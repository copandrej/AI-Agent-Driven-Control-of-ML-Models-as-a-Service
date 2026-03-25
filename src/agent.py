"""
MLOps Multi-Agent Flow Assistant
=================================
Flow-first architecture with deterministic routing and specialized agents.

Routes: deploy | monitor | ingress | ops | general
Agents: Orchestrator, Coding, Ops, Monitoring, Ingress — each with bounded tools.
"""

import hashlib
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

os.environ.setdefault("LITELLM_LOG", os.getenv("LITELLM_LOG", "CRITICAL"))

import yaml
from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, listen, router, start
from crewai.llm import LLM
from crewai.tools import BaseTool
from crewai_tools import FileReadTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────

VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"
FORCE_REACT = os.getenv("FORCE_REACT", "True").lower() == "true"
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "True").lower() == "true"
MEMORY_PATH = os.getenv("MEMORY_PATH", ".crewai/chroma")
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "mlops_memories")
MEMORY_RECALL_LIMIT = int(os.getenv("MEMORY_RECALL_LIMIT", "4"))
KEEP_MESSAGES = int(os.getenv("KEEP_MESSAGES", "5"))
KEEP_FULL_OUTPUTS = int(os.getenv("KEEP_FULL_OUTPUTS", "2"))
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge")
KNOWLEDGE_SCOPE = "/knowledge"
KNOWLEDGE_RECALL_LIMIT = int(os.getenv("KNOWLEDGE_RECALL_LIMIT", "3"))


# ── LLM Setup ─────────────────────────────────────────────────────────────


class ReActLLM(LLM):
    """Forces text-based ReAct tool calling instead of native function calling.
    Needed because Ollama models report supports_function_calling=True but
    produce empty {} with the OpenAI function-calling API."""

    def supports_function_calling(self) -> bool:
        return False


def resolve_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.4"))
    if provider == "ollama":
        os.environ["OLLAMA_API_BASE"] = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        model = f"ollama/{os.getenv('OLLAMA_MODEL', 'qwen3:8b')}"
    elif provider == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
        model = f"openrouter/{os.getenv('OPENROUTER_MODEL', 'z-ai/glm-4.5-air:free')}"
    else:
        raise RuntimeError("Set LLM_PROVIDER in .env (ollama or openrouter)")

    cls = ReActLLM if FORCE_REACT else LLM
    if provider == "ollama":
        return cls(model=model, temperature=temperature, timeout=180,
                   extra_body={"options": {"num_ctx": 65536}})
    return cls(model=model)


# ── Helpers ───────────────────────────────────────────────────────────────


def run_cmd(cmd: str, timeout: int = 40) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = r.stdout.strip()
        err = r.stderr.strip()
        if r.returncode == 0:
            return out or "(no output)"
        return err or out or "(command failed)"
    except Exception as e:
        return f"Error: {e}"


def clean_response(raw: str) -> str:
    """Strip <think> blocks, ReAct traces, and catch degenerate responses."""
    # Strip leaked <think> blocks (qwen3 thinking mode residue)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"</?think>", "", raw).strip()

    # Strip incomplete ReAct traces leaked when max_iter is hit
    if re.search(r"\nThought:", raw):
        result = re.split(r"\n(?:Thought|Action):", raw)[0].strip()
    elif raw.startswith("Thought:"):
        fa = re.search(r"Final Answer[:\s]+(.*)", raw, re.DOTALL | re.IGNORECASE)
        result = fa.group(1).strip() if fa else raw
    else:
        result = raw

    # Catch degenerate responses (just backticks, whitespace, etc.)
    if not result or re.fullmatch(r"[`\s]*", result):
        return "[Agent produced empty response — the model likely exhausted its output on internal reasoning. Try rephrasing the request.]"
    return result


def clean_yaml(text: str) -> str:
    t = text.strip()
    for prefix in ("```yaml", "```"):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


# ── Scoped Memory ─────────────────────────────────────────────────────────


class ScopedMemory:
    """Simple ChromaDB wrapper with scope-filtered recall."""

    def __init__(self, path: str, collection: str):
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(collection)

    def remember(self, scope: str, content: str) -> None:
        if not content.strip():
            return
        digest = hashlib.sha256(f"{scope}|{content}".encode()).hexdigest()
        self.collection.upsert(
            ids=[digest],
            documents=[content],
            metadatas=[{"scope": scope, "ts": datetime.now(timezone.utc).isoformat()}],
        )

    def recall(self, scope: str, query: str, limit: int = 4) -> list[str]:
        try:
            result = self.collection.query(
                query_texts=[query], n_results=limit, where={"scope": scope}
            )
            docs = result.get("documents", [[]])[0]
            return [d for d in docs if d.strip()]
        except Exception:
            return []

    def ingest_directory(self, directory: str, scope: str) -> int:
        """Chunk all .md files by ## headings and upsert into ChromaDB.
        Skips re-ingestion if the directory content hash hasn't changed."""
        from pathlib import Path

        md_files = sorted(Path(directory).glob("*.md"))
        if not md_files:
            return 0

        # Hash all file contents to detect changes
        content_hash = hashlib.sha256()
        for f in md_files:
            content_hash.update(f.read_bytes())
        digest = content_hash.hexdigest()

        # Check if already ingested (store hash as a sentinel document)
        sentinel_id = f"_knowledge_hash_{scope}"
        existing = self.collection.get(ids=[sentinel_id])
        if existing["documents"] and existing["documents"][0] == digest:
            return 0

        # Chunk by ## headings, keeping the heading with its content
        chunks = []
        for f in md_files:
            text = f.read_text()
            sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
            for section in sections:
                section = section.strip()
                if section and len(section) > 20:
                    chunks.append(section)

        # Upsert all chunks
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(f"{scope}|{chunk}".encode()).hexdigest()
            self.collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[{"scope": scope, "source": "knowledge"}],
            )

        # Store sentinel
        self.collection.upsert(
            ids=[sentinel_id], documents=[digest],
            metadatas=[{"scope": scope, "source": "_sentinel"}],
        )
        return len(chunks)


# ── Tools ─────────────────────────────────────────────────────────────────


class _CmdIn(BaseModel):
    command: str = Field(..., description="kubectl command without 'kubectl' prefix")


class KubectlReadTool(BaseTool):
    """Read-only kubectl: get, describe, logs, top."""
    name: str = "kubectl_read"
    description: str = (
        "Read-only kubectl access. Allowed verbs: get, describe, logs, top. "
        "Use -A for cluster-wide listing. Examples: 'get pods -A', 'describe pod X -n Y', 'top nodes'."
    )
    args_schema: type[BaseModel] = _CmdIn

    def _run(self, command: str) -> str:
        cmd = command.strip()
        if not cmd.startswith(("get ", "describe ", "logs ", "top ")):
            return "Denied: read-only tool — only get/describe/logs/top allowed."
        return run_cmd(f"kubectl {cmd}")


class KubectlWriteTool(BaseTool):
    """Mutation kubectl: create, scale, delete, rollout, expose, label, annotate, patch."""
    name: str = "kubectl_write"
    description: str = (
        "Mutating kubectl commands. Allowed: create, scale, delete, rollout, expose, "
        "label, annotate, patch. Always verify with kubectl_read after mutations. "
        "Examples: 'create deployment nginx --image=nginx', 'scale deployment X --replicas=3'."
    )
    args_schema: type[BaseModel] = _CmdIn

    def _run(self, command: str) -> str:
        cmd = command.strip()
        allowed = ("create ", "scale ", "delete ", "rollout ", "expose ", "label ", "annotate ", "patch ")
        if not cmd.startswith(allowed):
            return "Denied: allowed verbs are create/scale/delete/rollout/expose/label/annotate/patch."
        return run_cmd(f"kubectl {cmd}")


class _ManifestIn(BaseModel):
    manifest_yaml: str = Field(..., description="YAML manifest text")


class ApplyManifestTool(BaseTool):
    """Apply YAML manifest to the cluster."""
    name: str = "apply_manifest"
    description: str = (
        "Apply YAML manifest text to the Kubernetes cluster. "
        "Strips markdown fences automatically. Verifies pods after apply."
    )
    args_schema: type[BaseModel] = _ManifestIn

    def _run(self, manifest_yaml: str) -> str:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(clean_yaml(manifest_yaml))
                path = f.name
            out = run_cmd(f"kubectl apply -f {path}")
            verify = run_cmd("kubectl get pods -A -o wide")
            return f"Apply result:\n{out}\n\nCluster pods:\n{verify}"
        except Exception as e:
            return f"Error: {e}"


class _YamlIn(BaseModel):
    yaml_text: str = Field(..., description="YAML text to validate")


class YAMLValidatorTool(BaseTool):
    """Validate Kubernetes YAML structure."""
    name: str = "yaml_validator"
    description: str = "Validate Kubernetes YAML — returns VALID or INVALID with details."
    args_schema: type[BaseModel] = _YamlIn

    def _run(self, yaml_text: str) -> str:
        try:
            docs = [d for d in yaml.safe_load_all(clean_yaml(yaml_text)) if d]
            if not docs:
                return "INVALID: empty YAML"
            for d in docs:
                if not isinstance(d, dict):
                    return "INVALID: not a mapping"
                for k in ("apiVersion", "kind", "metadata"):
                    if k not in d:
                        return f"INVALID: missing {k}"
            return f"VALID: {', '.join(d['kind'] for d in docs)}"
        except Exception as e:
            return f"INVALID: {e}"


class _PromIn(BaseModel):
    query: str = Field(..., description="PromQL query")


class PrometheusQueryTool(BaseTool):
    """Run PromQL query against Prometheus."""
    name: str = "prometheus_query"
    description: str = "Run a PromQL query against the Prometheus HTTP API."
    args_schema: type[BaseModel] = _PromIn

    def _run(self, query: str) -> str:
        base = os.getenv("PROMETHEUS_URL", "").strip()
        if not base:
            return "PROMETHEUS_URL not configured"
        try:
            with urlopen(f"{base.rstrip('/')}/api/v1/query?query={quote_plus(query)}", timeout=20) as r:
                return r.read().decode()[:3000]
        except Exception as e:
            return f"Error: {e}"


class _AlertIn(BaseModel):
    state: str = Field(default="active", description="Alert state filter")


class GrafanaAlertsTool(BaseTool):
    """Fetch Grafana Alertmanager alerts."""
    name: str = "grafana_alerts"
    description: str = "Fetch alerts from Grafana Alertmanager."
    args_schema: type[BaseModel] = _AlertIn

    def _run(self, state: str = "active") -> str:
        base = os.getenv("GRAFANA_URL", "").strip()
        token = os.getenv("GRAFANA_TOKEN", "").strip()
        if not base:
            return "GRAFANA_URL not configured"
        url = f"{base.rstrip('/')}/api/alertmanager/grafana/api/v2/alerts?active={str(state == 'active').lower()}"
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urlopen(Request(url, headers=headers), timeout=20) as r:
                return r.read().decode()[:3000]
        except Exception as e:
            return f"Error: {e}"


class _VersionIn(BaseModel):
    deployment: str = Field(..., description="Deployment name")
    new_tag: str = Field(..., description="New image tag (e.g. 'v2')")
    namespace: str = Field(default="default", description="Namespace")


class ModelVersionTool(BaseTool):
    """Update the image tag of a deployment to switch model versions."""
    name: str = "model_version_update"
    description: str = (
        "Update the container image tag of a Kubernetes deployment to deploy a new model version. "
        "Example: deployment='ml-service', new_tag='v2' changes image to 'ml-service:v2'. "
        "Waits for rollout and verifies the update."
    )
    args_schema: type[BaseModel] = _VersionIn

    def _run(self, deployment: str, new_tag: str, namespace: str = "default") -> str:
        current = run_cmd(
            f"kubectl get deployment {deployment} -n {namespace} "
            f"-o jsonpath='{{.spec.template.spec.containers[0].image}}'"
        )
        base_image = current.rsplit(":", 1)[0] if ":" in current else current
        new_image = f"{base_image}:{new_tag}"
        result = run_cmd(
            f"kubectl set image deployment/{deployment} "
            f"{deployment}={new_image} -n {namespace}"
        )
        rollout = run_cmd(
            f"kubectl rollout status deployment/{deployment} -n {namespace} --timeout=60s"
        )
        verify = run_cmd(
            f"kubectl get deployment {deployment} -n {namespace} "
            f"-o jsonpath='{{.spec.template.spec.containers[0].image}}'"
        )
        return (
            f"Previous image: {current}\n"
            f"Updated to: {new_image}\n"
            f"Set image: {result}\n"
            f"Rollout: {rollout}\n"
            f"Verified image: {verify}"
        )


class _InferenceIn(BaseModel):
    url: str = Field(..., description="Full URL to the inference endpoint (e.g. http://localhost:30081/predict)")
    payload: str = Field(default="", description="JSON payload string to POST. Leave empty for GET requests.")


class InferenceClientTool(BaseTool):
    """Call an ML inference service endpoint to test it."""
    name: str = "inference_client"
    description: str = (
        "Call an ML service endpoint. Use for health checks (GET /health), "
        "predictions (POST /predict with JSON payload), or metrics (GET /metrics). "
        "Examples: url='http://localhost:30081/health' with empty payload for GET, "
        "url='http://localhost:30081/predict' with payload='{}' for POST (empty JSON uses random input)."
    )
    args_schema: type[BaseModel] = _InferenceIn

    def _run(self, url: str, payload: str = "") -> str:
        try:
            if payload.strip():
                data = payload.encode("utf-8")
                req = Request(url, data=data, headers={"Content-Type": "application/json"})
            else:
                req = Request(url)
            with urlopen(req, timeout=20) as r:
                return r.read().decode()[:3000]
        except Exception as e:
            return f"Error: {e}"


# ── Flow State & Routing ──────────────────────────────────────────────────


class AppState(BaseModel):
    user_input: str = ""
    history_context: str = ""
    intent: str = "general"
    coding_output: str = ""
    yaml_valid: bool = False
    final_response: str = ""


class MlOpsFlow(Flow[AppState]):

    def __init__(self, llm: LLM, memory: "ScopedMemory | None" = None):
        super().__init__()
        self.llm = llm
        self.memory = memory

    def _agent(self, role: str, goal: str, backstory: str, tools: list) -> Agent:
        return Agent(
            role=role, goal=goal, backstory=backstory,
            llm=self.llm, tools=tools, verbose=VERBOSE, max_iter=15,
        )

    def _run_crew(self, agent: Agent, description: str, expected: str, scope: str) -> str:
        prompt = self.state.history_context + description if self.state.history_context else description
        if self.memory:
            recalled = self.memory.recall(scope, self.state.user_input, MEMORY_RECALL_LIMIT)
            if recalled:
                prompt += "\n\nRelevant memory:\n" + "\n".join(f"- {r}" for r in recalled)
            knowledge = self.memory.recall(KNOWLEDGE_SCOPE, self.state.user_input, KNOWLEDGE_RECALL_LIMIT)
            if knowledge:
                prompt += "\n\nCluster knowledge:\n" + "\n".join(f"- {k}" for k in knowledge)
        task = Task(description=prompt, expected_output=expected, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], verbose=VERBOSE)
        try:
            raw = str(crew.kickoff())
        except ValueError as e:
            if "Invalid response from LLM call - None or empty" in str(e):
                return (
                    "[LLM returned empty response — likely hit max_iter with thinking mode active. "
                    "The model exhausted iterations and failed to emit a final answer. "
                    "Try rephrasing the request or simplifying the task.]"
                )
            raise

        result = clean_response(raw)
        if self.memory and result.strip():
            self.memory.remember(scope, f"Q: {self.state.user_input}\nA: {result[:1000]}")
        return result

    # ── Intent Classification ─────────────────────────────────────────

    @start()
    def classify_intent(self) -> str:
        orchestrator = self._agent(
            role="Orchestrator",
            goal="Classify user intent into exactly one route",
            backstory=(
                "You classify requests. Return ONLY one word:\n"
                "- deploy: ONLY when user needs a multi-resource YAML manifest written from scratch (e.g. deployment + service + ingress together)\n"
                "- ops: ANY kubectl operation — create/scale/delete/update deployment, list/describe pods/nodes, "
                "get logs, apply an existing manifest file, switch model version, call inference endpoints. "
                "If the user says 'create a deployment with image X' or 'deploy using manifest file Y', that is ops, NOT deploy.\n"
                "- monitor: Prometheus PromQL query, Grafana alerts, latency percentile analysis\n"
                "- ingress: ingress rules, traffic routing between services\n"
                "- general: greeting, question, conversation\n"
                "Rule of thumb: if it can be done with a single kubectl command, it is ops. "
                "Only use deploy when complex YAML generation is needed."
            ),
            tools=[KubectlReadTool()],
        )
        result = self._run_crew(
            agent=orchestrator,
            description=f"Classify this request into one word (deploy/ops/monitor/ingress/general): {self.state.user_input}",
            expected="Single word: deploy, ops, monitor, ingress, or general",
            scope="/project/mlops/orchestrator",
        )
        match = re.search(r"\b(deploy|ops|monitor|ingress|general)\b", result.strip().lower())
        self.state.intent = match.group(1) if match else "ops"
        return self.state.intent

    @router(classify_intent)
    def route_intent(self) -> str:
        return self.state.intent

    # ── Deploy Route: coding → validate → apply ───────────────────────

    @listen("deploy")
    def coding_step(self) -> str:
        coder = self._agent(
            role="Coding & Configuration Agent",
            goal="Generate valid Kubernetes YAML manifests",
            backstory=(
                "You write Kubernetes YAML for deployments, services, and ingress. "
                "Output ONLY valid YAML separated by ---. No markdown fences, no commentary. "
                "Always validate your output with yaml_validator before finishing."
            ),
            tools=[FileReadTool(), YAMLValidatorTool()],
        )
        self.state.coding_output = self._run_crew(
            agent=coder,
            description=f"Create Kubernetes YAML manifests for: {self.state.user_input}",
            expected="Valid Kubernetes YAML documents only",
            scope="/project/mlops/coding",
        )
        return self.state.coding_output

    @listen(coding_step)
    def validate_yaml(self) -> str:
        cleaned = clean_yaml(self.state.coding_output)
        self.state.coding_output = cleaned
        validator = YAMLValidatorTool()
        result = validator._run(cleaned)
        self.state.yaml_valid = result.startswith("VALID")
        return "yaml_valid" if self.state.yaml_valid else "yaml_invalid"

    @router(validate_yaml)
    def route_validation(self) -> str:
        return "yaml_valid" if self.state.yaml_valid else "yaml_invalid"

    @listen("yaml_valid")
    def ops_apply(self) -> str:
        ops = self._agent(
            role="Ops / Execution Agent",
            goal="Apply validated YAML to the cluster",
            backstory="You apply manifests and verify the result. Use apply_manifest, then verify with kubectl_read.",
            tools=[ApplyManifestTool(), KubectlReadTool()],
        )
        result = self._run_crew(
            agent=ops,
            description=f"Apply this YAML to the cluster and verify:\n\n{self.state.coding_output}",
            expected="Apply result with verification summary",
            scope="/project/mlops/ops",
        )
        self.state.final_response = result
        return result

    @listen("yaml_invalid")
    def handle_invalid_yaml(self) -> str:
        self.state.final_response = "YAML validation failed for your request. The generated YAML was invalid. Please try again with more specific instructions."
        return self.state.final_response

    # ── Ops Route ─────────────────────────────────────────────────────

    @listen("ops")
    def ops_step(self) -> str:
        ops = self._agent(
            role="Ops / Execution Agent",
            goal="Execute kubectl commands to manage cluster resources",
            backstory=(
                "You manage the Kubernetes cluster and ML model deployments. "
                "Use kubectl_read for queries (get, describe, logs, top). "
                "Use kubectl_write for mutations (create, scale, delete, rollout). "
                "Use model_version_update to switch ML model versions (e.g. v1→v2) by updating deployment image tags. "
                "Use inference_client to test ML service endpoints (health checks, predictions, metrics). "
                "The ML inference service runs on NodePort 30081: health at http://localhost:30081/health, "
                "predictions at http://localhost:30081/predict, metrics at http://localhost:30081/metrics. "
                "Always verify mutations with kubectl_read after executing them. "
                "Summarize results in clear natural language."
            ),
            tools=[KubectlReadTool(), KubectlWriteTool(), ModelVersionTool(), InferenceClientTool()],
        )
        result = self._run_crew(
            agent=ops,
            description=f"Handle this cluster operation: {self.state.user_input}",
            expected="Clear summary of the operation result",
            scope="/project/mlops/ops",
        )
        self.state.final_response = result
        return result

    # ── Monitor Route ─────────────────────────────────────────────────

    @listen("monitor")
    def monitor_step(self) -> str:
        monitor = self._agent(
            role="Monitoring Agent",
            goal="Query cluster health, metrics, and alerts",
            backstory=(
                "You monitor the cluster and ML services via Prometheus and Grafana when configured. "
                "Fall back to kubectl_read for pod status and resource usage. "
                "Use inference_client to check ML service health (http://localhost:30081/health) "
                "or scrape raw Prometheus metrics (http://localhost:30081/metrics). "
                "Key ML metrics: ml_requests_total (counter by model_version, predicted_class), "
                "ml_request_latency_seconds (histogram by model_version). "
                "PromQL examples: rate(ml_requests_total[5m]), "
                "histogram_quantile(0.95, rate(ml_request_latency_seconds_bucket[5m])). "
                "Recommend scaling actions when you detect issues."
            ),
            tools=[PrometheusQueryTool(), GrafanaAlertsTool(), KubectlReadTool(), InferenceClientTool()],
        )
        result = self._run_crew(
            agent=monitor,
            description=f"Handle this monitoring request: {self.state.user_input}",
            expected="Monitoring summary with actionable insights",
            scope="/project/mlops/prometheus",
        )
        self.state.final_response = result
        return result

    # ── Ingress Route ─────────────────────────────────────────────────

    @listen("ingress")
    def ingress_step(self) -> str:
        ingress = self._agent(
            role="Ingress / Networking Agent",
            goal="Manage ingress rules and traffic routing",
            backstory=(
                "You handle ingress configuration, traffic routing, and canary deployments. "
                "Use kubectl_read to inspect current ingress state and kubectl_write for changes."
            ),
            tools=[KubectlReadTool(), KubectlWriteTool()],
        )
        result = self._run_crew(
            agent=ingress,
            description=f"Handle this networking request: {self.state.user_input}",
            expected="Networking action result and status",
            scope="/project/mlops/ingress",
        )
        self.state.final_response = result
        return result

    # ── General Route ─────────────────────────────────────────────────

    @listen("general")
    def general_step(self) -> str:
        orchestrator = self._agent(
            role="MLOps Assistant",
            goal="Answer general questions helpfully",
            backstory="You are a helpful MLOps assistant. Answer conversational questions directly.",
            tools=[KubectlReadTool()],
        )
        result = self._run_crew(
            agent=orchestrator,
            description=f"Answer this question: {self.state.user_input}",
            expected="Helpful, concise response",
            scope="/project/mlops/orchestrator",
        )
        self.state.final_response = result
        return result


# ── Conversation Loop ─────────────────────────────────────────────────────


def build_context(history: list[dict]) -> str:
    if not history:
        return ""
    recent = history[-KEEP_MESSAGES:]
    parts = []
    for i, h in enumerate(recent):
        is_recent = i >= len(recent) - KEEP_FULL_OUTPUTS
        resp = h["full"] if is_recent else h["short"]
        parts.append(f"User: {h['user']}\nAssistant: {resp}")
    return (
        "=== CONVERSATION HISTORY ===\n"
        + "\n".join(parts)
        + "\n=== END HISTORY ===\n\n"
    )


def main() -> None:
    llm = resolve_llm()
    memory = ScopedMemory(MEMORY_PATH, MEMORY_COLLECTION) if MEMORY_ENABLED else None
    if memory and os.path.isdir(KNOWLEDGE_DIR):
        n = memory.ingest_directory(KNOWLEDGE_DIR, KNOWLEDGE_SCOPE)
        if n:
            print(f"Ingested {n} knowledge chunks from {KNOWLEDGE_DIR}/")
    history: list[dict] = []

    print("MLOps Flow Assistant (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        context = build_context(history)
        flow = MlOpsFlow(llm=llm, memory=memory)
        try:
            flow.kickoff(inputs={"user_input": user_input, "history_context": context})
            result = flow.state.final_response or "(no response)"
        except Exception as e:
            result = f"Error: {e}"

        history.append({
            "user": user_input,
            "short": result[:200],
            "full": result[:800],
        })
        if len(history) > KEEP_MESSAGES + 2:
            history.pop(0)

        print(f"\nAgent: {result}\n")


if __name__ == "__main__":
    main()
