"""
Kubernetes Multi-Agent Flow Assistant
======================================
Flow-first architecture with deterministic routing and specialized agents.

Routes: deploy | monitor | ingress | ops | general
Agents: Orchestrator, Coding, Ops, Monitoring, Ingress — each with bounded tools.
"""

import os
import re

os.environ.setdefault("LITELLM_LOG", os.getenv("LITELLM_LOG", "CRITICAL"))

from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, listen, router, start
from crewai.llm import LLM
from crewai_tools import FileReadTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .rag import ScopedMemory
from .tools import (
    ApplyManifestTool,
    GrafanaAlertsTool,
    InferenceClientTool,
    KubectlReadTool,
    KubectlWriteTool,
    ModelVersionTool,
    PrometheusQueryTool,
    YAMLValidatorTool,
    clean_yaml,
    run_cmd,
)

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

    def call(self, messages, *args, **kwargs):
        # Strict chat templates (e.g. Qwen3.5 on vLLM) reject system messages after the first
        if isinstance(messages, list):
            messages = [{**m, "role": "user"} if i and m.get("role") == "system" else m
                        for i, m in enumerate(messages)]
        return super().call(messages, *args, **kwargs)


def resolve_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.4"))
    if provider == "ollama":
        os.environ["OLLAMA_API_BASE"] = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        model = f"ollama/{os.getenv('OLLAMA_MODEL', 'qwen3:8b')}"
    elif provider == "openai_compat":
        base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8001/v1")
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_COMPAT_API_KEY", "dummy")
        model = f"openai/{os.getenv('OPENAI_COMPAT_MODEL', 'default')}"
        temperature = float(os.getenv("OPENAI_COMPAT_TEMPERATURE", os.getenv("OLLAMA_TEMPERATURE", "0.4")))
        # disable TLS verification for endpoints with self-signed certs
        if os.getenv("OPENAI_COMPAT_SSL_VERIFY", "True").lower() == "false":
            import litellm
            litellm.ssl_verify = False
    elif provider == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
        model = f"openrouter/{os.getenv('OPENROUTER_MODEL', 'z-ai/glm-4.5-air:free')}"
    else:
        raise RuntimeError("Set LLM_PROVIDER in .env (ollama, openai_compat, or openrouter)")

    cls = ReActLLM if FORCE_REACT else LLM
    if provider in ("ollama", "openai_compat"):
        return cls(model=model, temperature=temperature, timeout=180,
                   extra_body={"options": {"num_ctx": 65536}})
    return cls(model=model)


# ── Helpers ───────────────────────────────────────────────────────────────


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
            llm=self.llm, tools=tools, verbose=VERBOSE, max_iter=25,
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
                "Only use deploy when complex YAML generation is needed.\n"
                "IMPORTANT: Use kubectl_read to check current cluster state if you need context before classifying."
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
                "Use file_read to read existing manifests as templates before writing new ones. "
                "Output ONLY valid YAML separated by ---. No markdown fences, no commentary. "
                "ALWAYS call yaml_validator on your output before finishing — never skip validation."
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
            goal="Execute kubectl commands and tools to manage cluster resources — never reason about state without calling a tool first",
            backstory=(
                "You manage the Kubernetes cluster and ML model deployments. "
                "ALWAYS call a tool — never guess or reason from memory about cluster state.\n"
                "Tool usage rules:\n"
                "- kubectl_read: for ALL queries — get, describe, logs, top. Call this first before any mutation.\n"
                "- kubectl_write: for mutations — scale, delete, apply -f <path>. "
                "After deploying a new resource, use 'rollout status deployment/<name> --timeout=90s' to wait for it to become ready before health checking.\n"
                "- model_version_update: REQUIRED for any model version switch (e.g. v1→v2). Do not use kubectl_write for this.\n"
                "- inference_client: REQUIRED for any health check, prediction request, or metrics fetch from the ML service. "
                "ML service NodePort 30081: GET http://localhost:30081/health, "
                "POST http://localhost:30081/predict (empty payload {} uses random input), "
                "GET http://localhost:30081/metrics.\n"
                "Never skip tool calls. If asked to check health, call inference_client. "
                "If asked to switch model version, call model_version_update. "
                "If asked about pod status, call kubectl_read."
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
            goal="Query cluster health and metrics by calling tools — never report metrics from memory",
            backstory=(
                "You monitor the cluster and ML services. ALWAYS call a tool to retrieve live data.\n"
                "Tool usage rules:\n"
                "- prometheus_query: REQUIRED for any Prometheus/PromQL request. "
                "Key metrics: ml_requests_total (counter, labels: model_version, predicted_class), "
                "ml_request_latency_seconds (histogram, label: model_version). "
                "PromQL examples: rate(ml_requests_total[5m]), "
                "histogram_quantile(0.95, rate(ml_request_latency_seconds_bucket[5m])).\n"
                "- grafana_alerts: REQUIRED for Grafana alert state queries.\n"
                "- inference_client: REQUIRED for ML service health/metrics. "
                "GET http://localhost:30081/health, GET http://localhost:30081/metrics.\n"
                "- kubectl_read: for pod status, resource usage, events.\n"
                "Never answer with assumed values. If Prometheus is unavailable, fall back to "
                "inference_client for /metrics, then kubectl_read. Report what the tool returns."
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
            goal="Manage ingress rules and traffic routing by calling tools",
            backstory=(
                "You handle ingress configuration, traffic routing, and canary deployments. "
                "ALWAYS call kubectl_read first to inspect current ingress/service state before making changes. "
                "Use kubectl_write for mutations, then verify with kubectl_read. "
                "Never assume configuration — always read before writing."
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


def main() -> None:
    llm = resolve_llm()
    memory = ScopedMemory(MEMORY_PATH, MEMORY_COLLECTION) if MEMORY_ENABLED else None
    if memory and os.path.isdir(KNOWLEDGE_DIR):
        n = memory.ingest_directory(KNOWLEDGE_DIR, KNOWLEDGE_SCOPE)
        if n:
            print(f"Ingested {n} knowledge chunks from {KNOWLEDGE_DIR}/")
    history: list[dict] = []

    print("Kubernetes Flow Assistant (type 'exit' to quit)\n")

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
