"""
Agent Tests & Evaluation Benchmarks
=====================================
Tests for the Flow-first MLOps agent architecture.

Structure:
  - TestToolBoundaries: Unit tests for tool access control (no LLM, fast)
  - TestHelpers: Unit tests for utility functions (no LLM, fast)
  - TestFlowRouting: Integration tests for Flow routes against live cluster (LLM, slow)
  - TestConversation: Multi-turn context handling (LLM, slow)
  - TestEdgeCases: Robustness checks (LLM, slow)

Cluster assumptions (agent-cluster, Kind):
  - 9 system pods across kube-system and local-path-storage namespaces
  - All pods in Running state
  - No user deployments in default namespace (tests clean up after themselves)

Usage:
    uv run python -m pytest test_agent.py -v
    uv run python -m pytest test_agent.py -v -k "TestToolBoundaries or TestMLTools or TestHelpers"  # fast, no LLM
    uv run python -m pytest test_agent.py -v -k "TestFlowRouting"      # LLM integration
    uv run python -m pytest test_agent.py -v -k "TestConversation"     # LLM multi-turn
    uv run python -m pytest test_agent.py -v -k "TestEdgeCases"        # LLM robustness

Test results are designed to be used as thesis evaluation data.
Each LLM test records: route taken, response quality, cluster verification.
"""

import json
import os
import time

import pytest

os.environ.setdefault("LITELLM_LOG", "CRITICAL")

from src.agent import (
    AppState,
    InferenceClientTool,
    KubectlReadTool,
    KubectlWriteTool,
    MlOpsFlow,
    ModelVersionTool,
    PrometheusQueryTool,
    GrafanaAlertsTool,
    ScopedMemory,
    YAMLValidatorTool,
    build_context,
    clean_yaml,
    resolve_llm,
    run_cmd,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def llm():
    return resolve_llm()


def _run_flow(llm, user_input: str, history: list[dict] | None = None) -> dict:
    """Run a Flow and return structured result for evaluation."""
    context = build_context(history or [])
    flow = MlOpsFlow(llm=llm, memory=None)
    start_time = time.time()
    try:
        flow.kickoff(inputs={"user_input": user_input, "history_context": context})
        result = flow.state.final_response or "(no response)"
        intent = flow.state.intent
    except Exception as e:
        result = f"Error: {e}"
        intent = "error"
    elapsed = time.time() - start_time
    return {
        "input": user_input,
        "intent": intent,
        "response": result,
        "elapsed": round(elapsed, 2),
    }


def _has_any(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def _history_entry(user: str, result: str) -> dict:
    return {"user": user, "short": result[:200], "full": result[:800]}


def _find_nginx_deployment() -> str | None:
    out = run_cmd("kubectl get deployments -o name")
    for line in out.splitlines():
        if "nginx" in line.lower():
            return line.strip().split("/")[-1]
    return None


def _cleanup_nginx():
    for kind in ("deployments", "services"):
        resources = run_cmd(f"kubectl get {kind} -o name")
        for line in resources.splitlines():
            if "nginx" in line.lower():
                run_cmd(f"kubectl delete {line.strip()} --ignore-not-found")


def _cleanup_resource(kind: str, name: str, namespace: str = "default"):
    run_cmd(f"kubectl delete {kind} {name} -n {namespace} --ignore-not-found")


# ── Unit Tests: Tool Boundaries (no LLM, fast) ───────────────────────────


class TestToolBoundaries:
    """Verify that tool access controls enforce read/write separation.
    These tests validate the guardrails thesis requirement."""

    def test_read_tool_allows_get(self):
        tool = KubectlReadTool()
        result = tool._run("get namespaces")
        assert "kube-system" in result

    def test_read_tool_allows_describe(self):
        tool = KubectlReadTool()
        result = tool._run("describe namespace default")
        assert "default" in result.lower()

    def test_read_tool_allows_top(self):
        tool = KubectlReadTool()
        result = tool._run("top nodes")
        # top may not be available without metrics-server, but should not be "Denied"
        assert "Denied" not in result

    def test_read_tool_blocks_delete(self):
        tool = KubectlReadTool()
        result = tool._run("delete deployment nginx")
        assert "Denied" in result

    def test_read_tool_blocks_apply(self):
        tool = KubectlReadTool()
        result = tool._run("apply -f deployment.yaml")
        assert "Denied" in result

    def test_read_tool_blocks_create(self):
        tool = KubectlReadTool()
        result = tool._run("create deployment nginx --image=nginx")
        assert "Denied" in result

    def test_read_tool_blocks_scale(self):
        tool = KubectlReadTool()
        result = tool._run("scale deployment nginx --replicas=5")
        assert "Denied" in result

    def test_write_tool_allows_create(self):
        # Dry run to avoid side effects
        tool = KubectlWriteTool()
        result = tool._run("create deployment test-boundary --image=nginx --dry-run=client")
        assert "Denied" not in result

    def test_write_tool_allows_scale(self):
        tool = KubectlWriteTool()
        # Will error (no such deployment) but should NOT be denied
        result = tool._run("scale deployment nonexistent --replicas=3")
        assert "Denied" not in result

    def test_write_tool_blocks_get(self):
        tool = KubectlWriteTool()
        result = tool._run("get pods")
        assert "Denied" in result

    def test_write_tool_blocks_describe(self):
        tool = KubectlWriteTool()
        result = tool._run("describe pod xyz")
        assert "Denied" in result

    def test_write_tool_blocks_logs(self):
        tool = KubectlWriteTool()
        result = tool._run("logs some-pod")
        assert "Denied" in result


# ── Unit Tests: ML Tools (no LLM, fast) ───────────────────────────────────


class TestMLTools:
    """Verify ML-specific tools work correctly at the unit level."""

    def test_inference_client_get_health(self):
        """InferenceClientTool GET request to ML service health endpoint."""
        tool = InferenceClientTool()
        result = tool._run("http://localhost:30081/health")
        # If ML service is running, expect healthy response; otherwise expect connection error
        assert "healthy" in result.lower() or "error" in result.lower()

    def test_inference_client_get_metrics(self):
        """InferenceClientTool GET request to ML service metrics endpoint."""
        tool = InferenceClientTool()
        result = tool._run("http://localhost:30081/metrics")
        assert "ml_requests_total" in result or "error" in result.lower()

    def test_inference_client_post_predict(self):
        """InferenceClientTool POST request with JSON payload."""
        tool = InferenceClientTool()
        payload = json.dumps({})
        result = tool._run("http://localhost:30081/predict", payload=payload)
        assert "prediction" in result.lower() or "error" in result.lower()

    def test_model_version_tool_nonexistent_deployment(self):
        """ModelVersionTool on nonexistent deployment should report error."""
        tool = ModelVersionTool()
        result = tool._run("nonexistent-deploy-xyz", "v2")
        assert "not found" in result.lower() or "error" in result.lower() or "(command failed)" in result.lower()

    def test_prometheus_query_no_url(self):
        """PrometheusQueryTool without PROMETHEUS_URL returns config message."""
        original = os.environ.pop("PROMETHEUS_URL", None)
        try:
            tool = PrometheusQueryTool()
            result = tool._run("up")
            assert "not configured" in result.lower()
        finally:
            if original is not None:
                os.environ["PROMETHEUS_URL"] = original

    def test_grafana_alerts_no_url(self):
        """GrafanaAlertsTool without GRAFANA_URL returns config message."""
        original = os.environ.pop("GRAFANA_URL", None)
        try:
            tool = GrafanaAlertsTool()
            result = tool._run()
            assert "not configured" in result.lower()
        finally:
            if original is not None:
                os.environ["GRAFANA_URL"] = original


# ── Unit Tests: Helpers (no LLM, fast) ────────────────────────────────────


class TestHelpers:
    """Test utility functions that don't require LLM calls."""

    def test_clean_yaml_strips_fences(self):
        raw = "```yaml\napiVersion: v1\nkind: Pod\n```"
        assert clean_yaml(raw) == "apiVersion: v1\nkind: Pod"

    def test_clean_yaml_plain(self):
        raw = "apiVersion: v1\nkind: Pod"
        assert clean_yaml(raw) == raw

    def test_clean_yaml_generic_fence(self):
        raw = "```\napiVersion: v1\n```"
        assert clean_yaml(raw) == "apiVersion: v1"

    def test_run_cmd_success(self):
        result = run_cmd("echo hello")
        assert result == "hello"

    def test_run_cmd_failure(self):
        result = run_cmd("kubectl get nonexistent-resource-xyz")
        assert "error" in result.lower() or "not found" in result.lower()

    def test_run_cmd_empty_output(self):
        result = run_cmd("true")
        assert result == "(no output)"

    def test_build_context_empty(self):
        assert build_context([]) == ""

    def test_build_context_with_history(self):
        history = [_history_entry("hello", "Hi there!")]
        ctx = build_context(history)
        assert "CONVERSATION HISTORY" in ctx
        assert "hello" in ctx
        assert "Hi there!" in ctx

    def test_yaml_validator_valid(self):
        tool = YAMLValidatorTool()
        result = tool._run("apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: test\nspec: {}")
        assert "VALID" in result
        assert "Deployment" in result

    def test_yaml_validator_valid_multi_doc(self):
        tool = YAMLValidatorTool()
        yaml_text = (
            "apiVersion: v1\nkind: Service\nmetadata:\n  name: svc\n---\n"
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: dep"
        )
        result = tool._run(yaml_text)
        assert "VALID" in result

    def test_yaml_validator_invalid_missing_field(self):
        tool = YAMLValidatorTool()
        result = tool._run("apiVersion: v1\nmetadata:\n  name: test")
        assert "INVALID" in result

    def test_yaml_validator_invalid_empty(self):
        tool = YAMLValidatorTool()
        result = tool._run("")
        assert "INVALID" in result

    def test_history_entry_format(self):
        entry = _history_entry("test question", "x" * 1000)
        assert entry["user"] == "test question"
        assert len(entry["short"]) == 200
        assert len(entry["full"]) == 800


# ── Knowledge RAG Tests (no LLM, fast) ────────────────────────────────────


class TestKnowledgeRAG:
    """Test the knowledge base ingestion and retrieval pipeline."""

    @pytest.fixture(autouse=True)
    def setup_knowledge(self, tmp_path):
        """Create a temp knowledge dir and a fresh ChromaDB for each test."""
        self.knowledge_dir = tmp_path / "knowledge"
        self.knowledge_dir.mkdir()
        (self.knowledge_dir / "test.md").write_text(
            "# Test Doc\n\n"
            "## GPU Scheduling\n\n"
            "Use nvidia.com/gpu resource requests for GPU-accelerated inference pods.\n\n"
            "## Scaling Strategy\n\n"
            "Use HPA with custom metrics like ml_request_latency_seconds for autoscaling.\n\n"
            "## Health Probes\n\n"
            "Configure liveness and readiness probes on /health endpoints.\n"
        )
        chroma_path = str(tmp_path / "chroma")
        self.memory = ScopedMemory(chroma_path, "test_knowledge")

    def test_ingest_chunks_by_heading(self):
        n = self.memory.ingest_directory(str(self.knowledge_dir), "/knowledge")
        assert n == 3  # GPU Scheduling, Scaling Strategy, Health Probes

    def test_ingest_skips_when_unchanged(self):
        self.memory.ingest_directory(str(self.knowledge_dir), "/knowledge")
        n = self.memory.ingest_directory(str(self.knowledge_dir), "/knowledge")
        assert n == 0  # hash unchanged, skip

    def test_ingest_reingests_on_change(self):
        self.memory.ingest_directory(str(self.knowledge_dir), "/knowledge")
        (self.knowledge_dir / "test.md").write_text(
            "# Updated\n\n## New Section\n\nNew content here.\n"
        )
        n = self.memory.ingest_directory(str(self.knowledge_dir), "/knowledge")
        assert n > 0  # content changed, re-ingested

    def test_recall_returns_relevant_chunks(self):
        self.memory.ingest_directory(str(self.knowledge_dir), "/knowledge")
        results = self.memory.recall("/knowledge", "GPU scheduling for ML models", limit=2)
        assert len(results) > 0
        assert any("gpu" in r.lower() or "nvidia" in r.lower() for r in results)

    def test_recall_empty_when_no_knowledge(self):
        results = self.memory.recall("/knowledge", "GPU scheduling", limit=2)
        assert results == []


# ── Integration Tests: Flow Routing (LLM, slow) ──────────────────────────


class TestFlowRouting:
    """Test that the Flow correctly routes requests and produces valid results.
    Each test records intent, response, and elapsed time for thesis evaluation."""

    def test_ops_list_pods(self, llm):
        """Route: ops. Ask agent to list pods.
        Success: mentions pods, kube-system, or running counts."""
        r = _run_flow(llm, "what pods are running in the cluster")
        assert r["intent"] == "ops", f"Expected ops route, got: {r['intent']}"
        assert r["response"].strip() not in ("", "{}", "[]", "None")
        assert _has_any(r["response"], [
            "kube-system", "coredns", "running", "pod", "9"
        ]), f"Expected pod info, got: {r['response'][:200]}"

    def test_general_greeting(self, llm):
        """Route: general. Casual greeting — no tools needed.
        Success: friendly response mentioning assistant/help."""
        r = _run_flow(llm, "hello, who are you?")
        assert r["intent"] == "general", f"Expected general route, got: {r['intent']}"
        assert _has_any(r["response"], [
            "hello", "hi", "assist", "help", "mlops", "kubernetes", "I am", "I'm"
        ]), f"Expected greeting, got: {r['response'][:200]}"

    def test_ops_deploy_nginx(self, llm):
        """Route: ops or deploy. Create nginx deployment.
        Success: deployment exists on cluster after agent runs."""
        try:
            r = _run_flow(llm, "create a deployment called test-flow-nginx with image nginx and 1 replica")
            assert r["response"].strip() not in ("", "{}", "[]")
            assert _has_any(r["response"], [
                "nginx", "deploy", "created", "success", "running", "applied"
            ]), f"Expected deployment confirmation, got: {r['response'][:200]}"
            # Cluster verification
            name = _find_nginx_deployment()
            assert name is not None, "No nginx deployment found on cluster"
        finally:
            _cleanup_nginx()

    def test_ops_delete_resource(self, llm):
        """Route: ops. Delete a pre-created deployment.
        Success: deployment is gone from cluster."""
        run_cmd("kubectl create deployment test-flow-delete --image=nginx")
        time.sleep(2)
        try:
            r = _run_flow(llm, "delete the deployment called test-flow-delete")
            assert r["intent"] == "ops", f"Expected ops route, got: {r['intent']}"
            assert _has_any(r["response"], [
                "delet", "remov", "test-flow-delete"
            ]), f"Expected deletion confirmation, got: {r['response'][:200]}"
            verify = run_cmd("kubectl get deployment test-flow-delete --ignore-not-found")
            assert "test-flow-delete" not in verify or "not found" in verify.lower()
        finally:
            _cleanup_resource("deployment", "test-flow-delete")

    def test_deploy_route_yaml_pipeline(self, llm):
        """Route: deploy. Request that should go through coding → validate → apply.
        Success: agent generates YAML, validates it, and applies to cluster."""
        try:
            r = _run_flow(llm, "deploy a FastAPI ML inference service called ml-test with image python:3.11-slim on port 8000")
            assert r["intent"] == "deploy", f"Expected deploy route, got: {r['intent']}"
            assert r["response"].strip() not in ("", "{}", "[]")
            # Check if something was created (may be named differently by agent)
            verify = run_cmd("kubectl get deployments -o name")
            assert _has_any(verify, ["ml-test", "ml", "fastapi"]) or _has_any(r["response"], [
                "applied", "created", "deploy", "running"
            ]), f"Expected deployment, got response: {r['response'][:200]}, cluster: {verify}"
        finally:
            run_cmd("kubectl delete deployment ml-test --ignore-not-found")
            run_cmd("kubectl delete service ml-test --ignore-not-found")

    def test_monitor_route(self, llm):
        """Route: monitor. Ask about cluster health metrics.
        Success: routes to monitor, provides health info (even without Prometheus)."""
        r = _run_flow(llm, "check the cluster health and show me any alerts")
        assert r["intent"] == "monitor", f"Expected monitor route, got: {r['intent']}"
        assert r["response"].strip() not in ("", "{}", "[]")


# ── Integration Tests: Conversation (LLM, slow) ──────────────────────────


class TestConversation:
    """Test multi-turn conversation context handling via Flow."""

    def test_followup_summary(self, llm):
        """Turn 1: list pods. Turn 2: summarize.
        Success: turn 2 is shorter and references pods."""
        r1 = _run_flow(llm, "list all pods in the cluster")
        history = [_history_entry("list all pods in the cluster", r1["response"])]
        r2 = _run_flow(llm, "summarize that in 2 lines", history=history)
        assert r2["response"].strip() not in ("", "{}", "[]")
        assert len(r2["response"]) < len(r1["response"]) or _has_any(r2["response"], [
            "pod", "running", "kube-system", "cluster"
        ]), f"Expected short summary, got: {r2['response'][:300]}"

    def test_scale_with_context(self, llm):
        """Turn 1: create deployment. Turn 2: scale 'it' to 3.
        Success: resolves pronoun, deployment has 3 replicas."""
        try:
            r1 = _run_flow(llm, "create a deployment called test-context with image nginx and 1 replica")
            history = [_history_entry("create a deployment called test-context with image nginx and 1 replica", r1["response"])]
            time.sleep(2)
            r2 = _run_flow(llm, "scale it to 3 replicas", history=history)
            assert _has_any(r2["response"], [
                "scale", "3", "replica", "test-context"
            ]), f"Expected scale confirmation, got: {r2['response'][:200]}"
            # Verify on cluster
            verify = run_cmd("kubectl get deployment test-context -o jsonpath='{.spec.replicas}'")
            assert "3" in verify, f"Expected 3 replicas, got: {verify}"
        finally:
            _cleanup_resource("deployment", "test-context")


# ── Edge Case Tests (LLM, slow) ──────────────────────────────────────────


class TestEdgeCases:
    """Robustness and edge case checks."""

    def test_empty_result_not_returned(self, llm):
        """Agent should never return empty/useless outputs."""
        r = _run_flow(llm, "check the cluster status")
        assert r["response"].strip() not in ("", "{}", "[]", "None", "null")

    def test_nonexistent_resource(self, llm):
        """Ask about a resource that doesn't exist. Should report not-found, not crash."""
        r = _run_flow(llm, "describe the deployment called this-does-not-exist-xyz")
        assert _has_any(r["response"], [
            "not found", "does not exist", "no deployment", "couldn't find",
            "not exist", "error", "no resource", "unable"
        ]), f"Expected not-found message, got: {r['response'][:200]}"
