"""
Kubernetes & Inference Tools
============================
All BaseTool implementations used across architectures.
Add new tools here; import the subset you need in each architecture file.
"""

import os
import subprocess
import tempfile
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import yaml
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, model_validator


# ── Helpers (used internally by tools) ────────────────────────────────────


def run_cmd(cmd: str, timeout: int = 40) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = r.stdout.strip()
        err = r.stderr.strip()
        if r.returncode == 0:
            return out or err or "(no resources found)"
        return err or out or "(command failed)"
    except Exception as e:
        return f"Error: {e}"


def clean_yaml(text: str) -> str:
    t = text.strip()
    for prefix in ("```yaml", "```"):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


# ── kubectl tools ──────────────────────────────────────────────────────────


class _CmdIn(BaseModel):
    command: str = Field(..., description="kubectl command without 'kubectl' prefix")

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, v):
        return v[0] if isinstance(v, list) and v else v


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
        "label, annotate, patch, apply. Always verify with kubectl_read after mutations. "
        "Examples: 'create deployment nginx --image=nginx', 'scale deployment X --replicas=3', "
        "'apply -f k8s/ml_service/deployment.yaml'."
    )
    args_schema: type[BaseModel] = _CmdIn

    def _run(self, command: str) -> str:
        cmd = command.strip()
        allowed = ("create ", "scale ", "delete ", "rollout ", "expose ", "label ", "annotate ", "patch ", "apply ")
        if not cmd.startswith(allowed):
            return "Denied: allowed verbs are create/scale/delete/rollout/expose/label/annotate/patch/apply."
        return run_cmd(f"kubectl {cmd}")


# ── Manifest tool ──────────────────────────────────────────────────────────


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


# ── YAML validation tool ───────────────────────────────────────────────────


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


# ── Monitoring tools ───────────────────────────────────────────────────────


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


# ── ML-specific tools ──────────────────────────────────────────────────────


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
