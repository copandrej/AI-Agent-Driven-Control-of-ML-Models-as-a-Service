"""
Kubernetes & Inference Tools
============================
All BaseTool implementations used across architectures.
Add new tools here; import the subset you need in each architecture file.
"""

import os
import re
import shlex
import subprocess
import tempfile
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import yaml
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, model_validator


# ── Helpers (used internally by tools) ────────────────────────────────────


def run_cmd(cmd: str | list[str], timeout: int = 40) -> str:
    # list → exec directly (no shell), so ;, &&, |, $() in args can't chain commands
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
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


def kubectl_apply(manifest_yaml: str, *flags: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write(clean_yaml(manifest_yaml))
        f.flush()
        return run_cmd(["kubectl", "apply", *flags, "-f", f.name])


# ── kubectl tools ──────────────────────────────────────────────────────────


def _cmd_timeout(args: list[str], default: int = 40) -> int:
    """The command's own --timeout (e.g. rollout status --timeout=90s) plus a margin, otherwise the default."""
    for i, a in enumerate(args):
        value = a.split("=", 1)[1] if a.startswith("--timeout=") else args[i + 1] if a == "--timeout" and i + 1 < len(args) else None
        m = re.fullmatch(r"(\d+)(s|m)?", value or "")
        if m:
            return min(int(m.group(1)) * (60 if m.group(2) == "m" else 1) + 10, 310)
    return default


class _CmdIn(BaseModel):
    command: str = Field(..., description="kubectl command without 'kubectl' prefix")

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, v):
        return v[0] if isinstance(v, list) and v else v


P3_GATE = os.getenv("P3_GATE", "off").lower() == "on"
P3_REQUEST = ""  # the operator's request for the current task (set per task by the caller)
_POD = {"pod", "pods", "po"}
_VALUE_FLAGS = {"-n", "--namespace", "-l", "--selector", "--field-selector", "--grace-period", "--timeout", "--context"}


def _delete_targets(args: list[str]) -> tuple[list[str], list[str]]:
    """Resource kinds and names in `delete ...`: 'pod x', 'deploy,svc x', 'deploy/x pod/y'."""
    words, rest = [], iter(args[1:])
    for a in rest:
        if a in _VALUE_FLAGS:
            next(rest, None)
        elif not a.startswith("-"):
            words.append(a)
    if not words:
        return [], []
    if "/" in words[0]:
        pairs = [w.split("/", 1) for w in words if "/" in w]
        return [k for k, _ in pairs], [n for _, n in pairs]
    return words[0].split(","), words[1:]


_NEGATED = re.compile(r"\b(not|never|no|without)\b|n't", re.I)  # anywhere before the verb: 'Do not replace X, delete Y'


def delete_requested(names: list[str]) -> bool:
    """One sentence of the operator's request asks to delete or remove (not 'do not delete') and names every one of
    these objects."""
    for sentence in re.split(r"[.;\n]", P3_REQUEST) if names else []:
        asks = any(not _NEGATED.search(sentence[:m.start()])
                   for m in re.finditer(r"\b(delete|remove)\b", sentence, re.I))
        if asks and all(re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", sentence) for n in names):
            return True
    return False


def p3_denied(args: list[str]) -> str | None:
    """P3 no-delete gate: block deleting or force-replacing anything but Pods (controllers recreate pods), unless
    the request asks to delete that named object; the agent has to repair objects in place.
    Returns the denial message, or None when allowed."""
    if not P3_GATE or not args:
        return None
    if args[0] == "delete":
        kinds, names = _delete_targets(args)
        if ("-f" in args or any(a.startswith("--filename") for a in args) or not kinds or any(
                k.split(".")[0].lower() not in _POD for k in kinds)) and not delete_requested(names):
            return ("Denied (P3): deleting objects other than Pods is blocked by the no-delete policy unless the "
                    "request asks to delete that object. Repair the object in place (patch, apply, set) instead.")
    if (args[0] == "replace" and "--force" in args) or (args[0] == "apply" and any(a.startswith("--prune") for a in args)):
        return "Denied (P3): replace --force / apply --prune delete objects and are blocked by the no-delete policy."
    return None


class KubectlReadTool(BaseTool):
    """Read-only kubectl: get, describe, logs, top, rollout status/history."""
    name: str = "kubectl_read"
    description: str = (
        "Read-only kubectl access. Allowed: get, describe, logs, top, rollout status, rollout history. "
        "Use -A for cluster-wide listing. Examples: 'get pods -A', 'describe pod X -n Y', 'top nodes', "
        "'rollout status deployment/X --timeout=90s'."
    )
    args_schema: type[BaseModel] = _CmdIn

    def _run(self, command: str) -> str:
        try:
            args = shlex.split(command)
        except ValueError as e:
            return f"Error: {e}"
        if not args or not (args[0] in {"get", "describe", "logs", "top"}
                            or args[:2] in (["rollout", "status"], ["rollout", "history"])):
            return "Denied: read-only tool — only get/describe/logs/top/rollout status/rollout history allowed."
        return run_cmd(["kubectl", *args], timeout=_cmd_timeout(args))


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
        try:
            args = shlex.split(command)
        except ValueError as e:
            return f"Error: {e}"
        allowed = {"create", "scale", "delete", "rollout", "expose", "label", "annotate", "patch", "apply"}
        if not args or args[0] not in allowed:
            return "Denied: allowed verbs are create/scale/delete/rollout/expose/label/annotate/patch/apply."
        if denied := p3_denied(args):
            return denied
        return run_cmd(["kubectl", *args], timeout=_cmd_timeout(args))


class KubectlTool(BaseTool):
    """Unrestricted kubectl (no read/write separation) — bare baseline only."""
    name: str = "kubectl"
    description: str = "Run any kubectl command. Examples: 'get pods -A', 'scale deployment X --replicas=3'."
    args_schema: type[BaseModel] = _CmdIn

    def _run(self, command: str) -> str:
        try:
            args = shlex.split(command)
        except ValueError as e:
            return f"Error: {e}"
        if args[:1] == ["kubectl"]:
            args = args[1:]
        if denied := p3_denied(args):
            return denied
        return run_cmd(["kubectl", *args], timeout=_cmd_timeout(args))


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
            out = kubectl_apply(manifest_yaml)
            verify = run_cmd("kubectl get pods -A -o wide")
            return f"Apply result:\n{out}\n\nCluster pods:\n{verify}"
        except Exception as e:
            return f"Error: {e}"


# ── YAML validation tool ───────────────────────────────────────────────────


class _YamlIn(BaseModel):
    yaml_text: str = Field(..., description="YAML text to validate")


class YAMLValidatorTool(BaseTool):
    """Validate Kubernetes YAML structure, then server-side dry run (schema, admission)."""
    name: str = "yaml_validator"
    description: str = "Validate Kubernetes YAML (structure + server dry run) — returns VALID or INVALID with details."
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
            dry = kubectl_apply(yaml_text, "--dry-run=server")
            if re.search(r"^(error|Error from server)", dry, re.MULTILINE):
                return f"INVALID: {dry}"
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
            f"kubectl rollout status deployment/{deployment} -n {namespace} --timeout=60s", timeout=70
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
