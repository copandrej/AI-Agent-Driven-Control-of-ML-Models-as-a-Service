"""
Platform toolset (TOOLSET=platform)
===================================
The tools an ML-serving platform agent would have connected: the Kubernetes, Prometheus and Helm MCP servers
(names and arguments modelled on containers/kubernetes-mcp-server, pab1it0/prometheus-mcp-server and the Helm
MCP servers) and our own ML-platform tools (training jobs, model registry, autoscaling, batch inference).
All are functional on the cluster. Used to test bounded tools per agent under small context windows:
Single gets every server, each Multi specialist only the servers of its domain (src/agent.py).
"""

import hashlib
import json
import os
import time
from urllib.parse import urlencode
from urllib.request import urlopen

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from .tools import kubectl_apply, p3_denied, run_cmd

REGISTRY = "model-registry"  # ConfigMap in the default namespace, one key per model version
NS = {"namespace": ("Namespace", "default")}


def _kubectl(*args: str) -> str:
    if denied := p3_denied(list(args)):
        return denied
    return run_cmd(["kubectl", *args], timeout=60)


def _ns(namespace: str) -> list[str]:
    return ["-A"] if namespace in ("", "all") else ["-n", namespace]


def _prom(path: str, **params) -> str:
    base = os.getenv("PROMETHEUS_URL", "").strip()
    if not base:
        return "PROMETHEUS_URL not configured"
    try:
        with urlopen(f"{base.rstrip('/')}/api/v1/{path}?{urlencode(params)}", timeout=20) as r:
            return r.read().decode()[:3000]
    except Exception as e:
        return f"Error: {e}"


# ── ML platform ───────────────────────────────────────────────────────────


def _job_yaml(name: str, kind: str, script: str, namespace: str) -> str:
    return json.dumps({
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": namespace, "labels": {"ml-platform/kind": kind}},
        "spec": {"backoffLimit": 0, "template": {"spec": {"restartPolicy": "Never", "containers": [
            {"name": kind, "image": "busybox:1.36", "command": ["sh", "-c", script],
             "resources": {"requests": {"cpu": "50m", "memory": "32Mi"}}}]}}},
    })


def submit_training_job(name: str, model: str, dataset: str, epochs: int = 3, namespace: str = "default") -> str:
    acc = 0.80 + int(hashlib.sha1(f"{model}/{dataset}/{epochs}".encode()).hexdigest(), 16) % 150 / 1000
    script = (f"echo 'training {model} on {dataset} for {epochs} epochs'; sleep {5 * epochs}; "
              f"echo 'done: val_accuracy={acc:.3f} artifact=registry://{model}/{name}'")
    return kubectl_apply(_job_yaml(name, "training", script, namespace))


def get_training_job(name: str, namespace: str = "default") -> str:
    status = _kubectl("get", "job", name, "-n", namespace, "-o",
                      "jsonpath={.status.succeeded}/{.status.failed}/{.status.active}")
    return f"succeeded/failed/active: {status}\nlogs:\n{_kubectl('logs', f'job/{name}', '-n', namespace, '--tail=20')}"


def _registry() -> dict:
    out = run_cmd(["kubectl", "get", "configmap", REGISTRY, "-n", "default", "-o", "json"])
    try:
        return json.loads(out).get("data") or {}
    except ValueError:
        return {}


def _registry_put(key: str, entry: dict) -> str:
    if not _registry() and "NotFound" in run_cmd(["kubectl", "get", "configmap", REGISTRY, "-n", "default"]):
        run_cmd(["kubectl", "create", "configmap", REGISTRY, "-n", "default"])
    patch = json.dumps({"data": {key: json.dumps(entry)}})
    return run_cmd(["kubectl", "patch", "configmap", REGISTRY, "-n", "default", "--type", "merge", "-p", patch])


def register_model(model: str, version: str, image: str, val_accuracy: float = 0.0) -> str:
    entry = {"image": image, "stage": "staging", "val_accuracy": val_accuracy,
             "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return _registry_put(f"{model}.{version}", entry)


def list_model_versions(model: str) -> str:
    rows = {k: json.loads(v) for k, v in _registry().items() if k.split(".")[0] == model}
    return json.dumps(rows, indent=1) if rows else f"no versions of {model} in the registry"


def set_model_stage(model: str, version: str, stage: str) -> str:
    entry = _registry().get(f"{model}.{version}")
    if not entry:
        return f"Error: {model} {version} is not registered"
    return _registry_put(f"{model}.{version}", {**json.loads(entry), "stage": stage})


def set_autoscaling(deployment: str, min_replicas: int, max_replicas: int, cpu_percent: int = 70,
                    namespace: str = "default") -> str:
    return kubectl_apply(json.dumps({
        "apiVersion": "autoscaling/v2", "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": deployment, "namespace": namespace},
        "spec": {"scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": deployment},
                 "minReplicas": int(min_replicas), "maxReplicas": int(max_replicas),
                 "metrics": [{"type": "Resource", "resource": {"name": "cpu", "target": {
                     "type": "Utilization", "averageUtilization": int(cpu_percent)}}}]},
    }))


def set_resources(deployment: str, cpu_request: str = "", cpu_limit: str = "", memory_request: str = "",
                  memory_limit: str = "", namespace: str = "default") -> str:
    req = ",".join(f"{k}={v}" for k, v in (("cpu", cpu_request), ("memory", memory_request)) if v)
    lim = ",".join(f"{k}={v}" for k, v in (("cpu", cpu_limit), ("memory", memory_limit)) if v)
    args = ["set", "resources", f"deployment/{deployment}", "-n", namespace]
    return _kubectl(*args, *([f"--requests={req}"] if req else []), *([f"--limits={lim}"] if lim else []))


def run_batch_inference(name: str, requests: int = 20, service_url: str = "http://ml-service.default.svc/predict",
                        namespace: str = "default") -> str:
    script = (f"ok=0; for i in $(seq {requests}); do wget -qO- --post-data '{{}}' "
              f"--header 'Content-Type: application/json' {service_url} >/dev/null && ok=$((ok+1)); done; "
              f"echo \"batch done: $ok/{requests} predictions succeeded\"")
    return kubectl_apply(_job_yaml(name, "batch-inference", script, namespace))


# ── tool table: (name, description, {arg: (description, default or ... = required)}, handler) ──────

SERVERS = {
    "kubernetes": [
        ("pods_list", "List all the Kubernetes pods in the current cluster from all namespaces, or in one namespace.",
         {"namespace": ("Namespace, or 'all'", "all")}, lambda namespace: _kubectl("get", "pods", *_ns(namespace), "-o", "wide")),
        ("pods_get", "Get a Kubernetes Pod in the current or provided namespace with the provided name.",
         {"name": ("Pod name", ...), **NS}, lambda name, namespace: _kubectl("get", "pod", name, "-n", namespace, "-o", "yaml")),
        ("pods_log", "Get the logs of a Kubernetes Pod in the current or provided namespace with the provided name.",
         {"name": ("Pod name", ...), **NS, "container": ("Container name (optional)", "")},
         lambda name, namespace, container: _kubectl("logs", name, "-n", namespace, "--tail=100",
                                                     *(["-c", container] if container else []))),
        ("pods_delete", "Delete a Kubernetes Pod in the current or provided namespace with the provided name.",
         {"name": ("Pod name", ...), **NS}, lambda name, namespace: _kubectl("delete", "pod", name, "-n", namespace)),
        ("pods_exec", "Execute a command in a Kubernetes Pod in the current or provided namespace.",
         {"name": ("Pod name", ...), **NS, "command": ("Command as one shell string", ...)},
         lambda name, namespace, command: _kubectl("exec", name, "-n", namespace, "--", "sh", "-c", command)),
        ("resources_list", "List Kubernetes resources and objects in the current cluster by kind "
         "(e.g. Deployment, Service, ConfigMap, NetworkPolicy, Role).",
         {"kind": ("Resource kind", ...), "namespace": ("Namespace, or 'all'", "all")},
         lambda kind, namespace: _kubectl("get", kind, *_ns(namespace), "-o", "wide")),
        ("resources_get", "Get a Kubernetes resource in the current cluster by kind and name, as YAML.",
         {"kind": ("Resource kind", ...), "name": ("Resource name", ...), **NS},
         lambda kind, name, namespace: _kubectl("get", kind, name, "-n", namespace, "-o", "yaml")),
        ("resources_create_or_update", "Create or update a Kubernetes resource in the current cluster by providing "
         "a YAML or JSON representation of the resource.",
         {"resource": ("YAML or JSON of the resource", ...)}, lambda resource: kubectl_apply(resource)),
        ("resources_delete", "Delete a Kubernetes resource in the current cluster by kind and name.",
         {"kind": ("Resource kind", ...), "name": ("Resource name", ...), **NS},
         lambda kind, name, namespace: _kubectl("delete", kind, name, "-n", namespace)),
        ("events_list", "List all the Kubernetes events in the current cluster from all namespaces, or in one namespace.",
         {"namespace": ("Namespace, or 'all'", "all")},
         lambda namespace: _kubectl("get", "events", *_ns(namespace), "--sort-by=.lastTimestamp")),
        ("namespaces_list", "List all the Kubernetes namespaces in the current cluster.",
         {}, lambda: _kubectl("get", "namespaces")),
    ],
    "prometheus": [
        ("execute_query", "Execute a PromQL instant query against Prometheus.",
         {"query": ("PromQL query", ...)}, lambda query: _prom("query", query=query)),
        ("execute_range_query", "Execute a PromQL range query with start time, end time and step interval.",
         {"query": ("PromQL query", ...), "start": ("Start time, RFC3339 or unix seconds", ...),
          "end": ("End time, RFC3339 or unix seconds", ...), "step": ("Step, e.g. '30s'", "30s")},
         lambda query, start, end, step: _prom("query_range", query=query, start=start, end=end, step=step)),
        ("list_metrics", "List all metric names available in Prometheus.", {}, lambda: _prom("label/__name__/values")),
        ("get_metric_metadata", "Get metadata (type, help, unit) for a specific metric.",
         {"metric": ("Metric name", ...)}, lambda metric: _prom("metadata", metric=metric)),
        ("get_targets", "Get information about all scrape targets and their health.", {}, lambda: _prom("targets")),
    ],
    "helm": [
        ("helm_list", "List all the Helm releases in the current or provided namespace (or in all namespaces).",
         {"namespace": ("Namespace, or 'all'", "all")},
         lambda namespace: run_cmd(["helm", "list", *(["-A"] if namespace in ("", "all") else ["-n", namespace])])),
        ("helm_status", "Show the status of a Helm release.", {"release": ("Release name", ...), **NS},
         lambda release, namespace: run_cmd(["helm", "status", release, "-n", namespace])),
        ("helm_history", "Show the revision history of a Helm release.", {"release": ("Release name", ...), **NS},
         lambda release, namespace: run_cmd(["helm", "history", release, "-n", namespace])),
        ("helm_rollback", "Roll a Helm release back to a previous revision.",
         {"release": ("Release name", ...), "revision": ("Revision number", ...), **NS},
         lambda release, revision, namespace: run_cmd(["helm", "rollback", release, str(revision), "-n", namespace],
                                                      timeout=120)),
    ],
    "ml_platform": [
        ("submit_training_job", "Submit a model training job on the cluster (a Kubernetes Job). The job logs report "
         "the validation accuracy and the artifact when it finishes.",
         {"name": ("Job name", ...), "model": ("Model name, e.g. 'mobilenet_v2'", ...),
          "dataset": ("Dataset name or version", ...), "epochs": ("Number of epochs", 3), **NS}, submit_training_job),
        ("get_training_job", "Get the status and the last log lines of a training job.",
         {"name": ("Job name", ...), **NS}, get_training_job),
        ("list_training_jobs", "List training and batch inference jobs submitted through the platform.",
         {"namespace": ("Namespace, or 'all'", "all")},
         lambda namespace: _kubectl("get", "jobs", *_ns(namespace), "-l", "ml-platform/kind")),
        ("register_model", "Register a model version in the model registry (stage 'staging').",
         {"model": ("Model name", ...), "version": ("Version, e.g. 'v3'", ...),
          "image": ("Serving image, e.g. 'copandrej/ml-service:v3'", ...),
          "val_accuracy": ("Validation accuracy from training", 0.0)}, register_model),
        ("list_model_versions", "List the registered versions of a model with their stage, image and accuracy.",
         {"model": ("Model name", ...)}, list_model_versions),
        ("set_model_stage", "Set the registry stage of a model version: staging, production or archived.",
         {"model": ("Model name", ...), "version": ("Version", ...), "stage": ("staging | production | archived", ...)},
         set_model_stage),
        ("set_autoscaling", "Create or update a CPU-based HorizontalPodAutoscaler for a model serving deployment.",
         {"deployment": ("Deployment name", ...), "min_replicas": ("Minimum replicas", ...),
          "max_replicas": ("Maximum replicas", ...), "cpu_percent": ("Target CPU utilization %", 70), **NS},
         set_autoscaling),
        ("set_resources", "Set CPU/memory requests and limits of a model serving deployment (rolls the pods).",
         {"deployment": ("Deployment name", ...), "cpu_request": ("e.g. '500m'", ""), "cpu_limit": ("e.g. '2'", ""),
          "memory_request": ("e.g. '512Mi'", ""), "memory_limit": ("e.g. '1Gi'", ""), **NS}, set_resources),
        ("run_batch_inference", "Run a batch inference job that sends N prediction requests to a model service.",
         {"name": ("Job name", ...), "requests": ("Number of requests", 20),
          "service_url": ("Prediction URL", "http://ml-service.default.svc/predict"), **NS}, run_batch_inference),
    ],
}


def _make_tool(name: str, description: str, args: dict, handler) -> BaseTool:
    schema = create_model(f"{name}_args", **{
        a: ((type(d) if d is not ... else str), Field(d, description=desc)) for a, (desc, d) in args.items()})

    class PlatformTool(BaseTool):
        def _run(self, **kwargs) -> str:
            try:
                return handler(**{a: kwargs.get(a, d) for a, (_, d) in args.items()})
            except Exception as e:
                return f"Error: {e}"

    return PlatformTool(name=name, description=description, args_schema=schema)


def platform_tools(*servers: str) -> list[BaseTool]:
    """Tools of the given MCP servers (all servers when none is given); empty unless TOOLSET=platform."""
    if os.getenv("TOOLSET", "core").lower() != "platform":
        return []
    return [_make_tool(*spec) for s in (servers or SERVERS) for spec in SERVERS[s]]
