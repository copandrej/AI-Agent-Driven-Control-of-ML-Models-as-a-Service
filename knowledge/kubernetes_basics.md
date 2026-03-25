# Kubernetes Cluster Basics

## Core Architecture

A Kubernetes cluster consists of a control plane and worker nodes. The control plane manages scheduling, API access, and cluster state via etcd. Worker nodes run kubelet and a container runtime to host application pods.

## Key Resources

- **Pod**: Smallest deployable unit — one or more containers sharing network and storage.
- **Deployment**: Manages pod replicas with rolling updates and rollback support.
- **Service**: Stable network endpoint for a set of pods. Types: ClusterIP (internal), NodePort (external via node port), LoadBalancer (cloud provider).
- **Ingress**: HTTP/HTTPS routing rules that map hostnames and paths to backend services.
- **ConfigMap / Secret**: Inject configuration or sensitive data into pods as env vars or mounted files.
- **Namespace**: Logical isolation boundary for resources within a cluster.

## Common Operations

- `kubectl get pods -A` — list all pods across namespaces.
- `kubectl describe pod <name> -n <ns>` — detailed pod status, events, and conditions.
- `kubectl logs <pod> -n <ns>` — stream container logs.
- `kubectl scale deployment/<name> --replicas=N` — adjust replica count.
- `kubectl rollout status deployment/<name>` — watch a deployment rollout.
- `kubectl apply -f manifest.yaml` — declaratively apply a resource manifest.

## ML Inference on Kubernetes

ML models are typically served as HTTP microservices inside pods. Key concerns:

- **Resource requests/limits**: ML models often need large memory allocations even when idle. Always set explicit memory requests.
- **GPU scheduling**: Use `nvidia.com/gpu` resource requests and the NVIDIA device plugin DaemonSet for GPU-accelerated inference.
- **Health probes**: Use `/health` or `/readyz` endpoints as liveness and readiness probes so Kubernetes can route traffic only to ready model servers.
- **Scaling strategy**: Horizontal Pod Autoscaler (HPA) can scale on CPU, memory, or custom metrics like inference latency (`ml_request_latency_seconds`) and request rate (`ml_requests_total`).
- **Model versioning**: Update the container image tag in the deployment spec to switch model versions. Use rolling updates to avoid downtime.

## Monitoring with Prometheus

Prometheus scrapes `/metrics` endpoints from pods. Common patterns:

- Counter: `ml_requests_total` — total inference requests, labeled by `model_version` and `predicted_class`.
- Histogram: `ml_request_latency_seconds` — inference latency distribution.
- PromQL: `rate(ml_requests_total[5m])` for request rate, `histogram_quantile(0.95, rate(ml_request_latency_seconds_bucket[5m]))` for p95 latency.
- ServiceMonitor CRDs tell Prometheus which services to scrape.
