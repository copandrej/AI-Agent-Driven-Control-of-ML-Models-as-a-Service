# Master Thesis: AI-Agent-Driven Control of ML Models as a Service

Agentic system for managing Kubernetes cluster-based ML models through natural language, with a terminal chatbot interface.

## Architecture

- **Flow Backbone**: CrewAI `Flow` with deterministic routing (`deploy`, `ops`, `monitor`, `ingress`, `general`)
- **5 Specialized Agents**: Orchestrator, Coding, Ops, Monitoring, Ingress — each with bounded tool access
- **Ablation Baselines**: single-agent with all tools (`simple`), kubectl-only (`minimal`)
- **ML Service**: Pre-trained ImageNet classifiers (torchvision MobileNetV2/ResNet50, FastAPI, NodePort 30081)
- **Observability**: Prometheus and Grafana integration (when endpoints are configured)
- **Memory**: Optional scoped ChromaDB memory per agent
- **Autonomous Mode**: Watch loop for self-healing (Level 3)

## Project Structure

```
src/           — Agentic framework
  agent.py     — Flow-first architecture (5 agents, deterministic routing, all tools)
  simple.py    — Single-agent ablation baseline (all tools, no routing)
  minimal.py   — Fully ablated baseline (kubectl only)
  watch.py     — Autonomous watch mode (Level 3 self-healing loop)

eval/          — Evaluation framework
  tasks.yaml   — 25 benchmark tasks at L1/L2 complexity (L3 = watch_eval.py)
  runner.py    — Automated eval harness (flow/simple/minimal architectures)
  watch_eval.py — Fault injection evaluation for watch mode
  load_test.py — Async load tester for measuring latency

k8s/           — Cluster configuration & service manifests
  kind-config.yaml       — Kind cluster config (NodePort mappings)
  prometheus-values.yaml — Helm values for kube-prometheus-stack
  echo_service/          — Test workload (FastAPI matrix multiply, NodePort 30080)
  ml_service/            — ML inference service (torchvision, NodePort 30081)
  setup.sh               — All-in-one cluster bootstrap script

main.py        — Entrypoint → src/agent.py
watch.py       — Entrypoint → src/watch.py
test_agent.py  — Unit + integration tests
```

## Prerequisites

- Python 3.10+, Docker, kubectl, uv
- Kind (or any Kubernetes cluster), Helm
- Ollama (local LLM) or OpenRouter API key

## Setup

**1. Kind Cluster (builds images, installs Prometheus/Grafana, deploys all services):**
```bash
bash k8s/setup.sh
```

**2. Set up LLM** (Ollama example, or bring your own):
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull qwen3:8b
```

**3. Configure environment:**
```bash
cp .env.example .env
# Set up the variables yourself
```

**4. Install dependencies with uv:**
```bash
uv sync
```

**5. Run the agent:**
```bash
uv run python main.py
```

## Example Interactions

```
You: Show me all pods running in the cluster
Agent: Here are the running pods across all namespaces...

You: Update the ML service to model version v2
Agent: Updated ml-service image, switching MODEL_NAME to resnet50, rollout complete.

You: What is the P95 inference latency?
Agent: Current P95 latency from Prometheus: 0.032s
```

## Running Evaluations to Reproduce Results

```bash
uv run python eval/runner.py --archs flow --levels 1 2
uv run python eval/runner.py --archs simple --levels 1 2
uv run python eval/runner.py --archs minimal --levels 1
```

## Autonomous Watch Mode 

Monitors cluster health and ML service metrics, remediates issues automatically:

```bash
uv run python watch.py --interval 30 --dry-run
uv run python eval/watch_eval.py --all --output eval_watch_results.json
```

## References

- [CrewAI](https://docs.crewai.com/) 
- [Ollama](https://ollama.ai/) 
- [Kind](https://kind.sigs.k8s.io/) 
- [FastAPI](https://fastapi.tiangolo.com/)
