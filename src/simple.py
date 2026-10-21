"""
Simple Agent (Ablation Baseline)
=======================================
Single agent with all tools, no intent routing, no tool boundaries.
Used for benchmark comparison against the Flow-first architecture.
"""

import os

os.environ.setdefault("LITELLM_LOG", os.getenv("LITELLM_LOG", "CRITICAL"))

from crewai import Agent, Crew, Task
from crewai_tools import FileReadTool

from .agent import build_context, clean_response, resolve_llm, VERBOSE
from .tools import (
    ApplyManifestTool,
    GrafanaAlertsTool,
    InferenceClientTool,
    KubectlReadTool,
    KubectlWriteTool,
    ModelVersionTool,
    PrometheusQueryTool,
    YAMLValidatorTool,
)

ALL_TOOLS = [
    KubectlReadTool(),
    KubectlWriteTool(),
    ApplyManifestTool(),
    YAMLValidatorTool(),
    FileReadTool(),
    ModelVersionTool(),
    InferenceClientTool(),
    PrometheusQueryTool(),
    GrafanaAlertsTool(),
]


def run_simple(llm, user_input: str, history_context: str = "") -> str:
    agent = Agent(
        role="MLOps Agent",
        goal="Handle any Kubernetes and ML operations request by calling tools — never reason about cluster state without a tool call",
        backstory=(
            "You are a full-stack MLOps agent. ALWAYS use tools to get live data — never guess or assume cluster state.\n"
            "Tool usage rules:\n"
            "- kubectl_read: for ALL read operations (get, describe, logs, top). Call first before any mutation.\n"
            "- kubectl_write: for mutations (scale, delete, rollout). Always verify with kubectl_read after.\n"
            "- model_version_update: REQUIRED for model version switches. Do not use kubectl_write for this.\n"
            "- inference_client: REQUIRED for any ML service call. "
            "NodePort 30081: GET /health, POST /predict (payload '{}' uses random input), GET /metrics.\n"
            "- prometheus_query: REQUIRED for PromQL/Prometheus metric queries.\n"
            "- grafana_alerts: REQUIRED for Grafana alert state.\n"
            "- apply_manifest: REQUIRED for applying YAML manifests to the cluster.\n"
            "- yaml_validator: REQUIRED before applying any generated YAML.\n"
            "Never skip a tool call when the task requires live data or cluster mutation."
        ),
        llm=llm,
        tools=ALL_TOOLS,
        verbose=VERBOSE,
        max_iter=25,
        cache=False,  # tools read live cluster state; a cached result would be stale
    )
    prompt = history_context + user_input if history_context else user_input
    task = Task(
        description=prompt,
        expected_output="Clear, actionable response with verification from tool output",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=VERBOSE)
    return clean_response(str(crew.kickoff()))


def main() -> None:
    llm = resolve_llm()
    history: list[dict] = []
    keep = int(os.getenv("KEEP_MESSAGES", "5"))

    print("MLOps Simple Agent (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        context = build_context(history)
        try:
            result = run_simple(llm, user_input, context)
        except Exception as e:
            result = f"Error: {e}"

        history.append({"user": user_input, "short": result[:200], "full": result[:800]})
        if len(history) > keep + 2:
            history.pop(0)

        print(f"\nAgent: {result}\n")


if __name__ == "__main__":
    main()
