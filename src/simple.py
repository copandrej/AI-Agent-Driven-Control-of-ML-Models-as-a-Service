"""
MLOps Simple Agent (Ablation Baseline)
=======================================
Single agent with all tools, no intent routing, no tool boundaries.
Used for benchmark comparison against the Flow-first architecture.
"""

import os

os.environ.setdefault("LITELLM_LOG", os.getenv("LITELLM_LOG", "CRITICAL"))

from crewai import Agent, Crew, Task
from crewai_tools import FileReadTool

from .agent import (
    ApplyManifestTool,
    GrafanaAlertsTool,
    InferenceClientTool,
    KubectlReadTool,
    KubectlWriteTool,
    ModelVersionTool,
    PrometheusQueryTool,
    YAMLValidatorTool,
    build_context,
    clean_response,
    resolve_llm,
    VERBOSE,
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
        goal="Handle any Kubernetes and ML operations request",
        backstory=(
            "You are a full-stack MLOps agent with access to all tools. "
            "You can read and mutate the cluster, deploy manifests, manage ML model versions, "
            "test inference endpoints, and query Prometheus metrics. "
            "The ML inference service runs on NodePort 30081 with /health, /predict, /metrics endpoints. "
            "Always verify your actions with kubectl_read after mutations."
        ),
        llm=llm,
        tools=ALL_TOOLS,
        verbose=VERBOSE,
        max_iter=15,
    )
    prompt = history_context + user_input if history_context else user_input
    task = Task(
        description=prompt,
        expected_output="Clear, actionable response with verification",
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
