"""
Minimal Agent (Full Ablation Baseline)
=============================================
Single agent with only kubectl read/write tools.
No ML-specific tools, no intent routing, no YAML validation gate.
Represents the lowest capability baseline for ablation study.

Ablation study hierarchy:
  minimal  ← kubectl only (this file)
  simple   ← all tools, no routing
  flow     ← full architecture (src/agent.py)
"""

import os

os.environ.setdefault("LITELLM_LOG", os.getenv("LITELLM_LOG", "CRITICAL"))

from crewai import Agent, Crew, Task

from .agent import build_context, clean_response, resolve_llm, VERBOSE
from .tools import KubectlReadTool, KubectlWriteTool


def run_minimal(llm, user_input: str, history_context: str = "") -> str:
    agent = Agent(
        role="Kubernetes Agent",
        goal="Handle Kubernetes cluster operations by calling kubectl tools — never reason about state without a tool call",
        backstory=(
            "You are a Kubernetes agent with read and write access to the cluster. "
            "ALWAYS call a tool to get live cluster data — never assume or guess state.\n"
            "- kubectl_read: REQUIRED for ALL queries (get, describe, logs, top). Call this first.\n"
            "- kubectl_write: for mutations (create, scale, delete, rollout). Always follow with kubectl_read to verify.\n"
            "You have no access to ML service endpoints or monitoring tools. "
            "If asked about metrics or inference endpoints, report that those tools are unavailable."
        ),
        llm=llm,
        tools=[KubectlReadTool(), KubectlWriteTool()],
        verbose=VERBOSE,
        max_iter=15,
    )
    prompt = history_context + user_input if history_context else user_input
    task = Task(
        description=prompt,
        expected_output="Clear response based on kubectl tool output",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=VERBOSE)
    return clean_response(str(crew.kickoff()))


def main() -> None:
    llm = resolve_llm()
    history: list[dict] = []
    keep = int(os.getenv("KEEP_MESSAGES", "5"))

    print("MLOps Minimal Agent — kubectl only (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        context = build_context(history)
        try:
            result = run_minimal(llm, user_input, context)
        except Exception as e:
            result = f"Error: {e}"

        history.append({"user": user_input, "short": result[:200], "full": result[:800]})
        if len(history) > keep + 2:
            history.pop(0)

        print(f"\nAgent: {result}\n")


if __name__ == "__main__":
    main()
