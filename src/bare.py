"""
Bare Agent (Bare Baseline)
==========================
Single ReAct agent with one unrestricted kubectl tool and Prometheus.
No read/write separation, no ML tools, no routing, no domain prompt.
Evaluation baseline only.
"""

import os

os.environ.setdefault("LITELLM_LOG", os.getenv("LITELLM_LOG", "CRITICAL"))

from crewai import Agent, Crew, Task

from .agent import VERBOSE, clean_response
from .tools import KubectlTool, PrometheusQueryTool


def run_bare(llm, user_input: str, history_context: str = "") -> str:
    agent = Agent(
        role="Kubernetes Agent",
        goal="Complete the user's request on the Kubernetes cluster",
        backstory="You operate a Kubernetes cluster using kubectl and Prometheus.",
        llm=llm,
        tools=[KubectlTool(), PrometheusQueryTool()],
        verbose=VERBOSE,
        max_iter=25,
        cache=False,  # tools read live cluster state; a cached result would be stale
    )
    prompt = history_context + user_input if history_context else user_input
    task = Task(description=prompt, expected_output="Response based on tool output", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=VERBOSE)
    return clean_response(str(crew.kickoff()))
