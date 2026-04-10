"""
Dual Agent (Two-Stage Baseline)
======================================
Main agent handles operations directly; delegates new resource creation
to a specialized deploy sub-agent via a structured DEPLOY_TASK marker.
"""

import json
import os
import re

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


def _run_main_agent(llm, prompt: str) -> str:
    agent = Agent(
        role="MLOps Main Agent",
        goal="Handle Kubernetes and ML operations directly using tools. For new resource creation from scratch, emit a delegation spec.",
        backstory=(
            "You are an MLOps agent. Rules:\n"
            "- ALWAYS call a tool before reporting cluster state — never guess.\n"
            "- After every kubectl_write call, immediately call kubectl_read to verify the change took effect before concluding.\n"
            "- For simple mutations (scale, delete, patch, rollout restart, version switch, apply existing manifest): handle directly with tools.\n"
            "- For monitoring, metrics, health checks, logs, describe: handle directly with tools.\n"
            "- ONLY when a brand-new Kubernetes resource needs to be created FROM SCRATCH (new Deployment/Service/ConfigMap "
            "that does not yet exist AND no manifest file is specified): output exactly one line "
            "`DEPLOY_TASK: {JSON}` where JSON is a spec dict with relevant fields (name, image, replicas, namespace, port, etc.) "
            "extracted from the user request — then stop. Do NOT attempt to create it yourself.\n"
            "- If the user provides a manifest file path (e.g. 'apply k8s/ml_service/deployment.yaml'), "
            "use kubectl_write with `apply -f <path>` directly — do NOT emit DEPLOY_TASK."
        ),
        llm=llm,
        tools=[
            KubectlReadTool(),
            KubectlWriteTool(),
            ModelVersionTool(),
            InferenceClientTool(),
            PrometheusQueryTool(),
            GrafanaAlertsTool(),
        ],
        verbose=VERBOSE,
        max_iter=25,
    )
    task = Task(
        description=prompt,
        expected_output="Clear response based on tool output, or DEPLOY_TASK: {spec} if new resource creation is needed",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=VERBOSE)
    return clean_response(str(crew.kickoff()))


def _extract_deploy_spec(text: str) -> dict | None:
    match = re.search(r'DEPLOY_TASK:\s*(\{.*?\})', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _run_deploy_subagent(llm, spec: dict) -> str:
    agent = Agent(
        role="Deploy Sub-Agent",
        goal="Generate a valid Kubernetes manifest from the given spec, validate it, apply it to the cluster, and verify the result.",
        backstory=(
            "You are a Kubernetes deploy specialist. Follow this exact sequence:\n"
            "1. Use file_read to check k8s/ directory for an existing template manifest that matches the resource type "
            "(e.g. k8s/ml_service/deployment.yaml). Use it as a structural reference if found.\n"
            "2. Generate the YAML manifest from the provided spec. Include apiVersion, kind, metadata, spec. "
            "For a Deployment always include a matching Service.\n"
            "3. REQUIRED: call yaml_validator on the generated YAML. If INVALID, fix the YAML and validate again before proceeding.\n"
            "4. REQUIRED: call apply_manifest with the validated YAML.\n"
            "5. REQUIRED: call kubectl_read to verify the resource exists (e.g. `get deployment <name> -n <namespace>`). "
            "Report the actual output.\n"
            "6. Never report success without completing steps 3-5."
        ),
        llm=llm,
        tools=[
            FileReadTool(),
            YAMLValidatorTool(),
            ApplyManifestTool(),
            KubectlReadTool(),
        ],
        verbose=VERBOSE,
        max_iter=25,
    )
    task = Task(
        description=f"Deploy the following resource spec to the Kubernetes cluster:\n{json.dumps(spec, indent=2)}\n\nFollow the generate→validate→apply→verify sequence strictly.",
        expected_output="Apply result and kubectl verification confirming the resource is running",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=VERBOSE)
    return clean_response(str(crew.kickoff()))


def run_dual(llm, user_input: str, history_context: str = "") -> str:
    prompt = history_context + user_input if history_context else user_input
    result = _run_main_agent(llm, prompt)
    spec = _extract_deploy_spec(result)
    if spec is not None:
        return _run_deploy_subagent(llm, spec)
    return result


def main() -> None:
    llm = resolve_llm()
    history: list[dict] = []
    keep = int(os.getenv("KEEP_MESSAGES", "5"))

    print("MLOps Dual Agent (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        context = build_context(history)
        try:
            result = run_dual(llm, user_input, context)
        except Exception as e:
            result = f"Error: {e}"

        history.append({"user": user_input, "short": result[:200], "full": result[:800]})
        if len(history) > keep + 2:
            history.pop(0)

        print(f"\nAgent: {result}\n")


if __name__ == "__main__":
    main()
