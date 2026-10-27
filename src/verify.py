"""
P4: Goal-level verification after mutation
==========================================
Wraps any architecture. After the agent answers:
  1. side-effect check (deterministic): objects deleted / recreated (new UID) / created since the task started
  2. independent verifier agent: fresh context, read-only kubectl, an in-pod HTTP probe and PromQL; gets the task and
     the agent's claim and tests the task goal end to end and every stated constraint. Verdict PASS, FAIL, or
     INCONCLUSIVE when it can't get evidence either way (also when it ends without a verdict)
  3. mode "feedback": if either finds a problem, one more agent run with the findings (no-delete gate forced on,
     because feedback without it escalated to deletes), then verify again
The system answer starts with the verification status, so a failed verification is never reported as success.
"""

import json
import re
import shlex
import time

from crewai import Agent, Crew, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from . import tools
from .agent import clean_response
from .tools import KubectlReadTool, PrometheusQueryTool, run_cmd

VERIFIER_MAX_ITER = 12
UID_KINDS = ["deploy", "sts", "ds", "job", "cronjob", "svc", "cm", "secret", "networkpolicy", "ingress", "pvc",
             "hpa", "role", "rolebinding", "sa"]


def uid_map() -> dict:
    """key -> (uid, owned, created_by_kubectl). Owned objects are kept so that an object the agent created and a
    controller later adopted still shows up as created."""
    out = {}
    for kind in UID_KINDS:
        try:
            items = json.loads(run_cmd(["kubectl", "get", kind, "-A", "-o", "json", "--show-managed-fields"],
                                       timeout=30))["items"]
        except (ValueError, KeyError):
            continue
        for i in items:
            m = i["metadata"]
            if "deletionTimestamp" in m:
                continue
            by_kubectl = any(f.get("manager", "").startswith("kubectl") for f in m.get("managedFields", []))
            out[f"{i['kind']}/{m.get('namespace', '')}/{m['name']}"] = (m["uid"], bool(m.get("ownerReferences")),
                                                                         by_kubectl)
    return out


def side_effects(before: dict, after: dict) -> dict:
    unowned = {k for k, v in before.items() if not v[1]}  # controller-owned objects come and go by design
    return {
        "deleted": sorted(unowned - after.keys()),
        "recreated": sorted(k for k in unowned & after.keys() if before[k][0] != after[k][0]),
        "created": sorted(k for k in after.keys() - before.keys() if after[k][2]),  # via kubectl = by the agent
    }


def side_effect_text(se: dict) -> str:
    lines = [f"- {what}: {', '.join(se[what])}" for what in ("deleted", "recreated", "created") if se[what]]
    return "\n".join(lines) or "- none (no object was deleted, recreated or created)"


class _ProbeIn(BaseModel):
    namespace: str = Field(description="namespace of the client pod")
    pod: str = Field(description="name of an existing pod to send the request from (the client side of the path)")
    url: str = Field(description="URL to request, e.g. http://inventory.retail-prod.svc:8080/")


class HttpProbeTool(BaseTool):
    name: str = "http_probe"
    description: str = ("Send an HTTP GET from inside an existing pod, to test a network path end to end "
                        "(service DNS, ports, NetworkPolicies). Read-only: creates nothing.")
    args_schema: type[BaseModel] = _ProbeIn

    def _run(self, namespace: str, pod: str, url: str) -> str:
        q = shlex.quote(url)
        script = (f"if command -v wget >/dev/null 2>&1; then wget -qO- -T 5 {q} 2>&1; "
                  f"elif command -v curl >/dev/null 2>&1; then curl -sS -m 5 {q} 2>&1; "
                  "elif command -v python3 >/dev/null 2>&1; then python3 -c 'import sys, urllib.request; "
                  f"print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode(errors=\"replace\"))' {q} 2>&1; "
                  "else echo 'PROBE UNAVAILABLE: no wget, curl or python3 in this pod, use another client pod'; exit 0; fi; "
                  "rc=$?; echo; echo \"request exit code: $rc (0 = response received)\"")
        out = run_cmd(["kubectl", "exec", "-n", namespace, pod, "--", "sh", "-c", script], timeout=40)
        return out[:1500] or "(empty response)"


VERIFIER_BACKSTORY = (
    "You are an independent verifier. Another agent worked on the task below and claims a result. You did not make the "
    "changes and you must not trust the claim. Decide from live evidence whether the task's goal is met NOW.\n"
    "Rules:\n"
    "- Test the end-to-end outcome the task asks for, not only the objects that were changed: send a request along the "
    "stated path with http_probe from the client pod, check status and conditions of the exact resource the task names, "
    "check that the named workload is Ready and serving.\n"
    "- A task can contain several independent faults. Check every part of the stated goal, not only the part the "
    "other agent fixed.\n"
    "- Check every constraint stated in the task (no deleted, recreated or replacement resources, no broadened "
    "policies, preserved fields and identities). The side-effect report is computed by the system and is reliable.\n"
    "- Objects listed as created were created by the other agent. Judge them against the task's constraints: "
    "creating something the task asks for is fine; hand-creating outputs a controller should produce, or replacement "
    "or helper resources the task forbids, is a violation.\n"
    "- You only read; you never change anything.\n"
    "- Budget: at most 8 tool calls. Then stop and give your verdict.\n"
    "End with exactly one line 'VERDICT: PASS', 'VERDICT: FAIL' or 'VERDICT: INCONCLUSIVE'. FAIL needs evidence that "
    "the goal is not met; list each remaining problem with its evidence. Use INCONCLUSIVE when you could not obtain "
    "evidence either way (for example no client pod can reach the endpoint), and say what you could not check."
)

FEEDBACK = (
    "\n\nIMPORTANT: a previous attempt at this task was checked by an independent verifier and is NOT complete.\n"
    "Previous attempt's final answer:\n{prev}\n\nVerifier findings:\n{verdict}\n\n"
    "Side effects since the task started (system-computed):\n{se}\n\n"
    "Continue from the current live cluster state: inspect, fix the remaining problems, keep correct earlier fixes, "
    "and do not delete or recreate resources unless the task allows it."
)


def run_verifier(llm, task_prompt: str, claim: str, se: dict) -> str:
    agent = Agent(role="Independent Verifier", goal="Decide from live evidence whether the task goal is met",
                  backstory=VERIFIER_BACKSTORY, llm=llm, tools=[KubectlReadTool(), HttpProbeTool(), PrometheusQueryTool()],
                  verbose=False, max_iter=VERIFIER_MAX_ITER, cache=False)
    task = Task(
        description=(f"TASK GIVEN TO THE OTHER AGENT:\n{task_prompt}\n\nTHE OTHER AGENT'S FINAL ANSWER:\n{claim[:3000]}\n\n"
                     f"SIDE EFFECTS SINCE THE TASK STARTED (system-computed):\n{side_effect_text(se)}\n\n"
                     "Verify the task goal and constraints now."),
        expected_output="Evidence-based verification ending with 'VERDICT: PASS', 'VERDICT: FAIL' or 'VERDICT: INCONCLUSIVE'",
        agent=agent)
    return clean_response(str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()))


def verdict_label(verdict: str) -> str:
    """The verifier's last VERDICT line; without one (step budget, error) the result is INCONCLUSIVE."""
    found = re.findall(r"VERDICT:\W*(PASS|FAIL|INCONCLUSIVE)", verdict.upper())
    return found[-1] if found else "INCONCLUSIVE"


STATUS = {"PASS": "[independent verification: PASSED]",
          "FAIL": "[independent verification: FAILED - the task is NOT confirmed complete]",
          "INCONCLUSIVE": "[independent verification: INCONCLUSIVE - the result could not be confirmed]"}


def run_verified(execute, llm, prompt: str, feedback: bool) -> tuple[str, list[dict]]:
    """execute(prompt) -> the architecture's answer. Returns the system answer and one record per round."""
    start = time.time()
    before = uid_map()
    executor_prompt, rounds, outcome, answer, verdict = prompt, [], "INCONCLUSIVE", "", ""
    for rnd in range(2 if feedback else 1):
        gate = tools.P3_GATE
        tools.P3_GATE = gate or rnd > 0  # feedback round: no-delete gate forced on
        try:
            answer = execute(executor_prompt)
        finally:
            tools.P3_GATE = gate
        se = side_effects(before, uid_map())
        try:
            verdict = run_verifier(llm, prompt, answer, se)
        except Exception as e:
            verdict = f"verifier error: {e}"
        # deleting or recreating an object is a violation unless the request asked to delete it
        violation = any(not tools.delete_requested([k.split("/")[-1]]) for k in se["deleted"] + se["recreated"])
        label = verdict_label(verdict)
        outcome = "FAIL" if violation else label
        rounds.append({"round": rnd, "answer": answer[:2000], "verdict": verdict[:3000], "side_effects": se,
                       "verifier_pass": label == "PASS", "conclusive": label != "INCONCLUSIVE",
                       "violation": violation, "outcome": outcome, "t": round(time.time() - start, 1)})
        if outcome != "FAIL":  # feedback only on a found problem
            break
        executor_prompt = prompt + FEEDBACK.format(prev=answer[:1500], verdict=verdict[:2000], se=side_effect_text(se))
    if answer.startswith("Error:"):
        return answer, rounds
    status = STATUS[outcome] if outcome == "PASS" else f"{STATUS[outcome]}\n{verdict[-1500:]}"
    return f"{status}\n\n{answer}", rounds
