import json, os
from pathlib import Path
from gradio_client import Client

role = os.environ["ROLE"]
model = os.environ["MODEL"]
focus = os.environ.get("FOCUS", "invention discovery")

prompt = f"""
You are role {role} in CEREBRON Omega Farm 32 Invention Discovery.
Focus: {focus}.
Generate or audit invention/discovery candidates with strict evidence discipline.
Separate: ESTABLISHED / DERIVED / HYPOTHESIS / SPECULATIVE.
For each candidate state: problem, mechanism, why it may work, assumptions, dependencies,
novelty risk, feasibility, manufacturability or implementability, cost drivers, safety risks,
critical unknowns, falsification test, minimum decisive experiment, transfer limits and failure modes.
Do not claim patentability or freedom-to-operate without evidence.
Do not call simulations tests. Unknown remains unknown.
Return a concise but technically useful report.
"""

out = {
    "role": role,
    "model": model,
    "focus": focus,
    "inference_success": False,
    "status": "EXTERNAL_INFERENCE_FAILED",
}

try:
    client = Client(model)
    answer = client.predict(prompt, api_name="/chat")
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        raise RuntimeError("External model returned an empty output")
    out["inference_success"] = True
    out["status"] = "UNREVIEWED_EXTERNAL_AGENT_OUTPUT"
    out["output"] = answer
except Exception as exc:
    out["error"] = repr(exc)

Path("results").mkdir(exist_ok=True)
Path(f"results/{role}.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
