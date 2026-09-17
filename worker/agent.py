import json
import os
from pathlib import Path
from gradio_client import Client

role = os.environ["ROLE"]
space = os.environ.get("MODEL", "huggingface-projects/llama-3.2-3B-Instruct")
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
""".strip()

out = {
    "role": role,
    "model": space,
    "focus": focus,
    "provider": "huggingface-space-zerogpu",
    "api_name": "/generate",
    "inference_success": False,
    "status": "EXTERNAL_INFERENCE_FAILED",
}

try:
    client = Client(space)
    # Gradio 6 ChatInterface defaults to the function name. Both selected
    # public Spaces use fn=generate(message, history, max_new_tokens,
    # temperature, top_p, top_k, repetition_penalty).
    answer = client.predict(
        prompt,
        [],
        512,
        0.4,
        0.9,
        50,
        1.15,
        api_name="/generate",
    )
    if isinstance(answer, (list, tuple)):
        text = "\n".join(str(x) for x in answer if x is not None).strip()
    else:
        text = str(answer or "").strip()
    if not text:
        raise RuntimeError("External model returned empty output")
    out["inference_success"] = True
    out["status"] = "UNREVIEWED_EXTERNAL_AGENT_OUTPUT"
    out["output"] = text
except Exception as exc:
    out["error"] = repr(exc)

Path("results").mkdir(exist_ok=True)
Path(f"results/{role}.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(json.dumps({
    "role": role,
    "model": space,
    "status": out["status"],
    "inference_success": out["inference_success"],
}, ensure_ascii=False))
