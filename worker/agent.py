import json
import os
import urllib.request
import urllib.error
from pathlib import Path

role = os.environ["ROLE"]
model = os.environ.get("MODEL", "openai/gpt-4o")
focus = os.environ.get("FOCUS", "invention discovery")
token = os.environ.get("GITHUB_TOKEN", "")
endpoint = "https://models.github.ai/inference/chat/completions"

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
    "model": model,
    "focus": focus,
    "provider": "github-models",
    "endpoint": endpoint,
    "inference_success": False,
    "status": "EXTERNAL_INFERENCE_FAILED",
}

try:
    if not token:
        raise RuntimeError("GITHUB_TOKEN missing")

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "CLAIM <= EVIDENCE. UNKNOWN REMAINS UNKNOWN. VERIFY BEFORE COMMIT."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 1200,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))

    choices = body.get("choices") or []
    answer = ""
    if choices:
        answer = ((choices[0].get("message") or {}).get("content") or "").strip()

    if not answer:
        raise RuntimeError(f"Model returned no text; keys={sorted(body.keys())}")

    out["inference_success"] = True
    out["status"] = "UNREVIEWED_EXTERNAL_AGENT_OUTPUT"
    out["output"] = answer
    out["response_id"] = body.get("id")
    out["usage"] = body.get("usage")

except urllib.error.HTTPError as exc:
    try:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
    except Exception:
        detail = ""
    out["error"] = f"HTTPError({exc.code}): {detail}"
except Exception as exc:
    out["error"] = repr(exc)

Path("results").mkdir(exist_ok=True)
Path(f"results/{role}.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(json.dumps({
    "role": role,
    "model": model,
    "status": out["status"],
    "inference_success": out["inference_success"],
}, ensure_ascii=False))
