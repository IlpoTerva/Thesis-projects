import json
import re
from pathlib import Path

from app.config import settings
from pipelines.build_vector_index import _parse_pdf, RAW_DIR_CASES
from app.chat_lang import LLM_PLAIN



HEADING_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+(\S.*)$")
CONTEXT_MARKERS = ("Fault description", "Alarm description", "Remedy", "Diagnostic steps")

PROMPT = """Convert this troubleshooting manual section into JSON.

Rules:
- If the section lists "Case 1", "Case 2", ... make one entry per case with id "{sid}-<n>".
  If it is split by alarm codes instead (e.g. A05/F30001), make one entry per code group with id "{sid}-<code>".
  Otherwise make one entry with id "{sid}".
- "symptom": one short sentence describing what the operator OBSERVES, based only on the fault/alarm description.
- "alarm_codes": every alarm code that identifies this entry (e.g. "025000", "700021", "A08", "F30001"). Empty list if none.
- "steps": the diagnostic/remedy steps for this entry, in order, concise. Keep machine data numbers (MD30200),
  interface names (X54) and values exactly as written.
- Use ONLY information in the section. Do not add anything.

Return ONLY a JSON array of objects with keys: id, symptom, alarm_codes, steps.

Section {sid} {title}:
{text}
"""
def _split_sections(pdf: Path) -> list[dict]:
    sections, current = [], None
    for text, page, _ in _parse_pdf(pdf):
        for line in text.splitlines():
            m  = HEADING_RE.match(line.strip())
            if m and "...." not in line:
                if current:
                    sections.append(current)
                current = {"sid": m.group(1), "title": m.group(2).strip(), "page": page, "lines": []}
            elif current:
                current["lines"].append(line)
    if current:
        sections.append(current)

    return [
        s for s in sections
        if s["sid"].split(".")[0] in {"3", "4", "5"}
        and any(mk in "\n".join(s["lines"]) for mk in CONTEXT_MARKERS)
    ]     

def _parse_json(text:str) -> list[dict]|None:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    try:
        return json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        return None


def main():
    pdf = next(RAW_DIR_CASES.glob("*.pdf"))
    draft, failed = [], []

    for s in _split_sections(pdf=pdf):
        body = "\n".join(s["lines"])
        resp = LLM_PLAIN.invoke(PROMPT.format(sid=s["sid"], title=s["title"], text=body))
        entries = _parse_json(resp.content or "")
        if not entries:
            failed.append(f"{s['sid']} {s['title']}")
            continue
        for e in entries:
            e["section"] = f"{s['sid']} {s['title']}"
            e["page"] = s["page"]
            e["source"] = pdf.name
            e["alarm_text"] = ""
            draft.append(e)
        print(f"  {s['sid']} {s['title']}: {len(entries)} cases")

    out = Path(settings.CASES_JSON).with_name("cases.draft.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(draft)} entries to {out}")
    if failed:
        print(f"Failed - Do these by hand:" *failed, sep="\n ")

