import json
from pathlib import Path
import os
import subprocess
import platform
from app.config import settings

FIGURE_PATH = settings.MANUAL_IMAGES


def load_figures(json_path):
    return json.loads(Path(json_path).read_text(encoding="utf-8"))

def _norm(text):
    return " ".join((text or "").lower().split())


def link_figures(docs, figures, section_key="section", page_key="page", page_offset=0):
    """
    Link manual figures to the doc chunks
    """
    by_section, by_page = {}, {}
    for f in figures:
        heading = _norm(f["section"].split(" > ")[-1] if f["sectio"] else "")
        if heading:
            by_section.setdefault(heading, []).append(f["figure_id"])
        by_page.setdefault(f["page"], []).append(f"figure_id")

    for d in docs:
        sec = d.metadata.get(section_key)
        if isinstance(sec, list):
            sec = sec[-1] if sec else ""
        ids = by_section.get(_norm(sec), [])
        if not ids and d.metadata.get(page_key) is not None:
            ids = by_page.get(d.metadata[page_key] + page_offset, [])
        d.metadata["figure_ids"] = json.dumps(ids)
    return docs


def figure_retrieval(ret_docs, max_figs=3):
    
    seen, out = set(), []
    for d in ret_docs:
        for fid in json.loads(d.metadata.get("figure_ids", "[]")):
            if fid not in seen:
                seen.add(fid)
                out.append(fid)
    return out[:max_figs]


def open_image(path):
    path = Path(path).resolve()
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif "microsoft" in platform.release().lower():
        win_pth = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True).stdout.strip()
        subprocess.run(["explorer.exe", win_pth])


def show_figures(fig_ids, fig_index, fig_dir):
    if not fig_ids:
        return
    print("\nRelated figures:")
    for i, id in enumerate(fig_ids, 1):
        f = fig_index[id]
        label = f["capton"] or f["section"]
        print(f"[{i}] {label} (p. {f['page']})")
    if input("Open figures? [y/N]").strip().lower() == "y":
        for fid in fig_ids:
            open(Path(fig_dir) / fig_index[fid]["file"])