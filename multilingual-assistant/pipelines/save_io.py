from dataclasses import dataclass, asdict
import json
import os
from app.config import settings

@dataclass
class in_content:
    In: str = ""
    Out: str = ""
    Ground: str = "" #NOTE need to be added manually

@dataclass
class out_content:
    In: str = ""
    Out: str = ""


def save_content_in(content: in_content):
    if os.path.exists(settings.TRANSLATE_IN_FILE):
        with open(settings.TRANSLATE_IN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
        
    data.append(asdict(content))
    
    with open(settings.TRANSLATE_IN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_content_out(content: out_content):
    if os.path.exists(settings.TRANSLATE_OUT_FILE):
        with open(settings.TRANSLATE_OUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
        
    data.append(asdict(content))
    
    with open(settings.TRANSLATE_OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)