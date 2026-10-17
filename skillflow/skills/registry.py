import json
from pathlib import Path


def load_skills():
    return {data["name"]: data for path in Path(__file__).parent.glob("*.json")
            if (data := json.loads(path.read_text(encoding="utf-8")))}
