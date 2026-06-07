import json
from pathlib import Path
from typing import Dict


def load_json(path: str) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Json file not found: {path}")
    with path.open() as f:
        return json.load(f)


def output_json(data, output_path):
    with open(output_path, "w") as outputfile:
        json.dump(data, outputfile, indent=4)


def output_jsonl(records, filepath):
    """
    records: iterable of dicts
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
