from __future__ import annotations

import json
import os
import time
from json import JSONEncoder
from pathlib import Path
from typing import Any

_LOG_PATH: str | None = None


def get_log_path() -> str:
    global _LOG_PATH
    if _LOG_PATH is None:
        _LOG_PATH = os.environ.get("LOG_FILE_PATH", "/root/exact/logs/requests.jsonl")
    return _LOG_PATH


class _LogEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def log_entry(entry: dict[str, Any]) -> None:
    entry["_timestamp"] = time.time()
    line = json.dumps(entry, cls=_LogEncoder, ensure_ascii=False)
    path = get_log_path()
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(line)
            f.write("\n")
    except OSError:
        pass  # Silently skip if log path is unwritable (e.g. during local tests)
