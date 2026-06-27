import json
import re
from typing import Any


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_STRING_FIELD_TEMPLATE = r'"{field}"\s*:\s*"((?:\\.|[^"\\])*)"'
_LIST_FIELD_TEMPLATE = r'"{field}"\s*:\s*(\[[^\]]*\])'
_OBJECT_FIELD_TEMPLATE = r'"{field}"\s*:\s*(\{{.*?\}})'


def _salvage_partial_object(candidate: str) -> dict[str, Any] | None:
    result: dict[str, Any] = {}

    for field in ("answer", "unit", "explanation", "z3_code", "python_code"):
        match = re.search(_STRING_FIELD_TEMPLATE.format(field=re.escape(field)), candidate, flags=re.DOTALL)
        if match:
            try:
                result[field] = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError:
                result[field] = match.group(1)

    for field in ("premises_used",):
        match = re.search(_LIST_FIELD_TEMPLATE.format(field=re.escape(field)), candidate, flags=re.DOTALL)
        if match:
            try:
                result[field] = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    for field in ("reasoning",):
        match = re.search(_OBJECT_FIELD_TEMPLATE.format(field=re.escape(field)), candidate, flags=re.DOTALL)
        if match:
            try:
                result[field] = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    return result or None


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response.

    The prompts ask for JSON-only, but this protects us from accidental markdown fences
    or extra prose. Raises ValueError if no object can be parsed.
    """
    if not text or not text.strip():
        raise ValueError("empty LLM response")

    candidate = text.strip()
    fence = _CODE_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start == -1:
        raise ValueError("no JSON object found")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = candidate[start : idx + 1]
                parsed = json.loads(obj)
                if not isinstance(parsed, dict):
                    raise ValueError("parsed JSON is not an object")
                return parsed

    salvaged = _salvage_partial_object(candidate[start:])
    if salvaged is not None:
        return salvaged

    raise ValueError("unterminated JSON object")
