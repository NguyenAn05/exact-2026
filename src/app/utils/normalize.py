from __future__ import annotations

import re
from typing import Any
import json
import math

from app.logging_utils import log_entry
from app.schemas import PredictRequest, PredictResponseItem

_ASCII_UNIT_REPLACEMENTS = {
    "Ω": "ohm",
    "Ω": "ohm",
    "\u03a9": "ohm",
    "μ": "u",
    "µ": "u",
    "°": "degree",
}


def normalize_unit(unit: Any) -> str:
    value = "" if unit is None else str(unit).strip()
    for src, dst in _ASCII_UNIT_REPLACEMENTS.items():
        value = value.replace(src, dst)
    value = value.replace(" ", "")
    return value


def _strip_embedded_unit(answer: str, unit: str) -> str:
    normalized_unit = normalize_unit(unit)
    if not normalized_unit:
        return answer
    stripped = answer.strip()
    for candidate in {normalized_unit, normalized_unit.replace("ohm", "Ω")}:
        if candidate and stripped.endswith(candidate):
            stripped = stripped[: -len(candidate)].strip()
    return stripped


def _normalize_type2_answer(answer: str, unit: str) -> str:
    stripped = _strip_embedded_unit(answer, unit)
    try:
        value = float(stripped)
    except ValueError:
        return stripped

    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(value)))

    magnitude = abs(value)
    if magnitude != 0 and (magnitude >= 1e4 or magnitude < 1e-3):
        exponent = int(math.floor(math.log10(magnitude)))
        mantissa = value / (10 ** exponent)
        mantissa_text = f"{mantissa:.2f}"
        return f"{mantissa_text} × 10^{exponent}"

    text = format(value, ".10g")
    if "e" in text.lower():
        mantissa, exponent = re.split(r"[eE]", text, maxsplit=1)
        mantissa_value = float(mantissa)
        return f"{mantissa_value:.2f} × 10^{int(exponent)}"
    return text


def _normalize_choice(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _reasoning_text(raw: dict[str, Any]) -> str:
    """Extract only reasoning steps (FOL trace) for precise premise matching."""
    reasoning = raw.get("reasoning")
    if isinstance(reasoning, dict):
        steps = reasoning.get("steps")
        if isinstance(steps, list):
            return " ".join(_norm(str(s)) for s in steps)
    return ""


def _all_text(raw: dict[str, Any]) -> str:
    parts: list[str] = []
    exp = raw.get("explanation")
    if exp:
        parts.append(_norm(str(exp)))
    reasoning = raw.get("reasoning")
    if isinstance(reasoning, dict):
        steps = reasoning.get("steps")
        if isinstance(steps, list):
            parts.extend(_norm(str(s)) for s in steps)
    return " ".join(parts)


def _norm(text: str) -> str:
    """Normalize text for matching: lowercase, underscores→spaces."""
    return text.lower().replace("_", " ").replace("-", " ")


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Check if keyword (handling underscore/space equivalence) appears in text."""
    nk = _norm(keyword)
    nt = _norm(text)
    return nk in nt


def audit_premises_used(
    req: PredictRequest,
    raw: dict[str, Any],
    premises_used: list[int],
) -> list[int]:
    answer = str(raw.get("answer", "")).strip()
    text = _all_text(raw)
    premises = req.premises
    result = set(premises_used)

    # --- Rule 1: Uncertain + denial premise → only denial premise ---
    if _normalize_choice(answer) == "uncertain":
        denial_idx = [
            i for i, p in enumerate(premises)
            if any(kw in p.lower() for kw in [
                "no premise", "nothing", "not stated",
                "does not state", "not mention", "no information",
            ])
        ]
        if denial_idx:
            return denial_idx

    # --- Rule 2: Include premise referenced by index in text ---
    for i in range(len(premises)):
        if f"premise {i}" in text or f"(premise {i})" in text:
            result.add(i)

    # --- Rule 3: If answer is a number, prefer premise containing that number ---
    num_match = re.search(r"-?\d+(?:\.\d+)?", answer)
    if num_match:
        ans_num = num_match.group()
        for i, p in enumerate(premises):
            if ans_num in p:
                result.add(i)

    # --- Use reasoning steps (FOL trace) as the primary matching target ---
    rtext = _reasoning_text(raw) or _all_text(raw)

    # Words too common across premises to be useful for matching
    _COMMON = frozenset({
        "that", "this", "with", "from", "have", "been",
        "then", "than", "were", "will", "would", "could",
        "should", "must", "shall", "into", "over", "such",
        "each", "also", "after", "before", "about", "between",
        "their", "there", "when", "what", "does", "doing",
        "some", "more", "most", "other", "only", "very",
        "just", "they", "them", "who", "which", "whose",
        "may", "can", "has", "had", "not", "are", "all",
        "every", "where", "while", "these", "those",
        "researcher", "study", "alpha", "asha",
    })

    # --- Rule 4: Cross-check each premise against reasoning steps ---
    referenced: set[int] = set()
    premise_conclusions: list[str | None] = []

    for i, p in enumerate(premises):
        np = _norm(p)
        low = p.lower()
        premise_conclusions.append(None)

        # For If-Then rules: check if distinctive conclusion words appear in reasoning
        if low.startswith("if ") and "then " in low:
            conclusion = low.split("then", 1)[-1].strip().rstrip(".").rstrip(",")
            premise_conclusions[i] = conclusion
            conc_distinct = {
                w for w in re.findall(r"[a-zA-Z][a-zA-Z]{2,}", _norm(conclusion))
                if w not in _COMMON
            }
            if not conc_distinct:
                continue
            r_matches = sum(1 for w in conc_distinct if re.search(rf"\b{re.escape(w)}\b", rtext))
            if r_matches >= 1 and r_matches / len(conc_distinct) >= 0.3:
                referenced.add(i)

        # For factual premises: check distinctive words against reasoning steps
        else:
            distinctive = {
                w for w in re.findall(r"[a-zA-Z][a-zA-Z]{3,}", np)
                if w not in _COMMON
            }
            if not distinctive:
                continue
            r_matches = sum(1 for kw in distinctive if re.search(rf"\b{re.escape(kw)}\b", rtext))
            if r_matches >= 1 and r_matches / len(distinctive) >= 0.35:
                referenced.add(i)

    # --- Rule 5: Chain rules via distinctive word overlap ---
    for i, conc in enumerate(premise_conclusions):
        if conc is None or i in referenced:
            continue
        conc_distinct = {
            w for w in re.findall(r"[a-zA-Z][a-zA-Z]{2,}", _norm(conc))
            if w not in _COMMON
        }
        if not conc_distinct:
            continue
        r_matches = sum(1 for w in conc_distinct if re.search(rf"\b{re.escape(w)}\b", rtext))
        if r_matches >= 1:
            referenced.add(i)

    # --- Rule 6: Filter result ---
    final: set[int] = set()
    for i in result:
        if i in referenced:
            final.add(i)
        elif premises[i].lower().startswith("if "):
            final.add(i)  # keep rules implicitly
        elif f"premise {i}" in text or f"(premise {i})" in text:
            final.add(i)

    # --- Rule 7: Add high-confidence premises the model missed ---
    for i in referenced:
        if i not in final:
            final.add(i)

    if not final:
        final = result  # fallback

    final_sorted = sorted(final)
    if final_sorted != sorted(premises_used):
        log_entry({
            "event": "premise_audit",
            "query_id": req.query_id,
            "model_gave": sorted(premises_used),
            "auditor_fixed": final_sorted,
        })

    return final_sorted


def coerce_response(req: PredictRequest, raw: dict[str, Any]) -> PredictResponseItem:
    answer = str(raw.get("answer", "")).strip()
    if answer.startswith("{") and answer.endswith("}"):
        try:
            parsed_answer = json.loads(answer)
            if isinstance(parsed_answer, dict):
                if "value" in parsed_answer:
                    answer = str(parsed_answer["value"]).strip()
                if not raw.get("unit") and parsed_answer.get("unit") is not None:
                    raw["unit"] = parsed_answer["unit"]
        except json.JSONDecodeError:
            pass
    explanation = str(raw.get("explanation", "")).strip()
    if not explanation:
        explanation = "The system produced the final answer using the available query fields."

    reasoning = raw.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, dict):
        reasoning = {"type": "raw", "steps": [str(reasoning)]}

    if req.type == "type1":
        # The competition requires an exact option string when options is non-empty.
        if req.options:
            normalized = {_normalize_choice(opt): opt for opt in req.options}
            if _normalize_choice(answer) in normalized:
                answer = normalized[_normalize_choice(answer)]
            elif "Uncertain" in req.options:
                answer = "Uncertain"
            else:
                answer = req.options[0]

        premises_used_raw = raw.get("premises_used", [])
        premises_used: list[int] = []
        if isinstance(premises_used_raw, list):
            for item in premises_used_raw:
                try:
                    idx = int(item)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(req.premises) and idx not in premises_used:
                    premises_used.append(idx)

        premises_used = audit_premises_used(req, raw, premises_used)

        return PredictResponseItem(
            query_id=req.query_id,
            answer=answer or ("Uncertain" if not req.options else req.options[0]),
            unit="",
            explanation=explanation,
            premises_used=premises_used,
            reasoning=reasoning,
        )

    # Type 2: answer should be numerical/string value only; unit goes separately.
    return PredictResponseItem(
        query_id=req.query_id,
        answer=_normalize_type2_answer(answer or "0", raw.get("unit", "")),
        unit=normalize_unit(raw.get("unit", "")),
        explanation=explanation,
        premises_used=[],
        reasoning=reasoning,
    )
