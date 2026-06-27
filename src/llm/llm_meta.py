from __future__ import annotations

from typing import Any


def usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0)
    if total == 0:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def merge_usage(usages: dict[str, dict[str, int]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for step_usage in usages.values():
        totals["prompt_tokens"] += step_usage.get("prompt_tokens", 0)
        totals["completion_tokens"] += step_usage.get("completion_tokens", 0)
        totals["total_tokens"] += step_usage.get("total_tokens", 0)
    return totals
