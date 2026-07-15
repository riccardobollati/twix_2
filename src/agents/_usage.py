from __future__ import annotations


def split_tokens(usage: dict | None) -> tuple[int, int, int]:
    """Return (input_tokens, output_tokens, total_tokens) from a usage dict.

    Prefers the explicit input/output split. When only a total is reported,
    it is returned as total_tokens (third element) instead of being booked as
    output — total_tokens is non-zero only when the split is unknown.
    """
    if not isinstance(usage, dict):
        return 0, 0, 0
    inp = usage.get("input_tokens") or usage.get("inputTokens") or 0
    out = usage.get("output_tokens") or usage.get("outputTokens") or 0
    if inp or out:
        return inp, out, 0
    total = usage.get("total_tokens") or usage.get("totalTokens") or 0
    return 0, 0, total


def apply_usage(cost_entry: dict, input_tokens: int | None, output_tokens: int | None,
                total_tokens: int | None = None) -> dict:
    """Attach non-zero token counts to a cost entry in-place and return it."""
    if input_tokens:
        cost_entry["input_tokens"] = input_tokens
    if output_tokens:
        cost_entry["output_tokens"] = output_tokens
    if total_tokens:
        cost_entry["total_tokens"] = total_tokens
    return cost_entry
