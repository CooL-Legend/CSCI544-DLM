"""Extract and compare explicitly marked final answers without repairing outputs."""

from decimal import Decimal
import re


# Require a marked final answer: never silently score a number from the reasoning.
NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
FINAL_ANSWER = re.compile(r"####\s*\$?\s*(" + NUMBER + r")(.*)")
# Trailing units ("110 miles", "15 pounds", "20%") are allowed, but the tail must not
# continue the number or an expression ("1,2", "1/2", "3 1/2", "5 - 3", "12 x 3").
CONTINUATION = re.compile(r"\s*(?:[\d/*+\-=×÷^]|[.,]\s*\d|[x×]\s*\d)")


def extract_answer(text):
    """Normalize the number after the last #### marker (units allowed) as a decimal string, or None."""
    marked = [line.strip() for line in text.splitlines() if line.lstrip().startswith("####")]
    match = FINAL_ANSWER.fullmatch(marked[-1]) if marked else None
    if match is None or CONTINUATION.match(match.group(2)):
        return None
    value = Decimal(match.group(1).replace(",", ""))
    if value == 0:
        return "0"
    normalized = format(value, "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def score_answer(output, reference_solution):
    prediction = extract_answer(output)
    expected = extract_answer(reference_solution) if reference_solution is not None else None
    if reference_solution is not None and expected is None:
        raise ValueError("The GSM8K reference has no valid #### final answer.")
    return {"predicted_answer": prediction, "reference_answer": expected,
            "parse_success": prediction is not None, "reference_available": expected is not None,
            "exact_match": prediction == expected if expected is not None else None}
