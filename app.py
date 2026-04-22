from __future__ import annotations

import re
from typing import Optional, Tuple

from flask import Flask, jsonify, request

app = Flask(__name__)

# Precompiled regex for low-latency parsing of arithmetic expressions.
_ARITH_EXPR_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([+\-*/xX])\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*$"
)

# Search-mode regex to extract arithmetic expressions from noisy text.
_ARITH_EXPR_SEARCH_RE = re.compile(
    r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([+\-*/xX])\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))"
)

_OP_MAP = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "x": "mul",
    "X": "mul",
}

# Optional natural-language arithmetic patterns.
_NL_PATTERNS = [
    ("add", re.compile(r"^\s*(?:what is\s+)?([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:plus|add)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*\??\s*$", re.IGNORECASE)),
    ("sub", re.compile(r"^\s*(?:what is\s+)?([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:minus|subtract)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*\??\s*$", re.IGNORECASE)),
    ("mul", re.compile(r"^\s*(?:what is\s+)?([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:times|multiply)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*\??\s*$", re.IGNORECASE)),
    ("div", re.compile(r"^\s*(?:what is\s+)?([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:divided by|divide)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*\??\s*$", re.IGNORECASE)),
    ("add", re.compile(r"\b(?:sum of|add)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:and|,)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\b", re.IGNORECASE)),
    ("sub", re.compile(r"\b(?:difference between)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:and|,)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\b", re.IGNORECASE)),
    ("mul", re.compile(r"\b(?:product of|multiply)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:and|,)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\b", re.IGNORECASE)),
    ("div", re.compile(r"\b(?:quotient of)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:and|,)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\b", re.IGNORECASE)),
]

_SUBTRACT_FROM_RE = re.compile(
    r"\bsubtract\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*from\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\b",
    re.IGNORECASE,
)

_DIVIDE_BY_RE = re.compile(
    r"\bdivide\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*by\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\b",
    re.IGNORECASE,
)


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}"


def parse_arithmetic_query(query: str) -> Optional[Tuple[str, float, float]]:
    match = _ARITH_EXPR_RE.match(query)
    if match:
        left = float(match.group(1))
        op_token = match.group(2)
        right = float(match.group(3))
        operation = _OP_MAP.get(op_token)
        if operation:
            return operation, left, right

    m = _SUBTRACT_FROM_RE.search(query)
    if m:
        # "subtract 3 from 10" -> 10 - 3
        return "sub", float(m.group(2)), float(m.group(1))

    m = _DIVIDE_BY_RE.search(query)
    if m:
        # "divide 10 by 2" -> 10 / 2
        return "div", float(m.group(1)), float(m.group(2))

    for operation, pattern in _NL_PATTERNS:
        m = pattern.search(query)
        if m:
            return operation, float(m.group(1)), float(m.group(2))

    # Final fast fallback: extract inline expression from otherwise noisy text.
    match = _ARITH_EXPR_SEARCH_RE.search(query)
    if match:
        left = float(match.group(1))
        op_token = match.group(2)
        right = float(match.group(3))
        operation = _OP_MAP.get(op_token)
        if operation:
            return operation, left, right

    return None


def solve_arithmetic(operation: str, left: float, right: float) -> str:
    if operation == "add":
        result = left + right
        return f"The sum is {_format_number(result)}."
    if operation == "sub":
        result = left - right
        return f"The difference is {_format_number(result)}."
    if operation == "mul":
        result = left * right
        return f"The product is {_format_number(result)}."
    if operation == "div":
        if right == 0:
            return "The quotient is undefined."
        result = left / right
        return f"The quotient is {_format_number(result)}."
    return "I cannot determine the answer."


def llm_style_fallback(query: str, assets: list) -> str:
    # Minimal, deterministic fallback for non-arithmetic queries.
    _ = query
    _ = assets
    return "I cannot determine the answer."


def sanitize_output(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    cleaned = " ".join(text.replace("\n", " ").split())
    if not cleaned:
        return "I cannot determine the answer."

    # Keep only the first sentence-like segment, but ignore decimal points.
    split_match = re.search(r"[!?]|\.(?!\d)", cleaned)
    if split_match:
        cleaned = cleaned[: split_match.start() + 1]

    cleaned = cleaned.rstrip(".!? ") + "."
    return cleaned


def validate_payload(payload: object) -> Tuple[bool, Optional[str], Optional[str], Optional[list]]:
    if not isinstance(payload, dict):
        return False, "Request body must be a JSON object.", None, None

    query = payload.get("query")
    assets = payload.get("assets")

    if not isinstance(query, str) or not query.strip():
        return False, "'query' must be a non-empty string.", None, None

    if not isinstance(assets, list):
        return False, "'assets' must be an array.", None, None

    return True, None, query.strip(), assets


@app.route("/v1/answer", methods=["POST"])
def answer():
    payload = request.get_json(silent=True)
    is_valid, err, query, assets = validate_payload(payload)

    if not is_valid:
        return jsonify({"error": err}), 400

    parsed = parse_arithmetic_query(query)
    if parsed is not None:
        operation, left, right = parsed
        raw_output = solve_arithmetic(operation, left, right)
    else:
        raw_output = llm_style_fallback(query, assets)

    final_output = sanitize_output(raw_output)
    return jsonify({"output": final_output}), 200


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
