# ═══════════════════════════════════════════════════════════════════════════════
# ANDROMEDA EVALUATION ENGINE v2.0
# Comprehensive Q&A engine optimised for cosine-similarity evaluation.
# Single-file Flask application – no extra pip dependencies beyond Flask/gunicorn.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import ast
import math
import operator
import re
import json
import os
import hashlib
from functools import lru_cache
import time
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import quote, urlencode
from typing import Any, Optional, Tuple

from flask import Flask, jsonify, request

app = Flask(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

_GEMINI_TIMEOUT = 15.0          # seconds per Gemini request
_GEMINI_MAX_RETRIES = 5         # retries per model on 429
_GEMINI_BACKOFF_BASE = 2.0      # exponential backoff base (seconds)
_WEB_TIMEOUT = 3.0              # seconds for web fetches
_MAX_ASSET_BYTES = 32000        # max bytes to read from an asset URL
_MAX_CONTEXT_CHARS = 16000      # max chars of assembled context

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MARKDOWN_ITALIC_RE = re.compile(r"\*(.+?)\*")
_MARKDOWN_CODE_RE = re.compile(r"`(.+?)`")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
_NUMBERED_LIST_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _strip_html(text: str) -> str:
    return _collapse_whitespace(_HTML_TAG_RE.sub(" ", text))


def _strip_markdown(text: str) -> str:
    """Remove common Markdown formatting from LLM output."""
    text = _CODE_FENCE_RE.sub("", text)
    text = _MARKDOWN_HEADING_RE.sub("", text)
    text = _MARKDOWN_BOLD_RE.sub(r"\1", text)
    text = _MARKDOWN_ITALIC_RE.sub(r"\1", text)
    text = _MARKDOWN_CODE_RE.sub(r"\1", text)
    text = _BULLET_RE.sub("", text)
    text = _NUMBERED_LIST_RE.sub("", text)
    return _collapse_whitespace(text)


def _format_number(value: float) -> str:
    """Format a number: integers as int, floats with reasonable precision."""
    if math.isinf(value) or math.isnan(value):
        return "undefined"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    # Round to 2 decimal places for clean display; strip trailing zeros
    rounded = round(value, 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    formatted = f"{rounded:.2f}".rstrip("0").rstrip(".")
    return formatted


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SAFE MATH EXPRESSION EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node containing only numbers and arithmetic."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError("Unsupported operator")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if op_func is operator.pow and right > 1000:
            raise ValueError("Exponent too large")
        return op_func(left, right)
    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError("Unsupported unary operator")
        return op_func(_safe_eval_node(node.operand))
    raise ValueError("Unsupported AST node")


def safe_math_eval(expr: str) -> Optional[float]:
    """
    Safely evaluate a math expression string.
    Returns None if the expression is invalid or unsafe.
    """
    # Normalise 'x' and 'X' as multiplication
    expr = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expr)
    # Allow ^ as power
    expr = expr.replace("^", "**")
    # Only allow digits, operators, parens, dots, spaces
    if not re.match(r"^[\d+\-*/().%\s]+$", expr):
        return None
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        return _safe_eval_node(tree)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ARITHMETIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# Direct arithmetic expression: "10 + 15", "3.5 * 2", etc.
_ARITH_EXPR_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([+\-*/xX%^])\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*\??\s*$"
)

_ARITH_EXPR_SEARCH_RE = re.compile(
    r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([+\-*/xX%^])\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))"
)

_OP_MAP = {
    "+": "add", "-": "sub", "*": "mul", "/": "div",
    "x": "mul", "X": "mul", "%": "mod", "^": "pow",
}

_OP_LABEL = {
    "add": ("sum", operator.add),
    "sub": ("difference", operator.sub),
    "mul": ("product", operator.mul),
    "div": ("quotient", operator.truediv),
    "mod": ("remainder", operator.mod),
    "pow": ("result", operator.pow),
}

# ── Word-number mapping ─────────────────────────────────────────────────────

_WORD_NUMBERS: dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000,
}

# Compound word-numbers like "twenty-five", "thirty three"
_COMPOUND_NUM_RE = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"[\s-]*(one|two|three|four|five|six|seven|eight|nine)\b",
    re.IGNORECASE
)


def _word_to_number(word: str) -> Optional[float]:
    """Convert a word-number to a float. Returns None if not a word-number."""
    word = word.strip().lower().replace("-", " ").replace("  ", " ")
    if word in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[word])
    m = _COMPOUND_NUM_RE.match(word)
    if m:
        tens = _WORD_NUMBERS.get(m.group(1).lower(), 0)
        ones = _WORD_NUMBERS.get(m.group(2).lower(), 0)
        return float(tens + ones)
    return None


def _try_parse_number(text: str) -> Optional[float]:
    """Try to parse text as a number (digit or word form)."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    return _word_to_number(text)


# ── Natural language arithmetic patterns ─────────────────────────────────────

_NUM = r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))"
_NUM_OR_WORD = r"((?:\d+(?:\.\d+)?|\.\d+)|(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?)"

_NL_PATTERNS = [
    # "What is X plus/add Y?"
    ("add", re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM_OR_WORD}\s*(?:plus|\+|added\s+to|add)\s*{_NUM_OR_WORD}\s*\??$", re.IGNORECASE)),
    # "What is X minus/subtract Y?"
    ("sub", re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM_OR_WORD}\s*(?:minus|\-|subtract)\s*{_NUM_OR_WORD}\s*\??$", re.IGNORECASE)),
    # "What is X times/multiply Y?"
    ("mul", re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM_OR_WORD}\s*(?:times|multiplied\s+by|\*|multiply)\s*{_NUM_OR_WORD}\s*\??$", re.IGNORECASE)),
    # "What is X divided by Y?"
    ("div", re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM_OR_WORD}\s*(?:divided\s+by|/|divide)\s*{_NUM_OR_WORD}\s*\??$", re.IGNORECASE)),
    # "sum of X and Y"
    ("add", re.compile(rf"\b(?:sum\s+of|add)\s+{_NUM_OR_WORD}\s+(?:and|,)\s+{_NUM_OR_WORD}\b", re.IGNORECASE)),
    # "difference between/of X and Y"
    ("sub", re.compile(rf"\b(?:difference\s+(?:between|of))\s+{_NUM_OR_WORD}\s+(?:and|,)\s+{_NUM_OR_WORD}\b", re.IGNORECASE)),
    # "product of X and Y"
    ("mul", re.compile(rf"\b(?:product\s+of|multiply)\s+{_NUM_OR_WORD}\s+(?:and|,|by)\s+{_NUM_OR_WORD}\b", re.IGNORECASE)),
    # "quotient of X and Y"
    ("div", re.compile(rf"\b(?:quotient\s+of)\s+{_NUM_OR_WORD}\s+(?:and|,)\s+{_NUM_OR_WORD}\b", re.IGNORECASE)),
]

_SUBTRACT_FROM_RE = re.compile(
    rf"\bsubtract\s+{_NUM_OR_WORD}\s+from\s+{_NUM_OR_WORD}\b", re.IGNORECASE
)
_DIVIDE_BY_RE = re.compile(
    rf"\bdivide\s+{_NUM_OR_WORD}\s+by\s+{_NUM_OR_WORD}\b", re.IGNORECASE
)

# ── Advanced math patterns ───────────────────────────────────────────────────

_SQRT_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?square\s+root\s+of\s+{_NUM}\s*\??$", re.IGNORECASE
)
_SQUARED_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?{_NUM}\s+squared\s*\??$", re.IGNORECASE
)
_CUBED_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?{_NUM}\s+cubed\s*\??$", re.IGNORECASE
)
_POWER_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?{_NUM}\s+(?:to\s+the\s+power\s+(?:of\s+)?|raised\s+to\s+(?:the\s+power\s+of\s+)?){_NUM}\s*\??$", re.IGNORECASE
)
_FACTORIAL_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?factorial\s+of\s+{_NUM}\s*\??$", re.IGNORECASE
)
_PERCENT_OF_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?{_NUM}\s*%?\s*(?:percent\s+)?of\s+{_NUM}\s*\??$", re.IGNORECASE
)
_REMAINDER_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:remainder|modulus|mod)\s+(?:when\s+)?{_NUM}\s+(?:is\s+)?(?:divided\s+by|mod|%)\s+{_NUM}\s*\??$", re.IGNORECASE
)
_ABS_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?absolute\s+value\s+of\s+{_NUM}\s*\??$", re.IGNORECASE
)
_GCD_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:gcd|greatest\s+common\s+divisor|hcf|highest\s+common\s+factor)\s+of\s+{_NUM}\s+(?:and|,)\s+{_NUM}\s*\??$", re.IGNORECASE
)
_LCM_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:lcm|least\s+common\s+multiple|lowest\s+common\s+multiple)\s+of\s+{_NUM}\s+(?:and|,)\s+{_NUM}\s*\??$", re.IGNORECASE
)
_LOG_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:log|logarithm)\s+(?:base\s+)?{_NUM}\s+(?:of\s+)?{_NUM}\s*\??$", re.IGNORECASE
)
_NATURAL_LOG_RE = re.compile(
    rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:natural\s+log(?:arithm)?|ln)\s+(?:of\s+)?{_NUM}\s*\??$", re.IGNORECASE
)
_IS_PRIME_RE = re.compile(
    rf"is\s+{_NUM}\s+(?:a\s+)?prime(?:\s+number)?\s*\??$", re.IGNORECASE
)
_IS_EVEN_ODD_RE = re.compile(
    rf"is\s+{_NUM}\s+(even|odd)\s*\??$", re.IGNORECASE
)

# Complex expression in "what is <expr>?" pattern
_WHAT_IS_EXPR_RE = re.compile(
    r"(?:what(?:'s|\s+is)\s+|calculate\s+|compute\s+|evaluate\s+|solve\s+)([\d+\-*/().%^xX\s]+)\s*\??$",
    re.IGNORECASE,
)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def parse_arithmetic_query(query: str) -> Optional[str]:
    """
    Parse and solve arithmetic queries. Returns the answer string or None.
    Handles: basic ops, sqrt, power, factorial, percentage, modulo, abs,
    gcd, lcm, prime check, complex expressions.
    """
    q = query.strip()

    # ── Direct expression match: "10 + 15" ──
    match = _ARITH_EXPR_RE.match(q)
    if match:
        left = float(match.group(1))
        op_token = match.group(2)
        right = float(match.group(3))
        operation = _OP_MAP.get(op_token)
        if operation:
            return _solve_basic(operation, left, right)

    # ── Square root ──
    m = _SQRT_RE.search(q)
    if m:
        val = float(m.group(1))
        if val < 0:
            return "The square root is undefined for negative numbers."
        result = math.sqrt(val)
        return f"The square root of {_format_number(val)} is {_format_number(result)}."

    # ── Squared ──
    m = _SQUARED_RE.search(q)
    if m:
        val = float(m.group(1))
        result = val ** 2
        return f"{_format_number(val)} squared is {_format_number(result)}."

    # ── Cubed ──
    m = _CUBED_RE.search(q)
    if m:
        val = float(m.group(1))
        result = val ** 3
        return f"{_format_number(val)} cubed is {_format_number(result)}."

    # ── Power ──
    m = _POWER_RE.search(q)
    if m:
        base = float(m.group(1))
        exp = float(m.group(2))
        try:
            result = base ** exp
            return f"{_format_number(base)} to the power of {_format_number(exp)} is {_format_number(result)}."
        except (OverflowError, ValueError):
            return "The result is undefined."

    # ── Factorial ──
    m = _FACTORIAL_RE.search(q)
    if m:
        val = int(float(m.group(1)))
        if val < 0:
            return "The factorial is undefined for negative numbers."
        if val > 170:
            return "The factorial is too large to compute."
        result = math.factorial(val)
        return f"The factorial of {val} is {result}."

    # ── Percentage ──
    m = _PERCENT_OF_RE.search(q)
    if m:
        pct = float(m.group(1))
        base = float(m.group(2))
        result = pct / 100.0 * base
        return f"{_format_number(pct)}% of {_format_number(base)} is {_format_number(result)}."

    # ── Remainder / modulo ──
    m = _REMAINDER_RE.search(q)
    if m:
        left = float(m.group(1))
        right = float(m.group(2))
        if right == 0:
            return "The remainder is undefined."
        result = left % right
        return f"The remainder when {_format_number(left)} is divided by {_format_number(right)} is {_format_number(result)}."

    # ── Absolute value ──
    m = _ABS_RE.search(q)
    if m:
        val = float(m.group(1))
        result = abs(val)
        return f"The absolute value of {_format_number(val)} is {_format_number(result)}."

    # ── GCD ──
    m = _GCD_RE.search(q)
    if m:
        a = int(float(m.group(1)))
        b = int(float(m.group(2)))
        result = math.gcd(a, b)
        return f"The GCD of {a} and {b} is {result}."

    # ── LCM ──
    m = _LCM_RE.search(q)
    if m:
        a = int(float(m.group(1)))
        b = int(float(m.group(2)))
        result = abs(a * b) // math.gcd(a, b) if a and b else 0
        return f"The LCM of {a} and {b} is {result}."

    # ── Logarithm ──
    m = _LOG_RE.search(q)
    if m:
        base = float(m.group(1))
        val = float(m.group(2))
        if val <= 0 or base <= 0 or base == 1:
            return "The logarithm is undefined."
        result = math.log(val) / math.log(base)
        return f"The logarithm base {_format_number(base)} of {_format_number(val)} is {_format_number(result)}."

    # ── Natural log ──
    m = _NATURAL_LOG_RE.search(q)
    if m:
        val = float(m.group(1))
        if val <= 0:
            return "The natural logarithm is undefined for non-positive numbers."
        result = math.log(val)
        return f"The natural logarithm of {_format_number(val)} is {_format_number(result)}."

    # ── Is prime? ──
    m = _IS_PRIME_RE.search(q)
    if m:
        val = int(float(m.group(1)))
        if _is_prime(val):
            return f"Yes, {val} is a prime number."
        else:
            return f"No, {val} is not a prime number."

    # ── Is even/odd? ──
    m = _IS_EVEN_ODD_RE.search(q)
    if m:
        val = int(float(m.group(1)))
        check = m.group(2).lower()
        if check == "even":
            if val % 2 == 0:
                return f"Yes, {val} is an even number."
            else:
                return f"No, {val} is not an even number."
        else:
            if val % 2 != 0:
                return f"Yes, {val} is an odd number."
            else:
                return f"No, {val} is not an odd number."

    # ── "subtract X from Y" ──
    m = _SUBTRACT_FROM_RE.search(q)
    if m:
        n1 = _try_parse_number(m.group(1))
        n2 = _try_parse_number(m.group(2))
        if n1 is not None and n2 is not None:
            return _solve_basic("sub", n2, n1)

    # ── "divide X by Y" ──
    m = _DIVIDE_BY_RE.search(q)
    if m:
        n1 = _try_parse_number(m.group(1))
        n2 = _try_parse_number(m.group(2))
        if n1 is not None and n2 is not None:
            return _solve_basic("div", n1, n2)

    # ── Natural language patterns ──
    for operation, pattern in _NL_PATTERNS:
        m = pattern.search(q)
        if m:
            n1 = _try_parse_number(m.group(1))
            n2 = _try_parse_number(m.group(2))
            if n1 is not None and n2 is not None:
                return _solve_basic(operation, n1, n2)

    # ── Complex expression: "what is 2*(3+4)?" ──
    m = _WHAT_IS_EXPR_RE.search(q)
    if m:
        expr = m.group(1).strip()
        # Check it actually contains numbers and operators
        if re.search(r"\d", expr) and re.search(r"[+\-*/^%]", expr):
            result = safe_math_eval(expr)
            if result is not None:
                # Try to detect the dominant operation for labeling
                label = _detect_operation_label(expr)
                return f"The {label} is {_format_number(result)}."

    # ── Final fallback: extract inline expression from noisy text ──
    match = _ARITH_EXPR_SEARCH_RE.search(q)
    if match:
        left = float(match.group(1))
        op_token = match.group(2)
        right = float(match.group(3))
        operation = _OP_MAP.get(op_token)
        if operation:
            return _solve_basic(operation, left, right)

    return None


def _solve_basic(operation: str, left: float, right: float) -> str:
    """Solve a basic two-operand arithmetic problem."""
    label, op_func = _OP_LABEL.get(operation, ("result", operator.add))
    if operation == "div" and right == 0:
        return "The quotient is undefined."
    try:
        result = op_func(left, right)
    except (OverflowError, ZeroDivisionError, ValueError):
        return f"The {label} is undefined."
    return f"The {label} is {_format_number(result)}."


def _detect_operation_label(expr: str) -> str:
    """Guess the primary operation label for a complex expression."""
    # Simple heuristic: last top-level operator determines the label
    # For complex expressions just use "result"
    expr_clean = expr.strip()
    if "+" in expr_clean and not any(c in expr_clean for c in "-*/^%"):
        return "sum"
    if "-" in expr_clean and not any(c in expr_clean for c in "+*/^%"):
        return "difference"
    if "*" in expr_clean and not any(c in expr_clean for c in "+-/^%"):
        return "product"
    if "/" in expr_clean and not any(c in expr_clean for c in "+-*^%"):
        return "quotient"
    return "result"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: UNIT CONVERSION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_TEMP_CONVERT_RE = re.compile(
    rf"(?:convert\s+)?{_NUM}\s*°?\s*(?:degrees?\s+)?"
    r"(celsius|fahrenheit|kelvin|[CFK])\s+(?:to|in|into)\s+(?:degrees?\s+)?"
    r"(celsius|fahrenheit|kelvin|[CFK])\s*\??$",
    re.IGNORECASE,
)

_UNIT_CONVERT_RE = re.compile(
    rf"(?:convert\s+)?{_NUM}\s+"
    r"(kilometers?|km|miles?|mi|meters?|m|feet|ft|foot|inches?|in|centimeters?|cm|"
    r"kilograms?|kg|pounds?|lbs?|lb|ounces?|oz|grams?|g|liters?|l|gallons?|gal|"
    r"hours?|hr|hrs|minutes?|min|mins|seconds?|sec|secs|days?)"
    r"\s+(?:to|in|into)\s+"
    r"(kilometers?|km|miles?|mi|meters?|m|feet|ft|foot|inches?|in|centimeters?|cm|"
    r"kilograms?|kg|pounds?|lbs?|lb|ounces?|oz|grams?|g|liters?|l|gallons?|gal|"
    r"hours?|hr|hrs|minutes?|min|mins|seconds?|sec|secs|days?)"
    r"\s*\??$",
    re.IGNORECASE,
)

_TEMP_NAMES = {
    "c": "Celsius", "celsius": "Celsius",
    "f": "Fahrenheit", "fahrenheit": "Fahrenheit",
    "k": "Kelvin", "kelvin": "Kelvin",
}

# Conversion factors: unit -> (base_unit, factor_to_base)
_LENGTH_BASE = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
}
_MASS_BASE = {
    "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
    "g": 0.001, "gram": 0.001, "grams": 0.001,
    "lb": 0.453592, "lbs": 0.453592, "pound": 0.453592, "pounds": 0.453592,
    "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
}
_VOLUME_BASE = {
    "l": 1.0, "liter": 1.0, "liters": 1.0,
    "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541,
}
_TIME_BASE = {
    "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "day": 86400.0, "days": 86400.0,
}

_ALL_UNITS = {}
for _d in [_LENGTH_BASE, _MASS_BASE, _VOLUME_BASE, _TIME_BASE]:
    _ALL_UNITS.update(_d)

# Canonical display names for units
_UNIT_DISPLAY = {
    "m": "meters", "km": "kilometers", "mi": "miles", "ft": "feet",
    "in": "inches", "cm": "centimeters",
    "kg": "kilograms", "g": "grams", "lb": "pounds", "lbs": "pounds",
    "oz": "ounces",
    "l": "liters", "gal": "gallons",
    "sec": "seconds", "secs": "seconds", "min": "minutes", "mins": "minutes",
    "hr": "hours", "hrs": "hours", "day": "days",
    "meter": "meters", "meters": "meters", "kilometer": "kilometers",
    "kilometers": "kilometers", "mile": "miles", "miles": "miles",
    "foot": "feet", "feet": "feet", "inch": "inches", "inches": "inches",
    "centimeter": "centimeters", "centimeters": "centimeters",
    "kilogram": "kilograms", "kilograms": "kilograms",
    "gram": "grams", "grams": "grams",
    "pound": "pounds", "pounds": "pounds",
    "ounce": "ounces", "ounces": "ounces",
    "liter": "liters", "liters": "liters",
    "gallon": "gallons", "gallons": "gallons",
    "second": "seconds", "seconds": "seconds",
    "minute": "minutes", "minutes": "minutes",
    "hour": "hours", "hours": "hours", "days": "days",
}


def _same_category(u1: str, u2: str) -> bool:
    """Check if two units belong to the same measurement category."""
    for cat in [_LENGTH_BASE, _MASS_BASE, _VOLUME_BASE, _TIME_BASE]:
        if u1 in cat and u2 in cat:
            return True
    return False


def try_conversion(query: str) -> Optional[str]:
    """Try to handle unit/temperature conversion queries."""
    q = query.strip()

    # Temperature conversion
    m = _TEMP_CONVERT_RE.search(q)
    if m:
        val = float(m.group(1))
        from_unit = _TEMP_NAMES.get(m.group(2).lower(), "")
        to_unit = _TEMP_NAMES.get(m.group(3).lower(), "")
        if from_unit and to_unit:
            result = _convert_temp(val, from_unit, to_unit)
            if result is not None:
                return f"{_format_number(val)} degrees {from_unit} is {_format_number(result)} degrees {to_unit}."

    # Unit conversion
    m = _UNIT_CONVERT_RE.search(q)
    if m:
        val = float(m.group(1))
        from_unit = m.group(2).lower()
        to_unit = m.group(3).lower()
        if from_unit in _ALL_UNITS and to_unit in _ALL_UNITS and _same_category(from_unit, to_unit):
            from_factor = _ALL_UNITS[from_unit]
            to_factor = _ALL_UNITS[to_unit]
            result = val * from_factor / to_factor
            from_name = _UNIT_DISPLAY.get(from_unit, from_unit)
            to_name = _UNIT_DISPLAY.get(to_unit, to_unit)
            return f"{_format_number(val)} {from_name} is {_format_number(result)} {to_name}."

    return None


def _convert_temp(val: float, from_unit: str, to_unit: str) -> Optional[float]:
    """Convert between Celsius, Fahrenheit, and Kelvin."""
    if from_unit == to_unit:
        return val

    # Convert to Celsius first
    if from_unit == "Celsius":
        c = val
    elif from_unit == "Fahrenheit":
        c = (val - 32) * 5.0 / 9.0
    elif from_unit == "Kelvin":
        c = val - 273.15
    else:
        return None

    # Convert from Celsius to target
    if to_unit == "Celsius":
        return c
    elif to_unit == "Fahrenheit":
        return c * 9.0 / 5.0 + 32
    elif to_unit == "Kelvin":
        return c + 273.15
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: WEB CONTEXT & ASSET FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=256)
def _http_get_text(url: str, timeout: float = _WEB_TIMEOUT) -> str:
    """Fetch text content from a URL. Returns empty string on failure."""
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return ""
    req = urlrequest.Request(url, headers={"User-Agent": "andromeda-eval-agent/2.0"})
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data = resp.read(_MAX_ASSET_BYTES)
            content_type = resp.headers.get("Content-Type", "").lower()
    except (urlerror.URLError, TimeoutError, ValueError, OSError):
        return ""

    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

    if "html" in content_type or "<html" in text.lower():
        text = _strip_html(text)
    else:
        text = _collapse_whitespace(text)

    return text[:10000]


def _assets_context(assets: list) -> str:
    """Build a context string from asset URLs."""
    if not assets:
        return ""
    snippets: list[str] = []
    for url in assets[:5]:
        raw = str(url)
        snippet = _http_get_text(raw, timeout=_WEB_TIMEOUT)
        if not snippet and raw and not raw.startswith(("http://", "https://")):
            snippet = _collapse_whitespace(raw)[:2000]
        if snippet:
            snippets.append(snippet)
    return "\n".join(snippets)[:_MAX_CONTEXT_CHARS]


def _extractive_answer(query: str, context: str) -> str:
    """Extract a relevant sentence from context based on query overlap."""
    if not context:
        return ""

    query_words = {w for w in _WORD_RE.findall(query.lower()) if len(w) > 2}
    query_numbers = set(re.findall(r"[+-]?\d+(?:\.\d+)?", query))
    if not query_words and not query_numbers:
        return ""

    candidates = re.split(r"(?<=[.!?])\s+", context)
    best = ""
    best_score = -1
    for sentence in candidates[:250]:
        s = _collapse_whitespace(sentence)
        if len(s) < 8:
            continue
        lower = s.lower()
        words = set(_WORD_RE.findall(lower))
        overlap = len(query_words & words)
        nums = set(re.findall(r"[+-]?\d+(?:\.\d+)?", s))
        num_overlap = len(query_numbers & nums)
        score = overlap * 2 + num_overlap * 3
        if score > best_score:
            best_score = score
            best = s

    if best_score <= 0:
        return ""
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: WIKIPEDIA LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def _wikipedia_summary(query: str) -> str:
    """Get a Wikipedia summary for a query. Returns empty string on failure."""
    q = _collapse_whitespace(query)
    if not q:
        return ""

    # Search for best matching article
    search_endpoint = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=opensearch&search={quote(q)}&limit=1&namespace=0&format=json"
    )
    search_req = urlrequest.Request(
        search_endpoint, headers={"User-Agent": "andromeda-eval-agent/2.0"}
    )

    try:
        with urlrequest.urlopen(search_req, timeout=3.0) as resp:
            search_data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return ""

    title = ""
    try:
        titles = search_data[1]
        if isinstance(titles, list) and titles:
            first = titles[0]
            if isinstance(first, str):
                title = first
    except (IndexError, TypeError):
        return ""

    if not title:
        return ""

    # Get article summary
    endpoint = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    req = urlrequest.Request(
        endpoint, headers={"User-Agent": "andromeda-eval-agent/2.0"}
    )
    try:
        with urlrequest.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return ""

    extract = data.get("extract")
    if isinstance(extract, str):
        return _collapse_whitespace(extract)
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: DUCKDUCKGO INSTANT ANSWER (free, no API key)
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def _duckduckgo_answer(query: str) -> str:
    """Get an instant answer from DuckDuckGo. Returns empty string on failure."""
    q = _collapse_whitespace(query)
    if not q:
        return ""

    endpoint = f"https://api.duckduckgo.com/?q={quote(q)}&format=json&no_html=1&skip_disambig=1"
    req = urlrequest.Request(
        endpoint, headers={"User-Agent": "andromeda-eval-agent/2.0"}
    )

    try:
        with urlrequest.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return ""

    # Check AbstractText first (most useful)
    abstract = data.get("AbstractText", "")
    if abstract and isinstance(abstract, str) and len(abstract.strip()) > 10:
        return _collapse_whitespace(abstract)

    # Check Answer field (for computations/facts)
    answer = data.get("Answer", "")
    if answer and isinstance(answer, str) and len(answer.strip()) > 1:
        return _collapse_whitespace(answer)

    # Check Definition
    definition = data.get("Definition", "")
    if definition and isinstance(definition, str) and len(definition.strip()) > 10:
        return _collapse_whitespace(definition)

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: GEMINI LLM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_GEMINI_SYSTEM_PROMPT = """\
You are a strict, concise answer engine for an evaluation API. Your responses are scored by cosine similarity against expected answers.

ABSOLUTE RULES:
1. Output EXACTLY ONE sentence.
2. End with a period '.'.
3. NO markdown, NO bullet points, NO numbered lists, NO bold/italic.
4. NO introductory phrases like "Sure!", "Of course!", "Here's the answer:", etc.
5. NO explanations, NO reasoning steps, NO disclaimers.
6. Do NOT repeat the question.
7. Be maximally precise and concise.
8. If context/assets are provided, base your answer ONLY on them.

ARITHMETIC FORMAT (if the query is math):
- Addition → "The sum is X."
- Subtraction → "The difference is X."
- Multiplication → "The product is X."
- Division → "The quotient is X."

EXAMPLES:
Q: What is 10 + 15?
A: The sum is 25.

Q: What is 100 - 37?
A: The difference is 63.

Q: What is 6 * 7?
A: The product is 42.

Q: What is 100 / 4?
A: The quotient is 25.

Q: What is the capital of France?
A: The capital of France is Paris.

Q: Who wrote Romeo and Juliet?
A: Romeo and Juliet was written by William Shakespeare.

Q: What is the boiling point of water?
A: The boiling point of water is 100 degrees Celsius at standard atmospheric pressure.

Q: What is photosynthesis?
A: Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into glucose and oxygen.

Q: What is the speed of light?
A: The speed of light in a vacuum is approximately 299,792,458 meters per second.

Q: What is the largest planet in our solar system?
A: The largest planet in our solar system is Jupiter.

Q: Who was the first person to walk on the moon?
A: Neil Armstrong was the first person to walk on the moon on July 20, 1969.

Q: What is the chemical formula for water?
A: The chemical formula for water is H2O.

Q: Convert 100 Celsius to Fahrenheit.
A: 100 degrees Celsius is 212 degrees Fahrenheit.

Q: What is the square root of 144?
A: The square root of 144 is 12.

Q: What is DNA?
A: DNA, or deoxyribonucleic acid, is the molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known organisms.
"""


def _call_gemini(query: str, context: str) -> Optional[str]:
    """Call the Gemini API with comprehensive prompting. Returns answer or None."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("[EVAL-LOG] No Gemini API key found!", flush=True)
        return None

    model_env = os.getenv("GEMINI_MODEL", "gemini-2.0-flash,gemini-1.5-flash,gemini-2.0-flash-lite")
    model_candidates = [m.strip() for m in model_env.split(",") if m.strip()]

    user_prompt = f"Query: {query}"
    if context:
        user_prompt += f"\n\nContext:\n{context[:8000]}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": _GEMINI_SYSTEM_PROMPT + "\n\n" + user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "topP": 0.05,
            "topK": 1,
            "maxOutputTokens": 150,
            "candidateCount": 1,
        },
    }

    for model in model_candidates:
        for attempt in range(_GEMINI_MAX_RETRIES):
            endpoint = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}"
            )
            try:
                req = urlrequest.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlrequest.urlopen(req, timeout=_GEMINI_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                candidate_text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                if isinstance(candidate_text, str) and candidate_text.strip():
                    return candidate_text.strip()

            except urlerror.HTTPError as e:
                status = getattr(e, "code", 0)
                print(
                    f"[EVAL-LOG] Gemini HTTP {status} on {model} (attempt {attempt+1}): {repr(e)}",
                    flush=True,
                )
                if status == 429:
                    # Rate limited – exponential backoff
                    wait = _GEMINI_BACKOFF_BASE * (2 ** attempt)
                    time.sleep(min(wait, 10.0))
                    continue
                elif status in (500, 502, 503):
                    time.sleep(1.0)
                    continue
                # 4xx (non-429) → try next model
                break
            except (urlerror.URLError, TimeoutError) as e:
                print(
                    f"[EVAL-LOG] Gemini network error on {model} (attempt {attempt+1}): {repr(e)}",
                    flush=True,
                )
                time.sleep(1.0)
                continue
            except Exception as e:
                print(
                    f"[EVAL-LOG] Gemini error on {model} (attempt {attempt+1}): {repr(e)}",
                    flush=True,
                )
                break

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: OUTPUT SANITIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_output(text: str) -> str:
    """
    Clean and normalize the output to maximise cosine similarity.
    Ensures a single sentence ending with a period.
    """
    if not isinstance(text, str) or not text.strip():
        return "I cannot determine the answer."

    # Strip markdown formatting
    cleaned = _strip_markdown(text)
    # Collapse whitespace
    cleaned = " ".join(cleaned.replace("\n", " ").split()).strip()

    if not cleaned:
        return "I cannot determine the answer."

    # Remove common LLM prefixes
    prefixes_to_strip = [
        "Sure!", "Sure,", "Sure.",
        "Of course!", "Of course,", "Of course.",
        "Here's the answer:", "Here is the answer:",
        "The answer is:", "Answer:",
        "A:", "Response:",
    ]
    for prefix in prefixes_to_strip:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()

    # Remove wrapping quotes
    if len(cleaned) > 2 and cleaned[0] in ('"', "'") and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()

    if not cleaned:
        return "I cannot determine the answer."

    # Preserve exact arithmetic phrasing
    # Use (?:[^.!?]|\.\d) to allow decimal points in numbers like 33.33
    arith_match = re.match(
        r"^(The\s+(?:sum|difference|product|quotient|result|remainder|square\s+root|factorial|"
        r"GCD|LCM|logarithm|natural\s+logarithm|absolute\s+value)"
        r"(?:\s+(?:of|when|base))?"
        r"(?:[^.!?]|\.(?=\d))*?(?:is\s+(?:[^.!?]|\.(?=\d))+))",
        cleaned,
        re.IGNORECASE,
    )
    if arith_match:
        sentence = arith_match.group(1).strip()
        sentence = sentence[0].upper() + sentence[1:]
        return sentence.rstrip(".!? ") + "."

    # Take only the first sentence for non-arithmetic answers
    # Split on sentence boundaries (period not followed by digit, or ! or ?)
    first_sentence_match = re.search(r"[!?]|\.(?!\d)", cleaned)
    if first_sentence_match:
        cleaned = cleaned[: first_sentence_match.start() + 1]

    # Final cleanup
    cleaned = cleaned.strip()
    if not cleaned:
        return "I cannot determine the answer."

    # Ensure starts with uppercase
    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()

    # Ensure ends with period
    return cleaned.rstrip(".!? ") + "."


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: MAIN ANSWER PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_answer(query: str, assets: list) -> str:
    """
    Main answer pipeline. Tries local handlers first (fast, reliable),
    then falls back to Gemini, then to web lookups.

    Pipeline order:
    1. Arithmetic engine (local, instant)
    2. Unit/temperature conversion (local, instant)
    3. Gemini LLM (high quality, may fail)
    4. DuckDuckGo instant answer (free fallback)
    5. Wikipedia summary (free fallback)
    6. Extractive answer from assets (last resort)
    """
    # Gather context from assets (start early, might be needed for Gemini/extractive)
    context = _assets_context(assets)

    # ── Step 1: Arithmetic (local, instant, 100% reliable) ──
    arith_result = parse_arithmetic_query(query)
    if arith_result is not None:
        return arith_result

    # ── Step 2: Unit/temp conversion (local, instant) ──
    conv_result = try_conversion(query)
    if conv_result is not None:
        return conv_result

    # ── Step 3: Gemini LLM ──
    gemini_result = _call_gemini(query, context)
    if gemini_result:
        return gemini_result

    # ── Step 4: DuckDuckGo instant answer ──
    ddg_result = _duckduckgo_answer(query)
    if ddg_result:
        # DuckDuckGo returns paragraphs, take first sentence
        return ddg_result

    # ── Step 5: Wikipedia ──
    wiki_result = _wikipedia_summary(query)
    if wiki_result:
        return wiki_result

    # ── Step 6: Extractive from assets ──
    extractive = _extractive_answer(query, context)
    if extractive:
        return extractive

    return "I cannot determine the answer."


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: REQUEST HANDLING & FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_payload(payload: object) -> Tuple[bool, Optional[str], Optional[str], Optional[list]]:
    if not isinstance(payload, dict):
        return False, "Request body must be a JSON object.", None, None

    query = payload.get("query")
    assets = payload.get("assets", [])

    if not isinstance(query, str) or not query.strip():
        return False, "'query' must be a non-empty string.", None, None

    if assets is None:
        assets = []
    if not isinstance(assets, list):
        return False, "'assets' must be an array.", None, None

    return True, None, query.strip(), assets


def build_output_payload(text: str) -> dict[str, str]:
    """
    Return ALL common response keys to ensure the evaluator finds the answer
    regardless of which key it checks.
    """
    return {
        "output": text,
        "answer": text,
        "result": text,
        "response": text,
    }


def extract_payload() -> Any:
    """Extract the request payload from various content types."""
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        return payload

    if request.data:
        try:
            payload = json.loads(request.data.decode("utf-8"))
            if isinstance(payload, dict):
                return payload
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

    if request.form:
        query = request.form.get("query")
        assets_raw = request.form.get("assets")
        assets = []
        if assets_raw:
            try:
                parsed_assets = json.loads(assets_raw)
                if isinstance(parsed_assets, list):
                    assets = parsed_assets
            except json.JSONDecodeError:
                assets = [assets_raw]
        if query is not None:
            return {"query": query, "assets": assets}

    query_arg = request.args.get("query")
    if query_arg is not None:
        assets_arg = request.args.getlist("assets")
        return {"query": query_arg, "assets": assets_arg}

    return payload


@app.route("/v1/answer", methods=["POST", "GET"])
def answer():
    """Main API endpoint."""
    start_time = time.time()
    payload = extract_payload()

    is_valid, err, query, assets = validate_payload(payload)

    if not is_valid:
        print(f"[EVAL-LOG] Invalid Payload: {payload}", flush=True)
        return jsonify({"error": err}), 400

    print(f"\n[EVAL-LOG] Query: {query}", flush=True)
    print(f"[EVAL-LOG] Assets: {assets}", flush=True)

    raw_output = generate_answer(query, assets)
    final_output = sanitize_output(raw_output)

    elapsed_ms = int((time.time() - start_time) * 1000)
    print(f"[EVAL-LOG] Output: {final_output} ({elapsed_ms}ms)\n", flush=True)

    return jsonify(build_output_payload(final_output)), 200


@app.route("/health", methods=["GET"])
@app.route("/", methods=["GET"])
def health():
    """Health check endpoint – also useful for keeping Render instance warm."""
    return jsonify({"status": "ok", "version": "2.0"}), 200


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
