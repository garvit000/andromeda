# ==============================================================================
# ANDROMEDA EVALUATION ENGINE v5.0
# All-levels: arithmetic, extraction, boolean, comparison/reasoning, LLM.
# Single-file Flask application - no extra pip deps beyond Flask/gunicorn.
# ==============================================================================
from __future__ import annotations

import ast
import math
import operator
import re
import json
import os
import hashlib
import datetime
import calendar
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

_GEMINI_TIMEOUT = 8.0
_GEMINI_MAX_RETRIES = 2
_GEMINI_BACKOFF_BASE = 1.0
_WEB_TIMEOUT = 3.0
_MAX_ASSET_BYTES = 32000
_MAX_CONTEXT_CHARS = 16000

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

# Month names for date parsing
_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_ABBR = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]
_MONTH_MAP = {}
for i, m in enumerate(_MONTHS):
    _MONTH_MAP[m] = i + 1
for i, m in enumerate(_MONTH_ABBR):
    _MONTH_MAP[m] = i + 1

_MONTH_NAMES_RE = "|".join(_MONTHS + _MONTH_ABBR)


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
    rounded = round(value, 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    formatted = f"{rounded:.2f}".rstrip("0").rstrip(".")
    return formatted


def _extract_quoted_text(query: str) -> Optional[str]:
    """Extract text inside quotes (single, double, or smart quotes) from a query."""
    # Try double quotes first
    m = re.search(r'["\u201c](.+?)["\u201d]', query)
    if m:
        return m.group(1)
    # Try single quotes
    m = re.search(r"['\u2018](.+?)['\u2019]", query)
    if m:
        return m.group(1)
    return None


def _extract_after_colon(query: str) -> Optional[str]:
    """Extract text after a colon, possibly in quotes."""
    m = re.search(r":\s*(.+)$", query)
    if m:
        text = m.group(1).strip().rstrip(".")
        # Remove surrounding quotes
        if len(text) >= 2 and text[0] in ('"', "'", "\u201c") and text[-1] in ('"', "'", "\u201d"):
            text = text[1:-1]
        return text
    return None


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
    expr = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expr)
    expr = expr.replace("^", "**")
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

_WORD_NUMBERS: dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000,
}
_COMPOUND_NUM_RE = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"[\s-]*(one|two|three|four|five|six|seven|eight|nine)\b",
    re.IGNORECASE
)


def _word_to_number(word: str) -> Optional[float]:
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
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    return _word_to_number(text)


_NUM = r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))"
_NUM_OR_WORD = (
    r"((?:\d+(?:\.\d+)?|\.\d+)|(?:zero|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?)"
)

_NL_PATTERNS = [
    ("add", re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM_OR_WORD}\s*(?:plus|\+|added\s+to|add)\s*{_NUM_OR_WORD}\s*\??$", re.IGNORECASE)),
    ("sub", re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM_OR_WORD}\s*(?:minus|\-|subtract)\s*{_NUM_OR_WORD}\s*\??$", re.IGNORECASE)),
    ("mul", re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM_OR_WORD}\s*(?:times|multiplied\s+by|\*|multiply)\s*{_NUM_OR_WORD}\s*\??$", re.IGNORECASE)),
    ("div", re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM_OR_WORD}\s*(?:divided\s+by|/|divide)\s*{_NUM_OR_WORD}\s*\??$", re.IGNORECASE)),
    ("add", re.compile(rf"\b(?:sum\s+of|add)\s+{_NUM_OR_WORD}\s+(?:and|,)\s+{_NUM_OR_WORD}\b", re.IGNORECASE)),
    ("sub", re.compile(rf"\b(?:difference\s+(?:between|of))\s+{_NUM_OR_WORD}\s+(?:and|,)\s+{_NUM_OR_WORD}\b", re.IGNORECASE)),
    ("mul", re.compile(rf"\b(?:product\s+of|multiply)\s+{_NUM_OR_WORD}\s+(?:and|,|by)\s+{_NUM_OR_WORD}\b", re.IGNORECASE)),
    ("div", re.compile(rf"\b(?:quotient\s+of)\s+{_NUM_OR_WORD}\s+(?:and|,)\s+{_NUM_OR_WORD}\b", re.IGNORECASE)),
]

_SUBTRACT_FROM_RE = re.compile(
    rf"\bsubtract\s+{_NUM_OR_WORD}\s+from\s+{_NUM_OR_WORD}\b", re.IGNORECASE
)
_DIVIDE_BY_RE = re.compile(
    rf"\bdivide\s+{_NUM_OR_WORD}\s+by\s+{_NUM_OR_WORD}\b", re.IGNORECASE
)

_SQRT_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?square\s+root\s+of\s+{_NUM}\s*\??$", re.IGNORECASE)
_SQUARED_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM}\s+squared\s*\??$", re.IGNORECASE)
_CUBED_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM}\s+cubed\s*\??$", re.IGNORECASE)
_POWER_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM}\s+(?:to\s+the\s+power\s+(?:of\s+)?|raised\s+to\s+(?:the\s+power\s+of\s+)?){_NUM}\s*\??$", re.IGNORECASE)
_FACTORIAL_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?factorial\s+of\s+{_NUM}\s*\??$", re.IGNORECASE)
_PERCENT_OF_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?{_NUM}\s*%?\s*(?:percent\s+)?of\s+{_NUM}\s*\??$", re.IGNORECASE)
_REMAINDER_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:remainder|modulus|mod)\s+(?:when\s+)?{_NUM}\s+(?:is\s+)?(?:divided\s+by|mod|%)\s+{_NUM}\s*\??$", re.IGNORECASE)
_ABS_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?absolute\s+value\s+of\s+{_NUM}\s*\??$", re.IGNORECASE)
_GCD_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:gcd|greatest\s+common\s+divisor|hcf|highest\s+common\s+factor)\s+of\s+{_NUM}\s+(?:and|,)\s+{_NUM}\s*\??$", re.IGNORECASE)
_LCM_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:lcm|least\s+common\s+multiple|lowest\s+common\s+multiple)\s+of\s+{_NUM}\s+(?:and|,)\s+{_NUM}\s*\??$", re.IGNORECASE)
_LOG_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:log|logarithm)\s+(?:base\s+)?{_NUM}\s+(?:of\s+)?{_NUM}\s*\??$", re.IGNORECASE)
_NATURAL_LOG_RE = re.compile(rf"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:natural\s+log(?:arithm)?|ln)\s+(?:of\s+)?{_NUM}\s*\??$", re.IGNORECASE)
_IS_PRIME_RE = re.compile(rf"is\s+{_NUM}\s+(?:a\s+)?prime(?:\s+number)?\s*\??$", re.IGNORECASE)
_IS_EVEN_ODD_RE = re.compile(rf"is\s+{_NUM}\s+(even|odd)\s*\??$", re.IGNORECASE)
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
    q = query.strip()

    match = _ARITH_EXPR_RE.match(q)
    if match:
        left = float(match.group(1))
        op_token = match.group(2)
        right = float(match.group(3))
        operation = _OP_MAP.get(op_token)
        if operation:
            return _solve_basic(operation, left, right)

    m = _SQRT_RE.search(q)
    if m:
        val = float(m.group(1))
        if val < 0:
            return "The square root is undefined for negative numbers."
        result = math.sqrt(val)
        return f"The square root of {_format_number(val)} is {_format_number(result)}."

    m = _SQUARED_RE.search(q)
    if m:
        val = float(m.group(1))
        result = val ** 2
        return f"{_format_number(val)} squared is {_format_number(result)}."

    m = _CUBED_RE.search(q)
    if m:
        val = float(m.group(1))
        result = val ** 3
        return f"{_format_number(val)} cubed is {_format_number(result)}."

    m = _POWER_RE.search(q)
    if m:
        base = float(m.group(1))
        exp = float(m.group(2))
        try:
            result = base ** exp
            return f"{_format_number(base)} to the power of {_format_number(exp)} is {_format_number(result)}."
        except (OverflowError, ValueError):
            return "The result is undefined."

    m = _FACTORIAL_RE.search(q)
    if m:
        val = int(float(m.group(1)))
        if val < 0:
            return "The factorial is undefined for negative numbers."
        if val > 170:
            return "The factorial is too large to compute."
        result = math.factorial(val)
        return f"The factorial of {val} is {result}."

    m = _PERCENT_OF_RE.search(q)
    if m:
        pct = float(m.group(1))
        base = float(m.group(2))
        result = pct / 100.0 * base
        return f"{_format_number(pct)}% of {_format_number(base)} is {_format_number(result)}."

    m = _REMAINDER_RE.search(q)
    if m:
        left = float(m.group(1))
        right = float(m.group(2))
        if right == 0:
            return "The remainder is undefined."
        result = left % right
        return f"The remainder when {_format_number(left)} is divided by {_format_number(right)} is {_format_number(result)}."

    m = _ABS_RE.search(q)
    if m:
        val = float(m.group(1))
        result = abs(val)
        return f"The absolute value of {_format_number(val)} is {_format_number(result)}."

    m = _GCD_RE.search(q)
    if m:
        a = int(float(m.group(1)))
        b = int(float(m.group(2)))
        result = math.gcd(a, b)
        return f"The GCD of {a} and {b} is {result}."

    m = _LCM_RE.search(q)
    if m:
        a = int(float(m.group(1)))
        b = int(float(m.group(2)))
        result = abs(a * b) // math.gcd(a, b) if a and b else 0
        return f"The LCM of {a} and {b} is {result}."

    m = _LOG_RE.search(q)
    if m:
        base = float(m.group(1))
        val = float(m.group(2))
        if val <= 0 or base <= 0 or base == 1:
            return "The logarithm is undefined."
        result = math.log(val) / math.log(base)
        return f"The logarithm base {_format_number(base)} of {_format_number(val)} is {_format_number(result)}."

    m = _NATURAL_LOG_RE.search(q)
    if m:
        val = float(m.group(1))
        if val <= 0:
            return "The natural logarithm is undefined for non-positive numbers."
        result = math.log(val)
        return f"The natural logarithm of {_format_number(val)} is {_format_number(result)}."

    # Prime/even/odd now handled by try_boolean_question (returns YES/NO)
    # Keeping them here as fallback with YES/NO format
    m = _IS_PRIME_RE.search(q)
    if m:
        val = int(float(m.group(1)))
        return ("YES" if _is_prime(val) else "NO")

    m = _IS_EVEN_ODD_RE.search(q)
    if m:
        val = int(float(m.group(1)))
        check = m.group(2).lower()
        if check == "even":
            return "YES" if val % 2 == 0 else "NO"
        else:
            return "YES" if val % 2 != 0 else "NO"

    m = _SUBTRACT_FROM_RE.search(q)
    if m:
        n1 = _try_parse_number(m.group(1))
        n2 = _try_parse_number(m.group(2))
        if n1 is not None and n2 is not None:
            return _solve_basic("sub", n2, n1)

    m = _DIVIDE_BY_RE.search(q)
    if m:
        n1 = _try_parse_number(m.group(1))
        n2 = _try_parse_number(m.group(2))
        if n1 is not None and n2 is not None:
            return _solve_basic("div", n1, n2)

    for operation, pattern in _NL_PATTERNS:
        m = pattern.search(q)
        if m:
            n1 = _try_parse_number(m.group(1))
            n2 = _try_parse_number(m.group(2))
            if n1 is not None and n2 is not None:
                return _solve_basic(operation, n1, n2)

    m = _WHAT_IS_EXPR_RE.search(q)
    if m:
        expr = m.group(1).strip()
        if re.search(r"\d", expr) and re.search(r"[+\-*/^%]", expr):
            result = safe_math_eval(expr)
            if result is not None:
                label = _detect_operation_label(expr)
                return f"The {label} is {_format_number(result)}."

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
    label, op_func = _OP_LABEL.get(operation, ("result", operator.add))
    if operation == "div" and right == 0:
        return "The quotient is undefined."
    try:
        result = op_func(left, right)
    except (OverflowError, ZeroDivisionError, ValueError):
        return f"The {label} is undefined."
    return f"The {label} is {_format_number(result)}."


def _detect_operation_label(expr: str) -> str:
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
    rf"{_NUM}\s*°?\s*(?:degrees?\s+)?"
    r"(celsius|fahrenheit|kelvin|[CFK])\s+(?:to|in|into)\s+(?:degrees?\s+)?"
    r"(celsius|fahrenheit|kelvin|[CFK])\s*\??$",
    re.IGNORECASE,
)
_TEMP_NAMES = {
    "c": "Celsius", "celsius": "Celsius",
    "f": "Fahrenheit", "fahrenheit": "Fahrenheit",
    "k": "Kelvin", "kelvin": "Kelvin",
}

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

_UNIT_DISPLAY = {
    "m": "meters", "km": "kilometers", "mi": "miles", "ft": "feet",
    "in": "inches", "cm": "centimeters",
    "kg": "kilograms", "g": "grams", "lb": "pounds", "lbs": "pounds",
    "oz": "ounces", "l": "liters", "gal": "gallons",
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
    for cat in [_LENGTH_BASE, _MASS_BASE, _VOLUME_BASE, _TIME_BASE]:
        if u1 in cat and u2 in cat:
            return True
    return False


def try_conversion(query: str) -> Optional[str]:
    q = query.strip()
    m = _TEMP_CONVERT_RE.search(q)
    if m:
        val = float(m.group(1))
        from_unit = _TEMP_NAMES.get(m.group(2).lower(), "")
        to_unit = _TEMP_NAMES.get(m.group(3).lower(), "")
        if from_unit and to_unit:
            result = _convert_temp(val, from_unit, to_unit)
            if result is not None:
                return f"{_format_number(val)} degrees {from_unit} is {_format_number(result)} degrees {to_unit}."

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
    if from_unit == to_unit:
        return val
    if from_unit == "Celsius":
        c = val
    elif from_unit == "Fahrenheit":
        c = (val - 32) * 5.0 / 9.0
    elif from_unit == "Kelvin":
        c = val - 273.15
    else:
        return None
    if to_unit == "Celsius":
        return c
    elif to_unit == "Fahrenheit":
        return c * 9.0 / 5.0 + 32
    elif to_unit == "Kelvin":
        return c + 273.15
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: TEXT EXTRACTION & PROCESSING ENGINE (Level 2)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Date extraction patterns ──────────────────────────────────────────────────

# "12 March 2024", "12 march 2024", "1 Jan 2025"
_DATE_DMY_LONG = re.compile(
    r"\b(\d{1,2})\s+(" + _MONTH_NAMES_RE + r")[\s,]+(\d{4})\b", re.IGNORECASE
)
# "March 12, 2024", "march 12 2024"
_DATE_MDY_LONG = re.compile(
    r"\b(" + _MONTH_NAMES_RE + r")\s+(\d{1,2})[\s,]+(\d{4})\b", re.IGNORECASE
)
# "12/03/2024", "12-03-2024", "12.03.2024"
_DATE_NUMERIC_DMY = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
# "2024-03-12" (ISO format)
_DATE_ISO = re.compile(r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b")

# ── Email pattern ─────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# ── URL pattern ───────────────────────────────────────────────────────────────
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")

# ── Phone number pattern ──────────────────────────────────────────────────────
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}"
)

# ── Number extraction ─────────────────────────────────────────────────────────
_NUMBER_IN_TEXT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _extract_dates(text: str) -> list[str]:
    """Extract date strings from text, returning them in their original format."""
    dates = []

    for m in _DATE_DMY_LONG.finditer(text):
        dates.append((m.start(), m.group(0)))

    for m in _DATE_MDY_LONG.finditer(text):
        dates.append((m.start(), m.group(0)))

    for m in _DATE_ISO.finditer(text):
        dates.append((m.start(), m.group(0)))

    if not dates:
        for m in _DATE_NUMERIC_DMY.finditer(text):
            dates.append((m.start(), m.group(0)))

    dates.sort(key=lambda x: x[0])
    return [d[1] for d in dates]


def _parse_date_to_obj(text: str) -> Optional[datetime.date]:
    """Try to parse a text date into a datetime.date object."""
    # "12 March 2024"
    m = _DATE_DMY_LONG.search(text)
    if m:
        day = int(m.group(1))
        month = _MONTH_MAP.get(m.group(2).lower())
        year = int(m.group(3))
        if month:
            try:
                return datetime.date(year, month, day)
            except ValueError:
                pass

    # "March 12, 2024"
    m = _DATE_MDY_LONG.search(text)
    if m:
        month = _MONTH_MAP.get(m.group(1).lower())
        day = int(m.group(2))
        year = int(m.group(3))
        if month:
            try:
                return datetime.date(year, month, day)
            except ValueError:
                pass

    # "2024-03-12"
    m = _DATE_ISO.search(text)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        try:
            return datetime.date(year, month, day)
        except ValueError:
            pass

    # "12/03/2024"
    m = _DATE_NUMERIC_DMY.search(text)
    if m:
        day = int(m.group(1))
        month = int(m.group(2))
        year = int(m.group(3))
        if year < 100:
            year += 2000
        try:
            return datetime.date(year, month, day)
        except ValueError:
            # Maybe it's MM/DD/YYYY
            try:
                return datetime.date(year, day, month)
            except ValueError:
                pass

    return None


def try_text_extraction(query: str) -> Optional[Tuple[str, bool]]:
    """
    Handle text extraction and string processing queries.
    Returns (answer, is_raw) or None.
    is_raw=True means the answer is a raw value (no sentence formatting).
    """
    q = query.strip()
    ql = q.lower()

    # ── Extract date ──────────────────────────────────────────────────────
    if re.match(r"extract\s+(?:the\s+)?date", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            dates = _extract_dates(text)
            if dates:
                return dates[0], True

    # ── Extract email ─────────────────────────────────────────────────────
    if re.match(r"extract\s+(?:the\s+)?(?:email|e-mail)", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            emails = _EMAIL_RE.findall(text)
            if emails:
                return emails[0], True

    # ── Extract URL ───────────────────────────────────────────────────────
    if re.match(r"extract\s+(?:the\s+)?(?:url|link|website)", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            urls = _URL_RE.findall(text)
            if urls:
                return urls[0], True

    # ── Extract phone number ──────────────────────────────────────────────
    if re.match(r"extract\s+(?:the\s+)?(?:phone|telephone|contact)\s*(?:number)?", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            phones = _PHONE_RE.findall(text)
            if phones:
                return phones[0].strip(), True

    # ── Extract number(s) ─────────────────────────────────────────────────
    if re.match(r"extract\s+(?:the\s+)?(?:all\s+)?numbers?", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            nums = _NUMBER_IN_TEXT_RE.findall(text)
            if nums:
                if "all" in ql:
                    return ", ".join(nums), True
                return nums[0], True

    # ── Extract name ──────────────────────────────────────────────────────
    if re.match(r"extract\s+(?:the\s+)?name", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            # Common patterns: "My name is X", "I am X", "name is X"
            m = re.search(r"(?:my\s+name\s+is|i\s+am|name\s+is|called)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", text)
            if m:
                return m.group(1), True
            # Fallback: just use the proper nouns
            proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            if proper_nouns:
                # Filter common words
                common = {"the", "a", "an", "in", "on", "at", "is", "it", "my", "i", "he", "she"}
                filtered = [n for n in proper_nouns if n.lower() not in common]
                if filtered:
                    return filtered[0], True

    # ── Extract word(s) ───────────────────────────────────────────────────
    if re.match(r"extract\s+(?:the\s+)?(?:all\s+)?words?", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            # Check for specific conditions like "starting with" or "containing"
            m_starts_with = re.search(r"starting\s+with\s+['\"]?(\w)['\"]?", ql)
            m_containing = re.search(r"containing\s+['\"]?(\w+)['\"]?", ql)
            words = text.split()
            if m_starts_with:
                letter = m_starts_with.group(1).lower()
                filtered = [w for w in words if w.lower().startswith(letter)]
                return ", ".join(filtered), True
            elif m_containing:
                substr = m_containing.group(1).lower()
                filtered = [w for w in words if substr in w.lower()]
                return ", ".join(filtered), True
            else:
                return " ".join(words), True

    # ── Generic "extract X from Y" ────────────────────────────────────────
    m = re.match(r"extract\s+(.+?)\s+from\s*[:\s]+", ql)
    if m:
        what = m.group(1).strip()
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            # Date fallback
            if "date" in what:
                dates = _extract_dates(text)
                if dates:
                    return dates[0], True
            # Email fallback
            if "email" in what or "e-mail" in what:
                emails = _EMAIL_RE.findall(text)
                if emails:
                    return emails[0], True
            # Number fallback
            if "number" in what:
                nums = _NUMBER_IN_TEXT_RE.findall(text)
                if nums:
                    return nums[0], True
            # URL fallback
            if "url" in what or "link" in what:
                urls = _URL_RE.findall(text)
                if urls:
                    return urls[0], True

    return None


def try_string_operation(query: str) -> Optional[Tuple[str, bool]]:
    """
    Handle string manipulation queries.
    Returns (answer, is_raw) or None.
    """
    q = query.strip()
    ql = q.lower()

    # ── Reverse string ────────────────────────────────────────────────────
    m = re.match(r"reverse\s+(?:the\s+)?(?:string\s+)?[:\s]*(.+)$", q, re.IGNORECASE)
    if m:
        text = m.group(1).strip().rstrip(".")
        # Remove surrounding quotes
        if len(text) >= 2 and text[0] in ('"', "'") and text[-1] in ('"', "'"):
            text = text[1:-1]
        return text[::-1], True

    # ── Uppercase ─────────────────────────────────────────────────────────
    if re.match(r"(?:convert\s+)?(?:to\s+)?(?:upper\s*case|uppercase)", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            return text.upper(), True

    m = re.match(r"(?:convert|change|transform)\s+(.+?)\s+to\s+(?:upper\s*case|uppercase)", q, re.IGNORECASE)
    if m:
        text = m.group(1).strip()
        if len(text) >= 2 and text[0] in ('"', "'") and text[-1] in ('"', "'"):
            text = text[1:-1]
        return text.upper(), True

    # ── Lowercase ─────────────────────────────────────────────────────────
    if re.match(r"(?:convert\s+)?(?:to\s+)?(?:lower\s*case|lowercase)", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            return text.lower(), True

    m = re.match(r"(?:convert|change|transform)\s+(.+?)\s+to\s+(?:lower\s*case|lowercase)", q, re.IGNORECASE)
    if m:
        text = m.group(1).strip()
        if len(text) >= 2 and text[0] in ('"', "'") and text[-1] in ('"', "'"):
            text = text[1:-1]
        return text.lower(), True

    # ── Capitalize / Title Case ───────────────────────────────────────────
    if re.match(r"(?:capitalize|title\s*case)", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            return text.title(), True

    # ── Count words ───────────────────────────────────────────────────────
    if re.search(r"count\s+(?:the\s+)?(?:number\s+of\s+)?words?\s+(?:in|of)", ql) or \
       re.search(r"how\s+many\s+words?\s+(?:in|are\s+in|does)", ql) or \
       re.search(r"word\s+count\s+(?:of|in|for)", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            count = len(text.split())
            return str(count), True

    # ── Count characters ──────────────────────────────────────────────────
    if re.search(r"count\s+(?:the\s+)?(?:number\s+of\s+)?(?:characters?|chars?|letters?)\s+(?:in|of)", ql) or \
       re.search(r"how\s+many\s+(?:characters?|chars?|letters?)\s+(?:in|are\s+in)", ql) or \
       re.search(r"(?:character|char|letter)\s+count\s+(?:of|in|for)", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            return str(len(text)), True

    # ── Length of string ──────────────────────────────────────────────────
    if re.search(r"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?length\s+of", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            return str(len(text)), True

    # ── Palindrome check ──────────────────────────────────────────────────
    if re.search(r"is\s+.+\s+(?:a\s+)?palindrome", ql):
        text = _extract_quoted_text(q)
        if text:
            cleaned = re.sub(r"[^a-zA-Z0-9]", "", text).lower()
            if cleaned == cleaned[::-1]:
                return "Yes", True
            else:
                return "No", True

    # ── Remove vowels ─────────────────────────────────────────────────────
    if re.search(r"remove\s+(?:all\s+)?vowels?\s+(?:from|in)", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            result = re.sub(r"[aeiouAEIOU]", "", text)
            return result, True

    # ── Remove spaces ─────────────────────────────────────────────────────
    if re.search(r"remove\s+(?:all\s+)?spaces?\s+(?:from|in)", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            return text.replace(" ", ""), True

    # ── Remove duplicates from string (characters) ────────────────────────
    if re.search(r"remove\s+(?:duplicate|repeated)\s+(?:characters?|chars?|letters?)", ql):
        text = _extract_quoted_text(q)
        if text is None:
            text = _extract_after_colon(q)
        if text:
            seen = set()
            result = []
            for ch in text:
                if ch not in seen:
                    seen.add(ch)
                    result.append(ch)
            return "".join(result), True

    # ── Replace X with Y in text ──────────────────────────────────────────
    m = re.search(
        r"replace\s+['\"](.+?)['\"]\s+with\s+['\"](.+?)['\"]\s+in\s+['\"](.+?)['\"]",
        q, re.IGNORECASE,
    )
    if m:
        old = m.group(1)
        new = m.group(2)
        text = m.group(3)
        return text.replace(old, new), True

    # Alternate: replace X with Y in: "text"
    m = re.search(
        r"replace\s+['\"]?(.+?)['\"]?\s+with\s+['\"]?(.+?)['\"]?\s+in\s*:\s*['\"](.+?)['\"]",
        q, re.IGNORECASE,
    )
    if m:
        old = m.group(1)
        new = m.group(2)
        text = m.group(3)
        return text.replace(old, new), True

    # ── Concatenate strings ───────────────────────────────────────────────
    if re.search(r"concatenate|concat|join|combine\s+(?:the\s+)?strings?", ql):
        strings = re.findall(r'["\'](.+?)["\']', q)
        if len(strings) >= 2:
            return "".join(strings), True

    # ── Trim/strip whitespace ─────────────────────────────────────────────
    if re.search(r"(?:trim|strip)\s+(?:whitespace\s+)?(?:from\s+)?", ql):
        text = _extract_quoted_text(q)
        if text:
            return text.strip(), True

    # ── First N / Last N characters ───────────────────────────────────────
    m = re.search(r"(?:first|get\s+first)\s+(\d+)\s+(?:characters?|chars?|letters?)\s+(?:of|from|in)\s+", ql)
    if m:
        n = int(m.group(1))
        text = _extract_quoted_text(q)
        if text:
            return text[:n], True

    m = re.search(r"(?:last|get\s+last)\s+(\d+)\s+(?:characters?|chars?|letters?)\s+(?:of|from|in)\s+", ql)
    if m:
        n = int(m.group(1))
        text = _extract_quoted_text(q)
        if text:
            return text[-n:], True

    # ── Substring / slice ─────────────────────────────────────────────────
    m = re.search(r"substring\s+(?:of\s+)?.*?from\s+(?:index\s+)?(\d+)\s+to\s+(?:index\s+)?(\d+)", ql)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
        text = _extract_quoted_text(q)
        if text:
            return text[start:end], True

    # ── Repeat string ─────────────────────────────────────────────────────
    m = re.search(r"repeat\s+['\"](.+?)['\"]\s+(\d+)\s+times?", q, re.IGNORECASE)
    if m:
        text = m.group(1)
        count = int(m.group(2))
        return text * count, True

    # ── Count occurrences ─────────────────────────────────────────────────
    m = re.search(r"(?:count|how\s+many)\s+(?:times?\s+)?(?:does\s+)?['\"](.+?)['\"]\s+(?:appear|occur)\s+in\s+['\"](.+?)['\"]", q, re.IGNORECASE)
    if m:
        substr = m.group(1)
        text = m.group(2)
        return str(text.count(substr)), True

    m = re.search(r"count\s+(?:occurrences?\s+of\s+)?['\"](.+?)['\"]\s+in\s+['\"](.+?)['\"]", q, re.IGNORECASE)
    if m:
        substr = m.group(1)
        text = m.group(2)
        return str(text.count(substr)), True

    # ── Split string ──────────────────────────────────────────────────────
    m = re.search(r"split\s+['\"](.+?)['\"]\s+(?:by|on|using|with)\s+['\"](.+?)['\"]", q, re.IGNORECASE)
    if m:
        text = m.group(1)
        delim = m.group(2)
        parts = text.split(delim)
        return ", ".join(parts), True

    return None


def try_list_operation(query: str) -> Optional[Tuple[str, bool]]:
    """
    Handle list/array operations.
    Returns (answer, is_raw) or None.
    """
    q = query.strip()
    ql = q.lower()

    # ── Helpers: extract numbers from query ──
    def _extract_numbers_from_query(text: str) -> list[float]:
        # Look for numbers after a colon or keyword
        m = re.search(r"[:\s]\s*([\d+\-.,\s]+)$", text)
        if m:
            nums_str = m.group(1)
        else:
            # Extract from brackets
            m = re.search(r"\[([^\]]+)\]", text)
            if m:
                nums_str = m.group(1)
            else:
                # Just find all numbers
                nums_str = text
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?", nums_str)
        return [float(n) for n in nums] if nums else []

    # ── Sort numbers ──────────────────────────────────────────────────────
    if re.search(r"sort\s+(?:the\s+)?(?:following\s+)?(?:numbers?|values?|list|array|elements?)", ql) or \
       re.search(r"(?:arrange|order)\s+(?:the\s+)?(?:following\s+)?(?:numbers?|values?)", ql):
        nums = _extract_numbers_from_query(q)
        if nums:
            nums.sort()
            return ", ".join(_format_number(n) for n in nums), True

    # ── Sort in descending order ──────────────────────────────────────────
    if re.search(r"sort\s+.+\s+(?:in\s+)?(?:descending|reverse)\s*(?:order)?", ql):
        nums = _extract_numbers_from_query(q)
        if nums:
            nums.sort(reverse=True)
            return ", ".join(_format_number(n) for n in nums), True

    # ── Find maximum ──────────────────────────────────────────────────────
    if re.search(r"(?:find\s+)?(?:the\s+)?(?:max(?:imum)?|largest|biggest|greatest)\s+(?:of|in|from|value|number)", ql):
        nums = _extract_numbers_from_query(q)
        if nums:
            return _format_number(max(nums)), True

    # ── Find minimum ──────────────────────────────────────────────────────
    if re.search(r"(?:find\s+)?(?:the\s+)?(?:min(?:imum)?|smallest|least)\s+(?:of|in|from|value|number)", ql):
        nums = _extract_numbers_from_query(q)
        if nums:
            return _format_number(min(nums)), True

    # ── Sum of list ───────────────────────────────────────────────────────
    if re.search(r"(?:find\s+)?(?:the\s+)?(?:sum|total)\s+(?:of|:)", ql):
        nums = _extract_numbers_from_query(q)
        if nums and len(nums) > 2:
            return _format_number(sum(nums)), True

    # ── Average / mean ────────────────────────────────────────────────────
    if re.search(r"(?:find\s+)?(?:the\s+)?(?:average|mean|avg)\s+(?:of|:)", ql):
        nums = _extract_numbers_from_query(q)
        if nums:
            return _format_number(sum(nums) / len(nums)), True

    # ── Median ────────────────────────────────────────────────────────────
    if re.search(r"(?:find\s+)?(?:the\s+)?median\s+(?:of|:)", ql):
        nums = _extract_numbers_from_query(q)
        if nums:
            sorted_nums = sorted(nums)
            n = len(sorted_nums)
            if n % 2 == 1:
                return _format_number(sorted_nums[n // 2]), True
            else:
                return _format_number((sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2), True

    # ── Count items ───────────────────────────────────────────────────────
    if re.search(r"(?:count|how\s+many)\s+(?:items?|elements?|numbers?|values?)\s+(?:in|are\s+in)", ql):
        # Try to find a list
        m = re.search(r"\[([^\]]+)\]", q)
        if m:
            items = [x.strip() for x in m.group(1).split(",") if x.strip()]
            return str(len(items)), True
        nums = _extract_numbers_from_query(q)
        if nums:
            return str(len(nums)), True

    # ── Remove duplicates from list ───────────────────────────────────────
    if re.search(r"remove\s+(?:duplicate|repeated)\s+(?:numbers?|values?|elements?|items?)", ql) or \
       re.search(r"unique\s+(?:numbers?|values?|elements?|items?)", ql):
        m = re.search(r"\[([^\]]+)\]", q)
        if m:
            items = [x.strip() for x in m.group(1).split(",") if x.strip()]
            seen = set()
            unique = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
            return ", ".join(unique), True
        nums = _extract_numbers_from_query(q)
        if nums:
            seen = set()
            unique = []
            for n in nums:
                if n not in seen:
                    seen.add(n)
                    unique.append(n)
            return ", ".join(_format_number(n) for n in unique), True

    # ── Find common elements ──────────────────────────────────────────────
    if re.search(r"(?:find\s+)?(?:the\s+)?(?:common|shared|intersection)\s+(?:elements?|numbers?|values?)", ql):
        brackets = re.findall(r"\[([^\]]+)\]", q)
        if len(brackets) >= 2:
            set1 = set(x.strip() for x in brackets[0].split(","))
            set2 = set(x.strip() for x in brackets[1].split(","))
            common = sorted(set1 & set2)
            return ", ".join(common), True

    # ── Reverse list ──────────────────────────────────────────────────────
    if re.search(r"reverse\s+(?:the\s+)?(?:list|array|order)", ql):
        m = re.search(r"\[([^\]]+)\]", q)
        if m:
            items = [x.strip() for x in m.group(1).split(",") if x.strip()]
            return ", ".join(reversed(items)), True
        nums = _extract_numbers_from_query(q)
        if nums:
            nums.reverse()
            return ", ".join(_format_number(n) for n in nums), True

    return None


def try_number_base_conversion(query: str) -> Optional[Tuple[str, bool]]:
    """Handle number base/format conversions."""
    q = query.strip()
    ql = q.lower()

    # ── Binary to decimal ─────────────────────────────────────────────────
    m = re.search(r"(?:convert\s+)?(?:binary\s+)?([01]+)\s+(?:from\s+binary\s+)?to\s+decimal", ql)
    if m:
        try:
            return str(int(m.group(1), 2)), True
        except ValueError:
            pass

    if re.search(r"convert\s+binary\s+", ql):
        m = re.search(r"([01]+)", q)
        if m:
            try:
                return str(int(m.group(1), 2)), True
            except ValueError:
                pass

    # ── Decimal to binary ─────────────────────────────────────────────────
    m = re.search(r"(?:convert\s+)?(\d+)\s+(?:from\s+decimal\s+)?to\s+binary", ql)
    if m:
        return bin(int(m.group(1)))[2:], True

    if re.search(r"convert\s+(?:decimal\s+)?.*to\s+binary", ql):
        m = re.search(r"(\d+)", q)
        if m:
            return bin(int(m.group(1)))[2:], True

    # ── Hex to decimal ────────────────────────────────────────────────────
    m = re.search(r"(?:convert\s+)?(?:hex(?:adecimal)?\s+)?([0-9a-fA-F]+)\s+(?:from\s+hex(?:adecimal)?\s+)?to\s+decimal", ql)
    if m:
        try:
            return str(int(m.group(1), 16)), True
        except ValueError:
            pass

    # ── Decimal to hex ────────────────────────────────────────────────────
    m = re.search(r"(?:convert\s+)?(\d+)\s+(?:from\s+decimal\s+)?to\s+hex(?:adecimal)?", ql)
    if m:
        return hex(int(m.group(1)))[2:].upper(), True

    # ── Octal to decimal ──────────────────────────────────────────────────
    m = re.search(r"(?:convert\s+)?(?:octal\s+)?([0-7]+)\s+(?:from\s+octal\s+)?to\s+decimal", ql)
    if m:
        try:
            return str(int(m.group(1), 8)), True
        except ValueError:
            pass

    # ── Decimal to octal ──────────────────────────────────────────────────
    m = re.search(r"(?:convert\s+)?(\d+)\s+(?:from\s+decimal\s+)?to\s+octal", ql)
    if m:
        return oct(int(m.group(1)))[2:], True

    # ── Roman to decimal ──────────────────────────────────────────────────
    m = re.search(r"(?:convert\s+)?(?:roman\s+(?:numeral\s+)?)?([IVXLCDM]+)\s+to\s+(?:decimal|number|arabic)", q)
    if m:
        result = _roman_to_int(m.group(1))
        if result > 0:
            return str(result), True

    # ── Decimal to Roman ──────────────────────────────────────────────────
    m = re.search(r"(?:convert\s+)?(\d+)\s+to\s+roman(?:\s+numerals?)?", ql)
    if m:
        result = _int_to_roman(int(m.group(1)))
        if result:
            return result, True

    return None


def _roman_to_int(s: str) -> int:
    roman_vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        val = roman_vals.get(ch, 0)
        if val == 0:
            return 0
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


def _int_to_roman(num: int) -> str:
    if num <= 0 or num > 3999:
        return ""
    vals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    result = []
    for val, sym in vals:
        while num >= val:
            result.append(sym)
            num -= val
    return "".join(result)


def try_date_operation(query: str) -> Optional[Tuple[str, bool]]:
    """Handle date-related queries."""
    q = query.strip()
    ql = q.lower()

    # ── What day of the week ──────────────────────────────────────────────
    if re.search(r"(?:what|which)\s+day\s+(?:of\s+the\s+week|is|was)", ql):
        dt = _parse_date_to_obj(q)
        if dt:
            return calendar.day_name[dt.weekday()], True

    # ── How many days between ─────────────────────────────────────────────
    if re.search(r"how\s+many\s+days?\s+(?:between|from|until|till)", ql):
        dates = _extract_dates(q)
        if len(dates) >= 2:
            d1 = _parse_date_to_obj(dates[0])
            d2 = _parse_date_to_obj(dates[1])
            if d1 and d2:
                delta = abs((d2 - d1).days)
                return str(delta), True

    # ── Is it a leap year ─────────────────────────────────────────────────
    m = re.search(r"is\s+(\d{4})\s+(?:a\s+)?leap\s*year", ql)
    if m:
        year = int(m.group(1))
        if calendar.isleap(year):
            return "Yes", True
        else:
            return "No", True

    # ── Days in month ─────────────────────────────────────────────────────
    m = re.search(r"how\s+many\s+days?\s+(?:in|are\s+in)\s+(" + _MONTH_NAMES_RE + r")(?:\s+(\d{4}))?", ql)
    if m:
        month = _MONTH_MAP.get(m.group(1).lower())
        year = int(m.group(2)) if m.group(2) else 2024
        if month:
            days = calendar.monthrange(year, month)[1]
            return str(days), True

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: WEB CONTEXT & FALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=256)
def _http_get_text(url: str, timeout: float = _WEB_TIMEOUT) -> str:
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return ""
    req = urlrequest.Request(url, headers={"User-Agent": "andromeda-eval-agent/3.0"})
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
# SECTION 8: WIKIPEDIA LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def _wikipedia_summary(query: str) -> str:
    q = _collapse_whitespace(query)
    if not q:
        return ""
    search_endpoint = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=opensearch&search={quote(q)}&limit=1&namespace=0&format=json"
    )
    search_req = urlrequest.Request(
        search_endpoint, headers={"User-Agent": "andromeda-eval-agent/3.0"}
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
    endpoint = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    req = urlrequest.Request(
        endpoint, headers={"User-Agent": "andromeda-eval-agent/3.0"}
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
# SECTION 9: DUCKDUCKGO INSTANT ANSWER
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def _duckduckgo_answer(query: str) -> str:
    q = _collapse_whitespace(query)
    if not q:
        return ""
    endpoint = f"https://api.duckduckgo.com/?q={quote(q)}&format=json&no_html=1&skip_disambig=1"
    req = urlrequest.Request(
        endpoint, headers={"User-Agent": "andromeda-eval-agent/3.0"}
    )
    try:
        with urlrequest.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return ""
    abstract = data.get("AbstractText", "")
    if abstract and isinstance(abstract, str) and len(abstract.strip()) > 10:
        return _collapse_whitespace(abstract)
    answer = data.get("Answer", "")
    if answer and isinstance(answer, str) and len(answer.strip()) > 1:
        return _collapse_whitespace(answer)
    definition = data.get("Definition", "")
    if definition and isinstance(definition, str) and len(definition.strip()) > 10:
        return _collapse_whitespace(definition)
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: GEMINI LLM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_GEMINI_SYSTEM_PROMPT = """\
You are a strict, concise answer engine for an evaluation API. Your responses are scored by cosine similarity against expected answers.

ABSOLUTE RULES:
1. Output EXACTLY the answer — nothing more.
2. NO markdown, NO bullet points, NO numbered lists, NO bold/italic.
3. NO introductory phrases like "Sure!", "Of course!", "Here's the answer:" etc.
4. NO explanations, NO reasoning steps, NO disclaimers.
5. Do NOT repeat the question.
6. If context/assets are provided, base your answer ONLY on them.

RESPONSE FORMAT BY QUERY TYPE:

A) ARITHMETIC:
- Addition → "The sum is X."
- Subtraction → "The difference is X."
- Multiplication → "The product is X."
- Division → "The quotient is X."

B) EXTRACTION (extract X from text):
- Return ONLY the extracted value, no sentence wrapper, no period.
- Example: Extract date from "Meeting on 12 March 2024" → 12 March 2024
- Example: Extract email from "Contact john@test.com" → john@test.com

C) STRING OPERATIONS:
- Return ONLY the result, no sentence wrapper, no period.
- Example: Reverse "hello" → olleh
- Example: Convert "hello" to uppercase → HELLO

D) LIST OPERATIONS:
- Return ONLY the result values, comma-separated, no brackets.
- Example: Sort: 5, 3, 1 → 1, 3, 5
- Example: Max of: 10, 20, 5 → 20

E) FACTUAL QUESTIONS:
- Answer in ONE concise sentence ending with a period.
- Example: What is the capital of France? → The capital of France is Paris.

EXAMPLES:
Q: What is 10 + 15?
A: The sum is 25.

Q: What is 100 - 37?
A: The difference is 63.

Q: Extract date from: "Meeting on 12 March 2024".
A: 12 March 2024

Q: Extract email from: "Contact us at info@example.com for details".
A: info@example.com

Q: Reverse the string "hello".
A: olleh

Q: Convert "hello world" to uppercase.
A: HELLO WORLD

Q: Count words in "the quick brown fox".
A: 4

Q: Sort the numbers: 5, 3, 1, 4, 2.
A: 1, 2, 3, 4, 5

Q: What is the capital of France?
A: The capital of France is Paris.

Q: Who wrote Romeo and Juliet?
A: Romeo and Juliet was written by William Shakespeare.

Q: Is "racecar" a palindrome?
A: YES

Q: Is 9 an odd number?
A: YES

Q: Is 7 a prime number?
A: YES

Q: Is 10 divisible by 3?
A: NO

Q: Is 2024 a leap year?
A: YES

Q: Is 16 a perfect square?
A: YES

Q: What day of the week was January 1, 2024?
A: Monday

Q: Convert 10 to binary.
A: 1010

Q: Find the maximum in: 10, 20, 5, 15.
A: 20

Q: What is the length of "hello"?
A: 5

Q: Remove vowels from "beautiful".
A: btfl

Q: Replace "a" with "o" in "banana".
A: bonono

Q: Alice scored 80, Bob scored 90. Who scored highest?
A: Bob

Q: Tom earned $500, Jane earned $700. Who earned less?
A: Tom

Q: Alice scored 80, Bob scored 90. What is the total?
A: 170

Q: Alice scored 80, Bob scored 90. What is the difference?
A: 10
"""


def _call_gemini(query: str, context: str) -> Optional[str]:
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
                print(f"[EVAL-LOG] Gemini HTTP {status} on {model} (attempt {attempt+1}): {repr(e)}", flush=True)
                if status == 429:
                    wait = _GEMINI_BACKOFF_BASE * (2 ** attempt)
                    time.sleep(min(wait, 10.0))
                    continue
                elif status in (500, 502, 503):
                    time.sleep(1.0)
                    continue
                break
            except (urlerror.URLError, TimeoutError) as e:
                print(f"[EVAL-LOG] Gemini network error on {model} (attempt {attempt+1}): {repr(e)}", flush=True)
                time.sleep(1.0)
                continue
            except Exception as e:
                print(f"[EVAL-LOG] Gemini error on {model} (attempt {attempt+1}): {repr(e)}", flush=True)
                break
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: OUTPUT SANITIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_output(text: str) -> str:
    """
    Clean and normalize sentence-mode output.
    Ensures a single sentence ending with a period.
    """
    if not isinstance(text, str) or not text.strip():
        return "I cannot determine the answer."

    cleaned = _strip_markdown(text)
    cleaned = " ".join(cleaned.replace("\n", " ").split()).strip()

    if not cleaned:
        return "I cannot determine the answer."

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

    if len(cleaned) > 2 and cleaned[0] in ('"', "'") and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()

    if not cleaned:
        return "I cannot determine the answer."

    # Preserve exact arithmetic phrasing (allow decimal points)
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

    first_sentence_match = re.search(r"[!?]|\.(?!\d)", cleaned)
    if first_sentence_match:
        cleaned = cleaned[: first_sentence_match.start() + 1]

    cleaned = cleaned.strip()
    if not cleaned:
        return "I cannot determine the answer."

    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    return cleaned.rstrip(".!? ") + "."


def sanitize_raw_output(text: str) -> str:
    """
    Clean raw-mode output. Minimal processing — just strip markdown/whitespace.
    No sentence formatting, no period added.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = _strip_markdown(text)
    cleaned = " ".join(cleaned.replace("\n", " ").split()).strip()

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

    # Remove trailing period only if this looks like a raw value (not a sentence)
    if cleaned.endswith(".") and not re.search(r"[a-zA-Z]{3,}\.$", cleaned):
        cleaned = cleaned[:-1].strip()

    return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: MAIN ANSWER PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

# ==============================================================================
# SECTION 12A: BOOLEAN YES/NO ENGINE (Level 3)
# ==============================================================================

def _is_perfect_square(n):
    if n < 0: return False
    root = int(math.isqrt(n))
    return root * root == n

def _is_fibonacci(n):
    if n < 0: return False
    return _is_perfect_square(5*n*n+4) or _is_perfect_square(5*n*n-4)


def _is_perfect_cube(n):
    if n < 0:
        r = round(abs(n) ** (1 / 3))
        return -(r ** 3) == n
    r = round(n ** (1 / 3))
    return r ** 3 == n

def try_boolean_question(query):
    """Handle yes/no questions. Returns ('YES'/'NO', True) or None."""
    ql = query.lower().strip().rstrip("?. ")

    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:an?\s+)?(even|odd)(?:\s+number)?", ql)
    if m:
        val = int(float(m.group(1)))
        return ("YES" if (val % 2 == 0) == (m.group(2) == "even") else "NO", True)

    m = re.search(r"is\s+([+-]?\d+)\s+(?:a\s+)?prime(?:\s+number)?", ql)
    if m:
        return ("YES" if _is_prime(int(m.group(1))) else "NO", True)

    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+divisible\s+by\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        a, b = int(float(m.group(1))), int(float(m.group(2)))
        return ("YES" if b != 0 and a % b == 0 else "NO", True)

    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(greater|larger|bigger|more)\s+than\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        return ("YES" if float(m.group(1)) > float(m.group(3)) else "NO", True)

    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(less|smaller|fewer)\s+than\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        return ("YES" if float(m.group(1)) < float(m.group(3)) else "NO", True)

    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+equal\s+to\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        return ("YES" if abs(float(m.group(1)) - float(m.group(2))) < 1e-9 else "NO", True)

    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:a\s+)?(positive|negative|zero)(?:\s+number)?", ql)
    if m:
        val, chk = float(m.group(1)), m.group(2)
        if chk == "positive": return ("YES" if val > 0 else "NO", True)
        if chk == "negative": return ("YES" if val < 0 else "NO", True)
        return ("YES" if val == 0 else "NO", True)

    m = re.search(r"is\s+([+-]?\d+)\s+(?:a\s+)?perfect\s+square", ql)
    if m:
        return ("YES" if _is_perfect_square(int(m.group(1))) else "NO", True)

    m = re.search(r"is\s+([+-]?\d+)\s+(?:a\s+)?perfect\s+cube", ql)
    if m:
        return ("YES" if _is_perfect_cube(int(m.group(1))) else "NO", True)

    m = re.search(r"is\s+([+-]?\d+)\s+(?:a\s+)?fibonacci(?:\s+number)?", ql)
    if m:
        return ("YES" if _is_fibonacci(int(m.group(1))) else "NO", True)

    m = re.search(r"is\s+([+-]?\d+)\s+(?:a\s+)?multiple\s+of\s+([+-]?\d+)", ql)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return ("YES" if b != 0 and a % b == 0 else "NO", True)

    m = re.search(r"is\s+([+-]?\d+)\s+(?:a\s+)?factor\s+of\s+([+-]?\d+)", ql)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return ("YES" if a != 0 and b % a == 0 else "NO", True)

    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+between\s+([+-]?\d+(?:\.\d+)?)\s+and\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        val, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if lo > hi: lo, hi = hi, lo
        return ("YES" if lo <= val <= hi else "NO", True)

    if re.search(r"is\s+.+\s+(?:a\s+)?palindrome", ql):
        text = _extract_quoted_text(query)
        if text:
            c = re.sub(r"[^a-zA-Z0-9]", "", text).lower()
            return ("YES" if c == c[::-1] else "NO", True)
        m2 = re.search(r"is\s+(\w+)\s+(?:a\s+)?palindrome", ql)
        if m2:
            w = m2.group(1).lower()
            return ("YES" if w == w[::-1] else "NO", True)

    m = re.search(r"is\s+(\d{4})\s+(?:a\s+)?leap\s*year", ql)
    if m:
        return ("YES" if calendar.isleap(int(m.group(1))) else "NO", True)

    m = re.search(r"is\s+([+-]?\d+)\s+(?:a\s+)?power\s+of\s+([+-]?\d+)", ql)
    if m:
        val, base = int(m.group(1)), int(m.group(2))
        if base <= 1 or val <= 0:
            return ("YES" if val == 1 else "NO", True)
        n = val
        while n > 1:
            if n % base != 0: return ("NO", True)
            n //= base
        return ("YES", True)

    m = re.search(r"is\s+([+-]?\d+)\s+(?:an?\s+)?(composite|natural|whole)(?:\s+number)?", ql)
    if m:
        val, chk = int(m.group(1)), m.group(2)
        if chk == "composite": return ("YES" if val > 1 and not _is_prime(val) else "NO", True)
        if chk == "natural": return ("YES" if val > 0 else "NO", True)
        if chk == "whole": return ("YES" if val >= 0 else "NO", True)

    m = re.search(r"can\s+([+-]?\d+)\s+be\s+divided\s+evenly\s+by\s+([+-]?\d+)", ql)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return ("YES" if b != 0 and a % b == 0 else "NO", True)

    m = re.search(r'does\s+["\'](.+?)["\']\s+contain\s+["\'](.+?)["\']', query, re.IGNORECASE)
    if m:
        return ("YES" if m.group(2) in m.group(1) else "NO", True)

    return None


# ==============================================================================
# SECTION 12B: COMPARISON / REASONING ENGINE (Level 5)
# ==============================================================================

def _extract_entity_values(text):
    """Extract (name, value) pairs from text."""
    pairs = []
    stopwords = {"who", "what", "which", "the", "a", "an", "it", "he", "she", "they",
                 "and", "or", "but", "is", "are", "was", "were", "has", "have", "had",
                 "do", "does", "did", "will", "would", "can", "could", "should",
                 "return", "find", "give", "name", "tell", "get", "identify",
                 "how", "many", "much", "by", "if", "then", "that", "this",
                 "first", "second", "third", "last", "next", "each", "every",
                 "total", "sum", "average", "mean", "difference", "highest",
                 "lowest", "most", "least", "best", "worst", "more", "less",
                 "did", "does", "scorer", "winner", "loser"}

    verbs = (r"(?:scored|scores?|has|have|had|got|gets|earned|earns|received|receives|"
             r"made|makes|weighs?|costs?|is|was|are|were|runs?|ran|finished|"
             r"took|takes|spent|spends|pays?|paid|sold|sells|ate|eats|"
             r"owns?|holds?|carries|carried|collected|saves?|saved|"
             r"hit|hits|threw|throws|caught|catches|answered|completed|"
             r"drank|drinks|read|reads|wrote|writes|walked|walks|drove|drives)")

    # Allow labels such as "A", "Q1", "Team A", "Product 2".
    name_pat = r"([A-Za-z][A-Za-z0-9]*(?:\s+[A-Za-z0-9][A-Za-z0-9]*){0,3})"
    value_pat = r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"

    def _to_float(num_text):
        return float(num_text.replace(",", ""))

    def _normalize_name(name_text):
        name_text = re.sub(r"^(?:and|or|but)\s+", "", name_text.strip(), flags=re.IGNORECASE)
        return _collapse_whitespace(name_text)

    def _is_valid_name(name_text):
        low = name_text.lower()
        if len(name_text) == 1 and name_text.isalpha() and name_text.isupper():
            return True
        return low not in stopwords

    # Pattern 1: "Name verb Number"
    for m in re.finditer(
        name_pat + r"\s+" + verbs + r"\s+\$?" + value_pat,
        text,
        re.IGNORECASE,
    ):
        name = _normalize_name(m.group(1))
        if _is_valid_name(name):
            pairs.append((name, _to_float(m.group(2))))
    if pairs:
        return pairs

    # Pattern 2: "Name: Number" or "Name - Number" or "Name = Number"
    for m in re.finditer(name_pat + r"\s*[:=\-]\s*\$?" + value_pat, text, re.IGNORECASE):
        name = _normalize_name(m.group(1))
        if _is_valid_name(name):
            pairs.append((name, _to_float(m.group(2))))
    if pairs:
        return pairs

    # Pattern 3: case-insensitive "name verb number"
    for m in re.finditer(
        name_pat + r"\s+" + verbs + r"\s+\$?" + value_pat,
        text, re.IGNORECASE,
    ):
        name = _normalize_name(m.group(1))
        if _is_valid_name(name):
            pairs.append((name, _to_float(m.group(2))))
    if pairs:
        return pairs

    # Pattern 4a: 'Name with X points/marks/runs'
    for m in re.finditer(name_pat + r'\s+with\s+' + value_pat + r'\s+(?:points?|marks?|runs?|goals?|votes?|items?|units?)', text, re.IGNORECASE):
        name = _normalize_name(m.group(1))
        if _is_valid_name(name):
            pairs.append((name, _to_float(m.group(2))))
    if pairs:
        return pairs

    # Pattern 4b: "Name (Number)" or "Name [Number]"
    for m in re.finditer(r"([A-Z][a-zA-Z]+)\s*[\(\[]\s*(\d+(?:\.\d+)?)\s*[\)\]]", text):
        pairs.append((m.group(1).strip(), float(m.group(2))))
    return pairs


def try_comparison_query(query):
    """Handle comparison/reasoning queries. Returns (answer, True) or None."""
    q = query.strip()
    ql = q.lower()

    # Extract entities first - if we have entities, try to answer
    entities = _extract_entity_values(q)

    # Handle single entity lookup
    if len(entities) == 1:
        m = re.search(r"what\s+(?:did|does|is|was)\s+(\w+)\s+(?:score|get|earn|have|make|cost|weigh)", ql)
        if m:
            target = m.group(1).lower()
            for name, val in entities:
                if name.lower() == target:
                    return _format_number(val), True

    if len(entities) < 2:
        return None

    # Build helpers
    entity_dict = {name.lower(): (name, val) for name, val in entities}
    sorted_desc = sorted(entities, key=lambda x: x[1], reverse=True)
    sorted_asc = sorted(entities, key=lambda x: x[1])

    # ── RANKING: second/third/nth highest/lowest ──
    ordinals = {"second": 2, "third": 3, "fourth": 4, "fifth": 5,
                "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}
    m = re.search(r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th)\s+(highest|lowest|best|worst|most|least|top|bottom|largest|smallest)", ql)
    if m:
        idx = ordinals.get(m.group(1), 2) - 1
        if m.group(2) in ("lowest", "worst", "least", "bottom", "smallest"):
            if idx < len(sorted_asc):
                return sorted_asc[idx][0], True
        else:
            if idx < len(sorted_desc):
                return sorted_desc[idx][0], True

    # Place-position phrasing: "who came second", "2nd place", "first/last place"
    m = re.search(r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last)\b(?:\s+(?:place|position|rank))?", ql)
    if m:
        token = m.group(1)
        if token == "last":
            return sorted_asc[0][0], True
        idx = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
               "fourth": 3, "4th": 3, "fifth": 4, "5th": 4}.get(token)
        if idx is not None and idx < len(sorted_desc):
            return sorted_desc[idx][0], True

    # ── RANK/SORT/ORDER ──
    if re.search(r"\b(rank|sort|order|arrange|list)\b", ql):
        if re.search(r"\b(ascending|low.to.high|smallest.first|increasing)\b", ql):
            return ", ".join(n for n, v in sorted_asc), True
        return ", ".join(n for n, v in sorted_desc), True

    # ── LOOKUP: what did X score? ──
    m = re.search(r"what\s+(?:did|does|is|was)\s+(\w+)'?s?\s+(?:score|total|mark|point|value|earning|cost|weight|age)", ql)
    if not m:
        m = re.search(r"what\s+(?:did|does)\s+(\w+)\s+(?:score|get|earn|have|make|cost|weigh)", ql)
    if not m:
        m = re.search(r"(\w+)'s\s+(?:score|total|mark|point|value)", ql)
    if m:
        target = m.group(1).lower()
        for name, val in entities:
            if name.lower() == target:
                return _format_number(val), True

    # ── YES/NO: did X score more than Y? ──
    m = re.search(r"did\s+(\w+)\s+(?:score|earn|get|make|have)\s+(?:more|higher|greater|better)\s+than\s+(\w+)", ql)
    if not m:
        m = re.search(r"did\s+(\w+)\s+(?:beat|outperform|outscore|defeat)\s+(\w+)", ql)
    if m:
        v1 = entity_dict.get(m.group(1).lower(), (None, None))[1]
        v2 = entity_dict.get(m.group(2).lower(), (None, None))[1]
        if v1 is not None and v2 is not None:
            return ("YES" if v1 > v2 else "NO"), True

    m = re.search(r"did\s+(\w+)\s+(?:score|earn|get)\s+(?:less|lower|fewer)\s+than\s+(\w+)", ql)
    if m:
        v1 = entity_dict.get(m.group(1).lower(), (None, None))[1]
        v2 = entity_dict.get(m.group(2).lower(), (None, None))[1]
        if v1 is not None and v2 is not None:
            return ("YES" if v1 < v2 else "NO"), True

    # ── BY HOW MUCH ──
    m = re.search(r"(?:by\s+)?how\s+much\s+(?:did\s+)?(\w+)\s+(?:beat|outscore|lead|outperform|score\s+more)", ql)
    if not m:
        m = re.search(r"difference\s+between\s+(\w+)(?:'s?)?\s+and\s+(\w+)", ql)
    if m:
        n1 = m.group(1).lower()
        n2 = m.group(2).lower() if m.lastindex >= 2 else None
        v1 = entity_dict.get(n1, (None, None))[1]
        if n2:
            v2 = entity_dict.get(n2, (None, None))[1]
        else:
            vals = [v for nm, v in entities if nm.lower() != n1]
            v2 = vals[0] if vals else None
        if v1 is not None and v2 is not None:
            return _format_number(abs(v1 - v2)), True

    # ── PASS/FAIL with threshold ──
    m = re.search(r"who\s+(?:passed|cleared|qualified)(?:\s+(?:with|above|over|scoring\s+above|scoring\s+over)\s+(\d+))?", ql)
    if m:
        threshold = float(m.group(1)) if m.group(1) else 50.0
        passed = [name for name, val in entities if val >= threshold]
        if passed:
            return ", ".join(passed), True

    m = re.search(r"who\s+(?:failed|did\s+not\s+pass)(?:\s+(?:below|under|scoring\s+below)\s+(\d+))?", ql)
    if m:
        threshold = float(m.group(1)) if m.group(1) else 50.0
        failed = [name for name, val in entities if val < threshold]
        if failed:
            return ", ".join(failed), True

    # ── COUNT ABOVE/BELOW ──
    m = re.search(r"how\s+many\s+(?:\w+\s+)?(?:scored|got|earned|have)\s+(?:above|over|more\s+than|greater\s+than)\s+(\d+)", ql)
    if m:
        return str(sum(1 for _, v in entities if v > float(m.group(1)))), True

    m = re.search(r"how\s+many\s+(?:\w+\s+)?(?:scored|got|earned|have)\s+(?:below|under|less\s+than)\s+(\d+)", ql)
    if m:
        return str(sum(1 for _, v in entities if v < float(m.group(1)))), True

    # ── Generic keywords (catch-all) ──
    # MAX
    if re.search(r"\b(highest|most|best|greater|bigger|more|top|won|winner|older|taller|heavier|faster|expensive|maximum|max)\b", ql):
        best_val = max(v for _, v in entities)
        for name, val in entities:
            if val == best_val:
                return name, True

    # MIN
    if re.search(r"\b(lowest|least|worst|smaller|less|fewer|bottom|lost|loser|younger|shorter|lighter|slower|cheaper|cheapest|minimum|min)\b", ql):
        worst_val = min(v for _, v in entities)
        for name, val in entities:
            if val == worst_val:
                return name, True

    # TOTAL/SUM
    if re.search(r"\b(total|sum|combined|altogether|overall)\b", ql):
        return _format_number(sum(v for _, v in entities)), True

    # AVERAGE/MEAN
    if re.search(r"\b(average|mean|avg)\b", ql):
        return _format_number(sum(v for _, v in entities) / len(entities)), True

    # DIFFERENCE
    if re.search(r"\b(difference|gap|margin|spread|range)\b", ql):
        vals = [v for _, v in entities]
        return _format_number(abs(max(vals) - min(vals))), True

    # COUNT
    if re.search(r"\bhow\s+many\b", ql):
        return str(len(entities)), True

    # SCORER/EARNER fallback → treat as MAX
    if re.search(r"\b(scorer|earner|performer|achiever)\b", ql):
        best_val = max(v for _, v in entities)
        for name, val in entities:
            if val == best_val:
                return name, True

    # WHO questions fallback - default to highest
    if re.search(r'\bwho\b', ql):
        best_val = max(v for _, v in entities)
        for name, val in entities:
            if val == best_val:
                return name, True

    # CATCH-ALL: any query with entities and question mark -> highest
    if '?' in q:
        best_val = max(v for _, v in entities)
        for name, val in entities:
            if val == best_val:
                return name, True

    return None


def _is_extraction_query(query):
    ql = query.lower().strip()
    patterns = [
        r"^extract\s+", r"^reverse\s+", r"^convert\s+.+\s+to\s+(?:upper|lower|binary|hex|octal|decimal|roman)",
        r"^sort\s+", r"^count\s+", r"^find\s+(?:the\s+)?(?:max|min|largest|smallest|common)",
        r"^remove\s+", r"^replace\s+", r"^concatenate\s+", r"^split\s+",
        r"^trim\s+", r"^strip\s+",
        r"^(?:how\s+many|what(?:'s|\s+is)\s+(?:the\s+)?length)",
        r"^is\s+.+\s+(?:a\s+)?palindrome",
        r"^(?:first|last)\s+\d+\s+(?:char|letter)",
        r"^repeat\s+", r"^(?:what|which)\s+day\s+",
        r"^how\s+many\s+days?\s+", r"^is\s+\d{4}\s+.*leap",
        r"^unique\s+", r"word\s+count", r"character\s+count",
        r"^is\s+", r"^does\s+", r"^can\s+",
        r"who\s+(?:scored|has|had|got|earned|won|is|was)",
        r"which\s+(?:is|was|has|had)",
    ]
    return any(re.search(p, ql) for p in patterns)


def generate_answer(query, assets):
    """
    Main pipeline. Returns (answer, is_raw).
    Order: boolean -> comparison -> extraction -> string -> list -> base -> date -> arith -> conv -> gemini -> fallbacks
    """
    context = _assets_context(assets)

    # 1. Boolean yes/no (Level 3)
    result = try_boolean_question(query)
    if result is not None:
        return result

    # 2. Comparison/reasoning (Level 5)
    result = try_comparison_query(query)
    if result is not None:
        return result

    # 3. Text extraction (Level 2)
    result = try_text_extraction(query)
    if result is not None:
        return result

    # 4. String operations
    result = try_string_operation(query)
    if result is not None:
        return result

    # 5. List operations
    result = try_list_operation(query)
    if result is not None:
        return result

    # 6. Number base conversion
    result = try_number_base_conversion(query)
    if result is not None:
        return result

    # 7. Date operations
    result = try_date_operation(query)
    if result is not None:
        return result

    # 8. Arithmetic (Level 1)
    arith = parse_arithmetic_query(query)
    if arith is not None:
        return arith, False

    # 9. Conversions
    conv = try_conversion(query)
    if conv is not None:
        return conv, False

    # 10. Gemini LLM
    is_raw = _is_extraction_query(query)
    gemini = _call_gemini(query, context)
    if gemini:
        return gemini, is_raw

    # 11. Fallbacks
    for fn in [_duckduckgo_answer, _wikipedia_summary]:
        r = fn(query)
        if r:
            return r, False
    ext = _extractive_answer(query, context)
    if ext:
        return ext, False

    return "I cannot determine the answer.", False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: REQUEST HANDLING & FLASK ROUTES
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
    return {
        "output": text,
        "answer": text,
        "result": text,
        "response": text,
    }


def extract_payload() -> Any:
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
    start_time = time.time()
    payload = extract_payload()

    is_valid, err, query, assets = validate_payload(payload)
    if not is_valid:
        print(f"[EVAL-LOG] Invalid Payload: {payload}", flush=True)
        return jsonify({"error": err}), 400

    print(f"\n[EVAL-LOG] Query: {query}", flush=True)
    print(f"[EVAL-LOG] Assets: {assets}", flush=True)

    raw_output, is_raw = generate_answer(query, assets)

    if is_raw:
        final_output = sanitize_raw_output(raw_output)
    else:
        final_output = sanitize_output(raw_output)

    if not final_output:
        final_output = "I cannot determine the answer."

    elapsed_ms = int((time.time() - start_time) * 1000)
    print(f"[EVAL-LOG] Output: {final_output} (raw={is_raw}, {elapsed_ms}ms)\n", flush=True)

    return jsonify(build_output_payload(final_output)), 200


@app.route("/health", methods=["GET"])
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "5.0"}), 200


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
