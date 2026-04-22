# ==============================================================================
# ANDROMEDA EVALUATION ENGINE v4.0
# Comprehensive Q&A + Text Processing + Boolean engine for hackathon evaluation.
# Handles: arithmetic, conversions, text extraction, string ops, list ops,
#          boolean yes/no questions, base conversions, date ops, LLM fallback.
# Single-file Flask application - no extra pip deps beyond Flask/gunicorn.
# ==============================================================================
from __future__ import annotations

import ast
import math
import operator
import re
import json
import os
import datetime
import calendar
from functools import lru_cache
import time
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import quote
from typing import Any, Optional, Tuple

from flask import Flask, jsonify, request

app = Flask(__name__)


# ==============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# ==============================================================================

_GEMINI_TIMEOUT = 15.0
_GEMINI_MAX_RETRIES = 5
_GEMINI_BACKOFF_BASE = 2.0
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
_BULLET_RE = re.compile(r"^\s*[-*\u2022]\s+", re.MULTILINE)
_NUMBERED_LIST_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)

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


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================

def _collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _strip_html(text: str) -> str:
    return _collapse_whitespace(_HTML_TAG_RE.sub(" ", text))


def _strip_markdown(text: str) -> str:
    text = _CODE_FENCE_RE.sub("", text)
    text = _MARKDOWN_HEADING_RE.sub("", text)
    text = _MARKDOWN_BOLD_RE.sub(r"\1", text)
    text = _MARKDOWN_ITALIC_RE.sub(r"\1", text)
    text = _MARKDOWN_CODE_RE.sub(r"\1", text)
    text = _BULLET_RE.sub("", text)
    text = _NUMBERED_LIST_RE.sub("", text)
    return _collapse_whitespace(text)


def _format_number(value: float) -> str:
    if math.isinf(value) or math.isnan(value):
        return "undefined"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    rounded = round(value, 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _extract_quoted_text(query: str) -> Optional[str]:
    m = re.search(r'["\u201c](.+?)["\u201d]', query)
    if m:
        return m.group(1)
    m = re.search(r"['\u2018](.+?)['\u2019]", query)
    if m:
        return m.group(1)
    return None


def _extract_after_colon(query: str) -> Optional[str]:
    m = re.search(r":\s*(.+)$", query)
    if m:
        text = m.group(1).strip().rstrip(".")
        if len(text) >= 2 and text[0] in ('"', "'", "\u201c") and text[-1] in ('"', "'", "\u201d"):
            text = text[1:-1]
        return text
    return None


# ==============================================================================
# SECTION 3: SAFE MATH EXPRESSION EVALUATOR
# ==============================================================================

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


def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if op_func is operator.pow and right > 1000:
            raise ValueError
        return op_func(left, right)
    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError
        return op_func(_safe_eval_node(node.operand))
    raise ValueError


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


# ==============================================================================
# SECTION 4: ARITHMETIC ENGINE
# ==============================================================================

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

_WORD_NUMBERS = {
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
    re.IGNORECASE,
)


def _word_to_number(word):
    word = word.strip().lower().replace("-", " ").replace("  ", " ")
    if word in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[word])
    m = _COMPOUND_NUM_RE.match(word)
    if m:
        return float(_WORD_NUMBERS.get(m.group(1).lower(), 0) + _WORD_NUMBERS.get(m.group(2).lower(), 0))
    return None


def _try_parse_number(text):
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

_SUBTRACT_FROM_RE = re.compile(rf"\bsubtract\s+{_NUM_OR_WORD}\s+from\s+{_NUM_OR_WORD}\b", re.IGNORECASE)
_DIVIDE_BY_RE = re.compile(rf"\bdivide\s+{_NUM_OR_WORD}\s+by\s+{_NUM_OR_WORD}\b", re.IGNORECASE)

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
_WHAT_IS_EXPR_RE = re.compile(
    r"(?:what(?:'s|\s+is)\s+|calculate\s+|compute\s+|evaluate\s+|solve\s+)([\d+\-*/().%^xX\s]+)\s*\??$",
    re.IGNORECASE,
)


def _is_prime(n):
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


def _is_perfect_square(n):
    if n < 0:
        return False
    root = int(math.isqrt(n))
    return root * root == n


def _is_perfect_cube(n):
    if n < 0:
        cube_root = -int(round(abs(n) ** (1.0 / 3.0)))
    else:
        cube_root = int(round(n ** (1.0 / 3.0)))
    return cube_root ** 3 == n


def _is_fibonacci(n):
    if n < 0:
        return False
    # n is Fibonacci iff 5*n^2 + 4 or 5*n^2 - 4 is a perfect square
    return _is_perfect_square(5 * n * n + 4) or _is_perfect_square(5 * n * n - 4)


def parse_arithmetic_query(query):
    """Parse and solve arithmetic queries. Returns answer string or None.
    NOTE: Prime/even/odd checks are in try_boolean_question instead."""
    q = query.strip()

    match = _ARITH_EXPR_RE.match(q)
    if match:
        left, op_token, right = float(match.group(1)), match.group(2), float(match.group(3))
        operation = _OP_MAP.get(op_token)
        if operation:
            return _solve_basic(operation, left, right)

    for regex, handler in [
        (_SQRT_RE, lambda m: f"The square root of {_format_number(float(m.group(1)))} is {_format_number(math.sqrt(float(m.group(1))))}." if float(m.group(1)) >= 0 else "The square root is undefined for negative numbers."),
        (_SQUARED_RE, lambda m: f"{_format_number(float(m.group(1)))} squared is {_format_number(float(m.group(1)) ** 2)}."),
        (_CUBED_RE, lambda m: f"{_format_number(float(m.group(1)))} cubed is {_format_number(float(m.group(1)) ** 3)}."),
    ]:
        m = regex.search(q)
        if m:
            return handler(m)

    m = _POWER_RE.search(q)
    if m:
        base, exp = float(m.group(1)), float(m.group(2))
        try:
            return f"{_format_number(base)} to the power of {_format_number(exp)} is {_format_number(base ** exp)}."
        except (OverflowError, ValueError):
            return "The result is undefined."

    m = _FACTORIAL_RE.search(q)
    if m:
        val = int(float(m.group(1)))
        if val < 0:
            return "The factorial is undefined for negative numbers."
        if val > 170:
            return "The factorial is too large to compute."
        return f"The factorial of {val} is {math.factorial(val)}."

    m = _PERCENT_OF_RE.search(q)
    if m:
        pct, base = float(m.group(1)), float(m.group(2))
        return f"{_format_number(pct)}% of {_format_number(base)} is {_format_number(pct / 100.0 * base)}."

    m = _REMAINDER_RE.search(q)
    if m:
        left, right = float(m.group(1)), float(m.group(2))
        if right == 0:
            return "The remainder is undefined."
        return f"The remainder when {_format_number(left)} is divided by {_format_number(right)} is {_format_number(left % right)}."

    m = _ABS_RE.search(q)
    if m:
        val = float(m.group(1))
        return f"The absolute value of {_format_number(val)} is {_format_number(abs(val))}."

    m = _GCD_RE.search(q)
    if m:
        a, b = int(float(m.group(1))), int(float(m.group(2)))
        return f"The GCD of {a} and {b} is {math.gcd(a, b)}."

    m = _LCM_RE.search(q)
    if m:
        a, b = int(float(m.group(1))), int(float(m.group(2)))
        return f"The LCM of {a} and {b} is {abs(a * b) // math.gcd(a, b) if a and b else 0}."

    m = _LOG_RE.search(q)
    if m:
        base, val = float(m.group(1)), float(m.group(2))
        if val <= 0 or base <= 0 or base == 1:
            return "The logarithm is undefined."
        return f"The logarithm base {_format_number(base)} of {_format_number(val)} is {_format_number(math.log(val) / math.log(base))}."

    m = _NATURAL_LOG_RE.search(q)
    if m:
        val = float(m.group(1))
        if val <= 0:
            return "The natural logarithm is undefined for non-positive numbers."
        return f"The natural logarithm of {_format_number(val)} is {_format_number(math.log(val))}."

    # "subtract X from Y" / "divide X by Y"
    m = _SUBTRACT_FROM_RE.search(q)
    if m:
        n1, n2 = _try_parse_number(m.group(1)), _try_parse_number(m.group(2))
        if n1 is not None and n2 is not None:
            return _solve_basic("sub", n2, n1)

    m = _DIVIDE_BY_RE.search(q)
    if m:
        n1, n2 = _try_parse_number(m.group(1)), _try_parse_number(m.group(2))
        if n1 is not None and n2 is not None:
            return _solve_basic("div", n1, n2)

    for operation, pattern in _NL_PATTERNS:
        m = pattern.search(q)
        if m:
            n1, n2 = _try_parse_number(m.group(1)), _try_parse_number(m.group(2))
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
        left, op_token, right = float(match.group(1)), match.group(2), float(match.group(3))
        operation = _OP_MAP.get(op_token)
        if operation:
            return _solve_basic(operation, left, right)

    return None


def _solve_basic(operation, left, right):
    label, op_func = _OP_LABEL.get(operation, ("result", operator.add))
    if operation == "div" and right == 0:
        return "The quotient is undefined."
    try:
        result = op_func(left, right)
    except (OverflowError, ZeroDivisionError, ValueError):
        return f"The {label} is undefined."
    return f"The {label} is {_format_number(result)}."


def _detect_operation_label(expr):
    e = expr.strip()
    if "+" in e and not any(c in e for c in "-*/^%"):
        return "sum"
    if "-" in e and not any(c in e for c in "+*/^%"):
        return "difference"
    if "*" in e and not any(c in e for c in "+-/^%"):
        return "product"
    if "/" in e and not any(c in e for c in "+-*^%"):
        return "quotient"
    return "result"


# ==============================================================================
# SECTION 5: UNIT CONVERSION ENGINE
# ==============================================================================

_TEMP_CONVERT_RE = re.compile(
    rf"{_NUM}\s*[^a-zA-Z]*?(?:degrees?\s+)?"
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

_LENGTH_BASE = {"m": 1.0, "meter": 1.0, "meters": 1.0, "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0, "mi": 1609.344, "mile": 1609.344, "miles": 1609.344, "ft": 0.3048, "foot": 0.3048, "feet": 0.3048, "in": 0.0254, "inch": 0.0254, "inches": 0.0254, "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01}
_MASS_BASE = {"kg": 1.0, "kilogram": 1.0, "kilograms": 1.0, "g": 0.001, "gram": 0.001, "grams": 0.001, "lb": 0.453592, "lbs": 0.453592, "pound": 0.453592, "pounds": 0.453592, "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495}
_VOLUME_BASE = {"l": 1.0, "liter": 1.0, "liters": 1.0, "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541}
_TIME_BASE = {"sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0, "day": 86400.0, "days": 86400.0}
_ALL_UNITS = {}
for _d in [_LENGTH_BASE, _MASS_BASE, _VOLUME_BASE, _TIME_BASE]:
    _ALL_UNITS.update(_d)
_UNIT_DISPLAY = {"m": "meters", "km": "kilometers", "mi": "miles", "ft": "feet", "in": "inches", "cm": "centimeters", "kg": "kilograms", "g": "grams", "lb": "pounds", "lbs": "pounds", "oz": "ounces", "l": "liters", "gal": "gallons", "sec": "seconds", "secs": "seconds", "min": "minutes", "mins": "minutes", "hr": "hours", "hrs": "hours", "day": "days", "meter": "meters", "meters": "meters", "kilometer": "kilometers", "kilometers": "kilometers", "mile": "miles", "miles": "miles", "foot": "feet", "feet": "feet", "inch": "inches", "inches": "inches", "centimeter": "centimeters", "centimeters": "centimeters", "kilogram": "kilograms", "kilograms": "kilograms", "gram": "grams", "grams": "grams", "pound": "pounds", "pounds": "pounds", "ounce": "ounces", "ounces": "ounces", "liter": "liters", "liters": "liters", "gallon": "gallons", "gallons": "gallons", "second": "seconds", "seconds": "seconds", "minute": "minutes", "minutes": "minutes", "hour": "hours", "hours": "hours", "days": "days"}


def _same_category(u1, u2):
    for cat in [_LENGTH_BASE, _MASS_BASE, _VOLUME_BASE, _TIME_BASE]:
        if u1 in cat and u2 in cat:
            return True
    return False


def try_conversion(query):
    q = query.strip()
    m = _TEMP_CONVERT_RE.search(q)
    if m:
        val = float(m.group(1))
        from_u = _TEMP_NAMES.get(m.group(2).lower(), "")
        to_u = _TEMP_NAMES.get(m.group(3).lower(), "")
        if from_u and to_u:
            result = _convert_temp(val, from_u, to_u)
            if result is not None:
                return f"{_format_number(val)} degrees {from_u} is {_format_number(result)} degrees {to_u}."
    m = _UNIT_CONVERT_RE.search(q)
    if m:
        val = float(m.group(1))
        from_u, to_u = m.group(2).lower(), m.group(3).lower()
        if from_u in _ALL_UNITS and to_u in _ALL_UNITS and _same_category(from_u, to_u):
            result = val * _ALL_UNITS[from_u] / _ALL_UNITS[to_u]
            return f"{_format_number(val)} {_UNIT_DISPLAY.get(from_u, from_u)} is {_format_number(result)} {_UNIT_DISPLAY.get(to_u, to_u)}."
    return None


def _convert_temp(val, from_u, to_u):
    if from_u == to_u:
        return val
    c = val if from_u == "Celsius" else (val - 32) * 5.0 / 9.0 if from_u == "Fahrenheit" else val - 273.15 if from_u == "Kelvin" else None
    if c is None:
        return None
    return c if to_u == "Celsius" else c * 9.0 / 5.0 + 32 if to_u == "Fahrenheit" else c + 273.15 if to_u == "Kelvin" else None


# ==============================================================================
# SECTION 6: BOOLEAN YES/NO QUESTION ENGINE (Level 3)
# ==============================================================================

def try_boolean_question(query: str) -> Optional[Tuple[str, bool]]:
    """
    Handle yes/no boolean questions. Returns ("YES"/"NO", True) or None.
    All boolean answers are uppercase YES/NO in raw mode.
    """
    q = query.strip()
    ql = q.lower().rstrip("?. ")

    # -- Is X an even/odd number? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:an?\s+)?(even|odd)(?:\s+number)?", ql)
    if m:
        val = int(float(m.group(1)))
        check = m.group(2)
        if check == "even":
            return ("YES" if val % 2 == 0 else "NO", True)
        else:
            return ("YES" if val % 2 != 0 else "NO", True)

    # -- Is X a prime number? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:a\s+)?prime(?:\s+number)?", ql)
    if m:
        val = int(float(m.group(1)))
        return ("YES" if _is_prime(val) else "NO", True)

    # -- Is X divisible by Y? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+divisible\s+by\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        a, b = int(float(m.group(1))), int(float(m.group(2)))
        if b == 0:
            return ("NO", True)
        return ("YES" if a % b == 0 else "NO", True)

    # -- Is X greater/less than Y? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(greater|larger|bigger|more)\s+than\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        a, b = float(m.group(1)), float(m.group(3))
        return ("YES" if a > b else "NO", True)

    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(less|smaller|fewer)\s+than\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        a, b = float(m.group(1)), float(m.group(3))
        return ("YES" if a < b else "NO", True)

    # -- Is X equal to Y? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+equal\s+to\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return ("YES" if abs(a - b) < 1e-9 else "NO", True)

    # -- Is X positive/negative/zero? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:a\s+)?(positive|negative|zero)(?:\s+number)?", ql)
    if m:
        val = float(m.group(1))
        check = m.group(2)
        if check == "positive":
            return ("YES" if val > 0 else "NO", True)
        elif check == "negative":
            return ("YES" if val < 0 else "NO", True)
        else:
            return ("YES" if val == 0 else "NO", True)

    # -- Is X a perfect square? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:a\s+)?perfect\s+square", ql)
    if m:
        val = int(float(m.group(1)))
        return ("YES" if _is_perfect_square(val) else "NO", True)

    # -- Is X a perfect cube? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:a\s+)?perfect\s+cube", ql)
    if m:
        val = int(float(m.group(1)))
        return ("YES" if _is_perfect_cube(val) else "NO", True)

    # -- Is X a fibonacci number? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:a\s+)?fibonacci(?:\s+number)?", ql)
    if m:
        val = int(float(m.group(1)))
        return ("YES" if _is_fibonacci(val) else "NO", True)

    # -- Is X a multiple of Y? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:a\s+)?multiple\s+of\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        a, b = int(float(m.group(1))), int(float(m.group(2)))
        if b == 0:
            return ("NO", True)
        return ("YES" if a % b == 0 else "NO", True)

    # -- Is X a factor of Y? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:a\s+)?factor\s+of\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        a, b = int(float(m.group(1))), int(float(m.group(2)))
        if a == 0:
            return ("NO", True)
        return ("YES" if b % a == 0 else "NO", True)

    # -- Is X between Y and Z? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+between\s+([+-]?\d+(?:\.\d+)?)\s+and\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        val, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if lo > hi:
            lo, hi = hi, lo
        return ("YES" if lo <= val <= hi else "NO", True)

    # -- Is "X" a palindrome? --
    if re.search(r"is\s+.+\s+(?:a\s+)?palindrome", ql):
        text = _extract_quoted_text(q)
        if text:
            cleaned = re.sub(r"[^a-zA-Z0-9]", "", text).lower()
            return ("YES" if cleaned == cleaned[::-1] else "NO", True)
        # Try without quotes - maybe the word is just in the query
        m = re.search(r"is\s+(\w+)\s+(?:a\s+)?palindrome", ql)
        if m:
            word = m.group(1).lower()
            return ("YES" if word == word[::-1] else "NO", True)

    # -- Is YEAR a leap year? --
    m = re.search(r"is\s+(\d{4})\s+(?:a\s+)?leap\s*year", ql)
    if m:
        year = int(m.group(1))
        return ("YES" if calendar.isleap(year) else "NO", True)

    # -- Does "X" contain "Y"? --
    m = re.search(r'does\s+["\'](.+?)["\']\s+contain\s+["\'](.+?)["\']', q, re.IGNORECASE)
    if m:
        text, substr = m.group(1), m.group(2)
        return ("YES" if substr in text else "NO", True)

    # Alternate: does X contain Y (without quotes)
    m = re.search(r"does\s+(.+?)\s+contain\s+(.+?)$", ql)
    if m:
        text = m.group(1).strip().strip("'\"")
        substr = m.group(2).strip().strip("'\"").rstrip("?. ")
        return ("YES" if substr in text else "NO", True)

    # -- Is "X" empty? --
    m = re.search(r'is\s+["\']([^"\']*?)["\']\s+(?:an?\s+)?empty(?:\s+string)?', q, re.IGNORECASE)
    if m:
        return ("YES" if m.group(1) == "" else "NO", True)

    # -- Are X and Y equal? / Are X and Y the same? --
    m = re.search(r"are\s+([+-]?\d+(?:\.\d+)?)\s+and\s+([+-]?\d+(?:\.\d+)?)\s+(?:equal|the\s+same)", ql)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return ("YES" if abs(a - b) < 1e-9 else "NO", True)

    # -- Is X a vowel/consonant? --
    m = re.search(r"""is\s+['""]?([a-zA-Z])['""]?\s+(?:a\s+)?(vowel|consonant)""", ql)
    if m:
        letter = m.group(1).lower()
        check = m.group(2)
        is_vowel = letter in "aeiou"
        if check == "vowel":
            return ("YES" if is_vowel else "NO", True)
        else:
            return ("YES" if not is_vowel and letter.isalpha() else "NO", True)

    # -- Is X an integer/whole number? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:an?\s+)?(?:integer|whole\s+number)", ql)
    if m:
        val = float(m.group(1))
        return ("YES" if val == int(val) else "NO", True)

    # -- Can X be divided evenly by Y? --
    m = re.search(r"can\s+([+-]?\d+(?:\.\d+)?)\s+be\s+divided\s+evenly\s+by\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        a, b = int(float(m.group(1))), int(float(m.group(2)))
        if b == 0:
            return ("NO", True)
        return ("YES" if a % b == 0 else "NO", True)

    # -- Is X a power of Y? --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:a\s+)?power\s+of\s+([+-]?\d+(?:\.\d+)?)", ql)
    if m:
        val, base = int(float(m.group(1))), int(float(m.group(2)))
        if base <= 1 or val <= 0:
            return ("NO" if val != 1 else "YES", True)
        n = val
        while n > 1:
            if n % base != 0:
                return ("NO", True)
            n //= base
        return ("YES", True)

    # -- Generic "Is X a Y number" pattern (catch-all) --
    m = re.search(r"is\s+([+-]?\d+(?:\.\d+)?)\s+(?:an?\s+)?(even|odd|prime|positive|negative|zero|composite|natural|whole|rational)(?:\s+number)?", ql)
    if m:
        val = float(m.group(1))
        check = m.group(2)
        if check == "even":
            return ("YES" if int(val) % 2 == 0 else "NO", True)
        elif check == "odd":
            return ("YES" if int(val) % 2 != 0 else "NO", True)
        elif check == "prime":
            return ("YES" if _is_prime(int(val)) else "NO", True)
        elif check == "positive":
            return ("YES" if val > 0 else "NO", True)
        elif check == "negative":
            return ("YES" if val < 0 else "NO", True)
        elif check == "zero":
            return ("YES" if val == 0 else "NO", True)
        elif check == "composite":
            v = int(val)
            return ("YES" if v > 1 and not _is_prime(v) else "NO", True)
        elif check == "natural":
            return ("YES" if val > 0 and val == int(val) else "NO", True)
        elif check == "whole":
            return ("YES" if val >= 0 and val == int(val) else "NO", True)
        elif check == "rational":
            return ("YES", True)  # all representable numbers are rational

    return None


# ==============================================================================
# SECTION 7: TEXT EXTRACTION & PROCESSING ENGINE (Level 2)
# ==============================================================================

_DATE_DMY_LONG = re.compile(r"\b(\d{1,2})\s+(" + _MONTH_NAMES_RE + r")[\s,]+(\d{4})\b", re.IGNORECASE)
_DATE_MDY_LONG = re.compile(r"\b(" + _MONTH_NAMES_RE + r")\s+(\d{1,2})[\s,]+(\d{4})\b", re.IGNORECASE)
_DATE_NUMERIC_DMY = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
_DATE_ISO = re.compile(r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}")
_NUMBER_IN_TEXT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _extract_dates(text):
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


def _parse_date_to_obj(text):
    m = _DATE_DMY_LONG.search(text)
    if m:
        month = _MONTH_MAP.get(m.group(2).lower())
        if month:
            try:
                return datetime.date(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                pass
    m = _DATE_MDY_LONG.search(text)
    if m:
        month = _MONTH_MAP.get(m.group(1).lower())
        if month:
            try:
                return datetime.date(int(m.group(3)), month, int(m.group(2)))
            except ValueError:
                pass
    m = _DATE_ISO.search(text)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = _DATE_NUMERIC_DMY.search(text)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        try:
            return datetime.date(y, mo, d)
        except ValueError:
            try:
                return datetime.date(y, d, mo)
            except ValueError:
                pass
    return None


def try_text_extraction(query):
    q = query.strip()
    ql = q.lower()

    if re.match(r"extract\s+(?:the\s+)?date", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            dates = _extract_dates(text)
            if dates:
                return dates[0], True

    if re.match(r"extract\s+(?:the\s+)?(?:email|e-mail)", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            emails = _EMAIL_RE.findall(text)
            if emails:
                return emails[0], True

    if re.match(r"extract\s+(?:the\s+)?(?:url|link|website)", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            urls = _URL_RE.findall(text)
            if urls:
                return urls[0], True

    if re.match(r"extract\s+(?:the\s+)?(?:phone|telephone|contact)\s*(?:number)?", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            phones = _PHONE_RE.findall(text)
            if phones:
                return phones[0].strip(), True

    if re.match(r"extract\s+(?:the\s+)?(?:all\s+)?numbers?", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            nums = _NUMBER_IN_TEXT_RE.findall(text)
            if nums:
                return (", ".join(nums) if "all" in ql else nums[0]), True

    if re.match(r"extract\s+(?:the\s+)?name", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            m = re.search(r"(?:my\s+name\s+is|i\s+am|name\s+is|called)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", text)
            if m:
                return m.group(1), True
            proper = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            common = {"the", "a", "an", "in", "on", "at", "is", "it", "my", "i", "he", "she"}
            filtered = [n for n in proper if n.lower() not in common]
            if filtered:
                return filtered[0], True

    if re.match(r"extract\s+(?:the\s+)?(?:all\s+)?words?", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            m_starts = re.search(r"starting\s+with\s+['\"]?(\w)['\"]?", ql)
            m_contains = re.search(r"containing\s+['\"]?(\w+)['\"]?", ql)
            words = text.split()
            if m_starts:
                words = [w for w in words if w.lower().startswith(m_starts.group(1).lower())]
            elif m_contains:
                words = [w for w in words if m_contains.group(1).lower() in w.lower()]
            return ", ".join(words), True

    # Generic extract X from "text"
    m = re.match(r"extract\s+(.+?)\s+from\s*[:\s]+", ql)
    if m:
        what = m.group(1).strip()
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            if "date" in what:
                dates = _extract_dates(text)
                if dates:
                    return dates[0], True
            if "email" in what or "e-mail" in what:
                emails = _EMAIL_RE.findall(text)
                if emails:
                    return emails[0], True
            if "number" in what:
                nums = _NUMBER_IN_TEXT_RE.findall(text)
                if nums:
                    return nums[0], True
            if "url" in what or "link" in what:
                urls = _URL_RE.findall(text)
                if urls:
                    return urls[0], True

    return None


def try_string_operation(query):
    q = query.strip()
    ql = q.lower()

    # Reverse
    m = re.match(r"reverse\s+(?:the\s+)?(?:string\s+)?[:\s]*(.+)$", q, re.IGNORECASE)
    if m:
        text = m.group(1).strip().rstrip(".")
        if len(text) >= 2 and text[0] in ('"', "'") and text[-1] in ('"', "'"):
            text = text[1:-1]
        return text[::-1], True

    # Uppercase
    if re.match(r"(?:convert\s+)?(?:to\s+)?(?:upper\s*case|uppercase)", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            return text.upper(), True
    m = re.match(r"(?:convert|change|transform)\s+(.+?)\s+to\s+(?:upper\s*case|uppercase)", q, re.IGNORECASE)
    if m:
        text = m.group(1).strip().strip("'\"")
        return text.upper(), True

    # Lowercase
    if re.match(r"(?:convert\s+)?(?:to\s+)?(?:lower\s*case|lowercase)", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            return text.lower(), True
    m = re.match(r"(?:convert|change|transform)\s+(.+?)\s+to\s+(?:lower\s*case|lowercase)", q, re.IGNORECASE)
    if m:
        text = m.group(1).strip().strip("'\"")
        return text.lower(), True

    # Capitalize / title case
    if re.match(r"(?:capitalize|title\s*case)", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            return text.title(), True

    # Count words
    if re.search(r"count\s+(?:the\s+)?(?:number\s+of\s+)?words?\s+(?:in|of)", ql) or \
       re.search(r"how\s+many\s+words?\s+(?:in|are\s+in|does)", ql) or \
       re.search(r"word\s+count\s+(?:of|in|for)", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            return str(len(text.split())), True

    # Count characters
    if re.search(r"count\s+(?:the\s+)?(?:number\s+of\s+)?(?:characters?|chars?|letters?)\s+(?:in|of)", ql) or \
       re.search(r"how\s+many\s+(?:characters?|chars?|letters?)\s+(?:in|are\s+in)", ql) or \
       re.search(r"(?:character|char|letter)\s+count\s+(?:of|in|for)", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            return str(len(text)), True

    # Length
    if re.search(r"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?length\s+of", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            return str(len(text)), True

    # Remove vowels
    if re.search(r"remove\s+(?:all\s+)?vowels?\s+(?:from|in)", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            return re.sub(r"[aeiouAEIOU]", "", text), True

    # Remove spaces
    if re.search(r"remove\s+(?:all\s+)?spaces?\s+(?:from|in)", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            return text.replace(" ", ""), True

    # Remove duplicate chars
    if re.search(r"remove\s+(?:duplicate|repeated)\s+(?:characters?|chars?|letters?)", ql):
        text = _extract_quoted_text(q) or _extract_after_colon(q)
        if text:
            seen, result = set(), []
            for ch in text:
                if ch not in seen:
                    seen.add(ch)
                    result.append(ch)
            return "".join(result), True

    # Replace X with Y in text
    m = re.search(r"replace\s+['\"](.+?)['\"]\s+with\s+['\"](.+?)['\"]\s+in\s+['\"](.+?)['\"]", q, re.IGNORECASE)
    if m:
        return m.group(3).replace(m.group(1), m.group(2)), True
    m = re.search(r"replace\s+['\"]?(.+?)['\"]?\s+with\s+['\"]?(.+?)['\"]?\s+in\s*:\s*['\"](.+?)['\"]", q, re.IGNORECASE)
    if m:
        return m.group(3).replace(m.group(1), m.group(2)), True

    # Concatenate
    if re.search(r"concatenate|concat|join|combine\s+(?:the\s+)?strings?", ql):
        strings = re.findall(r'["\'](.+?)["\']', q)
        if len(strings) >= 2:
            return "".join(strings), True

    # Trim/strip
    if re.search(r"(?:trim|strip)\s+(?:whitespace\s+)?(?:from\s+)?", ql):
        text = _extract_quoted_text(q)
        if text:
            return text.strip(), True

    # First/last N characters
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

    # Substring/slice
    m = re.search(r"substring\s+(?:of\s+)?.*?from\s+(?:index\s+)?(\d+)\s+to\s+(?:index\s+)?(\d+)", ql)
    if m:
        text = _extract_quoted_text(q)
        if text:
            return text[int(m.group(1)):int(m.group(2))], True

    # Repeat
    m = re.search(r"repeat\s+['\"](.+?)['\"]\s+(\d+)\s+times?", q, re.IGNORECASE)
    if m:
        return m.group(1) * int(m.group(2)), True

    # Count occurrences
    m = re.search(r"(?:count|how\s+many)\s+(?:times?\s+)?(?:does\s+)?['\"](.+?)['\"]\s+(?:appear|occur)\s+in\s+['\"](.+?)['\"]", q, re.IGNORECASE)
    if m:
        return str(m.group(2).count(m.group(1))), True
    m = re.search(r"count\s+(?:occurrences?\s+of\s+)?['\"](.+?)['\"]\s+in\s+['\"](.+?)['\"]", q, re.IGNORECASE)
    if m:
        return str(m.group(2).count(m.group(1))), True

    # Split
    m = re.search(r"split\s+['\"](.+?)['\"]\s+(?:by|on|using|with)\s+['\"](.+?)['\"]", q, re.IGNORECASE)
    if m:
        return ", ".join(m.group(1).split(m.group(2))), True

    return None


def try_list_operation(query):
    q = query.strip()
    ql = q.lower()

    def _extract_numbers(text):
        m = re.search(r"[:\s]\s*([\d+\-.,\s]+)$", text)
        if m:
            src = m.group(1)
        else:
            m = re.search(r"\[([^\]]+)\]", text)
            src = m.group(1) if m else text
        return [float(n) for n in re.findall(r"[+-]?\d+(?:\.\d+)?", src)]

    if re.search(r"sort\s+(?:the\s+)?(?:following\s+)?(?:numbers?|values?|list|array|elements?)", ql) or \
       re.search(r"(?:arrange|order)\s+(?:the\s+)?(?:following\s+)?(?:numbers?|values?)", ql):
        nums = _extract_numbers(q)
        if nums:
            if re.search(r"descending|reverse", ql):
                nums.sort(reverse=True)
            else:
                nums.sort()
            return ", ".join(_format_number(n) for n in nums), True

    if re.search(r"sort\s+.+\s+(?:in\s+)?(?:descending|reverse)", ql):
        nums = _extract_numbers(q)
        if nums:
            nums.sort(reverse=True)
            return ", ".join(_format_number(n) for n in nums), True

    if re.search(r"(?:find\s+)?(?:the\s+)?(?:max(?:imum)?|largest|biggest|greatest)\s+(?:of|in|from|value|number)", ql):
        nums = _extract_numbers(q)
        if nums:
            return _format_number(max(nums)), True

    if re.search(r"(?:find\s+)?(?:the\s+)?(?:min(?:imum)?|smallest|least)\s+(?:of|in|from|value|number)", ql):
        nums = _extract_numbers(q)
        if nums:
            return _format_number(min(nums)), True

    if re.search(r"(?:find\s+)?(?:the\s+)?(?:sum|total)\s+(?:of|:)", ql):
        nums = _extract_numbers(q)
        if nums and len(nums) > 2:
            return _format_number(sum(nums)), True

    if re.search(r"(?:find\s+)?(?:the\s+)?(?:average|mean|avg)\s+(?:of|:)", ql):
        nums = _extract_numbers(q)
        if nums:
            return _format_number(sum(nums) / len(nums)), True

    if re.search(r"(?:find\s+)?(?:the\s+)?median\s+(?:of|:)", ql):
        nums = _extract_numbers(q)
        if nums:
            s = sorted(nums)
            n = len(s)
            return _format_number(s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2), True

    if re.search(r"(?:count|how\s+many)\s+(?:items?|elements?|numbers?|values?)\s+(?:in|are\s+in)", ql):
        m = re.search(r"\[([^\]]+)\]", q)
        if m:
            items = [x.strip() for x in m.group(1).split(",") if x.strip()]
            return str(len(items)), True
        nums = _extract_numbers(q)
        if nums:
            return str(len(nums)), True

    if re.search(r"remove\s+(?:duplicate|repeated)\s+(?:numbers?|values?|elements?|items?)", ql) or \
       re.search(r"unique\s+(?:numbers?|values?|elements?|items?)", ql):
        m = re.search(r"\[([^\]]+)\]", q)
        if m:
            items = [x.strip() for x in m.group(1).split(",") if x.strip()]
            seen, unique = set(), []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
            return ", ".join(unique), True
        nums = _extract_numbers(q)
        if nums:
            seen, unique = set(), []
            for n in nums:
                if n not in seen:
                    seen.add(n)
                    unique.append(n)
            return ", ".join(_format_number(n) for n in unique), True

    if re.search(r"(?:find\s+)?(?:the\s+)?(?:common|shared|intersection)\s+(?:elements?|numbers?|values?)", ql):
        brackets = re.findall(r"\[([^\]]+)\]", q)
        if len(brackets) >= 2:
            set1 = set(x.strip() for x in brackets[0].split(","))
            set2 = set(x.strip() for x in brackets[1].split(","))
            return ", ".join(sorted(set1 & set2)), True

    if re.search(r"reverse\s+(?:the\s+)?(?:list|array|order)", ql):
        m = re.search(r"\[([^\]]+)\]", q)
        if m:
            items = [x.strip() for x in m.group(1).split(",") if x.strip()]
            return ", ".join(reversed(items)), True

    return None


def try_number_base_conversion(query):
    q = query.strip()
    ql = q.lower()

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

    m = re.search(r"(?:convert\s+)?(\d+)\s+(?:from\s+decimal\s+)?to\s+binary", ql)
    if m:
        return bin(int(m.group(1)))[2:], True

    m = re.search(r"(?:convert\s+)?(?:hex(?:adecimal)?\s+)?([0-9a-fA-F]+)\s+(?:from\s+hex(?:adecimal)?\s+)?to\s+decimal", ql)
    if m:
        try:
            return str(int(m.group(1), 16)), True
        except ValueError:
            pass

    m = re.search(r"(?:convert\s+)?(\d+)\s+(?:from\s+decimal\s+)?to\s+hex(?:adecimal)?", ql)
    if m:
        return hex(int(m.group(1)))[2:].upper(), True

    m = re.search(r"(?:convert\s+)?(?:octal\s+)?([0-7]+)\s+(?:from\s+octal\s+)?to\s+decimal", ql)
    if m:
        try:
            return str(int(m.group(1), 8)), True
        except ValueError:
            pass

    m = re.search(r"(?:convert\s+)?(\d+)\s+(?:from\s+decimal\s+)?to\s+octal", ql)
    if m:
        return oct(int(m.group(1)))[2:], True

    m = re.search(r"(?:convert\s+)?(?:roman\s+(?:numeral\s+)?)?([IVXLCDM]+)\s+to\s+(?:decimal|number|arabic)", q)
    if m:
        result = _roman_to_int(m.group(1))
        if result > 0:
            return str(result), True

    m = re.search(r"(?:convert\s+)?(\d+)\s+to\s+roman(?:\s+numerals?)?", ql)
    if m:
        result = _int_to_roman(int(m.group(1)))
        if result:
            return result, True

    return None


def _roman_to_int(s):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        val = vals.get(ch, 0)
        if val == 0:
            return 0
        total += -val if val < prev else val
        prev = val
    return total


def _int_to_roman(num):
    if num <= 0 or num > 3999:
        return ""
    result = []
    for val, sym in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while num >= val:
            result.append(sym)
            num -= val
    return "".join(result)


def try_date_operation(query):
    q = query.strip()
    ql = q.lower()

    if re.search(r"(?:what|which)\s+day\s+(?:of\s+the\s+week|is|was)", ql):
        dt = _parse_date_to_obj(q)
        if dt:
            return calendar.day_name[dt.weekday()], True

    if re.search(r"how\s+many\s+days?\s+(?:between|from|until|till)", ql):
        dates = _extract_dates(q)
        if len(dates) >= 2:
            d1, d2 = _parse_date_to_obj(dates[0]), _parse_date_to_obj(dates[1])
            if d1 and d2:
                return str(abs((d2 - d1).days)), True

    m = re.search(r"how\s+many\s+days?\s+(?:in|are\s+in)\s+(" + _MONTH_NAMES_RE + r")(?:\s+(\d{4}))?", ql)
    if m:
        month = _MONTH_MAP.get(m.group(1).lower())
        year = int(m.group(2)) if m.group(2) else 2024
        if month:
            return str(calendar.monthrange(year, month)[1]), True

    return None


# ==============================================================================
# SECTION 8: WEB CONTEXT & FALLBACKS
# ==============================================================================

@lru_cache(maxsize=256)
def _http_get_text(url, timeout=_WEB_TIMEOUT):
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return ""
    req = urlrequest.Request(url, headers={"User-Agent": "andromeda-eval-agent/4.0"})
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data = resp.read(_MAX_ASSET_BYTES)
            ct = resp.headers.get("Content-Type", "").lower()
    except (urlerror.URLError, TimeoutError, ValueError, OSError):
        return ""
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    text = _strip_html(text) if ("html" in ct or "<html" in text.lower()) else _collapse_whitespace(text)
    return text[:10000]


def _assets_context(assets):
    if not assets:
        return ""
    snippets = []
    for url in assets[:5]:
        raw = str(url)
        snippet = _http_get_text(raw, timeout=_WEB_TIMEOUT)
        if not snippet and raw and not raw.startswith(("http://", "https://")):
            snippet = _collapse_whitespace(raw)[:2000]
        if snippet:
            snippets.append(snippet)
    return "\n".join(snippets)[:_MAX_CONTEXT_CHARS]


def _extractive_answer(query, context):
    if not context:
        return ""
    query_words = {w for w in _WORD_RE.findall(query.lower()) if len(w) > 2}
    query_numbers = set(re.findall(r"[+-]?\d+(?:\.\d+)?", query))
    if not query_words and not query_numbers:
        return ""
    candidates = re.split(r"(?<=[.!?])\s+", context)
    best, best_score = "", -1
    for s in candidates[:250]:
        s = _collapse_whitespace(s)
        if len(s) < 8:
            continue
        words = set(_WORD_RE.findall(s.lower()))
        nums = set(re.findall(r"[+-]?\d+(?:\.\d+)?", s))
        score = len(query_words & words) * 2 + len(query_numbers & nums) * 3
        if score > best_score:
            best_score, best = score, s
    return best if best_score > 0 else ""


# ==============================================================================
# SECTION 9: WIKIPEDIA & DUCKDUCKGO
# ==============================================================================

@lru_cache(maxsize=128)
def _wikipedia_summary(query):
    q = _collapse_whitespace(query)
    if not q:
        return ""
    try:
        search_req = urlrequest.Request(
            f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote(q)}&limit=1&namespace=0&format=json",
            headers={"User-Agent": "andromeda-eval-agent/4.0"},
        )
        with urlrequest.urlopen(search_req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        title = data[1][0] if isinstance(data[1], list) and data[1] else ""
    except Exception:
        return ""
    if not title:
        return ""
    try:
        req = urlrequest.Request(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
            headers={"User-Agent": "andromeda-eval-agent/4.0"},
        )
        with urlrequest.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        extract = data.get("extract", "")
        return _collapse_whitespace(extract) if isinstance(extract, str) else ""
    except Exception:
        return ""


@lru_cache(maxsize=128)
def _duckduckgo_answer(query):
    q = _collapse_whitespace(query)
    if not q:
        return ""
    try:
        req = urlrequest.Request(
            f"https://api.duckduckgo.com/?q={quote(q)}&format=json&no_html=1&skip_disambig=1",
            headers={"User-Agent": "andromeda-eval-agent/4.0"},
        )
        with urlrequest.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception:
        return ""
    for key in ["AbstractText", "Answer", "Definition"]:
        val = data.get(key, "")
        if isinstance(val, str) and len(val.strip()) > 5:
            return _collapse_whitespace(val)
    return ""


# ==============================================================================
# SECTION 10: GEMINI LLM ENGINE
# ==============================================================================

_GEMINI_SYSTEM_PROMPT = """\
You are a strict, concise answer engine. Your responses are scored by cosine similarity against expected answers.

ABSOLUTE RULES:
1. Output EXACTLY the answer - nothing more.
2. NO markdown, NO bullet points, NO numbered lists, NO bold/italic.
3. NO introductory phrases like "Sure!", "Of course!", etc.
4. NO explanations, NO reasoning steps, NO disclaimers.
5. Do NOT repeat the question.
6. If context/assets are provided, base your answer ONLY on them.

RESPONSE FORMAT BY QUERY TYPE:

A) ARITHMETIC:
- Addition -> "The sum is X."
- Subtraction -> "The difference is X."
- Multiplication -> "The product is X."
- Division -> "The quotient is X."

B) YES/NO QUESTIONS (Is X even? Is X prime? Is it a palindrome? etc.):
- Return ONLY "YES" or "NO" in uppercase. Nothing else.

C) EXTRACTION (extract X from text):
- Return ONLY the extracted value, no sentence wrapper, no period.

D) STRING/LIST OPERATIONS:
- Return ONLY the result, no sentence wrapper, no period.

E) FACTUAL QUESTIONS:
- Answer in ONE concise sentence ending with a period.

EXAMPLES:
Q: What is 10 + 15?
A: The sum is 25.

Q: Is 9 an odd number?
A: YES

Q: Is 4 an even number?
A: YES

Q: Is 7 a prime number?
A: YES

Q: Is 10 divisible by 3?
A: NO

Q: Is 15 greater than 10?
A: YES

Q: Is "racecar" a palindrome?
A: YES

Q: Is 2024 a leap year?
A: YES

Q: Is 16 a perfect square?
A: YES

Q: Extract date from: "Meeting on 12 March 2024".
A: 12 March 2024

Q: Reverse the string "hello".
A: olleh

Q: Sort the numbers: 5, 3, 1.
A: 1, 3, 5

Q: What is the capital of France?
A: The capital of France is Paris.

Q: Convert 10 to binary.
A: 1010

Q: Count words in "hello world".
A: 2

Q: Find the maximum in: 10, 20, 5.
A: 20
"""


def _call_gemini(query, context):
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
        "contents": [{"role": "user", "parts": [{"text": _GEMINI_SYSTEM_PROMPT + "\n\n" + user_prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 0.05, "topK": 1, "maxOutputTokens": 150, "candidateCount": 1},
    }
    for model in model_candidates:
        for attempt in range(_GEMINI_MAX_RETRIES):
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                req = urlrequest.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
                with urlrequest.urlopen(req, timeout=_GEMINI_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if isinstance(text, str) and text.strip():
                    return text.strip()
            except urlerror.HTTPError as e:
                status = getattr(e, "code", 0)
                print(f"[EVAL-LOG] Gemini HTTP {status} on {model} (attempt {attempt+1}): {repr(e)}", flush=True)
                if status == 429:
                    time.sleep(min(_GEMINI_BACKOFF_BASE * (2 ** attempt), 10.0))
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


# ==============================================================================
# SECTION 11: OUTPUT SANITIZATION
# ==============================================================================

def sanitize_output(text):
    """Clean sentence-mode output. Single sentence ending with period."""
    if not isinstance(text, str) or not text.strip():
        return "I cannot determine the answer."
    cleaned = _strip_markdown(text)
    cleaned = " ".join(cleaned.replace("\n", " ").split()).strip()
    if not cleaned:
        return "I cannot determine the answer."
    for prefix in ["Sure!", "Sure,", "Sure.", "Of course!", "Of course,", "Of course.", "Here's the answer:", "Here is the answer:", "The answer is:", "Answer:", "A:", "Response:"]:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    if len(cleaned) > 2 and cleaned[0] in ('"', "'") and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        return "I cannot determine the answer."
    arith_match = re.match(
        r"^(The\s+(?:sum|difference|product|quotient|result|remainder|square\s+root|factorial|"
        r"GCD|LCM|logarithm|natural\s+logarithm|absolute\s+value)"
        r"(?:\s+(?:of|when|base))?"
        r"(?:[^.!?]|\.(?=\d))*?(?:is\s+(?:[^.!?]|\.(?=\d))+))",
        cleaned, re.IGNORECASE,
    )
    if arith_match:
        sentence = arith_match.group(1).strip()
        return sentence[0].upper() + sentence[1:].rstrip(".!? ") + "."
    m = re.search(r"[!?]|\.(?!\d)", cleaned)
    if m:
        cleaned = cleaned[:m.start() + 1]
    cleaned = cleaned.strip()
    if not cleaned:
        return "I cannot determine the answer."
    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    return cleaned.rstrip(".!? ") + "."


def sanitize_raw_output(text):
    """Clean raw-mode output. Minimal processing, no sentence formatting."""
    if not isinstance(text, str) or not text.strip():
        return ""
    cleaned = _strip_markdown(text)
    cleaned = " ".join(cleaned.replace("\n", " ").split()).strip()
    for prefix in ["Sure!", "Sure,", "Sure.", "Of course!", "Of course,", "Of course.", "Here's the answer:", "Here is the answer:", "The answer is:", "Answer:", "A:", "Response:"]:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    if len(cleaned) > 2 and cleaned[0] in ('"', "'") and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
    # Remove trailing period if not a sentence
    if cleaned.endswith(".") and not re.search(r"[a-zA-Z]{3,}\.$", cleaned):
        cleaned = cleaned[:-1].strip()
    return cleaned


# ==============================================================================
# SECTION 12: MAIN ANSWER PIPELINE
# ==============================================================================

def _is_extraction_query(query):
    ql = query.lower().strip()
    patterns = [
        r"^extract\s+", r"^reverse\s+", r"^convert\s+.+\s+to\s+(?:upper|lower|binary|hex|octal|decimal|roman)",
        r"^sort\s+", r"^count\s+", r"^find\s+(?:the\s+)?(?:max|min|largest|smallest|common)",
        r"^remove\s+", r"^replace\s+", r"^concatenate\s+", r"^split\s+",
        r"^is\s+", r"^does\s+", r"^are\s+", r"^can\s+",
        r"^(?:how\s+many|what(?:'s|\s+is)\s+(?:the\s+)?length)",
        r"^(?:first|last)\s+\d+\s+(?:char|letter)",
        r"^repeat\s+", r"^(?:what|which)\s+day\s+", r"^how\s+many\s+days?\s+",
        r"^unique\s+", r"word\s+count", r"character\s+count",
    ]
    return any(re.search(p, ql) for p in patterns)


def generate_answer(query, assets):
    """
    Main pipeline. Returns (answer, is_raw).
    is_raw=True -> no sentence formatting.

    Order:
    1. Boolean yes/no (raw) -- Level 3
    2. Text extraction (raw) -- Level 2
    3. String operations (raw)
    4. List operations (raw)
    5. Number base conversion (raw)
    6. Date operations (raw)
    7. Arithmetic (sentence) -- Level 1
    8. Unit conversion (sentence)
    9. Gemini LLM
    10. Fallbacks
    """
    context = _assets_context(assets)

    # 1. Boolean yes/no
    result = try_boolean_question(query)
    if result is not None:
        return result

    # 2. Text extraction
    result = try_text_extraction(query)
    if result is not None:
        return result

    # 3. String operations
    result = try_string_operation(query)
    if result is not None:
        return result

    # 4. List operations
    result = try_list_operation(query)
    if result is not None:
        return result

    # 5. Number base conversion
    result = try_number_base_conversion(query)
    if result is not None:
        return result

    # 6. Date operations
    result = try_date_operation(query)
    if result is not None:
        return result

    # 7. Arithmetic
    arith = parse_arithmetic_query(query)
    if arith is not None:
        return arith, False

    # 8. Conversions
    conv = try_conversion(query)
    if conv is not None:
        return conv, False

    # 9. Gemini
    is_raw = _is_extraction_query(query)
    gemini = _call_gemini(query, context)
    if gemini:
        return gemini, is_raw

    # 10. Fallbacks
    ddg = _duckduckgo_answer(query)
    if ddg:
        return ddg, False
    wiki = _wikipedia_summary(query)
    if wiki:
        return wiki, False
    ext = _extractive_answer(query, context)
    if ext:
        return ext, False

    return "I cannot determine the answer.", False


# ==============================================================================
# SECTION 13: FLASK ROUTES
# ==============================================================================

def validate_payload(payload):
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


def build_output_payload(text):
    return {"output": text, "answer": text, "result": text, "response": text}


def extract_payload():
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
                parsed = json.loads(assets_raw)
                if isinstance(parsed, list):
                    assets = parsed
            except json.JSONDecodeError:
                assets = [assets_raw]
        if query is not None:
            return {"query": query, "assets": assets}
    query_arg = request.args.get("query")
    if query_arg is not None:
        return {"query": query_arg, "assets": request.args.getlist("assets")}
    return payload


@app.route("/v1/answer", methods=["POST", "GET"])
def answer():
    start = time.time()
    payload = extract_payload()
    is_valid, err, query, assets = validate_payload(payload)
    if not is_valid:
        print(f"[EVAL-LOG] Invalid Payload: {payload}", flush=True)
        return jsonify({"error": err}), 400
    print(f"\n[EVAL-LOG] Query: {query}", flush=True)
    print(f"[EVAL-LOG] Assets: {assets}", flush=True)
    raw_output, is_raw = generate_answer(query, assets)
    final = sanitize_raw_output(raw_output) if is_raw else sanitize_output(raw_output)
    if not final:
        final = "I cannot determine the answer."
    ms = int((time.time() - start) * 1000)
    print(f"[EVAL-LOG] Output: {final} (raw={is_raw}, {ms}ms)\n", flush=True)
    return jsonify(build_output_payload(final)), 200


@app.route("/health", methods=["GET"])
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "4.0"}), 200


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
