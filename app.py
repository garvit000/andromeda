from __future__ import annotations

import re
import json
import os
from functools import lru_cache
import time
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import quote
from typing import Any, Optional, Tuple

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

_BROAD_NL_PATTERNS = [
    ("add", re.compile(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:plus|added to)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))", re.IGNORECASE)),
    ("sub", re.compile(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:minus|subtracted from)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))", re.IGNORECASE)),
    ("mul", re.compile(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:times|multiplied by)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))", re.IGNORECASE)),
    ("div", re.compile(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:divided by)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))", re.IGNORECASE)),
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _format_number(value: float) -> str:
    if abs(value - int(value)) < 1e-9:
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

    for operation, pattern in _BROAD_NL_PATTERNS:
        m = pattern.search(query)
        if m:
            if operation == "sub" and "subtracted from" in m.group(0).lower():
                return operation, float(m.group(2)), float(m.group(1))
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


def _collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _strip_html(text: str) -> str:
    return _collapse_whitespace(_HTML_TAG_RE.sub(" ", text))


@lru_cache(maxsize=128)
def _http_get_text(url: str, timeout: float = 2.5) -> str:
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return ""
    req = urlrequest.Request(url, headers={"User-Agent": "andromeda-eval-agent/1.0"})
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data = resp.read(24000)
            content_type = resp.headers.get("Content-Type", "").lower()
    except (urlerror.URLError, TimeoutError, ValueError):
        return ""

    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

    if "html" in content_type or "<html" in text.lower():
        text = _strip_html(text)
    else:
        text = _collapse_whitespace(text)

    return text[:8000]


def _assets_context(assets: list) -> str:
    if not assets:
        return ""
    snippets: list[str] = []
    for url in assets[:5]:
        raw = str(url)
        snippet = _http_get_text(raw, timeout=2.0)
        if not snippet and raw and not raw.startswith(("http://", "https://")):
            snippet = _collapse_whitespace(raw)[:1200]
        if snippet:
            snippets.append(snippet)
    return "\n".join(snippets)[:12000]


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
    for sentence in candidates[:180]:
        s = _collapse_whitespace(sentence)
        if len(s) < 10:
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


def _wikipedia_summary(query: str) -> str:
    q = _collapse_whitespace(query)
    if not q:
        return ""

    search_endpoint = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=opensearch&search={quote(q)}&limit=1&namespace=0&format=json"
    )
    search_req = urlrequest.Request(search_endpoint, headers={"User-Agent": "andromeda-eval-agent/1.0"})

    try:
        with urlrequest.urlopen(search_req, timeout=2.2) as resp:
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
    req = urlrequest.Request(endpoint, headers={"User-Agent": "andromeda-eval-agent/1.0"})
    try:
        with urlrequest.urlopen(req, timeout=2.2) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return ""

    extract = data.get("extract")
    if isinstance(extract, str):
        return _collapse_whitespace(extract)
    return ""


def llm_style_fallback(query: str, assets: list) -> str:
    context = _assets_context(assets)

    # Fast-path local logic to save Gemini rate limits for complex questions
    parsed = parse_arithmetic_query(query)
    if parsed is not None:
        operation, left, right = parsed
        return solve_arithmetic(operation, left, right)

    api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if api_key:
        model_env = os.getenv("GEMINI_MODEL", "gemini-2.0-flash,gemini-1.5-flash")
        model_candidates = [m.strip() for m in model_env.split(",") if m.strip()]

        prompt = (
            "You are a strict API answering engine. Follow these rules EXACTLY:\n"
            "1. Focus purely on answering the query.\n"
            "2. If it is basic arithmetic:\n"
            "   - Addition -> 'The sum is X.'\n"
            "   - Subtraction -> 'The difference is X.'\n"
            "   - Multiplication -> 'The product is X.'\n"
            "   - Division -> 'The quotient is X.'\n"
            "   (Format X as a plain number).\n"
            "3. Base non-arithmetic answers ONLY on the provided Context if present.\n"
            "4. YOUR OUTPUT MUST BE EXACTLY ONE CONCISE SENTENCE.\n"
            "5. YOUR OUTPUT MUST END WITH A PERIOD '.'\n"
            "6. Provide NO explanations, NO introductory text, and NO reasoning steps.\n\n"
            f"Query: {query}\n"
            f"Context: {context if context else 'None'}"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "topP": 0.1,
                "maxOutputTokens": 100,
            },
        }

        for model in model_candidates:
            for attempt in range(6):
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
                    with urlrequest.urlopen(req, timeout=12.0) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    candidate_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    if isinstance(candidate_text, str) and candidate_text.strip():
                        return candidate_text.strip()
                except urlerror.HTTPError as e:
                    print(f"[EVAL-LOG] Gemini error on {model} (Attempt {attempt}): {repr(e)}", flush=True)
                    if e.code == 429:
                        time.sleep(2.5)
                        continue
                    break # non-429 error, move to next model
                except Exception as e:
                    print(f"[EVAL-LOG] Gemini error on {model} (Attempt {attempt}): {repr(e)}", flush=True)
                    break
    else:
        print("[EVAL-LOG] No Gemini API key found!", flush=True)

    extractive = _extractive_answer(query, context)
    if extractive:
        return extractive

    wiki = _wikipedia_summary(query)
    if wiki:
        return wiki

    return "I cannot determine the answer."


def sanitize_output(text: str) -> str:
    if not isinstance(text, str):
        return "I cannot determine the answer."

    cleaned = " ".join(text.replace("\n", " ").split()).strip()
    if not cleaned:
        return "I cannot determine the answer."

    # Preserve exact arithmetic phrasing and decimals while forcing final period.
    arith_match = re.match(
        r"^(The (sum|difference|product|quotient) is [+-]?(?:\d+(?:\.\d+)?|\.\d+|undefined))",
        cleaned,
        re.IGNORECASE,
    )
    if arith_match:
        sentence = arith_match.group(1)
        sentence = sentence[0].upper() + sentence[1:]
        return sentence.rstrip(".!? ") + "."

    # General single-sentence normalization for Gemini fallback.
    split_match = re.search(r"[!?]|\.(?!\d)", cleaned)
    if split_match:
        cleaned = cleaned[: split_match.start() + 1]

    return cleaned.rstrip(".!? ") + "."


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
    if _env_flag("RETURN_ALL_KEYS", False):
        return {
            "output": text,
            "result": text,
            "answer": text,
            "response": text,
        }
    return {"output": text}


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
    payload = extract_payload()

    is_valid, err, query, assets = validate_payload(payload)

    if not is_valid:
        print(f"[EVAL-LOG] Invalid Payload: {payload}", flush=True)
        return jsonify({"error": err}), 400

    print(f"\n[EVAL-LOG] Query: {query}", flush=True)
    print(f"[EVAL-LOG] Assets: {assets}", flush=True)

    raw_output = llm_style_fallback(query, assets)

    final_output = sanitize_output(raw_output)
    
    print(f"[EVAL-LOG] Output: {final_output}\n", flush=True)
    
    return jsonify(build_output_payload(final_output)), 200


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
