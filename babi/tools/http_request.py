"""Tool for making arbitrary HTTP requests.

Supports GET, POST, PUT, DELETE, PATCH methods with custom headers and body.
Useful for API calls and web service interactions.
"""

from __future__ import annotations

import json
import logging
import os

import httpx
from langchain_core.tools import tool

from babi.tools.github_url import check_github_url
from babi.utils.helpers import truncate

logger = logging.getLogger(__name__)


@tool
def http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: str | None = None,
) -> str:
    """Make an HTTP request (GET, POST, PUT, DELETE, PATCH) to any URL.

    Use for API calls with custom methods, headers, or request bodies.
    NOTE: Do NOT use this for github.com URLs — use github_api_request instead.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH).
        url: The URL to send the request to.
        headers: Optional request headers as key-value pairs.
        body: Optional request body (for POST/PUT/PATCH).
    """
    logger.debug("[HTTP_REQUEST] %s %s", method, url)

    # Intercept GitHub web URLs
    redirect = check_github_url(url)
    if redirect is not None:
        logger.debug("[HTTP_REQUEST] Redirected by GitHub URL checker")
        return redirect

    # Intercept GraphQL POST requests
    actual_body = body
    if url and "api.github.com/graphql" in url.lower() and method.upper() == "POST" and body:
        actual_body = _fix_graphql_body(body)

    try:
        req_headers = dict(headers) if headers else {}

        # Add auth header for GitHub API if missing
        if url and "api.github.com" in url.lower() and "Authorization" not in req_headers:
            token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
            if token:
                req_headers["Authorization"] = f"Bearer {token}"

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.request(
                method=method.upper() if method else "GET",
                url=url,
                headers=req_headers,
                content=actual_body,
            )

        resp_headers = dict(response.headers)
        return f"Status: {response.status_code}\nHeaders: {resp_headers}\nBody:\n{truncate(response.text, 16000)}"

    except httpx.HTTPError as e:
        return f"Error: {e}"


def _fix_graphql_body(body: str) -> str:
    """Reformat GraphQL query in JSON body to multi-line format."""
    try:
        parsed = json.loads(body)
        query = parsed.get("query")
        if isinstance(query, str):
            parsed["query"] = _format_graphql_query(query)
            return json.dumps(parsed)
    except Exception:
        pass
    return body


def _format_graphql_query(query: str) -> str:
    """Reformat a GraphQL query string to multi-line format."""
    result = []
    depth = 0
    i = 0
    while i < len(query):
        c = query[i]
        if c == "{":
            depth += 1
            result.append(" {\n")
            result.append("  " * depth)
            while i + 1 < len(query) and query[i + 1] == " ":
                i += 1
        elif c == "}":
            depth -= 1
            while result and result[-1].endswith(" "):
                result[-1] = result[-1][:-1]
            result.append("\n")
            if depth >= 0:
                result.append("  " * depth)
            result.append("}")
        else:
            result.append(c)
        i += 1
    return "".join(result).strip()
