"""
Apollo.io API client — server-side calls using Dan's own Apollo API key.

This lets the Sammy CRM API act as a proxy/orchestrator: the ChatGPT custom
GPT calls these Sammy CRM endpoints, which in turn call Apollo's REST API
directly (Apollo has no MCP/connector story for ChatGPT Actions — only plain
HTTPS is usable there), then cross-reference results against the Sammy CRM
Supabase tables to skip duplicates and create new records.

Auth: Apollo's current API requires the key in an `x-api-key` header (the
old `api_key` query-param method is being deprecated per Apollo's own UI
notice). Supplied via env var APOLLO_API_KEY, populated in production via
the publish_website `credentials` proxy as
CUSTOM_CRED_API_APOLLO_IO_TOKEN.
"""

import json
import os
import subprocess
from typing import Any, Dict, List, Optional

# In production (published site), publish_website's credentials proxy injects
# CUSTOM_CRED_API_APOLLO_IO_URL / _TOKEN. The URL var points at an
# agent-proxy passthrough host that injects the real Apollo auth server-side
# -- NOT the literal api.apollo.io host -- so requests MUST be sent to that
# URL, same pattern as the working Supabase client in main.py. In the dev
# sandbox (no credential env vars set), fall back to the real Apollo host,
# where the sandbox's own outbound proxy authenticates transparently.
APOLLO_BASE = (
    os.environ.get("CUSTOM_CRED_API_APOLLO_IO_URL")
    or "https://api.apollo.io"
) + "/api/v1"

APOLLO_API_KEY = (
    os.environ.get("CUSTOM_CRED_API_APOLLO_IO_TOKEN")
    or os.environ.get("APOLLO_API_KEY")
)


class ApolloError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _apollo_request(method: str, path: str, json_body: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    # In production (published site), APOLLO_API_KEY arrives via the
    # CUSTOM_CRED_API_APOLLO_IO_TOKEN env var injected by publish_website's
    # credentials proxy, and we must send it explicitly as x-api-key. In the
    # dev sandbox (no credential env var set), outbound HTTPS to api.apollo.io
    # is authenticated transparently by the sandbox's own credential proxy, so
    # no explicit header is needed there either way -- same pattern as the
    # Supabase client in main.py.
    url = f"{APOLLO_BASE}/{path}"
    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
    }
    if APOLLO_API_KEY:
        headers["x-api-key"] = APOLLO_API_KEY

    cmd = ["curl", "-s", "-w", "\n%{http_code}", "--max-time", str(timeout), "-X", method, url]
    for k, v in headers.items():
        cmd.extend(["-H", f"{k}: {v}"])
    if json_body is not None:
        cmd.extend(["--data-binary", json.dumps(json_body)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        raise ApolloError(504, "Apollo request timed out")

    stdout = result.stdout
    if "\n" not in stdout:
        raise ApolloError(502, f"Unexpected curl output: {stdout[:300]}")
    body, status_code_str = stdout.rsplit("\n", 1)
    try:
        status_code = int(status_code_str.strip())
    except ValueError:
        raise ApolloError(502, f"Could not parse curl status code: {stdout[:300]}")

    if status_code == 0:
        raise ApolloError(502, f"Could not reach Apollo: {result.stderr[:300]}")

    parsed: Any = None
    if body.strip():
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body

    if status_code >= 400:
        detail = parsed if isinstance(parsed, str) else json.dumps(parsed)
        raise ApolloError(status_code, detail)

    return parsed


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_people(
    person_titles: Optional[List[str]] = None,
    q_keywords: Optional[str] = None,
    organization_domains: Optional[List[str]] = None,
    organization_num_employees_ranges: Optional[List[str]] = None,
    person_locations: Optional[List[str]] = None,
    industries: Optional[List[str]] = None,
    page: int = 1,
    per_page: int = 10,
) -> Dict[str, Any]:
    """Apollo mixed_people/api_search -- find people by title/company/location/industry.
    Note: the endpoint is api_search, not the older/deprecated search path --
    the current Apollo key is scoped for api_search only and returns
    API_INACCESSIBLE on the old path."""
    body: Dict[str, Any] = {"page": page, "per_page": per_page}
    if person_titles:
        body["person_titles"] = person_titles
    if q_keywords:
        body["q_keywords"] = q_keywords
    if organization_domains:
        body["organization_domains"] = organization_domains
    if organization_num_employees_ranges:
        body["organization_num_employees_ranges"] = organization_num_employees_ranges
    if person_locations:
        body["person_locations"] = person_locations
    if industries:
        body["q_organization_keyword_tags"] = industries
    return _apollo_request("POST", "mixed_people/api_search", json_body=body)


def search_companies(
    q_keywords: Optional[str] = None,
    organization_num_employees_ranges: Optional[List[str]] = None,
    organization_locations: Optional[List[str]] = None,
    industries: Optional[List[str]] = None,
    page: int = 1,
    per_page: int = 10,
) -> Dict[str, Any]:
    """Apollo mixed_companies/search — find companies by size/location/industry/keyword."""
    body: Dict[str, Any] = {"page": page, "per_page": per_page}
    if q_keywords:
        body["q_organization_keyword_tags"] = [q_keywords]
    if organization_num_employees_ranges:
        body["organization_num_employees_ranges"] = organization_num_employees_ranges
    if organization_locations:
        body["organization_locations"] = organization_locations
    if industries:
        body["q_organization_keyword_tags"] = industries
    return _apollo_request("POST", "mixed_companies/search", json_body=body)


# ---------------------------------------------------------------------------
# Enrich
# ---------------------------------------------------------------------------


def enrich_person(
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    name: Optional[str] = None,
    organization_name: Optional[str] = None,
    domain: Optional[str] = None,
    linkedin_url: Optional[str] = None,
    email: Optional[str] = None,
) -> Dict[str, Any]:
    """Apollo people/match — enrich a single person to get email, title, LinkedIn, etc."""
    body: Dict[str, Any] = {}
    if first_name:
        body["first_name"] = first_name
    if last_name:
        body["last_name"] = last_name
    if name:
        body["name"] = name
    if organization_name:
        body["organization_name"] = organization_name
    if domain:
        body["domain"] = domain
    if linkedin_url:
        body["linkedin_url"] = linkedin_url
    if email:
        body["email"] = email
    return _apollo_request("POST", "people/match", json_body=body)


def enrich_company(domain: str) -> Dict[str, Any]:
    """Apollo organizations/enrich — enrich a single company by domain."""
    return _apollo_request("GET", f"organizations/enrich?domain={domain}")
