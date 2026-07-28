"""
Sammy CRM API — live read/write access to Dan Poore's recruiting/BD database.
Backed by Supabase (Postgres) via its REST API (PostgREST).
Built for use as a ChatGPT Custom GPT Action (OpenAPI schema in openapi.json).

Auth: pass the API key as a query parameter `api_key=...` on every request
(including POST/PATCH). Custom request headers are unreliable through some
hosting proxies, so query-param auth is the primary, guaranteed-to-work method.

Supabase access uses `curl` via subprocess rather than a Python HTTP client.
This sandbox's outbound proxy fails TLS verification for httpx/requests when
talking to Supabase (missing Authority Key Identifier on the intercepted
cert chain) but curl handles it fine — this pattern is kept for production
parity with what was verified during migration.
"""

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Supabase connection — supplied via env vars. In production (published site)
# these arrive as CUSTOM_CRED_<HOST>_URL / CUSTOM_CRED_<HOST>_TOKEN through the
# publish_website `credentials` proxy. Locally they can be set directly.
SUPABASE_URL = (
    os.environ.get("CUSTOM_CRED_VEVQMGPDALIYKZMZKMRZ_SUPABASE_CO_URL")
    or os.environ.get("SUPABASE_URL")
    or "https://vevqmgpdaliykzmzkmrz.supabase.co"
).rstrip("/")
SUPABASE_KEY = (
    os.environ.get("CUSTOM_CRED_VEVQMGPDALIYKZMZKMRZ_SUPABASE_CO_TOKEN")
    or os.environ.get("SUPABASE_KEY")
)
SUPABASE_REST = f"{SUPABASE_URL}/rest/v1"

API_KEY = os.environ.get("SAMMY_API_KEY")  # must be set in the hosting environment

app = FastAPI(
    title="Sammy CRM API",
    description="Live read/write access to Dan Poore's recruiting and business-development CRM.",
    version="2.0.0",
)

ALLOWED_TABLES = {
    "searches": "search_id",
    "candidates": "candidate_id",
    "scores": "score_id",
    "outreach": "outreach_id",
    "buyer_companies": "company_id",
    "buyer_contacts": "contact_id",
    "buyer_signals": "signal_id",
    "buyer_outreach": "outreach_id",
    "mpc_candidates": "mpc_id",
    "mpc_outreach": "mpc_outreach_id",
}

# Columns known to be integer/float typed per table, so query filter values and
# inserted values get coerced correctly for Postgres (everything else is text).
INTEGER_COLUMNS = {
    "scores": {"total_score"},
    "buyer_companies": {"employee_count", "fit_score"},
    "buyer_signals": {"signal_strength"},
}
FLOAT_COLUMNS: Dict[str, set] = {}

_SCHEMA_CACHE: Dict[str, List[str]] = {}
_TEXT_COLUMNS_CACHE: Dict[str, List[str]] = {}


# ---------------------------------------------------------------------------
# curl helper — the only reliable transport to Supabase from this environment
# ---------------------------------------------------------------------------


class SupabaseError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _supabase_request(
    method: str,
    path: str,
    params: Optional[Dict[str, str]] = None,
    json_body: Optional[Any] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Any:
    # No hard requirement on SUPABASE_KEY here: in the dev sandbox, outbound
    # HTTPS to the Supabase host is authenticated transparently by the
    # sandbox's credential proxy even with no key in-process. In production,
    # a missing key means Supabase itself will reject with 401, which we
    # surface as a clear SupabaseError below.
    url = f"{SUPABASE_REST}/{path}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"

    # In production, SUPABASE_KEY is a real key read from env vars and must be
    # sent explicitly. In the dev sandbox, outbound HTTPS to this host is
    # transparently authenticated by the sandbox's credential proxy, so no
    # explicit header is needed there — sending headers is harmless either way
    # since Supabase just uses whichever valid apikey/Authorization it sees.
    headers = {"Content-Type": "application/json"}
    if SUPABASE_KEY:
        headers["apikey"] = SUPABASE_KEY
        headers["Authorization"] = f"Bearer {SUPABASE_KEY}"
    if extra_headers:
        headers.update(extra_headers)

    cmd = ["curl", "-s", "-w", "\n%{http_code}", "--max-time", str(timeout), "-X", method, url]
    for k, v in headers.items():
        cmd.extend(["-H", f"{k}: {v}"])
    if json_body is not None:
        cmd.extend(["--data-binary", json.dumps(json_body)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        raise SupabaseError(504, "Supabase request timed out")

    stdout = result.stdout
    if "\n" not in stdout:
        raise SupabaseError(502, f"Unexpected curl output: {stdout[:300]}")
    body, status_code_str = stdout.rsplit("\n", 1)
    try:
        status_code = int(status_code_str.strip())
    except ValueError:
        raise SupabaseError(502, f"Could not parse curl status code: {stdout[:300]}")

    if status_code == 0:
        raise SupabaseError(502, f"Could not reach Supabase: {result.stderr[:300]}")

    parsed: Any = None
    if body.strip():
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body

    if status_code >= 400:
        detail = parsed if isinstance(parsed, str) else json.dumps(parsed)
        raise SupabaseError(status_code, detail)

    return parsed


# ---------------------------------------------------------------------------
# Schema introspection (cached — table shape doesn't change at runtime)
# ---------------------------------------------------------------------------


def table_columns(table: str) -> List[str]:
    if table in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[table]
    try:
        rows = _supabase_request("GET", table, params={"select": "*", "limit": "20"})
    except SupabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    if isinstance(rows, list) and rows:
        cols = list(rows[0].keys())
        # Infer text-typed columns from sampled rows' Python value types.
        # ilike/like only work on text columns in Postgres — applying them to
        # int/float/bool columns raises "operator does not exist". Sampling up
        # to 20 rows (instead of 1) avoids wrongly excluding a text column
        # just because its value happened to be null in a single sample row.
        text_cols = set()
        seen_non_null = set()
        for r in rows:
            for c, v in r.items():
                if v is None:
                    continue
                seen_non_null.add(c)
                if isinstance(v, str):
                    text_cols.add(c)
        _TEXT_COLUMNS_CACHE[table] = sorted(text_cols)
    else:
        # Table is empty — fall back to a HEAD-style probe via OpenAPI root is
        # overkill here; return empty and let callers pass through unknown-col
        # checks softly. This only affects brand-new empty tables.
        cols = []
        _TEXT_COLUMNS_CACHE[table] = []
    _SCHEMA_CACHE[table] = cols
    return cols


def table_text_columns(table: str) -> List[str]:
    if table not in _TEXT_COLUMNS_CACHE:
        table_columns(table)  # populates both caches
    return _TEXT_COLUMNS_CACHE.get(table, [])


def coerce_value(table: str, col: str, value: Any) -> Any:
    if value is None:
        return None
    if col in INTEGER_COLUMNS.get(table, set()):
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if col in FLOAT_COLUMNS.get(table, set()):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    return value


class RowUpsert(BaseModel):
    row: Dict[str, Any]


def _check_auth(api_key: Optional[str]):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: SAMMY_API_KEY not set")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Pass it as ?api_key=... on every request.")


def _require_table(table: str) -> str:
    if table not in ALLOWED_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown table '{table}'")
    return ALLOWED_TABLES[table]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    configured = bool(SUPABASE_KEY)
    return {
        "status": "ok",
        "backend": "supabase",
        "supabase_url": SUPABASE_URL,
        "supabase_key_configured": configured,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/tables")
def list_tables(api_key: Optional[str] = None):
    _check_auth(api_key)
    return {"tables": list(ALLOWED_TABLES.keys())}


@app.get("/tables/{table}/schema")
def table_schema(table: str, api_key: Optional[str] = None):
    _check_auth(api_key)
    _require_table(table)
    return {"table": table, "columns": table_columns(table)}


@app.get("/tables/{table}/rows")
def get_rows(
    table: str,
    api_key: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    filter_op: str = Query("=", pattern="^(=|!=|LIKE|>|<|>=|<=)$"),
):
    _check_auth(api_key)
    _require_table(table)

    op_map = {"=": "eq", "!=": "neq", "LIKE": "ilike", ">": "gt", "<": "lt", ">=": "gte", "<=": "lte"}
    params: Dict[str, str] = {"select": "*", "limit": str(limit), "offset": str(offset)}

    if filter_column:
        cols = table_columns(table)
        if cols and filter_column not in cols:
            raise HTTPException(status_code=400, detail=f"Unknown column '{filter_column}' on table '{table}'")
        pg_op = op_map[filter_op]
        value = f"*{filter_value}*" if filter_op == "LIKE" else filter_value
        params[filter_column] = f"{pg_op}.{value}"

    try:
        rows = _supabase_request("GET", table, params=params)
    except SupabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    rows = rows or []
    return {"table": table, "count": len(rows), "rows": rows}


@app.post("/tables/{table}/rows")
def create_row(table: str, payload: RowUpsert, api_key: Optional[str] = None):
    _check_auth(api_key)
    pk = _require_table(table)
    cols = table_columns(table)
    row = dict(payload.row)

    if pk not in row or not row[pk]:
        row[pk] = str(uuid.uuid4())

    now_iso = datetime.now(timezone.utc).isoformat()
    if (not cols or "last_updated" in cols) and "last_updated" not in row:
        row["last_updated"] = now_iso
    if (not cols or "first_seen_date" in cols) and "first_seen_date" not in row:
        row["first_seen_date"] = now_iso

    if cols:
        bad_cols = [k for k in row.keys() if k not in cols]
        if bad_cols:
            raise HTTPException(status_code=400, detail=f"Unknown columns for '{table}': {bad_cols}")

    row = {k: coerce_value(table, k, v) for k, v in row.items()}

    try:
        _supabase_request(
            "POST",
            table,
            json_body=row,
            extra_headers={"Prefer": "return=representation"},
        )
    except SupabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    return {"table": table, "created": row}


@app.patch("/tables/{table}/rows/{row_id}")
def update_row(table: str, row_id: str, payload: RowUpsert, api_key: Optional[str] = None):
    _check_auth(api_key)
    pk = _require_table(table)
    cols = table_columns(table)
    row = dict(payload.row)

    if cols and "last_updated" in cols:
        row["last_updated"] = datetime.now(timezone.utc).isoformat()

    if cols:
        bad_cols = [k for k in row.keys() if k not in cols]
        if bad_cols:
            raise HTTPException(status_code=400, detail=f"Unknown columns for '{table}': {bad_cols}")
    if not row:
        raise HTTPException(status_code=400, detail="No fields to update")

    row = {k: coerce_value(table, k, v) for k, v in row.items()}

    try:
        _supabase_request(
            "PATCH",
            table,
            params={pk: f"eq.{row_id}"},
            json_body=row,
            extra_headers={"Prefer": "return=representation"},
        )
        # Confirm the row existed — PostgREST returns 200 with [] if no match.
        check = _supabase_request("GET", table, params={pk: f"eq.{row_id}", "select": pk})
    except SupabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    if not check:
        raise HTTPException(status_code=404, detail=f"No row found in '{table}' with {pk} = '{row_id}'")

    return {"table": table, "updated_id": row_id, "fields": row}


@app.get("/search")
def search_all(
    q: str = Query(..., min_length=2),
    api_key: Optional[str] = None,
    tables: Optional[str] = Query(None, description="Comma-separated list of tables to search; default all"),
    limit_per_table: int = Query(10, le=50),
):
    """Free-text search across text columns of the specified tables (or all tables)."""
    _check_auth(api_key)
    target_tables = [t.strip() for t in tables.split(",")] if tables else list(ALLOWED_TABLES.keys())

    # Columns known to hold large JSON blobs — skip these in free-text search,
    # matching the previous SQLite implementation's exclusion list.
    skip_cols = {"rubric_json", "dimension_breakdown"}

    results = {}
    for table in target_tables:
        if table not in ALLOWED_TABLES:
            continue
        text_cols = [c for c in table_text_columns(table) if c not in skip_cols]
        if not text_cols:
            continue

        # PostgREST OR filter: or=(col1.ilike.*q*,col2.ilike.*q*,...)
        or_clause = ",".join(f"{c}.ilike.*{q}*" for c in text_cols)
        params = {"select": "*", "or": f"({or_clause})", "limit": str(limit_per_table)}
        try:
            rows = _supabase_request("GET", table, params=params)
        except SupabaseError:
            # Some columns may not support ilike (e.g. non-text types) — skip
            # this table rather than failing the whole search.
            continue
        if rows:
            results[table] = rows

    return {"query": q, "results": results}
