"""
Sammy CRM API — live read/write access to Dan Poore's recruiting/BD database.
Backed by a local SQLite file (data.db) in this project directory.
Built for use as a ChatGPT Custom GPT Action (OpenAPI schema in openapi.json).

Auth: pass the API key either as a query parameter `api_key=...` on any
request, or as a standard `Authorization: Bearer <key>` header (this is what
ChatGPT Actions' "Bearer" auth type sends, and is the method the GPT Action
uses). Note: a custom `X-Api-Key` header is stripped by this hosting proxy
(confirmed by direct header-echo testing) -- only `Authorization` survives,
so Bearer is the header-based option, not a custom header name.

History: this API originally ran against Supabase (Postgres) via PostgREST.
It was migrated back to local SQLite on 2026-07-28 to eliminate Supabase's
write-side rate limiting, which was repeatedly throttling bulk CRM imports.
The internal `_supabase_request` function name and PostgREST-style params
protocol (eq./ilike./gt. filters, select, limit/offset, Prefer headers) are
kept as-is so every call site in this file is unchanged -- only the
transport underneath (sqlite_backend.py) changed. All data from Supabase was
exported and merged into data.db before the cutover; no rows were lost.
"""

import json
import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import apollo_client

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Local SQLite backend (data.db in this project directory). Migrated off
# Supabase on 2026-07-28 -- see module docstring and sqlite_backend.py.
DB_PATH = os.path.join(os.path.dirname(__file__), "data.db")

API_KEY = os.environ.get("SAMMY_API_KEY") or "fP60UuK-NJRaFagWiFibmdXC91_WzUNhNq8ym2VhllM"  # falls back to the fixed key already configured in the Sammy GPT Action if the env var isn't set by the hosting environment

app = FastAPI(
    title="Sammy CRM API",
    description="Live read/write access to Dan Poore's recruiting and business-development CRM.",
    version="2.0.0",
)


@app.exception_handler(RequestValidationError)
async def _validation_error_as_400(request: Request, exc: RequestValidationError):
    """Convert FastAPI's default 422 body-validation errors into a plain 400
    with a single readable message, e.g. a completely missing 'row' key on
    POST /tables/{table}/rows. GPT Actions callers reason better about a flat
    400 + string detail than a nested 422 error-array."""
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        msg = first.get("msg", "Invalid request body")
        detail = f"{loc}: {msg}" if loc else msg
    else:
        detail = "Invalid request body"
    return JSONResponse(status_code=400, content={"detail": detail})

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
# Local SQLite transport (formerly a curl-to-Supabase transport; see
# sqlite_backend.py for the implementation). The function name, signature,
# and PostgREST-style params/headers protocol are kept identical so every
# call site below this line is completely unchanged by the migration.
# ---------------------------------------------------------------------------

from sqlite_backend import SupabaseError, supabase_request as _supabase_request  # noqa: E402


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


class ApolloPeopleSearchRequest(BaseModel):
    person_titles: Optional[List[str]] = None
    q_keywords: Optional[str] = None
    organization_domains: Optional[List[str]] = None
    organization_num_employees_ranges: Optional[List[str]] = None
    person_locations: Optional[List[str]] = None
    industries: Optional[List[str]] = None
    page: int = 1
    per_page: int = 10


class ApolloCompanySearchRequest(BaseModel):
    q_keywords: Optional[str] = None
    organization_num_employees_ranges: Optional[List[str]] = None
    organization_locations: Optional[List[str]] = None
    industries: Optional[List[str]] = None
    page: int = 1
    per_page: int = 10


class ApolloEnrichPersonRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    name: Optional[str] = None
    organization_name: Optional[str] = None
    domain: Optional[str] = None
    linkedin_url: Optional[str] = None
    email: Optional[str] = None


class ApolloEnrichCompanyRequest(BaseModel):
    domain: str


class ApolloImportRequest(BaseModel):
    target_table: str  # "buyer_contacts", "buyer_companies", or "candidates"
    people: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        max_length=25,
        description=(
            "REQUIRED when target_table is buyer_contacts or candidates. "
            "MAX 25 PER CALL. One object per person, built directly from "
            "apolloSearchPeople/apolloEnrichPerson results. For more than 25 "
            "people, call this action repeatedly with chunks of <=25 across "
            "separate messages -- check the response's 'remaining_hint' "
            "field and continue automatically if it is non-null."
        ),
    )
    companies: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        max_length=25,
        description=(
            "REQUIRED when target_table is buyer_companies. MAX 25 PER CALL. "
            "For more than 25 companies, call repeatedly with chunks of <=25."
        ),
    )
    source_label: str = "apollo_import"


def _check_auth(api_key: Optional[str] = None, request: Optional[Request] = None):
    """Accepts the key either as ?api_key=... or as an Authorization: Bearer
    header. Confirmed by direct header-echo testing that this hosting proxy
    strips a custom X-Api-Key header before it reaches the backend, but the
    standard Authorization header passes through untouched -- so Bearer auth
    (what ChatGPT Actions' "Bearer" auth type sends) is the reliable option
    for the GPT Action, while the query param remains available for direct
    testing/curl use.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: SAMMY_API_KEY not set")

    if api_key == API_KEY:
        return

    if request is not None:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token == API_KEY:
                return

    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key. Pass it as ?api_key=... or as 'Authorization: Bearer <key>'.",
    )


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
def list_tables(
    request: Request,
    api_key: Optional[str] = None,
):
    _check_auth(api_key, request)
    return {"tables": list(ALLOWED_TABLES.keys())}


def _exact_row_count(table: str) -> Optional[int]:
    """True total row count for a table via Postgres exact count, in a single
    request with almost no payload (Range: 0-0 asks for just 1 row back).
    Returns None if the count header could not be parsed (caller should treat
    that table's count as unknown rather than crashing the whole summary).
    """
    try:
        _, resp_headers = _supabase_request(
            "GET",
            table,
            params={"select": "*"},
            extra_headers={"Prefer": "count=exact", "Range-Unit": "items", "Range": "0-0"},
            return_headers=True,
        )
    except SupabaseError:
        return None
    content_range = resp_headers.get("content-range", "")
    # Format is "0-0/1234" (or "*/1234" if range unsatisfiable) — total is after the slash.
    if "/" in content_range:
        total_str = content_range.rsplit("/", 1)[-1]
        if total_str.isdigit():
            return int(total_str)
    return None


@app.get("/summary")
def summary(
    request: Request,
    api_key: Optional[str] = None,
):
    """One-call overview of the whole CRM: true row count for every table.
    Use this instead of looping /tables then /tables/{table}/rows for each
    table — that pattern is slow (11 sequential calls) and its per-table
    'count' field is capped by the page limit, not the real total.
    """
    _check_auth(api_key, request)
    # Run all 10 per-table count requests concurrently instead of sequentially.
    # Sequential curl subprocesses measured ~16.7s total for 10 tables -- well
    # past ChatGPT Actions' per-call timeout budget. Parallelizing collapses
    # this to roughly the duration of the single slowest table (~2s).
    tables = list(ALLOWED_TABLES.keys())
    with ThreadPoolExecutor(max_workers=len(tables)) as pool:
        results = list(pool.map(_exact_row_count, tables))
    counts: Dict[str, Optional[int]] = dict(zip(tables, results))
    return {
        "tables": counts,
        "total_rows": sum(v for v in counts.values() if v is not None),
        "note": "Counts are exact total row counts per table (not page-limited).",
    }


@app.get("/gpt-schema.json", include_in_schema=False)
def gpt_schema():
    """Serves the hand-written openapi.json (with explicit api_key query-param
    documentation on every operation) so ChatGPT's Action import points at a
    schema that unambiguously documents auth -- instead of FastAPI's
    auto-generated /openapi.json, which does not emit a securitySchemes block
    for plain function-parameter auth."""
    path = os.path.join(os.path.dirname(__file__), "openapi.json")
    return FileResponse(path, media_type="application/json")


@app.get("/tables/{table}/schema")
def table_schema(
    request: Request,
    table: str,
    api_key: Optional[str] = None,
):
    _check_auth(api_key, request)
    _require_table(table)
    return {"table": table, "columns": table_columns(table)}


@app.get("/tables/{table}/rows")
def get_rows(
    request: Request,
    table: str,
    api_key: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    filter_op: str = Query("=", pattern="^(=|!=|LIKE|>|<|>=|<=)$"),
):
    _check_auth(api_key, request)
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
    return {
        "table": table,
        "count": len(rows),
        "count_note": "count is the number of rows returned in THIS page only (capped by limit), not the table's total row count. Use GET /summary for total row counts across all tables.",
        "rows": rows,
    }


@app.post("/tables/{table}/rows")
def create_row(
    request: Request,
    table: str,
    payload: RowUpsert,
    api_key: Optional[str] = None,
):
    _check_auth(api_key, request)
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


class AddContactRequest(BaseModel):
    name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    linkedin_url: Optional[str] = None
    phone: Optional[str] = None
    source: str = "gpt_add_contact"


@app.post("/contacts/add")
def add_contact(
    request: Request,
    payload: AddContactRequest,
    api_key: Optional[str] = None,
):
    """Simplified single-contact create for buyer_contacts. Every field is a
    flat top-level parameter (no nested 'row' object) specifically so this is
    trivial for a GPT Action to call correctly: name, title, company, email,
    linkedin_url, phone, source. company is resolved to buyer_companies by
    name (creating a new minimal company row if it doesn't exist yet), so the
    caller never has to know or supply a company_id. Duplicate emails are
    skipped, not errored, so a bulk CSV import can call this once per row
    without pre-checking for existing contacts."""
    _check_auth(api_key, request)

    name = payload.name or f"{payload.first_name or ''} {payload.last_name or ''}".strip()
    if not name:
        raise HTTPException(status_code=400, detail="name (or first_name/last_name) is required")

    now_iso = datetime.now(timezone.utc).isoformat()

    if payload.email and _norm(payload.email) in _existing_contact_emails():
        return {
            "status": "skipped",
            "reason": "email already exists in buyer_contacts",
            "name": name,
            "email": payload.email,
        }

    if not payload.company or not _norm(payload.company):
        raise HTTPException(
            status_code=400,
            detail="company is required (buyer_contacts.company_id is a required field) -- pass the company name in the 'company' parameter",
        )
    try:
        company_id = _get_or_create_company_id(payload.company, now_iso, payload.source)
    except SupabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=f"company creation failed: {e.detail}")
    if not company_id:
        raise HTTPException(
            status_code=400,
            detail="company is required (buyer_contacts.company_id is a required field) -- pass the company name in the 'company' parameter",
        )

    cols = table_columns("buyer_contacts")
    row = {
        "contact_id": str(uuid.uuid4()),
        "company_id": company_id,
        "name": name,
        "title": payload.title,
        "email": payload.email,
        "linkedin_url": payload.linkedin_url,
        "phone": payload.phone,
        "source": payload.source,
        "first_seen_date": now_iso,
        "last_updated": now_iso,
    }
    row = {k: v for k, v in row.items() if v is not None and (not cols or k in cols)}
    try:
        _supabase_request_with_retry("POST", "buyer_contacts", json_body=row, extra_headers={"Prefer": "return=representation"})
    except SupabaseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    return {"status": "created", "contact": row}


class AddContactsBatchRequest(BaseModel):
    contacts: List[Dict[str, Any]] = Field(
        ...,
        max_length=25,
        description=(
            "One object per contact, MAX 25 PER CALL. Each needs at minimum a "
            "name (or first_name/last_name) and a company. Optional: title, "
            "email, linkedin_url, phone, source. "
            "IMPORTANT FOR LARGE IMPORTS (CSV or Apollo results with more than "
            "25 rows): split the full list into consecutive chunks of at most "
            "25 contacts each, and call this action ONCE PER CHUNK across "
            "separate messages -- never try to build one call with more than "
            "25 items, and never try to process a large import inside a single "
            "reply. After each call, check the response's 'remaining_hint' "
            "field: if it says more chunks are expected, immediately continue "
            "with the next chunk of up to 25 in your next message. It is always "
            "safe to resend a chunk you already sent -- contacts with emails "
            "already in the CRM are skipped as duplicates, not double-added."
        ),
    )
    source: str = "gpt_batch_import"


@app.post("/contacts/add-batch")
def add_contacts_batch(
    request: Request,
    payload: AddContactsBatchRequest,
    api_key: Optional[str] = None,
):
    """Import contacts from a CSV or Apollo result list, UP TO 25 PER CALL
    (hard limit -- larger arrays are rejected by the schema). For imports
    larger than 25 rows: split into consecutive chunks of <=25 and call this
    action once per chunk, each in its own message -- never try to hold a
    100-row import across only one reply. It is always safe to resend a
    chunk that was already imported; duplicate emails (already in the CRM,
    or repeated within the same call) are skipped, not errored, so retries
    and re-sent chunks never create duplicates. The response's
    'remaining_hint' field tells you whether to keep chunking: if it's
    non-null, there is likely more of the import left to send -- continue
    immediately with the next chunk in your next message rather than
    stopping or asking the user what to do next. Companies are
    resolved/created once per unique name within each chunk."""
    _check_auth(api_key, request)

    if not payload.contacts:
        raise HTTPException(status_code=400, detail="contacts array is empty or missing -- pass at least one contact object")

    now_iso = datetime.now(timezone.utc).isoformat()
    cols = table_columns("buyer_contacts")
    existing_emails = _existing_contact_emails()
    company_map = _company_id_by_name_map()
    seen_emails_this_batch: set = set()

    created: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for item in payload.contacts:
        name = item.get("name") or f"{item.get('first_name') or ''} {item.get('last_name') or ''}".strip()
        email = item.get("email")
        company_name = item.get("company") or item.get("organization_name")

        if not name:
            skipped.append({"input": item, "reason": "missing name"})
            continue
        # company_name may be a non-empty string that's pure whitespace (e.g.
        # "   ") -- catch that here with the same "missing company" reason
        # instead of letting it fall through to a confusing "could not
        # resolve or create" later.
        if not company_name or not _norm(company_name):
            skipped.append({"input": item, "reason": "missing company"})
            continue

        norm_email = _norm(email) if email else None
        if norm_email and (norm_email in existing_emails or norm_email in seen_emails_this_batch):
            skipped.append({"name": name, "email": email, "reason": "duplicate email"})
            continue

        norm_company = _norm(company_name)
        company_id = company_map.get(norm_company)
        if not company_id:
            try:
                company_id = _get_or_create_company_id(company_name, now_iso, payload.source, existing_map=company_map)
            except SupabaseError as e:
                skipped.append({"name": name, "company": company_name, "reason": f"company creation failed: {e.detail}"})
                continue
            if company_id:
                company_map[norm_company] = company_id
        if not company_id:
            skipped.append({"name": name, "company": company_name, "reason": "could not resolve or create company (no error raised, but no id returned)"})
            continue

        row = {
            "contact_id": str(uuid.uuid4()),
            "company_id": company_id,
            "name": name,
            "title": item.get("title"),
            "email": email,
            "linkedin_url": item.get("linkedin_url"),
            "phone": item.get("phone"),
            "source": item.get("source") or payload.source,
            "first_seen_date": now_iso,
            "last_updated": now_iso,
        }
        row = {k: v for k, v in row.items() if v is not None and (not cols or k in cols)}
        try:
            _supabase_request_with_retry("POST", "buyer_contacts", json_body=row, extra_headers={"Prefer": "return=representation"})
            created.append(row)
            if norm_email:
                seen_emails_this_batch.add(norm_email)
        except SupabaseError as e:
            skipped.append({"name": name, "reason": f"insert failed: {e.detail}"})

    # Since a single call is capped at 25 contacts (see AddContactsBatchRequest),
    # hitting exactly that cap is the signal that this was probably one chunk
    # of a larger import rather than the whole thing. Surface that directly in
    # the response instead of relying on the caller to remember a rule from
    # its own instructions -- this way the tool result itself tells the model
    # to keep going.
    remaining_hint = (
        "This call was capped at 25 contacts. If your full import has more "
        "rows after this chunk, send the next chunk of up to 25 now in a new "
        "call -- do not stop or ask the user, and do not resend rows already "
        "included in this or an earlier chunk."
        if len(payload.contacts) >= 25
        else None
    )
    return {
        "created_count": len(created),
        "skipped_count": len(skipped),
        "created": created,
        "skipped": skipped,
        "remaining_hint": remaining_hint,
    }


@app.patch("/tables/{table}/rows/{row_id}")
def update_row(
    request: Request,
    table: str,
    row_id: str,
    payload: RowUpsert,
    api_key: Optional[str] = None,
):
    _check_auth(api_key, request)
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
    request: Request,
    q: str = Query(..., min_length=2),
    api_key: Optional[str] = None,
    tables: Optional[str] = Query(None, description="Comma-separated list of tables to search; default all"),
    limit_per_table: int = Query(10, le=50),
):
    """Free-text search across text columns of the specified tables (or all tables)."""
    _check_auth(api_key, request)
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


# ---------------------------------------------------------------------------
# Apollo integration — search, enrich, and import-to-CRM
# ---------------------------------------------------------------------------


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _paginated_fetch(table: str, select: str, page_size: int = 1000) -> List[Dict[str, Any]]:
    """Fetch ALL rows from a table for a given select clause, paging past
    PostgREST's 1000-row default cap on unpaginated GETs. Any function that
    needs to check membership/dedupe against a whole table (existing emails,
    existing company names, etc.) must go through this instead of a single
    unpaginated GET -- tables with 1000+ rows will silently truncate
    otherwise, and dedupe checks will report false negatives for rows past
    the cutoff (i.e. duplicates get created instead of skipped)."""
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        try:
            rows = _supabase_request(
                "GET",
                table,
                params={"select": select, "limit": str(page_size), "offset": str(offset)},
            )
        except SupabaseError:
            break
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


def _supabase_request_with_retry(
    method: str,
    table: str,
    json_body: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    delays: Optional[List[float]] = None,
) -> Any:
    """Wraps _supabase_request with retry-with-backoff, specifically to
    survive transient PostgREST/Supabase rate_limit_exceeded responses under
    batch load (observed live: a handful of company/contact creates failing
    with rate_limit_exceeded within a single 25-row batch). Backs off longer
    on an actual rate-limit response, shorter on other transient errors."""
    if delays is None:
        delays = [0.5, 1.5, 3.0]
    last_error: Optional[SupabaseError] = None
    for attempt in range(len(delays) + 1):
        try:
            return _supabase_request(method, table, json_body=json_body, extra_headers=extra_headers)
        except SupabaseError as e:
            last_error = e
            if attempt < len(delays):
                is_rate_limit = "rate_limit" in str(e.detail).lower() or "rate limit" in str(e.detail).lower()
                time.sleep(delays[attempt] if is_rate_limit else 0.5)
                continue
    if last_error:
        raise last_error


def _existing_contact_emails() -> set:
    rows = _paginated_fetch("buyer_contacts", "email")
    return {_norm(r.get("email")) for r in rows if r.get("email")}


def _existing_candidate_emails() -> set:
    rows = _paginated_fetch("candidates", "work_email,personal_email")
    emails = set()
    for r in rows:
        if r.get("work_email"):
            emails.add(_norm(r["work_email"]))
        if r.get("personal_email"):
            emails.add(_norm(r["personal_email"]))
    return emails


def _existing_company_names() -> set:
    rows = _paginated_fetch("buyer_companies", "normalized_name,company_name")
    names = set()
    for r in rows:
        if r.get("normalized_name"):
            names.add(_norm(r["normalized_name"]))
        if r.get("company_name"):
            names.add(_norm(r["company_name"]))
    return names


def _company_id_by_name_map() -> Dict[str, str]:
    """normalized company name -> company_id, for resolving buyer_contacts.company_id
    from a plain organization_name string (e.g. from an Apollo search result).

    Paginates through ALL rows in buyer_companies. PostgREST caps an
    unpaginated GET at 1000 rows by default -- with 1000+ companies now in
    the table, a single unpaginated fetch silently missed the last ~46,
    which meant lookups for those companies always reported "not found" and
    the code would create duplicate rows for them instead of reusing the
    existing company_id. Paging in fixed-size chunks avoids that."""
    m: Dict[str, str] = {}
    rows = _paginated_fetch("buyer_companies", "company_id,normalized_name,company_name")
    for r in rows:
        cid = r.get("company_id")
        if not cid:
            continue
        if r.get("normalized_name"):
            m[_norm(r["normalized_name"])] = cid
        if r.get("company_name"):
            m[_norm(r["company_name"])] = cid
    return m


def _get_or_create_company_id(
    organization_name: Optional[str],
    now_iso: str,
    source_label: str,
    existing_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Resolve a company_id for buyer_contacts from a plain company name string.
    If the company already exists in buyer_companies (matched by normalized
    name), reuse its company_id. Otherwise create a minimal buyer_companies
    row for it and return the new company_id. Returns None only if no
    organization_name was given at all.

    Pass `existing_map` (from _company_id_by_name_map()) when calling this in
    a loop over many contacts -- e.g. a 100-row batch with ~90 unique
    companies was re-fetching and re-paginating the ENTIRE buyer_companies
    table from scratch on every single miss (90 full-table fetches for one
    batch). Each fetch is a real network round trip with its own timeout, so
    doing this dozens of times in one request was the most likely source of
    intermittent, non-reproducible "could not resolve or create company"
    failures under load -- not a data problem, a self-inflicted N+1 fetch
    storm. If existing_map is omitted, this fetches its own (single-call)
    lookup, for callers that only ever resolve one company."""
    if not organization_name:
        return None
    norm = _norm(organization_name)
    if not norm:
        return None
    if existing_map is None:
        existing_map = _company_id_by_name_map()
    if norm in existing_map:
        return existing_map[norm]
    cols = table_columns("buyer_companies")
    row = {
        "company_id": str(uuid.uuid4()),
        "company_name": organization_name,
        "normalized_name": norm,
        "source": source_label,
        "first_seen_date": now_iso,
        "last_updated": now_iso,
    }
    row = {k: v for k, v in row.items() if v is not None and (not cols or k in cols)}
    # Retry-with-backoff (including real Supabase rate-limit cooldowns) is
    # handled centrally by _supabase_request_with_retry -- see its docstring.
    # This raises the real SupabaseError on exhaustion so the caller's skip
    # message says WHY creation failed, not just that it failed.
    _supabase_request_with_retry("POST", "buyer_companies", json_body=row, extra_headers={"Prefer": "return=representation"})
    return row["company_id"]


@app.post("/apollo/search-people")
def apollo_search_people(
    request: Request,
    payload: ApolloPeopleSearchRequest,
    api_key: Optional[str] = None,
):
    """Search Apollo's people database by title, company domain, headcount,
    location, or industry. Returns raw Apollo results — use
    /apollo/import-to-crm afterward to dedupe and save the ones you want."""
    _check_auth(api_key, request)
    try:
        result = apollo_client.search_people(
            person_titles=payload.person_titles,
            q_keywords=payload.q_keywords,
            organization_domains=payload.organization_domains,
            organization_num_employees_ranges=payload.organization_num_employees_ranges,
            person_locations=payload.person_locations,
            industries=payload.industries,
            page=payload.page,
            per_page=payload.per_page,
        )
    except apollo_client.ApolloError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    people = result.get("people", []) or []
    existing_emails = _existing_contact_emails() | _existing_candidate_emails()
    simplified = []
    for p in people:
        email = p.get("email")
        org = p.get("organization") or {}
        simplified.append({
            "name": p.get("name"),
            "first_name": p.get("first_name"),
            "last_name": p.get("last_name"),
            "title": p.get("title"),
            "email": email,
            "linkedin_url": p.get("linkedin_url"),
            "organization_name": org.get("name"),
            "organization_domain": org.get("primary_domain") or org.get("website_url"),
            "already_in_crm": bool(email) and _norm(email) in existing_emails,
        })
    pagination = result.get("pagination", {})
    return {
        "query": payload.dict(exclude_none=True),
        "total_entries": pagination.get("total_entries"),
        "page": pagination.get("page"),
        "total_pages": pagination.get("total_pages"),
        "count_returned": len(simplified),
        "people": simplified,
        "note": "already_in_crm flags people whose email already exists in buyer_contacts or candidates. Call /apollo/import-to-crm with the ones you want to save.",
    }


@app.post("/apollo/search-companies")
def apollo_search_companies(
    request: Request,
    payload: ApolloCompanySearchRequest,
    api_key: Optional[str] = None,
):
    """Search Apollo's company database by headcount, location, industry, or
    keyword. Returns raw Apollo results — use /apollo/import-to-crm
    afterward to dedupe and save the ones you want."""
    _check_auth(api_key, request)
    try:
        result = apollo_client.search_companies(
            q_keywords=payload.q_keywords,
            organization_num_employees_ranges=payload.organization_num_employees_ranges,
            organization_locations=payload.organization_locations,
            industries=payload.industries,
            page=payload.page,
            per_page=payload.per_page,
        )
    except apollo_client.ApolloError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    companies = result.get("accounts", []) or result.get("organizations", []) or []
    existing_names = _existing_company_names()
    simplified = []
    for c in companies:
        name = c.get("name")
        simplified.append({
            "name": name,
            "website_url": c.get("website_url"),
            "linkedin_url": c.get("linkedin_url"),
            "industry": c.get("industry"),
            "estimated_num_employees": c.get("estimated_num_employees"),
            "city": c.get("city"),
            "state": c.get("state"),
            "founded_year": c.get("founded_year"),
            "phone": c.get("phone"),
            "already_in_crm": bool(name) and _norm(name) in existing_names,
        })
    pagination = result.get("pagination", {})
    return {
        "query": payload.dict(exclude_none=True),
        "total_entries": pagination.get("total_entries"),
        "page": pagination.get("page"),
        "total_pages": pagination.get("total_pages"),
        "count_returned": len(simplified),
        "companies": simplified,
        "note": "already_in_crm flags companies whose name already exists in buyer_companies. Call /apollo/import-to-crm with the ones you want to save.",
    }


@app.post("/apollo/enrich-person")
def apollo_enrich_person(
    request: Request,
    payload: ApolloEnrichPersonRequest,
    api_key: Optional[str] = None,
):
    """Enrich a single person via Apollo (by name+company, domain, LinkedIn
    URL, or email) to get their verified email, title, and org info."""
    _check_auth(api_key, request)
    try:
        result = apollo_client.enrich_person(
            first_name=payload.first_name,
            last_name=payload.last_name,
            name=payload.name,
            organization_name=payload.organization_name,
            domain=payload.domain,
            linkedin_url=payload.linkedin_url,
            email=payload.email,
        )
    except apollo_client.ApolloError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    return result


@app.post("/apollo/enrich-company")
def apollo_enrich_company(
    request: Request,
    payload: ApolloEnrichCompanyRequest,
    api_key: Optional[str] = None,
):
    """Enrich a single company via Apollo by domain — returns firmographic
    details (headcount, industry, socials, founded year, etc.)."""
    _check_auth(api_key, request)
    try:
        result = apollo_client.enrich_company(domain=payload.domain)
    except apollo_client.ApolloError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    return result


@app.post("/apollo/import-to-crm")
def apollo_import_to_crm(
    request: Request,
    payload: ApolloImportRequest,
    api_key: Optional[str] = None,
):
    """Take a list of Apollo people or companies (as returned by
    /apollo/search-people or /apollo/search-companies) and create the
    missing ones in Sammy CRM. Skips anyone/anything already present
    (matched by email for people, by name for companies). Target table
    must be 'buyer_contacts', 'candidates', or 'buyer_companies'."""
    _check_auth(api_key, request)

    # Guard against silent no-ops: if the caller says "import these" but sends
    # an empty/missing people or companies array, fail loudly instead of
    # returning a fake 0-created/0-skipped success. This was previously
    # silent -- the GPT could call this with an empty list (e.g. because the
    # schema didn't expose named fields, so it didn't know how to build the
    # objects) and get back what looked like a clean successful response.
    if payload.target_table in ("buyer_contacts", "candidates"):
        if not payload.people:
            raise HTTPException(
                status_code=400,
                detail=(
                    "people array is empty or missing. target_table='" + payload.target_table + "' "
                    "requires a non-empty 'people' array -- copy the actual person objects "
                    "(name, email, title, linkedin_url, organization_name) from the "
                    "apolloSearchPeople or apolloEnrichPerson response you just received "
                    "and pass them here. Nothing was imported."
                ),
            )
    elif payload.target_table == "buyer_companies":
        if not payload.companies:
            raise HTTPException(
                status_code=400,
                detail=(
                    "companies array is empty or missing. target_table='buyer_companies' "
                    "requires a non-empty 'companies' array -- copy the actual company objects "
                    "(name, website_url, city, state, founded_year) from the "
                    "apolloSearchCompanies response you just received and pass them here. "
                    "Nothing was imported."
                ),
            )

    now_iso = datetime.now(timezone.utc).isoformat()
    created = []
    skipped = []

    if payload.target_table == "buyer_contacts":
        existing = _existing_contact_emails()
        cols = table_columns("buyer_contacts")
        # Build the company lookup map ONCE for the whole import instead of
        # letting _get_or_create_company_id re-fetch and re-paginate all of
        # buyer_companies on every person -- same N+1 fetch-storm fix as the
        # /contacts/add-batch endpoint.
        company_map = _company_id_by_name_map()
        for p in (payload.people or []):
            email = p.get("email")
            name = p.get("name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
            if email and _norm(email) in existing:
                skipped.append({"name": name, "email": email, "reason": "email already in buyer_contacts"})
                continue
            # buyer_contacts.company_id is NOT NULL, but Apollo search/enrich
            # results only give a plain organization_name string. Resolve it
            # to an existing buyer_companies.company_id, or auto-create a
            # minimal company row if none exists yet, instead of failing the
            # whole import with a database constraint error.
            org_name = p.get("organization_name")
            try:
                company_id = _get_or_create_company_id(org_name, now_iso, payload.source_label, existing_map=company_map)
            except SupabaseError as e:
                skipped.append({"name": name, "email": email, "reason": f"company creation failed: {e.detail}"})
                continue
            if company_id and org_name:
                company_map[_norm(org_name)] = company_id
            if not company_id:
                skipped.append({
                    "name": name,
                    "email": email,
                    "reason": "no organization_name provided, and buyer_contacts.company_id is required -- include organization_name from the Apollo result",
                })
                continue
            row = {
                "contact_id": str(uuid.uuid4()),
                "company_id": company_id,
                "name": name,
                "title": p.get("title"),
                "email": email,
                "linkedin_url": p.get("linkedin_url"),
                "source": payload.source_label,
                "first_seen_date": now_iso,
                "last_updated": now_iso,
            }
            row = {k: v for k, v in row.items() if not cols or k in cols}
            try:
                _supabase_request_with_retry("POST", "buyer_contacts", json_body=row, extra_headers={"Prefer": "return=representation"})
                created.append(row)
                if email:
                    existing.add(_norm(email))
            except SupabaseError as e:
                skipped.append({"name": name, "email": email, "reason": f"insert failed: {e.detail}"})

    elif payload.target_table == "candidates":
        existing = _existing_candidate_emails()
        cols = table_columns("candidates")
        for p in (payload.people or []):
            email = p.get("email")
            name = p.get("name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
            if email and _norm(email) in existing:
                skipped.append({"name": name, "email": email, "reason": "email already in candidates"})
                continue
            row = {
                "candidate_id": str(uuid.uuid4()),
                "name": name,
                "linkedin_url": p.get("linkedin_url"),
                "current_title": p.get("title"),
                "current_company": p.get("organization_name"),
                "work_email": email,
                "source": payload.source_label,
                "first_seen_date": now_iso,
                "last_updated": now_iso,
            }
            row = {k: v for k, v in row.items() if not cols or k in cols}
            try:
                _supabase_request("POST", "candidates", json_body=row, extra_headers={"Prefer": "return=representation"})
                created.append(row)
                if email:
                    existing.add(_norm(email))
            except SupabaseError as e:
                skipped.append({"name": name, "email": email, "reason": f"insert failed: {e.detail}"})

    elif payload.target_table == "buyer_companies":
        existing = _existing_company_names()
        cols = table_columns("buyer_companies")
        for c in (payload.companies or []):
            name = c.get("name")
            if name and _norm(name) in existing:
                skipped.append({"name": name, "reason": "already in buyer_companies"})
                continue
            row = {
                "company_id": str(uuid.uuid4()),
                "company_name": name,
                "normalized_name": _norm(name),
                "website": c.get("website_url"),
                "hq_city": c.get("city"),
                "hq_state": c.get("state"),
                "year_founded": c.get("founded_year"),
                "source": payload.source_label,
                "first_seen_date": now_iso,
                "last_updated": now_iso,
            }
            row = {k: v for k, v in row.items() if v is not None and (not cols or k in cols)}
            try:
                _supabase_request("POST", "buyer_companies", json_body=row, extra_headers={"Prefer": "return=representation"})
                created.append(row)
                if name:
                    existing.add(_norm(name))
            except SupabaseError as e:
                skipped.append({"name": name, "reason": f"insert failed: {e.detail}"})
    else:
        raise HTTPException(status_code=400, detail="target_table must be one of: buyer_contacts, candidates, buyer_companies")

    input_count = len(payload.people or payload.companies or [])
    remaining_hint = (
        "This call was capped at 25 items. If your full import has more "
        "rows after this chunk, send the next chunk of up to 25 now in a "
        "new call -- do not stop or ask the user, and do not resend items "
        "already included in this or an earlier chunk."
        if input_count >= 25
        else None
    )
    return {
        "target_table": payload.target_table,
        "created_count": len(created),
        "skipped_count": len(skipped),
        "created": created,
        "skipped": skipped,
        "remaining_hint": remaining_hint,
    }
