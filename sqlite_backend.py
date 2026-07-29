"""
SQLite-backed drop-in replacement for the Supabase/PostgREST transport layer.

This module implements `_supabase_request`-compatible semantics against a
local SQLite file (data.db) instead of Supabase's REST API. It understands
the same PostgREST-style query language the rest of main.py already builds:

  - params={"select": "*", "limit": "50", "offset": "0"}
  - params={"<col>": "eq.<value>"}          -> WHERE col = value
  - params={"<col>": "neq.<value>"}         -> WHERE col != value
  - params={"<col>": "ilike.*value*"}       -> WHERE col LIKE '%value%' (case-insensitive)
  - params={"<col>": "gt.<value>"}, lt, gte, lte
  - params={"or": "(col1.ilike.*q*,col2.ilike.*q*)"}  -> OR'd ILIKE clauses
  - extra_headers={"Prefer": "count=exact", "Range": "0-0"}  -> exact count via
    a synthetic Content-Range response header, so _exact_row_count keeps working
    unmodified.
  - extra_headers={"Prefer": "return=representation"} on POST/PATCH -> returns
    the affected row(s) as a list, matching PostgREST's behavior.
  - method="POST" -> INSERT
  - method="PATCH" -> UPDATE (requires an eq. filter on the primary key column,
    same as the calling code already does)
  - method="GET" -> SELECT

No retry/backoff logic is needed here (that was to survive Supabase's network
rate limits) but _supabase_request_with_retry still works unmodified since it
just calls _supabase_request and only retries on SupabaseError.
"""

import os
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "data.db")

_local = threading.local()


class SupabaseError(Exception):
    """Kept as the same exception type/name main.py already imports and
    catches everywhere, so no call-site changes are needed."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        _local.conn = conn
    return conn


_OP_MAP = {
    "eq": "=",
    "neq": "!=",
    "gt": ">",
    "lt": "<",
    "gte": ">=",
    "lte": "<=",
}


def _parse_filter_value(raw: str) -> Tuple[str, Any]:
    """Given a PostgREST-style filter value like 'eq.foo' or 'ilike.*bar*',
    return (sql_op, python_value) ready to bind into a parameterized query."""
    if "." not in raw:
        # Shouldn't happen given how main.py builds params, but degrade
        # gracefully to an equality check on the raw string.
        return "=", raw
    op, _, value = raw.partition(".")
    if op == "ilike":
        # PostgREST wraps the search term in literal *stars*; convert to SQL LIKE %term%
        value = value.strip("*")
        return "LIKE", f"%{value}%"
    sql_op = _OP_MAP.get(op)
    if sql_op is None:
        return "=", value
    return sql_op, value


def _parse_or_clause(or_expr: str) -> Tuple[str, List[Any]]:
    """Parse PostgREST's or=(col1.ilike.*q*,col2.ilike.*q*,...) into a SQL
    fragment and bind params."""
    inner = or_expr.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    clauses = []
    binds: List[Any] = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
        col, _, rest = part.partition(".")
        sql_op, value = _parse_filter_value(rest)
        clauses.append(f'"{col}" {sql_op} ?')
        binds.append(value)
    return " OR ".join(clauses), binds


def _table_exists(table: str) -> bool:
    conn = _get_conn()
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table,)
    )
    return cur.fetchone() is not None


def _table_columns_raw(table: str) -> List[str]:
    conn = _get_conn()
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    return [r[1] for r in cur.fetchall()]


def supabase_request(
    method: str,
    path: str,
    params: Optional[Dict[str, str]] = None,
    json_body: Optional[Any] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    return_headers: bool = False,
):
    """Same signature/semantics as the original _supabase_request, executed
    against local SQLite instead of an HTTP call to Supabase."""
    params = params or {}
    extra_headers = extra_headers or {}
    table = path.strip("/").split("/")[0]

    if not _table_exists(table):
        raise SupabaseError(404, f"relation \"{table}\" does not exist")

    conn = _get_conn()

    try:
        if method == "GET":
            result = _handle_get(conn, table, params, extra_headers)
        elif method == "POST":
            result = _handle_post(conn, table, json_body, extra_headers)
        elif method == "PATCH":
            result = _handle_patch(conn, table, params, json_body, extra_headers)
        elif method == "DELETE":
            result = _handle_delete(conn, table, params)
        else:
            raise SupabaseError(405, f"Unsupported method: {method}")
    except sqlite3.Error as e:
        raise SupabaseError(500, f"SQLite error: {e}")

    if return_headers:
        data, resp_headers = result
        return data, resp_headers
    if isinstance(result, tuple):
        return result[0]
    return result


def _build_where(table: str, params: Dict[str, str]) -> Tuple[str, List[Any]]:
    reserved = {"select", "limit", "offset", "order"}
    where_clauses = []
    binds: List[Any] = []
    for key, raw_value in params.items():
        if key in reserved:
            continue
        if key == "or":
            frag, or_binds = _parse_or_clause(raw_value)
            if frag:
                where_clauses.append(f"({frag})")
                binds.extend(or_binds)
            continue
        sql_op, value = _parse_filter_value(raw_value)
        where_clauses.append(f'"{key}" {sql_op} ?')
        binds.append(value)
    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    return where_sql, binds


def _handle_get(conn, table, params, extra_headers):
    select = params.get("select", "*")
    limit = params.get("limit")
    offset = params.get("offset")
    where_sql, binds = _build_where(table, params)

    prefer = extra_headers.get("Prefer", "")
    range_header = extra_headers.get("Range")

    # Exact-count probe: Prefer: count=exact + Range: 0-0 asks for just the
    # count, mirroring Postgres/PostgREST's Content-Range response header.
    if "count=exact" in prefer:
        count_sql = f'SELECT COUNT(*) as c FROM "{table}"{where_sql}'
        cur = conn.execute(count_sql, binds)
        total = cur.fetchone()["c"]
        resp_headers = {"content-range": f"0-0/{total}"}
        # Still return a tiny row payload consistent with Range: 0-0 semantics.
        row_sql = f'SELECT * FROM "{table}"{where_sql} LIMIT 1'
        cur = conn.execute(row_sql, binds)
        rows = [dict(r) for r in cur.fetchall()]
        return rows, resp_headers

    sql = f'SELECT {select if select != "*" else "*"} FROM "{table}"{where_sql}'
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
        if offset is not None:
            sql += f" OFFSET {int(offset)}"
    cur = conn.execute(sql, binds)
    rows = [dict(r) for r in cur.fetchall()]
    return rows, {}


def _handle_post(conn, table, json_body, extra_headers):
    if json_body is None:
        raise SupabaseError(400, "Missing request body for INSERT")
    rows = json_body if isinstance(json_body, list) else [json_body]
    cols = _table_columns_raw(table)
    created = []
    for row in rows:
        row_cols = [c for c in row.keys() if not cols or c in cols]
        placeholders = ",".join("?" for _ in row_cols)
        colnames = ",".join(f'"{c}"' for c in row_cols)
        vals = [row[c] for c in row_cols]
        sql = f'INSERT INTO "{table}" ({colnames}) VALUES ({placeholders})'
        conn.execute(sql, vals)
        created.append(row)
    conn.commit()

    prefer = extra_headers.get("Prefer", "")
    if "return=representation" in prefer:
        return created, {}
    return None, {}


def _handle_patch(conn, table, params, json_body, extra_headers):
    if json_body is None:
        raise SupabaseError(400, "Missing request body for UPDATE")
    where_sql, binds = _build_where(table, params)
    if not where_sql:
        raise SupabaseError(400, "PATCH requires a filter (e.g. eq.<id>) to avoid updating all rows")

    cols = _table_columns_raw(table)
    set_cols = [c for c in json_body.keys() if not cols or c in cols]
    if not set_cols:
        return [], {}
    set_sql = ", ".join(f'"{c}" = ?' for c in set_cols)
    set_vals = [json_body[c] for c in set_cols]

    sql = f'UPDATE "{table}" SET {set_sql}{where_sql}'
    conn.execute(sql, set_vals + binds)
    conn.commit()

    prefer = extra_headers.get("Prefer", "")
    if "return=representation" in prefer:
        select_sql = f'SELECT * FROM "{table}"{where_sql}'
        cur = conn.execute(select_sql, binds)
        rows = [dict(r) for r in cur.fetchall()]
        return rows, {}
    return None, {}


def _handle_delete(conn, table, params):
    where_sql, binds = _build_where(table, params)
    if not where_sql:
        raise SupabaseError(400, "DELETE requires a filter to avoid deleting all rows")
    sql = f'DELETE FROM "{table}"{where_sql}'
    conn.execute(sql, binds)
    conn.commit()
    return None, {}
