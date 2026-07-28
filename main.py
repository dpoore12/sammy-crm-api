"""
Sammy CRM API — live read/write access to Dan Poore's recruiting/BD SQLite database.
Built for use as a ChatGPT Custom GPT Action (OpenAPI schema in openapi.json).
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

DB_PATH = os.environ.get("SAMMY_DB_PATH", os.path.join(os.path.dirname(__file__), "data.db"))
API_KEY = os.environ.get("SAMMY_API_KEY")  # must be set in the hosting environment

app = FastAPI(
    title="Sammy CRM API",
    description="Live read/write access to Dan Poore's recruiting and business-development CRM.",
    version="1.0.0",
)

ALLOWED_TABLES = {
    "searches": ["search_id"],
    "candidates": ["candidate_id"],
    "scores": ["score_id"],
    "outreach": ["outreach_id"],
    "buyer_companies": ["company_id"],
    "buyer_contacts": ["contact_id"],
    "buyer_signals": ["signal_id"],
    "buyer_outreach": ["outreach_id"],
    "mpc_candidates": ["mpc_id"],
    "mpc_outreach": ["mpc_outreach_id"],
}


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def check_auth(x_api_key: Optional[str]):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: SAMMY_API_KEY not set")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def table_columns(table: str) -> List[str]:
    conn = get_conn()
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return [row["name"] for row in cur.fetchall()]
    finally:
        conn.close()


class RowUpsert(BaseModel):
    row: Dict[str, Any]


class QueryRequest(BaseModel):
    sql: str


@app.get("/health")
def health():
    return {"status": "ok", "db_path": DB_PATH, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/tables")
def list_tables(x_api_key: Optional[str] = Header(None)):
    check_auth(x_api_key)
    return {"tables": list(ALLOWED_TABLES.keys())}


@app.get("/tables/{table}/schema")
def table_schema(table: str, x_api_key: Optional[str] = Header(None)):
    check_auth(x_api_key)
    if table not in ALLOWED_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown table '{table}'")
    return {"table": table, "columns": table_columns(table)}


@app.get("/tables/{table}/rows")
def get_rows(
    table: str,
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    filter_op: str = Query("=", regex="^(=|!=|LIKE|>|<|>=|<=)$"),
    x_api_key: Optional[str] = Header(None),
):
    check_auth(x_api_key)
    if table not in ALLOWED_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown table '{table}'")
    cols = table_columns(table)
    conn = get_conn()
    try:
        sql = f"SELECT * FROM {table}"
        params: List[Any] = []
        if filter_column:
            if filter_column not in cols:
                raise HTTPException(status_code=400, detail=f"Unknown column '{filter_column}' on table '{table}'")
            value = filter_value
            if filter_op == "LIKE":
                value = f"%{filter_value}%"
            sql += f" WHERE {filter_column} {filter_op} ?"
            params.append(value)
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cur = conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        return {"table": table, "count": len(rows), "rows": rows}
    finally:
        conn.close()


@app.post("/tables/{table}/rows")
def create_row(table: str, payload: RowUpsert, x_api_key: Optional[str] = Header(None)):
    check_auth(x_api_key)
    if table not in ALLOWED_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown table '{table}'")
    cols = table_columns(table)
    row = dict(payload.row)

    pk = ALLOWED_TABLES[table][0]
    if pk not in row or not row[pk]:
        row[pk] = str(uuid.uuid4())

    now_iso = datetime.now(timezone.utc).isoformat()
    if "last_updated" in cols and "last_updated" not in row:
        row["last_updated"] = now_iso
    if "first_seen_date" in cols and "first_seen_date" not in row:
        row["first_seen_date"] = now_iso

    bad_cols = [k for k in row.keys() if k not in cols]
    if bad_cols:
        raise HTTPException(status_code=400, detail=f"Unknown columns for '{table}': {bad_cols}")

    conn = get_conn()
    try:
        keys = list(row.keys())
        placeholders = ", ".join(["?"] * len(keys))
        col_list = ", ".join(keys)
        conn.execute(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})", [row[k] for k in keys])
        conn.commit()
        return {"table": table, "created": row}
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=str(e))
    finally:
        conn.close()


@app.patch("/tables/{table}/rows/{row_id}")
def update_row(table: str, row_id: str, payload: RowUpsert, x_api_key: Optional[str] = Header(None)):
    check_auth(x_api_key)
    if table not in ALLOWED_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown table '{table}'")
    cols = table_columns(table)
    pk = ALLOWED_TABLES[table][0]
    row = dict(payload.row)

    if "last_updated" in cols:
        row["last_updated"] = datetime.now(timezone.utc).isoformat()

    bad_cols = [k for k in row.keys() if k not in cols]
    if bad_cols:
        raise HTTPException(status_code=400, detail=f"Unknown columns for '{table}': {bad_cols}")
    if not row:
        raise HTTPException(status_code=400, detail="No fields to update")

    conn = get_conn()
    try:
        set_clause = ", ".join([f"{k} = ?" for k in row.keys()])
        params = list(row.values()) + [row_id]
        cur = conn.execute(f"UPDATE {table} SET {set_clause} WHERE {pk} = ?", params)
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"No row found in '{table}' with {pk} = '{row_id}'")
        return {"table": table, "updated_id": row_id, "fields": row}
    finally:
        conn.close()


@app.get("/search")
def search_all(
    q: str = Query(..., min_length=2),
    tables: Optional[str] = Query(None, description="Comma-separated list of tables to search; default all"),
    limit_per_table: int = Query(10, le=50),
    x_api_key: Optional[str] = Header(None),
):
    """Free-text search across text columns of the specified tables (or all tables)."""
    check_auth(x_api_key)
    target_tables = [t.strip() for t in tables.split(",")] if tables else list(ALLOWED_TABLES.keys())
    conn = get_conn()
    results = {}
    try:
        for table in target_tables:
            if table not in ALLOWED_TABLES:
                continue
            cols = table_columns(table)
            text_cols = [c for c in cols if c not in ("rubric_json", "dimension_breakdown")]
            if not text_cols:
                continue
            where_clause = " OR ".join([f"{c} LIKE ?" for c in text_cols])
            params = [f"%{q}%"] * len(text_cols)
            sql = f"SELECT * FROM {table} WHERE {where_clause} LIMIT ?"
            params.append(limit_per_table)
            try:
                cur = conn.execute(sql, params)
                rows = [dict(r) for r in cur.fetchall()]
                if rows:
                    results[table] = rows
            except sqlite3.OperationalError:
                continue
        return {"query": q, "results": results}
    finally:
        conn.close()
