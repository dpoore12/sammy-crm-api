# Sammy CRM API

Live read/write HTTP API over Dan Poore's Sammy CRM SQLite database (`data/sammy_crm.db`).
Built specifically to be wired into a ChatGPT Custom GPT via a GPT Action, so the GPT can
query and write real, current CRM data in real time — no manual re-uploads, no CSV exports.

## Tables exposed

`searches`, `candidates`, `scores`, `outreach`, `buyer_companies`, `buyer_contacts`,
`buyer_signals`, `buyer_outreach`, `mpc_candidates`, `mpc_outreach`

## Endpoints

- `GET /health` — status check (no auth)
- `GET /tables` — list available tables
- `GET /tables/{table}/schema` — column names for a table
- `GET /tables/{table}/rows` — read rows, with optional filter (`filter_column`, `filter_value`, `filter_op`), `limit`, `offset`
- `POST /tables/{table}/rows` — create a row (`{"row": {...}}`)
- `PATCH /tables/{table}/rows/{row_id}` — update a row by primary key (`{"row": {...}}`)
- `GET /search?q=...` — free-text search across all text columns, optionally scoped to specific `tables`

All endpoints except `/health` require header `x-api-key: <SAMMY_API_KEY>`.

## Environment variables

- `SAMMY_API_KEY` — required. Shared secret the GPT Action must send as `x-api-key`.
- `SAMMY_DB_PATH` — optional. Defaults to `data/sammy_crm.db` inside the repo.

## Running locally

```bash
pip install -r requirements.txt
export SAMMY_API_KEY="your-secret-key"
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Wiring into ChatGPT

See `openapi.json` — import it into a Custom GPT's "Actions" configuration, with
Authentication set to API Key, header name `x-api-key`, value = your `SAMMY_API_KEY`.
