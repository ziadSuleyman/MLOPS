# MLOps Task 1 — Olist Data → PostgreSQL

Qafza MLOps Training 2026/2027 · Task 1: load the Olist Brazilian E-Commerce
dataset (Kaggle: `olistbr/brazilian-ecommerce`) into a relational database.

**The ML problem this data serves:** predict whether an order will be
delivered late or on time (`order_delivered_customer_date` vs
`order_estimated_delivery_date`). Baseline from the loaded data: **8.11%**
of delivered orders arrived late.

## Stack

| Piece | Value |
|---|---|
| Database | PostgreSQL 16 (Docker container `olist-postgres`) |
| Port | `localhost:5433` (5432 was taken by a local Postgres) |
| Credentials | user `olist` / password `olist123` / db `olist` |
| Ingestion | Python 3.13 + psycopg 3, bulk `COPY` |

## Files

- `docker-compose.yml` — Postgres 16 + persistent volume `pgdata` + healthcheck
- `schema.sql` — the 9 tables, PKs/FKs, indexes (column order matches the CSVs)
- `ingest.py` — applies schema.sql then COPY-loads every CSV, parents first
- `test_queries.sql` — row counts, 2- and 3-table joins, late-delivery rate
- `olist_explore.ipynb` — Jupyter notebook: connects to the DB and runs the same queries with pandas
- `baseline_model.py` — leakage-aware baseline: chronological split, two information scenarios
- `REPORT_AR.md` / `report.html` — analysis report: business impact, lateness drivers, project ideas
- `data/` — the 9 CSV files from Kaggle (~120 MB)

## How to run

```bash
docker compose up -d          # start the database
python ingest.py              # (re)create tables + load all CSVs
docker exec -it olist-postgres psql -U olist -d olist   # open a SQL shell
```

Python connection string: `postgresql://olist:olist123@localhost:5433/olist`

## Starting up again after a shutdown

The container has **no restart policy**, so nothing comes back on its own after
a reboot — but the data does survive in the `mlops-task1_pgdata` Docker volume,
so there is never a need to re-download or re-ingest.

One command does everything (starts Docker if needed, waits for the database,
checks the data is present, opens JupyterLab):

```bash
powershell -ExecutionPolicy Bypass -File "C:\Users\Rings-1\MLOPS\mlops-task1\start.ps1"
```

Add `-NoJupyter` for the database only. To shut down: `stop.ps1`.

Manual equivalent, if you prefer to see each step:

```bash
docker compose up -d
docker exec olist-postgres psql -U olist -d olist -c "SELECT count(*) FROM orders;"
python -m jupyterlab --notebook-dir "C:\Users\Rings-1\MLOPS\mlops-task1"
```

`docker compose down` keeps the volume. Only `docker compose down -v` destroys
it — after that you must rerun `python ingest.py`.

To make the database start automatically with Windows, add
`restart: unless-stopped` to the `postgres` service in `docker-compose.yml`.

## Jupyter

```bash
python -m jupyterlab --notebook-dir "C:\Users\Rings-1\MLOPS\mlops-task1"
```

Opens on <http://localhost:8888/lab> — then open `olist_explore.ipynb`.
Requires `jupyterlab pandas sqlalchemy "psycopg[binary]"`. Inside the
notebook the connection uses the SQLAlchemy driver prefix:

```python
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine("postgresql+psycopg://olist:olist123@localhost:5433/olist")
df = pd.read_sql(text("SELECT * FROM orders LIMIT 5"), engine)
```

## Tables & relations (rows loaded)

```
customers (99,441) ──< orders (99,441) ──< order_items (112,650) >── products (32,951)
                              │                    └──────────────>── sellers (3,095)
                              ├──< order_payments (103,886)
                              └──< order_reviews (99,224)

geolocation (1,000,163)                 — zip prefix reference, has duplicate rows, no PK
product_category_name_translation (71)  — pt → en category names
```

Key relations: `orders.customer_id → customers`; `order_items` links each
order line to its `product_id` and `seller_id`; payments and reviews hang off
`orders.order_id`. `order_reviews` PK is `(review_id, order_id)` because one
review can cover several orders.

## Notes / gotchas

- Zip code prefixes are stored as `VARCHAR(5)` to keep leading zeros.
- `geolocation` has no primary key — the source file contains exact duplicates.
- Empty CSV fields are loaded as `NULL` (`COPY ... NULL ''`).
- Data is persisted in the `pgdata` Docker volume — it survives container
  restarts; `docker compose down -v` wipes it (then rerun `ingest.py`).
