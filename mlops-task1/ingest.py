"""Load the Olist CSV files into PostgreSQL.

Runs schema.sql (drop + create tables), then bulk-loads every CSV
with COPY, parents before children so the foreign keys resolve.
"""

import pathlib
import sys

import psycopg

DB_URL = "postgresql://olist:olist123@localhost:5433/olist"

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"

# (table, csv file) — parent tables first
FILES = [
    ("customers", "olist_customers_dataset.csv"),
    ("sellers", "olist_sellers_dataset.csv"),
    ("products", "olist_products_dataset.csv"),
    ("geolocation", "olist_geolocation_dataset.csv"),
    ("product_category_name_translation", "product_category_name_translation.csv"),
    ("orders", "olist_orders_dataset.csv"),
    ("order_items", "olist_order_items_dataset.csv"),
    ("order_payments", "olist_order_payments_dataset.csv"),
    ("order_reviews", "olist_order_reviews_dataset.csv"),
]


def main() -> None:
    missing = [f for _, f in FILES if not (DATA / f).exists()]
    if missing:
        sys.exit(f"Missing CSV files in {DATA}: {missing}")

    schema = (ROOT / "schema.sql").read_text(encoding="utf-8")

    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        print("Applying schema.sql ...")
        cur.execute(schema)

        for table, filename in FILES:
            with open(DATA / filename, encoding="utf-8") as f, cur.copy(
                f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
            ) as copy:
                while chunk := f.read(1 << 16):
                    copy.write(chunk)
            cur.execute(f"SELECT count(*) FROM {table}")
            print(f"  {table}: {cur.fetchone()[0]:,} rows")

    print("Done — all tables loaded and committed.")


if __name__ == "__main__":
    main()
