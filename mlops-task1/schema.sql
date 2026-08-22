-- Olist Brazilian E-Commerce — schema
-- Column order in every table matches the CSV column order exactly,
-- because ingest.py loads the files with COPY ... FORMAT csv.

DROP TABLE IF EXISTS
    order_reviews,
    order_payments,
    order_items,
    orders,
    products,
    sellers,
    customers,
    geolocation,
    product_category_name_translation
CASCADE;

CREATE TABLE customers (
    customer_id              VARCHAR(32) PRIMARY KEY,
    customer_unique_id       VARCHAR(32) NOT NULL,
    customer_zip_code_prefix VARCHAR(5)  NOT NULL,  -- keeps leading zeros
    customer_city            TEXT        NOT NULL,
    customer_state           CHAR(2)     NOT NULL
);

CREATE TABLE sellers (
    seller_id              VARCHAR(32) PRIMARY KEY,
    seller_zip_code_prefix VARCHAR(5)  NOT NULL,
    seller_city            TEXT        NOT NULL,
    seller_state           CHAR(2)     NOT NULL
);

CREATE TABLE products (
    product_id                 VARCHAR(32) PRIMARY KEY,
    product_category_name      TEXT,
    product_name_lenght        NUMERIC,
    product_description_lenght NUMERIC,
    product_photos_qty         NUMERIC,
    product_weight_g           NUMERIC,
    product_length_cm          NUMERIC,
    product_height_cm          NUMERIC,
    product_width_cm           NUMERIC
);

-- No primary key: the source file contains exact duplicate rows.
CREATE TABLE geolocation (
    geolocation_zip_code_prefix VARCHAR(5),
    geolocation_lat             DOUBLE PRECISION,
    geolocation_lng             DOUBLE PRECISION,
    geolocation_city            TEXT,
    geolocation_state           CHAR(2)
);

CREATE TABLE product_category_name_translation (
    product_category_name         TEXT PRIMARY KEY,
    product_category_name_english TEXT
);

CREATE TABLE orders (
    order_id                      VARCHAR(32) PRIMARY KEY,
    customer_id                   VARCHAR(32) NOT NULL REFERENCES customers (customer_id),
    order_status                  VARCHAR(20) NOT NULL,
    order_purchase_timestamp      TIMESTAMP   NOT NULL,
    order_approved_at             TIMESTAMP,
    order_delivered_carrier_date  TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP
);

CREATE TABLE order_items (
    order_id            VARCHAR(32) REFERENCES orders (order_id),
    order_item_id       INT,
    product_id          VARCHAR(32) REFERENCES products (product_id),
    seller_id           VARCHAR(32) REFERENCES sellers (seller_id),
    shipping_limit_date TIMESTAMP,
    price               NUMERIC(10, 2),
    freight_value       NUMERIC(10, 2),
    PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE order_payments (
    order_id             VARCHAR(32) REFERENCES orders (order_id),
    payment_sequential   INT,
    payment_type         VARCHAR(20),
    payment_installments INT,
    payment_value        NUMERIC(10, 2),
    PRIMARY KEY (order_id, payment_sequential)
);

-- review_id alone is NOT unique (one review can cover several orders),
-- so the key is the (review_id, order_id) pair.
CREATE TABLE order_reviews (
    review_id               VARCHAR(32),
    order_id                VARCHAR(32) REFERENCES orders (order_id),
    review_score            INT,
    review_comment_title    TEXT,
    review_comment_message  TEXT,
    review_creation_date    TIMESTAMP,
    review_answer_timestamp TIMESTAMP,
    PRIMARY KEY (review_id, order_id)
);

CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_items_product   ON order_items (product_id);
CREATE INDEX idx_items_seller    ON order_items (seller_id);
CREATE INDEX idx_geo_zip         ON geolocation (geolocation_zip_code_prefix);
