-- Verification queries for Task 1

-- 1) Row count per table
SELECT 'customers' AS table_name, count(*) FROM customers
UNION ALL SELECT 'sellers', count(*) FROM sellers
UNION ALL SELECT 'products', count(*) FROM products
UNION ALL SELECT 'geolocation', count(*) FROM geolocation
UNION ALL SELECT 'product_category_name_translation', count(*) FROM product_category_name_translation
UNION ALL SELECT 'orders', count(*) FROM orders
UNION ALL SELECT 'order_items', count(*) FROM order_items
UNION ALL SELECT 'order_payments', count(*) FROM order_payments
UNION ALL SELECT 'order_reviews', count(*) FROM order_reviews
ORDER BY table_name;

-- 2) A join: orders with their customers
SELECT o.order_id, o.order_status, o.order_purchase_timestamp,
       c.customer_city, c.customer_state
FROM orders o
JOIN customers c USING (customer_id)
LIMIT 10;

-- 3) A 3-table join: items with product category and seller state
SELECT oi.order_id, p.product_category_name, s.seller_state, oi.price
FROM order_items oi
JOIN products p USING (product_id)
JOIN sellers  s USING (seller_id)
LIMIT 10;

-- 4) The problem we are solving: how many delivered orders arrived late?
SELECT
    count(*)                                                              AS delivered_orders,
    count(*) FILTER (WHERE order_delivered_customer_date
                         > order_estimated_delivery_date)                 AS late_orders,
    round(100.0 * count(*) FILTER (WHERE order_delivered_customer_date
                                       > order_estimated_delivery_date)
                / count(*), 2)                                            AS late_pct
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL;
