-- ============================================================
-- DISCOUNT SENSITIVITY FEATURE TABLE (V10) - CORRECTED
-- 
-- FIX: Use only coupondiscountdistr (€ amount) for discount
--      Removed product_discount and shipping_discount (they are %)
--
-- Grain / PK: externalcustomerkey
-- ============================================================

CREATE TABLE public.discount_sensitivity_features_v10
DISTSTYLE KEY
DISTKEY (externalcustomerkey)
SORTKEY (externalcustomerkey)
AS

WITH
params AS (
    SELECT
        CURRENT_DATE::date AS as_of_date,
        DATEADD(year, -5, CURRENT_DATE::date) AS start_date
),

/* =======================
   1) FX RATES
   ======================= */
fx AS (
    SELECT currencycode, date::date AS rate_date, currencyrate
    FROM bi_asa_dev_dbo.currencyrates
),

/* =======================
   2) ORDERS (LAST 5Y)
   ======================= */
orders_base AS (
    SELECT
        o.id AS order_id,
        o.ordernumber,
        o.shop,
        o.created_date::date AS order_date,
        o.currency_code,
        o.currencyrate AS order_currencyrate
    FROM asa.orders o
    WHERE COALESCE(TRIM(o.testorder),'') = '0'
      AND o.state IN ('ARCHIVED','EXPORTED')
      AND o.created_date::date BETWEEN (SELECT start_date FROM params) AND (SELECT as_of_date FROM params)
),

/* =======================
   3) CIF ORDER -> CUSTOMER MAP
   ======================= */
cif_order_map AS (
    SELECT
        related_ordernumber AS ordernumber,
        external_customerkey AS externalcustomerkey
    FROM (
        SELECT
            related_ordernumber,
            external_customerkey,
            ROW_NUMBER() OVER (
                PARTITION BY related_ordernumber
                ORDER BY event_time DESC
            ) AS rn
        FROM poc_dw.customer_interactions_fact
        WHERE interaction_type = 'order'
          AND related_ordernumber IS NOT NULL
          AND external_customerkey IS NOT NULL
    ) t
    WHERE rn = 1
),

/* =======================
   4) ORDER ITEMS - CORRECTED DISCOUNT CALCULATION
   
   FIX: Use only coupondiscountdistr (already in € amount)
        Removed product_discount and shipping_discount (they are %)
   ======================= */
orderitems_net AS (
    SELECT
        ob.order_id,
        ob.ordernumber,
        ob.shop,
        ob.order_date,
        oi.type,
        COALESCE(oi.quantity, 0)::bigint AS quantity,
        
        -- Net price in EUR (what customer paid)
        CASE
            WHEN COALESCE(ob.order_currencyrate, fxr.currencyrate) IS NULL
              OR COALESCE(ob.order_currencyrate, fxr.currencyrate) = 0
            THEN NULL
            ELSE (COALESCE(oi.nettotalprice, 0) / COALESCE(ob.order_currencyrate, fxr.currencyrate))::decimal(18,6)
        END AS post_discount_net_eur,
        
        -- CORRECTED: Discount amount in EUR (coupon only)
        -- coupondiscountdistr is negative, so we use ABS()
        CASE
            WHEN COALESCE(ob.order_currencyrate, fxr.currencyrate) IS NULL
              OR COALESCE(ob.order_currencyrate, fxr.currencyrate) = 0
            THEN NULL
            ELSE (ABS(COALESCE(oi.coupondiscountdistr, 0)) / COALESCE(ob.order_currencyrate, fxr.currencyrate))::decimal(18,6)
        END AS discount_amt_eur

    FROM orders_base ob
    JOIN asa.orderitems oi ON oi.orderid = ob.order_id
    LEFT JOIN fx fxr
        ON fxr.currencycode = ob.currency_code
       AND fxr.rate_date = ob.order_date
    WHERE oi.type IN ('PRODUCT', 'SHIPPING')  -- Only product and shipping lines
),

/* =======================
   5) ORDER LEVEL AGGREGATION
   ======================= */
order_level AS (
    SELECT
        shop,
        order_id,
        ordernumber,
        order_date,
        SUM(CASE WHEN type = 'PRODUCT' THEN post_discount_net_eur ELSE 0 END) AS product_net,
        SUM(CASE WHEN type = 'SHIPPING' THEN post_discount_net_eur ELSE 0 END) AS shipping_net,
        SUM(COALESCE(discount_amt_eur, 0)) AS discount_amt_eur,
        SUM(CASE WHEN type = 'PRODUCT' THEN quantity ELSE 0 END) AS product_qty
    FROM orderitems_net
    GROUP BY 1, 2, 3, 4
),

/* =======================
   6) PRODUCT UNIT PRICE STATS
   ======================= */
product_lines AS (
    SELECT
        shop,
        ordernumber,
        quantity,
        post_discount_net_eur AS net_amount_eur,
        CASE
            WHEN quantity > 0 AND post_discount_net_eur IS NOT NULL
            THEN (post_discount_net_eur / NULLIF(quantity, 0))::decimal(18,6)
        END AS unit_price_net_eur
    FROM orderitems_net
    WHERE type = 'PRODUCT'
),

unit_price_order AS (
    SELECT
        shop,
        ordernumber,
        AVG(unit_price_net_eur) AS avg_unit_price_order_eur,
        MIN(unit_price_net_eur) AS min_unit_price_order_eur,
        MAX(unit_price_net_eur) AS max_unit_price_order_eur
    FROM product_lines
    GROUP BY 1, 2
),

/* =======================
   7) REFUNDS
   ======================= */
refund_orders AS (
    SELECT DISTINCT r.orderid AS order_id
    FROM asa.refunds r
),

order_level_ref AS (
    SELECT
        ol.*,
        CASE WHEN ro.order_id IS NOT NULL THEN 1 ELSE 0 END AS refunded_flag,
        (COALESCE(ol.product_net, 0) + COALESCE(ol.shipping_net, 0)) AS revenue_net
    FROM order_level ol
    LEFT JOIN refund_orders ro ON ro.order_id = ol.order_id
),

/* =======================
   8) MAP ORDERS -> CUSTOMER
   ======================= */
orders_mapped AS (
    SELECT
        m.externalcustomerkey,
        ol.shop,
        ol.ordernumber,
        ol.order_date,
        ol.revenue_net,
        ol.product_qty,
        ol.discount_amt_eur,
        ol.refunded_flag,
        upo.avg_unit_price_order_eur,
        upo.min_unit_price_order_eur,
        upo.max_unit_price_order_eur
    FROM order_level_ref ol
    JOIN cif_order_map m ON m.ordernumber = ol.ordernumber
    LEFT JOIN unit_price_order upo
        ON upo.shop = ol.shop AND upo.ordernumber = ol.ordernumber
),

/* =======================
   9) CUSTOMER-SHOP AGGREGATES
   ======================= */
cust_shop AS (
    SELECT
        externalcustomerkey,
        shop,
        MIN(order_date) AS first_order_date,
        MAX(order_date) AS last_order_date,
        
        COUNT(DISTINCT ordernumber) AS orders_lifetime,
        COUNT(DISTINCT CASE WHEN order_date >= DATEADD(day, -15, (SELECT as_of_date FROM params)) THEN ordernumber END) AS orders_15d,
        COUNT(DISTINCT CASE WHEN order_date >= DATEADD(day, -30, (SELECT as_of_date FROM params)) THEN ordernumber END) AS orders_30d,
        COUNT(DISTINCT CASE WHEN order_date >= DATEADD(month, -3, (SELECT as_of_date FROM params)) THEN ordernumber END) AS orders_3m,
        COUNT(DISTINCT CASE WHEN order_date >= DATEADD(month, -6, (SELECT as_of_date FROM params)) THEN ordernumber END) AS orders_6m,
        COUNT(DISTINCT CASE WHEN order_date >= DATEADD(month, -12, (SELECT as_of_date FROM params)) THEN ordernumber END) AS orders_12m,
        
        SUM(revenue_net) AS revenue_lifetime,
        SUM(CASE WHEN order_date >= DATEADD(day, -15, (SELECT as_of_date FROM params)) THEN revenue_net END) AS revenue_15d,
        SUM(CASE WHEN order_date >= DATEADD(day, -30, (SELECT as_of_date FROM params)) THEN revenue_net END) AS revenue_30d,
        SUM(CASE WHEN order_date >= DATEADD(month, -6, (SELECT as_of_date FROM params)) THEN revenue_net END) AS revenue_6m,
        SUM(CASE WHEN order_date >= DATEADD(month, -12, (SELECT as_of_date FROM params)) THEN revenue_net END) AS revenue_12m,
        
        SUM(product_qty) AS items_qty_lifetime,
        SUM(CASE WHEN order_date >= DATEADD(month, -6, (SELECT as_of_date FROM params)) THEN product_qty END) AS items_qty_6m,
        SUM(CASE WHEN order_date >= DATEADD(month, -12, (SELECT as_of_date FROM params)) THEN product_qty END) AS items_qty_12m,
        
        -- CORRECTED: Discount aggregations (coupon only)
        SUM(discount_amt_eur) AS discount_abs_lifetime,
        COUNT(DISTINCT CASE WHEN discount_amt_eur > 0 THEN ordernumber END) AS discounted_orders,
        MAX(discount_amt_eur) AS max_discount_single_order,
        
        COUNT(DISTINCT CASE WHEN refunded_flag = 1 THEN ordernumber END) AS refund_orders_lifetime,
        COUNT(DISTINCT CASE WHEN refunded_flag = 1 AND order_date >= DATEADD(month, -3, (SELECT as_of_date FROM params)) THEN ordernumber END) AS refund_orders_3m,
        COUNT(DISTINCT CASE WHEN refunded_flag = 1 AND order_date >= DATEADD(month, -6, (SELECT as_of_date FROM params)) THEN ordernumber END) AS refund_orders_6m,
        COUNT(DISTINCT CASE WHEN refunded_flag = 1 AND order_date >= DATEADD(month, -12, (SELECT as_of_date FROM params)) THEN ordernumber END) AS refund_orders_12m,
        
        AVG(avg_unit_price_order_eur) AS avg_unit_price_lifetime_eur,
        MIN(min_unit_price_order_eur) AS min_unit_price_lifetime_eur,
        MAX(max_unit_price_order_eur) AS max_unit_price_lifetime_eur
    FROM orders_mapped
    GROUP BY 1, 2
),

/* =======================
   10) SHARE OF ITEMS DISCOUNTED
   ======================= */
items_discount_share AS (
    SELECT
        externalcustomerkey,
        shop,
        SUM(CASE WHEN discount_amt_eur > 0 THEN product_qty ELSE 0 END)::decimal(18,6)
            / NULLIF(SUM(product_qty), 0)::decimal(18,6) AS share_of_items_discounted
    FROM orders_mapped
    GROUP BY 1, 2
),

/* =======================
   11) ROLLUP TO CUSTOMER LEVEL
   ======================= */
final AS (
    SELECT
        cs.externalcustomerkey,
        MIN(cs.first_order_date) AS first_order_date,
        MAX(cs.last_order_date) AS last_order_date,
        
        SUM(cs.orders_lifetime) AS orders_lifetime,
        SUM(cs.orders_15d) AS orders_15d,
        SUM(cs.orders_30d) AS orders_30d,
        SUM(cs.orders_3m) AS orders_3m,
        SUM(cs.orders_6m) AS orders_6m,
        SUM(cs.orders_12m) AS orders_12m,
        
        SUM(cs.revenue_lifetime) AS revenue_net_lifetime_eur,
        SUM(cs.revenue_15d) AS revenue_net_15d_eur,
        SUM(cs.revenue_30d) AS revenue_net_30d_eur,
        SUM(cs.revenue_6m) AS revenue_net_6m_eur,
        SUM(cs.revenue_12m) AS revenue_net_12m_eur,
        
        SUM(cs.items_qty_lifetime) AS items_qty_lifetime,
        SUM(cs.items_qty_6m) AS items_qty_6m,
        SUM(cs.items_qty_12m) AS items_qty_12m,
        
        -- CORRECTED: Discount aggregations
        SUM(cs.discount_abs_lifetime) AS discount_abs_lifetime_eur,
        SUM(cs.discounted_orders) AS discounted_orders_lifetime,
        MAX(cs.max_discount_single_order) AS max_discount_single_order,
        
        SUM(cs.refund_orders_lifetime) AS refund_orders_lifetime,
        SUM(cs.refund_orders_3m) AS refund_orders_3m,
        SUM(cs.refund_orders_6m) AS refund_orders_6m,
        SUM(cs.refund_orders_12m) AS refund_orders_12m,
        
        AVG(cs.avg_unit_price_lifetime_eur) AS avg_unit_price_lifetime_eur,
        MIN(cs.min_unit_price_lifetime_eur) AS min_unit_price_lifetime_eur,
        MAX(cs.max_unit_price_lifetime_eur) AS max_unit_price_lifetime_eur,
        
        COUNT(DISTINCT cs.shop) AS shops_included
    FROM cust_shop cs
    GROUP BY cs.externalcustomerkey
),

/* =======================
   12) CUSTOMER ATTRIBUTES
   ======================= */
cust_attrs AS (
    SELECT
        externalcustomerkey,
        MAX(country) AS country,
        MAX(COALESCE(NULLIF(gender, ''), 'Undefined')) AS gender
    FROM dw.dim_customers
    WHERE externalcustomerkey IS NOT NULL
    GROUP BY 1
),

/* =======================
   13) CUSTOMER-LEVEL SHARE OF ITEMS DISCOUNTED
   ======================= */
items_discount_share_cust AS (
    SELECT
        externalcustomerkey,
        AVG(share_of_items_discounted) AS share_of_items_discounted
    FROM items_discount_share
    GROUP BY 1
)

/* =======================
   FINAL SELECT
   ======================= */
SELECT
    f.externalcustomerkey,
    (SELECT as_of_date FROM params) AS as_of_date,
    f.shops_included,
    
    f.first_order_date,
    f.last_order_date,
    DATEDIFF(day, f.last_order_date, (SELECT as_of_date FROM params)) AS days_since_last_order,
    
    f.orders_lifetime,
    f.orders_15d,
    f.orders_30d,
    f.orders_3m,
    f.orders_6m,
    f.orders_12m,
    
    f.revenue_net_lifetime_eur,
    f.revenue_net_15d_eur,
    f.revenue_net_30d_eur,
    f.revenue_net_6m_eur,
    f.revenue_net_12m_eur,
    
    CASE WHEN f.orders_lifetime > 0 THEN f.revenue_net_lifetime_eur / f.orders_lifetime::decimal(18,6) END AS aov_net_lifetime_eur,
    CASE WHEN f.orders_12m > 0 THEN f.revenue_net_12m_eur / f.orders_12m::decimal(18,6) END AS aov_net_12m_eur,
    CASE WHEN f.orders_6m > 0 THEN f.revenue_net_6m_eur / f.orders_6m::decimal(18,6) END AS aov_net_6m_eur,
    
    f.items_qty_lifetime,
    f.items_qty_6m,
    f.items_qty_12m,
    
    CASE WHEN f.orders_lifetime > 0 THEN f.items_qty_lifetime / f.orders_lifetime::decimal(18,6) END AS avg_items_per_order_lifetime,
    CASE WHEN f.orders_12m > 0 THEN f.items_qty_12m / f.orders_12m::decimal(18,6) END AS avg_items_per_order_12m,
    CASE WHEN f.orders_6m > 0 THEN f.items_qty_6m / f.orders_6m::decimal(18,6) END AS avg_items_per_order_6m,
    
    f.discount_abs_lifetime_eur,
    
    -- CORRECTED: Discount rate (now will be between 0 and 1)
    CASE
        WHEN (f.revenue_net_lifetime_eur + f.discount_abs_lifetime_eur) > 0
        THEN f.discount_abs_lifetime_eur / (f.revenue_net_lifetime_eur + f.discount_abs_lifetime_eur)
    END AS discount_rate_lifetime,
    
    CASE WHEN f.orders_lifetime > 0 THEN f.discounted_orders_lifetime::decimal(18,6) / f.orders_lifetime::decimal(18,6) END AS share_of_orders_with_discount,
    
    idsc.share_of_items_discounted,
    
    CASE WHEN f.discounted_orders_lifetime > 0 THEN f.discount_abs_lifetime_eur / f.discounted_orders_lifetime::decimal(18,6) END AS avg_discount_per_order,
    f.max_discount_single_order,
    
    f.avg_unit_price_lifetime_eur,
    f.min_unit_price_lifetime_eur,
    f.max_unit_price_lifetime_eur,
    
    f.refund_orders_lifetime,
    f.refund_orders_12m,
    f.refund_orders_6m,
    f.refund_orders_3m,
    CASE WHEN f.orders_lifetime > 0 THEN f.refund_orders_lifetime::decimal(18,6) / f.orders_lifetime::decimal(18,6) END AS refund_rate_lifetime,
    CASE WHEN f.orders_12m > 0 THEN f.refund_orders_12m::decimal(18,6) / f.orders_12m::decimal(18,6) END AS refund_rate_12m,
    CASE WHEN f.orders_6m > 0 THEN f.refund_orders_6m::decimal(18,6) / f.orders_6m::decimal(18,6) END AS refund_rate_6m,
    CASE WHEN f.orders_3m > 0 THEN f.refund_orders_3m::decimal(18,6) / f.orders_3m::decimal(18,6) END AS refund_rate_3m,
    
    NULL::integer AS account_age_days,
    0::smallint AS registration_flag,
    
    ca.country,
    ca.gender

FROM final f
LEFT JOIN cust_attrs ca ON ca.externalcustomerkey = f.externalcustomerkey
LEFT JOIN items_discount_share_cust idsc ON idsc.externalcustomerkey = f.externalcustomerkey;