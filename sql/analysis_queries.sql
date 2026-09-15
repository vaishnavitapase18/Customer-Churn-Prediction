-- Day 1 SQL Analysis Queries
-- Database: churn_ltv_db
-- Table: customers
--
-- How to use:
-- 1. Open pgAdmin
-- 2. Connect to your local PostgreSQL server
-- 3. Open Query Tool for churn_ltv_db
-- 4. Copy/paste and run each query one at a time

-- ============================================================
-- 1. Total customers
-- ============================================================
SELECT COUNT(*) AS total_customers
FROM customers;

-- ============================================================
-- 2. Total churned customers
-- ============================================================
SELECT COUNT(*) AS total_churned
FROM customers
WHERE "Churn" = 'Yes';

-- ============================================================
-- 3. Total non-churned customers
-- ============================================================
SELECT COUNT(*) AS total_non_churned
FROM customers
WHERE "Churn" = 'No';

-- ============================================================
-- 4. Overall churn percentage
-- ============================================================
SELECT
    ROUND(
        100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS overall_churn_percentage
FROM customers;

-- ============================================================
-- 5. Customers by contract
-- ============================================================
SELECT
    "Contract",
    COUNT(*) AS customer_count
FROM customers
GROUP BY "Contract"
ORDER BY customer_count DESC;

-- ============================================================
-- 6. Churn by contract
-- ============================================================
SELECT
    "Contract",
    "Churn",
    COUNT(*) AS customer_count
FROM customers
GROUP BY "Contract", "Churn"
ORDER BY "Contract", "Churn";

-- ============================================================
-- 7. Churn rate by contract
-- ============================================================
SELECT
    "Contract",
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate_pct
FROM customers
GROUP BY "Contract"
ORDER BY churn_rate_pct DESC;

-- ============================================================
-- 8. Average monthly charges by churn
-- ============================================================
SELECT
    "Churn",
    ROUND(AVG("MonthlyCharges")::numeric, 2) AS avg_monthly_charges
FROM customers
GROUP BY "Churn";

-- ============================================================
-- 9. Average tenure by churn
-- ============================================================
SELECT
    "Churn",
    ROUND(AVG(tenure)::numeric, 2) AS avg_tenure_months
FROM customers
GROUP BY "Churn";

-- ============================================================
-- 10. Churn by internet service
-- ============================================================
SELECT
    "InternetService",
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate_pct
FROM customers
GROUP BY "InternetService"
ORDER BY churn_rate_pct DESC;

-- ============================================================
-- 11. Churn by payment method
-- ============================================================
SELECT
    "PaymentMethod",
    COUNT(*) AS total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate_pct
FROM customers
GROUP BY "PaymentMethod"
ORDER BY churn_rate_pct DESC;

-- ============================================================
-- 12. Customers with high monthly charges
-- -- (top 25% of MonthlyCharges — threshold computed from data)
-- ============================================================
SELECT *
FROM customers
WHERE "MonthlyCharges" >= (
    SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "MonthlyCharges")
    FROM customers
)
ORDER BY "MonthlyCharges" DESC;

-- ============================================================
-- 13. Customers with low tenure
-- -- (tenure of 12 months or less)
-- ============================================================
SELECT *
FROM customers
WHERE tenure <= 12
ORDER BY tenure ASC, "customerID";
