-- 20_log_odds.sql: Smoothed log-odds ratio for tokens distinguishing label 1 vs label 0
WITH c AS (
    SELECT
        token,
        SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS c1,
        SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS c0
    FROM tokens
    GROUP BY token
    HAVING (c1 + c0) >= 10
),
tot AS (
    SELECT
        SUM(c1) AS t1,
        SUM(c0) AS t0
    FROM c
)
SELECT
    token,
    c1,
    c0,
    ROUND(LN((c1 + 1.0) / (t1 + 1.0)) - LN((c0 + 1.0) / (t0 + 1.0)), 4) AS log_odds
FROM c, tot
ORDER BY log_odds DESC
LIMIT 50;
