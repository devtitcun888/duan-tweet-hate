-- 50_vi_log_odds.sql: Smoothed log-odds ratio for Vietnamese tokens (HATE vs CLEAN on train split)
WITH c AS (
    SELECT
        token,
        SUM(CASE WHEN label_3 = 'HATE' THEN 1 ELSE 0 END) AS c_hate,
        SUM(CASE WHEN label_3 = 'OFFENSIVE' THEN 1 ELSE 0 END) AS c_off,
        SUM(CASE WHEN label_3 = 'CLEAN' THEN 1 ELSE 0 END) AS c_clean
    FROM vi_tokens
    GROUP BY token
    HAVING (c_hate + c_clean) >= 10
),
tot AS (
    SELECT
        SUM(c_hate) AS t_hate,
        SUM(c_off) AS t_off,
        SUM(c_clean) AS t_clean
    FROM c
)
SELECT
    token,
    c_hate,
    c_off,
    c_clean,
    ROUND(LN((c_hate + 1.0) / (t_hate + 1.0)) - LN((c_clean + 1.0) / (t_clean + 1.0)), 4) AS log_odds_hate_vs_clean,
    ROUND(LN((c_off + 1.0) / (t_off + 1.0)) - LN((c_clean + 1.0) / (t_clean + 1.0)), 4) AS log_odds_off_vs_clean
FROM c, tot
ORDER BY log_odds_hate_vs_clean DESC
LIMIT 60;
