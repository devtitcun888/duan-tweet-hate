-- 10_token_freq.sql: Token frequency by label
SELECT
    token,
    label,
    COUNT(*) AS cnt
FROM tokens
GROUP BY token, label
ORDER BY cnt DESC;
