-- 60_vi_cooccur.sql: Words co-occurring with top-20 Vietnamese HATE tokens in HATE comments
SELECT
    b.token AS neighbor,
    COUNT(*) AS cnt,
    COUNT(DISTINCT b.comment_id) AS n_comments
FROM vi_tokens a
JOIN vi_tokens b
    ON a.comment_id = b.comment_id
    AND a.token <> b.token
WHERE a.label_3 = 'HATE'
  AND a.token IN (SELECT token FROM vi_top_hate_tokens LIMIT 20)
GROUP BY b.token
ORDER BY cnt DESC
LIMIT 50;
