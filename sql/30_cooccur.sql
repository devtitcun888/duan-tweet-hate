-- 30_cooccur.sql: Words co-occurring with top-20 hate tokens in label=1 tweets
SELECT
    b.token AS neighbor,
    COUNT(*) AS cnt,
    COUNT(DISTINCT b.tweet_id) AS n_tweets
FROM tokens a
JOIN tokens b
    ON a.tweet_id = b.tweet_id
    AND a.token <> b.token
WHERE a.label = 1
  AND a.token IN (SELECT token FROM top_hate_tokens LIMIT 20)
GROUP BY b.token
ORDER BY cnt DESC
LIMIT 50;
