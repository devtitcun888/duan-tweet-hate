-- 40_lexicon.sql: Proportion of tweets containing offensive lexicon terms by label
SELECT
    t.label,
    COUNT(DISTINCT t.id) AS n_tweets,
    COUNT(DISTINCT CASE WHEN k.term IS NOT NULL THEN t.id END) AS n_with_lexicon,
    ROUND(
        100.0 * COUNT(DISTINCT CASE WHEN k.term IS NOT NULL THEN t.id END) / COUNT(DISTINCT t.id),
        2
    ) AS pct_with_lexicon
FROM tweets t
LEFT JOIN tokens k0 ON k0.tweet_id = t.id
LEFT JOIN lexicon k ON k.term = k0.token
GROUP BY t.label
ORDER BY t.label;
