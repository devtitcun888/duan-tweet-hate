-- 00_schema.sql: Database schema for Tweet Hate Speech SQL Analysis (P3)
DROP TABLE IF EXISTS tweets;
DROP TABLE IF EXISTS tokens;
DROP TABLE IF EXISTS bigrams;
DROP TABLE IF EXISTS assigned;
DROP TABLE IF EXISTS lexicon;
DROP TABLE IF EXISTS top_hate_tokens;

CREATE TABLE IF NOT EXISTS tweets(
    id INTEGER PRIMARY KEY,
    text TEXT,
    label INTEGER,
    n_chars INT,
    n_words INT
);

CREATE TABLE IF NOT EXISTS tokens(
    tweet_id INT,
    label INT,
    token TEXT
);

CREATE TABLE IF NOT EXISTS bigrams(
    tweet_id INT,
    label INT,
    bigram TEXT
);

CREATE TABLE IF NOT EXISTS assigned(
    id INTEGER PRIMARY KEY,
    text TEXT,
    sentiment INT
);

CREATE TABLE IF NOT EXISTS lexicon(
    term TEXT PRIMARY KEY,
    category TEXT
);

CREATE TABLE IF NOT EXISTS top_hate_tokens(
    token TEXT PRIMARY KEY,
    c1 INT,
    c0 INT,
    log_odds REAL
);

CREATE INDEX IF NOT EXISTS ix_tokens_tok ON tokens(token, label);
CREATE INDEX IF NOT EXISTS ix_tokens_tweet ON tokens(tweet_id);
CREATE INDEX IF NOT EXISTS ix_bigrams_bg ON bigrams(bigram, label);
