"""Unit tests for src/clean.py (P4.1 — minimum 12 test cases)."""

import math
import pytest

from src.clean import clean_text, extract_meta_features, PRESERVED_WORDS


def test_url_and_mention_removal():
    out = clean_text("@user check https://t.co/x NOW!!!")
    assert "user" not in out.split()
    assert "http" not in out
    assert "check" in out.split()
    assert "now" in out.split()


def test_hashtag_and_character_lengthening():
    out = clean_text("#BlackLivesMatter sooooo true")
    assert "blacklivesmatter" in out.split()
    assert "soo" in out.split()
    assert "sooooo" not in out.split()


def test_rt_removal():
    out = clean_text("RT @user: hello world")
    assert "rt" not in out.split()
    assert "user" not in out.split()
    assert "hello" in out.split()


def test_preserve_negation_and_pronouns_with_stopwords_removed():
    out = clean_text("You are NOT welcome here", remove_stopwords=True)
    tokens = out.split()
    assert "you" in tokens
    assert "not" in tokens
    assert "welcome" in tokens


def test_preserve_gender_and_group_pronouns():
    out = clean_text("She and they told women and men no never", remove_stopwords=True)
    tokens = out.split()
    for word in ["she", "they", "woman", "men", "no", "never"]:
        assert word in tokens or "women" in tokens


def test_html_entities_unescaped():
    out = clean_text("love &amp; peace &lt;3 &gt; hate")
    assert "amp" not in out.split()
    assert "lt" not in out.split()
    assert "gt" not in out.split()
    assert "love" in out.split()
    assert "peace" in out.split()


def test_empty_and_none_inputs():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_numbers_and_punctuation_only():
    assert clean_text("1234567890 !@#$%^&*()") == ""


def test_single_letter_tokens_filtered_except_i_and_a():
    out = clean_text("I have a cat b c d x z", remove_stopwords=False)
    tokens = out.split()
    assert "i" in tokens
    assert "a" in tokens
    for ch in ["b", "c", "d", "x", "z"]:
        assert ch not in tokens


def test_intra_word_apostrophe_preserved():
    out = clean_text("I don't think it's fair", remove_stopwords=False, lemmatize=False)
    tokens = out.split()
    assert "don't" in tokens
    assert "it's" in tokens


def test_lemmatization():
    out = clean_text("dogs running catsBoxes", remove_stopwords=False, lemmatize=True)
    tokens = out.split()
    assert "dog" in tokens
    assert "run" in tokens


def test_www_url_removal():
    out = clean_text("Visit www.example.com/test?id=1 today")
    assert "www" not in out
    assert "example" not in out
    assert "visit" in out.split()


def test_extract_meta_features():
    meta = extract_meta_features("@user1 @user2 WOW!! Really? #tag1 #tag2 https://x.com")
    assert meta["n_mentions"] == 2
    assert meta["n_hashtags"] == 2
    assert meta["n_urls"] == 1
    assert meta["n_exclaim"] == 2
    assert meta["n_question"] == 1
    assert meta["caps_ratio"] > 0.0
    assert meta["n_words"] == 7
