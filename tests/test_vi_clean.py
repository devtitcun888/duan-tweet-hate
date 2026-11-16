"""Unit tests for Vietnamese text cleaning, teencode expansion, accent stripping, and profanity masking (V2 — >=15 cases)."""

import pytest

from src.vi_clean import strip_accents, vi_clean, vi_mask_profanity


def test_profane_punctuation_comment_no_crash():
    out = vi_clean("Đ.M thằng này ngu vl!!!")
    assert "thằng" in out
    assert "ngu" in out
    assert "đm" in out
    assert "vl" in out


def test_unaccented_vietnamese_preserved():
    out = vi_clean("khong biet gi het", segment=False, expand_teencode=False)
    assert out == "khong biet gi het"


def test_vietnamese_diacritics_preserved():
    out = vi_clean("Người Việt Nam yêu chuộng hòa bình", segment=True)
    assert "người" in out
    assert "việt_nam" in out or "việt" in out
    assert "hòa_bình" in out or "bình" in out


def test_teencode_expansion_ko_dc():
    out = vi_clean("ko đc nói như z", segment=False, expand_teencode=True)
    tokens = out.split()
    assert "không" in tokens
    assert "được" in tokens
    assert "vậy" in tokens


def test_teencode_disabled():
    out = vi_clean("ko đc nói", segment=False, expand_teencode=False)
    tokens = out.split()
    assert "ko" in tokens
    assert "đc" in tokens


def test_empty_and_none_inputs():
    assert vi_clean("") == ""
    assert vi_clean("   ") == ""
    assert vi_clean(None) == ""
    assert vi_clean(float("nan")) == ""


def test_pure_emoji_demojize_and_remove():
    out_demo = vi_clean("😡🤬", segment=False, handle_emoji="demojize")
    assert len(out_demo) > 0
    out_rem = vi_clean("😡🤬", segment=False, handle_emoji="remove")
    assert out_rem == ""


def test_url_and_mention_removal():
    out = vi_clean("@user_123 xem link https://vnexpress.net/tin-tuc ngay nhé", segment=False)
    assert "user" not in out
    assert "http" not in out
    assert "vnexpress" not in out
    assert "xem" in out.split()


def test_repeated_character_collapse_vietnamese():
    out = vi_clean("Đỉnhhhhh quáááá luôn", segment=False)
    assert "đỉnhh" in out.split()
    assert "quáá" in out.split()


def test_mixed_english_and_vietnamese():
    out = vi_clean("Livestream trên fb và ytb rất hay", segment=False, expand_teencode=True)
    assert "livestream" in out.split()
    assert "facebook" in out.split()
    assert "youtube" in out.split()


def test_word_segmentation_compound_words():
    out = vi_clean("Học sinh sinh viên trường đại học", segment=True)
    assert "_" in out


def test_nfc_normalization_consistency():
    # NFD decomposed string vs NFC composed string must yield identical output
    import unicodedata

    nfd_str = unicodedata.normalize("NFD", "tiếng việt có dấu")
    nfc_str = unicodedata.normalize("NFC", "tiếng việt có dấu")
    assert vi_clean(nfd_str) == vi_clean(nfc_str)


def test_strip_accents_function():
    s = "Đội tuyển bóng đá Việt Nam chiến thắng"
    stripped = strip_accents(s)
    assert stripped == "Doi tuyen bong da Viet Nam chien thang"


def test_vi_mask_profanity_masks_first_char_plus_stars():
    masked = vi_mask_profanity("Thằng này ngu vcl và mất_dạy")
    assert "n***" in masked
    assert "v***" in masked
    assert "m***" in masked
    assert "ngu" not in masked


def test_negation_preserved_by_default():
    out = vi_clean("Mày đừng có nói không đúng sự thật", segment=True, remove_stopwords=False)
    assert "đừng" in out
    assert "không" in out
    assert "mày" in out
