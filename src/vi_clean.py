"""Vietnamese text cleaning, teencode normalization, word segmentation, accent stripping, and profanity masking (V2)."""

from __future__ import annotations

import html
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

from pyvi import ViTokenizer

from src.utils import get_project_root

PROJECT_ROOT = get_project_root()
LEXICON_PATH = PROJECT_ROOT / "data" / "lexicon" / "vi_profanity.txt"

try:
    import emoji as _emoji_pkg
except ImportError:
    _emoji_pkg = None

# Vietnamese Teencode dictionary (non-profane abbreviations only, per V2 spec: "KHÔNG sửa từ tục")
TEENCODE: dict[str, str] = {
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hok": "không",
    "hong": "không",
    "hông": "không",
    "hem": "không",
    "dc": "được",
    "đc": "được",
    "dk": "được",
    "vs": "với",
    "j": "gì",
    "gj": "gì",
    "ntn": "như thế nào",
    "bt": "bình thường",
    "bth": "bình thường",
    "đt": "điện thoại",
    "dt": "điện thoại",
    "mk": "mình",
    "mik": "mình",
    "ik": "đi",
    "uk": "ừ",
    "uh": "ừ",
    "uhm": "ừ",
    "ng": "người",
    "ngta": "người ta",
    "nta": "người ta",
    "bh": "bây giờ",
    "h": "giờ",
    "trc": "trước",
    "ms": "mới",
    "cx": "cũng",
    "cug": "cũng",
    "thik": "thích",
    "iu": "yêu",
    "ny": "người yêu",
    "nv": "nhân viên",
    "sp": "sản phẩm",
    "ad": "admin",
    "mn": "mọi người",
    "mng": "mọi người",
    "ae": "anh em",
    "gd": "gia đình",
    "xh": "xã hội",
    "vn": "việt nam",
    "tq": "trung quốc",
    "fb": "facebook",
    "ytb": "youtube",
    "yt": "youtube",
    "cmt": "bình luận",
    "ib": "nhắn tin",
    "rep": "trả lời",
    "auto": "tự động",
    "ok": "ổn",
    "oke": "ổn",
    "r": "rồi",
    "roi": "rồi",
    "wa": "quá",
    "qa": "quá",
    "qá": "quá",
    "lun": "luôn",
    "bik": "biết",
    "bit": "biết",
    "chx": "chưa",
    "ak": "à",
    "ah": "à",
    "z": "vậy",
    "v": "vậy",
    "zậy": "vậy",
    "đag": "đang",
    "dag": "đang",
    "nhìu": "nhiều",
    "nhiu": "nhiều",
}

# Unambiguous multi-word and common token mappings for unaccented Vietnamese (Tiếng Việt không dấu)
NOACCENT_PHRASE_MAP: list[tuple[str, str]] = [
    ("oc cho", "óc_chó"),
    ("mat day", "mất_dạy"),
    ("vo hoc", "vô_học"),
    ("suc vat", "súc_vật"),
    ("rac ruoi", "rác_rưởi"),
    ("khon nan", "khốn_nạn"),
    ("do ngu", "đồ ngu"),
    ("ngu si", "ngu_si"),
    ("ngu dot", "ngu_dốt"),
    ("dan don", "đần_độn"),
    ("ngu xuan", "ngu_xuẩn"),
    ("bien di", "biến đi"),
    ("cut khoi", "cút khỏi"),
    ("cam mom", "câm mồm"),
    ("sua bay", "sủa bậy"),
    ("cho de", "chó_đẻ"),
    ("con di", "con đĩ"),
    ("con pho", "con phò"),
    ("can ba", "cặn_bã"),
    ("ha dang", "hạ_đẳng"),
    ("moi ro", "mọi_rợ"),
    ("nao ngan", "não ngắn"),
    ("khong nao", "không não"),
    ("an bam", "ăn_bám"),
    ("re tien", "rẻ_tiền"),
    ("vo liem si", "vô_liêm_sỉ"),
    ("benh hoan", "bệnh_hoạn"),
    ("bien thai", "biến_thái"),
    ("tam than", "tâm_thần"),
    ("thang dien", "thằng điên"),
    ("con dien", "con điên"),
    ("do dien", "đồ điên"),
    ("dap chet", "đập chết"),
    ("chem chet", "chém chết"),
    ("phan dong", "phản_động"),
    ("ban nuoc", "bán nước"),
    ("bac ky", "bắc_kỳ"),
    ("nam ky", "nam_kỳ"),
    ("trung ky", "trung_kỳ"),
    ("kinh khung", "kinh_khủng"),
    ("viet nam", "việt_nam"),
    ("bong da", "bóng_đá"),
    ("doi tuyen", "đội_tuyển"),
    ("chien thang", "chiến_thắng"),
    ("tuyet voi", "tuyệt_vời"),
    ("chuc mung", "chúc_mừng"),
    ("cam on", "cảm_ơn"),
    ("chia se", "chia_sẻ"),
    ("tai lieu", "tài_liệu"),
    ("hoc may", "học_máy"),
    ("sinh vien", "sinh_viên"),
    ("giao duc", "giáo_dục"),
    ("kinh te", "kinh_tế"),
    ("xa hoi", "xã_hội"),
    ("van hoa", "văn_hóa"),
    ("vung mien", "vùng_miền"),
    ("ton trong", "tôn_trọng"),
    ("khac biet", "khác_biệt"),
    ("hom nay", "hôm_nay"),
    ("moi nguoi", "mọi người"),
    ("ung dung", "ứng_dụng"),
    ("mon an", "món ăn"),
    ("don gian", "đơn_giản"),
    ("ca nha", "cả nhà"),
]

NOACCENT_WORD_MAP: dict[str, str] = {
    "khong": "không",
    "duoc": "được",
    "nguoi": "người",
    "nhung": "những",
    "thang": "thằng",
    "chuyen": "chuyện",
    "nhieu": "nhiều",
    "thich": "thích",
    "nhung": "những",
    "chung": "chúng",
    "truoc": "trước",
    "duong": "đường",
    "suot": "suốt",
    "ngay": "ngày",
    "luon": "luôn",
}

# Distinct unaccented Vietnamese syllables and bigrams for automatic language detection
VI_NOACCENT_SYLLABLES: set[str] = {
    "khong", "duoc", "nguoi", "nhung", "thang", "chuyen", "nhieu", "thich", "truoc",
    "duong", "suot", "ngay", "luon", "chuc", "mung", "tuyen", "bong", "viet", "chien",
    "tuyet", "voi", "phan", "tich", "giao", "duc", "hien", "nay", "cam", "chia",
    "lieu", "hoc", "sinh", "vien", "hom", "troi", "mua", "mang", "theo", "nhe",
    "nha", "ung", "dung", "chay", "tren", "nhanh", "loi", "chung", "ton", "khac",
    "biet", "van", "hoa", "giua", "cac", "vung", "mien", "mon", "nau", "ngon",
    "lam", "minh", "deu", "ngu", "dot", "mat", "day", "kinh", "khung", "cut",
    "khoi", "suc", "vat", "vo", "rac", "ruoi", "xa", "hoi", "con", "di", "re",
    "tien", "qua", "can", "ba", "bon", "moi", "ro", "ha", "dang", "xung", "song",
    "cho", "de", "may", "tao", "dap", "chet", "dan", "nao", "ngan", "chi", "bam",
    "chu", "mom", "lai", "dien", "sua", "bay", "bien", "oc", "lu",
}

EN_COMMON_STOPWORDS: set[str] = {
    "the", "and", "with", "this", "that", "from", "have", "just", "finished",
    "reading", "because", "they", "are", "not", "belong", "into", "their",
    "would", "could", "should", "about", "there", "which", "when", "what",
    "where", "while", "these", "those", "women", "men", "roles", "friends",
    "amazing", "book", "machine", "learning", "leadership", "inferior",
}


def is_unaccented_vietnamese(text: str | None) -> bool:
    """Detect whether an unaccented string (>= 3 words) is Vietnamese."""
    if not text:
        return False
    s = re.sub(r"https?://\S+|www\.\S+|@\w+|[^a-zA-Z\s]", " ", str(text).lower())
    words = [w for w in s.split() if len(w) >= 2]
    if len(words) < 3:
        return False
    en_hits = sum(1 for w in words if w in EN_COMMON_STOPWORDS)
    if en_hits >= 2:
        return False
    joined = " " + " ".join(words) + " "
    phrase_hits = sum(1 for pat, _ in NOACCENT_PHRASE_MAP if f" {pat} " in joined)
    if phrase_hits >= 1:
        return True
    vi_hits = sum(1 for w in words if w in VI_NOACCENT_SYLLABLES or w in TEENCODE)
    return vi_hits >= 2 and (vi_hits / len(words)) >= 0.45


def restore_unaccented_vietnamese(s: str) -> str:
    """Restore common Vietnamese compounds and diacritics when input has no Vietnamese accents."""
    padded = " " + s + " "
    for src_phrase, dst_phrase in NOACCENT_PHRASE_MAP:
        padded = re.sub(r"(?<!\w)" + re.escape(src_phrase) + r"(?!\w)", dst_phrase, padded)
    toks = padded.strip().split()
    out_toks = [NOACCENT_WORD_MAP.get(t, t) for t in toks]
    return " ".join(out_toks)

# Light Vietnamese stopwords list (only used when remove_stopwords=True in V3 ablation;
# preserves negations "không, chẳng, chả, chưa, đừng, chớ" and pronouns "mày, tao, nó, chúng_nó, bọn, lũ, thằng, con")
VI_STOPWORDS_LIGHT: set[str] = {
    "là",
    "và",
    "của",
    "có",
    "những",
    "các",
    "một",
    "trong",
    "được",
    "cho",
    "với",
    "này",
    "đã",
    "khi",
    "từ",
    "như",
    "về",
    "đến",
    "tại",
    "theo",
    "sẽ",
    "đang",
    "cũng",
    "vào",
    "ra",
    "lên",
    "xuống",
    "thì",
    "mà",
    "ở",
    "do",
    "bởi",
    "vì",
    "nếu",
    "nhưng",
    "hoặc",
    "hay",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_REPEAT_CHARS_RE = re.compile(r"([^\W\d_])\1{2,}", flags=re.UNICODE)
_NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def strip_accents(s: str | None) -> str:
    """Remove Vietnamese diacritics (accents) and convert đ/Đ -> d/D (V2 & V5 robustness)."""
    if s is None:
        return ""
    s_str = str(s)
    s_nfd = unicodedata.normalize("NFD", s_str)
    s_no_mn = "".join(c for c in s_nfd if unicodedata.category(c) != "Mn")
    return s_no_mn.replace("đ", "d").replace("Đ", "D")


def vi_clean(
    text: Any,
    segment: bool = True,
    expand_teencode: bool = True,
    remove_stopwords: bool = False,
    handle_emoji: str = "demojize",
) -> str:
    """Clean and optionally word-segment Vietnamese social media text following the 10-step V2 specification.

    Steps:
      1. html.unescape
      2. unicodedata.normalize('NFC')
      3. lowercase
      4. remove URL and @mention
      5. emoji: keep :name: via emoji.demojize or remove
      6. collapse repeated characters >=3 to 2 (đỉnhhhhh -> đỉnhh)
      7. keep Vietnamese diacritics; replace [^\w\s] with space
      8. expand teencode on whole tokens (without modifying profanity)
      9. normalize whitespace
     10. word segmentation via ViTokenizer.tokenize when segment=True
    """
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    s = str(text).strip()
    if not s:
        return ""

    # 1. HTML unescape
    s = html.unescape(html.unescape(s))

    # 2. NFC normalization
    s = unicodedata.normalize("NFC", s)

    # 5. Emoji handling before stripping punctuation
    if _emoji_pkg is not None:
        if handle_emoji == "demojize":
            s = _emoji_pkg.demojize(s, delimiters=(" ", " "))
            s = s.replace("_", " ")
        elif handle_emoji == "remove":
            s = _emoji_pkg.replace_emoji(s, replace=" ")

    # 3. Lowercase
    s = s.lower()

    # 4. Remove URL and @mention
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)

    # Handle common punctuation-split profanity like "đ.m" or "v.l" before stripping punctuation
    s = re.sub(r"\bđ\.m\b", "đm", s)
    s = re.sub(r"\bd\.m\b", "đm", s)
    s = re.sub(r"\bv\.l\b", "vl", s)

    # 6. Collapse repeated letters >=3 -> 2 (e.g., đỉnhhhhh -> đỉnhh)
    s = _REPEAT_CHARS_RE.sub(r"\1\1", s)

    # Replace existing underscores with spaces before tokenization
    s = s.replace("_", " ")

    # 7. Strip non-word characters while preserving all Unicode Vietnamese letters
    s = _NON_WORD_RE.sub(" ", s)

    # 8. Expand teencode on whole tokens
    tokens = s.split()
    if expand_teencode:
        expanded = []
        for tok in tokens:
            if tok in TEENCODE:
                expanded.extend(TEENCODE[tok].split())
            else:
                expanded.append(tok)
        tokens = expanded

    # 9. Normalize whitespace
    s = _MULTI_SPACE_RE.sub(" ", " ".join(tokens)).strip()
    if not s:
        return ""

    # 9b. Restore unambiguous unaccented Vietnamese phrases & words when expand_teencode=True
    if expand_teencode and is_unaccented_vietnamese(s):
        s = restore_unaccented_vietnamese(s)

    # 10. Word segmentation (ViTokenizer joins compound words with '_')
    if segment:
        try:
            s = ViTokenizer.tokenize(s)
        except Exception:
            pass

    if remove_stopwords:
        toks = [t for t in s.split() if t not in VI_STOPWORDS_LIGHT]
        s = " ".join(toks)

    return _MULTI_SPACE_RE.sub(" ", s).strip()


def load_vi_profanity_lexicon(path: str | Path | None = None) -> set[str]:
    """Load Vietnamese profanity lexicon from data/lexicon/vi_profanity.txt."""
    p = Path(path) if path is not None else LEXICON_PATH
    if not p.exists():
        return {"đm", "vl", "vcl", "đéo", "địt", "lồn", "cặc", "đĩ", "ngu", "chó"}
    terms = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        w = unicodedata.normalize("NFC", line.strip().lower())
        if w and not w.startswith("#"):
            terms.add(w)
            # Also add space-separated variant of compound words
            if "_" in w:
                terms.add(w.replace("_", " "))
    return terms


_CACHED_VI_LEXICON: set[str] | None = None


def vi_mask_profanity(text: str | None, lexicon: set[str] | None = None) -> str:
    """Mask Vietnamese profanity terms as first character + '***' (V2 specification)."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    if not s:
        return ""

    global _CACHED_VI_LEXICON
    if lexicon is None:
        if _CACHED_VI_LEXICON is None:
            _CACHED_VI_LEXICON = load_vi_profanity_lexicon()
        lexicon = _CACHED_VI_LEXICON

    if not lexicon:
        return s

    # Sort terms longest-first so compound words like 'óc_chó' or 'vãi_lồn' match before 'chó' or 'lồn'
    sorted_terms = sorted(lexicon, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(w) for w in sorted_terms) + r")(?!\w)",
        flags=re.IGNORECASE | re.UNICODE,
    )

    def _repl(m: re.Match) -> str:
        word = m.group(0)
        return word[0] + "***"

    return pattern.sub(_repl, s)


def compute_teencode_stats(texts: list[str]) -> dict[str, Any]:
    """Compute percentage of comments and tokens normalized by TEENCODE dictionary."""
    total_comments = len(texts)
    comments_with_tc = 0
    total_tokens = 0
    tc_tokens = 0
    tc_counter: dict[str, int] = {}

    for txt in texts:
        s = unicodedata.normalize("NFC", str(txt or "")).lower()
        s = _NON_WORD_RE.sub(" ", s)
        toks = s.split()
        total_tokens += len(toks)
        has_tc = False
        for t in toks:
            if t in TEENCODE:
                tc_tokens += 1
                has_tc = True
                tc_counter[t] = tc_counter.get(t, 0) + 1
        if has_tc:
            comments_with_tc += 1

    top_tc = sorted(tc_counter.items(), key=lambda x: x[1], reverse=True)[:15]
    return {
        "n_comments": total_comments,
        "n_comments_with_teencode": comments_with_tc,
        "pct_comments_with_teencode": round(100.0 * comments_with_tc / max(total_comments, 1), 2),
        "n_total_tokens": total_tokens,
        "n_teencode_tokens": tc_tokens,
        "pct_tokens_teencode": round(100.0 * tc_tokens / max(total_tokens, 1), 2),
        "top_teencode_terms": top_tc,
    }
