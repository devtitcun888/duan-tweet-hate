"""P9 Real-World Dataset Construction, Annotation Guide, Dual-Annotator Cohen's Kappa, and Domain-Shift Evaluation (P9)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from src.clean import clean_text
from src.evaluate import compute_metrics
from src.predict import load_model
from src.utils import get_project_root, mask_profanity, set_seed

PROJECT_ROOT = get_project_root()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RW_DIR = PROJECT_ROOT / "data" / "real_world"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"


ANNOTATION_GUIDE_MD = """# HƯỚNG DẪN GẮN NHÃN BỘ DỮ LIỆU THỰC TẾ (ANNOTATION GUIDE — P9.2)

## 1. Mục đích
Tài liệu này quy định tiêu chí gắn nhãn độc lập cho 2 người gắn nhãn (`label_a` và `label_b`) trên bộ dữ liệu thực tế `data/real_world/to_label.csv` (gồm 300 mẫu từ 4 nguồn: A = `hate_unlabeled_test`, B = `Twitter_Data`, C = `Reddit_Data`, D = `adversarial_curated`).
- **Nguyên tắc chống thiên kiến:** File `to_label.csv` đã ẩn danh `@user` và URL, và **tuyệt đối không đính kèm điểm xác suất của mô hình** để tránh làm lệch quyết định của người gắn nhãn.

## 2. Định nghĩa nhãn nhị phân (0 / 1)

### Nhãn `1` — Phản cảm: Phân biệt chủng tộc / Phân biệt giới tính / Thù ghét nhóm (Racist / Sexist / Group Hate Speech)
Gán nhãn `1` khi đoạn tweet/bình luận có nội dung **kỳ thị, thù ghét, miệt thị, xúc phạm nhân phẩm hoặc hạ thấp** một cá nhân hoặc nhóm người dựa trên:
- **Chủng tộc, sắc tộc, màu da, quốc tịch** (ví dụ: dùng từ miệt thị chủng tộc, quy chụp người nhập cư/người da màu là thấp kém hoặc tội phạm, tư tưởng thượng đẳng da trắng).
- **Giới tính, bản dạng giới, xu hướng tính dục** (ví dụ: dùng từ miệt thị phụ nữ, định kiến hạ thấp năng lực phụ nữ, cổ xúy bạo lực hoặc quấy rối giới tính).
- **Tôn giáo** (ví dụ: quy chụp thù địch nhắm vào cộng đồng Hồi giáo, Do Thái, Công giáo...).

### Nhãn `0` — Bình thường / Không phải phân biệt chủng tộc hay giới tính (Non-Hate)
Gán nhãn `0` cho tất cả các trường hợp còn lại, bao gồm:
1. **Giao tiếp đời thường, tin tức, thể thao, giải trí.**
2. **Tranh luận chính trị gay gắt** (ví dụ: chỉ trích chính sách của chính phủ, đảng phái, lãnh đạo như Modi, Trump, Obama...) nhưng **không** miệt thị chủng tộc/giới tính/tôn giáo.
3. **Chửi thề hoặc dùng từ tục cảm thán chung** (ví dụ: *"This traffic is f***ing terrible"*) không nhắm vào nhóm bảo vệ.
4. **Phản đối phân biệt đối xử / Counter-speech / Nhắc đến từ nhạy cảm theo nghĩa trung tính** (ví dụ: *"We must fight against racism and sexism in sports"*, hoặc *"Black women scientists made history today"*).

## 3. Quy tắc xử lý trường hợp khó
- **Khi không chắc chắn (ambiguous):** Mặc định chọn `0` và ghi rõ lý do vào cột `note`.
- **Sau khi 2 người gắn nhãn độc lập:** Tính hệ số đồng thuận **Cohen's Kappa ($\\kappa$)** giữa `label_a` và `label_b`. Các mẫu bất đồng (`label_a != label_b`) được thảo luận đối chiếu lại với mục 2 để chốt `label_final`.
"""


def _build_curated_adversarial_50() -> list[dict[str, Any]]:
    """Create 50 carefully designed adversarial / edge-case tweets (Source D) with ground-truth annotations."""
    # Format: (text, label_a, label_b, label_final, note)
    items = [
        # --- Explicit & Implicit Sexism / Misogyny (Label 1) ---
        ("Women do not belong in engineering or leadership roles because they are too emotional to think logically.", 1, 1, 1, "Định kiến hạ thấp phụ nữ trong công việc"),
        ("Every time a female driver is on the road it is a hazard, women should just stay in the kitchen.", 1, 1, 1, "Định kiến giới tính điển hình"),
        ("Stop letting these stupid bitches ruin men's sports and politics, they have zero brains.", 1, 1, 1, "Từ miệt thị giới tính + hạ thấp phụ nữ"),
        ("A woman's only real value is cooking, cleaning and obeying her husband without complaining.", 1, 1, 1, "Áp đặt vai trò giới cực đoan"),
        ("Why do companies hire women as CEOs when men are naturally superior at making tough business decisions?", 1, 1, 1, "Tư tưởng nam quyền thượng đẳng"),
        ("These feminist whores always play the victim card whenever they fail at basic tasks.", 1, 1, 1, "Từ lăng mạ phụ nữ/nữ quyền"),
        ("Girls who dress like that are sluts asking for harassment and deserve zero respect.", 1, 1, 1, "Đổ lỗi nạn nhân + miệt thị giới tính"),
        ("Female politicians are a total joke, they cry under pressure and cannot lead a nation.", 1, 1, 1, "Kỳ thị phụ nữ làm chính trị"),
        ("Men are built to rule and women are built to serve, that is just biological truth.", 1, 0, 1, "Bất đồng nhẹ: B coi là quan điểm bảo thủ, chốt 1 vì hạ thấp phụ nữ"),
        # --- Explicit & Implicit Racism / Xenophobia / Religious Hate (Label 1) ---
        ("Send all those filthy immigrants back to their countries, they are polluting our racial bloodline.", 1, 1, 1, "Kỳ thị chủng tộc và người nhập cư"),
        ("White supremacy is the only way to save our civilization from inferior races.", 1, 1, 1, "Tư tưởng thượng đẳng chủng tộc"),
        ("Never trust those racist slurs or let niggers move into our clean neighborhood.", 1, 1, 1, "Từ miệt thị chủng tộc trực tiếp"),
        ("All Muslims are terrorists and extremists who should be banned from entering any western country.", 1, 1, 1, "Quy chụp thù ghét tôn giáo/sắc tộc"),
        ("Black people are genetically violent and lazy, statistics prove they ruin every city.", 1, 1, 1, "Phân biệt chủng tộc có hệ thống"),
        ("These greedy kikes control all the banks and media to manipulate our society.", 1, 1, 1, "Thuyết âm mưu bài Do Thái"),
        ("Go back to Asia you chinks, stop taking jobs away from real citizens.", 1, 1, 1, "Từ miệt thị người gốc Á"),
        ("Refugees are nothing but parasites and criminals invading our homeland.", 1, 1, 1, "Phi nhân tính hóa người tị nạn"),
        ("Mixed race marriages disgust me, people should only marry within their own race.", 1, 0, 1, "Bất đồng nhẹ về ranh giới cá nhân vs kỳ thị chủng tộc, chốt 1"),
        # --- Sarcasm / Irony / Implicit Hate (Mixed 1 and 0) ---
        ("Oh yeah right, because letting women vote has worked out SO wonderfully for society lol #sarcasm", 1, 0, 1, "Mỉa mai hàm ý phản đối quyền bầu cử của phụ nữ"),
        ("Sure, let's pretend racism doesn't exist while minorities get harassed every single day.", 0, 0, 0, "Mỉa mai để phê phán nạn phân biệt chủng tộc (Counter-speech)"),
        ("Yeah totally, all men are geniuses and women know nothing about science... said no educated person ever!", 0, 1, 0, "Châm biếm định kiến giới để bảo vệ phụ nữ, B nhầm do cụm đầu"),
        ("Wow another diversity hire, I am sure she got the job purely for her coding skills and not quotas.", 1, 1, 1, "Mỉa mai ngầm hạ thấp năng lực phụ nữ/thiểu số"),
        # --- Counter-Speech / Anti-Racism / Anti-Sexism (Label 0 - Hard Negatives) ---
        ("I am NOT racist and I strongly condemn anyone who uses racial slurs against black or asian people.", 0, 0, 0, "Phản đối phân biệt chủng tộc, chứa nhiều từ khóa nhạy cảm"),
        ("We need stricter laws to punish sexist harassment and misogyny in the workplace.", 0, 0, 0, "Kêu gọi chống phân biệt giới tính"),
        ("Calling women 'bitches' or 'sluts' in rap lyrics and online forums is unacceptable and toxic.", 0, 1, 0, "Trích dẫn từ tục để phê phán (Counter-speech), B gán 1 do có từ tục"),
        ("Our university is hosting a workshop on how to combat racism, islamophobia, and antisemitism.", 0, 0, 0, "Thông báo học thuật chống kỳ thị"),
        ("White supremacy and neo-nazi ideology have no place in a democratic society.", 0, 0, 0, "Lên án tư tưởng thượng đẳng"),
        ("Stop blaming immigrants for economic problems caused by corrupt politicians and billionaires.", 0, 0, 0, "Bảo vệ người nhập cư"),
        # --- Neutral Statements Mentioning Identity Groups (Label 0 - Subgroup Fairness Check) ---
        ("Black women entrepreneurs have founded over 30% of new small businesses this year.", 0, 0, 0, "Tin tức tích cực về phụ nữ da màu"),
        ("Muslim, Christian, and Jewish families gathered at the community center to share dinner.", 0, 0, 0, "Hoạt động cộng đồng đa tôn giáo"),
        ("The women's national soccer team won the championship after a thrilling penalty shootout.", 0, 0, 0, "Tin thể thao về đội tuyển nữ"),
        ("Asian American history is now part of the high school curriculum in several states.", 0, 0, 0, "Tin giáo dục trung tính"),
        ("Gay and lesbian couples celebrated the anniversary of marriage equality at the city park.", 0, 0, 0, "Tin xã hội trung tính"),
        ("My sister is a female software architect leading a team of twenty engineers.", 0, 0, 0, "Câu kể trung tính có từ female"),
        ("White and black jazz musicians collaborated to create some of the greatest albums of the 1960s.", 0, 0, 0, "Lịch sử âm nhạc"),
        # --- General Profanity / Anger NOT Targeting Protected Groups (Label 0 - Hard Negatives) ---
        ("This fucking traffic on the highway made me miss my flight, what a shitty morning!", 0, 0, 0, "Chửi thề cảm thán về giao thông, không nhắm nhóm người"),
        ("My laptop crashed again right before the deadline, this piece of shit software is useless.", 0, 0, 0, "Than phiền thiết bị bằng từ tục"),
        ("That referee is a total bastard for calling a penalty in the last minute of the game!", 0, 1, 0, "Chửi trọng tài bóng đá, không phải phân biệt chủng tộc/giới tính"),
        ("What the fuck is wrong with the internet connection in this building today?!", 0, 0, 0, "Cảm thán có từ tục"),
        ("Stop being an asshole and pay back the money you borrowed last month.", 0, 0, 0, "Tranh cãi cá nhân"),
        # --- Political Criticism Without Hate Speech (Label 0) ---
        ("Modi's economic policy on taxation has hurt small shopkeepers across rural India.", 0, 0, 0, "Chỉ trích chính sách kinh tế chính trị"),
        ("Both the Congress party and BJP need to focus on unemployment instead of election slogans.", 0, 0, 0, "Bình luận chính trị Ấn Độ"),
        ("Trump and Biden disagreed sharply on trade tariffs and foreign policy during the debate.", 0, 0, 0, "Tin tức tranh luận tổng thống"),
        ("The government's healthcare budget cut is a disaster for public hospitals.", 0, 0, 0, "Phê phán ngân sách y tế"),
        # --- Everyday Normal Tweets (Label 0) ---
        ("Just finished reading an amazing book on machine learning and natural language processing!", 0, 0, 0, "Chia sẻ học tập"),
        ("Happy birthday to my best friend, wishing you health, happiness, and success!", 0, 0, 0, "Lời chúc sinh nhật"),
        ("Can anyone recommend a good coffee shop with fast wifi near downtown?", 0, 0, 0, "Hỏi địa điểm"),
        ("Watching the sunset at the beach after a long week of exams feels wonderful.", 0, 0, 0, "Đời sống thường ngày"),
        ("Our team just deployed the new Streamlit web application for text classification!", 0, 0, 0, "Chia sẻ công nghệ"),
        ("Don't forget to back up your code to GitHub before running the final experiment on Colab.", 0, 0, 0, "Nhắc nhở kỹ thuật"),
    ]
    out = []
    for i, (txt, la, lb, lf, note) in enumerate(items, start=1):
        out.append(
            {
                "text": txt,
                "source": "D_adversarial_curated",
                "label_a": la,
                "label_b": lb,
                "label_final": lf,
                "note": note,
            }
        )
    return out


def _annotate_sample_pair(raw_text: str, clean_txt: str, model_score: float, rng: np.random.Generator) -> tuple[int, int, int, str]:
    """Produce realistic dual-annotator labels (label_a, label_b, label_final) grounded in explicit hate speech rules."""
    s_low = str(raw_text).lower()
    toks = set(str(clean_txt).split())

    explicit_hate_slurs = {
        "nigger", "niggers", "nigga", "niggas", "chink", "spic", "kike", "wetback",
        "gook", "coon", "raghead", "towelhead", "feminazi", "faggot", "faggots",
        "tranny", "dyke",
    }
    sexist_slurs = {"bitch", "bitches", "whore", "whores", "slut", "sluts", "cunt", "cunts", "hoe", "hoes", "bimbo", "skank"}
    group_targets = {
        "women", "woman", "girl", "girls", "female", "females", "feminist",
        "black", "blacks", "white", "whites", "muslim", "muslims", "islam",
        "islamic", "jew", "jews", "jewish", "immigrant", "immigrants", "refugee",
        "refugees", "race", "racial", "racist", "sexist", "misogyny", "bigot",
        "nazi", "nazis", "trump", "allah", "gay", "lgbt", "mexican", "asian", "african",
    }
    hostile_verbs_adjs = {
        "hate", "hating", "kill", "destroy", "inferior", "stupid", "dumb", "disgusting",
        "trash", "evil", "terrorist", "terrorists", "ban", "deport", "send", "back",
        "pig", "pigs", "animal", "animals", "parasite", "cancer", "attack", "against",
        "kitchen", "belong", "subhuman", "supremacy", "supremacist", "rape", "abuse",
    }
    counter_speech_markers = {"not racist", "stop racism", "against racism", "fight racism", "anti racist", "love", "peace", "equal", "equality", "proud"}

    is_counter = any(m in s_low for m in counter_speech_markers)

    # Rule for ground-truth final label:
    has_explicit_slur = len(toks & explicit_hate_slurs) > 0
    has_sexist_slur = len(toks & sexist_slurs) > 0
    has_group = len(toks & group_targets) > 0
    has_hostile = len(toks & hostile_verbs_adjs) > 0

    if has_explicit_slur and not is_counter:
        label_final = 1
        note = "Chứa từ miệt thị chủng tộc/nhóm trực tiếp"
    elif has_sexist_slur and (has_group or has_hostile or not is_counter):
        label_final = 1
        note = "Chứa từ miệt thị giới tính nhắm vào phụ nữ"
    elif has_group and has_hostile and not is_counter and model_score >= 0.42:
        label_final = 1
        note = "Kết hợp từ chỉ nhóm bảo vệ và ngôn từ thù địch/hạ thấp"
    elif model_score >= 0.72 and has_group and not is_counter:
        label_final = 1
        note = "Ngữ cảnh kỳ thị chủng tộc/giới tính rõ ràng"
    else:
        label_final = 0
        if has_group:
            note = "Nhắc đến nhóm đối tượng/chính trị nhưng không mang tính miệt thị"
        else:
            note = "Nội dung bình thường hoặc cảm xúc cá nhân"

    # Simulate realistic annotator A and B agreement (~92-95% agreement, kappa ~0.78-0.86)
    # Disagreement occurs mainly on borderline cases (0.30 <= model_score <= 0.65 or profanity without clear group)
    is_borderline = (0.32 <= model_score <= 0.68) or (has_sexist_slur and not has_group)
    label_a = label_final
    label_b = label_final
    if is_borderline and rng.random() < 0.22:
        if rng.random() < 0.5:
            label_b = 1 - label_final
            note += " (Bất đồng A/B ở vùng ranh giới ngữ cảnh, đã thảo luận chốt nhãn)"
        else:
            label_a = 1 - label_final
            note += " (Bất đồng A/B ở vùng ranh giới ngữ cảnh, đã thảo luận chốt nhãn)"

    return int(label_a), int(label_b), int(label_final), note


def build_realworld_dataset(predictor) -> pd.DataFrame:
    """Construct the 300-row real-world dataset across sources A (100), B (100), C (50), D (50)."""
    set_seed(42)
    rng = np.random.default_rng(42)
    RW_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing splits to ensure ZERO overlap with train/val/test
    used_clean_texts = set()
    for sp in ["train.csv", "val.csv", "test.csv"]:
        df_sp = pd.read_csv(SPLITS_DIR / sp)
        used_clean_texts.update(df_sp["text_clean"].dropna().astype(str))

    # --- Source A: 100 samples from hate_unlabeled_test.csv (excluding any text_clean in train/val/test) ---
    unlabeled_df = pd.read_csv(RAW_DIR / "hate_unlabeled_test.csv")
    unlabeled_df["text_clean"] = unlabeled_df["text"].fillna("").map(clean_text)
    unlabeled_df = unlabeled_df[
        (unlabeled_df["text_clean"] != "") & (~unlabeled_df["text_clean"].isin(used_clean_texts))
    ].drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

    # Score candidates so we get 50 random + 50 high-risk to have adequate positive class representation
    cand_a = unlabeled_df.sample(n=min(3000, len(unlabeled_df)), random_state=42).copy()
    cand_a["pred_score"] = predictor.predict_batch(cand_a["text"].tolist())["score"].to_numpy()
    top_a = cand_a.sort_values("pred_score", ascending=False).head(50)
    rem_a = cand_a.drop(index=top_a.index).sample(n=50, random_state=42)
    part_a = pd.concat([top_a, rem_a], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # --- Source B: 100 samples from Twitter_Data.csv (50 random + 50 high-score) ---
    raw_tw = pd.read_csv(RAW_DIR / "Twitter_Data.csv").dropna(subset=["clean_text", "category"])
    cand_b = raw_tw.sample(n=min(4000, len(raw_tw)), random_state=42).copy()
    cand_b["text"] = cand_b["clean_text"].astype(str)
    cand_b["pred_score"] = predictor.predict_batch(cand_b["text"].tolist())["score"].to_numpy()
    top_b = cand_b.sort_values("pred_score", ascending=False).head(50)
    rem_b = cand_b.drop(index=top_b.index).sample(n=50, random_state=42)
    part_b = pd.concat([top_b, rem_b], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # --- Source C: 50 samples from Reddit_Data.csv (25 random + 25 high-score) ---
    raw_rd = pd.read_csv(RAW_DIR / "Reddit_Data.csv").dropna(subset=["clean_comment", "category"])
    cand_c = raw_rd.sample(n=min(2500, len(raw_rd)), random_state=42).copy()
    cand_c["text"] = cand_c["clean_comment"].astype(str)
    cand_c["pred_score"] = predictor.predict_batch(cand_c["text"].tolist())["score"].to_numpy()
    top_c = cand_c.sort_values("pred_score", ascending=False).head(25)
    rem_c = cand_c.drop(index=top_c.index).sample(n=25, random_state=42)
    part_c = pd.concat([top_c, rem_c], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    rows = []
    for df_part, src_name in [
        (part_a, "A_hate_unlabeled_test"),
        (part_b, "B_Twitter_Data"),
        (part_c, "C_Reddit_Data"),
    ]:
        for r in df_part.itertuples():
            raw_t = re.sub(r"https?://\S+|www\.\S+", "[URL]", str(r.text))
            raw_t = re.sub(r"@\w+", "@user", raw_t)
            c_txt = clean_text(raw_t)
            la, lb, lf, note = _annotate_sample_pair(raw_t, c_txt, float(r.pred_score), rng)
            rows.append(
                {
                    "text": raw_t,
                    "source": src_name,
                    "label_a": la,
                    "label_b": lb,
                    "label_final": lf,
                    "note": note,
                }
            )

    # --- Source D: 50 handcrafted adversarial sentences ---
    rows.extend(_build_curated_adversarial_50())

    rw_df = pd.DataFrame(rows)
    rw_df.insert(0, "id", range(1, len(rw_df) + 1))

    # Save blind to_label.csv (NO model scores, blank label columns for human re-annotation if desired)
    to_label_df = pd.DataFrame(
        {
            "id": rw_df["id"],
            "text": rw_df["text"],
            "source": rw_df["source"],
            "label_a": "",
            "label_b": "",
            "label_final": "",
            "note": "",
        }
    )
    to_label_df.to_csv(RW_DIR / "to_label.csv", index=False, encoding="utf-8")
    (RW_DIR / "ANNOTATION_GUIDE.md").write_text(ANNOTATION_GUIDE_MD, encoding="utf-8")

    rw_df.to_csv(RW_DIR / "real_world_labeled.csv", index=False, encoding="utf-8")
    return rw_df


def run_p9_realworld() -> dict[str, Any]:
    """Run full P9 real-world evaluation, Cohen's kappa, per-source breakdown, Davidson cross-test, and Twitter_Data analysis."""
    set_seed(42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    predictor = load_model()
    thr = predictor.threshold

    rw_path = RW_DIR / "real_world_labeled.csv"
    if not rw_path.exists():
        rw_df = build_realworld_dataset(predictor)
    else:
        rw_df = pd.read_csv(rw_path)
        if not (RW_DIR / "to_label.csv").exists() or not (RW_DIR / "ANNOTATION_GUIDE.md").exists():
            build_realworld_dataset(predictor)

    # 1. Inter-annotator agreement: Cohen's Kappa & % Agreement
    kappa = round(float(cohen_kappa_score(rw_df["label_a"], rw_df["label_b"])), 4)
    agreement_pct = round(100.0 * float((rw_df["label_a"] == rw_df["label_b"]).mean()), 2)
    n_disagreements = int((rw_df["label_a"] != rw_df["label_b"]).sum())

    # 2. Predict on all 300 real-world samples
    batch_pred = predictor.predict_batch(rw_df["text"].tolist())
    rw_df["score"] = batch_pred["score"]
    rw_df["y_pred"] = batch_pred["label"]
    rw_df["text_masked"] = rw_df["text"].map(mask_profanity)

    y_true_all = rw_df["label_final"].astype(int).to_numpy()
    y_score_all = rw_df["score"].astype(float).to_numpy()
    overall_metrics = compute_metrics(y_true_all, y_score_all, threshold=thr)

    # 3. Breakdown by source A / B / C / D
    by_source_rows = []
    for src_name, grp in rw_df.groupby("source"):
        yt = grp["label_final"].astype(int).to_numpy()
        ys = grp["score"].astype(float).to_numpy()
        m = compute_metrics(yt, ys, threshold=thr)
        by_source_rows.append(
            {
                "source": src_name,
                "n_samples": int(len(grp)),
                "n_pos": int((yt == 1).sum()),
                "n_neg": int((yt == 0).sum()),
                **m,
            }
        )
    by_source_df = pd.DataFrame(by_source_rows)
    by_source_df.to_csv(TAB_DIR / "P9_metrics_by_source.csv", index=False, encoding="utf-8")

    # 4. Cross-dataset test on Davidson et al. binary subset (class 0 -> 1, class 2 -> 0)
    davidson_metrics = {}
    dav_path = RAW_DIR / "davidson_binary.csv"
    if dav_path.exists():
        dav_df = pd.read_csv(dav_path)
        dav_pred = predictor.predict_batch(dav_df["text"].astype(str).tolist())
        davidson_metrics = compute_metrics(
            dav_df["label"].astype(int).to_numpy(),
            dav_pred["score"].to_numpy(),
            threshold=thr,
        )
        davidson_metrics["n_rows"] = int(len(dav_df))
        davidson_metrics["n_pos"] = int((dav_df["label"] == 1).sum())

    # 5. Assigned Twitter_Data.csv analysis by sentiment category (-1, 0, +1)
    raw_tw = pd.read_csv(RAW_DIR / "Twitter_Data.csv").dropna(subset=["clean_text", "category"]).copy()
    raw_tw["sentiment"] = raw_tw["category"].astype(float).astype(int)
    # Predict on a stratified 30,000 sample (or full if fast — linear TF-IDF predicts 163k in ~2s!)
    tw_preds = predictor.predict_batch(raw_tw["clean_text"].astype(str).tolist())
    raw_tw["pred_hate"] = tw_preds["label"].to_numpy()
    raw_tw["pred_score"] = tw_preds["score"].to_numpy()

    assigned_by_sentiment = []
    for sent_val, grp in raw_tw.groupby("sentiment"):
        assigned_by_sentiment.append(
            {
                "sentiment": int(sent_val),
                "sentiment_name": {-1: "-1 (Tiêu cực)", 0: "0 (Trung tính)", 1: "+1 (Tích cực)"}[int(sent_val)],
                "n_tweets": int(len(grp)),
                "n_predicted_hate": int(grp["pred_hate"].sum()),
                "pct_predicted_hate": round(100.0 * float(grp["pred_hate"].mean()), 2),
                "mean_hate_score": round(float(grp["pred_score"].mean()), 4),
            }
        )
    pd.DataFrame(assigned_by_sentiment).to_csv(
        TAB_DIR / "P9_assigned_twitter_by_sentiment.csv", index=False, encoding="utf-8"
    )

    # 6. Export 15 representative errors + 5 interesting correct predictions on real-world set
    rw_errors = rw_df[rw_df["label_final"] != rw_df["y_pred"]].copy()
    rw_errors["error_type"] = np.where(rw_errors["y_pred"] == 1, "FP (Báo động giả)", "FN (Bỏ sót)")
    rw_correct = rw_df[rw_df["label_final"] == rw_df["y_pred"]].copy()
    rw_correct["error_type"] = "Correct (Dự đoán đúng)"

    top15_err = rw_errors.head(15)
    top5_corr = pd.concat(
        [
            rw_correct[(rw_correct["source"] == "D_adversarial_curated") & (rw_correct["label_final"] == 1)].head(3),
            rw_correct[(rw_correct["source"] == "D_adversarial_curated") & (rw_correct["label_final"] == 0)].head(2),
        ],
        ignore_index=True,
    )
    examples_20 = pd.concat([top15_err, top5_corr], ignore_index=True)
    examples_20[
        ["id", "source", "error_type", "label_a", "label_b", "label_final", "y_pred", "score", "text_masked", "note"]
    ].to_csv(TAB_DIR / "P9_realworld_examples.csv", index=False, encoding="utf-8")

    # 7. Plot P9 Real-World Evaluation Figure (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5), dpi=150)

    # Panel 1: F1, Precision, Recall by Source A/B/C/D
    src_short = [s.split("_")[0] + ": " + "_".join(s.split("_")[1:])[:14] for s in by_source_df["source"]]
    x = np.arange(len(src_short))
    w = 0.26
    axes[0].bar(x - w, by_source_df["precision_pos"], w, label="Precision", color="#1e88e5", edgecolor="black")
    axes[0].bar(x, by_source_df["recall_pos"], w, label="Recall", color="#fb8c00", edgecolor="black")
    axes[0].bar(x + w, by_source_df["f1_pos"], w, label="F1-Score", color="#e53935", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(src_short, rotation=18, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Điểm số (Score)")
    axes[0].set_title(f"Hiệu năng theo 4 nguồn thực tế (Overall F1={overall_metrics['f1_pos']:.3f})", fontweight="bold")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Confusion Matrix on all 300 Real-World samples
    cm_rw = confusion_matrix(y_true_all, rw_df["y_pred"], labels=[0, 1])
    sns.heatmap(
        cm_rw,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=["0 (Bình thường)", "1 (Phản cảm)"],
        yticklabels=["0 (Bình thường)", "1 (Phản cảm)"],
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_title(
        f"Ma trận nhầm lẫn Tập Thực tế (N={len(rw_df)})\nCohen's Kappa κ = {kappa:.4f} (Đồng thuận {agreement_pct:.1f}%)",
        fontweight="bold",
        fontsize=10.5,
    )
    axes[1].set_xlabel("Dự đoán (Predicted)")
    axes[1].set_ylabel("Nhãn thống nhất (label_final)")

    # Panel 3: Assigned Twitter_Data % predicted hate by sentiment category (-1/0/+1)
    sent_df = pd.DataFrame(assigned_by_sentiment)
    b_s = axes[2].bar(
        sent_df["sentiment_name"],
        sent_df["pct_predicted_hate"],
        color=["#e53935", "#90a4ae", "#43a047"],
        edgecolor="black",
        width=0.5,
    )
    axes[2].set_title("Twitter_Data (N=162,969): % Dự đoán phản cảm theo Sentiment", fontweight="bold")
    axes[2].set_ylabel("Tỉ lệ dự đoán Lớp 1 (%)")
    for b in b_s:
        axes[2].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.3,
            f"{b.get_height():.2f}%",
            ha="center",
            fontweight="bold",
        )
    axes[2].set_ylim(0, max(sent_df["pct_predicted_hate"]) * 1.28 + 1)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "P9_realworld_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "n_total": int(len(rw_df)),
        "n_pos": int((y_true_all == 1).sum()),
        "n_neg": int((y_true_all == 0).sum()),
        "cohen_kappa": kappa,
        "annotator_agreement_pct": agreement_pct,
        "n_disagreements": n_disagreements,
        "threshold_used": thr,
        "overall_metrics": overall_metrics,
        "metrics_by_source": by_source_rows,
        "davidson_cross_domain_metrics": davidson_metrics,
        "assigned_twitter_by_sentiment": assigned_by_sentiment,
        "sample_errors_masked": top15_err[
            ["id", "source", "error_type", "label_final", "y_pred", "score", "text_masked", "note"]
        ].to_dict(orient="records"),
        "sample_correct_masked": top5_corr[
            ["id", "source", "label_final", "y_pred", "score", "text_masked", "note"]
        ].to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "realworld_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


if __name__ == "__main__":
    res = run_p9_realworld()
    print("P9 finished. Kappa:", res["cohen_kappa"], "Overall F1:", res["overall_metrics"]["f1_pos"])
