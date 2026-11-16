"""B2, B3 & V8 Generator: PowerPoint Presentation (reports/slides.pptx) & Speaker Script (reports/speaker_script.md) including Vietnamese Extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from src.utils import get_project_root

PROJECT_ROOT = get_project_root()
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

NAVY = RGBColor(20, 45, 85)
DARK_TEXT = RGBColor(30, 35, 45)
WHITE = RGBColor(255, 255, 255)


def _add_header(slide, prs, title_text: str, slide_num: int, total_slides: int = 15) -> None:
    banner = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, Inches(0.95)
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.fill.background()

    tf = banner.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"{slide_num}. {title_text}"
    p.font.name = "Arial"
    p.font.size = Pt(21)
    p.font.bold = True
    p.font.color.rgb = WHITE

    footer = slide.shapes.add_textbox(Inches(11.3), Inches(7.05), Inches(1.8), Inches(0.35))
    fp = footer.text_frame.paragraphs[0]
    fp.text = f"Slide {slide_num}/{total_slides}"
    fp.alignment = PP_ALIGN.RIGHT
    fp.font.name = "Arial"
    fp.font.size = Pt(11)
    fp.font.color.rgb = RGBColor(100, 110, 125)


def _add_bullets(slide, bullets: list[str], left: float, top: float, width: float, height: float, font_size: int = 16) -> None:
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(bullets[:5]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {line}"
        p.font.name = "Arial"
        p.font.size = Pt(font_size)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(10)


def _add_image_safe(slide, img_path: Path, left: float, top: float, width: float | None = None, height: float | None = None) -> None:
    if img_path.exists():
        kwargs = {}
        if width is not None:
            kwargs["width"] = Inches(width)
        if height is not None:
            kwargs["height"] = Inches(height)
        slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), **kwargs)


def _set_notes(slide, notes_lines: list[str]) -> None:
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = "\n".join(f"- {ln}" for ln in notes_lines)


def build_slides_and_script() -> None:
    facts: dict[str, Any] = json.loads((REPORTS_DIR / "facts.json").read_text(encoding="utf-8"))

    ds_tw = facts["datasets"]["assigned"]
    ds_hate = facts["datasets"]["hate"]
    sp = facts["preprocessing_and_splits"]
    bal = facts["balance_choice"]
    bm = facts["best_model"]
    ft = facts["final_test"]
    rw = facts["realworld"]
    sql_f = facts["sql_analysis"]
    bonus = facts["bonus_models"]
    vi = facts.get("vi", {})
    vi_ds = vi.get("dataset", {})
    vi_ft = vi.get("final_test", {})
    vi_rob = vi.get("robustness", {})
    vi_rw = vi.get("realworld", {})

    total_slides = 15
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Tiêu đề + nhóm/tên + giảng viên
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(0.9), Inches(1.2), Inches(11.5), Inches(5.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    vi_f1_str = f"{vi_ft['bin_A']['metrics_tuned']['f1_pos']:.4f}" if vi_ft else "N/A"
    lines_s1 = [
        ("BÁO CÁO ĐỒ ÁN MÔN HỌC: KHAI PHÁ VĂN BẢN & HỌC MÁY", 18, False, RGBColor(180, 210, 255)),
        ("GẮN NHÃN CÁC ĐOẠN TWEET CÓ NỘI DUNG PHÂN BIỆT CHỦNG TỘC, PHÂN BIỆT GIỚI TÍNH", 26, True, WHITE),
        ("(Twitter Sentimental Analysis — Binary Classification & Mở rộng Đa ngôn ngữ Tiếng Việt ViHSD)", 19, False, RGBColor(255, 213, 79)),
        (f"• Tiếng Anh: {bm['name']} (Thr = {bm['threshold']:.3f}) | Test F1 = {ft['f1_pos']:.4f} | PR-AUC = {ft['pr_auc']:.4f}", 16, False, WHITE),
        (f"• Mở rộng Tiếng Việt (ViHSD 33,400 dòng): Test bin_A F1 = {vi_f1_str} | multi3 Macro-F1 = {vi_ft['multi3']['f1_macro']:.4f}", 16, False, RGBColor(210, 225, 245)),
    ]
    for idx, (txt, sz, is_b, col) in enumerate(lines_s1):
        p = tf1.paragraphs[0] if idx == 0 else tf1.add_paragraph()
        p.text = txt
        p.font.name = "Arial"
        p.font.size = Pt(sz)
        p.font.bold = is_b
        p.font.color.rgb = col
        p.space_after = Pt(16)

    _set_notes(
        s1,
        [
            "Chào thầy/cô và các bạn. Hôm nay nhóm xin trình bày đề tài: Gắn nhãn các đoạn tweet có nội dung phân biệt chủng tộc, phân biệt giới tính, kết hợp phần mở rộng điểm cộng cho Tiếng Việt trên bộ dữ liệu ViHSD.",
            f"Kết quả chính: Tiếng Anh đạt Test F1 = {ft['f1_pos']:.4f} (PR-AUC = {ft['pr_auc']:.4f}); Tiếng Việt đạt Test F1 (bin_A) = {vi_f1_str} và Macro-F1 3 lớp = {vi_ft['multi3']['f1_macro']:.4f}.",
        ],
    )

    # =========================================================================
    # SLIDE 2: Bài toán & mục tiêu
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    _add_header(s2, prs, "Bài toán & Mục tiêu Nghiên cứu", 2, total_slides)
    _add_bullets(
        s2,
        [
            "Bối cảnh: Ngôn từ thù ghét chủng tộc/giới tính (English) và ngôn từ thù ghét/xúc phạm (Tiếng Việt) lan truyền nhanh trên MXH.",
            "Bài toán: Phân loại nhị phân — Nhãn 1 (Phản cảm / Thù ghét) vs Nhãn 0 (Bình thường / Sạch).",
            "Mục tiêu 1 (Câu 1): Khám phá dữ liệu & dùng truy vấn SQL tìm điểm chung định lượng của tweet phản cảm.",
            "Mục tiêu 2 (Câu 2 & 3): Làm sạch chống rò rỉ, cân bằng dữ liệu, so sánh 11 mô hình (9 cổ điển + BiLSTM + Transformer).",
            "Mục tiêu 3 (Câu 4 & Điểm cộng): Mở rộng Tiếng Việt (ViHSD 33,400 mẫu) và tích hợp ứng dụng Web Streamlit Đa ngôn ngữ.",
        ],
        left=0.7,
        top=1.35,
        width=11.8,
        height=5.4,
        font_size=17,
    )
    _set_notes(s2, ["Nêu rõ mục tiêu kép: hoàn thành chuẩn mực 4 câu của đề bài tiếng Anh và mở rộng sang tiếng Việt (V0→V8)."])

    # =========================================================================
    # SLIDE 3: Dữ liệu: nguồn, kích thước, vấn đề nhãn bộ được giao, hướng xử lý
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    _add_header(s3, prs, "Nguồn Dữ liệu & Phát hiện Quan trọng về Nhãn", 3, total_slides)
    _add_bullets(
        s3,
        [
            f"Bộ được giao (Twitter_Data.csv): {ds_tw['n_rows']:,} dòng, chỉ có 2 cột clean_text và category (-1/0/+1 sentiment chính trị).",
            "Phát hiện cốt lõi: Bộ được giao KHÔNG có nhãn racist/sexist (nhãn -1 là bình luận tiêu cực về bầu cử Ấn Độ/Modi).",
            f"Giải pháp (Phương án A): Huấn luyện trên bộ Tweets Hate Speech Detection ({ds_hate['n_rows']:,} dòng, nhãn 0/1 chuẩn).",
            f"Tận dụng bộ được giao & Reddit ({facts['datasets']['assigned_reddit']['n_rows']:,} dòng) cho Câu 1 (EDA) & Câu 3 (kiểm thử lệch miền).",
            f"Mở rộng Tiếng Việt: Bộ chuẩn ViHSD ({vi_ds.get('n', 33400):,} dòng, chia chuẩn 24,048 train / 2,672 dev / 6,680 test).",
        ],
        left=0.5,
        top=1.2,
        width=5.8,
        height=5.6,
        font_size=14,
    )
    _add_image_safe(s3, FIG_DIR / "P2_01_dataset_quality.png", left=6.45, top=1.35, width=6.5)
    _set_notes(s3, [f"Giải thích trung thực vấn đề nhãn của Twitter_Data.csv ({ds_tw['n_rows']:,} dòng) và lý do chọn Phương án A."])

    # =========================================================================
    # SLIDE 4: Thống kê mô tả (phân bố nhãn, độ dài)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    _add_header(s4, prs, "Thống kê Mô tả: Phân bố Nhãn & Đặc trưng Bề mặt", 4, total_slides)
    _add_bullets(
        s4,
        [
            f"Mất cân bằng nghiêm trọng: Lớp 0 = {ds_hate['label_distribution']['0']:,} (92.99%) vs Lớp 1 = {ds_hate['label_distribution']['1']:,} ({ds_hate['positive_rate_pct']:.2f}%), tỉ lệ {ds_hate['imbalance_ratio_neg_to_pos']}:1.",
            f"Trùng lặp cao: {ds_hate['n_duplicates_text']:,} câu trùng nguyên bản, 3,628 câu trùng sau làm sạch (do retweet).",
            "Độ dài trung bình: Lớp 1 dài ~89.2 ký tự (13.9 từ) so với Lớp 0 ~84.3 ký tự (13.3 từ).",
            "Kiểm định Chi² (p < 1e-10): 65.03% tweet phản cảm nhắc @user (vs 46.46% ở Lớp 0) và 18.78% có dấu '?' (vs 10.42%).",
            f"Độ tương đồng từ vựng Jaccard Top-5000 giữa bộ Hate và Twitter_Data chỉ đạt {facts['datasets']['domain_jaccard_top5000']:.4f}.",
        ],
        left=0.5,
        top=1.15,
        width=5.7,
        height=5.8,
        font_size=14,
    )
    _add_image_safe(s4, FIG_DIR / "P2_02_label_distribution.png", left=6.35, top=1.15, width=6.6)
    _add_image_safe(s4, FIG_DIR / "P2_04_surface_features.png", left=6.35, top=4.15, width=6.6)
    _set_notes(s4, [f"Nhấn mạnh con số {ds_hate['positive_rate_pct']:.2f}% lớp dương (tỉ lệ 13.26:1) và kiểm định Chi-square."])

    # =========================================================================
    # SLIDE 5: Điểm chung của tweet phản cảm (từ khóa SQL, WordCloud đã che)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    _add_header(s5, prs, "Phân tích SQL: Điểm chung của Tweet Phản cảm", 5, total_slides)
    lex0 = sql_f["lexicon_coverage"][0]["pct_with_lexicon"]
    lex1 = sql_f["lexicon_coverage"][1]["pct_with_lexicon"]
    _add_bullets(
        s5,
        [
            f"Cơ sở dữ liệu SQLite (db/tweets.db): chỉ nạp tập Train ({sql_f['n_train_tweets_in_db']:,} tweet, {sql_f['n_tokens_in_db']:,} token) để chống rò rỉ.",
            f"Truy vấn 40_lexicon.sql: {lex1:.2f}% tweet Lớp 1 chứa từ trong Lexicon xúc phạm, cao gấp 8.6 lần Lớp 0 ({lex0:.2f}%).",
            "Tuy nhiên 79.41% tweet phản cảm KHÔNG chứa từ tục trực tiếp ⇒ lọc từ khóa đơn thuần sẽ bỏ sót gần 4/5 mẫu.",
            "Truy vấn 20_log_odds.sql & 30_cooccur.sql: tập trung vào 3 cụm chủ đề (Chủng tộc, Giới tính/Phụ nữ, Tôn giáo/Chính trị).",
            "Lưu ý: Từ có tần suất cao ở Lớp 1 (như 'women', 'black', 'racist') không đồng nghĩa câu chứa nó luôn phản cảm.",
        ],
        left=0.5,
        top=1.15,
        width=5.7,
        height=5.8,
        font_size=14,
    )
    _add_image_safe(s5, FIG_DIR / "P3_sql_analysis.png", left=6.35, top=1.2, width=6.65)
    _add_image_safe(s5, FIG_DIR / "P2_06_wordcloud_by_label.png", left=6.35, top=4.05, width=6.65)
    _set_notes(s5, [f"Trình bày kết quả 4 câu lệnh SQL: {lex1:.2f}% vs {lex0:.2f}% trong Lexicon."])

    # =========================================================================
    # SLIDE 6: Tiền xử lý & cân bằng dữ liệu
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    _add_header(s6, prs, "Tiền xử lý 12 bước, Khử trùng & Cân bằng Dữ liệu", 6, total_slides)
    _add_bullets(
        s6,
        [
            "Pipeline clean_text 12 bước: xóa URL/@user/RT, #tag→tag, chuẩn hóa chữ kéo dài, lemmatize WordNet.",
            "Giữ lại từ phủ định (no, not, never) và đại từ/nhóm người (you, she, they, women, men) vì quyết định ngữ nghĩa kỳ thị.",
            f"Khử trùng trước khi chia tập: loại {sp['n_near_duplicate_rows_dropped']:,} mẫu trùng + {sp['n_conflict_rows_dropped']} mẫu mâu thuẫn nhãn ⇒ còn {sp['n_final_unique_clean']:,} mẫu sạch.",
            f"Không cân bằng (none): Recall chỉ đạt {bal['table'][0]['recall_mean']*100:.2f}%, F1 = {bal['table'][0]['f1_pos_mean']:.4f} (bỏ sót >86% mẫu phản cảm).",
            f"Chọn {bal['chosen_strategy'].upper()} (RandomOverSampler trong Pipeline): đạt CV F1 cao nhất = 0.5876 ± 0.0117.",
        ],
        left=0.5,
        top=1.2,
        width=5.8,
        height=5.6,
        font_size=14,
    )
    _add_image_safe(s6, FIG_DIR / "P4_balance_comparison.png", left=6.4, top=1.35, width=6.55)
    _set_notes(s6, [f"Nhấn mạnh việc khử trùng {sp['n_near_duplicate_rows_dropped']:,} tweet trước khi chia tập."])

    # =========================================================================
    # SLIDE 7: Thiết kế thực nghiệm & Hệ thống mô hình
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    _add_header(s7, prs, "Thiết kế Thực nghiệm Chống Rò rỉ & Hệ thống 11 Mô hình", 7, total_slides)
    _add_bullets(
        s7,
        [
            f"Chia tập Stratified 70/15/15 (seed=42): Train = {sp['splits']['train']['n_rows']:,} | Val = {sp['splits']['val']['n_rows']:,} | Test = {sp['splits']['test']['n_rows']:,} mẫu (6.52% lớp 1).",
            "Chống rò rỉ nghiêm ngặt: Vectorizer, Sampler, Hyperparameter & Ngưỡng chỉ học từ Train/Val; Test chỉ chạy 1 lần ở P8.",
            "Tiêu chí chính: F1 lớp dương (f1_pos) kết hợp PR-AUC (vì đoán toàn bộ lớp 0 vẫn đạt 93.48% Accuracy).",
            "9 Mô hình cổ điển: LR, Calibrated Linear SVM, Complement NB, RBF SVM, KNN, Decision Tree, Random Forest, XGBoost, LightGBM.",
            "Điểm cộng Học sâu / Transformer: BiLSTM (PyTorch), DistilBERT (English), PhoBERT-v2 & XLM-RoBERTa (Tiếng Việt).",
        ],
        left=0.5,
        top=1.2,
        width=5.8,
        height=5.6,
        font_size=14,
    )
    _add_image_safe(s7, FIG_DIR / "P4_length_before_after.png", left=6.4, top=1.4, width=6.5)
    _set_notes(s7, ["Giải thích quy trình chống Data Leakage 4 tầng và hệ thống 11 mô hình."])

    # =========================================================================
    # SLIDE 8: Bảng/biểu đồ so sánh kết quả
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    _add_header(s8, prs, "Kết quả So sánh các Mô hình trên Tập Validation (P5 & P7)", 8, total_slides)
    _add_bullets(
        s8,
        [
            "Top 1 (Trước tuning): LR (word+char) đạt Val F1 = 0.6486 (PR-AUC = 0.7090), vượt trội nhờ đặc trưng ký tự.",
            "Top 2 & 3 (Word TF-IDF): LR (F1 = 0.5832), Calibrated Linear SVM (F1 = 0.5727, Precision cao = 0.7529).",
            "Complement NB (word+char): đạt PR-AUC cao nhất trước tuning (0.7403) và Recall = 84.48%.",
            "Nhóm Cây/Boosting (LightGBM F1=0.5260, RF F1=0.5195, XGB F1=0.5034, DT F1=0.3836) kém hơn trên TF-IDF thưa.",
            f"Mô hình Điểm cộng (P7): BiLSTM đạt Val F1 = {bonus['bilstm']['f1_pos']:.4f}, DistilBERT đạt Val F1 = {bonus['distilbert']['f1_pos']:.4f}.",
        ],
        left=0.45,
        top=1.15,
        width=5.6,
        height=5.8,
        font_size=14,
    )
    _add_image_safe(s8, FIG_DIR / "P5_f1_prauc_bars.png", left=6.15, top=1.15, width=6.85)
    _add_image_safe(s8, FIG_DIR / "P5_roc_pr_curves.png", left=6.15, top=4.2, width=6.85)
    _set_notes(s8, ["Giải thích vì sao các mô hình tuyến tính vượt xa Decision Tree, Random Forest và KNN trên ma trận TF-IDF thưa."])

    # =========================================================================
    # SLIDE 9: Tinh chỉnh tham số & mô hình được chọn
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    _add_header(s9, prs, "Tinh chỉnh Siêu tham số (P6) & Tối ưu Ngưỡng Quyết định", 9, total_slides)
    _add_bullets(
        s9,
        [
            "RandomizedSearchCV (40 tổ hợp × 5-fold CV) trên tập Train cho Top-3 mô hình: LR, Linear SVM, Complement NB.",
            f"Cấu hình tốt nhất: {bm['name']} với ngram_range=(1,3), min_df=2, max_features=50,000, C={bm['params']['clf__C']}, squared_hinge.",
            f"Tối ưu ngưỡng trên Val: Hạ ngưỡng từ 0.50 xuống Threshold = {bm['threshold']:.3f} giúp cân bằng Precision (0.6877) & Recall (0.6282).",
            f"Cải thiện mạnh mẽ: Val F1 tăng từ {bm['val_metrics_before_tuning']['val_f1_before']:.4f} lên {bm['val_metrics_after_tuning_best_thr']['f1_pos']:.4f} (+8.39 điểm %).",
            "Huấn luyện lại best_model trên Train + Val (24,080 mẫu) và đóng gói vào models/best_model.joblib.",
        ],
        left=0.45,
        top=1.15,
        width=5.6,
        height=5.8,
        font_size=14,
    )
    _add_image_safe(s9, FIG_DIR / "P6_threshold_tuning.png", left=6.15, top=1.15, width=6.85)
    _add_image_safe(s9, FIG_DIR / "P6_before_after.png", left=6.15, top=4.15, width=6.85)
    _set_notes(s9, [f"Giải thích vì sao hạ ngưỡng xuống {bm['threshold']:.3f} trên tập Val giúp tăng Recall lên 0.6282 và F1 đạt đỉnh {bm['val_metrics_after_tuning_best_thr']['f1_pos']:.4f}."])

    # =========================================================================
    # SLIDE 10: Đánh giá Test & Chạy thử thực tế Tiếng Anh
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    _add_header(s10, prs, "Đánh giá Cuối Tập Test (P8) & Chạy thử Thực tế Tiếng Anh (P9)", 10, total_slides)
    ci_f1 = ft["ci95"]["f1_pos"]
    _add_bullets(
        s10,
        [
            f"Kết quả Test độc lập (N=4,250, chạy 1 lần): F1 = {ft['f1_pos']:.4f} (95% CI: [{ci_f1['ci95_low']:.4f}, {ci_f1['ci95_high']:.4f}]), PR-AUC = {ft['pr_auc']:.4f}, ROC-AUC = {ft['roc_auc']:.4f}.",
            f"Bộ thực tế (N={rw['n']} mẫu từ 4 nguồn A/B/C/D): 2 người gắn nhãn độc lập đạt đồng thuận {rw['agreement_pct']:.2f}%, Cohen's Kappa κ = {rw['kappa']:.4f}.",
            f"Hiệu năng thực tế tổng thể: F1 = {rw['overall_metrics']['f1_pos']:.4f} (Precision = {rw['overall_metrics']['precision_pos']:.4f}, Recall = {rw['overall_metrics']['recall_pos']:.4f}, PR-AUC = {rw['overall_metrics']['pr_auc']:.4f}).",
            "Phân tích lệch miền (Domain Shift): Nguồn A cùng miền đạt F1 cao hơn nguồn D (câu mỉa mai/phủ định hiểm hóc).",
            "Trên Twitter_Data (162,969 dòng): chỉ 7.51% câu sentiment = -1 được dự đoán là hate speech ⇒ khẳng định khác biệt bản chất.",
        ],
        left=0.45,
        top=1.15,
        width=5.6,
        height=5.8,
        font_size=13,
    )
    _add_image_safe(s10, FIG_DIR / "P8_test_evaluation.png", left=6.15, top=1.15, width=6.85)
    _add_image_safe(s10, FIG_DIR / "P9_realworld_evaluation.png", left=6.15, top=4.1, width=6.85)
    _set_notes(s10, [f"Trình bày con số chốt trên tập Test: F1 = {ft['f1_pos']:.4f} và thực tế 300 mẫu: Kappa = {rw['kappa']:.4f}, F1 = {rw['overall_metrics']['f1_pos']:.4f}."])

    # =========================================================================
    # SLIDE 11: Phân tích lỗi & kiểm tra thiên lệch
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    _add_header(s11, prs, "Phân tích Lỗi (FP/FN), Giải thích Trọng số & Thiên lệch Nhóm", 11, total_slides)
    _add_bullets(
        s11,
        [
            f"Trên tập Test (4,250 mẫu): TP = {ft['tp']}, TN = {ft['tn']}, FP (Báo động giả) = {ft['fp']}, FN (Bỏ sót) = {ft['fn']}.",
            "Nguyên nhân Bỏ sót (FN): Kỳ thị ngầm/định kiến không dùng từ tục (chiếm đa số), câu mỉa mai, hoặc từ lóng hiếm gặp.",
            "Nguyên nhân Báo động giả (FP): Câu bình thường/phản đối kỳ thị nhưng chứa từ nhạy cảm ('women', 'black', 'racist').",
            "Kiểm tra thiên lệch (Subgroup Bias): Câu Lớp 0 chứa từ chỉ chủng tộc/giới tính có FPR cao hơn mức FPR nền của tập Test.",
            "Giải pháp vận hành: Luôn hiển thị Top từ đóng góp (TF-IDF × Hệ số) và dùng con người duyệt lại các mẫu gần ngưỡng.",
        ],
        left=0.45,
        top=1.15,
        width=5.6,
        height=5.8,
        font_size=14,
    )
    _add_image_safe(s11, FIG_DIR / "P8_error_causes.png", left=6.15, top=1.15, width=6.85)
    _add_image_safe(s11, FIG_DIR / "P8_subgroup_bias.png", left=6.15, top=4.1, width=6.85)
    _set_notes(s11, ["Phân tích lỗi False Positive trên câu counter-speech và False Negative trên câu kỳ thị ngầm."])

    # =========================================================================
    # SLIDE 12: MỞ RỘNG TIẾNG VIỆT (V0 -> V8) — MANDATORY V8 SLIDE
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    _add_header(s12, prs, "Mở rộng Tiếng Việt (ViHSD — Phân loại & Độ bền Không dấu)", 12, total_slides)
    vi_ba_f1 = vi_ft["bin_A"]["metrics_tuned"]["f1_pos"]
    vi_ba_pr = vi_ft["bin_A"]["metrics_tuned"]["pr_auc"]
    vi_bb_f1 = vi_ft["bin_B"]["metrics_tuned"]["f1_pos"]
    vi_m3_f1 = vi_ft["multi3"]["f1_macro"]
    vi_noacc_f1 = vi_rob["no_accents"]["f1_pos"]
    vi_noacc_drop = vi_rob["no_accents"]["f1_drop"]
    vi_tc_f1 = vi_rob["teencode_with_normalizer"]["f1_pos"]
    _add_bullets(
        s12,
        [
            f"Dữ liệu ViHSD (Luu et al., 2021): {vi_ds['n']:,} bình luận (24,048 train / 2,672 dev / 6,680 test), 3 nhãn CLEAN (82.7%) / HATE (10.5%) / OFFENSIVE (6.8%).",
            "Khác biệt định nghĩa nhãn: ViHSD phát hiện ngôn từ thù ghét/xúc phạm cá nhân hoặc nhóm nói chung (không riêng racist/sexist).",
            f"Kết quả Test chuẩn: bin_A (HATE vs CLEAN) đạt F1 = {vi_ba_f1:.4f} (PR-AUC = {vi_ba_pr:.4f}); bin_B F1 = {vi_bb_f1:.4f}; multi3 (3 lớp) Macro-F1 = {vi_m3_f1:.4f}.",
            f"Độ bền (Robustness): Teencode có chuẩn hóa giữ F1 = {vi_tc_f1:.4f}; khi BỎ DẤU hoàn toàn (strip_accents), F1 giảm {vi_noacc_drop:.4f} (còn {vi_noacc_f1:.4f}).",
            f"Thực tế Tiếng Việt (N={vi_rw['n']} mẫu, 3 nguồn): Cohen's Kappa 3 lớp κ = {vi_rw['kappa']:.4f} (Weighted κ = {vi_rw['kappa_weighted']:.4f}), Real-World bin_A F1 = {vi_rw['overall_bin_A']['f1_pos']:.4f}.",
        ],
        left=0.45,
        top=1.15,
        width=5.6,
        height=5.8,
        font_size=13,
    )
    _add_image_safe(s12, FIG_DIR / "V5_test_and_robustness.png", left=6.15, top=1.15, width=6.85)
    _add_image_safe(s12, FIG_DIR / "V6_vi_realworld_eval.png", left=6.15, top=4.1, width=6.85)
    _set_notes(
        s12,
        [
            f"Trình bày phần mở rộng Tiếng Việt (V0→V8): bộ ViHSD 33,400 dòng, kết quả Test bin_A F1 = {vi_ba_f1:.4f}, multi3 Macro-F1 = {vi_m3_f1:.4f} (khớp mốc baseline công bố 0.60-0.66).",
            f"Nhấn mạnh thí nghiệm độ bền (Robustness): bỏ dấu tiếng Việt làm F1 giảm {vi_noacc_drop:.4f} điểm xuống {vi_noacc_f1:.4f}, giải thích lý do cần cảnh báo người dùng trên App.",
        ],
    )

    # =========================================================================
    # SLIDE 13: Demo ứng dụng Đa ngôn ngữ
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    _add_header(s13, prs, "Ứng dụng Web Streamlit Đa ngôn ngữ (English & Tiếng Việt)", 13, total_slides)
    _add_bullets(
        s13,
        [
            "Tích hợp MultiPredictor (V7): Tự động nhận diện ngôn ngữ (ưu tiên dấu Tiếng Việt + langdetect ≥ 0.80) hoặc chọn tay.",
            "Chặn an toàn câu không rõ ngôn ngữ ('unknown'): hiển thị cảnh báo đối với câu quá ngắn hoặc Tiếng Việt không dấu.",
            "Tab 1 (Phân loại 1 câu): 6 nút mẫu (2 Anh, 2 Việt có dấu, 1 Việt không dấu, 1 câu ngắn), hiển thị rõ mô hình & định nghĩa nhãn.",
            "Tab 2 (Hàng loạt CSV): Thêm cột lang và model_used; xử lý 1,000 câu hỗn hợp Anh-Việt chỉ trong < 1.5 giây.",
            "Tab 3 (Thống kê độc lập): Đối sánh riêng biệt bảng kết quả Tiếng Anh và Tiếng Việt, tuyệt đối không trộn lẫn số liệu.",
        ],
        left=0.45,
        top=1.15,
        width=5.6,
        height=5.8,
        font_size=14,
    )
    app_shot = FIG_DIR / "P10_app_tab1_hate.png"
    if app_shot.exists():
        _add_image_safe(s13, app_shot, left=6.15, top=1.2, width=6.85)
    _set_notes(s13, ["Demo trực tiếp 6 câu mẫu đa ngôn ngữ trên ứng dụng Streamlit."])

    # =========================================================================
    # SLIDE 14: Kết luận, hạn chế, hướng phát triển
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    _add_header(s14, prs, "Kết luận, Hạn chế & Hướng Phát triển", 14, total_slides)
    _add_bullets(
        s14,
        [
            f"Kết luận: Hoàn thành trọn vẹn P0→P12 (Tiếng Anh: Test F1 = {ft['f1_pos']:.4f}) và V0→V8 (Tiếng Việt ViHSD: Test bin_A F1 = {vi_ba_f1:.4f}, multi3 Macro-F1 = {vi_m3_f1:.4f}).",
            f"Kiểm thử thực tế có đo Cohen's Kappa trên cả 2 ngôn ngữ: Tiếng Anh (300 mẫu, κ = {rw['kappa']:.4f}) & Tiếng Việt (210 mẫu, κ = {vi_rw['kappa']:.4f}).",
            "Hạn chế: Khác biệt định nghĩa nhãn giữa 2 bộ dữ liệu; hiệu năng giảm khi người dùng gõ Tiếng Việt không dấu hoặc mỉa mai sâu.",
            "Đạo đức & Giấy phép: Che từ tục tự động (mask_profanity & vi_mask_profanity); dữ liệu ViHSD chỉ dùng cho nghiên cứu (Luu et al., 2021).",
            "Hướng phát triển: Fine-tune toàn phần PhoBERT-v2 / ViHateT5 trên GPU kết hợp khôi phục dấu tiếng Việt tự động (Diacritic Restoration).",
        ],
        left=0.7,
        top=1.35,
        width=11.8,
        height=5.4,
        font_size=15,
    )
    _set_notes(s14, ["Tóm tắt kết quả nổi bật trên cả Tiếng Anh và Tiếng Việt, nêu rõ hạn chế về văn bản không dấu và hướng tích hợp bộ khôi phục dấu."])

    # =========================================================================
    # SLIDE 15: Cảm ơn + Q&A
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    bg15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
    bg15.fill.solid()
    bg15.fill.fore_color.rgb = NAVY
    bg15.line.fill.background()

    tb15 = s15.shapes.add_textbox(Inches(0.9), Inches(1.6), Inches(11.5), Inches(4.5))
    tf15 = tb15.text_frame
    tf15.word_wrap = True
    lines_s15 = [
        ("XIN CHÂN THÀNH CẢM ƠN THẦY/CÔ VÀ CÁC BẠN ĐÃ LẮNG NGHE!", 27, True, WHITE),
        ("HỎI & ĐÁP (Q&A SESSION — 70 CÂU HỎI CHUẨN BỊ SẴN)", 22, True, RGBColor(255, 213, 79)),
        (
            f"🇬🇧 English: Clean = {sp['n_final_unique_clean']:,} (Pos = {sp['final_pos_rate_pct']:.2f}%) | {bm['name']} (Thr={bm['threshold']:.3f}) | "
            f"Test F1 = {ft['f1_pos']:.4f} | Real-World F1 = {rw['overall_metrics']['f1_pos']:.4f} (κ = {rw['kappa']:.4f})",
            15,
            False,
            WHITE,
        ),
        (
            f"🇻🇳 Tiếng Việt (ViHSD): N = {vi_ds['n']:,} | bin_A Test F1 = {vi_ba_f1:.4f} (PR-AUC = {vi_ba_pr:.4f}) | "
            f"multi3 Macro-F1 = {vi_m3_f1:.4f} | Real-World F1 = {vi_rw['overall_bin_A']['f1_pos']:.4f} (κ = {vi_rw['kappa']:.4f})",
            15,
            False,
            RGBColor(180, 215, 255),
        ),
    ]
    for idx, (txt, sz, is_b, col) in enumerate(lines_s15):
        p = tf15.paragraphs[0] if idx == 0 else tf15.add_paragraph()
        p.text = txt
        p.alignment = PP_ALIGN.CENTER
        p.font.name = "Arial"
        p.font.size = Pt(sz)
        p.font.bold = is_b
        p.font.color.rgb = col
        p.space_after = Pt(18)

    _set_notes(s15, ["Cảm ơn hội đồng và sẵn sàng trả lời 70 câu hỏi trong qa_bank.md."])

    out_pptx = REPORTS_DIR / "slides.pptx"
    prs.save(str(out_pptx))

    # Append Slide 12 Vietnamese script to speaker_script.md
    script_path = REPORTS_DIR / "speaker_script.md"
    if script_path.exists():
        old_script = script_path.read_text(encoding="utf-8")
        vi_section = f"""

---

## PHẦN 3: BỔ SUNG KỊCH BẢN THUYẾT TRÌNH SLIDE 12 — MỞ RỘNG TIẾNG VIỆT (ViHSD — V8) `[09:40 – 10:30 | 50 giây]`

- **Lời nói tại Slide 12 (Mở rộng Tiếng Việt):**
  "Để mở rộng giá trị ứng dụng của đề tài cho bối cảnh mạng xã hội tại Việt Nam, nhóm đã thực hiện trọn vẹn phần mở rộng Tiếng Việt từ V0 đến V8 trên bộ dữ liệu chuẩn **ViHSD** của nhóm tác giả Lưu Sơn Tùng và cộng sự (UIT-NLP, 2021) gồm **{vi_ds['n']:,} bình luận** chia chuẩn `24,048 train / 2,672 dev / 6,680 test`. Em xin lưu ý rõ sự khác biệt về định nghĩa nhãn: nếu bản tiếng Anh tập trung vào *racist/sexist*, thì bộ ViHSD phát hiện **ngôn từ thù ghét và xúc phạm tiếng Việt nói chung** gồm 3 nhãn `CLEAN (82.71%)`, `HATE (10.52%)` và `OFFENSIVE (6.77%)`.
  Sau khi xây dựng hàm làm sạch giữ nguyên dấu tiếng Việt, chuẩn hóa từ điển Teencode và tách từ bằng `ViTokenizer`, mô hình chốt tiếng Việt của nhóm đạt **Test F1 (`bin_A` — HATE vs CLEAN) = {vi_ba_f1:.4f}** (PR-AUC = **{vi_ba_pr:.4f}**), bài toán `bin_B` đạt **F1 = {vi_bb_f1:.4f}**, và bài toán 3 lớp `multi3` đạt **Macro-F1 = {vi_m3_f1:.4f}** (nằm đúng trong vùng baseline công bố quốc tế `0.60 – 0.675`).
  Đặc biệt, nhóm thực hiện thí nghiệm **Kiểm tra độ bền (Robustness)** ở biểu đồ góc trên bên phải: khi người dùng gõ teencode, nhờ bộ chuẩn hóa của nhóm, F1 giữ vững ở mức **{vi_tc_f1:.4f}**; nhưng khi người dùng **bỏ dấu hoàn toàn (`strip_accents`)**, F1 giảm **{vi_noacc_drop:.4f}** điểm xuống còn **{vi_noacc_f1:.4f}**. Trên bộ thực tế tiếng Việt gồm **{vi_rw['n']} bình luận** (Cohen's Kappa 3 lớp $\\kappa = {vi_rw['kappa']:.4f}$, Weighted $\\kappa = {vi_rw['kappa_weighted']:.4f}$), mô hình đạt **Real-World F1 = {vi_rw['overall_bin_A']['f1_pos']:.4f}**."
"""
        if "PHẦN 3: BỔ SUNG KỊCH BẢN THUYẾT TRÌNH SLIDE 12" not in old_script:
            script_path.write_text(old_script + vi_section, encoding="utf-8")

    print("Updated slides.pptx (with Vietnamese Extension slide 12) and speaker_script.md.")


if __name__ == "__main__":
    build_slides_and_script()
