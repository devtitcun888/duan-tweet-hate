"""B2 & B3 Generator: 15-Slide PowerPoint Presentation (reports/slides.pptx) & Speaker Script (reports/speaker_script.md)."""

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
ACCENT_RED = RGBColor(198, 40, 40)
LIGHT_BG = RGBColor(245, 247, 250)
WHITE = RGBColor(255, 255, 255)


def _add_header(slide, prs, title_text: str, slide_num: int, total_slides: int = 15) -> None:
    """Add a styled top banner and footer page number to a content slide."""
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
    p.font.size = Pt(22)
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
    """Add up to 5 concise bullet lines to a slide."""
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

    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.3), Inches(5.0))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    lines_s1 = [
        ("BÁO CÁO ĐỒ ÁN MÔN HỌC: KHAI PHÁ VĂN BẢN & HỌC MÁY", 18, False, RGBColor(180, 210, 255)),
        ("GẮN NHÃN CÁC ĐOẠN TWEET CÓ NỘI DUNG PHÂN BIỆT CHỦNG TỘC, PHÂN BIỆT GIỚI TÍNH", 28, True, WHITE),
        ("(Twitter Sentimental Analysis — Binary Classification / Text Mining)", 20, False, RGBColor(255, 213, 79)),
        (f"• Mô hình chốt: {bm['name']} (Threshold = {bm['threshold']:.3f}) | Test F1 = {ft['f1_pos']:.4f} | PR-AUC = {ft['pr_auc']:.4f}", 16, False, WHITE),
        ("• Thực hiện: Nhóm Đồ Án Khoa Học Dữ Liệu | Môi trường: Python & Google Colab (Seed = 42)", 15, False, RGBColor(210, 225, 245)),
    ]
    for idx, (txt, sz, is_b, col) in enumerate(lines_s1):
        p = tf1.paragraphs[0] if idx == 0 else tf1.add_paragraph()
        p.text = txt
        p.font.name = "Arial"
        p.font.size = Pt(sz)
        p.font.bold = is_b
        p.font.color.rgb = col
        p.space_after = Pt(18)

    _set_notes(
        s1,
        [
            "Chào thầy/cô và các bạn. Hôm nay nhóm xin trình bày đề tài: Gắn nhãn các đoạn tweet có nội dung phân biệt chủng tộc, phân biệt giới tính.",
            "Đây là bài toán phân loại nhị phân trong khai phá văn bản với dữ liệu mạng xã hội mất cân bằng mạnh.",
            f"Kết quả chính: Mô hình {bm['name']} đạt F1 lớp dương = {ft['f1_pos']:.4f} và PR-AUC = {ft['pr_auc']:.4f} trên tập Test độc lập.",
        ],
    )

    # =========================================================================
    # SLIDE 2: Bài toán & mục tiêu
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    _add_header(s2, prs, "Bài toán & Mục tiêu Nghiên cứu", 2)
    _add_bullets(
        s2,
        [
            "Bối cảnh: Ngôn từ thù ghét chủng tộc/giới tính (Racist/Sexist Hate Speech) lan truyền nhanh trên mạng xã hội X/Twitter.",
            "Bài toán: Phân loại nhị phân văn bản ngắn — Nhãn 1 (Phản cảm: kỳ thị chủng tộc/giới tính) vs Nhãn 0 (Bình thường).",
            "Mục tiêu 1 (Câu 1): Khám phá dữ liệu & dùng truy vấn SQL tìm điểm chung định lượng của tweet phản cảm.",
            "Mục tiêu 2 (Câu 2 & 3): Làm sạch chống rò rỉ, cân bằng dữ liệu, so sánh 9 mô hình cổ điển + 2 mô hình học sâu.",
            "Mục tiêu 3 (Câu 3 & 4): Kiểm thử thực tế 300 mẫu (đo Cohen's Kappa) và xây dựng ứng dụng Web Streamlit.",
        ],
        left=0.7,
        top=1.35,
        width=11.8,
        height=5.4,
        font_size=18,
    )
    _set_notes(
        s2,
        [
            "Nêu rõ vì sao cần tự động sàng lọc tweet phản cảm: khối lượng hàng trăm triệu tweet/ngày khiến kiểm duyệt thủ công không xuể.",
            "Bám sát 4 câu trong thang điểm rubric (2đ + 3đ + 3đ + 2đ = 10 điểm).",
        ],
    )

    # =========================================================================
    # SLIDE 3: Dữ liệu: nguồn, kích thước, vấn đề nhãn bộ được giao, hướng xử lý
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    _add_header(s3, prs, "Nguồn Dữ liệu & Phát hiện Quan trọng về Nhãn", 3)
    _add_bullets(
        s3,
        [
            f"Bộ được giao (Twitter_Data.csv): {ds_tw['n_rows']:,} dòng, chỉ có 2 cột clean_text và category (-1/0/+1 sentiment chính trị).",
            "Phát hiện cốt lõi: Bộ được giao KHÔNG có nhãn racist/sexist (nhãn -1 là bình luận tiêu cực về bầu cử Ấn Độ/Modi).",
            f"Giải pháp (Phương án A): Huấn luyện trên bộ Tweets Hate Speech Detection ({ds_hate['n_rows']:,} dòng, nhãn 0/1 chuẩn).",
            f"Tận dụng bộ được giao & Reddit ({facts['datasets']['assigned_reddit']['n_rows']:,} dòng) cho Câu 1 (EDA) & Câu 3 (kiểm thử lệch miền).",
            f"Bổ sung bộ Davidson et al. ({facts['datasets']['davidson_cross_domain']['n_rows']:,} dòng) để kiểm tra chéo ngoài miền.",
        ],
        left=0.5,
        top=1.2,
        width=5.8,
        height=5.6,
        font_size=15,
    )
    _add_image_safe(s3, FIG_DIR / "P2_01_dataset_quality.png", left=6.45, top=1.35, width=6.5)
    _set_notes(
        s3,
        [
            f"Giải thích trung thực với giảng viên: Twitter_Data.csv ({ds_tw['n_rows']:,} dòng) chỉ có sentiment -1/0/+1 về chính trị Ấn Độ.",
            f"Do đó nhóm áp dụng Phương án A: dùng hate_train.csv ({ds_hate['n_rows']:,} mẫu có nhãn racist/sexist 0/1) để huấn luyện, và dùng Twitter_Data cho phân tích EDA + kiểm thử thực tế khác miền.",
        ],
    )

    # =========================================================================
    # SLIDE 4: Thống kê mô tả (phân bố nhãn, độ dài)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    _add_header(s4, prs, "Thống kê Mô tả: Phân bố Nhãn & Đặc trưng Bề mặt", 4)
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
    _set_notes(
        s4,
        [
            f"Nhấn mạnh con số {ds_hate['positive_rate_pct']:.2f}% lớp dương (tỉ lệ 13.26:1).",
            "Chỉ ra hành vi tấn công cá nhân qua @user (65.03% vs 46.46%) và câu hỏi tu từ '?'.",
        ],
    )

    # =========================================================================
    # SLIDE 5: Điểm chung của tweet phản cảm (từ khóa SQL, WordCloud đã che)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    _add_header(s5, prs, "Phân tích SQL: Điểm chung của Tweet Phản cảm", 5)
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
    _set_notes(
        s5,
        [
            f"Trình bày kết quả 4 câu lệnh SQL: đặc biệt là tỉ lệ {lex1:.2f}% vs {lex0:.2f}% trong Lexicon.",
            "Giải thích vì sao 79.41% tweet phản cảm không dùng từ tục trực tiếp là lý do bắt buộc phải dùng Học máy.",
        ],
    )

    # =========================================================================
    # SLIDE 6: Tiền xử lý & cân bằng dữ liệu
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    _add_header(s6, prs, "Tiền xử lý 12 bước, Khử trùng & Cân bằng Dữ liệu", 6)
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
    _set_notes(
        s6,
        [
            f"Nhấn mạnh việc khử trùng {sp['n_near_duplicate_rows_dropped']:,} tweet trước khi chia tập giúp tránh F1 ảo.",
            "Mọi bước cân bằng (ROS/SMOTE) chỉ đặt trong imblearn.Pipeline và chỉ fit trên tập Train.",
        ],
    )

    # =========================================================================
    # SLIDE 7: Thiết kế thực nghiệm (chia tập, metric, vì sao F1/PR-AUC)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    _add_header(s7, prs, "Thiết kế Thực nghiệm & Tiêu chí Đánh giá", 7)
    _add_bullets(
        s7,
        [
            f"Chia tập Stratified 70/15/15 (seed=42): Train = {sp['splits']['train']['n_rows']:,} | Val = {sp['splits']['val']['n_rows']:,} | Test = {sp['splits']['test']['n_rows']:,} mẫu.",
            "Tỉ lệ lớp 1 đồng nhất tuyệt đối giữa 3 tập: 6.52% (kiểm chứng tự động qua unit test test_leakage.py).",
            "Chống rò rỉ nghiêm ngặt: Vectorizer, Sampler, Hyperparameter & Ngưỡng chỉ học từ Train/Val; Test chỉ chạy 1 lần ở P8.",
            "Vì sao KHÔNG dùng Accuracy? Đoán toàn bộ lớp 0 vẫn đạt 93.48% Accuracy nhưng F1 lớp dương = 0.0000.",
            "Tiêu chí chính: F1 lớp dương (f1_pos) kết hợp PR-AUC (nhạy với lớp thiểu số hơn ROC-AUC) và Recall lớp 1.",
        ],
        left=0.5,
        top=1.2,
        width=5.8,
        height=5.6,
        font_size=15,
    )
    _add_image_safe(s7, FIG_DIR / "P4_length_before_after.png", left=6.4, top=1.4, width=6.5)
    _set_notes(
        s7,
        [
            "Giải thích rõ quy trình chống Data Leakage 4 tầng: khử trùng trước chia tập -> Pipeline -> chọn ngưỡng trên Val -> chạy Test đúng 1 lần.",
            "Giải thích sự khác nhau giữa ROC-AUC và PR-AUC khi lớp âm chiếm 93.48%.",
        ],
    )

    # =========================================================================
    # SLIDE 8: Các mô hình đã thử
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    _add_header(s8, prs, "Hệ thống 11 Mô hình Thực nghiệm (Cổ điển & Học sâu)", 8)
    _add_bullets(
        s8,
        [
            "Nhóm Tuyến tính & Xác suất: Logistic Regression (LR), Calibrated Linear SVM, Complement Naive Bayes (CNB).",
            "Nhóm Phi tuyến & Khoảng cách: RBF-Kernel SVM (trên mẫu 8k), Cosine K-Nearest Neighbors (KNN).",
            "Nhóm Cây & Boosting: Decision Tree (DT), Random Forest (RF), XGBoost (hist), LightGBM.",
            "Biểu diễn đặc trưng: So sánh Word TF-IDF (1-2 gram, 50k chiều) và Hợp nhất Word + Char TF-IDF (2-5 char_wb).",
            "Mô hình Điểm cộng ngoài chương trình (P7): Mạng nơ-ron hồi quy BiLSTM (PyTorch) & Fine-tuned DistilBERT.",
        ],
        left=0.7,
        top=1.35,
        width=11.8,
        height=5.4,
        font_size=17,
    )
    _set_notes(
        s8,
        [
            "Tổng cộng dự án đã thực nghiệm 9 mô hình học máy cổ điển (kết hợp 2 loại vectorizer = 12 cấu hình) + 2 mô hình học sâu (BiLSTM, DistilBERT).",
            "Giải thích ngắn gọn lý thuyết từng nhóm mô hình.",
        ],
    )

    # =========================================================================
    # SLIDE 9: Bảng/biểu đồ so sánh kết quả
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    _add_header(s9, prs, "Kết quả So sánh các Mô hình trên Tập Validation", 9)
    _add_bullets(
        s9,
        [
            "Top 1 (Trước tuning): LR (word+char) đạt Val F1 = 0.6486 (PR-AUC = 0.7090), vượt trội nhờ đặc trưng ký tự.",
            "Top 2 & 3 (Word TF-IDF): LR (F1 = 0.5832), Calibrated Linear SVM (F1 = 0.5727, Precision cao = 0.7529).",
            "Complement NB (word+char): đạt PR-AUC cao nhất trước tuning (0.7403) và Recall = 84.48%.",
            "Nhóm Cây/Boosting (LightGBM F1=0.5260, RF F1=0.5195, XGB F1=0.5034, DT F1=0.3836) kém hơn trên TF-IDF thưa.",
            "Mô hình Điểm cộng (P7): BiLSTM đạt Val F1 = "
            + f"{bonus['bilstm']['f1_pos']:.4f}, DistilBERT đạt Val F1 = {bonus['distilbert']['f1_pos']:.4f}.",
        ],
        left=0.45,
        top=1.15,
        width=5.6,
        height=5.8,
        font_size=14,
    )
    _add_image_safe(s9, FIG_DIR / "P5_f1_prauc_bars.png", left=6.15, top=1.15, width=6.85)
    _add_image_safe(s9, FIG_DIR / "P5_roc_pr_curves.png", left=6.15, top=4.2, width=6.85)
    _set_notes(
        s9,
        [
            "Chỉ vào biểu đồ cột P5_f1_prauc_bars.png để giải thích vì sao các mô hình tuyến tính (LR, Linear SVM, NB) vượt xa Decision Tree, Random Forest và KNN trên ma trận TF-IDF thưa.",
        ],
    )

    # =========================================================================
    # SLIDE 10: Tinh chỉnh tham số & mô hình được chọn
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    _add_header(s10, prs, "Tinh chỉnh Siêu tham số (P6) & Tối ưu Ngưỡng Quyết định", 10)
    _add_bullets(
        s10,
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
    _add_image_safe(s10, FIG_DIR / "P6_threshold_tuning.png", left=6.15, top=1.15, width=6.85)
    _add_image_safe(s10, FIG_DIR / "P6_before_after.png", left=6.15, top=4.15, width=6.85)
    _set_notes(
        s10,
        [
            f"Giải thích vì sao ngưỡng 0.50 không tối ưu cho dữ liệu lệch lớp: tại ngưỡng 0.50, Linear SVM có Precision 0.7368 nhưng Recall chỉ 0.5054. Khi hạ ngưỡng xuống {bm['threshold']:.3f} trên tập Val, Recall tăng lên 0.6282 và F1 đạt đỉnh {bm['val_metrics_after_tuning_best_thr']['f1_pos']:.4f}.",
        ],
    )

    # =========================================================================
    # SLIDE 11: Chạy thử thực tế (bộ tự gắn nhãn, kappa, domain shift)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    _add_header(s11, prs, "Đánh giá Cuối trên Tập Test (P8) & Chạy thử Thực tế (P9)", 11)
    ci_f1 = ft["ci95"]["f1_pos"]
    _add_bullets(
        s11,
        [
            f"Kết quả Test độc lập (N=4,250, chạy 1 lần): F1 = {ft['f1_pos']:.4f} (95% CI: [{ci_f1['ci95_low']:.4f}, {ci_f1['ci95_high']:.4f}]), PR-AUC = {ft['pr_auc']:.4f}, ROC-AUC = {ft['roc_auc']:.4f}.",
            f"Bộ thực tế (N={rw['n']} mẫu từ 4 nguồn A/B/C/D): 2 người gắn nhãn độc lập đạt đồng thuận {rw['agreement_pct']:.2f}%, Cohen's Kappa κ = {rw['kappa']:.4f}.",
            f"Hiệu năng thực tế tổng thể: F1 = {rw['overall_metrics']['f1_pos']:.4f} (Precision = {rw['overall_metrics']['precision_pos']:.4f}, Recall = {rw['overall_metrics']['recall_pos']:.4f}, PR-AUC = {rw['overall_metrics']['pr_auc']:.4f}).",
            "Phân tích lệch miền (Domain Shift): Nguồn A cùng miền đạt F1 cao hơn nguồn D (câu mỉa mai/phủ định hiểm hóc).",
            "Trên Twitter_Data (162,969 dòng): chỉ một tỉ lệ nhỏ câu sentiment = -1 được dự đoán là hate speech ⇒ khẳng định khác biệt bản chất.",
        ],
        left=0.45,
        top=1.15,
        width=5.6,
        height=5.8,
        font_size=13,
    )
    _add_image_safe(s11, FIG_DIR / "P8_test_evaluation.png", left=6.15, top=1.15, width=6.85)
    _add_image_safe(s11, FIG_DIR / "P9_realworld_evaluation.png", left=6.15, top=4.1, width=6.85)
    _set_notes(
        s11,
        [
            f"Trình bày con số chốt trên tập Test: F1 = {ft['f1_pos']:.4f} (Precision = {ft['precision_pos']:.4f}, Recall = {ft['recall_pos']:.4f}).",
            f"Trình bày kết quả chạy thử thực tế 300 mẫu: Kappa = {rw['kappa']:.4f}, F1 = {rw['overall_metrics']['f1_pos']:.4f}.",
        ],
    )

    # =========================================================================
    # SLIDE 12: Phân tích lỗi & kiểm tra thiên lệch
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    _add_header(s12, prs, "Phân tích Lỗi (FP/FN), Giải thích Trọng số & Thiên lệch Nhóm", 12)
    _add_bullets(
        s12,
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
    _add_image_safe(s12, FIG_DIR / "P8_error_causes.png", left=6.15, top=1.15, width=6.85)
    _add_image_safe(s12, FIG_DIR / "P8_subgroup_bias.png", left=6.15, top=4.1, width=6.85)
    _set_notes(
        s12,
        [
            "Phân tích trung thực điểm yếu của mô hình: hiện tượng False Positive trên câu counter-speech (phản đối phân biệt chủng tộc) và False Negative trên câu kỳ thị ngầm.",
        ],
    )

    # =========================================================================
    # SLIDE 13: Demo ứng dụng
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    _add_header(s13, prs, "Ứng dụng Web Streamlit Hỗ trợ Sàng lọc Tweet (P10)", 13)
    _add_bullets(
        s13,
        [
            "Kiến trúc 3 Tab tiếng Việt trực quan, nạp mô hình bằng @st.cache_resource (suy luận 1,000 tweet < 0.1 giây).",
            "Tab 1 (Phân loại 1 Tweet): Nhập văn bản / chọn 5 câu mẫu → Nhãn, thanh xác suất so với ngưỡng 0.315, tô màu Top-8 từ đóng góp.",
            "Tab 2 (Phân loại Hàng loạt): Upload file CSV (tối đa 20,000 dòng) → Thanh tiến trình, thống kê tỉ lệ, tải file CSV kết quả.",
            "Tab 3 (Thống kê & Hồ sơ Mô hình): Hiển thị model_card.json, bảng so sánh 11 mô hình, biểu đồ đánh giá Test.",
            "Tích hợp cảnh báo đạo đức ở chân trang: 'Mô hình có thể sai; chỉ hỗ trợ sàng lọc, cần người kiểm tra.'",
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
    else:
        _add_image_safe(s13, FIG_DIR / "P8_feature_importance.png", left=6.15, top=1.2, width=6.85)
    _set_notes(
        s13,
        [
            "Chuyển sang màn hình trình duyệt để demo trực tiếp ứng dụng Streamlit theo 5 bước trong demo_script.md.",
            "Nếu mạng/máy chiếu gặp sự cố, sử dụng ngay ảnh chụp màn hình và kết quả đã lưu sẵn.",
        ],
    )

    # =========================================================================
    # SLIDE 14: Kết luận, hạn chế, hướng phát triển
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    _add_header(s14, prs, "Kết luận, Hạn chế & Hướng Phát triển", 14)
    _add_bullets(
        s14,
        [
            f"Kết luận: Hoàn thành trọn vẹn quy trình P0→P12 không rò rỉ dữ liệu; {bm['name']} đạt Test F1 = {ft['f1_pos']:.4f}, PR-AUC = {ft['pr_auc']:.4f}.",
            "Đóng góp nổi bật: Phát hiện & xử lý vấn đề nhãn bộ được giao; phân tích SQL + Lexicon; kiểm thử thực tế 300 mẫu (κ = "
            + f"{rw['kappa']:.4f}).",
            "Hạn chế 1: Nhãn nhị phân hiện gộp chung racist và sexist; mô hình TF-IDF còn nhạy với từ khóa nhóm người (Subgroup Bias).",
            "Hạn chế 2: Chưa bắt hết câu mỉa mai sâu (sarcasm) hoặc kỳ thị ngầm hoàn toàn không có từ khóa đặc trưng.",
            "Hướng phát triển: Mở rộng phân loại đa nhãn (Racist / Sexist / Offensive / Normal), kết hợp RoBERTa/HateBERT và Tiếng Việt (ViHateT5).",
        ],
        left=0.7,
        top=1.35,
        width=11.8,
        height=5.4,
        font_size=16,
    )
    _set_notes(
        s14,
        [
            "Tóm tắt 3 đóng góp lớn nhất của đồ án và 2 hạn chế thực tế cùng hướng phát triển lên bài toán đa lớp và tiếng Việt.",
        ],
    )

    # =========================================================================
    # SLIDE 15: Cảm ơn + Q&A
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    bg15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
    bg15.fill.solid()
    bg15.fill.fore_color.rgb = NAVY
    bg15.line.fill.background()

    tb15 = s15.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.2))
    tf15 = tb15.text_frame
    tf15.word_wrap = True
    lines_s15 = [
        ("XIN CHÂN THÀNH CẢM ƠN THẦY/CÔ VÀ CÁC BẠN ĐÃ LẮNG NGHE!", 28, True, WHITE),
        ("HỎI & ĐÁP (Q&A SESSION)", 24, True, RGBColor(255, 213, 79)),
        (
            f"Số liệu tóm tắt: Dữ liệu sạch = {sp['n_final_unique_clean']:,} mẫu (Lớp 1 = {sp['final_pos_rate_pct']:.2f}%) | "
            f"Mô hình = {bm['name']} (Thr = {bm['threshold']:.3f})",
            16,
            False,
            WHITE,
        ),
        (
            f"Val F1 (trước → sau tuning): {bm['val_metrics_before_tuning']['val_f1_before']:.4f} → {bm['val_metrics_after_tuning_best_thr']['f1_pos']:.4f} | "
            f"Test F1 = {ft['f1_pos']:.4f} | Thực tế F1 = {rw['overall_metrics']['f1_pos']:.4f} (Kappa = {rw['kappa']:.4f})",
            16,
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
        p.space_after = Pt(20)

    _set_notes(
        s15,
        [
            "Cảm ơn giảng viên và mời hội đồng đặt câu hỏi. Giữ màn hình ở slide này vì có sẵn 6 con số chốt để tra cứu nhanh.",
        ],
    )

    out_pptx = REPORTS_DIR / "slides.pptx"
    prs.save(str(out_pptx))

    # =========================================================================
    # B3: Generate reports/speaker_script.md
    # =========================================================================
    script_md = f"""# KỊCH BẢN THUYẾT TRÌNH CHI TIẾT TỪNG SLIDE (SPEAKER SCRIPT — P11 / B3)

> **Thời lượng chuẩn:** 10 – 12 phút (15 slides, trung bình 40–50 giây/slide).  
> **Quy tắc trình bày:** Không đọc nguyên văn chữ trên slide; nhìn vào giảng viên/hội đồng, chỉ tay vào biểu đồ khi nêu số liệu. Mọi con số dưới đây được đồng bộ 100% từ `reports/facts.json`.

---

## PHẦN 1: KỊCH BẢN ĐẦY ĐỦ 10–12 PHÚT

### Slide 1: Tiêu đề & Giới thiệu đề tài `[00:00 – 00:40 | 40 giây]`
- **Lời nói:** "Em xin kính chào thầy/cô và các bạn. Hôm nay nhóm em xin trình bày báo cáo đồ án môn học với đề tài: *Gắn nhãn các đoạn tweet có nội dung phân biệt chủng tộc, phân biệt giới tính (Twitter Sentimental Analysis)*. Đây là bài toán phân loại nhị phân trong khai phá văn bản thực tế. Kết quả cốt lõi của nhóm là xây dựng được quy trình học máy chống rò rỉ dữ liệu hoàn chỉnh, đạt F1 lớp dương trên tập Test là **{ft['f1_pos']:.4f}** và PR-AUC đạt **{ft['pr_auc']:.4f}**, kèm ứng dụng web sàng lọc trực quan."
- **Câu chuyển slide:** "Trước hết, em xin đi vào bối cảnh bài toán và 4 mục tiêu chính của đề tài ở Slide 2."

### Slide 2: Bài toán & Mục tiêu nghiên cứu `[00:40 – 01:20 | 40 giây]`
- **Lời nói:** "Trên mạng xã hội X/Twitter, các nội dung phân biệt chủng tộc và phân biệt giới tính gây tổn hại nghiêm trọng đến cộng đồng, trong khi kiểm duyệt thủ công không thể xử lý xuể hàng trăm triệu bài đăng mỗi ngày. Do đó, nhóm xây dựng mô hình phân loại nhị phân: gán nhãn `1` cho tweet phản cảm (racist/sexist) và nhãn `0` cho tweet bình thường. Nội dung báo cáo bám sát 4 câu trong đề bài: từ thu thập & phân tích SQL (Câu 1), làm sạch & cân bằng dữ liệu (Câu 2), huấn luyện & tối ưu 11 mô hình học máy và chạy thử thực tế (Câu 3), đến triển khai ứng dụng Streamlit (Câu 4)."
- **Câu chuyển slide:** "Khi bắt tay vào kiểm tra dữ liệu được giao ở Câu 1, nhóm đã phát hiện một vấn đề rất quan trọng về nhãn dữ liệu, xin mời thầy/cô xem ở Slide 3."

### Slide 3: Nguồn dữ liệu & Phát hiện về nhãn bộ được giao `[01:20 – 02:15 | 55 giây]`
- **Lời nói:** "Khi tải bộ dữ liệu được giao trong đề (`cosmos98/twitter-and-reddit-sentimental-analysis-dataset`), nhóm kiểm tra thực tế và xác nhận file `Twitter_Data.csv` có **{ds_tw['n_rows']:,} dòng**, nhưng chỉ gồm 2 cột `clean_text` và `category` nhận giá trị `-1, 0, +1`. Nội dung thực chất là bình luận cảm xúc chính trị xoay quanh bầu cử Ấn Độ và thủ tướng Modi, hoàn toàn **không có nhãn phân biệt chủng tộc hay giới tính**. Nếu lấy đại nhãn `-1` làm nhãn phản cảm thì mô hình sẽ học sai bản chất đề tài. Vì vậy, nhóm áp dụng **Phương án A**: sử dụng bộ dữ liệu chuẩn *Tweets Hate Speech Detection* (**{ds_hate['n_rows']:,} dòng**, có nhãn `0/1` racist/sexist) để huấn luyện mô hình; đồng thời giữ nguyên bộ `Twitter_Data` (**{ds_tw['n_rows']:,} dòng**), `Reddit_Data` (**{facts['datasets']['assigned_reddit']['n_rows']:,} dòng**) và bộ *Davidson et al.* (**{facts['datasets']['davidson_cross_domain']['n_rows']:,} dòng**) để làm thống kê Câu 1 và kiểm thử độ lệch miền ở Câu 3."
- **Câu chuyển slide:** "Chi tiết các con số thống kê mô tả của hai bộ dữ liệu này được thể hiện ở Slide 4."

### Slide 4: Thống kê mô tả: Phân bố nhãn & Đặc trưng bề mặt `[02:15 – 03:05 | 50 giây]`
- **Lời nói:** "Nhìn vào biểu đồ tròn ở góc phải trên, đặc điểm lớn nhất của bộ dữ liệu Hate Speech là **mất cân bằng lớp rất mạnh**: Lớp 0 (Bình thường) chiếm tới **{ds_hate['label_distribution']['0']:,} mẫu (92.99%)**, trong khi Lớp 1 (Phản cảm) chỉ có **{ds_hate['label_distribution']['1']:,} mẫu ({ds_hate['positive_rate_pct']:.2f}%)**, tức tỉ lệ lệch **{ds_hate['imbalance_ratio_neg_to_pos']}:1**. Ngoài ra, dữ liệu thô có **{ds_hate['n_duplicates_text']:,} dòng trùng nguyên bản**. Ở biểu đồ góc dưới, kiểm định thống kê Chi-square và Mann-Whitney U cho thấy sự khác biệt rất có ý nghĩa ($p < 10^{{-10}}$): **65.03%** tweet phản cảm có gắn thẻ `@user` (so với **46.46%** ở lớp bình thường) và **18.78%** chứa dấu hỏi `?` mang tính chất vấn, mỉa mai."
- **Câu chuyển slide:** "Để trả lời Yêu cầu 1 của đề bài là tìm ra điểm chung của các tweet phản cảm bằng SQL, nhóm đã xây dựng cơ sở dữ liệu SQLite ở Slide 5."

### Slide 5: Phân tích SQL: Điểm chung của Tweet phản cảm `[03:05 – 04:00 | 55 giây]`
- **Lời nói:** "Nhóm thiết kế cơ sở dữ liệu `db/tweets.db` gồm 5 bảng và **chỉ nạp tập Train ({sql_f['n_train_tweets_in_db']:,} tweet, {sql_f['n_tokens_in_db']:,} token)** để tuyệt đối không nhìn trộm tập Test. Kết quả chạy 4 truy vấn SQL cho thấy: Thứ nhất, ở truy vấn `40_lexicon.sql`, có **{lex1:.2f}%** tweet Lớp 1 chứa từ trong từ điển xúc phạm công khai, cao gấp **8.6 lần** so với mức **{lex0:.2f}%** của Lớp 0. Tuy nhiên, điều đó cũng có nghĩa là **79.41%** tweet phản cảm không dùng từ tục trực tiếp trong từ điển — nếu chỉ lọc bằng từ khóa cứng sẽ bỏ sót gần 4/5 số mẫu. Thứ hai, truy vấn `20_log_odds.sql` và `30_cooccur.sql` chỉ ra các từ đặc trưng nhất tập trung vào 3 chủ đề: miệt thị chủng tộc/màu da (`racist`, `white`, `black`, `n****r`), miệt thị giới tính (`sexist`, `woman`, `b***h`, `misogyny`), và xung đột tôn giáo/ý thức hệ (`allahsoil`, `islam`, `libtard`)."
- **Câu chuyển slide:** "Từ những phát hiện trên, nhóm thiết kế quy trình làm sạch văn bản 12 bước và cân bằng dữ liệu ở Slide 6."

### Slide 6: Tiền xử lý 12 bước, Khử trùng & Cân bằng dữ liệu `[04:00 – 04:55 | 55 giây]`
- **Lời nói:** "Trong hàm `clean_text()`, điểm kỹ thuật quan trọng nhất của nhóm là **loại bỏ các từ phủ định (`no, not, never`) và các đại từ/nhóm người (`you, she, they, women, men`) khỏi danh sách stopwords của NLTK**, vì trong bài toán hate speech, việc xóa chữ `not` hoặc chữ `women` sẽ làm đảo ngược hoàn toàn nghĩa câu. Đặc biệt, nhóm thực hiện khử trùng dựa trên văn bản đã làm sạch trước khi chia tập: loại bỏ **{sp['n_near_duplicate_rows_dropped']:,} mẫu trùng lặp/retweet** và **{sp['n_conflict_rows_dropped']} mẫu có nhãn mâu thuẫn**, thu được **{sp['n_final_unique_clean']:,} mẫu sạch duy nhất**. Khi so sánh 6 chiến lược cân bằng dữ liệu bằng 5-fold CV trên tập Train, nếu không cân bằng (`none`), Recall chỉ đạt **{bal['table'][0]['recall_mean']*100:.2f}%** (F1 = **{bal['table'][0]['f1_pos_mean']:.4f}**). Khi dùng `RandomOverSampler (ros)` hoặc `class_weight='balanced'`, F1 tăng vọt lên **0.5876** và **0.5742**."
- **Câu chuyển slide:** "Thiết kế chia tập chống rò rỉ và lý do chọn thước đo đánh giá được trình bày ở Slide 7."

### Slide 7: Thiết kế thực nghiệm & Tiêu chí đánh giá `[04:55 – 05:40 | 45 giây]`
- **Lời nói:** "Bộ dữ liệu sạch **{sp['n_final_unique_clean']:,} mẫu** được chia Stratified một lần duy nhất theo tỉ lệ 70/15/15: Tập Train có **{sp['splits']['train']['n_rows']:,} mẫu**, Tập Validation có **{sp['splits']['val']['n_rows']:,} mẫu**, và Tập Test có **{sp['splits']['test']['n_rows']:,} mẫu**, tất cả đều giữ đúng tỉ lệ lớp dương **6.52%**. Nhóm kiểm soát chống rò rỉ dữ liệu bằng `Pipeline` và kiểm chứng bằng unit test tự động: giao nhau giữa 3 tập bằng rỗng. Về tiêu chí đánh giá, nhóm chọn **F1-score lớp dương (`f1_pos`)** và **PR-AUC** làm thước đo chính thay vì Accuracy, vì dữ liệu lệch 93.5% lớp 0."
- **Câu chuyển slide:** "Trên nền tảng thực nghiệm đó, nhóm tiến hành huấn luyện 11 mô hình ở Slide 8."

### Slide 8: Hệ thống 11 Mô hình thực nghiệm `[05:40 – 06:20 | 40 giây]`
- **Lời nói:** "Nhóm thực nghiệm đầy đủ 9 mô hình cổ điển trải rộng trên 4 họ thuật toán: (1) Tuyến tính & Xác suất gồm Logistic Regression, Calibrated Linear SVM, Complement Naive Bayes; (2) Phi tuyến & Khoảng cách gồm RBF SVM và Cosine KNN; (3) Cây quyết định & Rừng ngẫu nhiên; (4) Gradient Boosting gồm XGBoost và LightGBM. Ngoài ra, để đạt điểm cộng mở rộng, nhóm cài đặt thêm 2 mô hình Học sâu bằng PyTorch và HuggingFace là mạng hồi quy hai chiều **BiLSTM** và mô hình ngôn ngữ **DistilBERT fine-tuned**."
- **Câu chuyển slide:** "Kết quả đọ sức giữa toàn bộ các mô hình này trên tập Validation được tổng hợp ở Slide 9."

### Slide 9: Kết quả so sánh các mô hình trên Tập Validation `[06:20 – 07:15 | 55 giây]`
- **Lời nói:** "Nhìn vào biểu đồ so sánh ở bên phải, chúng ta thấy một quy luật rất rõ: nhóm mô hình tuyến tính (`Logistic Regression`, `Linear SVM`, `Complement NB`) vượt trội hoàn toàn so với nhóm cây quyết định (`Random Forest` F1 = 0.5195, `Decision Tree` F1 = 0.3836) và `KNN` (F1 = 0.3902). Lý do là với ma trận TF-IDF 50,000 chiều cực kỳ thưa, siêu phẳng tuyến tính kết hợp được trọng số của hàng nghìn n-gram cùng lúc, trong khi cây quyết định chỉ rẽ nhánh từng từ đơn lẻ nên bị phân mảnh dữ liệu. Khi kết hợp thêm đặc trưng ký tự (`word+char`), Logistic Regression đạt F1 = **0.6486** ngay trước khi tuning. Với hai mô hình điểm cộng, BiLSTM đạt Val F1 = **{bonus['bilstm']['f1_pos']:.4f}** và DistilBERT đạt **{bonus['distilbert']['f1_pos']:.4f}**."
- **Câu chuyển slide:** "Tiếp theo, nhóm đưa 3 mô hình cổ điển dẫn đầu vào bước tinh chỉnh siêu tham số và tối ưu ngưỡng quyết định ở Slide 10."

### Slide 10: Tinh chỉnh siêu tham số (P6) & Tối ưu ngưỡng quyết định `[07:15 – 08:05 | 50 giây]`
- **Lời nói:** "Nhóm sử dụng `RandomizedSearchCV` với 40 tổ hợp tham số và 5-fold CV trên tập Train để tối ưu đồng thời cả tham số của bộ trích xuất TF-IDF (`ngram_range`, `min_df`, `max_features`) lẫn tham số mô hình. Sau đó, trên tập Validation, nhóm quét ngưỡng xác suất từ 0.05 đến 0.95. Biểu đồ góc trên bên phải minh họa rất rõ: với **Linear SVM (Calibrated)** sử dụng `ngram_range=(1,3)`, `min_df=2`, `C=0.50191`, tại ngưỡng mặc định 0.50 mô hình có Precision cao nhưng Recall chỉ đạt 50.54% (F1 = 0.5996). Khi tối ưu ngưỡng quyết định về **`threshold = {bm['threshold']:.3f}`**, Recall tăng lên **62.82%**, đưa Validation F1 đạt đỉnh **{bm['val_metrics_after_tuning_best_thr']['f1_pos']:.4f}** — tăng **+8.39 điểm %** so với baseline ban đầu. Do đó, `Linear SVM (Calibrated)` được chọn làm mô hình chốt `best_model`."
- **Câu chuyển slide:** "Sau khi khóa mô hình chốt, nhóm thực hiện đánh giá một lần duy nhất trên tập Test và chạy thử thực tế ở Slide 11."

### Slide 11: Đánh giá cuối trên Tập Test (P8) & Chạy thử thực tế (P9) `[08:05 – 09:00 | 55 giây]`
- **Lời nói:** "Trên tập Test độc lập gồm **4,250 mẫu**, mô hình chốt đạt **F1 = {ft['f1_pos']:.4f}** với khoảng tin cậy 95% Bootstrap là **[{ci_f1['ci95_low']:.4f}, {ci_f1['ci95_high']:.4f}]**, **Precision = {ft['precision_pos']:.4f}**, **Recall = {ft['recall_pos']:.4f}**, **PR-AUC = {ft['pr_auc']:.4f}** và **ROC-AUC = {ft['roc_auc']:.4f}**. Tiếp theo, để kiểm thử thực tế theo Yêu cầu 4, nhóm xây dựng bộ dữ liệu **300 mẫu** từ 4 nguồn: 100 mẫu từ tập test chưa gán nhãn, 100 mẫu từ `Twitter_Data`, 50 mẫu từ `Reddit_Data`, và 50 câu hiểm hóc tự biên soạn. Hai người gắn nhãn độc lập đạt độ đồng thuận **{rw['agreement_pct']:.2f}%**, hệ số **Cohen's Kappa $\\kappa = {rw['kappa']:.4f}$**. Khi chạy mô hình trên 300 mẫu thực tế này, F1 tổng thể đạt **{rw['overall_metrics']['f1_pos']:.4f}** (PR-AUC = **{rw['overall_metrics']['pr_auc']:.4f}**). Đặc biệt, biểu đồ góc dưới bên phải chứng minh trên 162,969 dòng `Twitter_Data.csv`, chỉ một phần nhỏ các câu `sentiment = -1` bị dự đoán là phản cảm, khẳng định thực nghiệm rằng cảm xúc tiêu cực chính trị hoàn toàn khác với phân biệt chủng tộc/giới tính."
- **Câu chuyển slide:** "Để hiểu sâu hơn mô hình sai ở đâu và có thiên lệch hay không, nhóm thực hiện phân tích lỗi ở Slide 12."

### Slide 12: Phân tích lỗi (FP/FN), Giải thích trọng số & Thiên lệch nhóm `[09:00 – 09:50 | 50 giây]`
- **Lời nói:** "Nhóm đã trích xuất các lỗi Báo động giả (FP = {ft['fp']}) và Bỏ sót (FN = {ft['fn']}) trên tập Test để phân loại nguyên nhân. Biểu đồ góc trên bên phải cho thấy: lỗi Bỏ sót (FN) chủ yếu rơi vào các câu **kỳ thị ngầm, định kiến xã hội hoặc châm biếm mỉa mai** mà không dùng từ tục trực tiếp; trong khi lỗi Báo động giả (FP) thường xảy ra khi câu bình thường hoặc câu **lên án nạn phân biệt chủng tộc (counter-speech)** có chứa các từ nhạy cảm như `women`, `black`, `racist`. Phân tích công bằng mô hình (Subgroup Bias) ở góc dưới cũng chỉ ra tỉ lệ báo động giả trên các câu bình thường có từ chỉ nhóm chủng tộc/giới tính cao hơn mức trung bình toàn tập Test — đây là điểm cần con người phối hợp kiểm duyệt."
- **Câu chuyển slide:** "Để hỗ trợ con người sàng lọc nhanh và minh bạch, nhóm đã đóng gói mô hình thành ứng dụng Web Streamlit ở Slide 13."

### Slide 13: Demo Ứng dụng Web Streamlit `[09:50 – 10:35 | 45 giây]`
- **Lời nói:** "Ứng dụng Streamlit của nhóm gồm 3 tab bằng tiếng Việt: Tab 1 cho phép kiểm tra từng đoạn tweet, hiển thị nhãn cảnh báo đỏ/xanh, thanh mức độ rủi ro so với ngưỡng `{bm['threshold']:.3f}` và đặc biệt là giải thích trực tiếp Top-8 từ đóng góp (`TF-IDF × Hệ số`) giúp người kiểm duyệt biết ngay vì sao mô hình đưa ra phán đoán. Tab 2 cho phép tải lên file CSV tới 20,000 dòng để gắn nhãn hàng loạt chỉ trong vài giây và xuất file kết quả. Tab 3 cung cấp toàn bộ hồ sơ mô hình `model_card.json` và bảng đối sánh thực nghiệm."
- **Câu chuyển slide:** "Từ toàn bộ kết quả trên, em xin rút ra kết luận và hướng phát triển ở Slide 14."

### Slide 14: Kết luận, Hạn chế & Hướng phát triển `[10:35 – 11:10 | 35 giây]`
- **Lời nói:** "Tóm lại, đồ án đã giải quyết trọn vẹn bài toán phân loại tweet phân biệt chủng tộc/giới tính: từ việc phát hiện đúng bản chất nhãn của bộ dữ liệu được giao, phân tích SQL tìm quy luật từ vựng, chống rò rỉ dữ liệu nghiêm ngặt, so sánh 11 mô hình để đạt Test F1 = **{ft['f1_pos']:.4f}**, đến kiểm thử thực tế 300 mẫu đạt Kappa = **{rw['kappa']:.4f}** và triển khai ứng dụng Streamlit. Trong tương lai, nhóm đề xuất mở rộng lên bài toán phân loại đa nhãn (tách riêng Racist, Sexist, Offensive) và kết hợp kiến trúc RoBERTa/HateBERT cùng dữ liệu *counter-speech* để giảm thiểu thiên lệch từ khóa."
- **Câu chuyển slide:** "Em xin kết thúc phần trình bày tại Slide 15."

### Slide 15: Cảm ơn & Hỏi đáp (Q&A) `[11:10 – 11:30 | 20 giây]`
- **Lời nói:** "Em xin chân thành cảm ơn thầy/cô và các bạn đã lắng nghe. Trên màn hình là bảng tóm tắt các con số cốt lõi của dự án, nhóm em rất mong nhận được các câu hỏi và góp ý từ thầy/cô ạ!"

---

## PHẦN 2: PHIÊN BẢN RÚT GỌN 5 PHÚT (DÙNG KHI GIẢNG VIÊN YÊU CẦU TRÌNH BÀY NHANH)

1. **Phút 1 (Slide 1–3 — Bài toán & Dữ liệu):** Nêu mục tiêu phân loại nhị phân tweet phân biệt chủng tộc/giới tính. Chỉ ra phát hiện quan trọng: `Twitter_Data.csv` ({ds_tw['n_rows']:,} dòng) chỉ có nhãn sentiment chính trị `-1/0/+1`, nên nhóm dùng bộ `hate_train.csv` ({ds_hate['n_rows']:,} dòng, tỉ lệ lớp 1 = **{ds_hate['positive_rate_pct']:.2f}%**, lệch **13.26:1**) để huấn luyện và dùng `Twitter_Data` để phân tích + kiểm thử lệch miền.
2. **Phút 2 (Slide 4–6 — SQL & Tiền xử lý chống rò rỉ):** Nêu kết quả SQL trên tập Train: **20.59%** tweet phản cảm chứa từ trong Lexicon (gấp **8.6 lần** lớp 0), nhưng **79.41%** không dùng từ tục trực tiếp. Pipeline làm sạch 12 bước giữ lại từ phủ định/đại từ, khử trùng **3,628 mẫu retweet** trước khi chia tập 70/15/15 ({sp['n_final_unique_clean']:,} mẫu sạch), và chọn `RandomOverSampler` qua 5-fold CV (F1 = **0.5876**).
3. **Phút 3 (Slide 8–10 — So sánh 11 Mô hình & Tinh chỉnh):** So sánh 9 mô hình cổ điển + BiLSTM & DistilBERT bằng F1 lớp dương và PR-AUC. Các mô hình tuyến tính vượt trội so với cây quyết định trên TF-IDF thưa. Sau `RandomizedSearchCV` và tối ưu ngưỡng trên Val (`threshold = {bm['threshold']:.3f}`), mô hình chốt **{bm['name']}** tăng Val F1 từ **0.5727** lên **{bm['val_metrics_after_tuning_best_thr']['f1_pos']:.4f}** (+8.39 điểm %).
4. **Phút 4 (Slide 11–12 — Kết quả Test, Thực tế & Phân tích lỗi):** Chạy Test độc lập (4,250 mẫu) đúng 1 lần đạt **F1 = {ft['f1_pos']:.4f}** (PR-AUC = **{ft['pr_auc']:.4f}**). Chạy thử bộ thực tế 300 mẫu (4 nguồn A/B/C/D, Cohen's Kappa $\\kappa = {rw['kappa']:.4f}$) đạt **F1 = {rw['overall_metrics']['f1_pos']:.4f}**. Chỉ rõ nguyên nhân lỗi FP (từ nhạy cảm trong câu phản đối kỳ thị) và FN (kỳ thị ngầm/mỉa mai).
5. **Phút 5 (Slide 13–15 — Demo App & Kết luận):** Demo nhanh 3 tab ứng dụng Streamlit (phân loại 1 câu kèm giải thích từ đóng góp, phân loại file CSV hàng loạt, bảng thống kê) và kết luận.
"""
    (REPORTS_DIR / "speaker_script.md").write_text(script_md, encoding="utf-8")
    print("Generated slides.pptx (15 slides) and speaker_script.md.")


if __name__ == "__main__":
    build_slides_and_script()
