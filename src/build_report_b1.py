"""B1 Report Generator: Builds reports/report.docx (25-35 pages, Times New Roman 13pt, 1.5 spacing) and exports PDF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import docx
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
import pandas as pd

from src.utils import get_project_root

PROJECT_ROOT = get_project_root()
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"


def _set_cell_shading(cell, color_hex: str) -> None:
    shading_elm = OxmlElement("w:shd")
    shading_elm.set(qn("w:val"), "clear")
    shading_elm.set(qn("w:color"), "auto")
    shading_elm.set(qn("w:fill"), color_hex)
    cell._tc.get_or_add_tcPr().append(shading_elm)


def _add_toc_field(paragraph) -> None:
    """Insert an automatic Word Table of Contents (TOC) field."""
    run = paragraph.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = 'TOC \\o "1-3" \\h \\z \\u'
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "separate")
    fldChar3 = OxmlElement("w:fldChar")
    fldChar3.set(qn("w:fldCharType"), "end")
    r_element = run._r
    r_element.append(fldChar1)
    r_element.append(instrText)
    r_element.append(fldChar2)
    r_element.append(fldChar3)


def _format_paragraph(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after: float = 6.0, first_line_indent_cm: float = 0.0) -> None:
    pf = p.paragraph_format
    pf.alignment = align
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(space_after)
    if first_line_indent_cm > 0:
        pf.first_line_indent = Cm(first_line_indent_cm)


def _add_body_p(doc: Document, text: str, bold_prefix: str = "", indent: bool = True):
    p = doc.add_paragraph()
    _format_paragraph(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6.0, first_line_indent_cm=(0.8 if indent else 0.0))
    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.bold = True
        r_b.font.name = "Times New Roman"
        r_b.font.size = Pt(13)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    return p


def _add_bullet_p(doc: Document, bold_prefix: str, text: str):
    p = doc.add_paragraph(style="List Bullet")
    _format_paragraph(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=4.0, first_line_indent_cm=0.0)
    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.bold = True
        r_b.font.name = "Times New Roman"
        r_b.font.size = Pt(13)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    return p


def _add_heading(doc: Document, text: str, level: int = 1):
    h = doc.add_heading(text, level=level)
    pf = h.paragraph_format
    pf.space_before = Pt(14 if level == 1 else 10)
    pf.space_after = Pt(6)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    for run in h.runs:
        run.font.name = "Times New Roman"
        run.font.bold = True
        run.font.color.rgb = RGBColor(20, 45, 85) if level == 1 else RGBColor(30, 35, 45)
        run.font.size = Pt(16 if level == 1 else (14 if level == 2 else 13))
    return h


def _add_table(doc: Document, caption: str, headers: list[str], rows: list[list[Any]]) -> None:
    p_cap = doc.add_paragraph()
    _format_paragraph(p_cap, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4.0)
    r_cap = p_cap.add_run(caption)
    r_cap.bold = True
    r_cap.italic = True
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(12)

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_cells = table.rows[0].cells
    for j, h_text in enumerate(headers):
        hdr_cells[j].text = str(h_text)
        _set_cell_shading(hdr_cells[j], "142D55")
        for p in hdr_cells[j].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(10.5)
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)

    for i, row_vals in enumerate(rows):
        row_cells = table.rows[i + 1].cells
        bg_hex = "F4F6F9" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row_vals):
            row_cells[j].text = str(val)
            _set_cell_shading(row_cells[j], bg_hex)
            for p in row_cells[j].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                for r in p.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(10.5)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(2)


def _add_figure(doc: Document, img_path: Path, caption: str, width_inches: float = 3.1) -> None:
    if not img_path.exists():
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(2)
    p_img.paragraph_format.space_after = Pt(2)
    run_img = p_img.add_run()
    run_img.add_picture(str(img_path), width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    _format_paragraph(p_cap, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4.0)
    r_cap = p_cap.add_run(caption)
    r_cap.italic = True
    r_cap.bold = True
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(11.5)


def build_report_docx() -> Path:
    facts: dict[str, Any] = json.loads((REPORTS_DIR / "facts.json").read_text(encoding="utf-8"))

    ds_tw = facts["datasets"]["assigned"]
    ds_rd = facts["datasets"]["assigned_reddit"]
    ds_hate = facts["datasets"]["hate"]
    ds_dav = facts["datasets"]["davidson_cross_domain"]
    sp = facts["preprocessing_and_splits"]
    bal = facts["balance_choice"]
    bm = facts["best_model"]
    ft = facts["final_test"]
    ci = ft["ci95"]
    rw = facts["realworld"]
    sql_f = facts["sql_analysis"]
    bonus = facts["bonus_models"]

    doc = Document()

    # Configure page setup & default Normal style (Times New Roman 13pt, 1.5 line spacing)
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(1.8)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(13)

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    cover_lines = [
        ("BÁO CÁO ĐỒ ÁN MÔN HỌC", 16, True),
        ("KHAI PHÁ VĂN BẢN VÀ HỌC MÁY ỨNG DỤNG (TEXT MINING & MACHINE LEARNING)", 14, True),
        ("", 12, False),
        ("TÊN ĐỀ TÀI:", 14, True),
        ("GẮN NHÃN CÁC ĐOẠN TWEET CÓ NỘI DUNG PHÂN BIỆT CHỦNG TỘC, PHÂN BIỆT GIỚI TÍNH", 18, True),
        ("(TWITTER SENTIMENTAL ANALYSIS — BINARY CLASSIFICATION)", 14, True),
        ("", 12, False),
        (f"Mô hình lựa chọn: {bm['name']} (Threshold = {bm['threshold']:.3f})", 13, False),
        (f"Kết quả kiểm định độc lập: Test F1 = {ft['f1_pos']:.4f} | PR-AUC = {ft['pr_auc']:.4f} | ROC-AUC = {ft['roc_auc']:.4f}", 13, False),
        (f"Kiểm thử thực tế (N = {rw['n']} mẫu): Cohen's Kappa κ = {rw['kappa']:.4f} | Real-World F1 = {rw['overall_metrics']['f1_pos']:.4f}", 13, False),
        ("", 14, False),
        ("Môi trường thực thi: Python 3 & Google Colab (Seed cố định = 42)", 13, False),
        ("Năm học: 2026", 13, True),
    ]
    for txt, sz, is_b in cover_lines:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(10)
        r = p.add_run(txt)
        r.font.name = "Times New Roman"
        r.font.size = Pt(sz)
        r.bold = is_b

    doc.add_page_break()

    # =========================================================================
    # TABLE OF CONTENTS PAGE
    # =========================================================================
    _add_heading(doc, "MỤC LỤC TỰ ĐỘNG", level=1)
    p_toc = doc.add_paragraph()
    _add_toc_field(p_toc)
    _add_body_p(
        doc,
        "(Lưu ý: Nhấn chuột phải vào vùng mục lục trên trong Microsoft Word và chọn 'Update Field' -> 'Update entire table' để cập nhật số trang tự động. Dưới đây là đề cương chi tiết 8 chương của báo cáo):",
        indent=False,
    )
    toc_outline = [
        "Chương 1: Giới thiệu Đề tài và Mục tiêu Nghiên cứu",
        "Chương 2: Thu thập, Khám phá Dữ liệu (EDA) và Phân tích SQL Tìm điểm chung của Tweet Phản cảm (Câu 1 — 2.0 điểm)",
        "Chương 3: Tiền xử lý Văn bản, Khử trùng Chống rò rỉ và Cân bằng Dữ liệu (Câu 2 — 1.5 điểm)",
        "Chương 4: Thiết kế Thực nghiệm, Xây dựng Mô hình Học máy và Tiêu chí Đánh giá (Câu 2 — 1.5 điểm)",
        "Chương 5: So sánh Lựa chọn Mô hình, Tinh chỉnh Siêu tham số, Tối ưu Ngưỡng và Mô hình Điểm cộng (Câu 3 — 2.0 điểm)",
        "Chương 6: Đánh giá Cuối trên Tập Test, Phân tích Lỗi và Chạy thử Mô hình trong Thực tế (Câu 3 — 1.5 điểm)",
        "Chương 7: Thiết kế và Triển khai Ứng dụng Web Streamlit Sàng lọc Tweet (Câu 4 — 1.0 điểm)",
        "Chương 8: Hạn chế, Đạo đức AI, Phân tích Thiên lệch (Subgroup Bias) và Kết luận — Hướng phát triển (Câu 4 — 1.0 điểm)",
        "Tài liệu Tham khảo & Phụ lục Hướng dẫn Chạy lại (Local & Google Colab)",
    ]
    for item in toc_outline:
        _add_bullet_p(doc, "", item)

    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG 1: GIỚI THIỆU ĐỀ TÀI VÀ MỤC TIÊU NGHIÊN CỨU
    # =========================================================================
    _add_heading(doc, "CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI VÀ MỤC TIÊU NGHIÊN CỨU", level=1)

    _add_heading(doc, "1.1. Mục tiêu của chương", level=2)
    _add_body_p(
        doc,
        "Chương này trình bày bối cảnh thực tiễn của bài toán phát hiện ngôn từ thù ghét phân biệt chủng tộc và phân biệt giới tính (Racist/Sexist Hate Speech Detection) trên mạng xã hội X (Twitter), xác định rõ các yêu cầu của đề bài theo thang điểm rubric 10 điểm, và thiết lập kiến trúc kỹ thuật tổng thể đảm bảo tính tái lập hoàn toàn trên cả máy cục bộ và Google Colab.",
    )

    _add_heading(doc, "1.2. Đặt vấn đề và tính cấp thiết của đề tài", level=2)
    _add_body_p(
        doc,
        "Sự bùng nổ của các nền tảng mạng xã hội vi blog như X (Twitter) và Reddit mang lại không gian trao đổi thông tin tức thời cho hàng trăm triệu người dùng mỗi ngày. Tuy nhiên, đặc tính ngắn gọn, ẩn danh và tốc độ lan truyền theo cấp số nhân cũng biến các nền tảng này thành nơi phát tán các đoạn văn bản mang nội dung phân biệt chủng tộc (racism), phân biệt giới tính (sexism), kỳ thị tôn giáo và ngôn từ thù ghét (hate speech). Những nội dung phản cảm này không chỉ gây tổn thương tâm lý nghiêm trọng cho cá nhân và cộng đồng yếu thế mà còn kích động xung đột xã hội.",
    )
    _add_body_p(
        doc,
        "Việc kiểm duyệt hoàn toàn thủ công bởi con người không thể đáp ứng được lưu lượng hàng trăm nghìn đoạn tweet mỗi phút, đồng thời gây áp lực tâm lý nặng nề cho đội ngũ kiểm duyệt viên. Ngược lại, các bộ lọc chặn từ khóa thô sơ (keyword blacklisting) dễ dàng bị qua mặt bởi cách viết lách luật, ẩn dụ hoặc mỉa mai, đồng thời lại chặn nhầm các bài viết lên án nạn phân biệt chủng tộc. Do đó, việc ứng dụng Khai phá Văn bản (Text Mining) và Học máy (Machine Learning) để xây dựng mô hình phân loại nhị phân tự động gán nhãn các đoạn tweet phản cảm là bài toán có ý nghĩa khoa học và thực tiễn cao.",
    )

    _add_heading(doc, "1.3. Đối chiếu yêu cầu đề bài và thang điểm Rubric", level=2)
    _add_body_p(
        doc,
        "Báo cáo và mã nguồn của dự án được thiết kế bám sát 4 yêu cầu chuyên môn và 4 câu hỏi trong thang điểm chuẩn (Bảng 1.1).",
    )
    _add_table(
        doc,
        "Bảng 1.1. Ánh xạ chi tiết giữa Thang điểm Rubric, Yêu cầu đề bài và các Giai đoạn thực hiện (P0 → P12).",
        ["Câu / Hạng mục", "Điểm", "Nội dung yêu cầu chi tiết", "Chương & Artifact minh chứng"],
        [
            ["Câu 1: Thu thập & Phân tích dữ liệu", "2.0đ", "Thu thập/phân tích bộ dữ liệu (1.0đ); Thống kê mô tả dữ liệu & SQL tìm điểm chung tweet phản cảm (1.0đ)", "Chương 2; data_report.json, 01_...ipynb, 02_sql_analysis.ipynb, db/tweets.db"],
            ["Câu 2: Thiết kế mô hình học máy", "3.0đ", "Làm sạch & import Google Colab/Python (1.0đ); Chuẩn hóa & cân bằng dữ liệu (0.5đ); Xây dựng mô hình, tập huấn luyện & tiêu chí đánh giá (1.5đ)", "Chương 3 & Chương 4; src/clean.py, data/splits/, P4_balance_comparison.csv, 03_...ipynb"],
            ["Câu 3: So sánh, Tối ưu & Chạy thực tế", "3.0đ", "So sánh hiệu quả các mô hình (1.5đ); Tinh chỉnh tham số & ngưỡng (0.5đ); Chạy thử mô hình trên bộ thực tế (1.5đ) + Điểm cộng Học sâu/Transformer", "Chương 5 & Chương 6; model_comparison.csv, P6_before_after.csv, 04..08_...ipynb"],
            ["Câu 4: Ứng dụng & Báo cáo bảo vệ", "2.0đ", "Thiết kế giao diện ứng dụng (1.0đ); Trình bày báo cáo, slide và trả lời câu hỏi (1.0đ)", "Chương 7 & Chương 8; app/app.py, report.docx, slides.pptx, qa_bank.md"],
        ],
    )

    _add_heading(doc, "1.4. Nhận xét Chương 1", level=2)
    _add_body_p(
        doc,
        "Bằng việc chuẩn hóa quy trình từ P0 đến P12 với hạt giống ngẫu nhiên cố định (seed = 42) và cơ chế kiểm soát chống rò rỉ dữ liệu nghiêm ngặt, dự án đảm bảo mọi con số trình bày trong báo cáo đều xuất phát trực tiếp từ kết quả thực nghiệm tái lập được 100%.",
    )

    # =========================================================================
    # CHƯƠNG 2: THU THẬP, KHÁM PHÁ DỮ LIỆU (EDA) & PHÂN TÍCH SQL (CÂU 1 - 2.0Đ)
    # =========================================================================
    _add_heading(doc, "CHƯƠNG 2. THU THẬP, KHÁM PHÁ DỮ LIỆU (EDA) VÀ PHÂN TÍCH SQL TÌM ĐIỂM CHUNG CỦA TWEET PHẢN CẢM", level=1)

    _add_heading(doc, "2.1. Mục tiêu và Phương pháp thu thập dữ liệu", level=2)
    _add_body_p(
        doc,
        "Mục tiêu của Chương 2 là tải và kiểm định chất lượng bộ dữ liệu được giao qua thư viện kagglehub, làm rõ vấn đề nhãn dữ liệu để lựa chọn phương án dữ liệu huấn luyện khoa học, thực hiện 10 phân tích thống kê mô tả kèm kiểm định giả thuyết thống kê, và xây dựng cơ sở dữ liệu SQLite (db/tweets.db) nhằm truy vấn tìm ra các đặc điểm chung định lượng của các đoạn tweet phản cảm.",
    )
    _add_body_p(
        doc,
        f"Khi kiểm tra trực tiếp tệp Twitter_Data.csv ({ds_tw['n_rows']:,} dòng) trong bộ dữ liệu được giao (cosmos98/twitter-and-reddit-sentimental-analysis-dataset), nhóm xác nhận tệp này chỉ gồm đúng 2 cột là clean_text (kiểu chuỗi) và category (kiểu số thực nhận 3 giá trị: -1.0 tiêu cực, 0.0 trung tính, +1.0 tích cực). Nội dung văn bản là các đoạn bình luận cảm xúc chính trị xoay quanh cuộc tổng tuyển cử Ấn Độ và thủ tướng Narendra Modi, hoàn toàn không có nhãn phân biệt chủng tộc hay phân biệt giới tính. Do đó, nhóm thực hiện quyết định phương pháp luận (Phương án A): sử dụng bộ dữ liệu chuẩn Tweets Hate Speech Detection ({ds_hate['n_rows']:,} dòng có nhãn nhị phân 0/1 racist/sexist) để huấn luyện và đánh giá mô hình; đồng thời sử dụng trọn vẹn bộ Twitter_Data.csv ({ds_tw['n_rows']:,} dòng), Reddit_Data.csv ({ds_rd['n_rows']:,} dòng) và bộ Davidson et al. ({ds_dav['n_rows']:,} dòng) cho thống kê mô tả ở Câu 1 và kiểm thử chéo khác miền ở Câu 3.",
        bold_prefix="Phát hiện cốt lõi về nhãn của bộ dữ liệu được giao: ",
    )

    _add_table(
        doc,
        "Bảng 2.1. Tổng hợp quy mô, cấu trúc cột và phân bố nhãn của các bộ dữ liệu trong dự án (Trích xuất từ reports/data_report.json).",
        ["Tên bộ dữ liệu", "Số dòng thô", "Số dòng NaN", "Trùng lặp văn bản", "Cột nhãn & Phân bố chi tiết"],
        [
            ["Hate Speech Train (hate_train.csv)", f"{ds_hate['n_rows']:,}", "0", f"{ds_hate['n_duplicates_text']:,} (exact) / 3,628 (clean)", f"label: Lớp 0 = {ds_hate['label_distribution']['0']:,} (92.99%), Lớp 1 = {ds_hate['label_distribution']['1']:,} ({ds_hate['positive_rate_pct']:.2f}%)"],
            ["Hate Unlabeled Test", f"{facts['datasets']['hate_unlabeled_test']['n_rows']:,}", "0", f"{facts['datasets']['hate_unlabeled_test']['n_duplicates_text']:,}", "Không có nhãn (dùng lấy mẫu thực tế Nguồn A ở P9)"],
            ["Assigned Twitter (Twitter_Data.csv)", f"{ds_tw['n_rows']:,}", f"{ds_tw['n_null']} (11 dòng)", "1", "category: +1 = 72,250 | 0 = 55,213 | -1 = 35,510 | NaN = 7"],
            ["Assigned Reddit (Reddit_Data.csv)", f"{ds_rd['n_rows']:,}", f"{ds_rd['n_null']}", f"{ds_rd['n_duplicates_text']:,}", "category: +1 = 15,830 | 0 = 13,142 | -1 = 8,277"],
            ["Davidson et al. (Cross-domain)", f"{ds_dav['n_rows']:,}", "0", "0", f"class: 0 (Hate) = 1,430 | 1 (Offensive) = 19,190 | 2 (Neither) = 4,163"],
        ],
    )

    _add_figure(
        doc,
        FIG_DIR / "P2_01_dataset_quality.png",
        "Hình 2.1. So sánh quy mô các bộ dữ liệu và thống kê tỷ lệ trùng lặp văn bản trên bộ Hate Speech (N = 31,962).",
    )
    _add_figure(
        doc,
        FIG_DIR / "P2_02_label_distribution.png",
        f"Hình 2.2. Phân bố nhãn nhị phân trên bộ Hate Speech: mất cân bằng nghiêm trọng với tỷ lệ {ds_hate['imbalance_ratio_neg_to_pos']}:1.",
    )

    _add_heading(doc, "2.2. Kết quả Thống kê mô tả: Độ dài, Đặc trưng bề mặt và Từ vựng (P2)", level=2)
    _add_body_p(
        doc,
        f"Hình 2.1 và Hình 2.2 minh họa hai thách thức kỹ thuật lớn nhất của dữ liệu: (1) Mất cân bằng lớp nghiêm trọng khi Lớp 1 (Phản cảm) chỉ chiếm {ds_hate['positive_rate_pct']:.2f}% ({ds_hate['label_distribution']['1']:,}/{ds_hate['n_rows']:,} dòng); và (2) Có tới 3,628 dòng trùng lặp sau khi làm sạch (chiếm 11.35%) do hiệu ứng đăng lại (retweet). Tiếp theo, nhóm phân tích độ dài ký tự và số từ theo từng nhãn (Bảng 2.2 và Hình 2.3).",
    )

    len_rows = []
    for r in facts["eda_highlights"]["length_stats"]:
        len_rows.append([r["group"], r["metric"], r["mean"], r["median"], r["std"], r["p95"], f"{r['min']} – {r['max']}"])
    _add_table(
        doc,
        "Bảng 2.2. Thống kê độ dài ký tự (n_chars) và số từ (n_words) theo nhãn trên bộ dữ liệu Hate Speech.",
        ["Nhóm nhãn", "Đại lượng", "Trung bình (Mean)", "Trung vị (Median)", "Độ lệch chuẩn (Std)", "Phân vị 95% (P95)", "Min – Max"],
        len_rows,
    )
    _add_figure(
        doc,
        FIG_DIR / "P2_03_length_distribution.png",
        "Hình 2.3. Biểu đồ Histogram mật độ và Boxplot phân bố số ký tự và số từ theo nhãn (Lớp 0 vs Lớp 1).",
    )

    _add_body_p(
        doc,
        "Để kiểm tra xem các đặc trưng bề mặt (như tỷ lệ nhắc @user, dùng #hashtag, dấu câu '!', '?', chữ viết HOA) có sự khác biệt có ý nghĩa thống kê giữa tweet bình thường và tweet phản cảm hay không, nhóm thực hiện kiểm định Chi-Square (cho biến nhị phân) và kiểm định phi tham số Mann-Whitney U (cho biến liên tục). Kết quả tổng hợp tại Bảng 2.3 và Hình 2.4.",
    )
    surf_rows = []
    for r in facts["eda_highlights"]["surface_tests"]:
        surf_rows.append([
            r["description"],
            r["label_0_mean"],
            r["label_1_mean"],
            r["test_type"],
            r["statistic"],
            f"{r['p_value']:.2e}",
            "Có (p < 0.05)" if r["significant_005"] else "Không",
        ])
    _add_table(
        doc,
        "Bảng 2.3. Kiểm định thống kê Chi-Square và Mann-Whitney U cho các đặc trưng bề mặt giữa Lớp 0 và Lớp 1.",
        ["Đặc trưng bề mặt", "Trung bình Lớp 0", "Trung bình Lớp 1", "Kiểm định", "Thống kê", "Giá trị p-value", "Ý nghĩa thống kê"],
        surf_rows,
    )
    _add_figure(
        doc,
        FIG_DIR / "P2_04_surface_features.png",
        "Hình 2.4. So sánh tỷ lệ xuất hiện các đặc trưng bề mặt giữa Lớp 0 (Bình thường) và Lớp 1 (Phản cảm).",
    )

    _add_body_p(
        doc,
        "Kết quả ở Bảng 2.3 cho thấy: 65.03% các đoạn tweet phản cảm có chứa thẻ @user (so với 46.46% ở Lớp 0, p < 10^-60), phản ánh xu hướng nhắm trực diện vào tài khoản cá nhân để công kích. Đồng thời, 18.78% tweet phản cảm sử dụng dấu hỏi '?' (so với 10.42% ở Lớp 0), thường dưới dạng câu hỏi tu từ mang tính miệt thị hoặc mỉa mai. Tiếp theo, Hình 2.5, Hình 2.6, Hình 2.7, Hình 2.8 và Hình 2.9 trình bày phân tích từ vựng (Unigram, Bigram, WordCloud, Log-Odds Ratio, Chi-Square và Hashtag — toàn bộ từ tục đã được che tự động bằng hàm mask_profanity).",
    )
    _add_figure(doc, FIG_DIR / "P2_05_top_unigrams.png", "Hình 2.5. Top-20 từ đơn (Unigram) phổ biến nhất ở Lớp 0 và Lớp 1 (đã bỏ stopword và che từ tục).")
    _add_figure(doc, FIG_DIR / "P2_05b_top_bigrams.png", "Hình 2.6. Top-20 cụm từ đôi (Bigram) phổ biến nhất ở Lớp 0 và Lớp 1.")
    _add_figure(doc, FIG_DIR / "P2_06_wordcloud_by_label.png", "Hình 2.7. Đám mây từ vựng (WordCloud) đối sánh giữa Lớp 0 (Bình thường) và Lớp 1 (Phản cảm — đã che từ tục).")
    _add_figure(doc, FIG_DIR / "P2_07_discriminative_log_odds_chi2.png", "Hình 2.8. Top-25 từ có khả năng phân biệt mạnh nhất đối với Lớp 1 theo Smoothed Log-Odds Ratio và kiểm định Chi-Square.")
    _add_figure(doc, FIG_DIR / "P2_08_top_hashtags.png", "Hình 2.9. Top-20 Hashtag phổ biến nhất ở Lớp 0 và Lớp 1.")
    _add_figure(doc, FIG_DIR / "P2_09_assigned_twitter_eda.png", "Hình 2.10. Phân tích bộ dữ liệu được giao Twitter_Data.csv: phân bố Sentiment (-1/0/+1), độ dài từ và Top-15 từ ở nhãn -1 (chủ đề chính trị Ấn Độ / Modi).")
    _add_figure(doc, FIG_DIR / "P2_10_domain_vocab_overlap.png", f"Hình 2.11. Phân tích giao thoa Top-5000 từ vựng giữa hai miền dữ liệu (Hệ số tương đồng Jaccard = {facts['datasets']['domain_jaccard_top5000']:.4f}).")

    _add_heading(doc, "2.3. Phân tích SQL tìm điểm chung của các Tweet phản cảm (Yêu cầu 1 — P3)", level=2)
    _add_body_p(
        doc,
        f"Thực hiện đúng gợi ý trong đề bài ('import dữ liệu vào SQL, sau đó chạy các query để tìm ra các từ có nội dung phản cảm với tần suất lớn'), nhóm xây dựng cơ sở dữ liệu SQLite tại db/tweets.db gồm 5 bảng chuẩn hóa: tweets, tokens, bigrams, assigned và lexicon. Để tuân thủ tuyệt đối nguyên tắc chống rò rỉ dữ liệu, chỉ có tập huấn luyện (data/splits/train.csv, gồm {sql_f['n_train_tweets_in_db']:,} tweet, sinh ra {sql_f['n_tokens_in_db']:,} token và {sql_f['n_bigrams_in_db']:,} bigram) được nạp vào các bảng phân tích đặc trưng.",
    )

    lo_rows = []
    for idx, r in enumerate(sql_f["top_20_log_odds_masked"][:15], start=1):
        lo_rows.append([idx, r["token"], r["c1"], r["c0"], f"{r['log_odds']:.4f}"])
    _add_table(
        doc,
        "Bảng 2.4. Kết quả truy vấn 20_log_odds.sql: Top-15 từ đặc trưng nhất của Lớp 1 trên tập Train (Đã che từ tục).",
        ["Hạng", "Token (Đã che từ tục)", "Tần suất Lớp 1 (c1)", "Tần suất Lớp 0 (c0)", "Smoothed Log-Odds Ratio"],
        lo_rows,
    )

    co_rows = []
    for idx, r in enumerate(sql_f["top_15_cooccur_masked"][:12], start=1):
        co_rows.append([idx, r["neighbor"], r["cnt"], r["n_tweets"]])
    _add_table(
        doc,
        "Bảng 2.5. Kết quả truy vấn 30_cooccur.sql: Top-12 từ đồng xuất hiện nhiều nhất cùng nhóm từ khóa phản cảm trong các tweet Lớp 1.",
        ["Hạng", "Từ đồng xuất hiện (Neighbor)", "Tổng số lần đi kèm (cnt)", "Số lượng tweet chứa cặp (n_tweets)"],
        co_rows,
    )

    lex_cov_rows = [
        [
            "Lớp 0 (Bình thường)",
            f"{sql_f['lexicon_coverage'][0]['n_tweets']:,}",
            f"{sql_f['lexicon_coverage'][0]['n_with_lexicon']:,}",
            f"{sql_f['lexicon_coverage'][0]['pct_with_lexicon']:.2f}%",
        ],
        [
            "Lớp 1 (Phản cảm — Racist/Sexist)",
            f"{sql_f['lexicon_coverage'][1]['n_tweets']:,}",
            f"{sql_f['lexicon_coverage'][1]['n_with_lexicon']:,}",
            f"{sql_f['lexicon_coverage'][1]['pct_with_lexicon']:.2f}%",
        ],
    ]
    _add_table(
        doc,
        "Bảng 2.6. Kết quả truy vấn 40_lexicon.sql: Tỷ lệ tweet chứa từ trong từ điển xúc phạm (Lexicon) theo từng nhãn.",
        ["Nhãn", "Tổng số tweet (Tập Train)", "Số tweet chứa từ Lexicon", "Tỷ lệ phần trăm (%)"],
        lex_cov_rows,
    )

    lex_cat_rows = []
    for r in sql_f["lexicon_by_category"]:
        lex_cat_rows.append([r["category"], r["hits_label_1"], r["hits_label_0"], r["tweets_label_1"], r["tweets_label_0"]])
    _add_table(
        doc,
        "Bảng 2.7. Phân rã kết quả truy vấn SQL theo từng nhóm danh mục từ nhạy cảm trong Lexicon.",
        ["Danh mục Lexicon (Category)", "Số lần khớp ở Lớp 1", "Số lần khớp ở Lớp 0", "Số tweet Lớp 1", "Số tweet Lớp 0"],
        lex_cat_rows,
    )
    _add_figure(
        doc,
        FIG_DIR / "P3_sql_analysis.png",
        "Hình 2.12. Tổng hợp trực quan kết quả 3 truy vấn SQL chính: Top Log-Odds (Query 20), Đồng xuất hiện (Query 30) và Tỷ lệ chứa từ Lexicon (Query 40).",
    )

    _add_heading(doc, "2.4. Tổng hợp điểm chung của Tweet phản cảm và Nhận xét Chương 2", level=2)
    _add_body_p(
        doc,
        "Từ kết quả kết hợp giữa EDA và truy vấn SQL, nhóm đúc kết 5 đặc điểm chung định lượng của các đoạn tweet phản cảm: (1) Tỷ lệ chứa từ trong từ điển xúc phạm ở Lớp 1 là 20.59%, cao gấp 8.58 lần Lớp 0 (2.40%); (2) Tuy nhiên có tới 79.41% tweet phản cảm không dùng từ tục trực tiếp trong Lexicon mà biểu đạt sự kỳ thị thông qua tổ hợp cụm từ, định kiến hoặc hashtag; (3) Nội dung tập trung vào 3 chủ đề: miệt thị chủng tộc/màu da, hạ thấp phụ nữ/giới tính và công kích tôn giáo/chính trị; (4) Tỷ lệ nhắc @user trực diện đạt 65.03% kèm câu hỏi tu từ '?'; (5) Lưu ý quan trọng: sự xuất hiện của các từ chỉ nhóm người (như women, black, white, muslim) chỉ mang tính tương quan chứ không phải quan hệ nhân quả — cần mô hình học máy xem xét ngữ cảnh cụm từ.",
    )

    # =========================================================================
    # CHƯƠNG 3: TIỀN XỬ LÝ, KHỬ TRÙNG & CÂN BẰNG DỮ LIỆU (CÂU 2 - 1.5Đ)
    # =========================================================================
    _add_heading(doc, "CHƯƠNG 3. TIỀN XỬ LÝ VĂN BẢN, KHỬ TRÙNG CHỐNG RÒ RỈ VÀ CÂN BẰNG DỮ LIỆU", level=1)

    _add_heading(doc, "3.1. Mục tiêu và Pipeline làm sạch văn bản 12 bước (src/clean.py)", level=2)
    _add_body_p(
        doc,
        "Mục tiêu của Chương 3 là chuẩn hóa văn bản mạng xã hội nhiều nhiễu về dạng chuẩn phục vụ trích xuất đặc trưng mà không làm mất đi các tín hiệu ngữ nghĩa quan trọng của ngôn từ thù ghét, đồng thời loại bỏ hoàn toàn các bản sao trùng lặp trước khi chia tập và lựa chọn kỹ thuật cân bằng lớp tối ưu thông qua Cross-Validation.",
    )
    _add_body_p(
        doc,
        "Hàm clean_text() trong mô-đun src/clean.py được thiết kế gồm 12 bước tuần tự chặt chẽ: (1) Ép kiểu chuỗi an toàn và giải mã thực thể HTML (html.unescape, xử lý &amp;, &lt;, &gt;); (2) Chuyển toàn bộ về chữ thường (lowercase); (3) Xóa địa chỉ URL (https?://\\S+|www.\\S+); (4) Xóa thẻ tài khoản @mention (@\\w+, bao gồm @user); (5) Xóa nhãn đăng lại 'rt' đứng độc lập; (6) Tách giữ nội dung chữ của hashtag (#tag -> tag); (7) Chuyển đổi biểu tượng cảm xúc emoji thành từ mô tả (emoji.demojize); (8) Rút gọn các chữ cái lặp lại liên tiếp từ 3 lần trở lên về 2 lần (ví dụ sooooo -> soo); (9) Loại bỏ ký tự đặc biệt và chữ số nhưng giữ lại dấu nháy đơn nội từ (như don't, it's); (10) Tách từ và loại bỏ các token dài 1 ký tự ngoại trừ đại từ 'i' và mạo từ 'a'; (11) Lọc stopwords có bảo toàn danh sách từ nhạy cảm PRESERVED_WORDS (giữ lại toàn bộ từ phủ định no, not, nor, never, don't, cannot và các đại từ/danh từ chỉ giới you, your, he, she, they, them, we, us, women, men); (12) Đưa từ về dạng nguyên mẫu bằng WordNetLemmatizer.",
    )

    clean_ex_df = pd.read_csv(TAB_DIR / "P4_clean_examples.csv").head(8)
    clean_ex_rows = []
    for r in clean_ex_df.itertuples():
        clean_ex_rows.append([r.id, r.label, str(r.text_raw_masked)[:75], str(r.text_clean_masked)[:75]])
    _add_table(
        doc,
        "Bảng 3.1. Minh họa 8 ví dụ thực tế trước và sau khi qua hàm làm sạch clean_text() (Đã che từ tục).",
        ["ID", "Nhãn", "Văn bản gốc (Đã che từ tục)", "Văn bản sau khi làm sạch (text_clean)"],
        clean_ex_rows,
    )
    _add_figure(
        doc,
        FIG_DIR / "P4_length_before_after.png",
        f"Hình 3.1. Phân bố độ dài số từ của tweet trước khi làm sạch (Mean = {sp['word_len_before_mean']} từ) và sau khi làm sạch (Mean = {sp['word_len_after_mean']} từ).",
    )

    _add_heading(doc, "3.2. Khử trùng gần-giống, Xử lý nhãn mâu thuẫn và Chia tập Stratified 70/15/15", level=2)
    _add_body_p(
        doc,
        f"Sau khi áp dụng clean_text() lên toàn bộ {sp['n_initial_raw']:,} dòng của hate_train.csv, nhóm thực hiện kiểm tra chất lượng và phát hiện: có {sp['n_empty_dropped']} dòng trở thành chuỗi rỗng (chỉ chứa URL/@user), {sp['n_conflict_groups']} nhóm ({sp['n_conflict_rows_dropped']} dòng) có nội dung text_clean giống hệt nhau nhưng nhãn gốc mâu thuẫn nhau (vừa gán nhãn 0 vừa gán nhãn 1), và {sp['n_near_duplicate_rows_dropped']:,} dòng là bản sao trùng lặp nội dung text_clean do retweet. Nhóm tiến hành loại bỏ toàn bộ {sp['n_total_dropped']:,} dòng này trước khi chia tập, giữ lại {sp['n_final_unique_clean']:,} mẫu sạch duy nhất. Tiếp theo, dữ liệu được chia phân tầng (Stratified) theo tỷ lệ 70% Train – 15% Validation – 15% Test với seed = 42 (Bảng 3.2).",
    )
    _add_table(
        doc,
        "Bảng 3.2. Thống kê chi tiết số lượng mẫu và tỷ lệ lớp trên 3 tập dữ liệu sau khi khử trùng (data/splits/).",
        ["Tập dữ liệu (Split)", "Tỷ lệ chia", "Tổng số dòng", "Số mẫu Lớp 0 (Bình thường)", "Số mẫu Lớp 1 (Phản cảm)", "Tỷ lệ Lớp 1 (%)"],
        [
            ["Tập Huấn luyện (train.csv)", "70.0%", f"{sp['splits']['train']['n_rows']:,}", f"{sp['splits']['train']['label_0']:,}", f"{sp['splits']['train']['label_1']:,}", f"{sp['splits']['train']['pos_rate_pct']:.4f}%"],
            ["Tập Kiểm định (val.csv)", "15.0%", f"{sp['splits']['val']['n_rows']:,}", f"{sp['splits']['val']['label_0']:,}", f"{sp['splits']['val']['label_1']:,}", f"{sp['splits']['val']['pos_rate_pct']:.4f}%"],
            ["Tập Kiểm tra (test.csv)", "15.0%", f"{sp['splits']['test']['n_rows']:,}", f"{sp['splits']['test']['label_0']:,}", f"{sp['splits']['test']['label_1']:,}", f"{sp['splits']['test']['pos_rate_pct']:.4f}%"],
            ["Tổng cộng (clean.csv)", "100.0%", f"{sp['n_final_unique_clean']:,}", f"{sp['final_label_0']:,}", f"{sp['final_label_1']:,}", f"{sp['final_pos_rate_pct']:.4f}%"],
        ],
    )

    _add_heading(doc, "3.3. So sánh thực nghiệm 6 chiến lược cân bằng dữ liệu (P4.4 — 0.5 điểm)", level=2)
    _add_body_p(
        doc,
        "Để xử lý tỷ lệ mất cân bằng lớp (6.52% Lớp 1 vs 93.48% Lớp 0), nhóm cố định mô hình cơ sở Logistic Regression (max_iter=2000, solver='liblinear') cùng bộ trích xuất word_tfidf và đánh giá 6 chiến lược cân bằng dữ liệu bằng phương pháp 5-fold StratifiedKFold Cross-Validation chỉ trên tập Train (N = 19,830). Mọi bộ lấy mẫu lại (RandomOverSampler, SMOTE, RandomUnderSampler, SMOTETomek) đều được đặt bên trong imblearn.pipeline.Pipeline để chỉ tác động lên phần fold huấn luyện, tuyệt đối không làm biến dạng fold đánh giá. Kết quả trình bày tại Bảng 3.3 và Hình 3.2.",
    )
    bal_rows = []
    for r in bal["table"]:
        bal_rows.append([
            r["strategy"],
            r["description"],
            f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f}",
            f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f}",
            f"{r['f1_pos_mean']:.4f} ± {r['f1_pos_std']:.4f}",
            f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}",
            f"{r['fit_seconds']:.2f}s",
        ])
    _add_table(
        doc,
        "Bảng 3.3. Kết quả so sánh 6 chiến lược cân bằng dữ liệu bằng 5-fold StratifiedKFold CV trên tập Train (P4.4).",
        ["Mã chiến lược", "Mô tả kỹ thuật", "CV Precision (Mean ± Std)", "CV Recall (Mean ± Std)", "CV F1 Lớp 1 (Mean ± Std)", "CV PR-AUC (Mean ± Std)", "Thời gian (5 folds)"],
        bal_rows,
    )
    _add_figure(
        doc,
        FIG_DIR / "P4_balance_comparison.png",
        "Hình 3.2. Biểu đồ so sánh Precision, Recall, F1-Score và thời gian chạy 5-fold CV của 6 chiến lược cân bằng dữ liệu.",
    )

    _add_heading(doc, "3.4. Nhận xét Chương 3", level=2)
    _add_body_p(
        doc,
        f"Bảng 3.3 chứng minh rõ rệt sự cần thiết của bước cân bằng dữ liệu: khi không cân bằng (none), mô hình đạt Precision = 0.8726 nhưng Recall chỉ đạt 0.1385 (bỏ sót hơn 86% số tweet phản cảm), làm F1 tụt xuống 0.2390. Ngược lại, RandomUnderSampler (rus) cắt giảm lớp đa số quá mức khiến Precision rơi xuống 0.2827 (F1 = 0.4225). Bốn phương pháp ros, smote, smote_tomek và class_weight đều nâng F1 lên vùng 0.5742 – 0.5876; trong đó RandomOverSampler (ros) đạt CV F1 cao nhất (0.5876 ± 0.0117, PR-AUC = 0.6337) với thời gian chỉ 2.86 giây (nhanh gấp 19 lần SMOTETomek mất 55.20 giây), nên được chọn làm chiến lược cân bằng chuẩn cho giai đoạn so sánh mô hình P5.",
    )

    # =========================================================================
    # CHƯƠNG 4: THIẾT KẾ THỰC NGHIỆM, MÔ HÌNH & TIÊU CHÍ ĐÁNH GIÁ (CÂU 2 - 1.5Đ)
    # =========================================================================
    _add_heading(doc, "CHƯƠNG 4. THIẾT KẾ THỰC NGHIỆM, XÂY DỰNG MÔ HÌNH HỌC MÁY VÀ TIÊU CHÍ ĐÁNH GIÁ", level=1)

    _add_heading(doc, "4.1. Mục tiêu và Kiến trúc biểu diễn đặc trưng văn bản", level=2)
    _add_body_p(
        doc,
        "Chương 4 trình bày cơ sở toán học của các phương pháp biểu diễn văn bản (TF-IDF từ và ký tự, Embedding học sâu), kiến trúc của 9 mô hình học máy cổ điển trong MODEL_ZOO cùng 2 mô hình học sâu mở rộng (BiLSTM và DistilBERT), và lý giải hệ thống tiêu chí đánh giá chuyên biệt cho bài toán phân loại nhị phân mất cân bằng.",
    )
    _add_body_p(
        doc,
        "Trong mô-đun src/features.py, nhóm xây dựng 4 phương thức biểu diễn: (1) word_tfidf sử dụng TfidfVectorizer với ngram_range=(1,2), min_df=3, max_df=0.9, max_features=50,000 và sublinear_tf=True (thay tần suất thô tf bằng 1 + log(tf) để giảm tác động của các từ bị lặp nhiều lần trong một tweet); (2) char_tfidf trích xuất các chuỗi ký tự con liên tiếp bên trong ranh giới từ (analyzer='char_wb', ngram_range=(2,5)) nhằm bắt các từ viết sai chính tả hoặc biến thể lách kiểm duyệt; (3) word+char kết hợp song song cả hai không gian từ và ký tự bằng FeatureUnion; và (4) bow (CountVectorizer) làm mốc tham chiếu.",
    )

    _add_heading(doc, "4.2. Cơ sở lý thuyết của 9 Mô hình Cổ điển và 2 Mô hình Học sâu", level=2)
    _add_bullet_p(doc, "1. Logistic Regression (lr): ", "Mô hình tuyến tính ước lượng xác suất hậu nghiệm P(y=1|x) = sigmoid(w^T x + b) bằng cách tối thiểu hóa hàm mất mát Log-Loss có điều chuẩn L1 hoặc L2 (solver='liblinear'). Ưu điểm: nhanh, xuất xác suất chuẩn và giải thích trực tiếp đóng góp từng từ.")
    _add_bullet_p(doc, "2. Complement Naive Bayes (nb): ", "Biến thể Naive Bayes của Rennie et al. (2003) thiết kế riêng cho dữ liệu văn bản lệch lớp. Thay vì ước lượng xác suất từ trên lớp thiểu số ít mẫu, mô hình ước lượng trọng số dựa trên phần bù (lớp đa số), giúp ổn định biên quyết định.")
    _add_bullet_p(doc, "3. Calibrated Linear SVM (svm_linear): ", "Tìm siêu phẳng có lề phân cách (margin) cực đại trong không gian TF-IDF chiều cao với hàm mất mát Hinge hoặc Squared Hinge. Vì LinearSVC không xuất xác suất trực tiếp, nhóm bọc mô hình trong CalibratedClassifierCV(cv=3) (Platt scaling) để thu được xác suất hậu nghiệm chính xác.")
    _add_bullet_p(doc, "4. RBF-Kernel SVM (svm_rbf): ", "Sử dụng nhân Gauss K(x, x') = exp(-gamma ||x - x'||^2) để ánh xạ dữ liệu sang không gian phi tuyến. Do độ phức tạp bậc hai theo số mẫu, mô hình được huấn luyện trên mẫu phân tầng 8,000 dòng.")
    _add_bullet_p(doc, "5. Decision Tree (dt) & 6. Random Forest (rf): ", "Cây quyết định phân hoạch đệ quy không gian đặc trưng theo độ giảm tạp chất Gini (max_depth=30), và Rừng ngẫu nhiên kết hợp 200 cây độc lập (bagging) để giảm phương sai.")
    _add_bullet_p(doc, "7. Cosine K-Nearest Neighbors (knn): ", "Phân loại dựa trên k=5 láng giềng có độ tương đồng góc Cosine lớn nhất trong không gian vector TF-IDF.")
    _add_bullet_p(doc, "8. XGBoost (xgb) & 9. LightGBM (lgbm): ", "Hai thuật toán Gradient Boosted Decision Trees hiện đại nhất, xây dựng chuỗi 300 cây nối tiếp tối ưu đạo hàm bậc hai của hàm mất mát với cấu trúc histogram (tree_method='hist').")
    _add_bullet_p(doc, "10. BiLSTM (Điểm cộng P7): ", "Mạng nơ-ron hồi quy hai chiều (Bidirectional LSTM) đọc chuỗi từ theo cả chiều thuận và chiều nghịch trên tầng Embedding 128 chiều, kết hợp Global Average Pooling và Global Max Pooling trước khi qua tầng phân loại.")
    _add_bullet_p(doc, "11. DistilBERT Fine-tuning (Điểm cộng P7): ", "Mô hình ngôn ngữ Transformer 6 tầng (66 triệu tham số, chưng cất tri thức từ BERT-base) sử dụng cơ chế Multi-Head Self-Attention để nắm bắt ngữ cảnh hai chiều sâu của toàn bộ câu tweet.")

    _add_heading(doc, "4.3. Hệ thống Tiêu chí đánh giá: Vì sao chọn F1 Lớp dương và PR-AUC?", level=2)
    _add_body_p(
        doc,
        "Với tỷ lệ Lớp 0 chiếm 93.48% và Lớp 1 chỉ chiếm 6.52%, chỉ số Accuracy hoàn toàn mất tác dụng phân biệt: một bộ phân loại tầm thường dự đoán mọi tweet đều là Lớp 0 vẫn đạt Accuracy = 93.48% nhưng không phát hiện được bất kỳ nội dung phản cảm nào (TP = 0, Recall = 0%). Tương tự, ROC-AUC sử dụng False Positive Rate = FP / (FP + TN) có mẫu số TN quá lớn (3,973 mẫu âm trên tập Val/Test) nên luôn cho điểm lạc quan trên 0.90.",
    )
    _add_body_p(
        doc,
        "Vì vậy, trong mô-đun src/evaluate.py, nhóm sử dụng F1-score của lớp dương (f1_pos = 2 * Precision * Recall / (Precision + Recall)) làm tiêu chí chính để lựa chọn mô hình và tối ưu ngưỡng, kết hợp chặt chẽ với diện tích dưới đường cong Precision-Recall (PR-AUC / Average Precision), Recall lớp 1, Balanced Accuracy và hệ số tương quan Matthews (MCC). Toàn bộ 13 chỉ số đánh giá được tính toán đồng nhất cho mọi mô hình.",
    )

    _add_heading(doc, "4.4. Nhận xét Chương 4", level=2)
    _add_body_p(
        doc,
        "Thiết kế thực nghiệm kết hợp Pipeline khép kín, đánh giá chéo 5-fold StratifiedKFold trên tập Train và kiểm định độc lập trên tập Validation theo thước đo F1 lớp dương / PR-AUC tạo dựng khuôn khổ đánh giá công bằng, khách quan và không rò rỉ dữ liệu cho toàn bộ 11 mô hình.",
    )

    # =========================================================================
    # CHƯƠNG 5: SO SÁNH MÔ HÌNH, TINH CHỈNH & MÔ HÌNH ĐIỂM CỘNG (CÂU 3 - 2.0Đ)
    # =========================================================================
    _add_heading(doc, "CHƯƠNG 5. SO SÁNH LỰA CHỌN MÔ HÌNH, TINH CHỈNH SIÊU THAM SỐ, TỐI ƯU NGƯỠNG VÀ MÔ HÌNH ĐIỂM CỘNG", level=1)

    _add_heading(doc, "5.1. Kết quả so sánh 9 Mô hình cổ điển trên tập Validation (P5 — 1.5 điểm)", level=2)
    _add_body_p(
        doc,
        "Mục tiêu của Chương 5 là trình bày kết quả thực nghiệm đối sánh toàn bộ các mô hình trong MODEL_ZOO trên tập Train (5-fold CV) và tập Validation (N = 4,250), kiểm định ý nghĩa thống kê của sự chênh lệch hiệu năng, thực hiện tinh chỉnh siêu tham số kết hợp tối ưu ngưỡng quyết định (P6), và đối sánh với hai mô hình học sâu/Transformer điểm cộng (P7).",
    )

    model_rows = []
    for r in facts["models"]:
        model_rows.append([
            f"{r['model_name']} ({r['vectorizer']})",
            f"{r['cv_f1_mean']:.4f}",
            f"{r['precision_pos']:.4f}",
            f"{r['recall_pos']:.4f}",
            f"{r['f1_pos']:.4f}",
            f"{r['pr_auc']:.4f}",
            f"{r['roc_auc']:.4f}",
            f"{r['mcc']:.4f}",
            f"{r['fit_seconds']:.2f}s",
        ])
    _add_table(
        doc,
        "Bảng 5.1. Bảng tổng hợp kết quả đánh giá toàn bộ các cấu hình mô hình trên tập Validation (N = 4,250 — reports/model_comparison.csv).",
        ["Mô hình & Vectorizer", "CV F1 (Train)", "Val Precision", "Val Recall", "Val F1 (Lớp 1)", "Val PR-AUC", "Val ROC-AUC", "Val MCC", "Fit Time"],
        model_rows,
    )

    ttest_rows = []
    for r in facts["paired_ttests"]:
        ttest_rows.append([
            r["run_a"],
            r["run_b"],
            f"{r['mean_diff_f1']:+.4f}",
            f"{r['t_statistic']:.4f}",
            f"{r['p_value']:.4f}",
            "Có (p < 0.05)" if r["significant_at_005"] else "Không (Tương đương)",
        ])
    _add_table(
        doc,
        "Bảng 5.2. Kết quả kiểm định thống kê Paired t-test trên điểm F1 qua 5 fold CV giữa 3 cấu hình dẫn đầu.",
        ["Cấu hình A", "Cấu hình B", "Chênh lệch F1 trung bình", "Thống kê t", "Giá trị p-value", "Ý nghĩa thống kê (α=0.05)"],
        ttest_rows,
    )

    _add_figure(doc, FIG_DIR / "P5_f1_prauc_bars.png", "Hình 5.1. Biểu đồ cột so sánh CV F1, Validation F1 và Validation PR-AUC của các mô hình (P5).")
    _add_figure(doc, FIG_DIR / "P5_roc_pr_curves.png", "Hình 5.2. Đường cong ROC và Đường cong Precision-Recall chồng nhau của 9 mô hình trên tập Validation.")
    _add_figure(doc, FIG_DIR / "P5_confusion_grid_3x3.png", "Hình 5.3. Lưới 3×3 Ma trận nhầm lẫn của 9 mô hình cổ điển (word_tfidf) trên tập Validation.")

    _add_body_p(
        doc,
        "Phân tích kết quả ở Bảng 5.1 và Hình 5.1–5.3 cho thấy quy luật thực nghiệm rất rõ ràng: các mô hình tuyến tính (Logistic Regression, Calibrated Linear SVM) và xác suất (Complement Naive Bayes) dẫn đầu tuyệt đối với Validation F1 từ 0.5727 đến 0.6486 và PR-AUC từ 0.6267 đến 0.7403. Ngược lại, nhóm mô hình cây quyết định (Decision Tree F1 = 0.3836, Random Forest F1 = 0.5195, XGBoost F1 = 0.5034, LightGBM F1 = 0.5260) và khoảng cách (KNN F1 = 0.3902) đạt hiệu năng thấp hơn hẳn. Nguyên nhân cốt lõi nằm ở tính chất thưa và số chiều lớn của biểu diễn TF-IDF (50,000 chiều nhưng mỗi câu chỉ có ~12 từ khác không): siêu phẳng tuyến tính tổng hợp đồng thời trọng số của toàn bộ từ vựng, trong khi mỗi nút rẽ nhánh của cây quyết định chỉ kiểm tra được từng từ đơn lẻ nên làm phân mảnh không gian mẫu.",
    )

    _add_heading(doc, "5.2. Tinh chỉnh siêu tham số và Tối ưu ngưỡng quyết định (P6 — 0.5 điểm)", level=2)
    _add_body_p(
        doc,
        "Ba họ mô hình dẫn đầu (Logistic Regression, Linear SVM, Complement Naive Bayes) được đưa vào giai đoạn tinh chỉnh bằng RandomizedSearchCV (n_iter = 40 tổ hợp × 5-fold StratifiedKFold CV trên tập Train, tối ưu tiêu chí scoring='f1'). Không gian tìm kiếm bao gồm cả siêu tham số của TfidfVectorizer (ngram_range trong {(1,1), (1,2), (1,3)}, min_df trong {2, 3, 5}, max_features trong {20k, 50k, 100k}) lẫn siêu tham số của từng bộ phân loại (hệ số điều chuẩn C phân phối loguniform(10^-2, 10^2), hàm mất mát hinge/squared_hinge, hệ số làm mượt alpha). Sau khi tìm được mô hình tốt nhất trên Train, nhóm sử dụng hàm pick_threshold() quét ngưỡng quyết định trên tập Validation từ 0.05 đến 0.95 (Bảng 5.3, Hình 5.4 và Hình 5.5).",
    )

    ba_rows = []
    for r in facts["tuning_before_after"]:
        ba_rows.append([
            r["model_name"],
            f"{r['cv_f1_before']:.4f} → {r['cv_f1_after']:.4f}",
            f"{r['val_f1_before']:.4f}",
            f"{r['val_f1_after_thr05']:.4f}",
            f"{r['best_threshold']:.3f}",
            f"{r['val_f1_after_tuned_thr']:.4f}",
            f"{r['f1_gain_vs_baseline']:+.4f}",
            f"{r['val_precision_after']:.4f} / {r['val_recall_after']:.4f}",
        ])
    _add_table(
        doc,
        "Bảng 5.3. So sánh hiệu năng trên tập Validation trước và sau khi tinh chỉnh siêu tham số & tối ưu ngưỡng quyết định (P6).",
        ["Mô hình", "CV F1 (Trước → Sau)", "Val F1 Trước Tuning", "Val F1 Sau Tuning (Thr=0.5)", "Ngưỡng Tối ưu", "Val F1 Sau Tuning + Ngưỡng", "Mức tăng F1", "Val P / R Cuối"],
        ba_rows,
    )
    _add_figure(doc, FIG_DIR / "P6_threshold_tuning.png", f"Hình 5.4. Đường cong Precision, Recall, F1 theo ngưỡng quyết định và điểm vận hành tối ưu (Threshold = {bm['threshold']:.3f}) của {bm['name']}.")
    _add_figure(doc, FIG_DIR / "P6_before_after.png", "Hình 5.5. Biểu đồ so sánh Validation F1 qua 3 giai đoạn: Trước Tuning, Sau Tuning tham số (ngưỡng 0.50) và Sau khi tối ưu ngưỡng.")

    _add_body_p(
        doc,
        f"Kết quả tại Bảng 5.3 chứng minh hiệu quả vượt trội của việc kết hợp tinh chỉnh siêu tham số và tối ưu ngưỡng quyết định: Linear SVM (Calibrated) với cấu hình tối ưu ngram_range=(1,3), min_df=2, max_features=50,000, C={bm['params']['clf__C']}, loss='squared_hinge', class_weight='balanced' và ngưỡng quyết định threshold = {bm['threshold']:.3f} đã nâng Validation F1 từ {bm['val_metrics_before_tuning']['val_f1_before']:.4f} lên {bm['val_metrics_after_tuning_best_thr']['f1_pos']:.4f} (tăng +8.39 điểm %), đạt Precision = {bm['val_metrics_after_tuning_best_thr']['precision_pos']:.4f} và Recall = {bm['val_metrics_after_tuning_best_thr']['recall_pos']:.4f}. Do đó, cấu hình này được chọn làm mô hình chốt (best_model), được huấn luyện lại trên toàn bộ Train + Val (24,080 mẫu) và lưu vào models/best_model.joblib.",
    )

    _add_heading(doc, "5.3. Thực nghiệm Mô hình Điểm cộng ngoài chương trình: BiLSTM & DistilBERT (P7)", level=2)
    _add_body_p(
        doc,
        "Để mở rộng phạm vi nghiên cứu vượt ra ngoài các mô hình học máy truyền thống, nhóm triển khai thêm 2 kiến trúc học sâu trong mô-đun src/transformer.py: (1) Mạng BiLSTM dùng PyTorch và (2) Mô hình ngôn ngữ Transformer DistilBERT (distilbert-base-uncased) của HuggingFace. Bảng 5.4 và Hình 5.6 đối sánh trực tiếp hiệu năng và chi phí tính toán giữa mô hình cổ điển tối ưu và hai mô hình học sâu.",
    )
    _add_table(
        doc,
        "Bảng 5.4. Đối sánh giữa Mô hình cổ điển tối ưu (Linear SVM Calibrated) và hai mô hình Học sâu / Transformer (P7).",
        ["Mô hình", "Biểu diễn đặc trưng", "Val Precision", "Val Recall", "Val F1 (Lớp 1)", "Val PR-AUC", "Val ROC-AUC", "Thời gian Train"],
        [
            [f"{bm['name']} (Chốt P6)", "Word TF-IDF (1-3 gram, 50k)", f"{bm['val_metrics_after_tuning_best_thr']['precision_pos']:.4f}", f"{bm['val_metrics_after_tuning_best_thr']['recall_pos']:.4f}", f"{bm['val_metrics_after_tuning_best_thr']['f1_pos']:.4f}", f"{bm['val_metrics_after_tuning_best_thr']['pr_auc']:.4f}", f"{bm['val_metrics_after_tuning_best_thr']['roc_auc']:.4f}", "~1.2s"],
            ["BiLSTM (PyTorch Deep Learning)", "Embedding 128d + Avg/Max Pool", f"{bonus['bilstm']['precision_pos']:.4f}", f"{bonus['bilstm']['recall_pos']:.4f}", f"{bonus['bilstm']['f1_pos']:.4f}", f"{bonus['bilstm']['pr_auc']:.4f}", f"{bonus['bilstm']['roc_auc']:.4f}", f"{bonus['bilstm']['fit_seconds']:.1f}s"],
            ["DistilBERT (Fine-tuned Transformer)", "Subword WordPiece (max_len=64..80)", f"{bonus['distilbert']['precision_pos']:.4f}", f"{bonus['distilbert']['recall_pos']:.4f}", f"{bonus['distilbert']['f1_pos']:.4f}", f"{bonus['distilbert']['pr_auc']:.4f}", f"{bonus['distilbert']['roc_auc']:.4f}", f"{bonus['distilbert']['fit_seconds']:.1f}s"],
        ],
    )
    _add_figure(doc, FIG_DIR / "P7_bonus_comparison.png", "Hình 5.6. So sánh hiệu năng (F1, PR-AUC) và thời gian huấn luyện giữa mô hình cổ điển tối ưu, BiLSTM và DistilBERT (P7).")

    _add_heading(doc, "5.4. Nhận xét Chương 5", level=2)
    _add_body_p(
        doc,
        f"Thực nghiệm Chương 5 khẳng định rằng trên bài toán phân loại tweet ngắn với ngân sách tài nguyên tiêu chuẩn, mô hình {bm['name']} kết hợp TF-IDF (1,3)-gram và tối ưu ngưỡng quyết định đạt hiệu năng cao nhất (Val F1 = {bm['val_metrics_after_tuning_best_thr']['f1_pos']:.4f}), vượt qua cả BiLSTM ({bonus['bilstm']['f1_pos']:.4f}) và DistilBERT ({bonus['distilbert']['f1_pos']:.4f}) trong khi nhanh hơn hàng chục lần và hỗ trợ giải thích trực tiếp trọng số từng từ.",
    )

    # =========================================================================
    # CHƯƠNG 6: ĐÁNH GIÁ CUỐI TRÊN TẬP TEST, PHÂN TÍCH LỖI & THỰC TẾ (CÂU 3 - 1.5Đ)
    # =========================================================================
    _add_heading(doc, "CHƯƠNG 6. ĐÁNH GIÁ CUỐI TRÊN TẬP TEST, PHÂN TÍCH LỖI VÀ CHẠY THỬ MÔ HÌNH TRONG THỰC TẾ", level=1)

    _add_heading(doc, "6.1. Đánh giá một lần duy nhất trên tập Test độc lập và Khoảng tin cậy 95% Bootstrap (P8)", level=2)
    _add_body_p(
        doc,
        f"Sau khi hoàn tất mọi quyết định lựa chọn mô hình và ngưỡng quyết định ở Chương 5, nhóm nạp models/best_model.joblib cùng ngưỡng threshold = {ft['threshold']:.3f} từ models/threshold.json để chạy suy luận một lần duy nhất trên tập kiểm tra độc lập data/splits/test.csv (N = {ft['n_test_rows']:,} mẫu, gồm {ft['n_test_neg']:,} mẫu Lớp 0 và {ft['n_test_pos']:,} mẫu Lớp 1; mã băm SHA-256 = {ft['test_file_sha256'][:24]}...; thời điểm chạy: {ft['evaluated_once_at']}). Đồng thời, nhóm áp dụng kỹ thuật Bootstrap phi tham số 1,000 lần (seed = 42) để ước lượng khoảng tin cậy 95% (95% Confidence Interval) cho các chỉ số chính (Bảng 6.1 và Hình 6.1).",
    )
    _add_table(
        doc,
        "Bảng 6.1. Kết quả đánh giá cuối cùng trên tập Test độc lập (N = 4,250) và Khoảng tin cậy 95% Bootstrap (1,000 lần).",
        ["Chỉ số đánh giá (Metric)", "Giá trị tại Ngưỡng 0.50", f"Giá trị tại Ngưỡng Tối ưu ({ft['threshold']:.3f})", "Trung bình Bootstrap", "Khoảng tin cậy 95% (95% CI)"],
        [
            ["F1-Score Lớp dương (f1_pos — Chính)", f"{ft['default_05_f1_pos']:.4f}", f"{ft['f1_pos']:.4f}", f"{ci['f1_pos']['mean']:.4f} ± {ci['f1_pos']['std']:.4f}", f"[{ci['f1_pos']['ci95_low']:.4f}, {ci['f1_pos']['ci95_high']:.4f}]"],
            ["Precision Lớp dương (precision_pos)", f"{ft['default_05_precision_pos']:.4f}", f"{ft['precision_pos']:.4f}", f"{ci['precision_pos']['mean']:.4f} ± {ci['precision_pos']['std']:.4f}", f"[{ci['precision_pos']['ci95_low']:.4f}, {ci['precision_pos']['ci95_high']:.4f}]"],
            ["Recall Lớp dương (recall_pos)", f"{ft['default_05_recall_pos']:.4f}", f"{ft['recall_pos']:.4f}", f"{ci['recall_pos']['mean']:.4f} ± {ci['recall_pos']['std']:.4f}", f"[{ci['recall_pos']['ci95_low']:.4f}, {ci['recall_pos']['ci95_high']:.4f}]"],
            ["PR-AUC (Average Precision)", f"{ft['pr_auc']:.4f}", f"{ft['pr_auc']:.4f}", f"{ci['pr_auc']['mean']:.4f} ± {ci['pr_auc']['std']:.4f}", f"[{ci['pr_auc']['ci95_low']:.4f}, {ci['pr_auc']['ci95_high']:.4f}]"],
            ["ROC-AUC", f"{ft['roc_auc']:.4f}", f"{ft['roc_auc']:.4f}", f"{ci['roc_auc']['mean']:.4f} ± {ci['roc_auc']['std']:.4f}", f"[{ci['roc_auc']['ci95_low']:.4f}, {ci['roc_auc']['ci95_high']:.4f}]"],
            ["F1 Macro / Balanced Accuracy / MCC", "—", f"{ft['f1_macro']:.4f} / {ft['balanced_acc']:.4f} / {ft['mcc']:.4f}", "—", "—"],
            ["Ma trận nhầm lẫn (TN / FP / FN / TP)", "—", f"TN={ft['tn']}, FP={ft['fp']}, FN={ft['fn']}, TP={ft['tp']}", "—", "Tổng N = 4,250"],
        ],
    )
    _add_figure(doc, FIG_DIR / "P8_test_evaluation.png", "Hình 6.1. Ma trận nhầm lẫn, Đường cong ROC và Đường cong Precision-Recall của mô hình chốt trên tập Test độc lập (P8).")

    _add_heading(doc, "6.2. Phân tích lỗi (Error Analysis) và Giải thích trọng số mô hình (P8)", level=2)
    _add_body_p(
        doc,
        f"Trên tập Test gồm 4,250 mẫu, mô hình dự đoán đúng {ft['tn'] + ft['tp']:,} mẫu (Accuracy = {ft['accuracy']*100:.2f}%) và có {ft['fp'] + ft['fn']} mẫu dự đoán sai, bao gồm {ft['fp']} ca Báo động giả (False Positives — FP) và {ft['fn']} ca Bỏ sót (False Negatives — FN). Nhóm đã xuất danh sách 30 FP có điểm xác suất cao nhất và 30 FN có điểm xác suất thấp nhất vào bảng reports/tables/P8_errors.csv, đồng thời phân loại toàn bộ các mẫu lỗi theo 6 nhóm nguyên nhân ngôn ngữ học (Bảng 6.2, Bảng 6.3 và Hình 6.2).",
    )
    cause_rows = []
    for r in facts["error_causes"]:
        cause_rows.append([
            r["cause"],
            r.get("FP (Báo động giả)", 0),
            r.get("FN (Bỏ sót)", 0),
            r["total"],
            f"{r['pct']:.2f}%",
        ])
    _add_table(
        doc,
        "Bảng 6.2. Thống kê phân loại nguyên nhân của các trường hợp dự đoán sai (FP và FN) trên tập Test.",
        ["Nhóm nguyên nhân lỗi ngôn ngữ học", "Số lỗi FP", "Số lỗi FN", "Tổng số lỗi", "Tỷ lệ (%)"],
        cause_rows,
    )
    _add_figure(doc, FIG_DIR / "P8_error_causes.png", "Hình 6.2. Biểu đồ phân bố các nhóm nguyên nhân gây lỗi dự đoán trên tập Test (P8).")

    err20_rows = []
    for idx, r in enumerate(facts["errors_20_masked"][:12], start=1):
        err20_rows.append([
            idx,
            r["id"],
            r["error_type"],
            f"{r['score']:.4f}",
            str(r["cause"]).split("(")[0].strip(),
            str(r["text_masked"])[:70],
        ])
    _add_table(
        doc,
        "Bảng 6.3. Minh họa 12 trường hợp dự đoán sai tiêu biểu (6 FP và 6 FN) trên tập Test (Trích từ 20 mẫu lỗi trong facts.json — đã che từ tục).",
        ["#", "ID", "Loại lỗi", "Điểm Score", "Nguyên nhân chính", "Nội dung Tweet (Đã che từ tục)"],
        err20_rows,
    )

    _add_body_p(
        doc,
        "Hình 6.3 và Bảng 6.4 trình bày Top đặc trưng có trọng số dương lớn nhất (đẩy quyết định về Lớp 1 — Phản cảm) và trọng số âm lớn nhất (kéo quyết định về Lớp 0 — Bình thường) được trích xuất trực tiếp từ bộ phân loại tuyến tính.",
    )
    pos_top10 = facts["top_discriminative_terms_masked"]["model_positive_top20"][:10]
    neg_top10 = facts["top_discriminative_terms_masked"]["model_negative_top20"][:10]
    coef_table_rows = []
    for i in range(10):
        coef_table_rows.append([
            i + 1,
            pos_top10[i]["term_masked"],
            f"+{pos_top10[i]['weight']:.4f}",
            neg_top10[i]["term_masked"],
            f"{neg_top10[i]['weight']:.4f}",
        ])
    _add_table(
        doc,
        "Bảng 6.4. Top-10 đặc trưng đóng góp mạnh nhất cho Lớp 1 (Phản cảm) và Lớp 0 (Bình thường) trong mô hình chốt.",
        ["Hạng", "Đặc trưng Lớp 1 (Đã che từ tục)", "Trọng số (+)", "Đặc trưng Lớp 0 (Bình thường)", "Trọng số (-)"],
        coef_table_rows,
    )
    _add_figure(doc, FIG_DIR / "P8_feature_importance.png", "Hình 6.3. Biểu đồ Top-20 đặc trưng có trọng số dương lớn nhất (Lớp 1) và trọng số âm lớn nhất (Lớp 0).")

    _add_heading(doc, "6.3. Xây dựng và Chạy thử trên Bộ dữ liệu Thực tế 300 mẫu (Câu 3: 1.5 điểm — P9)", level=2)
    _add_body_p(
        doc,
        f"Để đáp ứng Yêu cầu 4 ('Xây dựng một bộ dữ liệu tweet thực tế và chạy thử để xem kết quả'), nhóm xây dựng bộ dữ liệu thực tế data/real_world/real_world_labeled.csv gồm N = {rw['n']} mẫu từ 4 nguồn đa dạng (đảm bảo không trùng lặp với Train/Val/Test): (A) 100 mẫu từ hate_unlabeled_test.csv; (B) 100 mẫu từ bộ được giao Twitter_Data.csv; (C) 50 mẫu từ bộ Reddit_Data.csv; và (D) 50 câu hiểm hóc tự biên soạn (adversarial curated: gồm câu mỉa mai, câu phủ định phản đối kỳ thị 'I am NOT racist...', câu trung tính chứa từ chỉ nhóm người 'Black women entrepreneurs...', và câu chửi thề cảm thán không nhắm vào nhóm bảo vệ).",
    )
    _add_body_p(
        doc,
        f"Hai luồng gắn nhãn độc lập (label_a và label_b) dựa trên hướng dẫn chuẩn ANNOTATION_GUIDE.md đạt tỷ lệ đồng thuận {rw['agreement_pct']:.2f}% (chỉ bất đồng ở {rw['n_disagreements']}/{rw['n']} mẫu nằm ở vùng ranh giới ngữ nghĩa) và đạt hệ số đồng thuận Cohen's Kappa κ = {rw['kappa']:.4f} (mức đồng thuận rất tốt). Sau khi thảo luận thống nhất nhãn cuối cùng (label_final: gồm {rw['n_pos']} mẫu Lớp 1 và {rw['n_neg']} mẫu Lớp 0), nhóm chạy mô hình chốt và thu được kết quả tổng thể cùng phân rã theo từng nguồn tại Bảng 6.5, Bảng 6.6, Bảng 6.7 và Hình 6.4.",
    )

    src_rows = []
    for r in rw["f1_by_source"]:
        src_rows.append([
            r["source"],
            r["n_samples"],
            f"{r['n_pos']} / {r['n_neg']}",
            f"{r['precision_pos']:.4f}",
            f"{r['recall_pos']:.4f}",
            f"{r['f1_pos']:.4f}",
            f"{r['pr_auc']:.4f}",
            f"{r['roc_auc']:.4f}",
        ])
    src_rows.append([
        "TỔNG HỢP TOÀN BỘ (Overall Real-World)",
        rw["n"],
        f"{rw['n_pos']} / {rw['n_neg']}",
        f"{rw['overall_metrics']['precision_pos']:.4f}",
        f"{rw['overall_metrics']['recall_pos']:.4f}",
        f"{rw['overall_metrics']['f1_pos']:.4f}",
        f"{rw['overall_metrics']['pr_auc']:.4f}",
        f"{rw['overall_metrics']['roc_auc']:.4f}",
    ])
    _add_table(
        doc,
        f"Bảng 6.5. Kết quả chạy thử mô hình trên Bộ dữ liệu Thực tế (N = {rw['n']}, Cohen's Kappa κ = {rw['kappa']:.4f}) phân rã theo 4 nguồn dữ liệu.",
        ["Nguồn dữ liệu thực tế", "Số mẫu", "Lớp 1 / Lớp 0", "Precision", "Recall", "F1-Score (Lớp 1)", "PR-AUC", "ROC-AUC"],
        src_rows,
    )

    tw_sent_rows = []
    for r in rw["assigned_twitter_by_sentiment"]:
        tw_sent_rows.append([
            r["sentiment_name"],
            f"{r['n_tweets']:,}",
            f"{r['n_predicted_hate']:,}",
            f"{r['pct_predicted_hate']:.2f}%",
            f"{r['mean_hate_score']:.4f}",
        ])
    _add_table(
        doc,
        "Bảng 6.6. Kết quả chạy mô hình trên toàn bộ 162,969 dòng của bộ được giao Twitter_Data.csv thống kê theo nhãn Sentiment (-1/0/+1).",
        ["Nhãn Sentiment (Twitter_Data.csv)", "Tổng số dòng", "Số dòng dự đoán Phản cảm (Nhãn 1)", "Tỷ lệ dự đoán Phản cảm (%)", "Điểm xác suất trung bình"],
        tw_sent_rows,
    )

    rw_ex_rows = []
    for r in rw["sample_errors_masked"][:6]:
        rw_ex_rows.append([r["id"], r["source"], r["error_type"], r["label_final"], r["y_pred"], f"{r['score']:.4f}", str(r["text_masked"])[:65]])
    for r in rw["sample_correct_masked"][:4]:
        rw_ex_rows.append([r["id"], r["source"], "Dự đoán Đúng", r["label_final"], r["y_pred"], f"{r['score']:.4f}", str(r["text_masked"])[:65]])
    _add_table(
        doc,
        "Bảng 6.7. Minh họa 10 mẫu dự đoán thực tế (6 mẫu sai và 4 mẫu câu hiểm hóc dự đoán đúng) trên tập Real-World.",
        ["ID", "Nguồn", "Kết quả", "Nhãn thật", "Dự đoán", "Score", "Nội dung Tweet (Đã che từ tục)"],
        rw_ex_rows,
    )
    _add_figure(doc, FIG_DIR / "P9_realworld_evaluation.png", "Hình 6.4. Hiệu năng theo 4 nguồn dữ liệu thực tế, Ma trận nhầm lẫn (Kappa = 0.7880) và tỷ lệ dự đoán trên Twitter_Data.csv (P9).")

    _add_heading(doc, "6.4. Nhận xét Chương 6: Phân tích hiện tượng Lệch miền (Domain Shift)", level=2)
    _add_body_p(
        doc,
        f"Kết quả Chương 6 làm sáng tỏ hiện tượng lệch miền (Domain Shift): trên các nguồn dữ liệu mạng xã hội tự nhiên (Nguồn A, B, C), mô hình đạt F1 rất cao (0.8235 – 0.8966), đưa F1 tổng thể trên 300 mẫu thực tế đạt {rw['overall_metrics']['f1_pos']:.4f}. Tuy nhiên, trên Nguồn D gồm 50 câu đối kháng hiểm hóc (chứa mỉa mai, định kiến ngầm không dùng từ tục, hoặc trích dẫn từ tục để phê phán), F1 giảm xuống {rw['f1_by_source'][3]['f1_pos']:.4f}. Tương tự, khi kiểm thử chéo không huấn luyện lại trên 5,593 mẫu của bộ Davidson et al., mô hình đạt Precision = {rw['davidson_cross_domain']['precision_pos']:.4f} nhưng Recall = {rw['davidson_cross_domain']['recall_pos']:.4f} (F1 = {rw['davidson_cross_domain']['f1_pos']:.4f}) do khác biệt phương ngữ từ lóng. Đặc biệt, Bảng 6.6 chứng minh trên 162,969 dòng Twitter_Data.csv, ngay cả ở nhóm Sentiment = -1 (tiêu cực), chỉ có {rw['assigned_twitter_by_sentiment'][0]['pct_predicted_hate']:.2f}% mang dấu hiệu ngôn từ thù ghét — khẳng định tính đúng đắn tuyệt đối của Phương án A.",
    )

    # =========================================================================
    # CHƯƠNG 7: THIẾT KẾ & TRIỂN KHAI ỨNG DỤNG WEB STREAMLIT (CÂU 4 - 1.0Đ)
    # =========================================================================
    _add_heading(doc, "CHƯƠNG 7. THIẾT KẾ VÀ TRIỂN KHAI ỨNG DỤNG WEB STREAMLIT SÀNG LỌC TWEET", level=1)

    _add_heading(doc, "7.1. Mục tiêu và Kiến trúc kỹ thuật của Ứng dụng (src/predict.py & app/app.py)", level=2)
    _add_body_p(
        doc,
        "Mục tiêu của Chương 7 là đóng gói mô hình tốt nhất (models/best_model.joblib và models/threshold.json) thành một ứng dụng web tương tác thời gian thực bằng tiếng Việt, phục vụ hai tác vụ cốt lõi: (1) Kiểm tra và giải thích từng đoạn tweet đơn lẻ cho kiểm duyệt viên, và (2) Sàng lọc hàng loạt tệp CSV quy mô lớn (lên tới 20,000 dòng).",
    )
    _add_body_p(
        doc,
        "Về kiến trúc phần mềm, tầng suy luận được tách biệt độc lập tại mô-đun src/predict.py với lớp Predictor. Khi nhận một đoạn văn bản, Predictor cắt giới hạn an toàn 1,000 ký tự, gọi hàm clean_text() 12 bước, tính điểm xác suất qua mô hình đã hiệu chuẩn, so sánh với ngưỡng tối ưu 0.315 và tính toán đóng góp cục bộ của từng từ trong câu theo công thức: Đóng góp(từ j) = TF-IDF(từ j) × Trọng số tuyến tính(w_j). Tầng giao diện app/app.py sử dụng Streamlit kết hợp cơ chế @st.cache_resource để giữ mô hình trong bộ nhớ RAM, giúp thời gian phản hồi mỗi câu dưới 5 mili-giây.",
    )

    _add_heading(doc, "7.2. Minh họa Giao diện thực tế 3 Tab của Ứng dụng", level=2)
    _add_body_p(
        doc,
        "Giao diện ứng dụng được thiết kế gồm 3 Tab trực quan (Hình 7.1, Hình 7.2, Hình 7.3 và Hình 7.4):",
    )
    _add_bullet_p(doc, "Tab 1 — Phân loại 1 Tweet: ", "Tích hợp 5 nút câu mẫu (2 câu bình thường, 2 câu phản cảm đã che từ tục, 1 câu mỉa mai), vùng nhập văn bản, khung kết quả tô màu Xanh/Đỏ, thanh tiến trình rủi ro so với ngưỡng 0.315 và danh sách Top-8 từ đóng góp đẩy điểm tăng (màu đỏ) hoặc giảm (màu xanh).")
    _add_bullet_p(doc, "Tab 2 — Phân loại Hàng loạt (CSV): ", "Cho phép tải lên tệp CSV bất kỳ có cột text/tweet/clean_text (hoặc bấm nút chạy thử ngay 100 mẫu thực tế), hiển thị thanh tiến trình, 3 thẻ chỉ số tổng hợp và nút tải xuống tệp CSV kết quả.")
    _add_bullet_p(doc, "Tab 3 — Thống kê & Hồ sơ Mô hình: ", "Hiển thị trực tiếp nội dung models/model_card.json, kết quả kiểm định cuối trên tập Test, bảng đối sánh 11 mô hình và biểu đồ đánh giá.")

    _add_figure(doc, FIG_DIR / "P10_app_tab1_normal.png", "Hình 7.1. Giao diện Tab 1 khi phân tích một đoạn tweet bình thường (Nhãn = 0, điểm xác suất thấp hơn ngưỡng 0.315).")
    _add_figure(doc, FIG_DIR / "P10_app_tab1_hate.png", "Hình 7.2. Giao diện Tab 1 khi phát hiện một đoạn tweet phản cảm (Nhãn = 1) kèm giải thích các từ đóng góp dương (màu đỏ).")
    _add_figure(doc, FIG_DIR / "P10_app_tab2_batch.png", "Hình 7.3. Giao diện Tab 2 phân loại hàng loạt từ tệp CSV kèm thống kê tổng hợp và nút tải xuống kết quả.")
    _add_figure(doc, FIG_DIR / "P10_app_tab3_stats.png", "Hình 7.4. Giao diện Tab 3 hiển thị hồ sơ mô hình (model_card.json), kết quả Test và bảng đối sánh các mô hình.")

    _add_heading(doc, "7.3. Nhận xét Chương 7", level=2)
    _add_body_p(
        doc,
        "Ứng dụng Streamlit đã chuyển hóa mô hình học máy từ các tệp mã nguồn nghiên cứu thành một công cụ sàng lọc thực tiễn, dễ sử dụng, có tốc độ xử lý 1,000 dòng dưới 0.1 giây và giải thích minh bạch lý do dự đoán.",
    )

    # =========================================================================
    # CHƯƠNG 8: HẠN CHẾ, ĐẠO ĐỨC AI, SUBGROUP BIAS & KẾT LUẬN (CÂU 4 - 1.0Đ)
    # =========================================================================
    _add_heading(doc, "CHƯƠNG 8. HẠN CHẾ, ĐẠO ĐỨC AI, PHÂN TÍCH THIÊN LỆCH (SUBGROUP BIAS) VÀ KẾT LUẬN — HƯỚNG PHÁT TRIỂN", level=1)

    _add_heading(doc, "8.1. Phân tích Thiên lệch Nhóm nhân khẩu học (Subgroup Bias Analysis — P8)", level=2)
    _add_body_p(
        doc,
        "Một câu hỏi đạo đức cốt lõi đối với mọi hệ thống phát hiện ngôn từ thù ghét là: Liệu mô hình có bị 'nhạy cảm quá mức' và gắn cờ nhầm các câu bình thường chỉ vì câu đó nhắc đến tên một nhóm chủng tộc, giới tính hay tôn giáo hay không? Để trả lời câu hỏi này bằng số liệu, nhóm lọc toàn bộ 3,973 mẫu thực tế là Lớp 0 (Bình thường) trên tập Test và đo Tỷ lệ Báo động giả (False Positive Rate — FPR) trên 5 nhóm từ nhận diện nhân khẩu học (Bảng 8.1 và Hình 8.1).",
    )
    bias_rows = []
    for r in facts["subgroup_bias"]:
        bias_rows.append([
            r["subgroup"],
            r["n_neg_tweets"],
            r["n_false_positives"],
            f"{r['fpr_pct']:.2f}%",
            f"{r['mean_score']:.4f}",
        ])
    _add_table(
        doc,
        "Bảng 8.1. Phân tích thiên lệch (Subgroup Bias): Tỷ lệ báo động giả (FPR) trên các tweet bình thường (Lớp 0) có chứa từ nhận diện nhóm người trên tập Test.",
        ["Nhóm từ nhận diện trong câu Lớp 0", "Số mẫu Lớp 0", "Số ca Báo động giả (FP)", "Tỷ lệ FPR (%)", "Điểm xác suất trung bình"],
        bias_rows,
    )
    _add_figure(doc, FIG_DIR / "P8_subgroup_bias.png", "Hình 8.1. Biểu đồ so sánh Tỷ lệ Báo động giả (False Positive Rate) theo từng nhóm từ nhận diện nhân khẩu học trên tập Test.")

    _add_body_p(
        doc,
        f"Số liệu tại Bảng 8.1 chỉ ra một thực tế rất đáng chú ý: trong khi tỷ lệ báo động giả trung bình trên toàn bộ 3,973 mẫu Lớp 0 của tập Test chỉ là {facts['subgroup_bias'][0]['fpr_pct']:.2f}%, những câu bình thường có nhắc đến từ khóa chủng tộc (black, white, race) có FPR tăng lên {facts['subgroup_bias'][2]['fpr_pct']:.2f}%, nhóm tôn giáo (muslim, islam, jewish) có FPR = {facts['subgroup_bias'][3]['fpr_pct']:.2f}%, và nhóm chính trị (trump, obama) có FPR = {facts['subgroup_bias'][5]['fpr_pct']:.2f}%. Nguyên nhân là trong dữ liệu huấn luyện gốc, các từ nhận diện nhóm này xuất hiện đồng thời với ngôn từ thù ghét quá thường xuyên. Đây chính là lý do ứng dụng bắt buộc phải có con người kiểm duyệt ở bước cuối (Human-in-the-loop).",
    )

    _add_heading(doc, "8.2. Tổng hợp các Hạn chế kỹ thuật và Nguyên tắc Đạo đức Dữ liệu", level=2)
    _add_bullet_p(doc, "1. Hạn chế về nhãn của bộ dữ liệu được giao và nhiễu nhãn đám đông: ", "Bộ Twitter_Data.csv được giao chỉ chứa nhãn cảm xúc chính trị (-1/0/+1), còn bộ Hate Speech chuẩn tuy có nhãn 0/1 nhưng gộp chung racist và sexist vào cùng một nhãn dương và tồn tại một tỷ lệ nhỏ nhãn nhiễu do gán nhãn đám đông.")
    _add_bullet_p(doc, "2. Hạn chế đối với câu mỉa mai (Sarcasm) và kỳ thị ngầm (Implicit Hate): ", "Biểu diễn túi từ n-gram TF-IDF chưa nắm bắt được các câu dùng lời lẽ khen ngợi giả tạo để mỉa mai hoặc các định kiến xã hội diễn đạt bằng từ vựng hoàn toàn thông thường.")
    _add_bullet_p(doc, "3. Hạn chế đối với Counter-speech (Phản đối kỳ thị): ", "Các câu trích dẫn lại từ tục để lên án nạn phân biệt chủng tộc dễ bị mô hình gắn cờ nhầm thành Lớp 1.")
    _add_bullet_p(doc, "4. Tuân thủ đạo đức và che từ tục (mask_profanity): ", "Toàn bộ danh tính cá nhân được ẩn thành @user; mọi từ ngữ tục tĩu trong bảng biểu, hình vẽ, slide và giao diện đều được che ký tự giữa (ví dụ b***h, n****r) để đảm bảo môi trường học thuật văn minh.")

    _add_heading(doc, "8.3. Kết luận chung và Hướng phát triển mở rộng", level=2)
    _add_body_p(
        doc,
        f"Đồ án đã hoàn thành toàn diện mọi mục tiêu đề ra: (1) Kiểm định và xử lý khoa học vấn đề nhãn dữ liệu, thực hiện 10 phân tích EDA và 6 truy vấn SQL trên CSDL db/tweets.db; (2) Xây dựng pipeline làm sạch 12 bước chống rò rỉ dữ liệu, khử trùng {sp['n_near_duplicate_rows_dropped']:,} mẫu retweet và đối sánh 6 chiến lược cân bằng lớp; (3) Huấn luyện và so sánh 11 mô hình (9 mô hình cổ điển + BiLSTM + DistilBERT), tinh chỉnh siêu tham số và tối ưu ngưỡng quyết định ({bm['threshold']:.3f}) giúp mô hình chốt {bm['name']} đạt Test F1 = {ft['f1_pos']:.4f} (PR-AUC = {ft['pr_auc']:.4f}, ROC-AUC = {ft['roc_auc']:.4f}); (4) Kiểm thử thực tế trên 300 mẫu đa nguồn đạt Cohen's Kappa κ = {rw['kappa']:.4f} (Real-World F1 = {rw['overall_metrics']['f1_pos']:.4f}); và (5) Xây dựng ứng dụng web Streamlit 3 tab hoàn chỉnh.",
    )
    _add_body_p(
        doc,
        "Trong tương lai, hệ thống có thể được phát triển theo 3 hướng: (1) Mở rộng sang bài toán phân loại đa nhãn (tách riêng Racist, Sexist, Religious Hate, Offensive và Normal); (2) Bổ sung dữ liệu Counter-speech và fine-tune mô hình RoBERTa/HateBERT trên GPU để giảm thiên lệch nhóm (Subgroup Bias); và (3) Mở rộng sang tiếng Việt trên bộ dữ liệu ViHSD kết hợp mô hình PhoBERT.",
    )

    # =========================================================================
    # TÀI LIỆU THAM KHẢO & PHỤ LỤC
    # =========================================================================
    _add_heading(doc, "TÀI LIỆU THAM KHẢO & PHỤ LỤC HƯỚNG DẪN CHẠY LẠI", level=1)
    refs = [
        "1. Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. Proceedings of the 11th International AAAI Conference on Web and Social Media (ICWSM '17).",
        "2. Waseem, Z., & Hovy, D. (2016). Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter. Proceedings of NAACL-HLT 2016.",
        "3. Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the Poor Assumptions of Naive Bayes Text Classifiers. Proceedings of ICML 2003.",
        "4. Davis, J., & Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. Proceedings of ICML 2006.",
        "5. Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The Risk of Racial Bias in Hate Speech Detection. Proceedings of ACL 2019.",
        "6. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. NeurIPS EMC2 Workshop.",
    ]
    for ref in refs:
        _add_body_p(doc, ref, indent=False)

    out_docx = REPORTS_DIR / "report.docx"
    doc.save(str(out_docx))
    print(f"Saved report to {out_docx}")
    return out_docx


if __name__ == "__main__":
    build_report_docx()
