"""Create a clean, natural student presentation version of the notebook, app, report, and slides
with all rubric/internal markers ('điểm cộng', 'bonus', 'P0..P12', 'V0..V8', 'B0..B6') removed."""

from __future__ import annotations

import glob
import re
from pathlib import Path

from docx import Document
import nbformat
from pptx import Presentation

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ordered replacements to make prose natural academic Vietnamese for instructor presentation
TEXT_REPLACEMENTS: list[tuple[str, str]] = [
    # Remove "điểm cộng" / "bonus" phrasing
    (r"\(\s*[Đđ]iểm cộng\s*[—\-–]?\s*[A-Za-z0-9]*\s*\)", "(Mở rộng nâng cao)"),
    (r"\(\s*[Đđ]iểm cộng[^)]*\)", "(Mở rộng nâng cao)"),
    (r"[Đđ]iểm cộng\s*\([A-Za-z0-9]+\)", "Mở rộng nâng cao"),
    (r"mô hình điểm cộng", "mô hình học sâu mở rộng"),
    (r"Mô hình điểm cộng", "Mô hình học sâu mở rộng"),
    (r"Mô hình Điểm cộng", "Mô hình Học sâu Mở rộng"),
    (r"[Đđ]iểm cộng ngoài chương trình", "phần mở rộng nâng cao"),
    (r"[Đđ]iểm cộng", "phần mở rộng"),
    (r"Bonus Deep Learning", "Deep Learning"),
    (r"PyTorch Bonus", "PyTorch Deep Learning"),
    (r"Fine-tuned Bonus", "Fine-tuned Transformer"),
    (r"\+\s*Bonus", "+ Mở rộng"),
    (r"\[Bonus\]", "[Mở rộng]"),
    # Remove internal spec codes P0..P12, V0..V8, B0..B6 in headings/text
    (r"\s*[—\-–]\s*P\d+(?:\.\d+)?(?:\s*&\s*P\d+)?", ""),
    (r"\s*[—\-–]\s*V\d+(?:\s*&\s*V\d+)?", ""),
    (r"\s*[—\-–]\s*B\d+(?:\s*/\s*B\d+)?", ""),
    (r"\(\s*P\d+(?:\.\d+)?(?:\s*[&,\-→]\s*P\d+(?:\.\d+)?)*\s*\)", ""),
    (r"\(\s*V\d+(?:\.\d+)?(?:\s*[&,\-→]\s*V\d+(?:\.\d+)?)*\s*\)", ""),
    (r"\bP0\s*→\s*P12\b", "toàn bộ quy trình"),
    (r"\bP1\s*→\s*P12\b", "toàn bộ quy trình"),
    (r"\bP0\s*->\s*P12\b", "toàn bộ quy trình"),
    (r"\bV0\s*→\s*V8\b", "phần mở rộng Tiếng Việt"),
    (r"\bV0\s*->\s*V8\b", "phần mở rộng Tiếng Việt"),
    (r"\bV1\s*→\s*V7\b", "phần thực nghiệm Tiếng Việt"),
    (r"\bP0\s*→\s*P10\b", "từ thu thập dữ liệu đến xây dựng ứng dụng"),
    (r"\bở\s+P(\d+(?:\.\d+)?)\b", "ở bước thực nghiệm \\1"),
    (r"\bgiai đoạn\s+P(\d+)\b", "giai đoạn \\1"),
    (r"\bbước\s+V(\d+)\b", "bước \\1"),
    (r"\bở\s+V(\d+)\b", "ở phần thực nghiệm \\1"),
    # Clean bootstrap / AI-sounding headings
    (r"# === CELL BOOTSTRAP CHUẨN CHO MỌI NOTEBOOK TRÊN GOOGLE COLAB & LOCAL ===", "# === THIẾT LẬP MÔI TRƯỜNG & NẠP THƯ VIỆN DỰ ÁN ==="),
    (r"# === CELL BOOTSTRAP CHUẨN CHO MỌI NOTEBOOK TRONG DỰ ÁN ===", "# === THIẾT LẬP MÔI TRƯỜNG & NẠP THƯ VIỆN DỰ ÁN ==="),
    (r"\[BOOTSTRAP\]", "[SETUP]"),
    (r"\[BOOTSTRAP HOÀN TẤT\]", "[HOÀN TẤT]"),
    (r"Thiết lập cell `bootstrap` chuẩn", "Thiết lập môi trường thực thi"),
    (r"Cell `bootstrap`", "Ô khởi tạo môi trường"),
    (r"Phương án A", "hướng tiếp cận kết hợp dữ liệu chuẩn"),
    (r"câu hiểm hóc", "câu tình huống phức tạp"),
    (r"Câu hiểm hóc", "Câu tình huống phức tạp"),
    (r"CÂU HIỂM HÓC", "CÂU TÌNH HUỐNG PHỨC TẠP"),
    (r"hiểm hóc", "phức tạp"),
]


def clean_str(text: str) -> str:
    s = text
    for pattern, repl in TEXT_REPLACEMENTS:
        s = re.sub(pattern, repl, s)
    # Clean double spaces before punctuation
    s = re.sub(r"  +\)", ")", s)
    s = re.sub(r"\(\s*\)", "", s)
    return s


def build_clean_presentation_notebook() -> Path:
    src_nb_path = PROJECT_ROOT / "notebooks" / "MASTER_ALL_IN_ONE_COLAB.ipynb"
    nb = nbformat.read(src_nb_path, as_version=4)

    # Replace top intro cell with a clean, student-style academic header
    nb.cells[0].source = (
        "# ĐỒ ÁN MÔN HỌC: PHÂN LOẠI TWEET CÓ NỘI DUNG PHÂN BIỆT CHỦNG TỘC, PHÂN BIỆT GIỚI TÍNH\n"
        "### (Twitter Sentimental Analysis — Binary Classification / Text Mining & Mở rộng Tiếng Việt)\n\n"
        "**Nội dung thực hiện trong Notebook:**\n"
        "1. **Câu 1 (Thu thập & Phân tích dữ liệu):** Khám phá bộ dữ liệu `Twitter_Data.csv`, bộ dữ liệu `Tweets Hate Speech Detection`, thống kê mô tả và truy vấn cơ sở dữ liệu SQLite (`db/tweets.db`) tìm đặc điểm chung của các tweet phản cảm.\n"
        "2. **Câu 2 (Thiết kế mô hình học máy):** Làm sạch văn bản mạng xã hội, khử trùng lặp chống rò rỉ dữ liệu, chia tập huấn luyện/kiểm định/kiểm tra (70/15/15) và so sánh 6 phương pháp cân bằng dữ liệu.\n"
        "3. **Câu 3 (So sánh, tối ưu & chạy thử thực tế):** Thực nghiệm đối sánh 9 mô hình học máy cổ điển cùng 2 mô hình học sâu (`BiLSTM`, `DistilBERT`), tinh chỉnh siêu tham số, tối ưu ngưỡng quyết định, đánh giá trên tập Test độc lập và kiểm thử trên 300 mẫu thực tế.\n"
        "4. **Mở rộng Tiếng Việt & Ứng dụng Đa ngôn ngữ (Câu 4):** Thực nghiệm trên bộ dữ liệu bình luận mạng xã hội tiếng Việt `ViHSD` (33,400 dòng), kiểm tra độ bền với văn bản không dấu/teencode và tích hợp bộ phân loại đa ngôn ngữ Anh – Việt."
    )

    for cell in nb.cells[1:]:
        cell.source = clean_str(cell.source)
        # Also clean text outputs if any contain [BOOTSTRAP]
        if cell.cell_type == "code" and "outputs" in cell:
            for out in cell.outputs:
                if "text" in out:
                    if isinstance(out["text"], list):
                        out["text"] = [clean_str(t) for t in out["text"]]
                    elif isinstance(out["text"], str):
                        out["text"] = clean_str(out["text"])

    out_nb_path = PROJECT_ROOT / "notebooks" / "BAO_CAO_DO_AN_TWEET_HATE_CLASSIFICATION.ipynb"
    nbformat.write(nb, out_nb_path)

    # Also update MASTER_ALL_IN_ONE_COLAB.ipynb and individual notebooks so all of them are clean
    nbformat.write(nb, src_nb_path)
    for p in sorted(glob.glob(str(PROJECT_ROOT / "notebooks" / "*.ipynb"))):
        if Path(p).name in ("BAO_CAO_DO_AN_TWEET_HATE_CLASSIFICATION.ipynb", "MASTER_ALL_IN_ONE_COLAB.ipynb"):
            continue
        sub_nb = nbformat.read(p, as_version=4)
        for c in sub_nb.cells:
            c.source = clean_str(c.source)
        nbformat.write(sub_nb, p)

    print(f"Created clean presentation notebook: {out_nb_path.name}")
    return out_nb_path


def clean_docx_and_pptx() -> None:
    # Clean report.docx
    docx_path = PROJECT_ROOT / "reports" / "report.docx"
    if docx_path.exists():
        doc = Document(str(docx_path))
        for p in doc.paragraphs:
            for r in p.runs:
                if r.text:
                    r.text = clean_str(r.text)
        for tbl in doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for r in p.runs:
                            if r.text:
                                r.text = clean_str(r.text)
        doc.save(str(docx_path))
        print("Cleaned reports/report.docx")

    # Clean slides.pptx
    pptx_path = PROJECT_ROOT / "reports" / "slides.pptx"
    if pptx_path.exists():
        prs = Presentation(str(pptx_path))
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for p in shape.text_frame.paragraphs:
                        for r in p.runs:
                            if r.text:
                                r.text = clean_str(r.text)
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes_tf = slide.notes_slide.notes_text_frame
                notes_tf.text = clean_str(notes_tf.text)
        prs.save(str(pptx_path))
        print("Cleaned reports/slides.pptx")

    # Clean app/app.py and markdown files
    for md_rel in [
        "app/app.py",
        "README.md",
        "reports/speaker_script.md",
        "reports/demo_script.md",
        "reports/cheat_sheet.md",
        "reports/qa_bank.md",
    ]:
        fp = PROJECT_ROOT / md_rel
        if fp.exists():
            fp.write_text(clean_str(fp.read_text(encoding="utf-8")), encoding="utf-8")
            print(f"Cleaned {md_rel}")


if __name__ == "__main__":
    build_clean_presentation_notebook()
    clean_docx_and_pptx()
