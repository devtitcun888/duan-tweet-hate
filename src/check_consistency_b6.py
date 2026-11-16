"""B6 Cross-Check & Consistency Verifier: Verifies all numbers in report.docx, slides.pptx, speaker_script.md, qa_bank.md, demo_script.md, cheat_sheet.md, and README.md against reports/facts.json."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from docx import Document
from pptx import Presentation

from src.utils import get_project_root

PROJECT_ROOT = get_project_root()
REPORTS_DIR = PROJECT_ROOT / "reports"


def _extract_docx_text(path: Path) -> str:
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def _extract_pptx_text(path: Path) -> str:
    prs = Presentation(str(path))
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            parts.append(slide.notes_slide.notes_text_frame.text)
    return "\n".join(parts)


def run_consistency_check() -> dict[str, Any]:
    facts = json.loads((REPORTS_DIR / "facts.json").read_text(encoding="utf-8"))

    doc_texts = {
        "reports/report.docx": _extract_docx_text(REPORTS_DIR / "report.docx"),
        "reports/slides.pptx": _extract_pptx_text(REPORTS_DIR / "slides.pptx"),
        "reports/speaker_script.md": (REPORTS_DIR / "speaker_script.md").read_text(encoding="utf-8"),
        "reports/qa_bank.md": (REPORTS_DIR / "qa_bank.md").read_text(encoding="utf-8"),
        "reports/demo_script.md": (REPORTS_DIR / "demo_script.md").read_text(encoding="utf-8"),
        "reports/cheat_sheet.md": (REPORTS_DIR / "cheat_sheet.md").read_text(encoding="utf-8"),
        "README.md": (PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
    }

    # Check 1: Zero occurrences of "[CHƯA CÓ SỐ]" across all deliverables
    placeholder_hits = {}
    for name, txt in doc_texts.items():
        cnt = txt.count("[CHƯA CÓ SỐ]")
        placeholder_hits[name] = cnt

    # Check 2: Key metrics from facts.json verified across documents
    ft = facts["final_test"]
    bm = facts["best_model"]
    rw = facts["realworld"]
    sp = facts["preprocessing_and_splits"]
    ds_tw = facts["datasets"]["assigned"]
    ds_hate = facts["datasets"]["hate"]

    key_facts = [
        (
            "Quy mô bộ dữ liệu được giao (Twitter_Data.csv)",
            f"{ds_tw['n_rows']:,}",
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
        ),
        (
            "Quy mô bộ dữ liệu Hate Speech thô (hate_train.csv)",
            f"{ds_hate['n_rows']:,}",
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
        ),
        (
            "Quy mô bộ dữ liệu sạch sau khử trùng (clean.csv)",
            f"{sp['n_final_unique_clean']:,}",
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
        ),
        (
            "Số dòng trùng lặp gần-giống bị loại (P4)",
            f"{sp['n_near_duplicate_rows_dropped']:,}",
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md"],
        ),
        (
            "Tên mô hình chốt (best_model)",
            bm["name"],
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
        ),
        (
            "Ngưỡng quyết định tối ưu (best_threshold)",
            f"{bm['threshold']:.3f}",
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/demo_script.md", "reports/cheat_sheet.md", "README.md"],
        ),
        (
            "Validation F1 sau tinh chỉnh & tối ưu ngưỡng (P6)",
            f"{bm['val_metrics_after_tuning_best_thr']['f1_pos']:.4f}",
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
        ),
        (
            "Test F1 Lớp dương (Final Test F1 — P8)",
            f"{ft['f1_pos']:.4f}",
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/demo_script.md", "reports/cheat_sheet.md", "README.md"],
        ),
        (
            "Test PR-AUC (P8)",
            f"{ft['pr_auc']:.4f}",
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
        ),
        (
            "Test ROC-AUC (P8)",
            f"{ft['roc_auc']:.4f}",
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md"],
        ),
        (
            "Hệ số đồng thuận Cohen's Kappa tập Thực tế (P9)",
            f"{rw['kappa']:.4f}",
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
        ),
        (
            "Real-World F1 tổng thể trên 300 mẫu (P9)",
            f"{rw['overall_metrics']['f1_pos']:.4f}",
            ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
        ),
    ]

    if "vi" in facts:
        vi = facts["vi"]
        vi_ft = vi["final_test"]
        vi_rw = vi["realworld"]
        key_facts.extend(
            [
                (
                    "Quy mô bộ dữ liệu Tiếng Việt ViHSD (V1)",
                    f"{vi['dataset']['n']:,}",
                    ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
                ),
                (
                    "Test F1 Tiếng Việt bài toán chính bin_A (V5)",
                    f"{vi_ft['bin_A']['metrics_tuned']['f1_pos']:.4f}",
                    ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
                ),
                (
                    "Test Macro-F1 Tiếng Việt 3 lớp multi3 (V5)",
                    f"{vi_ft['multi3']['f1_macro']:.4f}",
                    ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
                ),
                (
                    "Độ bền Không dấu (strip_accents Test F1 — V5)",
                    f"{vi['robustness']['no_accents']['f1_pos']:.4f}",
                    ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
                ),
                (
                    "Cohen's Kappa 3 lớp Tập Thực tế Tiếng Việt (V6)",
                    f"{vi_rw['kappa']:.4f}",
                    ["reports/report.docx", "reports/slides.pptx", "reports/speaker_script.md", "reports/qa_bank.md", "reports/cheat_sheet.md", "README.md"],
                ),
            ]
        )

    verification_rows = []
    all_passed = True
    for label, expected_str, target_docs in key_facts:
        missing_in = []
        for doc_name in target_docs:
            if expected_str not in doc_texts[doc_name]:
                missing_in.append(doc_name)
                all_passed = False
        verification_rows.append(
            {
                "metric_name": label,
                "facts_json_value": expected_str,
                "checked_files": len(target_docs),
                "status": "KHỚP 100% (PASS)" if not missing_in else f"LỆCH / THIẾU Ở: {missing_in}",
            }
        )

    total_placeholders = sum(placeholder_hits.values())
    if total_placeholders > 0:
        all_passed = False

    md_lines = [
        "# BÁO CÁO KIỂM TRA CHÉO ĐỘ NHẤT QUÁN SỐ LIỆU (CONSISTENCY CHECK — P11 / B6)",
        "",
        f"- **Thời điểm kiểm tra tự động:** `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        "- **Nguồn đối chiếu gốc (Single Source of Truth):** `reports/facts.json`",
        f"- **Tổng số chỗ còn `[CHƯA CÓ SỐ]` trên toàn bộ 7 tài liệu:** **`{total_placeholders}`** (Yêu cầu = `0`)",
        f"- **Kết luận tổng thể:** **{'ĐẠT TUYỆT ĐỐI 100% (PASS — ZERO DISCREPANCIES)' if all_passed else 'CẦN KIỂM TRA LẠI'}**",
        "",
        "---",
        "",
        "## 1. BẢNG ĐỐI CHIẾU TỪNG CON SỐ CHỐT GIỮA `facts.json` VÀ BỘ TÀI LIỆU",
        "",
        "| # | Chỉ số / Con số kiểm tra | Giá trị trong `facts.json` | Số tài liệu đối chiếu | Trạng thái kiểm tra |",
        "|---:|---|---|---:|---|",
    ]
    for i, r in enumerate(verification_rows, start=1):
        md_lines.append(
            f"| {i} | {r['metric_name']} | `{r['facts_json_value']}` | {r['checked_files']} | **{r['status']}** |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 2. KIỂM TRA KHÔNG CÒN PLACEHOLDER `[CHƯA CÓ SỐ]` TRONG TỪNG TÀI LIỆU",
            "",
            "| Tài liệu | Độ dài (ký tự) | Số lượng `[CHƯA CÓ SỐ]` | Trạng thái |",
            "|---|---:|---:|---|",
        ]
    )
    for doc_name, txt in doc_texts.items():
        cnt = placeholder_hits[doc_name]
        status = "✅ HỢP LỆ (0 placeholder)" if cnt == 0 else f"❌ CÒN {cnt} PLACEHOLDER"
        md_lines.append(f"| `{doc_name}` | {len(txt):,} | {cnt} | {status} |")

    out_path = REPORTS_DIR / "consistency_check.md"
    out_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    assert all_passed, f"Consistency check failed! Details written to {out_path}"
    print("Consistency check PASSED 100%.")
    return {"all_passed": all_passed, "rows": verification_rows}


if __name__ == "__main__":
    run_consistency_check()
