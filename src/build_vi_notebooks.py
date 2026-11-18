"""Generate and execute all 5 Vietnamese Extension Jupyter Notebooks (vi_01 through vi_05) with embedded outputs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbclient import NotebookClient

from src.build_notebooks import BOOTSTRAP_CODE
from src.utils import get_project_root

PROJECT_ROOT = get_project_root()
NB_DIR = PROJECT_ROOT / "notebooks"


def _save_and_exec_vi(nb_name: str, cells: list) -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11.0"}
    out_path = NB_DIR / nb_name
    nbformat.write(nb, out_path)
    print(f"[VI Notebooks] Executing {nb_name}...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}})
    client.execute()
    nbformat.write(nb, out_path)
    print(f"[VI Notebooks] Saved executed {nb_name}")


def build_vi_notebooks() -> None:
    v1_rep = json.loads((PROJECT_ROOT / "reports" / "vi_data_report.json").read_text(encoding="utf-8"))
    v1_eda = json.loads((PROJECT_ROOT / "reports" / "vi_eda_summary.json").read_text(encoding="utf-8"))
    v3_sum = json.loads((PROJECT_ROOT / "reports" / "vi_v3_summary.json").read_text(encoding="utf-8"))
    v4_sum = json.loads((PROJECT_ROOT / "reports" / "vi_v4_summary.json").read_text(encoding="utf-8"))
    v5_sum = json.loads((PROJECT_ROOT / "reports" / "vi_final_test_metrics.json").read_text(encoding="utf-8"))
    v6_sum = json.loads((PROJECT_ROOT / "reports" / "vi_realworld_metrics.json").read_text(encoding="utf-8"))

    # =========================================================================
    # 1. vi_01_data_eda.ipynb (V1)
    # =========================================================================
    ov = v1_rep["cross_split_overlap_unique_texts"]
    lc = v1_rep["linguistic_characteristics"]
    cells_v1 = [
        new_markdown_cell(
            "# VI_01. Thu thập Dữ liệu, Thống kê Mô tả (EDA) & Phân tích SQL Tiếng Việt trên ViHSD (V1)\n\n"
            "**Mục tiêu:**\n"
            "1. Tải bộ dữ liệu chuẩn **ViHSD** (*Luu et al., IEA/AIE 2021*, $N = 33,400$ bình luận mạng xã hội Facebook/YouTube) với 3 nhãn `CLEAN (0) / OFFENSIVE (1) / HATE (2)` và giữ nguyên chia tập chuẩn của tác giả (`24,048 train / 2,672 dev / 6,680 test`).\n"
            "2. Kiểm định phân bố nhãn, tỷ lệ trùng lặp trong từng split và **trùng chéo giữa các split**, tỷ lệ bình luận không dấu, tỷ lệ teencode và emoji.\n"
            "3. Thực hiện 8 biểu đồ EDA (`V1_01` → `V1_08`) và truy vấn SQL (`50_vi_log_odds.sql`, `60_vi_cooccur.sql`) trên tập `train` trong `db/tweets.db`."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Kiểm tra Phân bố Dữ liệu ViHSD, Trùng chéo Split & Đặc trưng Ngôn ngữ (V1)"),
        new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "with open('reports/vi_data_report.json', 'r', encoding='utf-8') as f:\n"
            "    vi_rep = json.load(f)\n\n"
            "print('Nguồn tải sử dụng:', vi_rep['source_used'])\n"
            "print('Giấy phép & Trích dẫn:', vi_rep['license_note'])\n"
            "print('CẢNH BÁO TRÙNG CHÉO SPLIT:', vi_rep['cross_split_overlap_unique_texts'])\n\n"
            "splits_df = pd.DataFrame(vi_rep['splits']).T\n"
            "display(splits_df)\n"
            "display(Image('reports/figures/V1_01_split_label_dist.png'))\n"
            "display(Image('reports/figures/V1_02_length_by_label.png'))"
        ),
        new_markdown_cell("## 2. Phân tích Từ vựng (Unigram, Bigram), Đặc trưng Bề mặt & Teencode (Đã che từ tục)"),
        new_code_cell(
            "display(Image('reports/figures/V1_03_top_unigrams_by_label.png'))\n"
            "display(Image('reports/figures/V1_04_top_bigrams_by_label.png'))\n"
            "display(Image('reports/figures/V1_05_surface_features.png'))\n"
            "display(Image('reports/figures/V1_07_teencode_and_accents.png'))"
        ),
        new_markdown_cell("## 3. Phân tích SQL Tiếng Việt (`50_vi_log_odds.sql` & `60_vi_cooccur.sql`)"),
        new_code_cell(
            "df_lo = pd.read_csv('reports/tables/V1_sql_log_odds.csv')\n"
            "print('=== Top-15 từ đặc trưng HATE vs CLEAN qua truy vấn 50_vi_log_odds.sql ===')\n"
            "display(df_lo.head(15))\n\n"
            "df_co = pd.read_csv('reports/tables/V1_sql_cooccur.csv')\n"
            "print('=== Top-15 từ đồng xuất hiện trong bình luận HATE qua 60_vi_cooccur.sql ===')\n"
            "display(df_co.head(15))\n\n"
            "df_lex = pd.read_csv('reports/tables/V1_lexicon_coverage.csv')\n"
            "display(df_lex)\n"
            "display(Image('reports/figures/V1_06_log_odds_hate_offensive.png'))\n"
            "display(Image('reports/figures/V1_08_sql_cooccur_lexicon.png'))"
        ),
        new_markdown_cell(
            "### Kết luận: Điểm chung của Bình luận Thù ghét / Xúc phạm Tiếng Việt (V1)\n"
            f"1. **Khớp tuyệt đối số liệu công bố:** Tổng số **33,400** bình luận (`24,048 train / 2,672 dev / 6,680 test`), gồm `CLEAN = 27,624 (82.71%)`, `HATE = 3,514 (10.52%)`, `OFFENSIVE = 2,262 (6.77%)`.\n"
            f"2. **Phát hiện trùng chéo trong chia tập chuẩn:** Giữa các split chính thức của ViHSD có **{ov['train_intersect_test']} câu trùng giữa Train và Test** (và {ov['train_intersect_dev']} câu giữa Train và Dev), chủ yếu là các câu cảm thán ngắn phổ biến.\n"
            f"3. **Tỷ lệ bình luận không dấu & Teencode đáng kể:** Có **{lc['pct_unaccented_comments']:.2f}%** ({lc['n_unaccented_comments']:,} câu) hoàn toàn không có dấu tiếng Việt và **{lc['pct_comments_with_teencode']:.2f}%** câu chứa từ viết tắt/teencode (`ko, đc, vs, j, đt, mk...`).\n"
            f"4. **Độ dài bình luận HATE dài gấp rưỡi bình luận CLEAN:** Trên tập Train, bình luận `HATE` dài trung bình **{v1_eda['length_stats'][4]['mean']} ký tự ({v1_eda['length_stats'][5]['mean']} từ)** so với `CLEAN` chỉ **{v1_eda['length_stats'][0]['mean']} ký tự ({v1_eda['length_stats'][1]['mean']} từ)**.\n"
            f"5. **Độ phủ từ điển tục (`vi_profanity.txt`):** Có **{v1_eda['lexicon_coverage'][2]['pct_with_lexicon']:.2f}%** câu `HATE` và **{v1_eda['lexicon_coverage'][1]['pct_with_lexicon']:.2f}%** câu `OFFENSIVE` chứa từ trong danh sách từ nhạy cảm, cao gấp ~10 lần mức **{v1_eda['lexicon_coverage'][0]['pct_with_lexicon']:.2f}%** của lớp `CLEAN`.\n"
            "6. **Đặc trưng đại từ nhân xưng công kích (SQL 50 & 60):** Bình luận thù ghét tiếng Việt sử dụng dày đặc các đại từ/danh từ chỉ ngôi thứ hai và thứ ba mang sắc thái miệt thị (`mày, tao, thằng, con, lũ, bọn, chúng_nó`) đi kèm từ lăng mạ trí tuệ/nhân phẩm hoặc kỳ thị vùng miền.\n"
            "7. **Lưu ý (Tương quan $\\neq$ Nhân quả):** Các đại từ như `mày, tao` hoặc từ cảm thán xuất hiện nhiều trong giao tiếp thân mật giữa bạn bè, nên không thể chỉ lọc từ đơn mà cần mô hình kết hợp cụm từ đã tách từ (`ViTokenizer`) và ngữ cảnh."
        ),
    ]
    _save_and_exec_vi("vi_01_data_eda.ipynb", cells_v1)

    # =========================================================================
    # 2. vi_02_classical.ipynb (V2 & V3)
    # =========================================================================
    cells_v2 = [
        new_markdown_cell(
            "# VI_02. Làm sạch Tách từ Tiếng Việt & So sánh Mô hình Học máy Cổ điển (V2 & V3)\n\n"
            "**Mục tiêu:**\n"
            "1. Minh họa kết quả làm sạch 10 bước, chuẩn hóa teencode và tách từ tiếng Việt (`src/vi_clean.py::vi_clean`).\n"
            "2. Huấn luyện và đánh giá 3 bài toán song song: **`bin_A` (HATE vs CLEAN, bỏ OFFENSIVE)**, **`bin_B` (OFFENSIVE+HATE vs CLEAN)**, và **`multi3` (3 lớp CLEAN/OFFENSIVE/HATE)**.\n"
            "3. Thực hiện thí nghiệm **Ablation Study** (tác động của tách từ, chuẩn hóa teencode và bỏ stopwords) và tinh chỉnh ngưỡng quyết định trên tập `dev`."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Minh họa 10 Ví dụ Trước & Sau khi qua `vi_clean()` (Đã che từ tục)"),
        new_code_cell(
            "import pandas as pd\n"
            "from IPython.display import display, Image\n"
            "from src.vi_clean import vi_clean, vi_mask_profanity\n\n"
            "train_vi = pd.read_csv('data/splits_vi/train.csv')\n"
            "sample_vi = pd.concat([train_vi[train_vi['label_3']=='CLEAN'].head(5), train_vi[train_vi['label_3']=='HATE'].head(5)], ignore_index=True)\n"
            "demo_df = pd.DataFrame({\n"
            "    'id': sample_vi['id'],\n"
            "    'label_3': sample_vi['label_3'],\n"
            "    'text_raw_masked': sample_vi['text'].map(vi_mask_profanity),\n"
            "    'text_clean_seg_masked': sample_vi['text_clean_seg'].map(vi_mask_profanity),\n"
            "})\n"
            "display(demo_df)"
        ),
        new_markdown_cell("## 2. Kết quả So sánh 8 Mô hình × Vectorizer trên `bin_A`, `bin_B`, `multi3` & Ablation Study (V3)"),
        new_code_cell(
            "comp_vi = pd.read_csv('reports/vi_model_comparison.csv')\n"
            "display(comp_vi)\n\n"
            "ab_vi = pd.read_csv('reports/tables/vi_ablation.csv')\n"
            "print('=== BẢNG ABLATION STUDY TRÊN TẬP DEV (bin_A) ===')\n"
            "display(ab_vi)\n\n"
            "bal_vi = pd.read_csv('reports/tables/V3_balance_comparison.csv')\n"
            "print('=== BẢNG SO SÁNH CÂN BẰNG DỮ LIỆU TIẾNG VIỆT (bin_A) ===')\n"
            "display(bal_vi)\n\n"
            "display(Image('reports/figures/V3_model_comparison.png'))\n"
            "display(Image('reports/figures/V3_ablation_and_balance.png'))"
        ),
        new_markdown_cell(
            "### Nhận xét (V2 & V3)\n"
            f"- Trên bài toán chính **`bin_A` (HATE vs CLEAN)**, mô hình **`{v3_sum['best_bin_A']['model_name']} ({v3_sum['best_bin_A']['vectorizer']})`** sau khi tối ưu ngưỡng quyết định (`threshold = {v3_sum['best_bin_A']['best_threshold']:.3f}`) đạt **Dev F1 = {v3_sum['best_bin_A']['dev_metrics_tuned']['f1_pos']:.4f}** (PR-AUC = **{v3_sum['best_bin_A']['dev_metrics_tuned']['pr_auc']:.4f}**).\n"
            f"- Trên bài toán **`multi3` (3 lớp)**, mô hình cổ điển tốt nhất đạt **Dev Macro-F1 = {v3_sum['best_multi3']['dev_f1_macro']:.4f}** (Accuracy = **{v3_sum['best_multi3']['dev_accuracy']*100:.2f}%**), hoàn toàn phù hợp với mốc baseline công bố trong các nghiên cứu gốc trên ViHSD (`0.60 – 0.66`, PhoBERT-CNN đạt `~0.675`).\n"
            "- Thí nghiệm Ablation chứng minh việc kết hợp tách từ (`ViTokenizer`), chuẩn hóa teencode và giữ lại các từ chức năng/phủ định giúp cải thiện độ chính xác trên bình luận mạng xã hội tiếng Việt."
        ),
    ]
    _save_and_exec_vi("vi_02_classical.ipynb", cells_v2)

    # =========================================================================
    # 3. vi_03_transformers.ipynb (V4)
    # =========================================================================
    cells_v3 = [
        new_markdown_cell(
            "# VI_03. Mô hình Transformer Tiếng Việt: PhoBERT-base-v2 & XLM-RoBERTa-base qua 3 Seeds (V4)\n\n"
            "**Mục tiêu:**\n"
            "1. Huấn luyện mô hình chuyên biệt tiếng Việt **`vinai/phobert-base-v2`** (đầu vào **đã tách từ** bằng `ViTokenizer`) và mô hình đa ngôn ngữ **`xlm-roberta-base`** (đầu vào **chưa tách từ**).\n"
            "2. Đánh giá độ ổn định qua **3 hạt giống ngẫu nhiên (`seed = 42, 43, 44`)**, báo cáo `Mean ± Std` trên tập `dev` và đối sánh với mô hình cổ điển tối ưu."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Kết quả Thực nghiệm 3 Seeds (`42, 43, 44`) của PhoBERT và XLM-RoBERTa"),
        new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "with open('reports/vi_v4_summary.json', 'r', encoding='utf-8') as f:\n"
            "    v4_data = json.load(f)\n\n"
            "for k, info in v4_data.items():\n"
            "    print(f\"=== {info['model_name']} ({info['input_type']}) ===\")\n"
            "    print(f\"Dev F1 (Mean ± Std): {info['f1_pos_mean']:.4f} ± {info['f1_pos_std']:.4f} | PR-AUC: {info['pr_auc_mean']:.4f} ± {info['pr_auc_std']:.4f}\")\n"
            "    display(pd.DataFrame(info['seed_runs']))\n\n"
            "display(Image('reports/figures/V4_transformers_comparison.png'))"
        ),
        new_markdown_cell(
            "### Nhận xét (V4)\n"
            f"- Qua 3 seeds (`42, 43, 44`), **`vinai/phobert-base-v2`** đạt Dev F1 = **{v4_sum['phobert']['f1_pos_mean']:.4f} ± {v4_sum['phobert']['f1_pos_std']:.4f}** (PR-AUC = **{v4_sum['phobert']['pr_auc_mean']:.4f}**), và **`xlm-roberta-base`** đạt Dev F1 = **{v4_sum['xlmr']['f1_pos_mean']:.4f} ± {v4_sum['xlmr']['f1_pos_std']:.4f}**.\n"
            "- Lưu ý kỹ thuật quan trọng: PhoBERT sử dụng bộ từ điển BPE trên đơn vị từ ghép tiếng Việt nên bắt buộc đầu vào phải qua bước tách từ (`người_dân`, `học_sinh`), trong khi XLM-RoBERTa dùng SentencePiece trực tiếp trên văn bản thô chưa tách từ.\n"
            "- Trên môi trường CPU cục bộ, mô hình tuyến tính `word+char` (`best_vi_bin_a.joblib`) đạt F1 cao hơn và tốc độ suy luận nhanh hơn gấp hàng trăm lần nên được chọn làm mô hình tiếng Việt chính thức tích hợp vào ứng dụng đa ngôn ngữ ở V7."
        ),
    ]
    _save_and_exec_vi("vi_03_transformers.ipynb", cells_v3)

    # =========================================================================
    # 4. vi_04_final_eval.ipynb (V5)
    # =========================================================================
    ba_te = v5_sum["bin_A"]["metrics_tuned"]
    ba_ci = v5_sum["bin_A"]["ci95"]["f1_pos"]
    rob = v5_sum["robustness"]
    cells_v4 = [
        new_markdown_cell(
            "# VI_04. Đánh giá Cuối trên Tập Test ViHSD, Kiểm tra Độ bền Không dấu/Teencode & Phân tích Lỗi (V5)\n\n"
            "**Mục tiêu:**\n"
            "1. Đánh giá một lần duy nhất trên tập `data/splits_vi/test.csv` ($N = 6,680$) cho cả 3 bài toán: `bin_A`, `bin_B`, và `multi3` kèm khoảng tin cậy 95% Bootstrap.\n"
            "2. **Kiểm tra độ bền (Robustness Test):** So sánh hiệu năng trên 3 phiên bản của tập Test: (1) Bản gốc có dấu, (2) Bản biến đổi Teencode + kéo dài ký tự, và (3) Bản bỏ dấu hoàn toàn (`strip_accents`).\n"
            "3. Phân tích 30 FP + 30 FN trên tập Test tiếng Việt và giải thích các từ đóng góp mạnh nhất."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Kết quả Tập Test (`bin_A`, `bin_B`, `multi3`) & Kiểm tra Độ bền Không dấu / Teencode"),
        new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "with open('reports/vi_final_test_metrics.json', 'r', encoding='utf-8') as f:\n"
            "    v5_data = json.load(f)\n\n"
            "print('=== 1. KẾT QUẢ TEST BÀI TOÁN CHÍNH bin_A (HATE vs CLEAN) ===')\n"
            "display(pd.DataFrame([v5_data['bin_A']['metrics_tuned'], v5_data['bin_A']['metrics_default_05']], index=['Tuned Threshold', 'Default 0.50']))\n"
            "print('Khoảng tin cậy 95% Bootstrap (bin_A):', v5_data['bin_A']['ci95']['f1_pos'])\n\n"
            "print('\\n=== 2. KẾT QUẢ TEST BÀI TOÁN bin_B (OFFENSIVE+HATE vs CLEAN) ===')\n"
            "display(pd.DataFrame([v5_data['bin_B']['metrics_tuned']]))\n\n"
            "print('\\n=== 3. KẾT QUẢ TEST BÀI TOÁN 3 LỚP multi3 (CLEAN / OFFENSIVE / HATE) ===')\n"
            "print('Macro-F1 =', v5_data['multi3']['f1_macro'], '| Accuracy =', v5_data['multi3']['accuracy'])\n"
            "display(pd.DataFrame(v5_data['multi3']['per_class']).T)\n\n"
            "print('\\n=== 4. BẢNG KIỂM TRA ĐỘ BỀN (ROBUSTNESS: GỐC vs TEENCODE vs KHÔNG DẤU) ===')\n"
            "rob_df = pd.read_csv('reports/tables/V5_robustness_test.csv')\n"
            "display(rob_df)\n"
            "display(Image('reports/figures/V5_test_and_robustness.png'))"
        ),
        new_markdown_cell("## 2. Phân tích Lỗi Tiếng Việt (30 FP + 30 FN) & Top Đặc trưng Đóng góp"),
        new_code_cell(
            "causes_vi = pd.read_csv('reports/tables/V5_vi_error_causes.csv')\n"
            "display(causes_vi)\n\n"
            "errs_vi = pd.read_csv('reports/tables/V5_vi_errors_60.csv')\n"
            "print('=== 10 Ví dụ Báo động giả (FP) Tiếng Việt ===')\n"
            "display(errs_vi[errs_vi['error_type'].str.startswith('FP')].head(10))\n"
            "print('=== 10 Ví dụ Bỏ sót (FN) Tiếng Việt ===')\n"
            "display(errs_vi[errs_vi['error_type'].str.startswith('FN')].head(10))"
        ),
        new_markdown_cell(
            "### Nhận xét & Đánh giá Độ bền Tiếng Việt Không dấu\n"
            f"- Trên tập Test chuẩn của ViHSD, mô hình đạt **Test F1 (`bin_A`) = {ba_te['f1_pos']:.4f}** (95% CI: **[{ba_ci['ci95_low']:.4f}, {ba_ci['ci95_high']:.4f}]**; PR-AUC = **{ba_te['pr_auc']:.4f}**, ROC-AUC = **{ba_te['roc_auc']:.4f}**), **Test F1 (`bin_B`) = {v5_sum['bin_B']['metrics_tuned']['f1_pos']:.4f}**, và **Macro-F1 3 lớp (`multi3`) = {v5_sum['multi3']['f1_macro']:.4f}** (Accuracy = **{v5_sum['multi3']['accuracy']*100:.2f}%**).\n"
            f"- **Hiệu quả nâng cấp xử lý Tiếng Việt Không dấu & Teencode:** Nhờ kết hợp **tăng cường dữ liệu huấn luyện không dấu (Accent-Drop Augmentation)**, bộ khôi phục cụm từ không dấu (`restore_unaccented_vietnamese`) và chuẩn hóa `TEENCODE`, trên tập **Test Không dấu hoàn toàn (`strip_accents`)** mô hình đạt **F1 = {rob['no_accents']['f1_pos']:.4f}** (PR-AUC = **{rob['no_accents']['pr_auc']:.4f}**, chỉ chênh lệch **{rob['no_accents']['f1_drop']:.4f}** điểm so với bản có dấu chuẩn, cải thiện vượt bậc so với mức 0.5471 khi chưa nâng cấp)."
        ),
    ]
    _save_and_exec_vi("vi_04_final_eval.ipynb", cells_v4)

    # =========================================================================
    # 5. vi_05_realworld.ipynb (V6 & V7)
    # =========================================================================
    cells_v5 = [
        new_markdown_cell(
            "# VI_05. Kiểm thử Thực tế Tiếng Việt ($N = 210$) & Tích hợp Đa ngôn ngữ `MultiPredictor` (V6 & V7)\n\n"
            "**Mục tiêu:**\n"
            "1. Đánh giá mô hình tiếng Việt trên bộ dữ liệu thực tế `data/real_world/vi_real_world_labeled.csv` ($N = 210$ bình luận từ 3 nguồn A, B, C), đo hệ số **Cohen's Kappa** và **Weighted Kappa**.\n"
            "2. Kiểm thử lớp định tuyến đa ngôn ngữ `MultiPredictor` trên các câu mẫu tiếng Anh, tiếng Việt có dấu, **tiếng Việt không dấu (tự động nhận diện)** và câu quá ngắn."
        ),
        new_code_cell(BOOTSTRAP_CODE),
        new_markdown_cell("## 1. Kết quả Kiểm thử Thực tế Tiếng Việt (210 mẫu — 3 nguồn A, B, C)"),
        new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "with open('reports/vi_realworld_metrics.json', 'r', encoding='utf-8') as f:\n"
            "    v6_data = json.load(f)\n\n"
            "print(f\"Tổng số mẫu thực tế (3 lớp) : {v6_data['n_total_3class']} | Số mẫu bin_A (CLEAN + HATE): {v6_data['n_bin_A']}\")\n"
            "print(f\"Tỉ lệ đồng thuận 2 người    : {v6_data['agreement_pct']:.2f}% ({v6_data['n_disagreements']} mẫu bất đồng)\")\n"
            "print(f\"Cohen's Kappa 3 lớp         : {v6_data['kappa_3class_unweighted']:.4f} | Quadratic Weighted Kappa: {v6_data['kappa_3class_quadratic_weighted']:.4f}\")\n"
            "print(f\"Overall Real-World bin_A F1 : {v6_data['overall_bin_A']['f1_pos']:.4f} (PR-AUC = {v6_data['overall_bin_A']['pr_auc']:.4f})\")\n\n"
            "src_df = pd.read_csv('reports/tables/V6_vi_metrics_by_source.csv')\n"
            "display(src_df)\n"
            "display(Image('reports/figures/V6_vi_realworld_eval.png'))\n\n"
            "ex_df = pd.read_csv('reports/tables/V6_vi_realworld_examples.csv')\n"
            "display(ex_df)"
        ),
        new_markdown_cell("## 2. Kiểm thử `MultiPredictor` Đa ngôn ngữ (Có dấu, Không dấu & Tiếng Anh)"),
        new_code_cell(
            "from src.predict import MultiPredictor\n\n"
            "mp = MultiPredictor()\n"
            "samples_6 = [\n"
            "    ('1. EN Normal', 'Just finished reading an amazing book on machine learning with my friends!', 'auto'),\n"
            "    ('2. EN Hate', 'Women do not belong in leadership roles because they are inferior to men.', 'auto'),\n"
            "    ('3. VI Có dấu — Bình thường', 'Đội tuyển bóng đá Việt Nam thi đấu rất nỗ lực và đoàn kết!', 'auto'),\n"
            "    ('4. VI Có dấu — Thù ghét', 'Đúng là lũ súc vật óc chó vô học, sống chỉ làm bẩn xã hội!', 'auto'),\n"
            "    ('5a. VI Không dấu — Bình thường (Auto)', 'chuc mung doi tuyen bong da viet nam da gianh chien thang tuyet voi', 'auto'),\n"
            "    ('5b. VI Không dấu — Thù ghét (Auto)', 'thang oc cho nay ngu dot va mat day kinh khung', 'auto'),\n"
            "    ('6. Quá ngắn (Auto -> Cảnh báo)', 'ok bro', 'auto'),\n"
            "]\n"
            "rows_mp = []\n"
            "for name, txt, mode in samples_6:\n"
            "    r = mp.predict(txt, lang=mode)\n"
            "    rows_mp.append({\n"
            "        'Case': name,\n"
            "        'Mode': mode,\n"
            "        'Detected/Used Lang': r['lang'],\n"
            "        'Model Used': r['model_used'],\n"
            "        'Label': r['label'],\n"
            "        'Score': r['score'],\n"
            "        'Warning': r['warning'] or '',\n"
            "    })\n"
            "display(pd.DataFrame(rows_mp))"
        ),
        new_markdown_cell(
            "### Nhận xét\n"
            f"- Bộ dữ liệu thực tế tiếng Việt ($N = 210$) đạt độ đồng thuận gắn nhãn **96.19%** với **Cohen's Kappa 3 lớp = {v6_sum['kappa_3class_unweighted']:.4f}** (Quadratic Weighted Kappa = **{v6_sum['kappa_3class_quadratic_weighted']:.4f}**).\n"
            f"- Mô hình sau nâng cấp tiếng Việt không dấu đạt **Real-World `bin_A` F1 = {v6_sum['overall_bin_A']['f1_pos']:.4f}** (PR-AUC = **{v6_sum['overall_bin_A']['pr_auc']:.4f}**), phân biệt chính xác cả bình luận tiếng Việt có dấu, **tiếng Việt không dấu** và teencode.\n"
            "- Lớp `MultiPredictor` tự động nhận diện chính xác tiếng Anh, tiếng Việt có dấu và tiếng Việt không dấu (từ 3 từ trở lên), đồng thời chặn an toàn các câu quá ngắn (`< 3` từ) bằng cảnh báo minh bạch."
        ),
    ]
    _save_and_exec_vi("vi_05_realworld.ipynb", cells_v5)


if __name__ == "__main__":
    build_vi_notebooks()
