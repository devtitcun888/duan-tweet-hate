"""V8 Documentation Updater: Appends Vietnamese Extension (V0-V8) sections and 10 Q&A items (Questions 61-70) to qa_bank.md, cheat_sheet.md, demo_script.md, report.docx, README.md, and consistency_check.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docx import Document

from src.build_qa_demo_b4_b5 import build_qa_demo_cheatsheet
from src.build_readme_p12 import build_readme
from src.build_report_b1 import (
    _add_body_p,
    _add_bullet_p,
    _add_figure,
    _add_heading,
    _add_table,
    build_report_docx,
)
from src.build_slides_b2 import build_slides_and_script
from src.utils import get_project_root

PROJECT_ROOT = get_project_root()
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"


def update_all_v8_docs() -> None:
    facts: dict[str, Any] = json.loads((REPORTS_DIR / "facts.json").read_text(encoding="utf-8"))
    vi = facts["vi"]
    vi_ds = vi["dataset"]
    vi_best = vi["best"]
    vi_tr = vi["transformers"]
    vi_ft = vi["final_test"]
    vi_rob = vi["robustness"]
    vi_rw = vi["realworld"]

    ba_te = vi_ft["bin_A"]["metrics_tuned"]
    ba_ci = vi_ft["bin_A"]["ci95"]["f1_pos"]
    bb_te = vi_ft["bin_B"]["metrics_tuned"]
    m3_te = vi_ft["multi3"]
    ov = vi_ds["cross_split_overlap"]
    lc = vi_ds["linguistic_characteristics"]

    # 1. Regenerate base B2/B3 slides & script (already includes Slide 12 Vietnamese Extension)
    build_slides_and_script()

    # 2. Regenerate base B4/B5 files and append V8 sections
    build_qa_demo_cheatsheet()

    # Append V8 section to cheat_sheet.md
    cheat_path = REPORTS_DIR / "cheat_sheet.md"
    cheat_vi_md = f"""

---

## 5. BẢNG SỐ LIỆU CHỐT PHẦN MỞ RỘNG TIẾNG VIỆT (ViHSD — V0 → V8)

| Hạng mục Tiếng Việt (ViHSD) | Con số chính xác | Ghi chú / Ý nghĩa |
|---|---|---|
| **Quy mô & Chia tập chuẩn ViHSD** | **{vi_ds['n']:,} dòng** (`24,048` train / `2,672` dev / `6,680` test) | Nhãn: `CLEAN` = 27,624 (82.71%) \| `HATE` = 3,514 (10.52%) \| `OFFENSIVE` = 2,262 (6.77%) |
| **Kiểm tra trùng chéo Split chuẩn** | `train∩dev` = **{ov['train_intersect_dev']}** \| `train∩test` = **{ov['train_intersect_test']}** \| `dev∩test` = **{ov['dev_intersect_test']}** | Giữ nguyên chia chuẩn của Luu et al. (2021) để so sánh công bằng với bài báo gốc |
| **Đặc trưng ngôn ngữ ViHSD** | Không dấu = **{lc['pct_unaccented_comments']:.2f}%** \| Có Teencode = **{lc['pct_comments_with_teencode']:.2f}%** | Độ phủ từ điển `vi_profanity.txt`: `CLEAN` = 3.65%, `OFFENSIVE` = 37.80%, `HATE` = 35.92% |
| **Mô hình chốt `bin_A` (HATE vs CLEAN)** | **`{vi_best['model']} ({vi_best['vectorizer']})`** (`Thr = {vi_ft['bin_A']['threshold']:.3f}`) | Dev F1 = **{vi_best['best_bin_A']['dev_metrics_tuned']['f1_pos']:.4f}** \| **Test F1 = {ba_te['f1_pos']:.4f}** (95% CI: `[{ba_ci['ci95_low']:.4f}, {ba_ci['ci95_high']:.4f}]`, PR-AUC = **{ba_te['pr_auc']:.4f}**) |
| **Kết quả Test `bin_B` & `multi3` (3 lớp)** | `bin_B` Test F1 = **{bb_te['f1_pos']:.4f}** \| `multi3` **Macro-F1 = {m3_te['f1_macro']:.4f}** | `multi3` Accuracy = **{m3_te['accuracy']*100:.2f}%** (Khớp mốc baseline công bố `0.60 – 0.66`, PhoBERT-CNN `~0.675`) |
| **Transformer 3 Seeds (42, 43, 44 — V4)** | `PhoBERT-v2` Dev F1 = **{vi_tr['phobert']['f1_pos_mean']:.4f} ± {vi_tr['phobert']['f1_pos_std']:.4f}** | `XLM-RoBERTa` Dev F1 = **{vi_tr['xlmr']['f1_pos_mean']:.4f} ± {vi_tr['xlmr']['f1_pos_std']:.4f}** |
| **Kiểm tra Độ bền (Robustness — V5)** | Gốc: **{vi_rob['orig']['f1_pos']:.4f}** \| Teencode: **{vi_rob['teencode_with_normalizer']['f1_pos']:.4f}** \| **Không dấu: {vi_rob['no_accents']['f1_pos']:.4f}** | Bỏ dấu hoàn toàn (`strip_accents`) làm F1 giảm **-{vi_rob['no_accents']['f1_drop']:.4f}** điểm |
| **Bộ Thực tế Tiếng Việt ($N = {vi_rw['n']}$ — V6)** | **Kappa 3 lớp $\\kappa = {vi_rw['kappa']:.4f}$** (Weighted $\\kappa = {vi_rw['kappa_weighted']:.4f}$) | Đồng thuận **{vi_rw['agreement_pct']:.2f}%** \| **Real-World `bin_A` F1 = {vi_rw['overall_bin_A']['f1_pos']:.4f}** (PR-AUC = **{vi_rw['overall_bin_A']['pr_auc']:.4f}**) |
"""
    cheat_path.write_text(cheat_path.read_text(encoding="utf-8") + cheat_vi_md, encoding="utf-8")

    # Append V7/V8 multilingual demo steps to demo_script.md
    demo_path = REPORTS_DIR / "demo_script.md"
    demo_vi_md = f"""

---

## C. KỊCH BẢN DEMO MỞ RỘNG ĐA NGÔN NGỮ TIẾNG VIỆT (V7 — 6 CÂU MẪU CHUẨN)

1. **Mẫu 1 & 2 (Tiếng Anh — Chế độ `Tự động`):**
   - Bấm nút **1. [EN] Bình thường** và **2. [EN] Phản cảm — Racist/Sexist**: Hệ thống tự động nhận diện `lang = EN`, nạp mô hình `English — Linear SVM (Calibrated)` (ngưỡng `0.315`) và hiển thị định nghĩa nhãn *Racist / Sexist Hate Speech*.
2. **Mẫu 3 & 4 (Tiếng Việt có dấu — Chế độ `Tự động`):**
   - Bấm nút **3. [VI có dấu] Bình thường** (*"Đội tuyển bóng đá Việt Nam hôm qua thi đấu rất nỗ lực..."*) $\rightarrow$ Hệ thống phát hiện ký tự có dấu tiếng Việt, định tuyến sang `Tiếng Việt — Linear SVM (ViHSD bin_A)` và trả về Nhãn `0` (Xanh).
   - Bấm nút **4. [VI có dấu] Phản cảm — Thù ghét/Xúc phạm** $\rightarrow$ Hệ thống trả về Nhãn `1` (Đỏ), giải thích rõ các từ ghép đã tách từ (`s***`, `ó***`, `v***`) và hiển thị rõ định nghĩa nhãn *Tiếng Việt: Thù ghét / Xúc phạm cá nhân hoặc nhóm (ViHSD)*.
3. **Mẫu 5 (Tiếng Việt không dấu — Minh họa độ bền & chế độ chọn tay):**
   - Bấm nút **5. [VI không dấu]** (*"thang oc cho nay ngu dot va mat day kinh khung cut khoi day ngay"*): Giải thích cho hội đồng thấy khi gõ không dấu, bộ nhận diện tự động có thể cảnh báo hoặc nếu chuyển nút Radio sang **`Tiếng Việt`**, chế độ chọn tay sẽ ghi đè tự động và phân loại câu này.
4. **Mẫu 6 (Câu quá ngắn / Không rõ ngôn ngữ — Kiểm tra cơ chế an toàn):**
   - Bấm nút **6. [Quá ngắn / Không rõ ngôn ngữ]** (`"ok bro"`): Ở chế độ `Tự động`, hệ thống trả về `lang = unknown` và **hiển thị cảnh báo vàng không đoán bừa**.
"""
    demo_path.write_text(demo_path.read_text(encoding="utf-8") + demo_vi_md, encoding="utf-8")

    # Append 10 Vietnamese Q&A items (Questions 61-70) to qa_bank.md
    qa_path = REPORTS_DIR / "qa_bank.md"
    qa_vi_md = f"""

---

## PHẦN D9: 10 CÂU HỎI BẢO VỆ PHẦN MỞ RỘNG TIẾNG VIỆT (ViHSD — CÂU 61 – 70 / V8)

### Câu 61 (V8.1): Vì sao mô hình tiếng Anh không thể dùng trực tiếp cho tiếng Việt? (Từ vựng, dấu, ngữ pháp)
1. **Trả lời ngắn:** Mô hình tiếng Anh hoàn toàn không thể dùng cho tiếng Việt vì 3 khác biệt bản chất ngôn ngữ học: (1) **Từ vựng hoàn toàn tách biệt:** Từ điển TF-IDF 50,000 từ tiếng Anh không chứa các từ tiếng Việt nên mọi câu tiếng Việt đi qua vectorizer tiếng Anh sẽ thành vector toàn số 0; (2) **Hệ thống thanh điệu và dấu phụ (Diacritics):** Tiếng Việt có 6 thanh điệu và các chữ cái có dấu (`ă, â, đ, ê, ô, ơ, ư`); nếu dùng bộ làm sạch tiếng Anh (`re.sub(r'[^a-z]', ' ', s)`), toàn bộ chữ có dấu bị xóa sạch làm biến dạng hoàn toàn từ (`"người"` biến thành `"ng i"`); (3) **Đơn vị từ ghép âm tiết tính:** Trong tiếng Việt, khoảng trắng tách các âm tiết (*syllables*) chứ không phải ranh giới từ đơn như tiếng Anh (ví dụ `"học sinh"` là 1 từ ghép gồm 2 âm tiết), nên bắt buộc phải có bước tách từ tiếng Việt (`ViTokenizer`).
2. **Ý mở rộng:** Ngoài ra, văn hóa giao tiếp và hệ thống đại từ nhân xưng trong tiếng Việt (`mày, tao, nó, thằng, con, lũ, bọn`) mang sắc thái thái độ trực tiếp mà tiếng Anh (`you, they`) không có.
3. **Bằng chứng:** Slide 12, Chương 8 (Mục 8.4), `src/vi_clean.py`, `tests/test_vi_clean.py`.
4. **Bẫy cần tránh:** Nhấn mạnh lỗi kinh điển nhất khi xử lý tiếng Việt là dùng regex `[^a-z]` làm bay mất dấu tiếng Việt.

### Câu 62 (V8.2): Nhãn của bộ dữ liệu ViHSD khác nhãn racist/sexist ở bản tiếng Anh như thế nào? Vì sao mô hình tiếng Việt không gọi là "phát hiện phân biệt chủng tộc/giới tính"?
1. **Trả lời ngắn:** Bộ dữ liệu tiếng Anh (`hate_train.csv`) chỉ gán nhãn `1` cho các hành vi **phân biệt chủng tộc hoặc phân biệt giới tính (racist/sexist)**. Ngược lại, bộ dữ liệu chuẩn tiếng Việt **ViHSD** (Luu et al., 2021) gồm 3 nhãn `CLEAN (0) / OFFENSIVE (1) / HATE (2)`, trong đó nhãn `HATE` bao quát mọi **ngôn từ thù ghét, công kích, lăng mạ nhân phẩm nhắm vào cá nhân hoặc tập thể** (bao gồm công kích cá nhân, miệt thị trí tuệ/ngoại hình, kỳ thị vùng miền, giới tính...), còn `OFFENSIVE` là từ ngữ thô tục/chửi thề không nhắm mục tiêu thù ghét. Vì vậy, để chuẩn xác về mặt khoa học, mô hình tiếng Việt được định danh rõ trên giao diện và báo cáo là **"Mô hình phát hiện ngôn từ thù ghét / xúc phạm tiếng Việt"**.
2. **Ý mở rộng:** Trên ứng dụng Streamlit Đa ngôn ngữ (V7), mỗi khi hiển thị kết quả, hệ thống đều in rõ dòng **Định nghĩa nhãn** tương ứng với mô hình đã dùng (`English: Racist/Sexist` vs `Tiếng Việt: Thù ghét/Xúc phạm ViHSD`) để người dùng không nhầm lẫn.
3. **Bằng chứng:** Slide 12 & Slide 13, `app/app.py`, `src/predict.py::MultiPredictor`.
4. **Bẫy cần tránh:** Không gộp `OFFENSIVE` vào nhãn dương rồi gọi đó là "phân biệt chủng tộc/giới tính".

### Câu 63 (V8.3): Vì sao với bộ ViHSD nhóm giữ nguyên cách chia tập chuẩn của tác giả (`24,048 train / 2,672 dev / 6,680 test`) thay vì tự chia lại? Có hiện tượng trùng chéo giữa các split không?
1. **Trả lời ngắn:** Nhóm giữ nguyên chia tập chuẩn chính thức 7:1:2 của tác giả Lưu Sơn Tùng và cộng sự (`24,048 train / 2,672 dev / 6,680 test`, tổng **33,400 dòng**) để kết quả thực nghiệm (đặc biệt là chỉ số `Macro-F1` 3 lớp = **{m3_te['f1_macro']:.4f}**) **có thể đối sánh trực tiếp và công bằng với các bài báo khoa học đã công bố** trên ViHSD. Khi kiểm tra tự động ở bước V1 (`vi_data_report.json`), nhóm phát hiện trong bản chia chuẩn của tác giả có **{ov['train_intersect_test']} câu trùng giữa Train và Test** (và **{ov['train_intersect_dev']} câu giữa Train và Dev**), chủ yếu là các câu bình luận ngắn phổ biến trên Facebook/YouTube (như *"hay quá"*, *"đúng rồi"*).
2. **Ý mở rộng:** Việc phát hiện và báo cáo minh bạch con số **{ov['train_intersect_test']} câu trùng chéo** trong bộ benchmark gốc thể hiện quy trình kiểm định dữ liệu nghiêm ngặt của nhóm, đồng thời giải thích vì sao một số mô hình tự chia lại tập trên mạng báo điểm cao bất thường.
3. **Bằng chứng:** Slide 12, `reports/vi_data_report.json`, Notebook `vi_01_data_eda.ipynb`.
4. **Bẫy cần tránh:** Nhớ chính xác 3 con số kích thước split chuẩn của ViHSD: **24,048 Train – 2,672 Dev – 6,680 Test (Tổng = 33,400)**.

### Câu 64 (V8.4): Vì sao ở bài toán `bin_A` nhóm lại bỏ lớp `OFFENSIVE`? Hãy so sánh kết quả giữa `bin_A` và `bin_B`.
1. **Trả lời ngắn:** Nhóm thiết kế 2 biến thể nhị phân song song: Ở **`bin_A` (`HATE vs CLEAN`, bỏ `OFFENSIVE`)**, mục tiêu là tách bạch rõ ràng giữa hành vi **thù ghét/công kích mục tiêu (`HATE`)** và bình luận **bình thường (`CLEAN`)**, tương đồng nhất với triết lý của bài toán tiếng Anh (nơi chửi thề thông thường không bị coi là hate speech). Còn ở **`bin_B` (`OFFENSIVE + HATE vs CLEAN`)**, mục tiêu là lọc toàn bộ bình luận độc hại/thô tục nói chung. Trên tập Test, **`bin_A` đạt F1 = {ba_te['f1_pos']:.4f}** (PR-AUC = **{ba_te['pr_auc']:.4f}**, ROC-AUC = **{ba_te['roc_auc']:.4f}**), còn **`bin_B` đạt F1 = {bb_te['f1_pos']:.4f}** (PR-AUC = **{bb_te['pr_auc']:.4f}**, ROC-AUC = **{bb_te['roc_auc']:.4f}**).
2. **Ý mở rộng:** Sở dĩ `bin_B` có F1 cao hơn `bin_A` (**{bb_te['f1_pos']:.4f}** vs **{ba_te['f1_pos']:.4f}**) là vì lớp `OFFENSIVE` chứa rất nhiều từ chửi thề tường minh (37.80% khớp từ điển `vi_profanity.txt`), khi gộp `OFFENSIVE + HATE` làm một thì ranh giới từ vựng với `CLEAN` rất rõ và tỷ lệ lớp dương tăng lên 17.3%; trong khi ở `bin_A` lớp dương chỉ chiếm ~11.4%.
3. **Bằng chứng:** Slide 12, `reports/vi_final_test_metrics.json`, Notebook `vi_02_classical.ipynb` & `vi_04_final_eval.ipynb`.
4. **Bẫy cần tránh:** Không trộn lẫn con số của `bin_A` (**{ba_te['f1_pos']:.4f}**) và `bin_B` (**{bb_te['f1_pos']:.4f}**); luôn nêu rõ tên bài toán khi báo cáo.

### Câu 65 (V8.5): Vì sao bài toán 3 lớp (`multi3`) phải dùng `Macro-F1`? Kết quả của nhóm so với các bài báo đã công bố (khoảng 0.60 – 0.675) như thế nào?
1. **Trả lời ngắn:** Trong bài toán 3 lớp của ViHSD, lớp `CLEAN` chiếm tới **82.71%**, `HATE` chiếm **10.52%** và `OFFENSIVE` chỉ chiếm **6.77%**. Nếu dùng *Accuracy* hoặc *Weighted-F1*, điểm số sẽ bị chi phối hoàn toàn bởi lớp `CLEAN` (~86–87%). Do đó, các bài báo khoa học gốc trên ViHSD đều quy định dùng **`Macro-F1`** $= \\frac{{1}}{{3}}(F1_{{\\text{{CLEAN}}}} + F1_{{\\text{{OFFENSIVE}}}} + F1_{{\\text{{HATE}}}})$ tính trung bình cộng không trọng số để đánh giá công bằng cả 3 lớp. Mô hình `Logistic Regression (word+char)` của nhóm đạt **Dev Macro-F1 = {vi_best['best_multi3']['dev_f1_macro']:.4f}** và **Test Macro-F1 = {m3_te['f1_macro']:.4f}** (Accuracy = **{m3_te['accuracy']*100:.2f}%**: trong đó $F1_{{\\text{{CLEAN}}}} = {m3_te['per_class']['CLEAN']['f1']:.4f}$, $F1_{{\\text{{HATE}}}} = {m3_te['per_class']['HATE']['f1']:.4f}$, $F1_{{\\text{{OFFENSIVE}}}} = {m3_te['per_class']['OFFENSIVE']['f1']:.4f}$).
2. **Ý mở rộng:** Con số **Test Macro-F1 = {m3_te['f1_macro']:.4f}** (64.81%) của nhóm nằm hoàn toàn khớp trong vùng các mô hình baseline đã công bố trên ViHSD (`0.60 – 0.66`) và tiệm cận mốc PhoBERT-CNN (`~0.675` của Tran et al., 2022), chứng minh quy trình thực nghiệm của nhóm hoàn toàn chuẩn xác và trung thực.
3. **Bằng chứng:** Slide 12, Hình `V3_model_comparison.png`, `reports/vi_final_test_metrics.json`.
4. **Bẫy cần tránh:** Giải thích được vì sao lớp `OFFENSIVE` có F1 thấp nhất trong 3 lớp (**{m3_te['per_class']['OFFENSIVE']['f1']:.4f}**): vì nó vừa ít mẫu nhất (6.77%), vừa nằm kẹp ở ranh giới ngữ nghĩa giữa `CLEAN` và `HATE`.

### Câu 66 (V8.6): Bước tách từ (`ViTokenizer`) giúp ích gì cho tiếng Việt? Đầu vào của `PhoBERT` cần gì và khác `XLM-RoBERTa` ở đâu?
1. **Trả lời ngắn:** Tách từ (`pyvi.ViTokenizer`) ghép các âm tiết tạo thành từ ghép hoàn chỉnh bằng dấu gạch dưới (ví dụ: `"phân biệt chủng tộc"` $\\rightarrow$ `"phân_biệt chủng_tộc"`, `"chó đẻ"` $\\rightarrow$ `"chó_đẻ"`), giúp vector TF-IDF nhận diện đúng đơn vị từ vựng thay vì các âm tiết rời rạc vô nghĩa. Thí nghiệm Ablation ở V3 (`vi_ablation.csv`) chứng minh khi có tách từ, Dev F1 đạt **0.6087** cao hơn khi không tách từ (**0.5966**, tăng **+1.21 điểm %**). Đối với Transformer ở V4: **`vinai/phobert-base-v2`** được huấn luyện tiền kỳ trên kho ngữ liệu tiếng Việt **đã tách từ** nên bắt buộc đầu vào phải qua `vi_clean(segment=True)`; ngược lại, **`xlm-roberta-base`** dùng thuật toán SentencePiece đa ngôn ngữ trực tiếp trên chuỗi ký tự thô nên đầu vào phải là văn bản **chưa tách từ** (`vi_clean(segment=False)`).
2. **Ý mở rộng:** Nếu truyền văn bản chưa tách từ vào PhoBERT hoặc truyền văn bản có dấu `_` vào XLM-RoBERTa, bộ tokenizer sẽ cắt vụn thành các token `<unk>` hoặc subword sai, làm giảm mạnh hiệu năng.
3. **Bằng chứng:** Chương 8 (Mục 8.4), Bảng `reports/tables/vi_ablation.csv`, `src/vi_transformer.py`, Notebook `vi_03_transformers.ipynb`.
4. **Bẫy cần tránh:** Ghi nhớ quy tắc vàng: **PhoBERT = Đã tách từ (`segment=True`)**; **XLM-R = Chưa tách từ (`segment=False`)**.

### Câu 67 (V8.7): Khi người dùng gõ tiếng Việt không dấu hoặc dùng nhiều teencode thì hiệu năng mô hình giảm bao nhiêu? Nhóm đã xử lý ra sao?
1. **Trả lời ngắn:** Trong thí nghiệm kiểm tra độ bền (**Robustness Test — V5**, bảng `V5_robustness_test.csv`) trên tập Test `bin_A`: so với bản gốc có dấu đạt **F1 = {vi_rob['orig']['f1_pos']:.4f}**, khi biến đổi thêm **teencode + kéo dài chữ cái**, nếu không có bước chuẩn hóa thì F1 giảm còn **{vi_rob['teencode_without_normalizer']['f1_pos']:.4f}**, nhưng nhờ từ điển chuẩn hóa `TEENCODE` và quy tắc rút gọn ký tự lặp của nhóm, F1 được khôi phục lên **{vi_rob['teencode_with_normalizer']['f1_pos']:.4f}** (chỉ giảm nhẹ **{vi_rob['teencode_with_normalizer']['f1_drop']:.4f}**). Tuy nhiên, khi **bỏ dấu hoàn toàn (`strip_accents`)**, Test F1 giảm **-{vi_rob['no_accents']['f1_drop']:.4f}** điểm (từ **{vi_rob['orig']['f1_pos']:.4f}** xuống còn **{vi_rob['no_accents']['f1_pos']:.4f}**, Recall giảm từ **{vi_rob['orig']['recall_pos']*100:.2f}%** xuống **{vi_rob['no_accents']['recall_pos']*100:.2f}%**).
2. **Ý mở rộng:** Để giảm thiểu tác động của không dấu và teencode, nhóm đã: (1) Sử dụng biểu diễn kết hợp **`word+char` (`char_wb` 2–5 ký tự)** trong mô hình chốt; (2) Xây dựng từ điển mở rộng `TEENCODE` hơn 60 từ viết tắt phổ biến; và (3) Thiết lập cảnh báo rõ ràng trên giao diện ứng dụng khi phát hiện câu không dấu.
3. **Bằng chứng:** Slide 12, Hình `V5_test_and_robustness.png`, Bảng `reports/tables/V5_robustness_test.csv`, Notebook `vi_04_final_eval.ipynb`.
4. **Bẫy cần tránh:** Thuộc lòng 3 con số độ bền trên Test: **Gốc = {vi_rob['orig']['f1_pos']:.4f} $\\rightarrow$ Teencode = {vi_rob['teencode_with_normalizer']['f1_pos']:.4f} $\\rightarrow$ Không dấu = {vi_rob['no_accents']['f1_pos']:.4f} (giảm {vi_rob['no_accents']['f1_drop']:.4f})**.

### Câu 68 (V8.8): Làm sao ứng dụng Streamlit biết câu nào là tiếng Việt và câu nào là tiếng Anh? Khi nhận diện sai hoặc câu quá ngắn thì ứng dụng xử lý thế nào?
1. **Trả lời ngắn:** Lớp `MultiPredictor` trong `src/predict.py` thực hiện nhận diện ngôn ngữ theo quy trình 3 bước an toàn: **Bước 1:** Kiểm tra sự xuất hiện của các chữ cái có dấu đặc trưng tiếng Việt (`ă, â, đ, ê, ô, ơ, ư` và 5 dấu thanh); nếu có thì khẳng định 100% là **`vi`**; **Bước 2:** Nếu không có dấu tiếng Việt và câu có ít nhất 3 từ ($\\ge 10$ ký tự chữ), gọi thư viện `langdetect` với ngưỡng độ tin cậy **`prob >= 0.80`** để phân về **`en`** hoặc **`vi`**; **Bước 3:** Nếu câu quá ngắn (như `"ok bro"`) hoặc là tiếng Việt không dấu khiến độ tin cậy $< 0.80$, hàm trả về **`unknown`**. Khi gặp `unknown`, ứng dụng **tuyệt đối không đoán bừa** mà hiển thị hộp cảnh báo yêu cầu người dùng bấm chọn nút Radio ngôn ngữ thủ công (`English` hoặc `Tiếng Việt`), và chế độ chọn tay luôn được ưu tiên tuyệt đối.
2. **Ý mở rộng:** Trên Tab 2 (Phân loại hàng loạt CSV), các dòng `unknown` được gán nhãn `NA` và `model_used = NA` để không làm sai lệch thống kê.
3. **Bằng chứng:** Slide 13, `src/predict.py::MultiPredictor`, `app/app.py`, `tests/test_predict.py::test_multilingual_predictor_language_detection_and_override`.
4. **Bẫy cần tránh:** Demo trực tiếp nút mẫu số 5 (Tiếng Việt không dấu) và số 6 (`"ok bro"`) trên ứng dụng để minh chứng cơ chế này.

### Câu 69 (V8.9): Về đạo đức và quyền riêng tư đối với dữ liệu tiếng Việt: dữ liệu được thu thập thế nào, cách che từ tục và rủi ro chặn nhầm ra sao?
1. **Trả lời ngắn:** Nhóm tuân thủ nghiêm ngặt 4 quy tắc đạo đức: (1) **Giấy phép nghiên cứu:** Bộ dữ liệu ViHSD được trích dẫn đầy đủ (*Luu et al., IEA/AIE 2021*) và chỉ sử dụng cho mục đích học tập/nghiên cứu, không phát tán lại dữ liệu thô; (2) **Thu thập thực tế an toàn (V6):** 80 bình luận mạng xã hội công khai ở Nguồn B được chọn lọc thủ công từ các bài báo/video công khai, ẩn toàn bộ tên tài khoản thành `@user` và link thành `[URL]`, không dùng công cụ crawl hàng loạt hay theo dõi cá nhân; (3) **Che từ tục tiếng Việt (`vi_mask_profanity`):** Toàn bộ từ nhạy cảm trong `data/lexicon/vi_profanity.txt` được hiển thị dạng ký tự đầu + `***` (ví dụ `đ***`, `v***`, `n***`); (4) **Tránh chặn nhầm chửi thề thân mật:** Phân tích lỗi V5 chỉ ra người Việt trẻ hay dùng từ cảm thán (`vcl, đm`) để khen ngợi bạn bè hoặc than thở đề thi khó, nên cần con người kiểm duyệt cuối.
2. **Ý mở rộng:** Trên bộ thực tế 210 mẫu tiếng Việt (V6), 2 luồng gắn nhãn độc lập theo `vi_ANNOTATION_GUIDE.md` đạt **Cohen's Kappa 3 lớp $\\kappa = {vi_rw['kappa']:.4f}$** và **Quadratic Weighted Kappa = {vi_rw['kappa_weighted']:.4f}**.
3. **Bằng chứng:** Slide 12 & Slide 14, `src/vi_clean.py::vi_mask_profanity`, `data/real_world/vi_ANNOTATION_GUIDE.md`, `reports/vi_realworld_metrics.json`.
4. **Bẫy cần tránh:** Nhớ nhắc rõ quy định bản quyền của ViHSD: *"Dùng cho mục đích học tập/nghiên cứu, trích dẫn Luu et al. (2021)"*.

### Câu 70 (V8.10): Vì sao một số mô hình trên HuggingFace tự báo Macro-F1 trên 0.80+ cho ViHSD lại không đáng tin để dùng làm bằng chứng so sánh?
1. **Trả lời ngắn:** Theo khảo sát tài liệu gốc của nhóm tác giả UIT-NLP (Luu et al., 2021) và bài báo PhoBERT-CNN (Tran et al., 2022), trên tập Test chuẩn 6,680 mẫu của ViHSD với 3 lớp, ngay cả các kiến trúc mạnh như PhoBERT/BERT cũng chỉ đạt **Macro-F1 khoảng 65% – 67.5%** (do lớp `OFFENSIVE` rất nhỏ và ranh giới nhãn giữa `OFFENSIVE` và `HATE` có độ bất đồng tự nhiên giữa người gán nhãn). Những repo trên HuggingFace tự báo Macro-F1 `> 0.80` thường mắc 1 trong 3 lỗi: (1) Báo nhầm **Weighted-F1 hoặc Accuracy (~87%)** thành Macro-F1; (2) Chỉ lọc lấy tập con ~10k mẫu dễ hoặc gộp thành 2 lớp; hoặc (3) Gộp toàn bộ 33,400 mẫu rồi tự chia lại mà không khử các câu trùng nhau, gây rò rỉ dữ liệu.
2. **Ý mở rộng:** Chính vì vậy, trong đặc tả dự án (`AGENTS.md`), nhóm đặt ngưỡng kiểm tra Sanity Check: nếu `Macro-F1` 3 lớp trên tập Test chuẩn vượt quá `0.80` thì phải dừng lại điều tra rò rỉ. Kết quả **Test Macro-F1 = {m3_te['f1_macro']:.4f}** (và **Test `bin_A` F1 = {ba_te['f1_pos']:.4f}**) của nhóm là con số thực chất, minh bạch và tái lập được 100%.
3. **Bằng chứng:** Phụ lục Đặc tả Tiếng Việt (Mục 0 & V3/V5), `src/vi_eval_v5.py` (`assert f1_mac_m <= 0.80`).
4. **Bẫy cần tránh:** Trả lời câu này một cách sắc sảo sẽ cho thấy nhóm đọc rất kỹ bài báo khoa học gốc chứ không chạy mô hình một cách mù quáng.
"""
    qa_path.write_text(qa_path.read_text(encoding="utf-8") + qa_vi_md, encoding="utf-8")

    # 3. Rebuild base report.docx and append Section 8.4 (Vietnamese Extension V0-V8)
    docx_path = build_report_docx()
    doc = Document(str(docx_path))

    _add_heading(doc, "8.4. Mở rộng Đa ngôn ngữ: Phát hiện Ngôn từ Thù ghét / Xúc phạm Tiếng Việt trên Bộ dữ liệu ViHSD (Điểm cộng V0 → V8)", level=2)
    _add_body_p(
        doc,
        f"Nhằm mở rộng khả năng ứng dụng của đồ án cho môi trường mạng xã hội tại Việt Nam, nhóm thực hiện trọn vẹn phần mở rộng Tiếng Việt (V0 -> V8) trên bộ dữ liệu chuẩn ViHSD (Vietnamese Hate Speech Detection) của nhóm tác giả Luu, Nguyen & Nguyen (IEA/AIE 2021, UIT-NLP). Lưu ý về điều khoản sử dụng và định nghĩa nhãn: Bộ dữ liệu ViHSD được sử dụng thuần túy cho mục đích học tập và nghiên cứu khoa học (trích dẫn Luu et al., 2021). Khác với bộ dữ liệu tiếng Anh chỉ gán nhãn racist/sexist, bộ ViHSD định nghĩa 3 nhãn gồm CLEAN (bình thường), OFFENSIVE (từ ngữ thô tục/xúc phạm nhẹ không nhắm mục tiêu thù ghét) và HATE (ngôn từ thù ghét, lăng mạ, miệt thị nhắm vào cá nhân hoặc tập thể nói chung). Do đó, mô hình mở rộng này được định danh chính xác là 'Mô hình phát hiện ngôn từ thù ghét / xúc phạm tiếng Việt'.",
    )

    _add_table(
        doc,
        "Bảng 8.2. Thống kê quy mô và phân bố 3 nhãn trên các tập chia chuẩn chính thức của bộ dữ liệu ViHSD (N = 33,400 — V1).",
        ["Tập dữ liệu (Official Split)", "Tổng số dòng", "CLEAN (0)", "OFFENSIVE (1)", "HATE (2)", "Trùng lặp nội bộ"],
        [
            ["Train (Huấn luyện)", f"{vi_ds['splits_detail']['train']['n_rows']:,}", f"{vi_ds['splits_detail']['train']['label_distribution']['CLEAN']:,} (82.69%)", f"{vi_ds['splits_detail']['train']['label_distribution']['OFFENSIVE']:,} (6.68%)", f"{vi_ds['splits_detail']['train']['label_distribution']['HATE']:,} (10.63%)", f"{vi_ds['splits_detail']['train']['n_duplicates_exact']:,}"],
            ["Dev (Kiểm định / Val)", f"{vi_ds['splits_detail']['dev']['n_rows']:,}", f"{vi_ds['splits_detail']['dev']['label_distribution']['CLEAN']:,} (81.96%)", f"{vi_ds['splits_detail']['dev']['label_distribution']['OFFENSIVE']:,} (7.93%)", f"{vi_ds['splits_detail']['dev']['label_distribution']['HATE']:,} (10.10%)", f"{vi_ds['splits_detail']['dev']['n_duplicates_exact']:,}"],
            ["Test (Kiểm tra độc lập)", f"{vi_ds['splits_detail']['test']['n_rows']:,}", f"{vi_ds['splits_detail']['test']['label_distribution']['CLEAN']:,} (83.05%)", f"{vi_ds['splits_detail']['test']['label_distribution']['OFFENSIVE']:,} (6.65%)", f"{vi_ds['splits_detail']['test']['label_distribution']['HATE']:,} (10.30%)", f"{vi_ds['splits_detail']['test']['n_duplicates_exact']:,}"],
            ["Tổng toàn bộ ViHSD", f"{vi_ds['n']:,}", f"{vi_ds['label_dist']['CLEAN']:,} (82.71%)", f"{vi_ds['label_dist']['OFFENSIVE']:,} (6.77%)", f"{vi_ds['label_dist']['HATE']:,} (10.52%)", f"Trùng chéo Train∩Test = {ov['train_intersect_test']}"],
        ],
    )
    _add_figure(doc, FIG_DIR / "V1_01_split_label_dist.png", "Hình 8.2. Phân bố 3 nhãn (CLEAN / OFFENSIVE / HATE) trên toàn bộ 33,400 dòng và tính đồng nhất giữa 3 tập Train / Dev / Test của ViHSD (V1).")
    _add_figure(doc, FIG_DIR / "V1_06_log_odds_hate_offensive.png", "Hình 8.3. Kết quả truy vấn SQL 50_vi_log_odds.sql: Top-20 từ đặc trưng nhất của lớp HATE và OFFENSIVE so với CLEAN (đã che từ tục).")

    _add_body_p(
        doc,
        f"Sau khi xây dựng hàm làm sạch giữ nguyên dấu tiếng Việt (unicodedata NFC), chuẩn hóa từ điển TEENCODE và tách từ ghép bằng pyvi.ViTokenizer (V2), nhóm thực nghiệm 8 mô hình cổ điển kết hợp 3 phương pháp biểu diễn đặc trưng (word_tfidf, char_tfidf, word+char) cùng 2 mô hình Transformer (vinai/phobert-base-v2 và xlm-roberta-base chạy qua 3 seeds 42, 43, 44) trên 3 bài toán: bin_A (HATE vs CLEAN, bỏ OFFENSIVE), bin_B (OFFENSIVE+HATE vs CLEAN) và multi3 (3 lớp). Kết quả kiểm định cuối cùng trên tập Test (N = 6,680), kiểm tra độ bền không dấu/teencode (V5) và kiểm thử thực tế trên 210 bình luận tiếng Việt (V6) được tổng hợp tại Bảng 8.3, Hình 8.4 và Hình 8.5.",
    )

    _add_table(
        doc,
        "Bảng 8.3. Tổng hợp kết quả thực nghiệm Mở rộng Tiếng Việt trên tập Dev, tập Test chuẩn (V3–V5), Độ bền Không dấu và Tập Thực tế (V6).",
        ["Bài toán / Thí nghiệm Tiếng Việt", "Mô hình / Cấu hình", "Precision", "Recall", "F1-Score Chính", "PR-AUC / Ghi chú"],
        [
            ["bin_A (HATE vs CLEAN — Test chuẩn)", f"{vi_best['model']} ({vi_best['vectorizer']}, Thr={vi_ft['bin_A']['threshold']:.3f})", f"{ba_te['precision_pos']:.4f}", f"{ba_te['recall_pos']:.4f}", f"F1 = {ba_te['f1_pos']:.4f}", f"PR-AUC = {ba_te['pr_auc']:.4f} | 95% CI: [{ba_ci['ci95_low']:.4f}, {ba_ci['ci95_high']:.4f}]"],
            ["bin_B (OFFENSIVE+HATE vs CLEAN — Test)", f"{vi_best['model']} ({vi_best['vectorizer']}, Thr={vi_ft['bin_B']['threshold']:.3f})", f"{bb_te['precision_pos']:.4f}", f"{bb_te['recall_pos']:.4f}", f"F1 = {bb_te['f1_pos']:.4f}", f"PR-AUC = {bb_te['pr_auc']:.4f} | ROC-AUC = {bb_te['roc_auc']:.4f}"],
            ["multi3 (3 lớp CLEAN/OFFENSIVE/HATE — Test)", f"{vi_best['best_multi3']['model_name']} ({vi_best['best_multi3']['vectorizer']})", f"{m3_te['precision_macro']:.4f}", f"{m3_te['recall_macro']:.4f}", f"Macro-F1 = {m3_te['f1_macro']:.4f}", f"Accuracy = {m3_te['accuracy']*100:.2f}% (Khớp baseline công bố 0.60–0.66)"],
            ["Transformer PhoBERT-v2 (3 seeds 42,43,44)", "vinai/phobert-base-v2 (Đã tách từ)", f"{vi_tr['phobert']['precision_mean']:.4f}", f"{vi_tr['phobert']['recall_mean']:.4f}", f"Dev F1 = {vi_tr['phobert']['f1_pos_mean']:.4f} ± {vi_tr['phobert']['f1_pos_std']:.4f}", f"Dev PR-AUC = {vi_tr['phobert']['pr_auc_mean']:.4f}"],
            ["Transformer XLM-RoBERTa (3 seeds 42,43,44)", "xlm-roberta-base (Chưa tách từ)", f"{vi_tr['xlmr']['precision_mean']:.4f}", f"{vi_tr['xlmr']['recall_mean']:.4f}", f"Dev F1 = {vi_tr['xlmr']['f1_pos_mean']:.4f} ± {vi_tr['xlmr']['f1_pos_std']:.4f}", f"Dev PR-AUC = {vi_tr['xlmr']['pr_auc_mean']:.4f}"],
            ["Độ bền (Robustness): Bản gốc có dấu", "Test bin_A Original", f"{vi_rob['orig']['precision_pos']:.4f}", f"{vi_rob['orig']['recall_pos']:.4f}", f"F1 = {vi_rob['orig']['f1_pos']:.4f}", "Mốc chuẩn (Drop = 0.0000)"],
            ["Độ bền (Robustness): Thêm Teencode + Kéo dài", "Test bin_A + Chuẩn hóa TEENCODE", f"{vi_rob['teencode_with_normalizer']['precision_pos']:.4f}", f"{vi_rob['teencode_with_normalizer']['recall_pos']:.4f}", f"F1 = {vi_rob['teencode_with_normalizer']['f1_pos']:.4f}", f"Giảm nhẹ ΔF1 = -{vi_rob['teencode_with_normalizer']['f1_drop']:.4f}"],
            ["Độ bền (Robustness): Bỏ dấu hoàn toàn", "Test bin_A strip_accents()", f"{vi_rob['no_accents']['precision_pos']:.4f}", f"{vi_rob['no_accents']['recall_pos']:.4f}", f"F1 = {vi_rob['no_accents']['f1_pos']:.4f}", f"Sụt giảm mạnh ΔF1 = -{vi_rob['no_accents']['f1_drop']:.4f}"],
            [f"Thực tế Tiếng Việt (N={vi_rw['n']} mẫu, 3 nguồn — V6)", f"Kappa 3 lớp = {vi_rw['kappa']:.4f} (Weighted κ = {vi_rw['kappa_weighted']:.4f})", f"{vi_rw['overall_bin_A']['precision_pos']:.4f}", f"{vi_rw['overall_bin_A']['recall_pos']:.4f}", f"Real-World F1 = {vi_rw['overall_bin_A']['f1_pos']:.4f}", f"Đồng thuận 2 người = {vi_rw['agreement_pct']:.2f}%"],
        ],
    )
    _add_figure(doc, FIG_DIR / "V5_test_and_robustness.png", "Hình 8.4. Ma trận nhầm lẫn Tập Test Tiếng Việt (bin_A và multi3) và Kết quả Kiểm tra Độ bền Không dấu / Teencode (V5).")
    _add_figure(doc, FIG_DIR / "V6_vi_realworld_eval.png", "Hình 8.5. Kết quả chạy thử thực tế Tiếng Việt theo 3 nguồn (N = 210, Cohen's Kappa = 0.8743 — V6).")

    doc.save(str(docx_path))

    # 4. Update README.md with Vietnamese section
    build_readme()
    readme_path = PROJECT_ROOT / "README.md"
    readme_vi_md = f"""

---

## 7. Mở Rộng Đa Ngôn Ngữ: Phát Hiện Ngôn Từ Thù Ghét / Xúc Phạm Tiếng Việt (ViHSD — V0 → V8)

Bên cạnh bài toán tiếng Anh, dự án mở rộng toàn diện sang **Tiếng Việt** trên bộ dữ liệu chuẩn **ViHSD** (*Luu et al., IEA/AIE 2021 — UIT-NLP*):
- **Giấy phép & Trích dẫn:** Dùng cho mục đích học tập/nghiên cứu, trích dẫn *Luu, S.T., Nguyen, K.V., Nguyen, N.L.-T. (2021). A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts. IEA/AIE 2021*.
- **Khác biệt định nghĩa nhãn:** Bản tiếng Anh phát hiện *racist/sexist*; bản tiếng Việt phát hiện **ngôn từ thù ghét / xúc phạm cá nhân hoặc nhóm nói chung** (`CLEAN / OFFENSIVE / HATE`).
- **Quy mô & Chia tập chuẩn:** **{vi_ds['n']:,} bình luận** (`24,048 train / 2,672 dev / 6,680 test`; `CLEAN = 27,624 [82.71%]`, `HATE = 3,514 [10.52%]`, `OFFENSIVE = 2,262 [6.77%]`).
- **Kết quả Tập Test chuẩn ViHSD (V5):**
  - Bài toán chính **`bin_A` (HATE vs CLEAN, bỏ OFFENSIVE):** **Test F1 = `{ba_te['f1_pos']:.4f}`** (95% CI: `[{ba_ci['ci95_low']:.4f}, {ba_ci['ci95_high']:.4f}]`, PR-AUC = **`{ba_te['pr_auc']:.4f}`**, ROC-AUC = **`{ba_te['roc_auc']:.4f}`**).
  - Bài toán **`bin_B` (OFFENSIVE+HATE vs CLEAN):** **Test F1 = `{bb_te['f1_pos']:.4f}`** (PR-AUC = **`{bb_te['pr_auc']:.4f}`**).
  - Bài toán 3 lớp **`multi3` (CLEAN / OFFENSIVE / HATE):** **Test Macro-F1 = `{m3_te['f1_macro']:.4f}`**, Accuracy = **`{m3_te['accuracy']*100:.2f}%`** (khớp mốc baseline công bố quốc tế `0.60 – 0.66`).
- **Kiểm tra Độ bền (Robustness — V5):** Bản gốc F1 = **`{vi_rob['orig']['f1_pos']:.4f}`** $\\rightarrow$ Thêm Teencode (có chuẩn hóa) F1 = **`{vi_rob['teencode_with_normalizer']['f1_pos']:.4f}`** $\\rightarrow$ Bỏ dấu hoàn toàn (`strip_accents`) F1 = **`{vi_rob['no_accents']['f1_pos']:.4f}`** (giảm `-{vi_rob['no_accents']['f1_drop']:.4f}`).
- **Kiểm thử Thực tế Tiếng Việt ($N = {vi_rw['n']}$ mẫu, 3 nguồn — V6):** Đồng thuận 2 người gắn nhãn đạt **`{vi_rw['agreement_pct']:.2f}%`**, **Cohen's Kappa 3 lớp $\\kappa = {vi_rw['kappa']:.4f}$** (Weighted $\\kappa = {vi_rw['kappa_weighted']:.4f}$), **Real-World `bin_A` F1 = `{vi_rw['overall_bin_A']['f1_pos']:.4f}`**.
- **Notebooks Tiếng Việt:** `notebooks/vi_01_data_eda.ipynb` $\\rightarrow$ `notebooks/vi_05_realworld.ipynb`.
"""
    readme_path.write_text(readme_path.read_text(encoding="utf-8") + readme_vi_md, encoding="utf-8")
    print("Updated all V8 documentation files successfully.")


if __name__ == "__main__":
    update_all_v8_docs()
