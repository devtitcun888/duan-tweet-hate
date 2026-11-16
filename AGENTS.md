# AGENT SPEC (bản chi tiết cho AI thực thi): Phân loại tweet phân biệt chủng tộc/giới tính

> **Cách dùng:** đưa file này cho agent (Antigravity) làm `AGENTS.md`/tài liệu gốc. Agent làm tuần tự P0 → P12. Mỗi task có: *Đầu vào → Việc làm → Đầu ra → Nghiệm thu (Acceptance)*. Không sang task sau khi task trước chưa qua nghiệm thu. Phần ⚠️ HUMAN = bắt buộc dừng, chờ người dùng.
> File này thay thế/chi tiết hóa `PLAN_Tweet_Hate_Classification.md` và dùng chung với `BO_TAI_LIEU_BAO_CAO_VA_HOI_DAP.md` (P11).

---

## 0. LUẬT VẬN HÀNH CỦA AGENT (đọc kỹ, áp dụng toàn dự án)

1. **Ngôn ngữ:** code/comment/tên biến tiếng Anh; markdown giải thích, nhận xét biểu đồ, báo cáo bằng **tiếng Việt**.
2. **Không bịa số.** Mọi con số trong notebook/báo cáo phải sinh từ code và lưu vào `reports/*.json|csv`. Chưa có thì ghi `[CHƯA CÓ SỐ]`.
3. **Seed = 42** ở mọi nơi (`random`, `numpy`, `sklearn`, `torch`). Hàm `set_seed()` ở `src/utils.py`.
4. **Chống rò rỉ dữ liệu (quan trọng nhất):**
   - Chia train/val/test **một lần**, lưu thành `data/splits/{train,val,test}.csv` (kèm `id`). Mọi notebook đọc lại các file này, không tự chia lại.
   - Vectorizer, scaler, SMOTE, chọn ngưỡng: **fit chỉ trên train** (dùng `Pipeline`/`imblearn.Pipeline`).
   - Test set chỉ dùng ở P8 và **chỉ một lần** để báo cáo số cuối. Ghi lại thời điểm chạy trong `reports/final_test_metrics.json`.
   - Khử trùng/tweet gần giống (retweet) **trước** khi chia (xem P4).
5. **Sanity check:** nếu F1 lớp dương > 0.90 trên bộ này ⇒ nghi rò rỉ, dừng và điều tra.
6. **Che từ tục** khi hiển thị trong notebook/figure/báo cáo: dùng `mask_profanity(text)` trong `src/utils.py` (giữ ký tự đầu/cuối, thay giữa bằng `*`). Dữ liệu thô không sửa.
7. **Mỗi task xong:** (a) lưu artifact vào đúng thư mục, (b) ghi 3-5 dòng "Nhận xét" vào cuối notebook, (c) `git commit -m "P{n}.{m}: <tóm tắt>"`.
8. **Log thực nghiệm:** mọi lần train ghi 1 dòng vào `reports/experiments.csv` (`timestamp, model, vectorizer, balance, params_json, cv_f1_mean, cv_f1_std, val_f1, val_pr_auc, fit_seconds`).
9. **Lỗi/thiếu dữ liệu:** không im lặng thay thế. Nếu một nguồn dữ liệu tải lỗi, thử đường dự phòng ghi trong P1; hết cách thì dừng và báo người dùng.
10. **Không dùng test set để chọn gì cả** (mô hình, ngưỡng, tham số).
11. **Tương thích Colab:** mọi notebook mở đầu bằng cell `bootstrap` (mục 2.3). Không hardcode đường dẫn máy cá nhân; dùng `Path(PROJECT_DIR)`.
12. **Đạo đức:** không đưa danh tính thật, không crawl dữ liệu vi phạm điều khoản. Tweet phản cảm chỉ dùng cho nghiên cứu.

---

## 1. SỰ THẬT ĐÃ XÁC MINH VỀ DỮ LIỆU & QUYẾT ĐỊNH

### 1.1 Bộ được giao: `cosmos98/twitter-and-reddit-sentimental-analysis-dataset`
- `Twitter_Data.csv`: ~162,973 dòng, 2 cột `clean_text` (str) và `category` (float: -1.0 / 0.0 / 1.0). Nội dung là tweet **sentiment chính trị** (nhiều tweet nhắc Modi). **Không có nhãn racist/sexist.** Có NaN.
- `Reddit_Data.csv`: cột `clean_comment`, `category` (-1/0/1).
- ⇒ **Không dùng làm nhãn huấn luyện cho đề tài.** Dùng để: (a) làm Câu 1 (phân tích, thống kê), (b) tập "thực tế khác miền" để chạy thử (P9).

### 1.2 Bộ huấn luyện có nhãn racist/sexist (Phương án A – **đã chọn**)
- **Tweets Hate Speech Detection** (nguồn gốc: Analytics Vidhya / repo `sharmaroshan/Twitter-Sentiment-Analysis`, bản trên HuggingFace: `tweets-hate-speech-detection/tweets_hate_speech_detection`).
  - Cột `label` (1 = racist/sexist, 0 = không), cột `tweet`.
  - `train`: 31,962 mẫu có nhãn; `test`: 17,197 mẫu (**bản gốc không có nhãn** ⇒ agent phải kiểm tra; nếu không có nhãn thì **không dùng để đánh giá**, chỉ dùng để thử suy luận hàng loạt trong app).
  - Mất cân bằng mạnh, lớp 1 khoảng 7% (agent phải đo thật bằng `value_counts`).
  - Người dùng bị ẩn thành `@user`. Có nhiều tweet trùng/gần trùng (retweet).
  - ⚠️ Lưu ý chất lượng nhãn: nhãn do crowd/mô hình gán, chứa nhiễu ⇒ ghi trong "Hạn chế".
- **Bộ phụ (tùy chọn, điểm cộng, kiểm tra chéo miền):** Davidson et al. *Hate Speech and Offensive Language* (Kaggle/HF `hate_speech_offensive`, cột `tweet`, `class`: 0=hate, 1=offensive, 2=neither; ~24.8k dòng). Cách ánh xạ cho cross-test: `class==0 → 1`, `class==2 → 0`, **bỏ** `class==1` (offensive ≠ racist/sexist). Ghi rõ khác định nghĩa nhãn.
- Phương án dự phòng nếu HF lỗi: tải `train_tweet.csv` từ GitHub `sharmaroshan/Twitter-Sentiment-Analysis` hoặc Kaggle (tìm "Twitter Sentiment Analysis Hatred Speech"). Tên cột có thể là `label,tweet` hoặc `id,label,tweet`. Agent chuẩn hóa về `id,text,label`.

### 1.3 Quyết định ghi vào báo cáo
> "Bộ được giao chỉ có nhãn sentiment chính trị nên được dùng cho phân tích dữ liệu và kiểm thử khác miền; mô hình được huấn luyện trên bộ hate-speech có nhãn racist/sexist 0/1."

---

## 2. CẤU TRÚC REPO & MÔI TRƯỜNG

### 2.1 Cây thư mục
```
tweet-hate/
├─ AGENTS.md                      # = file này
├─ README.md                      # hướng dẫn chạy lại (P12)
├─ requirements.txt
├─ config.yaml                    # đường dẫn, seed, tham số chung
├─ run_all.py                     # chạy lại toàn bộ pipeline không-GPU
├─ data/{raw,interim,processed,splits,real_world}/
├─ db/tweets.db
├─ sql/{00_schema.sql,10_token_freq.sql,20_log_odds.sql,30_cooccur.sql,40_lexicon.sql}
├─ src/
│   ├─ __init__.py
│   ├─ utils.py          # set_seed, mask_profanity, load_config, log_experiment
│   ├─ data_io.py        # load_assigned(), load_hate(), load_davidson()
│   ├─ clean.py          # clean_text(), extract_meta_features()
│   ├─ features.py       # build_vectorizer()
│   ├─ models.py         # MODEL_ZOO dict, build_pipeline()
│   ├─ evaluate.py       # compute_metrics(), plot_*()
│   ├─ threshold.py      # pick_threshold()
│   ├─ predict.py        # load_model(), predict_text()
│   └─ transformer.py    # fine-tune DistilBERT
├─ notebooks/
│   ├─ 00_colab_bootstrap.ipynb
│   ├─ 01_data_collection_eda.ipynb
│   ├─ 02_sql_analysis.ipynb
│   ├─ 03_preprocess_balance.ipynb
│   ├─ 04_model_comparison.ipynb
│   ├─ 05_tuning_threshold.ipynb
│   ├─ 06_transformer_bonus.ipynb
│   ├─ 07_final_eval_error_analysis.ipynb
│   └─ 08_realworld_test.ipynb
├─ app/app.py
├─ models/               # best_model.joblib, threshold.json, (transformer/)
├─ reports/{figures/,tables/,facts.json,experiments.csv,...}
└─ tests/{test_clean.py,test_leakage.py,test_predict.py}
```
