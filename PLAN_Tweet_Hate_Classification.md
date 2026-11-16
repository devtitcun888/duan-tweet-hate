# KẾ HOẠCH CHI TIẾT: Gắn nhãn tweet phân biệt chủng tộc / giới tính (Binary Classification, Text Mining)

> Dùng file này làm "master plan" đưa cho AI agent (Antigravity). Mỗi Giai đoạn có sẵn PROMPT để copy. Làm tuần tự, mỗi giai đoạn xong phải có file đầu ra + kiểm tra (Checkpoint) mới sang giai đoạn sau.

---

## 0. PHÂN TÍCH ĐỀ BÀI (đọc trước khi làm)

### 0.1 Vấn đề lớn nhất: dữ liệu được giao KHÔNG có nhãn "phân biệt chủng tộc/giới tính"
Bộ `cosmos98/twitter-and-reddit-sentimental-analysis-dataset` (file `Twitter_Data.csv`) theo hiểu biết của mình gồm 2 cột `clean_text` và `category` (-1 tiêu cực / 0 trung tính / 1 tích cực), là **sentiment** về chính trị (tweet nói về Modi), ~163k dòng, có NaN. Nhãn -1 (tiêu cực) **không đồng nghĩa** với phân biệt chủng tộc/giới tính.
=> Việc đầu tiên: chạy `df.columns`, `df.category.value_counts()` để xác nhận. Nếu đúng như trên thì phải chọn chiến lược:

| Phương án | Mô tả | Ưu | Nhược |
|---|---|---|---|
| **A (khuyến nghị)** | Dùng bộ được giao cho Câu 1 (phân tích, thống kê, tìm từ phản cảm) + dùng **thêm 1 bộ hate-speech có nhãn racist/sexist** (dạng label 0/1, ví dụ bộ "Twitter Sentiment Analysis – Hatred Speech" hoặc Davidson/Waseem hate speech trên Kaggle/HuggingFace) để huấn luyện | Có nhãn thật, đúng đề, điểm cao | Phải giải thích rõ trong báo cáo |
| B | Tạo nhãn yếu (weak label) cho bộ được giao: tweet có category=-1 **và** chứa từ trong lexicon phản cảm → nhãn 1, còn lại → 0 | Chỉ dùng 1 bộ | Nhãn nhiễu, mô hình chỉ học lại lexicon, kết quả "ảo" |
| C | Kết hợp A + B: dùng bộ hate-speech để train, bộ được giao (gắn nhãn yếu + kiểm tay) làm tập kiểm tra chéo | Mạnh nhất, có phần "thực tế" | Tốn thời gian hơn |

**Chọn A, thêm C nếu còn thời gian.** Ghi rõ lý do trong báo cáo (đây cũng là điểm thể hiện tư duy phản biện với giảng viên). Nên hỏi/báo giảng viên về việc này.

### 0.2 Ánh xạ rubric → công việc
| Câu | Điểm | Việc cần làm | Mục trong plan |
|---|---|---|---|
| 1 | 2.0 | Thu thập + phân tích dữ liệu (1đ); thống kê mô tả (1đ) | GĐ 1, 2 |
| 2 | 3.0 | Làm sạch + import Colab (1đ); chuẩn hóa + cân bằng (0.5đ); xây mô hình, chia tập, tiêu chí đánh giá (1.5đ) | GĐ 3, 4, 5 |
| 3 | 3.0 | So sánh mô hình (1.5đ); tinh chỉnh tham số (0.5đ); chạy thử thực tế (1.5đ) | GĐ 5, 6, 7 |
| 4 | 2.0 | Giao diện ứng dụng (1đ); báo cáo + Q&A (1đ) | GĐ 8, 9 |

### 0.3 Lưu ý đạo đức & kỹ thuật
- Dữ liệu có từ ngữ xúc phạm: chỉ dùng cho nghiên cứu, **che bớt (mask) từ tục** trong slide/báo cáo (vd `n****r`, `b***h`).
- Không public tweet có thông tin cá nhân; ẩn `@user`, URL.
- X (Twitter) API hiện hạn chế/trả phí → bộ "tweet thực tế" nên lấy từ: (1) phần test chưa dùng, (2) tập tự gắn nhãn tay, (3) Reddit/nguồn công khai khác. Ghi rõ nguồn.

---

## 1. CẤU TRÚC DỰ ÁN & MÔI TRƯỜNG

```
project/
├─ data/raw/  data/interim/  data/processed/
├─ db/tweets.db                 # SQLite (phần ghi chú đề: SQL để tìm từ tần suất cao)
├─ sql/                         # các file query
├─ notebooks/
│   ├─ 01_eda.ipynb
│   ├─ 02_preprocess.ipynb
│   ├─ 03_modeling.ipynb
│   ├─ 04_tuning_eval.ipynb
│   └─ 05_realworld_test.ipynb
├─ src/ (clean.py, features.py, train.py, predict.py)
├─ models/ (best_model.joblib, vectorizer.joblib)
├─ app/ (app.py)               # Streamlit hoặc Gradio
├─ reports/ (figures/, report.docx|pdf, slides.pptx)
└─ requirements.txt
```
**Kết nối Google Colab:** mount Drive, đặt `PROJECT_DIR=/content/drive/MyDrive/TweetHate`, mỗi notebook đầu tiên chạy cell setup (mount, `pip install`, set seed=42). Đẩy code `src/` lên GitHub rồi `git clone` trong Colab (hoặc chép vào Drive) để Antigravity và Colab dùng chung.

**PROMPT GĐ 1 (cho agent):**
> Tạo cấu trúc thư mục như trên. Viết `requirements.txt` (pandas, numpy, scikit-learn, imbalanced-learn, nltk, matplotlib, seaborn, wordcloud, xgboost, lightgbm, joblib, kagglehub, streamlit/gradio, transformers, torch, datasets). Tạo notebook `01_eda.ipynb` với cell setup cho Google Colab: mount Drive, cài thư viện, set seed 42, tải dataset bằng `kagglehub.dataset_download("cosmos98/twitter-and-reddit-sentimental-analysis-dataset")`, copy file vào `data/raw/`. In ra đường dẫn, danh sách file.

**Checkpoint:** Notebook chạy được từ đầu đến cuối trên Colab, thấy file `Twitter_Data.csv`.

---

## 2. GĐ 2 – THU THẬP & PHÂN TÍCH DỮ LIỆU (Câu 1: 2đ)

### 2.1 Thu thập (1đ)
1. Tải bộ được giao (kagglehub, đã có code trong đề).
2. Tải bộ hate-speech có nhãn racist/sexist (phương án A) → `data/raw/hate_*.csv`. Ghi nguồn, giấy phép, số dòng, tên cột.
3. (Tùy chọn) Reddit_Data.csv trong cùng dataset làm nguồn "thực tế" thứ hai.
4. Báo cáo mô tả: nguồn, kích thước, cột, kiểu dữ liệu, cách thu thập.

### 2.2 Thống kê mô tả (1đ) – đầu ra là bảng + biểu đồ lưu `reports/figures/`
- `shape`, `info`, số NaN, số dòng trùng, phân bố nhãn (bar + pie), độ mất cân bằng (tỉ lệ lớp).
- Độ dài tweet (ký tự, từ): histogram, boxplot **theo nhãn**.
- Top-N từ / bigram theo nhãn (bar chart), WordCloud theo nhãn.
- Tỉ lệ có URL, @mention, hashtag, emoji, chữ HOA, dấu `!`/`?` theo nhãn.
- So sánh: từ phân biệt nhất giữa 2 lớp (log-odds ratio hoặc chi2).

### 2.3 Tìm "điểm chung của tweet phản cảm" (Yêu cầu 1 của đề)
- Dùng **SQL (SQLite)**: import 2 bảng `tweets_assigned`, `tweets_hate`; viết query:
  - Tần suất từ (tách từ bằng Python rồi insert bảng `word_counts(word,label,cnt)`).
  - Top 50 từ ở nhãn 1 nhưng hiếm ở nhãn 0 (tỉ lệ cnt1/cnt0).
  - Co-occurrence: từ nào hay đi cùng từ bị gắn nhãn phản cảm.
  - Tweet chứa từ trong lexicon (danh sách từ xúc phạm công khai, vd. lexicon Hatebase/ danh sách bad-words tiếng Anh) theo nhãn.
- Kết luận dạng văn bản: "tweet phản cảm thường ngắn hơn/dài hơn, nhiều @mention, chứa nhóm từ X, Y, nhắm vào nhóm đối tượng Z…".

**PROMPT GĐ 2:**
> Trong `01_eda.ipynb`: load hai bộ dữ liệu, kiểm tra cột và NaN, in cảnh báo nếu `Twitter_Data.csv` chỉ có nhãn sentiment (-1/0/1) và không có nhãn racist/sexist. Làm đủ thống kê mô tả ở mục 2.2, mỗi biểu đồ có tiêu đề + nhận xét 2-3 câu bằng tiếng Việt, lưu PNG. Sau đó import vào SQLite `db/tweets.db`, viết file `sql/word_freq.sql` với các query ở mục 2.3, chạy bằng `pd.read_sql`, hiển thị top từ. Tổng hợp 5-8 phát hiện chính vào cell markdown cuối. Che bớt từ tục khi hiển thị (mask ký tự giữa).

**Checkpoint:** Có ≥10 biểu đồ, file `.db`, các query SQL chạy được, đoạn "điểm chung tweet phản cảm".

---

## 3. GĐ 3 – LÀM SẠCH & CHUẨN HÓA (Câu 2: 1đ + 0.5đ)

Pipeline `src/clean.py` (viết thành hàm, dùng lại khi chạy app):
1. Bỏ NaN, trùng; ép kiểu nhãn int.
2. Lowercase; xóa URL, `@user`, `RT`, HTML entity; giữ nội dung hashtag (bỏ dấu `#`).
3. Chuẩn hóa kéo dài chữ ("sooooo"→"soo"), giữ lại cờ `has_exclaim`, `all_caps_ratio` làm đặc trưng phụ.
4. Bỏ ký tự đặc biệt/số; xử lý emoji (demojize hoặc bỏ); tokenize.
5. Stopwords: **cẩn thận** — giữ các từ phủ định/đại từ (not, you, they, women, men…) vì quan trọng với hate-speech. So sánh có/không bỏ stopword.
6. Lemmatize (WordNet). Lưu `data/processed/clean.csv` với cột `text_raw`, `text_clean`, `label`.

**Chuẩn hóa/biểu diễn đặc trưng:** TF-IDF (unigram+bigram, `min_df=3`, `max_features` 20-50k), thử thêm char n-gram; (nâng cao) Word2Vec/GloVe trung bình, và embedding của BERT.

**Cân bằng dữ liệu (0.5đ):** hiển thị tỉ lệ lớp. Hate-speech thường lệch (~7% lớp 1). Thử và so sánh: `class_weight='balanced'`, RandomOverSampler, SMOTE (chỉ trên TF-IDF/ embedding, **chỉ áp dụng lên tập train**), RandomUnderSampler. Chọn bằng F1 trên validation.

**PROMPT GĐ 3:**
> Viết `src/clean.py` với hàm `clean_text(text, remove_stopwords=False)` theo mục 3, viết unit test nhỏ trên 10 câu mẫu. Chạy trên cả hai bộ dữ liệu, lưu `data/processed/`. Trong `02_preprocess.ipynb` hiển thị 10 ví dụ trước/sau làm sạch, thống kê độ dài trước/sau, và so sánh các cách cân bằng dữ liệu ở trên bằng 1 mô hình Logistic Regression cơ sở. TUYỆT ĐỐI không để rò rỉ dữ liệu (fit TF-IDF và SMOTE chỉ trên train).

**Checkpoint:** `clean.csv` sạch, bảng so sánh kỹ thuật cân bằng, không data leakage.

---

## 4. GĐ 4 – THIẾT KẾ THỰC NGHIỆM (Câu 2: 1.5đ)

- **Chia tập:** train/val/test = 70/15/15 (hoặc 80/20 + CV 5-fold) với `stratify=y`, `random_state=42`. Test set giữ nguyên, chỉ dùng 1 lần cuối.
- **Tiêu chí đánh giá** (do lệch lớp nên *không* dùng mỗi accuracy): Precision, **Recall (lớp phản cảm)**, **F1 (lớp 1 + macro)**, ROC-AUC, **PR-AUC**, confusion matrix, thời gian train/predict. Chọn **F1 lớp 1** là tiêu chí chính; giải thích vì sao.
- Dùng `Pipeline` (TfidfVectorizer + model) để tránh rò rỉ; `StratifiedKFold`.

---

## 5. GĐ 5 – XÂY DỰNG & SO SÁNH MÔ HÌNH (Câu 3: 1.5đ)

Mô hình "đã học" (baseline bắt buộc):
1. Logistic Regression
2. Multinomial / Complement Naive Bayes
3. Linear SVM (LinearSVC / SVC)
4. Decision Tree, Random Forest
5. KNN (có thể chậm, dùng mẫu nhỏ)
6. Gradient Boosting: XGBoost / LightGBM
**Điểm cộng (tìm hiểu ngoài):**
7. Mô hình học sâu: BiLSTM/GRU với embedding.
8. **DistilBERT/BERT/RoBERTa fine-tune** (chạy GPU Colab T4, 2-3 epoch, max_len 64-128), hoặc mô hình HateBERT / `cardiffnlp/twitter-roberta-base-hate` làm tham chiếu.

Đầu ra: bảng so sánh (model × Precision/Recall/F1/AUC/thời gian), biểu đồ cột, đường ROC & PR chồng nhau, confusion matrix từng mô hình, kết luận mô hình tốt nhất và vì sao (xét cả tốc độ để chạy app).

**PROMPT GĐ 5:**
> Trong `03_modeling.ipynb`, dùng Pipeline cho 6 mô hình cổ điển ở mục 5, huấn luyện trên train, đánh giá trên validation với các metric ở mục 4, lưu bảng kết quả `reports/model_comparison.csv` và biểu đồ ROC/PR. Thêm ô riêng cho DistilBERT (HuggingFace Trainer, bật GPU, fp16) như mô hình điểm cộng. Mỗi ô có nhận xét tiếng Việt. Không chạm vào test set.

---

## 6. GĐ 6 – TINH CHỈNH THAM SỐ (Câu 3: 0.5đ)

- `GridSearchCV`/`RandomizedSearchCV`/Optuna cho 2-3 mô hình tốt nhất, `scoring='f1'`, `cv=5`.
 - LR: `C`, `penalty`, `class_weight`; SVM: `C`, `loss`; NB: `alpha`; RF/XGB: `n_estimators`, `max_depth`, `learning_rate`; TF-IDF: `ngram_range`, `min_df`, `max_features`.
- **Tinh chỉnh ngưỡng quyết định** (threshold) theo đường PR để tăng recall.
- Báo cáo trước/sau tinh chỉnh; đánh giá cuối trên **test set**; phân tích lỗi: 20 dự đoán sai (FP/FN) và nguyên nhân (mỉa mai, từ không thấy, ngữ cảnh).
- Lưu: `joblib.dump(best_pipeline, "models/best_model.joblib")`.

---

## 7. GĐ 7 – CHẠY THỬ THỰC TẾ (Câu 3: 1.5đ)

Xây bộ tweet thực tế (đề cho phép dùng phần test hoặc thu thập):
1. Lấy ~200-500 tweet/bình luận: (a) phần dữ liệu test chưa dùng; (b) mẫu từ `Twitter_Data.csv`/`Reddit_Data.csv` (bộ khác miền dữ liệu); (c) tự viết/thu thập vài chục câu hiểm hóc (mỉa mai, từ lóng, phủ định).
2. **Gắn nhãn tay** bởi 2 người, tính Cohen's kappa, thống nhất nhãn. Che thông tin cá nhân.
3. Chạy mô hình, báo cáo metric + bảng ví dụ đúng/sai + nhận xét về việc mô hình "đổi miền" (domain shift) làm giảm hiệu năng ra sao.
4. Đề xuất cải tiến (thêm dữ liệu, BERT, xử lý mỉa mai).

**PROMPT GĐ 7:**
> Tạo `05_realworld_test.ipynb`: load `models/best_model.joblib`, nạp file `data/real_world_labeled.csv` (cột `text,label`), chạy `clean_text` rồi dự đoán, in classification report, confusion matrix, 15 ví dụ sai kèm xác suất. Viết hàm `predict_proba_text(text)` trong `src/predict.py` dùng lại cho app.

---

## 8. GĐ 8 – ỨNG DỤNG DEMO (Câu 4: 1đ)

- Công cụ: **Streamlit** (hoặc Gradio, chạy được ngay trong Colab; chia sẻ qua cloudflared/ngrok hoặc Hugging Face Spaces).
- Chức năng: (1) ô nhập tweet → nhãn + xác suất + thanh mức độ tin cậy; (2) làm nổi bật từ đóng góp nhiều (hệ số LR/LIME); (3) tải lên CSV → gắn nhãn hàng loạt, tải kết quả; (4) tab thống kê: biểu đồ phân bố & top từ; (5) cảnh báo "mô hình có thể sai, chỉ hỗ trợ".
- Giao diện sạch, có ví dụ mẫu (đã che từ tục).

**PROMPT GĐ 8:**
> Viết `app/app.py` bằng Streamlit, load `models/best_model.joblib`, có 3 tab: "Phân loại 1 tweet" (kèm xác suất + từ đóng góp), "Phân loại file CSV", "Thống kê". Bố cục rõ ràng, tiếng Việt. Viết hướng dẫn chạy trong Colab (cloudflared) và chạy local `streamlit run app/app.py`. Chụp ảnh màn hình để đưa vào báo cáo.

---

## 9. GĐ 9 – BÁO CÁO & TRÌNH BÀY (Câu 4: 1đ)

**Cấu trúc báo cáo:** 1) Giới thiệu & mục tiêu; 2) Dữ liệu (nguồn, vấn đề nhãn, EDA); 3) Tiền xử lý & cân bằng; 4) Mô hình & tiêu chí; 5) Kết quả so sánh + tinh chỉnh; 6) Chạy thử thực tế & phân tích lỗi; 7) Ứng dụng demo; 8) Kết luận, hạn chế, hướng phát triển; Phụ lục: link Colab, GitHub.
**Slide (10-14 trang):** vấn đề → dữ liệu → điểm chung tweet phản cảm → pipeline → bảng so sánh → mô hình chọn → thử thực tế → demo app → hạn chế.
**Link Colab:** chia sẻ "Anyone with the link – Viewer", chạy `Runtime → Run all` thử trước khi nộp; ghi sẵn cell tải dữ liệu để người chấm tự chạy được.

**Chuẩn bị câu hỏi bảo vệ:** Vì sao dùng F1 mà không dùng accuracy? Vì sao chọn mô hình này? SMOTE ở đâu và tại sao chỉ trên train? Tại sao bộ được giao không có nhãn phân biệt? Mô hình sai ở loại tweet nào? Hạn chế đạo đức/thiên lệch (bias) của dữ liệu?

---

## 10. LỊCH GỢI Ý (≈ 2 tuần)

| Ngày | Việc |
|---|---|
| 1 | GĐ 1-2 (setup, tải dữ liệu, xác nhận vấn đề nhãn) |
| 2-3 | GĐ 2 (EDA, SQL, điểm chung) |
| 4 | GĐ 3 (làm sạch, cân bằng) |
| 5-6 | GĐ 4-5 (mô hình, so sánh) |
| 7 | GĐ 6 (tinh chỉnh) + BERT (điểm cộng) |
| 8-9 | GĐ 7 (tạo bộ thực tế, gắn nhãn, chạy thử) |
| 10-11 | GĐ 8 (app) |
| 12-13 | GĐ 9 (báo cáo, slide, tập trả lời) |
| 14 | Chạy lại toàn bộ trên Colab, kiểm tra link, nộp |

## 11. CHECKLIST CUỐI
- [ ] Đã xác nhận cột/nhãn dataset và ghi quyết định (A/B/C) vào báo cáo
- [ ] Không rò rỉ dữ liệu; test set chỉ dùng cuối
- [ ] Đủ metric (P/R/F1/AUC/PR-AUC), có ≥6 mô hình + 1 mô hình ngoài (điểm cộng)
- [ ] Có SQL + kết luận "điểm chung" tweet phản cảm
- [ ] Bộ thực tế có nhãn tay + kappa
- [ ] App chạy được, có ảnh minh họa
- [ ] Notebook Colab "Run all" thành công, link chia sẻ mở được
