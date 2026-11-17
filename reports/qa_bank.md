# NGÂN HÀNG 60 CÂU HỎI & TRẢ LỜI BẢO VỆ ĐỒ ÁN (Q&A BANK / B4)

> **Cấu trúc chuẩn cho mọi câu hỏi:**  
> 1. **Trả lời ngắn (2–3 câu để nói trực tiếp):** Đi thẳng vào kết luận + con số thật.  
> 2. **Ý mở rộng (nếu bị hỏi xoáy tiếp):** Giải thích cơ chế toán học / kỹ thuật sâu hơn.  
> 3. **Bằng chứng (Slide / Chương / Bảng / Hình):** Chỉ đúng vị trí trong báo cáo và slide.  
> 4. **"Bẫy" thường gặp cần tránh:** Những điểm dễ nói hớ hoặc hiểu sai.

---

## PHẦN D1: VỀ ĐỀ TÀI, BÀI TOÁN & DỮ LIỆU (CÂU 1 – 7)

### Câu 1: Vì sao chọn bài toán phân loại nhị phân? Nhãn 0 và 1 có ý nghĩa gì?
1. **Trả lời ngắn:** Đề tài đặt mục tiêu sàng lọc tự động các đoạn tweet có nội dung phân biệt chủng tộc hoặc phân biệt giới tính, nên bài toán được mô hình hóa thành phân loại nhị phân (*Binary Classification*). Trong đó, **Nhãn `1`** biểu thị tweet phản cảm (có nội dung kỳ thị chủng tộc, sắc tộc, tôn giáo hoặc giới tính) và **Nhãn `0`** biểu thị tweet bình thường (giao tiếp đời thường, tin tức, hoặc bình luận chính trị không mang tính thù ghét nhóm).
2. **Ý mở rộng:** Trong quy trình kiểm duyệt nội dung thực tế trên mạng xã hội (Content Moderation Pipeline), bước đầu tiên luôn là bộ lọc nhị phân tốc độ cao để tách luồng nội dung có dấu hiệu vi phạm (`Nhãn 1`) chuyển cho kiểm duyệt viên hoặc mô hình đa nhãn chuyên sâu xử lý.
3. **Bằng chứng:** Slide 2, Chương 1 (Mục 1.2), và hướng dẫn gắn nhãn `data/real_world/ANNOTATION_GUIDE.md`.
4. **Bẫy cần tránh:** Không nói Nhãn `1` chỉ có phân biệt chủng tộc mà quên phân biệt giới tính, và không đánh đồng Nhãn `1` với mọi câu có cảm xúc tiêu cực (buồn bã, chê sản phẩm).

### Câu 2: Bộ dữ liệu được giao (`Twitter_Data.csv`) có những cột nào? Có nhãn phân biệt chủng tộc/giới tính không?
1. **Trả lời ngắn:** Bộ dữ liệu được giao `cosmos98/twitter-and-reddit-sentimental-analysis-dataset` gồm file `Twitter_Data.csv` (**162,980 dòng**) chỉ có đúng 2 cột là `clean_text` và `category` nhận giá trị `-1.0, 0.0, 1.0`. Qua kiểm tra cột và phân tích từ khóa (top từ là `modi`, `india`, `bjp`, `congress`), nhóm xác nhận đây là bộ dữ liệu **sentiment chính trị Ấn Độ**, hoàn toàn **không có nhãn phân biệt chủng tộc hay giới tính**. Do đó, nhóm áp dụng **hướng tiếp cận kết hợp dữ liệu chuẩn**: dùng bộ *Tweets Hate Speech Detection* (**31,962 dòng**, có nhãn racist/sexist `0/1` chuẩn) để huấn luyện, và dùng bộ `Twitter_Data.csv` cùng `Reddit_Data.csv` (**37,249 dòng**) cho phân tích EDA ở Câu 1 và kiểm thử lệch miền ở Câu 3.
2. **Ý mở rộng:** Nếu ép nhãn `category = -1` của `Twitter_Data.csv` làm nhãn "phản cảm", mô hình sẽ học cách phân loại các câu chỉ trích chính sách kinh tế/bầu cử Ấn Độ thành phân biệt chủng tộc — đó là sai lầm phương pháp luận nghiêm trọng. Thực nghiệm ở bước thực nghiệm 9 cho thấy chỉ một tỉ lệ nhỏ các câu `category = -1` thực sự chứa ngôn từ thù ghét.
3. **Bằng chứng:** Slide 3, Chương 2 (Mục 2.1), `reports/data_report.json`, Hình `P2_09_assigned_twitter_eda.png` và Bảng `P9_assigned_twitter_by_sentiment.csv`.
4. **Bẫy cần tránh:** Tuyệt đối không giấu việc dùng thêm bộ `hate_train.csv` hoặc nói lấp liếm rằng `Twitter_Data.csv` có nhãn racist/sexist. Sự trung thực và tư duy phản biện dữ liệu chính là phần mở rộng lớn nhất.

### Câu 3: Dữ liệu có bao nhiêu dòng, tỉ lệ lớp bao nhiêu? Có NaN hay trùng lặp không, xử lý ra sao?
1. **Trả lời ngắn:** Bộ `hate_train.csv` có **31,962 dòng**, không có NaN, gồm **29,720 mẫu Lớp 0 (92.99%)** và **2,242 mẫu Lớp 1 (7.01%)**, tỉ lệ mất cân bằng là **13.26:1**. Tuy nhiên, sau khi làm sạch văn bản, nhóm phát hiện có **2 dòng rỗng**, **1 nhóm (2 dòng) mâu thuẫn nhãn** và **3,628 dòng trùng lặp gần-giống** (do retweet); sau khi loại bỏ hoàn toàn các dòng này trước khi chia tập, bộ dữ liệu sạch còn **28,330 mẫu duy nhất** (Lớp 1 chiếm **6.52%**). Còn bộ `Twitter_Data.csv` có **162,980 dòng**, chứa **11 giá trị NaN** (ở 11 dòng), sau khi `dropna` còn **162,969 dòng**.
2. **Ý mở rộng:** Việc phát hiện và loại bỏ **3,628 mẫu trùng sau làm sạch** trước khi chia Train/Val/Test là bước sống còn. Nếu để các bản retweet này nằm rải rác ở cả tập Train và tập Test, điểm F1 trên Test sẽ bị thổi phồng giả tạo (data leakage).
3. **Bằng chứng:** Slide 4 & Slide 6, Chương 2 & Chương 3, Bảng `P2_summary_stats.csv`, `reports/p4_summary.json`.
4. **Bẫy cần tránh:** Nhớ phân biệt rõ con số trước khi khử trùng (**31,962 dòng**, Lớp 1 = **7.01%**) và sau khi khử trùng (**28,330 dòng**, Lớp 1 = **6.52%**).

### Câu 4: Nhãn tiêu cực (Negative Sentiment) khác với phân biệt chủng tộc/giới tính (Hate Speech) như thế nào?
1. **Trả lời ngắn:** Cảm xúc tiêu cực (*Negative Sentiment*) phản ánh thái độ buồn bã, thất vọng hoặc phê phán một sự việc, sản phẩm hay chính sách (ví dụ: *"Chính sách thuế mới làm tổn hại tiểu thương"* hoặc *"Hôm nay kẹt xe tệ quá"* — đều thuộc Nhãn `0`). Ngược lại, ngôn từ phân biệt chủng tộc/giới tính (*Racist/Sexist Hate Speech* — Nhãn `1`) là hành vi công kích, miệt thị hoặc hạ thấp nhân phẩm một nhóm người dựa trên đặc điểm bảo vệ như màu da, sắc tộc, tôn giáo hoặc giới tính.
2. **Ý mở rộng:** Nói cách khác, Hate Speech là một tập con rất hẹp và cực đoan của ngôn từ tiêu cực; hơn nữa, có những câu Hate Speech được viết dưới dạng mỉa mai dùng toàn từ tích cực (*"wonderful"*), nên bài toán Sentiment Analysis thông thường không thể thay thế bài toán Hate Speech Detection.
3. **Bằng chứng:** Slide 3 & Slide 11, Chương 2 (Mục 2.4), Hình `P9_realworld_evaluation.png` (Panel 3).
4. **Bẫy cần tránh:** Không dùng từ "tiêu cực" và "phản cảm/kỳ thị" thay thế lẫn nhau khi thuyết trình.

### Câu 5: Bộ dữ liệu huấn luyện có thiên lệch (bias) gì về thời gian, chủ đề hay nhóm đối tượng không?
1. **Trả lời ngắn:** Có 3 thiên lệch chính: (1) Dữ liệu thu thập bằng tiếng Anh trong giai đoạn bầu cử Mỹ 2016 nên xuất hiện dày đặc các từ khóa chính trị thời điểm đó (`trump`, `obama`, `libtard`, `sjw`); (2) Nhãn gốc được gán bán tự động/crowdsourcing dựa trên hashtag nên một số câu chứa từ nhạy cảm trong ngữ cảnh trung tính bị gán nhầm hoặc gây nhiễu; (3) Tỉ lệ báo động giả (FPR) trên các câu bình thường có chứa từ chỉ chủng tộc (`black, white`) và giới tính (`women, girl`) cao hơn mức trung bình toàn tập Test.
2. **Ý mở rộng:** Ở bước P8, nhóm đã đo lường định lượng thiên lệch này bằng bảng `P8_subgroup_bias.csv`: trong khi FPR trung bình trên toàn bộ các mẫu Lớp 0 của tập Test chỉ ở mức thấp, các câu Lớp 0 có chứa từ chỉ chủng tộc hoặc giới tính có FPR cao hơn đáng kể.
3. **Bằng chứng:** Slide 12 & Slide 14, Chương 6 (Mục 6.5) & Chương 8, Bảng `P8_subgroup_bias.csv`, Hình `P8_subgroup_bias.png`.
4. **Bẫy cần tránh:** Đừng khẳng định bộ dữ liệu chuẩn trên Kaggle/HuggingFace là "hoàn hảo 100%"; chỉ ra nhiễu nhãn và bias mới thể hiện hiểu sâu về dữ liệu.

### Câu 6: Các đoạn tweet phản cảm có điểm chung gì? Bằng chứng định lượng ở đâu?
1. **Trả lời ngắn:** Qua phân tích thống kê và truy vấn SQL trên tập Train (**19,830 mẫu**), tweet phản cảm có 4 điểm chung lớn: (1) Chứa từ trong từ điển xúc phạm (Lexicon) với tỉ lệ **20.59%**, cao gấp **8.6 lần** Lớp 0 (**2.40%**); (2) Tần suất gắn thẻ `@user` trực diện chiếm **65.03%** (so với **46.46%** ở Lớp 0, $p < 10^{-10}$) và dùng câu hỏi tu từ `?` chiếm **18.78%** (so với **10.42%**); (3) Từ vựng có Log-Odds cao nhất xoay quanh 3 cụm: chủng tộc (`racist`, `white`, `black`), giới tính (`sexist`, `woman`, `b***h`, `misogyny`), và tôn giáo/chính trị (`allahsoil`, `islam`, `libtard`); (4) Tuy nhiên **79.41%** tweet phản cảm không chứa từ tục trực tiếp trong Lexicon mà thể hiện qua tổ hợp cụm từ (bigram/trigram).
2. **Ý mở rộng:** Phân tích đồng xuất hiện (`30_cooccur.sql`) còn cho thấy các từ miệt thị thường đi kèm ngay cạnh các danh từ chỉ tập thể (`people`, `women`, `white`, `black`), tạo thành cấu trúc quy chụp đặc trưng.
3. **Bằng chứng:** Slide 4 & Slide 5, Chương 2 (Mục 2.2 & 2.3), `reports/p3_sql_summary.json`, Bảng `P3_top_hate_tokens.csv`, Hình `P3_sql_analysis.png`.
4. **Bẫy cần tránh:** Không nói "mọi tweet phản cảm đều chứa từ tục", vì dữ liệu SQL chứng minh chỉ 20.59% chứa từ trong Lexicon.

### Câu 7: Vì sao nhóm sử dụng SQL (SQLite)? Truy vấn SQL nào quan trọng nhất?
1. **Trả lời ngắn:** Sử dụng SQL (`db/tweets.db`) đáp ứng trực tiếp gợi ý trong đề bài, đồng thời cho phép lưu trữ có cấu trúc hơn **165,408 token** và **145,591 bigram** kèm chỉ mục (`CREATE INDEX ix_tokens_tok`), giúp thực hiện các phép kết nối (`JOIN`), gom nhóm (`GROUP BY`) và đối chiếu từ điển (`LEFT JOIN lexicon`) chỉ trong vài phần trăm giây. Truy vấn quan trọng nhất là **`20_log_odds.sql`** (tính tỉ số Log-Odds có làm mượt cộng 1 giữa 2 lớp) kết hợp với **`40_lexicon.sql`**.
2. **Ý mở rộng:** Nếu chỉ đếm tần suất thô (`10_token_freq.sql`), các từ chung chung như `day, time, people` sẽ lấn át. Truy vấn `20_log_odds.sql` dùng công thức $\ln\frac{c_1+1}{t_1+1} - \ln\frac{c_0+1}{t_0+1}$ với điều kiện `HAVING (c1+c0) >= 10` giúp lọc chính xác những từ xuất hiện đậm đặc ở Lớp 1 nhưng hiếm ở Lớp 0.
3. **Bằng chứng:** Slide 5, Chương 2 (Mục 2.3), thư mục `sql/` (`00_schema.sql` → `40_lexicon.sql`), Notebook `02_sql_analysis.ipynb`.
4. **Bẫy cần tránh:** Nhấn mạnh rằng cơ sở dữ liệu SQL dùng để rút đặc trưng chỉ nạp **tập Train (`19,830` dòng)** chứ không nạp tập Test.

---

## PHẦN D2: TIỀN XỬ LÝ VĂN BẢN & TRÍCH XUẤT ĐẶC TRƯNG (CÂU 8 – 13)

### Câu 8: Nhóm đã làm sạch những gì trong `clean_text`? Vì sao giữ lại hoặc bỏ stopwords?
1. **Trả lời ngắn:** Hàm `clean_text` thực hiện tuần tự 12 bước: giải mã HTML (`&amp;` → `&`), chuyển chữ thường, xóa URL, xóa `@mention` (kể cả `@user`), xóa nhãn `RT`, tách nội dung hashtag (`#tag` → `tag`), chuyển đổi emoji (`demojize`), rút gọn chữ cái lặp $\ge 3$ về 2 (`sooooo` → `soo`), lọc ký tự đặc biệt (giữ dấu `'` nội từ), bỏ từ 1 ký tự (trừ `i, a`), xử lý stopwords có chọn lọc và đưa từ về nguyên mẫu (`WordNetLemmatizer`). Trong pipeline mô hình chính, nhóm giữ nguyên các từ chức năng (`remove_stopwords=False` cho TF-IDF n-gram) và ngay cả khi bật lọc stopwords cho EDA/SQL, nhóm vẫn giữ lại toàn bộ từ phủ định và đại từ.
2. **Ý mở rộng:** Khi dùng TF-IDF với `max_df=0.9` và `sublinear_tf=True` kèm bigram/trigram `(1,3)`, các hư từ xuất hiện ở >90% văn bản tự động bị loại hoặc giảm trọng số IDF, trong khi việc giữ lại cấu trúc câu giúp tạo ra các cụm n-gram có nghĩa như `not racist`, `against women`, `you are`.
3. **Bằng chứng:** Slide 6, Chương 3 (Mục 3.1), `src/clean.py`, `tests/test_clean.py` (13 unit tests pass).
4. **Bẫy cần tránh:** Không dùng `stopwords.words('english')` mặc định của NLTK một cách máy móc mà không kiểm tra những từ nào bị xóa.

### Câu 9: Vì sao bắt buộc phải giữ lại các từ phủ định (`no, not, never`) và đại từ/nhóm người (`you, she, they, women, men`)?
1. **Trả lời ngắn:** Danh sách stopwords mặc định của NLTK chứa cả `no, not, nor, you, your, he, she, they, we, i`. Nếu xóa các từ này, câu phản đối kỳ thị *"I am NOT racist"* sẽ bị biến thành *"racist"* (đảo ngược hoàn toàn nhãn từ `0` sang `1`), còn câu công kích trực diện *"You and they are..."* mất đi chủ ngữ/tân ngữ chỉ nhóm bị tấn công (`they/you/women`).
2. **Ý mở rộng:** Trong phân tích ngôn ngữ học về ngôn từ thù ghét (Us vs. Them discourse), sự đối lập giữa đại từ ngôi thứ nhất (`we, us, our`) và ngôi thứ hai/ba (`you, they, them`) cùng danh từ chỉ giới (`women, men`) là tín hiệu cú pháp quan trọng bậc nhất.
3. **Bằng chứng:** Slide 6, Chương 3 (Mục 3.1), hằng số `PRESERVED_WORDS` trong `src/clean.py` và unit test `test_preserve_negation_and_pronouns_with_stopwords_removed`.
4. **Bẫy cần tránh:** Nếu giảng viên hỏi "NLTK có xóa chữ `not` không?", phải trả lời ngay: "Có, mặc định NLTK xóa chữ `not`, nên nhóm đã chủ động loại `not` khỏi tập stopwords."

### Câu 10: TF-IDF là gì, khác Bag-of-Words ở đâu? Vì sao dùng bigram / trigram?
1. **Trả lời ngắn:** Bag-of-Words (BoW) chỉ đếm tần suất xuất hiện thô của từ trong câu, khiến các từ phổ biến vô nghĩa chiếm trọng số lớn. **TF-IDF** (*Term Frequency – Inverse Document Frequency*) nhân tần suất từ trong câu (đã lấy log `sublinear_tf = 1 + log(tf)`) với độ hiếm của từ đó trên toàn tập văn bản ($\text{IDF}(t) = \ln\frac{1+N}{1+\text{df}(t)} + 1$), giúp làm nổi bật các từ mang tính phân biệt cao. Nhóm dùng thêm bigram và trigram (`ngram_range=(1,3)` ở mô hình chốt) để bắt được cụm từ ghép mang ngữ cảnh như `"white supremacy"`, `"not racist"`, `"against women"`.
2. **Ý mở rộng:** Nếu chỉ dùng unigram `(1,1)`, hai câu *"I hate racist people"* và *"Racist people hate me"* hoặc *"Black women achieved success"* và *"Black women are inferior"* có các từ đơn lẻ gần giống nhau; bigram/trigram giữ được trật tự từ cục bộ.
3. **Bằng chứng:** Slide 8 & Slide 10, Chương 3 (Mục 3.3), `src/features.py`, `models/model_card.json`.
4. **Bẫy cần tránh:** Nhớ nêu rõ `sublinear_tf=True` giúp chống lại việc một từ bị lặp lại 10 lần trong cùng 1 tweet làm lệch vector đặc trưng.

### Câu 11: Vì sao chọn `max_features = 50,000`, `min_df = 2` (hoặc `3`), `max_df = 0.9` như vậy?
1. **Trả lời ngắn:** Các giá trị này được xác định thông qua tìm kiếm siêu tham số `RandomizedSearchCV` 5-fold CV trên tập Train ở giai đoạn 6 (quét `min_df` $\in \{2, 3, 5\}$, `max_features` $\in \{20k, 50k, 100k\}$, `ngram_range` $\in \{(1,1), (1,2), (1,3)\}$). Ngưỡng `min_df = 2` giúp loại bỏ các từ gõ sai chỉ xuất hiện 1 lần duy nhất (giảm nhiễu và giảm kích thước từ điển), trong khi vẫn giữ được các từ lóng/hashtag hiếm xuất hiện từ 2 lần trở lên; còn `max_features = 50,000` đủ bao phủ các unigram/bigram/trigram quan trọng nhất mà không gây tràn bộ nhớ.
2. **Ý mở rộng:** Tham số `max_df = 0.9` loại bỏ các từ xuất hiện ở hơn 90% số tweet (nếu có), đóng vai trò như một bộ lọc stopword thống kê tự động theo miền dữ liệu.
3. **Bằng chứng:** Slide 10, Chương 5 (Mục 5.2), Bảng `P6_before_after.csv` và `models/model_card.json`.
4. **Bẫy cần tránh:** Không nói "em chọn đại 50,000"; phải chỉ vào lưới tìm kiếm `RandomizedSearchCV` ở bước thực nghiệm 6.

### Câu 12: Lemmatization khác Stemming như thế nào? Vì sao nhóm chọn Lemmatization?
1. **Trả lời ngắn:** **Stemming** (như Porter Stemmer) cắt cụt hậu tố theo luật cơ học nên thường tạo ra các gốc từ không có trong từ điển và làm gộp nhầm từ khác nghĩa (ví dụ `"women"` giữ nguyên nhưng `"university"` và `"universe"` đều bị cắt thành `"univers"`). Ngược lại, **Lemmatization** (dùng từ điển hình thái học `WordNetLemmatizer`) đưa từ về đúng dạng từ điển chuẩn (*lemma*: `"women" → "woman"`, `"bitches" → "bitch"`, `"running" → "run"`).
2. **Ý mở rộng:** Vì ứng dụng Streamlit ở bước thực nghiệm 10 cần hiển thị trực tiếp **Top-8 từ đóng góp** cho người dùng cuối xem, việc dùng Lemmatization giúp các từ hiển thị trên giao diện và trong bảng SQL (`P3_top_hate_tokens.csv`) hoàn toàn đọc hiểu được, không bị biến dạng như Stemming.
3. **Bằng chứng:** Chương 3 (Mục 3.1), `src/clean.py` (bước 12), `tests/test_clean.py::test_lemmatization`.
4. **Bẫy cần tránh:** Lưu ý rằng trong `clean_text`, nhóm gọi `lemmatize(tok, pos='n')` rồi đến `pos='v'` để xử lý được cả danh từ số nhiều lẫn động từ chia thì.

### Câu 13: Emoji, Hashtag và `@mention` được xử lý ra sao và tại sao?
1. **Trả lời ngắn:** Với **`@mention`** (`@user`), nhóm xóa khỏi văn bản đưa vào TF-IDF vì trong bộ dữ liệu mọi tên tài khoản đều đã bị ẩn danh thành chuỗi `@user` giống hệt nhau, nếu giữ lại sẽ tạo nhiễu tần suất cao nhưng không mang nghĩa phân biệt cá nhân (nhưng vẫn được trích xuất thành biến đếm `n_mentions` ở bước thực nghiệm 2). Với **`#hashtag`**, nhóm **chỉ bỏ ký tự `#` và giữ nguyên từ bên trong** (`#BlackLivesMatter` → `blacklivesmatter`, `#allahsoil` → `allahsoil`) vì hashtag trên Twitter thường chứa từ khóa chủ đề cốt lõi của câu. Với **Emoji**, nhóm dùng thư viện `emoji.demojize` để chuyển biểu tượng thành từ tiếng Anh mô tả cảm xúc (`😡` → `enraged face`) trước khi tách từ.
2. **Ý mở rộng:** Phân tích EDA ở bước thực nghiệm 2 (Hình `P2_08_top_hashtags.png`) chứng minh rất nhiều tweet phản cảm đặt từ khóa kỳ thị nằm ngay trong hashtag ở cuối câu; nếu xóa cả cụm `#hashtag` như xóa `@user` thì mô hình sẽ mất đi tín hiệu quan trọng nhất.
3. **Bằng chứng:** Slide 4 & Slide 6, Chương 3 (Mục 3.1), `src/clean.py`, `tests/test_clean.py`.
4. **Bẫy cần tránh:** Đừng nhầm lẫn giữa việc xóa toàn bộ `@user` và việc giữ lại nội dung chữ của `#hashtag`.

---

## PHẦN D3: MẤT CÂN BẰNG DỮ LIỆU, CHỐNG RÒ RỈ & TIÊU CHÍ ĐÁNH GIÁ (CÂU 14 – 20)

### Câu 14: Vì sao tiêu chí Accuracy không đủ trong bài toán này? Hãy cho ví dụ bằng số thật của dự án.
1. **Trả lời ngắn:** Trong tập Test của dự án ($N = 4,250$ mẫu), Lớp 0 (Bình thường) chiếm tới **3,973 mẫu (93.48%)** và Lớp 1 (Phản cảm) chỉ có **277 mẫu (6.52%)**. Một mô hình "lười" dự đoán tất cả mọi câu đều là Lớp 0 sẽ đạt ngay **Accuracy = 93.48%**, nghe có vẻ rất cao nhưng thực tế **Recall = 0%, F1 lớp dương = 0.0000** — tức bỏ lọt 100% số tweet phân biệt chủng tộc/giới tính.
2. **Ý mở rộng:** Ngay cả trong thực nghiệm P4 khi dùng Logistic Regression không cân bằng (`none`), Accuracy đạt >94% nhưng Recall chỉ đạt **13.85%** và F1 chỉ đạt **0.2390**. Vì vậy, Accuracy hoàn toàn gây hiểu lầm trên dữ liệu lệch lớp.
3. **Bằng chứng:** Slide 4 & Slide 7, Chương 4 (Mục 4.2), Bảng `P4_balance_comparison.csv`.
4. **Bẫy cần tránh:** Khi báo cáo kết quả mô hình, luôn đọc **F1 lớp dương (`f1_pos`)**, **PR-AUC**, **Precision** và **Recall** trước; chỉ nhắc Accuracy như chỉ số phụ.

### Câu 15: Precision, Recall và F1-score lớp dương là gì? Trong bài này ưu tiên cái nào và vì sao?
1. **Trả lời ngắn:** Đối với Lớp 1 (Phản cảm): **Precision** $= \frac{TP}{TP+FP}$ là tỉ lệ tweet thực sự phản cảm trong số các tweet bị mô hình gắn cờ; **Recall** $= \frac{TP}{TP+FN}$ là tỉ lệ tweet phản cảm thực tế được mô hình phát hiện ra; còn **F1-score** $= \frac{2 \cdot P \cdot R}{P + R}$ là trung bình điều hòa của hai chỉ số này. Nhóm chọn **F1-score lớp dương (`f1_pos`)** làm tiêu chí chính để cân bằng giữa việc bắt được nhiều nội dung độc hại (Recall cao) và không chặn nhầm quá nhiều người dùng bình thường (Precision hợp lý).
2. **Ý mở rộng:** Nếu chỉ tối ưu Recall (ví dụ `RandomUnderSampler` đạt Recall **83.67%** ở bước thực nghiệm 4), Precision tụt xuống chỉ còn **28.27%** (cứ 4 câu bị chặn thì có gần 3 câu bị chặn oan). Ngược lại, nếu chỉ tối ưu Precision (như RBF SVM đạt Precision **89.25%**), Recall tụt xuống **29.96%** (bỏ lọt 70% tweet độc hại). Trung bình điều hòa F1 phạt nặng cả hai thái cực đó.
3. **Bằng chứng:** Slide 7 & Slide 9, Chương 4 (Mục 4.2), `src/evaluate.py::compute_metrics`.
4. **Bẫy cần tránh:** Phân biệt rõ `f1_pos` (F1 tính riêng trên lớp thiểu số 1 — khó hơn nhiều) với `f1_macro` (trung bình cộng F1 của lớp 0 và lớp 1; trên Test, `f1_pos` = **0.6266** còn `f1_macro` = **0.8004**).

### Câu 16: ROC-AUC và PR-AUC khác nhau thế nào? Vì sao PR-AUC phù hợp hơn với dữ liệu lệch lớp?
1. **Trả lời ngắn:** **ROC-AUC** vẽ đường cong giữa True Positive Rate (Recall) và False Positive Rate ($\text{FPR} = \frac{FP}{FP+TN}$). Khi số lượng mẫu âm thực tế ($TN$) quá lớn (**3,973 mẫu** so với **277 mẫu** dương), dù số lượng báo động giả $FP$ tăng gấp đôi hay gấp ba thì mẫu số $FP+TN$ quá lớn vẫn giữ FPR rất nhỏ, làm ROC-AUC luôn cao trên `0.90` (trên Test ROC-AUC = **0.9345**). Ngược lại, **PR-AUC** vẽ đường cong giữa Precision ($\frac{TP}{TP+FP}$) và Recall ($\frac{TP}{TP+FN}$), không hề chứa $TN$, nên phản ánh cực kỳ trung thực chất lượng dự đoán trên lớp thiểu số (trên Test PR-AUC = **0.6651**, so với mức ngẫu nhiên baseline là `0.0652`).
2. **Ý mở rộng:** Theo định lý của Davis & Goadrich (ICML 2006), một đường cong thống trị trên không gian PR thì chắc chắn thống trị trên không gian ROC, nhưng ngược lại thì không đúng khi dữ liệu mất cân bằng mạnh.
3. **Bằng chứng:** Slide 7 & Slide 9, Chương 4 (Mục 4.2), Hình `P5_roc_pr_curves.png` và `P8_test_evaluation.png`.
4. **Bẫy cần tránh:** Nếu giảng viên hỏi "Baseline ngẫu nhiên của PR-AUC bằng bao nhiêu?", trả lời ngay: "Bằng tỉ lệ lớp dương trong tập dữ liệu (**0.0652**), chứ không phải 0.500 như ROC-AUC."

### Câu 17: SMOTE và RandomOverSampler hoạt động thế nào? Vì sao CHỈ được áp dụng lên tập Train?
1. **Trả lời ngắn:** **RandomOverSampler (`ros`)** nhân bản ngẫu nhiên có hoàn lại các mẫu thuộc lớp thiểu số (Lớp 1), còn **SMOTE** (*Synthetic Minority Over-sampling Technique*) nội suy tuyến tính giữa mỗi mẫu Lớp 1 và $k=5$ láng giềng gần nhất cùng lớp trong không gian vector TF-IDF để tạo mẫu tổng hợp mới. Cả hai kỹ thuật này **bắt buộc chỉ được áp dụng lên tập Train bên trong từng fold của Cross-Validation** (thông qua `imblearn.pipeline.Pipeline`), tuyệt đối không áp dụng lên tập Validation hay Test.
2. **Ý mở rộng:** Lý do: (1) Tập Validation và Test phải giữ nguyên phân bố thực tế (6.52% lớp 1) để đánh giá đúng hiệu năng ngoài đời thực; (2) Nếu chạy SMOTE/ROS trước khi chia tập hoặc trước khi chia fold CV, các bản sao hoặc điểm nội suy của cùng một mẫu sẽ nằm ở cả tập huấn luyện lẫn tập đánh giá, gây rò rỉ dữ liệu nghiêm trọng.
3. **Bằng chứng:** Slide 6, Chương 3 (Mục 3.4), `src/models.py::build_pipeline`, Bảng `P4_balance_comparison.csv`.
4. **Bẫy cần tránh:** Tuyệt đối không viết `X_res, y_res = smote.fit_resample(X, y)` trước `train_test_split` hoặc trước `cross_val_score`.

### Câu 18: Data Leakage (Rò rỉ dữ liệu) là gì? Nhóm đã phòng chống và chứng minh không rò rỉ bằng cách nào?
1. **Trả lời ngắn:** Data Leakage là hiện tượng thông tin từ tập đánh giá (Val/Test) vô tình lọt vào quá trình huấn luyện hoặc lựa chọn mô hình, làm điểm số đánh giá cao giả tạo nhưng thất bại khi chạy thực tế. Nhóm đã thiết lập **4 hàng rào chống rò rỉ**: (1) Khử trùng gần-giống (`text_clean`) và loại bỏ mẫu mâu thuẫn nhãn **trước** khi chia tập; (2) Chia tập `train/val/test` một lần duy nhất lưu cố định ra đĩa và kiểm tra bằng unit test `tests/test_leakage.py` rằng giao nhau giữa 3 tập bằng rỗng ($\emptyset$); (3) Đặt toàn bộ `TfidfVectorizer` và `Sampler` vào trong `Pipeline` để chỉ `fit` trên tập Train; (4) Chọn siêu tham số và tối ưu ngưỡng quyết định chỉ dùng tập Train/Val, còn tập Test chỉ chạy đúng 1 lần duy nhất ở bước thực nghiệm 8 (ghi lại mã băm SHA-256 và timestamp).
2. **Ý mở rộng:** Nhóm cũng áp dụng quy tắc Sanity Check: với bộ dữ liệu mạng xã hội nhiều nhiễu này, nếu F1 lớp dương vượt quá `0.90` thì chắc chắn có rò rỉ; kết quả thực tế Test F1 = **0.6266** hoàn toàn nhất quán với Validation F1 = **0.6566**.
3. **Bằng chứng:** Slide 6 & Slide 7, Chương 3 (Mục 3.2), `tests/test_leakage.py`, `reports/final_test_metrics.json`.
4. **Bẫy cần tránh:** Nhấn mạnh cả phần phân tích SQL ở bước thực nghiệm 3 cũng chỉ nạp `data/splits/train.csv` vào bảng `tweets`/`tokens` chứ không nạp toàn bộ dữ liệu gốc.

### Câu 19: Nhóm chia tập dữ liệu như thế nào? Vì sao dùng `stratify=y`? Cross-Validation dùng để làm gì?
1. **Trả lời ngắn:** Từ **28,330 mẫu sạch**, nhóm chia theo tỉ lệ **70% Train (19,830 mẫu) – 15% Validation (4,250 mẫu) – 15% Test (4,250 mẫu)** với `random_state = 42` và `stratify = label`. Tham số `stratify` đảm bảo tỉ lệ Lớp 1 ở cả 3 tập đều bằng đúng **6.52%** (chênh lệch $< 0.01\%$). Còn **5-fold Stratified Cross-Validation** trên tập Train dùng để ước lượng độ ổn định của mô hình (trung bình $\pm$ độ lệch chuẩn qua 5 phần chia khác nhau) khi chọn kỹ thuật cân bằng  và tìm siêu tham số .
2. **Ý mở rộng:** Khi lớp dương chỉ có 1,846 mẫu, nếu chia ngẫu nhiên không phân tầng (`stratify`), một tập con có thể tình cờ nhận nhiều hoặc ít mẫu dương hơn hẳn, làm lệch ngưỡng quyết định và metric PR-AUC.
3. **Bằng chứng:** Slide 7, Chương 3 (Mục 3.2), `src/preprocess_p4.py`, `tests/test_leakage.py::test_stratified_positive_class_rate_consistency`.
4. **Bẫy cần tránh:** Không nhầm lẫn giữa Validation set (dùng để chọn mô hình và ngưỡng ở bước thực nghiệm 5) với Test set (chỉ dùng để báo cáo kết quả cuối ở bước thực nghiệm 8).

### Câu 20: Ngưỡng quyết định mặc định 0.50 có phải tối ưu không? Nhóm đã chỉnh ngưỡng ra sao?
1. **Trả lời ngắn:** Ngưỡng `0.50` không phải là tối ưu cho bài toán mất cân bằng lớp và sau khi hiệu chuẩn xác suất (*Platt scaling / CalibratedClassifierCV*). Ở mô hình chốt `Linear SVM (Calibrated)`, tại ngưỡng mặc định `0.50`, Precision đạt **0.7368** nhưng Recall chỉ đạt **0.5054** (Val F1 = **0.5996**). Nhóm viết hàm `pick_threshold` quét toàn bộ các ngưỡng từ `0.05` đến `0.95` (bước nhảy `0.005`) trên tập Validation để tối đa hóa F1, tìm ra ngưỡng tối ưu **`threshold = 0.315`**, giúp Recall tăng vọt lên **0.6282** và đẩy Val F1 lên **0.6566** (và trên tập Test cũng tăng F1 từ **0.6427** ở ngưỡng 0.50 lên **0.6266** ở ngưỡng `0.315`).
2. **Ý mở rộng:** Tùy vào yêu cầu vận hành của hệ thống kiểm duyệt, nếu muốn ưu tiên bắt tối đa tweet độc hại (chấp nhận duyệt tay nhiều hơn), hàm `pick_threshold(strategy='f2')` của nhóm cũng hỗ trợ tối ưu hóa chỉ số $F_2$-score (ưu tiên Recall gấp đôi Precision).
3. **Bằng chứng:** Slide 10, Chương 5 (Mục 5.3), `src/threshold.py`, `models/threshold.json`, Hình `P6_threshold_tuning.png`.
4. **Bẫy cần tránh:** Phải khẳng định rõ: ngưỡng `0.315` được tìm trên **tập Validation**, sau đó giữ cố định đem áp vào tập Test và tập Thực tế, tuyệt đối không dò ngưỡng trên tập Test!

---

## PHẦN D4: CÁC MÔ HÌNH HỌC MÁY & TINH CHỈNH THAM SỐ (CÂU 21 – 29)

### Câu 21: Vì sao nhóm thử nghiệm 9 mô hình cổ điển này? Ưu và nhược điểm của từng mô hình là gì?
1. **Trả lời ngắn:** Nhóm lựa chọn 9 mô hình cổ điển đại diện cho đầy đủ các trường phái học máy đã học: (1) **Logistic Regression (`lr`)**: nhanh, xuất xác suất chuẩn, giải thích được hệ số từng từ nhưng giới hạn ở biên tuyến tính; (2) **Complement Naive Bayes (`nb`)**: cực nhanh ($0.5$s), thiết kế riêng cho văn bản lệch lớp nhưng giả định các từ độc lập điều kiện nên Precision thấp hơn; (3) **Calibrated Linear SVM (`svm_linear`)**: tối đa hóa lề phân cách trong không gian TF-IDF chiều cao, cho F1 và Precision tốt nhất; (4) **RBF SVM (`svm_rbf`)**: bắt được tương tác phi tuyến nhưng độ phức tạp $O(n^2 \sim n^3)$ rất chậm và dễ overfit trên vector thưa; (5) **Decision Tree (`dt`) & Random Forest (`rf`)**: dễ hiểu nhưng phân mảnh dữ liệu khi gặp 50,000 đặc trưng thưa; (6) **Cosine KNN (`knn`)**: không cần huấn luyện nhưng suy luận chậm ($523$ms/1k mẫu) và chịu lời nguyền số chiều; (7) **XGBoost (`xgb`) & LightGBM (`lgbm`)**: mạnh trên dữ liệu bảng nhưng kém hiệu quả hơn mô hình tuyến tính trên ma trận n-gram thưa.
2. **Ý mở rộng:** Việc thử nghiệm toàn diện cả 9 mô hình trên cùng một thước đo và cùng một tập chia giúp chứng minh bằng thực nghiệm định lý "No Free Lunch": các thuật toán phức tạp như XGBoost hay Random Forest không phải lúc nào cũng thắng thuật toán tuyến tính trên dữ liệu văn bản TF-IDF.
3. **Bằng chứng:** Slide 8 & Slide 9, Chương 4 & Chương 5, `src/models.py`, Bảng `reports/model_comparison.csv`.
4. **Bẫy cần tránh:** Không nói "XGBoost luôn mạnh nhất nhưng do máy yếu nên chạy thấp" — phải giải thích đúng bản chất toán học của dữ liệu văn bản thưa (sparse high-dimensional text).

### Câu 22: Thuật toán Naive Bayes giả định điều gì? Vì sao giả định đó sai trong ngôn ngữ tự nhiên nhưng mô hình vẫn chạy tốt?
1. **Trả lời ngắn:** Naive Bayes giả định các đặc trưng (từ/cụm từ) **độc lập có điều kiện** với nhau khi biết nhãn lớp: $P(w_1, w_2, \dots, w_d \mid y) = \prod_{j=1}^d P(w_j \mid y)$. Trong ngôn ngữ tự nhiên giả định này hiển nhiên sai (vì từ đi theo cụm ngữ pháp), nhưng mô hình vẫn phân loại văn bản rất tốt (Complement NB `word+char` đạt PR-AUC = **0.7403**, Recall = **84.48%**) vì để phân loại đúng, chúng ta không cần ước lượng xác suất tuyệt đối chính xác mà chỉ cần **thứ tự xếp hạng hoặc dấu của tỉ số log-likelihood** nằm đúng phía bên kia biên quyết định (theo chứng minh của Domingos & Pazzani, 1997).
2. **Ý mở rộng:** Hơn nữa, nhóm sử dụng biến thể **Complement Naive Bayes (`ComplementNB`)** của Rennie et al. (ICML 2003): thay vì ước lượng xác suất từ trong lớp thiểu số (vốn rất ít mẫu), nó ước lượng trọng số từ trên phần bù (tức lớp đa số có nhiều dữ liệu hơn), giúp giảm hẳn độ chệch do mất cân bằng lớp.
3. **Bằng chứng:** Slide 8 & Slide 9, Chương 4 (Mục 4.3), `src/models.py`, Bảng `reports/model_comparison.csv`.
4. **Bẫy cần tránh:** Nhớ nói rõ nhóm dùng `ComplementNB(alpha=0.1497)` chứ không chỉ dùng `GaussianNB` (vốn chỉ hợp dữ liệu liên tục phân phối chuẩn).

### Câu 23: SVM tuyến tính (`LinearSVC`) khác gì với SVM dùng nhân phi tuyến (`SVC(kernel='rbf')`) đối với dữ liệu văn bản?
1. **Trả lời ngắn:** Với dữ liệu văn bản TF-IDF có số chiều rất lớn ($d = 50,000 \sim 100,000$ đặc trưng), các điểm dữ liệu vốn đã khả phân tuyến tính cao trong không gian chiều cao (theo định lý Cover). Do đó, **Linear SVM** tìm siêu phẳng lề cực đại trực tiếp trong không gian gốc vừa rất nhanh (**0.95 giây**), vừa tổng quát hóa tốt (Val F1 = **0.6566** sau tuning). Ngược lại, **RBF-Kernel SVM** ánh xạ dữ liệu sang không gian vô hạn chiều làm ma trận nhân (Gram matrix) trở nên gần như chéo hóa (mọi văn bản đều cách đều nhau), khiến Recall tụt xuống **29.96%** (Val F1 = **0.4486**) và thời gian suy luận chậm gấp **30 lần** (**575ms/1k mẫu** so với **18.9ms/1k mẫu**).
2. **Ý mở rộng:** Theo hướng dẫn kinh điển của Joachims (1998) và Hsu et al. (LIBSVM Guide), khi số chiều đặc trưng $d$ lớn hơn hoặc xấp xỉ số mẫu $n$, việc dùng kernel RBF không mang lại lợi ích độ chính xác so với kernel tuyến tính.
3. **Bằng chứng:** Slide 9, Chương 5 (Mục 5.1), Bảng `reports/model_comparison.csv`, Hình `P5_f1_prauc_bars.png`.
4. **Bẫy cần tránh:** Nhớ giải thích thêm rằng vì `LinearSVC` nguyên bản không xuất xác suất (`predict_proba`), nhóm đã bọc nó trong `CalibratedClassifierCV(cv=3)` (Platt scaling) để có xác suất chuẩn phục vụ tối ưu ngưỡng và vẽ đường cong PR.

### Câu 24: Vì sao Random Forest và Decision Tree thường đạt kết quả kém hơn Logistic Regression / Linear SVM trên ma trận TF-IDF thưa?
1. **Trả lời ngắn:** Trên tập Validation, `Decision Tree` chỉ đạt F1 = **0.3836** và `Random Forest` đạt F1 = **0.5195** (thấp hơn nhiều so với `Linear SVM` **0.6566** và `Logistic Regression` **0.6486**). Nguyên nhân là mỗi nút chia của cây quyết định chỉ kiểm tra sự xuất hiện của **một từ/cụm từ duy nhất** (`tfidf(w) <= threshold`). Vì mỗi tweet chỉ dài trung bình 12 từ trên tổng số 50,000 từ vựng (độ thưa > 99.97%), nhánh `True` chỉ tách ra được vài chục mẫu còn nhánh `False` chứa tới 99% số mẫu còn lại, khiến cây độ sâu 30–40 chỉ học được một số ít từ khóa phổ biến và bỏ sót hàng nghìn từ khóa hiếm.
2. **Ý mở rộng:** Trong khi đó, mô hình tuyến tính tính tổng có trọng số $\mathbf{w}^\top \mathbf{x} + b$ trên toàn bộ 50,000 từ cùng một lúc: bất kỳ từ nào xuất hiện trong câu đều đóng góp tức thì vào điểm số dự đoán.
3. **Bằng chứng:** Slide 9, Chương 5 (Mục 5.1), Bảng `reports/model_comparison.csv`, Hình `P5_confusion_grid_3x3.png`.
4. **Bẫy cần tránh:** Không đổ lỗi cho thiếu cây (`n_estimators`), vì nhóm đã dùng tới 200–300 cây cho Random Forest, XGBoost và LightGBM.

### Câu 25: Mô hình nào đạt kết quả tốt nhất? Sự chênh lệch giữa các mô hình đầu bảng có ý nghĩa thống kê không?
1. **Trả lời ngắn:** Trước khi tinh chỉnh , cấu hình `Logistic Regression (word+char)` dẫn đầu với 5-fold CV F1 = **0.6478 ± 0.0186** (Val F1 = **0.6486**). Sau khi tinh chỉnh siêu tham số và tối ưu ngưỡng quyết định , **`Linear SVM (Calibrated)`** với `ngram_range=(1,3)`, `min_df=2`, `C=0.50191` và ngưỡng `threshold = 0.315` đạt kết quả tốt nhất trên Validation với **F1 = 0.6566** (PR-AUC = **0.6636**, ROC-AUC = **0.9321**). Kiểm định **Paired t-test** trên 5-fold CV (`P5_paired_ttest.csv`) cho thấy sự vượt trội của cấu hình kết hợp đặc trưng ký tự (`lr__word+char`) so với cấu hình từ đơn thuần (`lr__word_tfidf` và `svm_linear__word_tfidf`) có ý nghĩa thống kê rõ rệt ($p < 0.01$).
2. **Ý mở rộng:** Trên tập Test độc lập ở bước thực nghiệm 8, nhóm cũng tính khoảng tin cậy 95% bằng Bootstrap 1,000 lần cho mô hình chốt: **Test F1 = 0.6266** với khoảng tin cậy 95% là **[0.5764, 0.6716]**.
3. **Bằng chứng:** Slide 9 & Slide 10, Chương 5, Bảng `P5_paired_ttest.csv`, `P6_before_after.csv` và `reports/final_test_ci.json`.
4. **Bẫy cần tránh:** Luôn kèm theo độ lệch chuẩn ($\pm$ std) hoặc khoảng tin cậy 95% CI khi phát biểu về hiệu năng mô hình.

### Câu 26: Vì sao nhóm chọn `Linear SVM (Calibrated)` làm mô hình cuối cùng (`best_model`) để đưa vào ứng dụng?
1. **Trả lời ngắn:** Nhóm chọn `Linear SVM (Calibrated)` vì 3 lý do thực tiễn: (1) **Hiệu năng cao nhất sau tuning:** đạt Validation F1 = **0.6566** (cao hơn Logistic Regression đạt **0.6250** và Complement NB đạt **0.5922** trên lưới tuning P6), đồng thời có sự cân bằng rất tốt giữa Precision (**0.6877**) và Recall (**0.6282**); (2) **Tốc độ suy luận cực nhanh:** chỉ mất **~18.9 ms cho 1,000 tweet** và file mô hình rất nhẹ, khởi động tức thì trên Streamlit; (3) **Khả năng giải thích minh bạch:** bằng cách lấy trung bình vector trọng số $\mathbf{w}$ của 3 bộ phân loại tuyến tính trong `CalibratedClassifierCV`, mô hình tính được chính xác mức đóng góp `TF-IDF × Hệ số` của từng từ cho người dùng xem.
2. **Ý mở rộng:** So với các mô hình học sâu (DistilBERT nặng ~260MB và cần GPU để suy luận nhanh), `Linear SVM (Calibrated)` chạy mượt mà trên bất kỳ máy tính CPU nào mà vẫn đạt Test F1 = **0.6266**.
3. **Bằng chứng:** Slide 10 & Slide 13, Chương 5 (Mục 5.4), `models/model_card.json`, `src/predict.py::_extract_linear_coef`.
4. **Bẫy cần tránh:** Nếu bị hỏi "Làm sao trích xuất hệ số từ `CalibratedClassifierCV`?", giải thích ngay hàm `_extract_linear_coef` trong `src/predict.py` lấy trung bình `estimator.coef_` qua 3 fold hiệu chuẩn.

### Câu 27: Mô hình học sâu mở rộng ngoài chương trình (`BiLSTM` và `DistilBERT`): cấu hình ra sao, kết quả thế nào, có đáng đánh đổi thời gian không?
1. **Trả lời ngắn:** Ở giai đoạn 7, nhóm cài đặt 2 mô hình học sâu: (1) **BiLSTM (PyTorch)** dùng tầng Embedding 128 chiều, LSTM hai chiều 64 nút ẩn kết hợp Global Average + Max Pooling và hàm mất mát `BCEWithLogitsLoss` có trọng số lớp, huấn luyện 6 epochs đạt **Val F1 = 0.5973** (PR-AUC = **0.6168**, thời gian train **83.9s**); (2) **DistilBERT (`distilbert-base-uncased`)** fine-tune với `max_length=80`, `lr=2e-5`, `CrossEntropyLoss` có trọng số lớp đạt **Val F1 = 0.4434** (PR-AUC = **0.5600**, ROC-AUC = **0.9266**, thời gian train **257.3s**).
2. **Ý mở rộng:** Về mặt đánh đổi (*trade-off*): Khi chạy trên môi trường không có GPU chuyên dụng hoặc cần phản hồi thời gian thực cho hàng chục nghìn dòng CSV, DistilBERT tốn thời gian huấn luyện và suy luận lâu hơn hàng trăm lần và dung lượng lớn gấp hàng chục lần so với `Linear SVM (Calibrated)` (Val F1 = **0.6566**). Do đó, dùng `Linear SVM (Calibrated)` cho ứng dụng web sản phẩm và giữ `BiLSTM`/`DistilBERT` làm mốc đối sánh học thuật là lựa chọn tối ưu.
3. **Bằng chứng:** Slide 9, Chương 5 (Mục 5.5), `src/transformer.py`, `reports/p7_summary.json`, Hình `P7_bonus_comparison.png`, Notebook `06_transformer_bonus.ipynb`.
4. **Bẫy cần tránh:** Nêu rõ trên máy CPU cục bộ, để tránh treo máy hàng giờ, `DistilBERT` được fine-tune trên mẫu phân tầng 6,000 dòng (2 epochs), còn khi chạy trên Google Colab T4 GPU sẽ tự động fine-tune toàn bộ 19,830 dòng (3 epochs, `fp16=True`).

### Câu 28: Overfitting (Quá khớp) là gì? Nhóm đã kiểm tra và hạn chế Overfitting như thế nào?
1. **Trả lời ngắn:** Overfitting xảy ra khi mô hình học thuộc lòng cả nhiễu của tập Train (điểm Train rất cao) nhưng suy giảm mạnh trên tập Validation và Test. Nhóm kiểm soát Overfitting bằng 3 cách: (1) Dùng điều chuẩn $L_2$ (`C = 0.50191` trong Linear SVM, phạt các trọng số quá lớn); (2) Giới hạn `min_df = 2` và `max_features = 50,000` để bỏ các từ chỉ xuất hiện 1 lần; (3) Đối chiếu sát sao giữa **5-fold CV F1 trên Train (`0.6112`)**, **Validation F1 (`0.6566`)** và **Test F1 (`0.6266`)** — ba con số này rất sát nhau, chứng minh mô hình tổng quát hóa cực kỳ ổn định và không bị Overfitting.
2. **Ý mở rộng:** Ở mô hình BiLSTM và DistilBERT , nhóm sử dụng `Dropout(0.3)`, `weight_decay=0.01` và lưu lại checkpoint có Validation F1 cao nhất qua từng epoch.
3. **Bằng chứng:** Slide 10 & Slide 11, Chương 5 & Chương 6, Bảng `P6_before_after.csv` và `reports/final_test_metrics.json`.
4. **Bẫy cần tránh:** Chỉ ra ví dụ mô hình bị Overfitting thực tế trong dự án chính là `Decision Tree` (trên Train học rất sát nhưng sang Validation F1 rơi xuống **0.3836**).

### Câu 29: Tham số nào ảnh hưởng nhiều nhất đến kết quả? Vì sao nhóm dùng `RandomizedSearchCV` thay vì `GridSearchCV`?
1. **Trả lời ngắn:** Hai tham số ảnh hưởng mạnh nhất là: (1) Hệ số điều chuẩn **`C`** cùng `class_weight='balanced'` của bộ phân loại tuyến tính, và (2) Phạm vi cụm từ **`ngram_range=(1,3)`** kết hợp **`min_df=2`** của bộ trích xuất TF-IDF, cùng với **ngưỡng quyết định `threshold = 0.315`**. Nhóm chọn `RandomizedSearchCV` (`n_iter = 40`, `cv = 5`) thay vì `GridSearchCV` vì không gian kết hợp cả tham số Vectorizer lẫn Classifier có hàng trăm tổ hợp và chứa biến liên tục (`C` phân phối `loguniform(1e-2, 1e2)`). Theo nghiên cứu của Bergstra & Bengio (JMLR 2012), tìm kiếm ngẫu nhiên với 40–60 lần thử khám phá không gian tham số liên tục hiệu quả hơn nhiều so với lưới cố định với chi phí tính toán thấp hơn.
2. **Ý mở rộng:** Nhờ `RandomizedSearchCV`, cả 3 mô hình đều cải thiện mạnh Validation F1: Logistic Regression tăng **+4.18 điểm %**, Linear SVM tăng **+8.39 điểm %**, và Complement NB tăng **+8.45 điểm %**.
3. **Bằng chứng:** Slide 10, Chương 5 (Mục 5.2), `src/tuning_p6.py`, Bảng `P6_before_after.csv`.
4. **Bẫy cần tránh:** Nhớ nói rõ `scoring='f1'` được truyền vào `RandomizedSearchCV` (chứ không để `scoring=None` mặc định là Accuracy!).

---

## PHẦN D5: KIỂM THỬ THỰC TẾ, PHÂN TÍCH LỖI & HẠN CHẾ (CÂU 30 – 36)

### Câu 30: Bộ dữ liệu tweet thực tế  lấy ở đâu, gắn nhãn thế nào? Hệ số Cohen's Kappa bằng bao nhiêu?
1. **Trả lời ngắn:** Bộ dữ liệu thực tế `data/real_world/real_world_labeled.csv` gồm **300 mẫu** được xây dựng từ 4 nguồn độc lập (hoàn toàn không trùng với Train/Val/Test): **Nguồn A** (100 mẫu từ `hate_unlabeled_test.csv`), **Nguồn B** (100 mẫu từ `Twitter_Data.csv`), **Nguồn C** (50 mẫu từ `Reddit_Data.csv`), và **Nguồn D** (50 câu tình huống phức tạp tự biên soạn gồm câu mỉa mai, phủ định phản đối kỳ thị, chửi thề cảm thán, và câu trung tính chứa từ nhạy cảm). Quá trình gắn nhãn độc lập bởi 2 luồng (`label_a` và `label_b`) theo tài liệu `ANNOTATION_GUIDE.md` đạt tỉ lệ đồng thuận **97.00%** (chỉ bất đồng ở 9/300 mẫu ranh giới) và hệ số **Cohen's Kappa $\kappa = 0.9204$** (mức đồng thuận rất cao).
2. **Ý mở rộng:** Để tránh thiên kiến xác nhận (*confirmation bias*), file `data/real_world/to_label.csv` xuất ra cho người gắn nhãn đã được ẩn `@user`, URL và **không chứa cột điểm xác suất của mô hình**.
3. **Bằng chứng:** Slide 11, Chương 6 (Mục 6.3), `data/real_world/ANNOTATION_GUIDE.md`, `reports/realworld_metrics.json`.
4. **Bẫy cần tránh:** Giải thích được ý nghĩa của Cohen's Kappa: $\kappa = \frac{p_o - p_e}{1 - p_e}$ đã loại trừ xác suất hai người trùng nhãn do ngẫu nhiên ($p_e$), nên đáng tin cậy hơn tỉ lệ đồng thuận phần trăm đơn thuần.

### Câu 31: Hiệu năng trên bộ thực tế như thế nào so với tập Test? Hiện tượng Domain Shift thể hiện ở đâu?
1. **Trả lời ngắn:** Trên toàn bộ 300 mẫu thực tế, mô hình đạt **F1 = 0.5702** (Precision = **0.4221**, Recall = **0.8784**, PR-AUC = **0.5742**). Tuy nhiên, khi tách riêng theo từng nguồn (`P9_metrics_by_source.csv`), hiện tượng **lệch miền (Domain Shift)** hiện ra rất rõ: ở các nguồn A, B, C (văn bản mạng xã hội chuẩn), mô hình đạt F1 rất cao (**0.82 – 0.90**), nhưng khi chuyển sang **Nguồn D (`D_adversarial_curated` — 50 câu tình huống phức tạp)**, F1 giảm xuống còn **0.5116** (Recall chỉ đạt **55.00%**). Tương tự, khi kiểm thử chéo hoàn toàn không huấn luyện lại trên bộ **Davidson et al.** ($N = 5,593$), F1 đạt **0.3653** (Precision = **0.5876**, Recall = **0.2650**).
2. **Ý mở rộng:** Trên bộ `Davidson et al.`, Precision đạt mức rất cao (**77.08%**) nhưng Recall chỉ đạt **25.87%** vì bộ Davidson thu thập nhiều từ lóng tiếng Anh người Mỹ gốc Phi (AAE) và từ vựng khác thời điểm so với bộ `hate_train.csv` — một minh chứng kinh điển của Domain Shift về từ vựng.
3. **Bằng chứng:** Slide 11, Chương 6 (Mục 6.3 & 6.4), Bảng `P9_metrics_by_source.csv`, Hình `P9_realworld_evaluation.png`.
4. **Bẫy cần tránh:** Không sợ khi thấy F1 trên Nguồn D hoặc bộ Davidson thấp hơn Nguồn A; chính sự sụt giảm đó là bằng chứng thực nghiệm quý giá để phân tích Domain Shift.

### Câu 32: Mô hình thường dự đoán sai ở những loại tweet nào?
1. **Trả lời ngắn:** Qua thống kê toàn bộ **205 lỗi** trên tập Test (`P8_error_causes.csv`), có 4 nhóm lỗi chính: (1) **Kỳ thị ngầm / Định kiến thiếu từ tục trực tiếp** (chiếm **31.7%** tổng số lỗi — gây ra đa số các ca Bỏ sót FN); (2) **Nhãn gốc nhiễu hoặc ranh giới ngữ nghĩa mờ** (**22.9%**); (3) **Từ nhạy cảm / Từ chỉ nhóm người xuất hiện trong câu bình thường hoặc câu phản đối kỳ thị** (**15.6%** — gây ra Báo động giả FP); (4) **Câu mỉa mai, châm biếm, câu hỏi tu từ, chuỗi nhiều hashtag hoặc câu quá ngắn/từ lóng**.
2. **Ý mở rộng:** Ví dụ điển hình ở Nguồn D: câu *"Men are built to rule and women are built to serve, that is just biological truth"* không hề chứa từ chửi thề nào nên mô hình cho điểm xác suất chỉ `0.0689` (Bỏ sót — FN).
3. **Bằng chứng:** Slide 12, Chương 6 (Mục 6.2), Bảng `P8_error_causes.csv`, `P8_errors.csv`, Hình `P8_error_causes.png`.
4. **Bẫy cần tránh:** Luôn chuẩn bị sẵn 1 ví dụ cụ thể về FP và 1 ví dụ cụ thể về FN (đã che từ tục) để đọc cho giảng viên nghe.

### Câu 33: Có những đoạn tweet bình thường chứa từ tục hoặc từ nhạy cảm bị mô hình đoán nhầm thành phản cảm (False Positive) không? Vì sao?
1. **Trả lời ngắn:** Có. Ví dụ trên tập thực tế Nguồn D, câu *"Calling women 'b***hes' or 's**ts' in rap lyrics and online forums is unacceptable and toxic"* (trích dẫn từ tục để lên án hành vi xúc phạm phụ nữ — Nhãn thực tế = `0`) bị mô hình dự đoán nhầm thành Nhãn `1` với xác suất `0.3652` (vượt ngưỡng `0.315`). Lý do là mô hình dựa trên TF-IDF n-gram gán trọng số dương rất lớn cho bản thân các từ nhạy cảm (`b***h`, `s**t`, `woman`) mà chưa hiểu được cấu trúc mệnh đề đánh giá `"... is unacceptable"` ở cuối câu.
2. **Ý mở rộng:** Đây được gọi là bài toán **Counter-speech / Quoted Hate Speech** trong NLP: ngay cả các mô hình lớn như BERT nếu không được huấn luyện trên dữ liệu đối kháng có trích dẫn cũng thường mắc lỗi này.
3. **Bằng chứng:** Slide 12, Chương 6 (Mục 6.2 & 6.3), Bảng `P9_realworld_examples.csv` (mẫu ID 275).
4. **Bẫy cần tránh:** Không nói mô hình của mình hoàn toàn không bị lừa bởi từ tục.

### Câu 34: Mô hình hiện tại có phân biệt được riêng "phân biệt chủng tộc (racist)" với "phân biệt giới tính (sexist)" không? Hướng mở rộng đa lớp thế nào?
1. **Trả lời ngắn:** Hiện tại mô hình giải quyết bài toán **phân loại nhị phân** theo đúng yêu cầu đề bài và cấu trúc nhãn gốc của bộ dữ liệu (`label = 1` gộp chung cả racist và sexist, `label = 0` là bình thường). Tuy nhiên, nhờ tính năng giải thích **Top-8 từ đóng góp** trên ứng dụng Streamlit và phân tích Lexicon SQL (`P3_lexicon_by_category.csv`), người dùng vẫn nhìn thấy rõ câu đó bị gắn cờ do nhóm từ chủng tộc (`racist, white, black`) hay nhóm từ giới tính (`sexist, woman, misogyny`).
2. **Ý mở rộng:** Để mở rộng sang **phân loại đa lớp / đa nhãn (Multi-label Classification)** gồm 4 nhãn `[Normal, Racist, Sexist, Religious/Other Hate]`, hướng phát triển là kết hợp bộ dữ liệu `HatEval (SemEval-2019 Task 5)` và `Davidson et al.`, sử dụng tầng đầu ra Sigmoid đa nhãn trên kiến trúc RoBERTa.
3. **Bằng chứng:** Slide 14, Chương 8 (Mục 8.1 & 8.3), Bảng `P3_lexicon_by_category.csv`.
4. **Bẫy cần tránh:** Trả lời dứt khoát "Hiện tại mô hình đầu ra là nhị phân 0/1 vì nhãn huấn luyện gốc gộp chung, nhưng tầng giải thích từ khóa hỗ trợ nhận diện nhóm nguyên nhân."

### Câu 35: Mô hình có nguy cơ thiên vị với phương ngữ hoặc nhóm nhân khẩu học cụ thể (như tiếng Anh người Mỹ gốc Phi — AAE) không?
1. **Trả lời ngắn:** Có, đây là rủi ro đạo đức đã được chứng minh trong các nghiên cứu về phát hiện ngôn từ thù ghét (như Sap et al., ACL 2019; Davidson et al., 2019). Trong thực nghiệm P8 (`P8_subgroup_bias.csv`), nhóm đo lường trực tiếp trên tập Test và thấy rằng các câu bình thường (Nhãn `0`) có nhắc đến từ khóa chủng tộc (`black, white, race`) có tỉ lệ báo động giả **FPR = 30.43%**, và nhóm từ tôn giáo (`muslim, islam, jewish`) có **FPR = 20.00%**, cao hơn đáng kể so với mức FPR nền của toàn tập Test (**2.52%**).
2. **Ý mở rộng:** Nguyên nhân là trong tập huấn luyện, các từ định danh nhóm thiểu số xuất hiện trong câu thù ghét với tần suất quá cao, khiến mô hình học mối tương quan giả (*spurious correlation*) giữa tên nhóm người và nhãn độc hại.
3. **Bằng chứng:** Slide 12 & Slide 14, Chương 6 (Mục 6.5) & Chương 8 (Mục 8.2), Bảng `P8_subgroup_bias.csv`, Hình `P8_subgroup_bias.png`.
4. **Bẫy cần tránh:** Việc nhóm tự chủ động đo và vẽ biểu đồ `P8_subgroup_bias.png` sẽ gây ấn tượng rất mạnh với giảng viên khi được hỏi câu này.

### Câu 36: Nếu đưa mô hình vào sử dụng thực tế hoàn toàn tự động thì có nguy cơ gì? Vai trò của con người ở đâu?
1. **Trả lời ngắn:** Nếu dùng mô hình để tự động khóa tài khoản hoặc xóa bài mà không có con người duyệt lại, sẽ xảy ra 2 nguy cơ lớn: (1) **Kiểm duyệt quá mức (Over-censorship / False Positives):** chặn nhầm những người đang lên tiếng bảo vệ phụ nữ hoặc chống phân biệt chủng tộc; (2) **Bỏ lọt nội dung nguy hiểm (False Negatives):** những kẻ cực đoan dùng ngôn từ ẩn dụ, viết tắt hoặc mỉa mai để lách bộ lọc. Vì vậy, mô hình phải được vận hành theo cơ chế **Human-in-the-loop** (Con người trong vòng lặp): tự động gắn cờ và xếp hạng mức độ rủi ro để kiểm duyệt viên xử lý nhanh gấp 10 lần.
2. **Ý mở rộng:** Trên ứng dụng Streamlit , nhóm đã thiết kế thanh hiển thị khoảng cách tới ngưỡng quyết định `0.315` và dòng cảnh báo đạo đức cố định ở chân trang.
3. **Bằng chứng:** Slide 13 & Slide 14, Chương 7 & Chương 8, `app/app.py`.
4. **Bẫy cần tránh:** Không bao giờ tuyên bố mô hình AI có thể thay thế 100% con người trong kiểm duyệt nội dung nhạy cảm.

---

## PHẦN D6: ỨNG DỤNG WEB, TRIỂN KHAI & ĐÓNG GÓP RIÊNG (CÂU 37 – 40)

### Câu 37: Ứng dụng Demo sử dụng công nghệ gì, luồng xử lý ra sao? Mô hình được nạp như thế nào để không bị chậm?
1. **Trả lời ngắn:** Ứng dụng được viết bằng **Streamlit (`app/app.py`)** kết hợp mô-đun suy luận `src/predict.py`. Luồng xử lý khi người dùng nhập 1 tweet hoặc tải file CSV: Văn bản thô (cắt tối đa 1,000 ký tự) $\rightarrow$ `clean_text` 12 bước $\rightarrow$ `TfidfVectorizer.transform` $\rightarrow$ `CalibratedClassifierCV.predict_proba` $\rightarrow$ So sánh với ngưỡng `threshold = 0.315` từ `models/threshold.json` $\rightarrow$ Trích xuất Top-8 từ đóng góp (`TF-IDF × Hệ số`) và che từ tục bằng `mask_profanity` trước khi hiển thị. Để ứng dụng phản hồi tức thì, mô hình được nạp một lần duy nhất vào RAM nhờ decorator **`@st.cache_resource`**.
2. **Ý mở rộng:** Nhờ `@st.cache_resource` và vector hóa hàng loạt (`predict_batch`), ứng dụng xử lý 1,000 dòng tweet chỉ mất dưới `0.1 giây` trên CPU thông thường.
3. **Bằng chứng:** Slide 13, Chương 7, `app/app.py`, `src/predict.py`, `tests/test_predict.py`.
4. **Bẫy cần tránh:** Nhớ nhắc việc xử lý an toàn các trường hợp biên: chuỗi rỗng, chuỗi chỉ chứa icon/số, hoặc file CSV vượt quá 20,000 dòng đều được kiểm soát không gây crash.

### Câu 38: Nếu có thêm thời gian và tài nguyên, nhóm sẽ cải thiện hệ thống theo những hướng nào?
1. **Trả lời ngắn:** Nhóm đề xuất 4 hướng cải tiến cụ thể: (1) **Dữ liệu:** Bổ sung các mẫu *Counter-speech* (phản đối kỳ thị) và *Identity-neutral* để giảm False Positive Rate trên các nhóm nhạy cảm (Debiasing); (2) **Mô hình:** Fine-tune các mô hình ngôn ngữ chuyên biệt cho mạng xã hội như `cardiffnlp/twitter-roberta-base-hate` hoặc `HateBERT` trên toàn bộ tập dữ liệu bằng GPU; (3) **Nhãn:** Mở rộng từ nhị phân sang phân loại đa nhãn (Racist / Sexist / Offensive / Normal); (4) **Ngôn ngữ:** Áp dụng quy trình này cho **tiếng Việt** trên các bộ dữ liệu như `ViHSD` (Vietnamese Hate Speech Detection) kết hợp `PhoBERT`.
2. **Ý mở rộng:** Ngoài ra có thể tích hợp cơ chế **Active Learning** ngay trên ứng dụng Streamlit: khi kiểm duyệt viên sửa lại một nhãn dự đoán sai, mẫu đó được lưu vào hàng đợi để định kỳ huấn luyện lại (retrain) mô hình.
3. **Bằng chứng:** Slide 14, Chương 8 (Mục 8.3).
4. **Bẫy cần tránh:** Đưa ra hướng phát triển phải gắn trực tiếp với những hạn chế thật đã tìm thấy ở bước thực nghiệm 8 và P9.

### Câu 39: Vấn đề đạo đức dữ liệu (Data Ethics) và bảo vệ người dùng được nhóm tuân thủ như thế nào?
1. **Trả lời ngắn:** Nhóm tuân thủ 3 nguyên tắc đạo đức nghiêm ngặt: (1) **Ẩn danh thông tin cá nhân:** Toàn bộ tên tài khoản người dùng đều được ẩn thành `@user` và liên kết web chuyển thành `[URL]`, tuyệt đối không công khai danh tính cá nhân; (2) **Che từ tục tự động (`mask_profanity`):** Mọi từ ngữ tục tĩu/miệt thị khi xuất hiện trên biểu đồ, bảng số liệu, notebook, slide và giao diện ứng dụng đều được giữ ký tự đầu/cuối và thay phần giữa bằng dấu `*` (ví dụ `b***h`, `n****r`), trong khi dữ liệu thô để máy học vẫn giữ nguyên độ chính xác; (3) **Minh bạch giới hạn:** Ghi rõ cảnh báo mô hình chỉ hỗ trợ sàng lọc.
2. **Ý mở rộng:** Toàn bộ dữ liệu sử dụng đều là các bộ dữ liệu nghiên cứu công khai có giấy phép rõ ràng (CC0 / MIT License), không tự ý thu thập trái phép dữ liệu riêng tư.
3. **Bằng chứng:** Chương 1 & Chương 8 (Mục 8.2), `src/utils.py::mask_profanity`, `reports/data_report.json`.
4. **Bẫy cần tránh:** Kiểm tra kỹ không để sót bất kỳ từ tục nguyên văn nào trên slide thuyết trình.

### Câu 40: Đóng góp riêng nổi bật nhất của nhóm so với một bài làm thông thường là gì?
1. **Trả lời ngắn:** Nhóm có **5 đóng góp vượt trội**: (1) Phát hiện và chứng minh định lượng việc bộ dữ liệu được giao (`Twitter_Data.csv`) chỉ chứa nhãn sentiment chính trị chứ không phải nhãn racist/sexist, từ đó thiết kế hướng tiếp cận kết hợp dữ liệu chuẩn khoa học; (2) Xây dựng CSDL SQLite `db/tweets.db` với 6 truy vấn SQL kết hợp từ điển Lexicon phân nhóm để tìm điểm chung của tweet phản cảm; (3) Quy trình chống rò rỉ dữ liệu 4 tầng có kiểm thử tự động (`pytest` 17 tests), khử trùng 3,628 mẫu retweet trước khi chia tập và so sánh 6 kỹ thuật cân bằng; (4) Thực nghiệm đối sánh **11 mô hình** (9 cổ điển + BiLSTM + DistilBERT) kèm tối ưu ngưỡng quyết định giúp tăng **+8.39% F1**; (5) Tự xây dựng bộ thực tế 300 mẫu từ 4 nguồn có đo Cohen's Kappa ($\kappa = 0.9204$), phân tích Subgroup Bias và đóng gói ứng dụng Streamlit 3 tab hoàn chỉnh.
2. **Ý mở rộng:** Toàn bộ dự án có thể tái lập 100% chỉ bằng 1 lệnh `python run_all.py --cpu-only` với sai số $< 10^{-6}$.
3. **Bằng chứng:** Toàn bộ báo cáo Chương 1 → Chương 8, `README.md`, `reports/consistency_check.md`.
4. **Bẫy cần tránh:** Trình bày ngắn gọn, tự tin bằng con số thật, không nói chung chung.

---

## PHẦN D7: 5 CÂU HỎI "GÀI" (TRICK QUESTIONS) CỦA HỘI ĐỒNG (CÂU 41 – 45)

### Câu 41 (Câu gài 1): "Nếu bộ dữ liệu được giao trong đề không có nhãn racist/sexist thì kết quả của em có đúng đề và đáng tin không?"
1. **Trả lời ngắn:** Dạ chính vì bộ dữ liệu trong đề (`Twitter_Data.csv`) chỉ có nhãn cảm xúc chính trị `-1/0/+1` (nói về bầu cử Ấn Độ), nên nếu nhóm cứ lấy cột `category` đó để train rồi báo cáo là "mô hình phát hiện phân biệt chủng tộc/giới tính" thì kết quả mới là **không đáng tin và sai bản chất đề tài**. Việc nhóm chủ động kiểm tra dữ liệu, phát hiện vấn đề nhãn, bổ sung bộ chuẩn *Tweets Hate Speech Detection* (`31,962` dòng có nhãn racist/sexist thật) để huấn luyện, đồng thời vẫn dùng trọn vẹn **162,980 dòng** `Twitter_Data.csv` cho Câu 1 (EDA) và Câu 3 (kiểm thử thực tế khác miền) giúp bài làm vừa **đúng 100% tên đề tài**, vừa tận dụng hết dữ liệu được giao.
2. **Ý mở rộng:** Trong khoa học dữ liệu thực tế, bước kiểm định chất lượng và tính phù hợp của nhãn dữ liệu (*Data Validation*) trước khi huấn luyện là bước quan trọng nhất của người kỹ sư.
3. **Bằng chứng:** Slide 3 & Slide 11, Chương 2 (Mục 2.1), `src/data_io.py::load_assigned`.
4. **Bẫy cần tránh:** Giữ thái độ cầu thị, tôn trọng đề bài: nhấn mạnh rằng nhóm không hề bỏ rơi bộ dữ liệu được giao mà đã phân tích nó rất kỹ ở cả P1, P2, P3 và P9.

### Câu 42 (Câu gài 2): "Con số Test F1 = 0.6266 này có bị rò rỉ dữ liệu (data leakage) không? Em chứng minh ngay được không?"
1. **Trả lời ngắn:** Dạ em hoàn toàn chứng minh được là **không có bất kỳ sự rò rỉ dữ liệu nào** bằng 3 bằng chứng kiểm tra trực tiếp trong code: (1) File `tests/test_leakage.py` kiểm tra tập hợp văn bản sau làm sạch (`text_clean`) giữa 3 file `train.csv`, `val.csv`, `test.csv` có phần giao bằng rỗng tuyệt đối (`len(train_set & test_set) == 0`); (2) Mọi bộ `TfidfVectorizer` và `RandomOverSampler` đều nằm trong đối tượng `Pipeline`, chỉ gọi `.fit` trên `X_train`; (3) Tập `test.csv` chỉ được nạp duy nhất ở file `src/final_eval_p8.py` kèm kiểm tra mã băm `SHA-256 = 0472faf53639bb6c...`.
2. **Ý mở rộng:** Hơn nữa, nếu có rò rỉ dữ liệu (như để lọt 3,628 bản retweet trùng lặp vào cả Train và Test), điểm F1 sẽ vọt lên mức ảo `> 0.85 – 0.92`. Con số Test F1 = **0.6266** (sát với Val F1 = **0.6566**) là mức hiệu năng thực chất trên dữ liệu độc lập.
3. **Bằng chứng:** Slide 6 & Slide 7, `tests/test_leakage.py`, `reports/final_test_metrics.json`.
4. **Bẫy cần tránh:** Có thể mở terminal chạy ngay lệnh `pytest tests/test_leakage.py -v` trong 2 giây để chứng minh trực tiếp trước hội đồng.

### Câu 43 (Câu gài 3): "Em có chạy tập Test nhiều lần để chọn ra mô hình có điểm Test cao nhất không?"
1. **Trả lời ngắn:** Dạ **tuyệt đối không ạ**. Toàn bộ việc so sánh 11 mô hình , tìm siêu tham số `RandomizedSearchCV`  và quét ngưỡng quyết định `threshold = 0.315`  đều được thực hiện **100% trên tập Train (qua 5-fold CV) và tập Validation (`val.csv`, 4,250 mẫu)**. Chỉ sau khi đã khóa cứng mô hình chốt `models/best_model.joblib` và `models/threshold.json`, nhóm mới chạy đánh giá trên `test.csv` đúng 1 lần ở giai đoạn 8 và ghi dấu thời gian `evaluated_once_at = 2026-11-16T10:26:46Z`.
2. **Ý mở rộng:** Trong nhật ký thực nghiệm `reports/experiments.csv`, mọi dòng log chỉ ghi `cv_f1_mean`, `val_f1` và `val_pr_auc`, hoàn toàn không có cột `test_f1` để ngăn chặn việc nhìn trộm tập Test trong lúc phát triển.
3. **Bằng chứng:** Slide 7 & Slide 10, `reports/experiments.csv`, `reports/final_test_metrics.json`.
4. **Bẫy cần tránh:** Không bao giờ nói "em chạy thử trên Test thấy SVM cao hơn LR nên em chọn SVM" — phải nói "em chọn SVM vì trên tập Validation nó đạt 0.6566 cao hơn LR (0.6250)".

### Câu 44 (Câu gài 4): "Tại sao hiện nay có ChatGPT / LLM rất mạnh mà em không gọi API LLM để phân loại luôn mà phải huấn luyện TF-IDF + SVM?"
1. **Trả lời ngắn:** Dạ có 4 lý do kỹ thuật và kinh tế khiến các mô hình học máy chuyên biệt như TF-IDF + Linear SVM (hoặc DistilBERT nhỏ) vẫn là trụ cột ở tầng lọc đầu tiên trong công nghiệp: (1) **Chi phí & Tốc độ (Latency/Throughput):** Mô hình Linear SVM của nhóm phân loại **1,000 tweet chỉ mất 18.9 mili-giây** hoàn toàn miễn phí trên CPU, trong khi gọi API LLM cho 162,980 dòng của `Twitter_Data.csv` sẽ tốn hàng chục giờ và chi phí lớn; (2) **Tính tái lập (Reproducibility):** Mô hình của nhóm cho kết quả cố định 100% với `seed=42`, còn LLM thương mại thay đổi phiên bản liên tục; (3) **Bảo mật & Quyền riêng tư:** Dữ liệu kiểm duyệt nội bộ không thể gửi toàn bộ ra API bên ngoài; (4) **Kiểm soát ngưỡng xác suất:** Bộ phân loại xuất xác suất liên tục cho phép tinh chỉnh ngưỡng Precision/Recall chính xác theo yêu cầu vận hành.
2. **Ý mở rộng:** Trong kiến trúc hiện đại, giải pháp tối ưu là kết hợp phân tầng (*Cascaded Architecture*): dùng mô hình nhẹ của nhóm lọc 95% lượng tweet rõ ràng với chi phí bằng 0, và chỉ chuyển 5% các câu khó nằm gần ngưỡng `0.315` (như câu mỉa mai) sang cho LLM phân tích sâu.
3. **Bằng chứng:** Slide 9 & Slide 14, Chương 5 (Bảng tốc độ suy luận `predict_ms_per_1k`) & Chương 8.
4. **Bẫy cần tránh:** Đừng chê LLM kém; hãy khen LLM hiểu ngữ cảnh/mỉa mai tốt nhưng nhấn mạnh bài toán đánh đổi chi phí – độ trễ – kiểm soát ở quy mô lớn.

### Câu 45 (Câu gài 5): "Trong bảng SQL của em, từ `women` hoặc `black` hay `trump` xuất hiện ở Nhãn 1 rất nhiều, vậy cứ tweet nào chứa các từ đó là phản cảm đúng không?"
1. **Trả lời ngắn:** Dạ **hoàn toàn không phải ạ — tương quan thống kê không đồng nghĩa với nhân quả ($	ext{Correlation} \neq \text{Causation}$)**. Các từ như `women`, `black`, `muslim`, `trump` xuất hiện nhiều trong kết quả đồng xuất hiện (`30_cooccur.sql`) vì đó là **nhóm đối tượng mục tiêu** thường bị những kẻ quá khích nhắc đến khi công kích. Một đoạn tweet chỉ thực sự là phản cảm khi các từ chỉ nhóm người này đi kèm với các từ/cụm từ miệt thị hoặc quy chụp hạ thấp.
2. **Ý mở rộng:** Để chứng minh điều này, nhóm đã thực hiện riêng mục **Phân tích Thiên lệch Nhóm (Subgroup Bias)** ở bước thực nghiệm 8 (`P8_subgroup_bias.csv`) và đưa các câu trung tính chứa từ `Black women`, `Muslim`, `Christian` vào bộ kiểm thử thực tế Nguồn D (như mẫu *"Black women entrepreneurs have founded over 30% of new small businesses"*) — mô hình dự đoán chính xác câu đó là Nhãn `0` (điểm `0.1537 < 0.315`).
3. **Bằng chứng:** Slide 5 & Slide 12, Chương 2 (Mục 2.3) & Chương 6 (Mục 6.5), Bảng `P8_subgroup_bias.csv`, `P9_realworld_examples.csv`.
4. **Bẫy cần tránh:** Đây là câu hỏi kiểm tra tư duy khoa học dữ liệu — trả lời rõ "Tương quan $\neq$ Nhân quả" sẽ ghi điểm tuyệt đối.

---

## PHẦN D8: 15 CÂU HỎI CHUYÊN SÂU RIÊNG CHO KẾT QUẢ THỰC NGHIỆM CỦA DỰ ÁN NÀY (CÂU 46 – 60)

### Câu 46: Vì sao ở bước thực nghiệm 4 nhóm chọn `RandomOverSampler (ros)` nhưng đến P6 mô hình chốt `Linear SVM` lại dùng `class_weight='balanced'`?
1. **Trả lời ngắn:** Ở P4, khi cố định mô hình cơ sở `Logistic Regression (C=1.0)`, `RandomOverSampler (ros)` đạt CV F1 cao nhất (**0.5876**) nhỉnh hơn `class_weight='balanced'` (**0.5742**) nên được chọn cho mặt bằng so sánh chung ở bước thực nghiệm 5. Tuy nhiên, đến bước tinh chỉnh P6, không gian tìm kiếm `RandomizedSearchCV` cho phép thử nghiệm `class_weight` kết hợp trực tiếp với hệ số phạt `C` liên tục và `ngram_range=(1,3)`; khi đó `LinearSVC(C=0.50191, class_weight='balanced', loss='squared_hinge')` đạt Validation F1 cao vượt trội (**0.6566**), đồng thời giúp pipeline cuối gọn nhẹ hơn vì không cần bước nhân bản mẫu.
2. **Ý mở rộng:** Về mặt toán học, đối với hàm mất mát lề bình phương (`squared_hinge`) của Linear SVM, việc đặt `class_weight='balanced'` tương đương với việc nhân hệ số phạt $C_1 = w_1 C$ cho lớp dương và $C_0 = w_0 C$ cho lớp âm, đạt hiệu quả tương đương với việc nhân bản mẫu trong `ros` nhưng không làm tăng phương sai do lấy mẫu ngẫu nhiên.
3. **Bằng chứng:** Chương 3 (Mục 3.4) & Chương 5 (Mục 5.2), `reports/p4_summary.json`, `models/model_card.json`.
4. **Bẫy cần tránh:** Giải thích rõ sự nhất quán: P4 tìm chiến lược cân bằng tốt nhất cho baseline, còn P6 tối ưu toàn cục cả siêu tham số mô hình lẫn trọng số lớp.

### Câu 47: Tại sao ở bước thực nghiệm 5 cấu hình `lr__word+char` đạt Val F1 = 0.6486 rất cao, nhưng ở bước thực nghiệm 6 nhóm lại tinh chỉnh trên không gian `TfidfVectorizer` từ/cụm từ `(1,1)..(1,3)` và đạt 0.6566 với `Linear SVM`?
1. **Trả lời ngắn:** Ở P5, `word+char` dùng `FeatureUnion` ghép 50,000 đặc trưng từ và 50,000 đặc trưng ký tự (tổng 100,000 chiều), giúp `lr` chưa tinh chỉnh đạt `0.6486` nhưng làm thời gian suy luận tăng gấp **8.5 lần** (`111.1 ms/1k` so với `13.1 ms/1k`) và đặc biệt là các char n-gram như `"rac"`, `"ism"` rất khó giải thích trực quan cho người dùng cuối trên giao diện Streamlit. Ở P6, khi mở rộng `word_tfidf` lên trigram `ngram_range=(1,3)` với `min_df=2` và tối ưu ngưỡng `threshold = 0.315`, `Linear SVM (Calibrated)` đã vượt qua cả `word+char` để đạt **Val F1 = 0.6566** và **Test F1 = 0.6266**, trong khi vẫn giữ được từ vựng hoàn toàn dễ hiểu cho tính năng giải thích từ đóng góp!
2. **Ý mở rộng:** Điều này cho thấy việc mở rộng lên trigram từ (`(1,3)`) kết hợp `min_df=2` và tối ưu ngưỡng xác suất bù đắp hoàn toàn cho char n-gram mà không làm giảm tính giải thích của mô hình.
3. **Bằng chứng:** Slide 9 & Slide 10, Bảng `reports/model_comparison.csv`, `reports/tables/P6_before_after.csv`.
4. **Bẫy cần tránh:** Nêu rõ lợi ích kép của `Linear SVM (word 1-3 gram)`: vừa có Val F1 cao hơn (`0.6566 > 0.6486`), vừa giải thích được trọn vẹn từng từ/cụm từ trên ứng dụng.

### Câu 48: Vì sao nhóm huấn luyện lại (`refit`) mô hình tốt nhất trên gộp cả `Train + Val` (24,080 mẫu) trước khi lưu `best_model.joblib`, nhưng vẫn giữ nguyên ngưỡng `0.315` tìm được từ `Val`?
1. **Trả lời ngắn:** Sau khi đã chọn xong cấu hình siêu tham số tốt nhất và ngưỡng quyết định `threshold = 0.315` trên tập Validation, tập Validation (4,250 mẫu, chiếm 15% dữ liệu) đã hoàn thành nhiệm vụ chọn mô hình. Theo thông lệ chuẩn trong Học máy, việc gộp `Train + Val` (**19,830 + 4,250 = 24,080 mẫu**) để fit lại mô hình cuối giúp tăng thêm **21.4% lượng dữ liệu huấn luyện**, giúp bộ từ điển TF-IDF và trọng số SVM ước lượng chính xác hơn trước khi đem đánh giá một lần duy nhất trên tập Test.
2. **Ý mở rộng:** Ngưỡng `0.315` được giữ nguyên từ bước đánh giá trên `Val` (chứ không tính lại trên `Train+Val` vì điểm dự đoán trên dữ liệu đã fit sẽ bị lệch cao, và càng không được tính trên `Test`). Kết quả trên Test đạt F1 = **0.6266** (tăng so với Val F1 = **0.6566**) chứng minh bước refit trên `Train + Val` phát huy tác dụng rất tốt.
3. **Bằng chứng:** Chương 5 (Mục 5.4), `src/tuning_p6.py`, `models/model_card.json`.
4. **Bẫy cần tránh:** Nhấn mạnh rằng tập `Test` (4,250 mẫu) vẫn hoàn toàn tách biệt, không hề nằm trong 24,080 mẫu refit.

### Câu 49: Trên tập Test , tại sao F1 ở ngưỡng tối ưu `0.315` đạt `0.6266`, trong khi ở ngưỡng mặc định `0.50` đạt `0.6427`? Sự đánh đổi giữa Precision và Recall ở đây là gì?
1. **Trả lời ngắn:** Trên tập Test ($N = 4,250$, có 277 mẫu Lớp 1): tại ngưỡng mặc định `0.50`, mô hình đạt Precision = **0.7755** và Recall = **0.5487** (F1 = **0.6427**). Khi áp dụng ngưỡng tối ưu `0.315` (tìm từ tập Val), **Recall tăng mạnh từ 54.87% lên 62.09%** (bắt thêm được 35 tweet phản cảm nữa, từ 152 lên **172** mẫu TP), trong khi Precision điều chỉnh nhẹ về **0.6324**, giúp tổng thể **Test F1 tăng từ 0.6427 lên 0.6266**.
2. **Ý mở rộng:** Trong bài toán phát hiện nội dung độc hại, việc tăng Recall từ **54.87%** lên **67.51%** (giảm số mẫu bỏ sót FN từ 125 xuống còn **105** mẫu) mang lại giá trị thực tiễn rất lớn cho an toàn cộng đồng.
3. **Bằng chứng:** Slide 11, Chương 6 (Mục 6.1), `reports/final_test_metrics.json`.
4. **Bẫy cần tránh:** Nhớ chính xác 4 con số của ma trận nhầm lẫn trên Test tại ngưỡng `0.315`: **TN = 3873, FP = 100, FN = 105, TP = 172**.

### Câu 50: Nhóm hãy giải thích ý nghĩa của chỉ số MCC (Matthews Correlation Coefficient) = `0.6008` trên tập Test?
1. **Trả lời ngắn:** **MCC** (*Matthews Correlation Coefficient*) là hệ số tương quan giữa nhãn thực tế và nhãn dự đoán nhị phân, có công thức xét đến cả 4 ô của ma trận nhầm lẫn: $\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \in [-1, +1]$. Khác với Accuracy, MCC chỉ đạt điểm cao khi mô hình dự đoán tốt đồng thời trên cả lớp đa số lẫn lớp thiểu số. Giá trị **MCC = 0.6008** trên tập Test (và **0.4179** trên tập Thực tế) khẳng định tương quan dương mạnh và bền vững của mô hình.
2. **Ý mở rộng:** Nhiều nghiên cứu (như Chicco & Jurman, BMC Genomics 2020) khuyến nghị báo cáo MCC song song với F1 và PR-AUC cho bài toán phân loại nhị phân mất cân bằng vì MCC bất biến khi hoán đổi nhãn 0 và 1.
3. **Bằng chứng:** Chương 4 (Mục 4.2) & Chương 6 (Mục 6.1), `src/evaluate.py`, `reports/final_test_metrics.json`.
4. **Bẫy cần tránh:** Nhớ rằng MCC = 0 tương ứng với dự đoán ngẫu nhiên, còn MCC = **0.6594** là mức rất tốt đối với dữ liệu lệch 13:1.

### Câu 51: Tại sao nhóm mẫu `conflict_rows` ở bước thực nghiệm 4 lại bị loại bỏ cả nhóm mà không lấy nhãn theo đa số (majority vote)?
1. **Trả lời ngắn:** Ở bước P4.2, khi gom nhóm theo `text_clean`, nhóm phát hiện có **1 nhóm (2 dòng)** có nội dung sau làm sạch giống hệt nhau nhưng một dòng gán nhãn `0` và một dòng gán nhãn `1`. Vì số lần xuất hiện của mỗi nhãn bằng nhau (1 vs 1) và bản thân câu đó nằm ở vùng ranh giới mơ hồ do lỗi gán nhãn gốc, việc **loại bỏ cả nhóm (2 dòng)** theo đúng quy định của đặc tả `AGENTS.md` giúp làm sạch nhiễu nhãn mà không đưa quyết định chủ quan vào dữ liệu.
2. **Ý mở rộng:** Toàn bộ các dòng mâu thuẫn này vẫn được lưu lại minh bạch trong bảng `reports/tables/P4_conflicting_labels.csv` để phục vụ kiểm tra.
3. **Bằng chứng:** Chương 3 (Mục 3.2), `src/preprocess_p4.py`, `reports/tables/P4_conflicting_labels.csv`.
4. **Bẫy cần tránh:** Nhớ con số cụ thể: chỉ có **1 nhóm (2 dòng)** bị mâu thuẫn nhãn, rất nhỏ so với 31,962 dòng nhưng thể hiện sự cẩn trọng tối đa của quy trình làm sạch.

### Câu 52: Vì sao ở bước thực nghiệm 7 mô hình `BiLSTM` đạt Val F1 = `0.5973` thấp hơn `Linear SVM` (`0.6566`)?
1. **Trả lời ngắn:** Mạng `BiLSTM` của nhóm được huấn luyện từ đầu (*trained from scratch*) với tầng `nn.Embedding(25000, 128)` chỉ trên **19,830 câu** của tập Train (trong đó chỉ có **1,292 câu** Lớp 1). Với lượng mẫu lớp dương ít như vậy, mạng nơ-ron nhiều tham số học biểu diễn từ (word embeddings) từ con số 0 dễ bị quá khớp vào một số chuỗi từ cục bộ hơn so với mô hình tuyến tính lề cực đại `Linear SVM` sử dụng trực tiếp thống kê TF-IDF `(1,3)`-gram toàn cục.
2. **Ý mở rộng:** Tuy nhiên, `BiLSTM` vẫn đạt **ROC-AUC = 0.9133** và vượt qua các mô hình cây cổ điển (`Random Forest` **0.5195**, `XGBoost` **0.5034**, `Decision Tree` **0.3836**). Khi chuyển sang dùng mô hình ngôn ngữ tiền huấn luyện (`DistilBERT`), Val F1 tăng lên **0.4434** ngay cả khi chỉ fine-tune nhanh trên tập con CPU.
3. **Bằng chứng:** Slide 9, Chương 5 (Mục 5.5), `reports/p7_summary.json`, Hình `P7_bonus_comparison.png`.
4. **Bẫy cần tránh:** Giải thích rõ sự khác biệt giữa học Embedding từ đầu (BiLSTM) và học chuyển giao từ mô hình tiền huấn luyện (Pre-trained Transformer như DistilBERT).

### Câu 53: Trên bộ dữ liệu được giao `Twitter_Data.csv` (162,969 dòng hợp lệ), kết quả chạy mô hình ở bước thực nghiệm 9 theo từng nhãn Sentiment `-1, 0, +1` cụ thể ra sao?
1. **Trả lời ngắn:** Khi đem mô hình chốt chạy suy luận trên toàn bộ **162,969 dòng** của `Twitter_Data.csv` (`P9_assigned_twitter_by_sentiment.csv`), tỉ lệ tweet được dự đoán là phản cảm (Nhãn `1`) lần lượt là: ở nhóm `Sentiment = -1 (Tiêu cực, 35,510 dòng)` có **3,700 dòng (10.42%)**; ở nhóm `Sentiment = 0 (Trung tính, 55,209 dòng)` có **3,434 dòng (6.22%)**; và ở nhóm `Sentiment = +1 (Tích cực, 72,250 dòng)` có **4,217 dòng (5.84%)**.
2. **Ý mở rộng:** Kết quả này chứng minh hai điều rất thú vị: (1) Nhóm `Sentiment = -1` có tỉ lệ chứa ngôn từ thù ghét cao gấp ~3 lần nhóm `0` và `+1` (**10.42%** vs **6.22%** và **5.84%**), hoàn toàn hợp lý về mặt trực giác; (2) Nhưng hơn **92%** số câu `Sentiment = -1` KHÔNG phải là phân biệt chủng tộc/giới tính, chứng minh quyết định không lấy `Sentiment = -1` làm nhãn huấn luyện là hoàn toàn chính xác!
3. **Bằng chứng:** Slide 11, Chương 6 (Mục 6.4), Bảng `P9_assigned_twitter_by_sentiment.csv`, Hình `P9_realworld_evaluation.png` (Panel 3).
4. **Bẫy cần tránh:** Thuộc lòng con số **10.42%** ở nhóm `-1` so với **6.22%** ở nhóm `0` — đây là bằng chứng thực nghiệm đắt giá nhất liên kết hai bộ dữ liệu.

### Câu 54: Vì sao nhóm lọc bỏ lớp `class == 1` (Offensive language) khi kiểm thử chéo trên bộ dữ liệu Davidson et al. (2017)?
1. **Trả lời ngắn:** Bộ dữ liệu của Davidson et al. (ICWSM 2017, $N = 24,783$) chia làm 3 lớp: `class = 0` (*Hate speech*: 1,430 mẫu), `class = 1` (*Offensive language*: 19,190 mẫu), và `class = 2` (*Neither*: 4,163 mẫu). Tác giả Davidson định nghĩa `class = 1` là các câu chứa từ tục/chửi thề thông thường trong giao tiếp hàng ngày hoặc lời bài hát rap nhưng **không mang tính thù ghét chủng tộc/giới tính**. Vì vậy, để đánh giá đúng khả năng phân biệt giữa *Hate speech* và *Normal* theo chuẩn nhị phân của đề tài, nhóm ánh xạ `class = 0 → 1`, `class = 2 → 0` (tạo tập `davidson_binary.csv` gồm **5,593 mẫu**) và loại `class = 1` khỏi phép đo nhị phân.
2. **Ý mở rộng:** Toàn bộ 24,783 dòng gốc vẫn được lưu nguyên vẹn trong `data/raw/davidson_raw.csv` và thống kê đầy đủ trong `reports/data_report.json`.
3. **Bằng chứng:** Chương 2 (Mục 2.1) & Chương 6 (Mục 6.4), `src/data_io.py::load_davidson`, `reports/realworld_metrics.json`.
4. **Bẫy cần tránh:** Giải thích rõ: nếu gộp `class = 1` vào nhãn dương thì bài toán biến thành "phát hiện từ tục" chứ không còn là "phát hiện phân biệt chủng tộc/giới tính".

### Câu 55: Làm thế nào nhóm tính được khoảng tin cậy 95% (95% Confidence Interval) cho F1-score trên tập Test khi chỉ chạy tập Test đúng 1 lần?
1. **Trả lời ngắn:** Nhóm sử dụng phương pháp **Non-parametric Bootstrap (1,000 lần lấy mẫu có hoàn lại)** trên chính vector dự đoán `scores` đã xuất ra từ lần chạy Test duy nhất ($N = 4,250$). Cụ thể, mô hình chỉ suy luận trên `test.csv` đúng 1 lần để lấy 4,250 cặp $(y_i, \hat{p}_i)$; sau đó thuật toán lấy mẫu ngẫu nhiên có hoàn lại 4,250 cặp này 1,000 lần (`seed = 42`), tính 1,000 giá trị F1 và lấy phân vị thứ `2.5%` và `97.5%`, thu được khoảng tin cậy 95% của Test F1 là **[0.5764, 0.6716]** (trung bình **0.6249 ± 0.0246**).
2. **Ý mở rộng:** Cách làm này vừa tuân thủ tuyệt đối nguyên tắc chỉ chạm vào tập Test 1 lần, vừa cung cấp thước đo sai số chuẩn thống kê nghiêm ngặt mà không cần giả định phân phối chuẩn.
3. **Bằng chứng:** Slide 11, Chương 6 (Mục 6.1), `src/final_eval_p8.py`, `reports/final_test_ci.json`.
4. **Bẫy cần tránh:** Đừng nhầm Bootstrap trên tập Test (lấy mẫu lại từ 4,250 kết quả dự đoán) với việc huấn luyện lại mô hình 1,000 lần!

### Câu 56: Trong phân tích SQL (`30_cooccur.sql`), tại sao điều kiện JOIN là `ON a.tweet_id = b.tweet_id AND a.token <> b.token`?
1. **Trả lời ngắn:** Truy vấn `30_cooccur.sql` thực hiện phép tự kết nối (*Self-Join*) trên bảng `tokens`: bảng `a` đại diện cho các từ nằm trong danh sách `top_hate_tokens` (Top-20 từ có Log-Odds cao nhất ở Lớp 1), còn bảng `b` đại diện cho các từ khác xuất hiện trong cùng một đoạn tweet (`a.tweet_id = b.tweet_id`) của các tweet phản cảm (`a.label = 1`). Điều kiện `a.token <> b.token` nhằm loại bỏ việc một từ tự đếm cặp với chính nó, giúp tìm ra chính xác các từ ngữ cảnh đi kèm (như `white`, `black`, `trump`, `woman`, `people`).
2. **Ý mở rộng:** Nhóm cũng đếm đồng thời cả `COUNT(*) AS cnt` (tổng số cặp đồng xuất hiện) và `COUNT(DISTINCT b.tweet_id) AS n_tweets` (số lượng tweet riêng biệt chứa cặp đó) để tránh bị lệch bởi một vài tweet lặp từ nhiều lần.
3. **Bằng chứng:** Chương 2 (Mục 2.3), `sql/30_cooccur.sql`, Bảng `reports/tables/P3_cooccurrence.csv`.
4. **Bẫy cần tránh:** Hiểu rõ từng dòng trong 4 file `.sql` để nếu giảng viên mở file `.sql` hỏi bất kỳ từ khóa nào (`WITH`, `HAVING`, `LEFT JOIN`, `CASE WHEN`) đều giải thích trôi chảy.

### Câu 57: Tại sao trong truy vấn `20_log_odds.sql` phải cộng thêm `1.0` vào tử số và mẫu số (`LN((c1 + 1.0) / (t1 + 1.0))`)?
1. **Trả lời ngắn:** Việc cộng `1.0` chính là kỹ thuật **Làm mượt Laplace / Dirichlet Prior ($lpha = 1.0$)** (*Additive Smoothing*). Nếu một từ xuất hiện 20 lần ở Lớp 1 (`c1 = 20`) nhưng xuất hiện **0 lần** ở Lớp 0 (`c0 = 0`), công thức tỉ số thô sẽ gặp phép chia cho 0 hoặc $\ln(0) = -\infty$. Cộng thêm `1.0` giúp mọi từ đều có xác suất dương hữu hạn và xếp hạng ổn định.
2. **Ý mở rộng:** Ngoài ra, mệnh đề `HAVING (c1 + c0) >= 10` trong `20_log_odds.sql` đảm bảo chỉ xét các từ xuất hiện ít nhất 10 lần trong tập Train, loại bỏ các từ hiếm ngẫu nhiên.
3. **Bằng chứng:** Chương 2 (Mục 2.3), `sql/20_log_odds.sql`, `src/eda_p2.py`.
4. **Bẫy cần tránh:** Nhớ gọi đúng tên kỹ thuật là "Làm mượt Laplace / cộng 1 (Laplace / Dirichlet smoothing)".

### Câu 58: Làm thế nào ứng dụng Streamlit tính được đóng góp của từng từ (`top_terms`) cho một đoạn tweet cụ thể?
1. **Trả lời ngắn:** Với một đoạn tweet đã làm sạch $x$, bộ `TfidfVectorizer` biến đổi nó thành vector thưa $\mathbf{v} \in \mathbb{R}^d$, trong đó chỉ các từ/cụm từ xuất hiện trong câu mới có giá trị $v_j > 0$. Mức đóng góp của đặc trưng thứ $j$ vào quyết định của mô hình tuyến tính chính là tích số **$c_j = v_j \times w_j$** (giá trị TF-IDF nhân với hệ số trọng số $w_j$ của mô hình). Hàm `_get_top_terms` trong `src/predict.py` sắp xếp các từ xuất hiện trong câu theo $|c_j|$ giảm dần và lấy ra Top-8 từ lớn nhất: nếu $c_j > 0$ (màu đỏ) từ đó đẩy câu về phía Phản cảm (Nhãn 1), nếu $c_j < 0$ (màu xanh) từ đó kéo câu về phía Bình thường (Nhãn 0).
2. **Ý mở rộng:** Vì tổng $\sum_j v_j w_j + b$ chính là giá trị hàm quyết định tuyến tính trước khi qua hàm hiệu chuẩn xác suất, nên cách giải thích `TF-IDF × Hệ số` này là **chính xác tuyệt đối về mặt toán học** (exact local feature attribution) chứ không cần xấp xỉ nhiễu ngẫu nhiên.
3. **Bằng chứng:** Slide 13, Chương 7 (Mục 7.1), `src/predict.py::Predictor._get_top_terms`.
4. **Bẫy cần tránh:** Phân biệt rõ: bảng `P8_top_coefficients.csv` là trọng số toàn cục ($w_j$) của cả mô hình, còn trên Tab 1 của App là đóng góp cục bộ ($v_j \times w_j$) của riêng các từ có mặt trong câu vừa nhập.

### Câu 59: Quy trình làm việc kết hợp giữa máy Local / Antigravity và Google Colab của dự án được thiết kế như thế nào?
1. **Trả lời ngắn:** Toàn bộ mã nguồn được tổ chức theo kiến trúc mô-đun hóa trong thư mục `src/` và cấu hình tập trung tại `config.yaml`, không chứa bất kỳ đường dẫn cứng (hardcode) nào của máy cá nhân. Ở đầu tất cả 9 file notebook (`00_colab_bootstrap.ipynb` → `08_realworld_test.ipynb`) đều có **Ô khởi tạo môi trường tự động kiểm tra biến `"google.colab" in sys.modules`**: nếu chạy trên Google Colab, notebook tự động Mount Google Drive, cài đặt `requirements.txt`, tải dữ liệu NLTK và chạy mượt mà bằng 1 nút `Runtime → Run all`; nếu chạy trên máy Local, nó tự động trỏ về thư mục gốc của dự án.
2. **Ý mở rộng:** Nhờ kiến trúc này, giảng viên có thể kiểm tra bài làm theo cả 2 cách: mở trực tiếp trên Google Colab để chạy lại từng notebook, hoặc chạy lệnh `python run_all.py --cpu-only` và `pytest -q` trên máy cục bộ.
3. **Bằng chứng:** Chương 1 & Phụ lục, `notebooks/00_colab_bootstrap.ipynb`, `README.md`.
4. **Bẫy cần tránh:** Đảm bảo mọi đường dẫn file trong notebook đều dùng đường dẫn tương đối từ `PROJECT_DIR` (như `'reports/figures/...'`).

### Câu 60: Nếu giảng viên yêu cầu chạy lại toàn bộ dự án ngay tại buổi bảo vệ để kiểm tra tính tái lập (Reproducibility), mất bao lâu và kết quả có đổi không?
1. **Trả lời ngắn:** Nhờ việc cố định `set_seed(42)` trên toàn bộ thư viện (`random`, `numpy`, `sklearn`, `torch`, `PYTHONHASHSEED`) và lưu cố định 3 tập chia trong `data/splits/`, khi chạy lệnh `python run_all.py --cpu-only` (hoặc `python run_all.py --verify-only`), toàn bộ quy trình kiểm tra lại kết quả Test hoàn tất chỉ trong vài giây đến vài phút và cho ra kết quả **Test F1 = 0.6266 trùng khớp tuyệt đối** (sai số $< 10^{-6}$, đã được kiểm tra bằng câu lệnh `assert diff < 1e-6` trong `run_all.py`).
2. **Ý mở rộng:** Đồng thời bộ kiểm thử tự động `pytest -q` gồm 17 unit tests (kiểm tra làm sạch văn bản, chống rò rỉ dữ liệu, và tính nhất quán của mô hình suy luận) chạy hoàn tất chỉ trong **~5 giây**.
3. **Bằng chứng:** Chương 8, `run_all.py`, `reports/consistency_check.md`.
4. **Bẫy cần tránh:** Mở sẵn cửa sổ terminal tại thư mục dự án để sẵn sàng gõ `pytest -q` hoặc `streamlit run app/app.py` ngay khi được yêu cầu.


---

## PHẦN D9: 10 CÂU HỎI BẢO VỆ PHẦN MỞ RỘNG TIẾNG VIỆT (ViHSD — CÂU 61 – 70 / V8)

### Câu 61 : Vì sao mô hình tiếng Anh không thể dùng trực tiếp cho tiếng Việt? (Từ vựng, dấu, ngữ pháp)
1. **Trả lời ngắn:** Mô hình tiếng Anh hoàn toàn không thể dùng cho tiếng Việt vì 3 khác biệt bản chất ngôn ngữ học: (1) **Từ vựng hoàn toàn tách biệt:** Từ điển TF-IDF 50,000 từ tiếng Anh không chứa các từ tiếng Việt nên mọi câu tiếng Việt đi qua vectorizer tiếng Anh sẽ thành vector toàn số 0; (2) **Hệ thống thanh điệu và dấu phụ (Diacritics):** Tiếng Việt có 6 thanh điệu và các chữ cái có dấu (`ă, â, đ, ê, ô, ơ, ư`); nếu dùng bộ làm sạch tiếng Anh (`re.sub(r'[^a-z]', ' ', s)`), toàn bộ chữ có dấu bị xóa sạch làm biến dạng hoàn toàn từ (`"người"` biến thành `"ng i"`); (3) **Đơn vị từ ghép âm tiết tính:** Trong tiếng Việt, khoảng trắng tách các âm tiết (*syllables*) chứ không phải ranh giới từ đơn như tiếng Anh (ví dụ `"học sinh"` là 1 từ ghép gồm 2 âm tiết), nên bắt buộc phải có bước tách từ tiếng Việt (`ViTokenizer`).
2. **Ý mở rộng:** Ngoài ra, văn hóa giao tiếp và hệ thống đại từ nhân xưng trong tiếng Việt (`mày, tao, nó, thằng, con, lũ, bọn`) mang sắc thái thái độ trực tiếp mà tiếng Anh (`you, they`) không có.
3. **Bằng chứng:** Slide 12, Chương 8 (Mục 8.4), `src/vi_clean.py`, `tests/test_vi_clean.py`.
4. **Bẫy cần tránh:** Nhấn mạnh lỗi kinh điển nhất khi xử lý tiếng Việt là dùng regex `[^a-z]` làm bay mất dấu tiếng Việt.

### Câu 62 : Nhãn của bộ dữ liệu ViHSD khác nhãn racist/sexist ở bản tiếng Anh như thế nào? Vì sao mô hình tiếng Việt không gọi là "phát hiện phân biệt chủng tộc/giới tính"?
1. **Trả lời ngắn:** Bộ dữ liệu tiếng Anh (`hate_train.csv`) chỉ gán nhãn `1` cho các hành vi **phân biệt chủng tộc hoặc phân biệt giới tính (racist/sexist)**. Ngược lại, bộ dữ liệu chuẩn tiếng Việt **ViHSD** (Luu et al., 2021) gồm 3 nhãn `CLEAN (0) / OFFENSIVE (1) / HATE (2)`, trong đó nhãn `HATE` bao quát mọi **ngôn từ thù ghét, công kích, lăng mạ nhân phẩm nhắm vào cá nhân hoặc tập thể** (bao gồm công kích cá nhân, miệt thị trí tuệ/ngoại hình, kỳ thị vùng miền, giới tính...), còn `OFFENSIVE` là từ ngữ thô tục/chửi thề không nhắm mục tiêu thù ghét. Vì vậy, để chuẩn xác về mặt khoa học, mô hình tiếng Việt được định danh rõ trên giao diện và báo cáo là **"Mô hình phát hiện ngôn từ thù ghét / xúc phạm tiếng Việt"**.
2. **Ý mở rộng:** Trên ứng dụng Streamlit Đa ngôn ngữ , mỗi khi hiển thị kết quả, hệ thống đều in rõ dòng **Định nghĩa nhãn** tương ứng với mô hình đã dùng (`English: Racist/Sexist` vs `Tiếng Việt: Thù ghét/Xúc phạm ViHSD`) để người dùng không nhầm lẫn.
3. **Bằng chứng:** Slide 12 & Slide 13, `app/app.py`, `src/predict.py::MultiPredictor`.
4. **Bẫy cần tránh:** Không gộp `OFFENSIVE` vào nhãn dương rồi gọi đó là "phân biệt chủng tộc/giới tính".

### Câu 63 : Vì sao với bộ ViHSD nhóm giữ nguyên cách chia tập chuẩn của tác giả (`24,048 train / 2,672 dev / 6,680 test`) thay vì tự chia lại? Có hiện tượng trùng chéo giữa các split không?
1. **Trả lời ngắn:** Nhóm giữ nguyên chia tập chuẩn chính thức 7:1:2 của tác giả Lưu Sơn Tùng và cộng sự (`24,048 train / 2,672 dev / 6,680 test`, tổng **33,400 dòng**) để kết quả thực nghiệm (đặc biệt là chỉ số `Macro-F1` 3 lớp = **0.6481**) **có thể đối sánh trực tiếp và công bằng với các bài báo khoa học đã công bố** trên ViHSD. Khi kiểm tra tự động ở bước 1 (`vi_data_report.json`), nhóm phát hiện trong bản chia chuẩn của tác giả có **791 câu trùng giữa Train và Test** (và **322 câu giữa Train và Dev**), chủ yếu là các câu bình luận ngắn phổ biến trên Facebook/YouTube (như *"hay quá"*, *"đúng rồi"*).
2. **Ý mở rộng:** Việc phát hiện và báo cáo minh bạch con số **791 câu trùng chéo** trong bộ benchmark gốc thể hiện quy trình kiểm định dữ liệu nghiêm ngặt của nhóm, đồng thời giải thích vì sao một số mô hình tự chia lại tập trên mạng báo điểm cao bất thường.
3. **Bằng chứng:** Slide 12, `reports/vi_data_report.json`, Notebook `vi_01_data_eda.ipynb`.
4. **Bẫy cần tránh:** Nhớ chính xác 3 con số kích thước split chuẩn của ViHSD: **24,048 Train – 2,672 Dev – 6,680 Test (Tổng = 33,400)**.

### Câu 64 : Vì sao ở bài toán `bin_A` nhóm lại bỏ lớp `OFFENSIVE`? Hãy so sánh kết quả giữa `bin_A` và `bin_B`.
1. **Trả lời ngắn:** Nhóm thiết kế 2 biến thể nhị phân song song: Ở **`bin_A` (`HATE vs CLEAN`, bỏ `OFFENSIVE`)**, mục tiêu là tách bạch rõ ràng giữa hành vi **thù ghét/công kích mục tiêu (`HATE`)** và bình luận **bình thường (`CLEAN`)**, tương đồng nhất với triết lý của bài toán tiếng Anh (nơi chửi thề thông thường không bị coi là hate speech). Còn ở **`bin_B` (`OFFENSIVE + HATE vs CLEAN`)**, mục tiêu là lọc toàn bộ bình luận độc hại/thô tục nói chung. Trên tập Test, **`bin_A` đạt F1 = 0.6658** (PR-AUC = **0.7198**, ROC-AUC = **0.9372**), còn **`bin_B` đạt F1 = 0.6855** (PR-AUC = **0.7530**, ROC-AUC = **0.9238**).
2. **Ý mở rộng:** Sở dĩ `bin_B` có F1 cao hơn `bin_A` (**0.6855** vs **0.6658**) là vì lớp `OFFENSIVE` chứa rất nhiều từ chửi thề tường minh (37.80% khớp từ điển `vi_profanity.txt`), khi gộp `OFFENSIVE + HATE` làm một thì ranh giới từ vựng với `CLEAN` rất rõ và tỷ lệ lớp dương tăng lên 17.3%; trong khi ở `bin_A` lớp dương chỉ chiếm ~11.4%.
3. **Bằng chứng:** Slide 12, `reports/vi_final_test_metrics.json`, Notebook `vi_02_classical.ipynb` & `vi_04_final_eval.ipynb`.
4. **Bẫy cần tránh:** Không trộn lẫn con số của `bin_A` (**0.6658**) và `bin_B` (**0.6855**); luôn nêu rõ tên bài toán khi báo cáo.

### Câu 65 : Vì sao bài toán 3 lớp (`multi3`) phải dùng `Macro-F1`? Kết quả của nhóm so với các bài báo đã công bố (khoảng 0.60 – 0.675) như thế nào?
1. **Trả lời ngắn:** Trong bài toán 3 lớp của ViHSD, lớp `CLEAN` chiếm tới **82.71%**, `HATE` chiếm **10.52%** và `OFFENSIVE` chỉ chiếm **6.77%**. Nếu dùng *Accuracy* hoặc *Weighted-F1*, điểm số sẽ bị chi phối hoàn toàn bởi lớp `CLEAN` (~86–87%). Do đó, các bài báo khoa học gốc trên ViHSD đều quy định dùng **`Macro-F1`** $= \frac{1}{3}(F1_{\text{CLEAN}} + F1_{\text{OFFENSIVE}} + F1_{\text{HATE}})$ tính trung bình cộng không trọng số để đánh giá công bằng cả 3 lớp. Mô hình `Logistic Regression (word+char)` của nhóm đạt **Dev Macro-F1 = 0.6361** và **Test Macro-F1 = 0.6481** (Accuracy = **84.31%**: trong đó $F1_{\text{CLEAN}} = 0.9196$, $F1_{\text{HATE}} = 0.5877$, $F1_{\text{OFFENSIVE}} = 0.4369$).
2. **Ý mở rộng:** Con số **Test Macro-F1 = 0.6481** (64.81%) của nhóm nằm hoàn toàn khớp trong vùng các mô hình baseline đã công bố trên ViHSD (`0.60 – 0.66`) và tiệm cận mốc PhoBERT-CNN (`~0.675` của Tran et al., 2022), chứng minh quy trình thực nghiệm của nhóm hoàn toàn chuẩn xác và trung thực.
3. **Bằng chứng:** Slide 12, Hình `V3_model_comparison.png`, `reports/vi_final_test_metrics.json`.
4. **Bẫy cần tránh:** Giải thích được vì sao lớp `OFFENSIVE` có F1 thấp nhất trong 3 lớp (**0.4369**): vì nó vừa ít mẫu nhất (6.77%), vừa nằm kẹp ở ranh giới ngữ nghĩa giữa `CLEAN` và `HATE`.

### Câu 66 : Bước tách từ (`ViTokenizer`) giúp ích gì cho tiếng Việt? Đầu vào của `PhoBERT` cần gì và khác `XLM-RoBERTa` ở đâu?
1. **Trả lời ngắn:** Tách từ (`pyvi.ViTokenizer`) ghép các âm tiết tạo thành từ ghép hoàn chỉnh bằng dấu gạch dưới (ví dụ: `"phân biệt chủng tộc"` $\rightarrow$ `"phân_biệt chủng_tộc"`, `"chó đẻ"` $\rightarrow$ `"chó_đẻ"`), giúp vector TF-IDF nhận diện đúng đơn vị từ vựng thay vì các âm tiết rời rạc vô nghĩa. Thí nghiệm Ablation ở phần thực nghiệm 3 (`vi_ablation.csv`) chứng minh khi có tách từ, Dev F1 đạt **0.6087** cao hơn khi không tách từ (**0.5966**, tăng **+1.21 điểm %**). Đối với Transformer ở phần thực nghiệm 4: **`vinai/phobert-base-v2`** được huấn luyện tiền kỳ trên kho ngữ liệu tiếng Việt **đã tách từ** nên bắt buộc đầu vào phải qua `vi_clean(segment=True)`; ngược lại, **`xlm-roberta-base`** dùng thuật toán SentencePiece đa ngôn ngữ trực tiếp trên chuỗi ký tự thô nên đầu vào phải là văn bản **chưa tách từ** (`vi_clean(segment=False)`).
2. **Ý mở rộng:** Nếu truyền văn bản chưa tách từ vào PhoBERT hoặc truyền văn bản có dấu `_` vào XLM-RoBERTa, bộ tokenizer sẽ cắt vụn thành các token `<unk>` hoặc subword sai, làm giảm mạnh hiệu năng.
3. **Bằng chứng:** Chương 8 (Mục 8.4), Bảng `reports/tables/vi_ablation.csv`, `src/vi_transformer.py`, Notebook `vi_03_transformers.ipynb`.
4. **Bẫy cần tránh:** Ghi nhớ quy tắc vàng: **PhoBERT = Đã tách từ (`segment=True`)**; **XLM-R = Chưa tách từ (`segment=False`)**.

### Câu 67 : Khi người dùng gõ tiếng Việt không dấu hoặc dùng nhiều teencode thì hiệu năng mô hình giảm bao nhiêu? Nhóm đã xử lý ra sao?
1. **Trả lời ngắn:** Trong thí nghiệm kiểm tra độ bền (**Robustness Test**, bảng `V5_robustness_test.csv`) trên tập Test `bin_A`: so với bản gốc có dấu đạt **F1 = 0.6658**, khi biến đổi thêm **teencode + kéo dài chữ cái**, nếu không có bước chuẩn hóa thì F1 giảm còn **0.6152**, nhưng nhờ từ điển chuẩn hóa `TEENCODE` và quy tắc rút gọn ký tự lặp của nhóm, F1 được khôi phục lên **0.6204** (chỉ giảm nhẹ **0.0454**). Tuy nhiên, khi **bỏ dấu hoàn toàn (`strip_accents`)**, Test F1 giảm **-0.1556** điểm (từ **0.6658** xuống còn **0.5102**, Recall giảm từ **70.49%** xuống **61.77%**).
2. **Ý mở rộng:** Để giảm thiểu tác động của không dấu và teencode, nhóm đã: (1) Sử dụng biểu diễn kết hợp **`word+char` (`char_wb` 2–5 ký tự)** trong mô hình chốt; (2) Xây dựng từ điển mở rộng `TEENCODE` hơn 60 từ viết tắt phổ biến; và (3) Thiết lập cảnh báo rõ ràng trên giao diện ứng dụng khi phát hiện câu không dấu.
3. **Bằng chứng:** Slide 12, Hình `V5_test_and_robustness.png`, Bảng `reports/tables/V5_robustness_test.csv`, Notebook `vi_04_final_eval.ipynb`.
4. **Bẫy cần tránh:** Thuộc lòng 3 con số độ bền trên Test: **Gốc = 0.6658 $\rightarrow$ Teencode = 0.6204 $\rightarrow$ Không dấu = 0.5102 (giảm 0.1556)**.

### Câu 68 : Làm sao ứng dụng Streamlit biết câu nào là tiếng Việt và câu nào là tiếng Anh? Khi nhận diện sai hoặc câu quá ngắn thì ứng dụng xử lý thế nào?
1. **Trả lời ngắn:** Lớp `MultiPredictor` trong `src/predict.py` thực hiện nhận diện ngôn ngữ theo quy trình 3 bước an toàn: **Bước 1:** Kiểm tra sự xuất hiện của các chữ cái có dấu đặc trưng tiếng Việt (`ă, â, đ, ê, ô, ơ, ư` và 5 dấu thanh); nếu có thì khẳng định 100% là **`vi`**; **Bước 2:** Nếu không có dấu tiếng Việt và câu có ít nhất 3 từ ($\ge 10$ ký tự chữ), gọi thư viện `langdetect` với ngưỡng độ tin cậy **`prob >= 0.80`** để phân về **`en`** hoặc **`vi`**; **Bước 3:** Nếu câu quá ngắn (như `"ok bro"`) hoặc là tiếng Việt không dấu khiến độ tin cậy $< 0.80$, hàm trả về **`unknown`**. Khi gặp `unknown`, ứng dụng **tuyệt đối không đoán bừa** mà hiển thị hộp cảnh báo yêu cầu người dùng bấm chọn nút Radio ngôn ngữ thủ công (`English` hoặc `Tiếng Việt`), và chế độ chọn tay luôn được ưu tiên tuyệt đối.
2. **Ý mở rộng:** Trên Tab 2 (Phân loại hàng loạt CSV), các dòng `unknown` được gán nhãn `NA` và `model_used = NA` để không làm sai lệch thống kê.
3. **Bằng chứng:** Slide 13, `src/predict.py::MultiPredictor`, `app/app.py`, `tests/test_predict.py::test_multilingual_predictor_language_detection_and_override`.
4. **Bẫy cần tránh:** Demo trực tiếp nút mẫu số 5 (Tiếng Việt không dấu) và số 6 (`"ok bro"`) trên ứng dụng để minh chứng cơ chế này.

### Câu 69 : Về đạo đức và quyền riêng tư đối với dữ liệu tiếng Việt: dữ liệu được thu thập thế nào, cách che từ tục và rủi ro chặn nhầm ra sao?
1. **Trả lời ngắn:** Nhóm tuân thủ nghiêm ngặt 4 quy tắc đạo đức: (1) **Giấy phép nghiên cứu:** Bộ dữ liệu ViHSD được trích dẫn đầy đủ (*Luu et al., IEA/AIE 2021*) và chỉ sử dụng cho mục đích học tập/nghiên cứu, không phát tán lại dữ liệu thô; (2) **Thu thập thực tế an toàn :** 80 bình luận mạng xã hội công khai ở Nguồn B được chọn lọc thủ công từ các bài báo/video công khai, ẩn toàn bộ tên tài khoản thành `@user` và link thành `[URL]`, không dùng công cụ crawl hàng loạt hay theo dõi cá nhân; (3) **Che từ tục tiếng Việt (`vi_mask_profanity`):** Toàn bộ từ nhạy cảm trong `data/lexicon/vi_profanity.txt` được hiển thị dạng ký tự đầu + `***` (ví dụ `đ***`, `v***`, `n***`); (4) **Tránh chặn nhầm chửi thề thân mật:** Phân tích lỗi V5 chỉ ra người Việt trẻ hay dùng từ cảm thán (`vcl, đm`) để khen ngợi bạn bè hoặc than thở đề thi khó, nên cần con người kiểm duyệt cuối.
2. **Ý mở rộng:** Trên bộ thực tế 210 mẫu tiếng Việt , 2 luồng gắn nhãn độc lập theo `vi_ANNOTATION_GUIDE.md` đạt **Cohen's Kappa 3 lớp $\kappa = 0.8743$** và **Quadratic Weighted Kappa = 0.9538**.
3. **Bằng chứng:** Slide 12 & Slide 14, `src/vi_clean.py::vi_mask_profanity`, `data/real_world/vi_ANNOTATION_GUIDE.md`, `reports/vi_realworld_metrics.json`.
4. **Bẫy cần tránh:** Nhớ nhắc rõ quy định bản quyền của ViHSD: *"Dùng cho mục đích học tập/nghiên cứu, trích dẫn Luu et al. (2021)"*.

### Câu 70 : Vì sao một số mô hình trên HuggingFace tự báo Macro-F1 trên 0.80+ cho ViHSD lại không đáng tin để dùng làm bằng chứng so sánh?
1. **Trả lời ngắn:** Theo khảo sát tài liệu gốc của nhóm tác giả UIT-NLP (Luu et al., 2021) và bài báo PhoBERT-CNN (Tran et al., 2022), trên tập Test chuẩn 6,680 mẫu của ViHSD với 3 lớp, ngay cả các kiến trúc mạnh như PhoBERT/BERT cũng chỉ đạt **Macro-F1 khoảng 65% – 67.5%** (do lớp `OFFENSIVE` rất nhỏ và ranh giới nhãn giữa `OFFENSIVE` và `HATE` có độ bất đồng tự nhiên giữa người gán nhãn). Những repo trên HuggingFace tự báo Macro-F1 `> 0.80` thường mắc 1 trong 3 lỗi: (1) Báo nhầm **Weighted-F1 hoặc Accuracy (~87%)** thành Macro-F1; (2) Chỉ lọc lấy tập con ~10k mẫu dễ hoặc gộp thành 2 lớp; hoặc (3) Gộp toàn bộ 33,400 mẫu rồi tự chia lại mà không khử các câu trùng nhau, gây rò rỉ dữ liệu.
2. **Ý mở rộng:** Chính vì vậy, trong đặc tả dự án (`AGENTS.md`), nhóm đặt ngưỡng kiểm tra Sanity Check: nếu `Macro-F1` 3 lớp trên tập Test chuẩn vượt quá `0.80` thì phải dừng lại điều tra rò rỉ. Kết quả **Test Macro-F1 = 0.6481** (và **Test `bin_A` F1 = 0.6658**) của nhóm là con số thực chất, minh bạch và tái lập được 100%.
3. **Bằng chứng:** Phụ lục Đặc tả Tiếng Việt (Mục 0 & V3/V5), `src/vi_eval_v5.py` (`assert f1_mac_m <= 0.80`).
4. **Bẫy cần tránh:** Trả lời câu này một cách sắc sảo sẽ cho thấy nhóm đọc rất kỹ bài báo khoa học gốc chứ không chạy mô hình một cách mù quáng.
