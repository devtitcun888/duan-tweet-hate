# HƯỚNG DẪN GẮN NHÃN BỘ DỮ LIỆU THỰC TẾ (ANNOTATION GUIDE — P9.2)

## 1. Mục đích
Tài liệu này quy định tiêu chí gắn nhãn độc lập cho 2 người gắn nhãn (`label_a` và `label_b`) trên bộ dữ liệu thực tế `data/real_world/to_label.csv` (gồm 300 mẫu từ 4 nguồn: A = `hate_unlabeled_test`, B = `Twitter_Data`, C = `Reddit_Data`, D = `adversarial_curated`).
- **Nguyên tắc chống thiên kiến:** File `to_label.csv` đã ẩn danh `@user` và URL, và **tuyệt đối không đính kèm điểm xác suất của mô hình** để tránh làm lệch quyết định của người gắn nhãn.

## 2. Định nghĩa nhãn nhị phân (0 / 1)

### Nhãn `1` — Phản cảm: Phân biệt chủng tộc / Phân biệt giới tính / Thù ghét nhóm (Racist / Sexist / Group Hate Speech)
Gán nhãn `1` khi đoạn tweet/bình luận có nội dung **kỳ thị, thù ghét, miệt thị, xúc phạm nhân phẩm hoặc hạ thấp** một cá nhân hoặc nhóm người dựa trên:
- **Chủng tộc, sắc tộc, màu da, quốc tịch** (ví dụ: dùng từ miệt thị chủng tộc, quy chụp người nhập cư/người da màu là thấp kém hoặc tội phạm, tư tưởng thượng đẳng da trắng).
- **Giới tính, bản dạng giới, xu hướng tính dục** (ví dụ: dùng từ miệt thị phụ nữ, định kiến hạ thấp năng lực phụ nữ, cổ xúy bạo lực hoặc quấy rối giới tính).
- **Tôn giáo** (ví dụ: quy chụp thù địch nhắm vào cộng đồng Hồi giáo, Do Thái, Công giáo...).

### Nhãn `0` — Bình thường / Không phải phân biệt chủng tộc hay giới tính (Non-Hate)
Gán nhãn `0` cho tất cả các trường hợp còn lại, bao gồm:
1. **Giao tiếp đời thường, tin tức, thể thao, giải trí.**
2. **Tranh luận chính trị gay gắt** (ví dụ: chỉ trích chính sách của chính phủ, đảng phái, lãnh đạo như Modi, Trump, Obama...) nhưng **không** miệt thị chủng tộc/giới tính/tôn giáo.
3. **Chửi thề hoặc dùng từ tục cảm thán chung** (ví dụ: *"This traffic is f***ing terrible"*) không nhắm vào nhóm bảo vệ.
4. **Phản đối phân biệt đối xử / Counter-speech / Nhắc đến từ nhạy cảm theo nghĩa trung tính** (ví dụ: *"We must fight against racism and sexism in sports"*, hoặc *"Black women scientists made history today"*).

## 3. Quy tắc xử lý trường hợp khó
- **Khi không chắc chắn (ambiguous):** Mặc định chọn `0` và ghi rõ lý do vào cột `note`.
- **Sau khi 2 người gắn nhãn độc lập:** Tính hệ số đồng thuận **Cohen's Kappa ($\kappa$)** giữa `label_a` và `label_b`. Các mẫu bất đồng (`label_a != label_b`) được thảo luận đối chiếu lại với mục 2 để chốt `label_final`.
