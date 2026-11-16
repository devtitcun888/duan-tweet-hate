# HƯỚNG DẪN GẮN NHÃN BÌNH LUẬN TIẾNG VIỆT THỰC TẾ (VI_ANNOTATION_GUIDE — V6)

## 1. Mục đích & Lưu ý Đạo đức
Hướng dẫn này dùng cho 2 người gắn nhãn độc lập (`label_a`, `label_b`) trên bộ dữ liệu thực tế tiếng Việt `data/real_world/vi_to_label.csv` ($N = 210$ bình luận từ 3 nguồn: A = mẫu phân tầng từ tập test ViHSD, B = bình luận mạng xã hội công khai đã ẩn danh, C = câu hiểm hóc tự soạn).
- **Khác biệt định nghĩa nhãn so với bản tiếng Anh:** Mô hình tiếng Việt được huấn luyện theo chuẩn **ViHSD (Luu et al., 2021)** nhằm phát hiện **ngôn từ thù ghét / xúc phạm tiếng Việt nói chung** (nhắm vào cá nhân, nhóm, vùng miền, giới tính...), không giới hạn ở racist/sexist như bộ tiếng Anh.
- Toàn bộ tên người dùng đã được thay bằng `@user` và đường dẫn thay bằng `[URL]`.

## 2. Định nghĩa 3 mức nhãn (CLEAN / OFFENSIVE / HATE) theo chuẩn ViHSD

### 1) `CLEAN` (Mã số `0` — Bình thường / Sạch)
- Bình luận trao đổi thông tin, khen ngợi, góp ý văn minh, tin tức, hoặc tranh luận xã hội không chứa từ ngữ tục tĩu hay công kích cá nhân/nhóm.
- Bao gồm cả **câu phản đối ngôn từ thù ghét (Counter-speech)** (ví dụ: *"Chúng ta cần lên án hành vi phân biệt vùng miền trên mạng"*).
- **Quy tắc mặc định:** Khi không chắc chắn giữa `CLEAN` và `OFFENSIVE`, chọn `CLEAN` và ghi chú vào cột `note`.

### 2) `OFFENSIVE` (Mã số `1` — Xúc phạm / Thô tục nhưng không thù ghét mục tiêu)
- Bình luận có sử dụng từ ngữ thô tục, chửi thề cảm thán (ví dụ: `v***`, `đ***`, `n***`) hoặc lời lẽ thiếu lịch sự nhưng **không nhắm vào việc hạ thấp nhân phẩm, miệt thị hay kích động thù ghét một cá nhân/nhóm cụ thể** (ví dụ: *"Đ*** hôm nay đề thi khó v*** làm tao toát mồ hôi"*).

### 3) `HATE` (Mã số `2` — Ngôn từ thù ghét / Công kích cá nhân hoặc nhóm)
- Bình luận tấn công trực diện, lăng mạ, miệt thị nhân phẩm, ngoại hình, trí tuệ, vùng miền, giới tính, tôn giáo của một cá nhân hoặc một tập thể, hoặc đe dọa bạo lực (ví dụ: *"Lũ s*** v*** vô học này xứng đáng bị đuổi khỏi xã hội"*).

## 3. Ánh xạ sang Bài toán Nhị phân (`bin_A` và `bin_B`)
- **`bin_A` (HATE vs CLEAN, bỏ OFFENSIVE):** `HATE -> 1`, `CLEAN -> 0`.
- **`bin_B` (OFFENSIVE + HATE vs CLEAN):** `OFFENSIVE, HATE -> 1`, `CLEAN -> 0`.
