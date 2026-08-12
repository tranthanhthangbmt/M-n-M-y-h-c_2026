# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 1 - Học máy (Chuẩn Đại học)

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **~55 Frames (Slides)** (Nằm trong khoảng 40 - 60 slides theo yêu cầu).  
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_01.md`).  
**Tích hợp hình ảnh:** Phân bổ toàn bộ 26 hình ảnh (`Hinh_1-1` đến `Hinh_1-26` trong thư mục `Figures/CH01/`) vào các slide tương ứng.

---

## 1. Bố cục Phân chương Tiết học (Sections & TOC)

Bộ slide Chương 1 được chia thành **3 Tiết học (3 Sections)** rõ ràng với tổng cộng 45 Frames:

### Tiết 1 (14 Frames): Khái quát \& Ứng dụng của Học máy
- **Mục tiêu:** Hiểu rõ định nghĩa học máy, lý do áp dụng và các ứng dụng thực tiễn.
- **Nội dung các slide thực tế:**
  1. Trang bìa (`\titlepage`)
  2. Mục lục nội dung chương (`\tableofcontents`)
  3. Mục tiêu bài học
  4. Slide chuyển ý (Khái quát & Ứng dụng)
  5. Đặt vấn đề: Bức tranh tổng quan
  6. Học máy là gì?
  7. Cách tiếp cận truyền thống vs. Học máy
  8. Cách tiếp cận bằng Học máy
  9. Tại sao sử dụng học máy?
  10. Tự động thích ứng với thay đổi
  11. Học máy giúp con người học hỏi
  12. Ví dụ về các ứng dụng (1/2)
  13. Ví dụ về các ứng dụng (2/2)
  14. Slide chuyển ý (Phân loại Hệ thống Học máy)

### Tiết 2 (15 Frames): Phân loại Hệ thống Học máy
- **Mục tiêu:** Nắm vững các mô hình: Có giám sát, Không giám sát, Học tăng cường, Batch vs Online.
- **Nội dung các slide thực tế:**
  15. Tiêu chí phân loại hệ thống học máy
  16. Học có giám sát (Supervised Learning)
  17. Phân loại (Classification) vs Hồi quy (Regression)
  18. Học không giám sát (Unsupervised Learning)
  19. Phân cụm (Clustering)
  20. Giảm chiều dữ liệu (Dimensionality Reduction)
  21. Phát hiện bất thường (Anomaly Detection) \& Luật kết hợp
  22. Minh họa: Bất thường \& Luật kết hợp
  23. Học bán giám sát (Semi-supervised Learning)
  24. Học tự giám sát (Self-supervised Learning)
  25. Học tăng cường (Reinforcement Learning)
  26. Học theo lô vs Học trực tuyến (Batch vs Online)
  27. Minh họa: Học trực tuyến
  28. Học dựa trên thực thể (Instance-based Learning)
  29. Học dựa trên mô hình (Model-based Learning)
  30. Minh họa: Học dựa trên mô hình (Biểu đồ phân tán & Mô hình tuyến tính)

### Tiết 3 (13 Frames): Thách thức, Kiểm thử \& Xác thực
- **Mục tiêu:** Nhận diện các rủi ro về dữ liệu/mô hình và phương pháp kiểm thử (Train/Test/Validation).
- **Nội dung các slide thực tế:**
  30. Slide chuyển ý (Thách thức, Kiểm thử & Xác thực)
  31. Những thách thức chính của học máy
  32. Dữ liệu huấn luyện không đủ (Dữ liệu tồi)
  33. Dữ liệu huấn luyện không đại diện
  34. Dữ liệu chất lượng kém
  35. Đặc trưng không liên quan
  36. Overfitting (Khớp quá mức)
  37. Điều chuẩn (Regularization)
  38. Underfitting (Chưa khớp)
  39. Kiểm thử và xác thực (Testing and Validating)
  40. Tinh chỉnh mô hình và Tập xác thực (Validation Set)
  41. Sự không khớp dữ liệu (Data Mismatch) & Tập train-dev
  42. Không có bữa trưa miễn phí (No Free Lunch)
  43. Tổng kết Chương 1

---

## 2. Kiến trúc Kỹ thuật (`v3_generate_slides.py`)
- Python Script sẽ được viết lại hoàn toàn để map **100% nội dung chữ và hình** vào cấu trúc 55 slide cố định như trên, tránh việc sinh tự động tràn lan làm slide quá dài (169 trang như bản trước).
- Sẽ sử dụng `\begin{columns}` để trình bày song song chữ & hình (giống `slide_AIAcc_Day01.md`), giúp slide đẹp hơn, tránh việc hình ảnh chiếm trọn 1 slide đơn điệu.

> [!IMPORTANT]  
> Xin thầy xác nhận **Proceed** để tôi tiến hành:
> 1. Lưu lại cấu trúc này vào file `planSlide_ML_Chap01.md`.
> 2. Viết kịch bản sinh slide thế hệ mới để đạt chuẩn ~55 frames và có bố cục chữ - hình song song hoàn hảo.

---

## 3. Nhật ký thực hiện (Hoàn thành)

- **Đã hoàn thành:** Kịch bản python (`v3_generate_slides.py`) đã được viết lại, phân đoạn thành slide cân đối.
- **Tối ưu hình ảnh:** 
  - Tích hợp 25 bức ảnh của Chương 1 bằng môi trường `\begin{columns}` hiển thị văn bản 1 bên, hình minh họa 1 bên. (*Ghi chú: Ảnh 1-21 bị khuyết trong thư mục gốc nên đã tự động lược bỏ*).
  - Tách các slide chứa 2 ảnh minh họa ("Phát hiện bất thường & Luật kết hợp" và "Học theo lô vs Học trực tuyến") thành 2 slide riêng biệt (1 slide chữ, 1 slide chứa 2 ảnh song song với kích thước lớn) để sinh viên dễ quan sát hơn.
  - Xóa bỏ tiền tố "Figure: " mặc định ở tiêu đề ảnh bằng cấu hình `\setbeamertemplate{caption}{\raggedright\insertcaption\par}`.
- **Biên dịch thành công:** Đã xuất PDF 42 trang hoàn chỉnh (`Slide_ML_Chap01.pdf`).
- **Rà soát & Nâng cấp:** Đã đối chiếu với nội dung trang web (`chuong_01.md`) và phát hiện thiếu một số phần: 
  - Đã thêm nội dung "Học tự giám sát" (Self-supervised Learning).
  - Đã thêm nội dung "Sự không khớp dữ liệu" (Data Mismatch) và "Tập train-dev".
  - Sửa lại sự sai lệch hình ảnh giữa markdown và slide (Sửa Hình 1-11, Hình 1-12, Hình 1-17, Hình 1-18, Hình 1-26). Tổng số slide cập nhật là ~45 slide.
