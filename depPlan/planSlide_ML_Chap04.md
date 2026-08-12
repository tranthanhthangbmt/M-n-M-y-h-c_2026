# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 4 - Huấn luyện Mô hình

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **~60 Frames (Slides)** (Phù hợp với lượng lớn 25 hình ảnh và kiến thức).  
**Nguồn nội dung chữ:** Trích xuất từ tài liệu trang web của chương (`machineLearningWeb/docs/chuong_04.md`).  
**Tích hợp hình ảnh:** Phân bổ toàn bộ 25 hình ảnh (`Hinh_4-1` đến `Hinh_4-25` trong thư mục `Figures/CH04/`) vào các slide.

---

## 1. Bố cục Phân chương Tiết học (Sections & TOC)

Bộ slide Chương 4 là một chương rất dài và quan trọng, do đó được chia thành **4 Tiết học (4 Sections)** rõ ràng với tổng cộng 59 Frames, đặc biệt kế thừa triệt để quy tắc **Tách slide hình ảnh**:

### Tiết 1 (11 Frames): Hồi quy tuyến tính & Phương trình chuẩn tắc
- **Mục tiêu:** Hiểu mô hình toán học của Hồi quy tuyến tính, Hàm chi phí MSE và cách giải bằng Phương trình chuẩn tắc / SVD.
- **Nội dung:**
  1. Trang bìa (`\titlepage`)
  2. Mục lục nội dung chương (`\tableofcontents`)
  3. Mục tiêu bài học
  4. Slide chuyển ý (Hồi quy tuyến tính)
  5. Hồi quy tuyến tính (Mô hình toán học dạng vector)
  6. Hàm chi phí MSE
  7. Phương trình chuẩn tắc
  8. Minh họa: Tập dữ liệu tuyến tính ngẫu nhiên (Hình 4-1)
  9. Minh họa: Dự đoán của Hồi quy tuyến tính (Hình 4-2)
  10. Thực hiện với Scikit-Learn và SVD (Singular Value Decomposition)
  11. Độ phức tạp tính toán

### Tiết 2 (17 Frames): Hạ Gradient (Gradient Descent)
- **Mục tiêu:** Nắm vững nguyên lý và phân biệt 3 loại Gradient Descent (Batch, Stochastic, Mini-batch).
- **Nội dung:**
  12. Slide chuyển ý (Gradient Descent)
  13. Gradient Descent là gì?
  14. Minh họa: Nguyên lý hoạt động của GD (Hình 4-3)
  15. Ảnh hưởng của Tốc độ học (Learning Rate) (Chỉ chứa Text - *Quy tắc tách slide*)
  16. Minh họa: Tốc độ học quá nhỏ và quá cao (Hình 4-4 & Hình 4-5 song song - *Quy tắc tách slide*)
  17. Các cạm bẫy của Gradient Descent (Hình 4-6)
  18. Tầm quan trọng của Điều chỉnh tỷ lệ đặc trưng (Scaling) (Hình 4-7)
  19. Hạ Gradient theo lô (Batch GD)
  20. Minh họa: Batch GD với các tốc độ học khác nhau (Hình 4-8)
  21. Hạ Gradient ngẫu nhiên (Stochastic GD)
  22. Đặc điểm của SGD (Hình 4-9)
  23. Minh họa: 20 bước đầu tiên của SGD (Hình 4-10)
  24. Hạ Gradient theo Mini-Batch
  25. So sánh 3 thuật toán Gradient Descent
  26. Minh họa: Đường đi trong không gian tham số (Hình 4-11)
  27. Bảng so sánh tổng hợp các thuật toán hồi quy
  28. Slide chuyển ý

### Tiết 3 (17 Frames): Hồi quy đa thức, Đường cong học tập & Chính quy hóa
- **Mục tiêu:** Mở rộng sang dữ liệu phi tuyến, phát hiện Overfitting qua Đường cong học tập và khắc phục bằng các kỹ thuật Regularization.
- **Nội dung:**
  29. Hồi quy đa thức
  30. Khớp Hồi quy đa thức trên dữ liệu phi tuyến (Chỉ chứa Text - *Quy tắc tách slide*)
  31. Minh họa: Tạo và Khớp dữ liệu phi tuyến (Hình 4-12 & Hình 4-13 song song - *Quy tắc tách slide*)
  32. Hiện tượng Quá khớp (Overfitting) với đa thức bậc cao (Hình 4-14)
  33. Đánh giá mô hình bằng Đường cong học tập (Chỉ chứa Text - *Quy tắc tách slide*)
  34. Minh họa: Đường cong học tập - Dưới khớp và Quá khớp (Hình 4-15 & Hình 4-16 song song - *Quy tắc tách slide*)
  35. Sự đánh đổi giữa Độ chệch (Bias) và Phương sai (Variance)
  36. Slide chuyển ý (Chính quy hóa Mô hình tuyến tính)
  37. Hồi quy Ridge (Chuẩn hóa Tikhonov)
  38. Minh họa: Hồi quy Ridge (Hình 4-17)
  39. Hồi quy Lasso
  40. Minh họa: Hồi quy Lasso (Hình 4-18)
  41. Điểm khác biệt giữa Lasso và Ridge
  42. Minh họa: Phân tích Lasso vs Ridge trong không gian tham số (Hình 4-19)
  43. Hồi quy Elastic Net
  44. Dừng sớm (Early Stopping)
  45. Minh họa: Chính quy hóa bằng Dừng sớm (Hình 4-20)

### Tiết 4 (14 Frames): Hồi quy Logistic & Softmax
- **Mục tiêu:** Ứng dụng mô hình Hồi quy cho bài toán Phân loại (Nhị phân và Đa lớp).
- **Nội dung:**
  46. Slide chuyển ý (Hồi quy Logistic)
  47. Ước tính xác suất
  48. Hàm Logistic (Sigmoid) (Hình 4-21)
  49. Hàm huấn luyện và Hàm chi phí (Log loss)
  50. Đạo hàm riêng của hàm chi phí Logistic
  51. Tập dữ liệu hoa Diên vĩ (Iris Dataset) (Hình 4-22)
  52. Các đường ranh giới quyết định (Chỉ chứa Text - *Quy tắc tách slide*)
  53. Minh họa: Ranh giới quyết định 1 chiều và 2 chiều (Hình 4-23 & Hình 4-24 song song - *Quy tắc tách slide*)
  54. Hồi quy Softmax (Hồi quy Logistic đa thức)
  55. Điểm số Softmax và Hàm Softmax
  56. Dự đoán và Hàm chi phí Cross Entropy
  57. Minh họa: Ranh giới quyết định của Hồi quy Softmax (Hình 4-25)
  58. Tổng kết Chương 4

---

## 2. Kiến trúc Kỹ thuật (Kịch bản sinh mã)
- Viết file mã nguồn Python (VD: `v6_generate_slides_ch04.py`) theo các nguyên tắc đã chốt từ Chương 1-3:
  - **Sử dụng môi trường chia cột:** `\begin{columns}` (tỷ lệ 0.5 - 0.5) cho các slide có 1 hình ảnh.
  - **Quy tắc tách slide nhiều hình (Bắt buộc):** Tiếp tục áp dụng triệt để quy tắc: Gom các hình có chủ đề tương đồng thành 1 cặp và tách thành slide độc lập. (Gồm 4 cặp đã xác định: Hình 4-4/4-5, Hình 4-12/4-13, Hình 4-15/4-16, Hình 4-23/4-24).
  - **Quy tắc biên dịch (Bắt buộc):** Chạy lệnh `xelatex` **2 lần liên tiếp** để đảm bảo Mục lục hiển thị đầy đủ ở slide "Nội dung Chương trình".
  - Bố trí cấu hình ảnh: `\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]`.

## 3. Nhật ký thực hiện (Hoàn thành)
- [x] Soạn bản nháp kiến trúc kịch bản tạo slide (Python script `v6_generate_slides_ch04.py`).
- [x] Trích xuất nội dung từ `chuong_04.md`.
- [x] Ghép nối 25 hình ảnh tương ứng (Kế thừa quy tắc tách slide song song).
- [x] Chạy lệnh `xelatex` 2 lần liên tiếp để biên dịch PDF.
- [x] Kiểm tra và đối soát lỗi hiển thị hình ảnh.
