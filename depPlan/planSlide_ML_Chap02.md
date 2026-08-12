# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 2 - Dự án Học máy Từ đầu đến cuối

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **~50 Frames (Slides)** (Nằm trong khoảng 40 - 60 slides theo yêu cầu).  
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_02.md`).  
**Tích hợp hình ảnh:** Phân bổ toàn bộ 20 hình ảnh (`Hinh_2-1` đến `Hinh_2-20` trong thư mục `Figures/CH02/`) vào các slide tương ứng.

---

## 1. Bố cục Phân chương Tiết học (Sections & TOC)

Bộ slide Chương 2 được chia thành **3 Tiết học (3 Sections)** rõ ràng với tổng cộng khoảng 50 Frames:

### Tiết 1 (16 Frames): Giới thiệu Dự án & Khởi tạo Dữ liệu
- **Mục tiêu:** Nắm được quy trình từ lúc xác định vấn đề, chọn thước đo hiệu suất đến lấy và chia tách dữ liệu.
- **Nội dung các slide thực tế:**
  1. Trang bìa (`\titlepage`)
  2. Mục lục nội dung chương (`\tableofcontents`)
  3. Mục tiêu bài học
  4. Slide chuyển ý (Giới thiệu Dự án & Bức tranh lớn)
  5. Làm việc với dữ liệu thực (Hình 2-1. Giá nhà California)
  6. Nhìn vào bức tranh lớn: Xác định vấn đề
  7. Pipeline hệ thống học máy (Hình 2-2)
  8. Chọn một Thước đo Hiệu suất (RMSE, MAE)
  9. Kiểm tra các Giả định
  10. Slide chuyển ý (Lấy dữ liệu & Môi trường làm việc)
  11. Môi trường Google Colab (Hình 2-3 & 2-4)
  12. Tải và Xem dữ liệu (Hình 2-6)
  13. Xem tóm tắt dữ liệu (Hình 2-7 & 2-8)
  14. Tạo tập kiểm thử (Test set): Tại sao cần thiết?
  15. Phân phối Thu nhập (Hình 2-9)
  16. Lấy mẫu phân tầng so với ngẫu nhiên (Hình 2-10)

### Tiết 2 (18 Frames): Khám phá & Chuẩn bị Dữ liệu
- **Mục tiêu:** Nắm bắt kỹ thuật trực quan hóa dữ liệu địa lý, tìm kiếm tương quan và xây dựng Data Pipeline (làm sạch, biến đổi).
- **Nội dung các slide thực tế:**
  17. Slide chuyển ý (Tự do khám phá và trực quan hóa)
  18. Trực quan hóa dữ liệu địa lý (Hình 2-11)
  19. Phân tích mật độ địa lý (Hình 2-12)
  20. Giá nhà theo dân số & Vị trí (Hình 2-13)
  21. Tìm kiếm các tương quan (Correlation)
  22. Ma trận phân tán (Scatter Matrix - Hình 2-14)
  23. Thu nhập trung bình vs Giá nhà trung bình (Hình 2-15)
  24. Phân phối và độ lệch chuẩn của dữ liệu (Hình 2-16)
  25. Thử nghiệm với các kết hợp đặc trưng
  26. Slide chuyển ý (Chuẩn bị dữ liệu cho học máy)
  27. Làm sạch dữ liệu (Missing Values)
  28. Xử lý các thuộc tính văn bản và phân loại (One-Hot Encoding)
  29. Co giãn Đặc trưng (Feature Scaling)
  30. Biến đổi Đặc trưng để gần với Gaussian (Hình 2-17)
  31. Biến đổi Đặc trưng bằng RBF (Hình 2-18)
  32. Đặc trưng từ phân cụm bằng K-Means (Hình 2-19)
  33. Bộ biến đổi tùy chỉnh (Custom Transformers)
  34. Các pipeline biến đổi (Transformation Pipelines)

### Tiết 3 (16 Frames): Chọn, Huấn luyện, Tinh chỉnh & Triển khai
- **Mục tiêu:** Chọn mô hình, đánh giá, tinh chỉnh siêu tham số và triển khai dự án thực tế ra Production.
- **Nội dung các slide thực tế:**
  35. Slide chuyển ý (Chọn và Huấn luyện Mô hình)
  36. Huấn luyện trên Tập huấn luyện (Linear Regression, Decision Tree)
  37. Đánh giá tốt hơn bằng kiểm định chéo (Cross-Validation)
  38. Thử nghiệm với Rừng ngẫu nhiên (Random Forest)
  39. Slide chuyển ý (Tinh chỉnh mô hình)
  40. Tìm kiếm theo lưới (Grid Search)
  41. Tìm kiếm ngẫu nhiên (Random Search)
  42. Các phương pháp tập hợp (Ensemble Methods)
  43. Phân tích các mô hình tốt nhất và lỗi của chúng
  44. Đánh giá hệ thống trên tập kiểm thử
  45. Slide chuyển ý (Triển khai, Giám sát và Bảo trì)
  46. Triển khai mô hình (Model Deployment)
  47. Kiến trúc Triển khai (Hình 2-20)
  48. Giám sát hệ thống (Monitoring)
  49. Tự động hóa quá trình bảo trì và cập nhật
  50. Tổng kết Chương 2

---

## 2. Kiến trúc Kỹ thuật (Kịch bản sinh mã)
- Dựa trên kinh nghiệm từ Chương 1, file mã nguồn sinh slide (VD: `v4_generate_slides_ch02.py`) sẽ được viết theo nguyên tắc:
  - Sử dụng môi trường `\begin{columns}` (tỷ lệ 0.5 - 0.5) để văn bản ở bên trái và hình minh họa ở bên phải đối với slide có 1 hình.
  - **Quy tắc cho slide nhiều hình:** Nếu một chủ đề có 2 hình ảnh trở lên, bắt buộc tách thành 2 slide: một slide chỉ chứa văn bản giải thích, một slide kế tiếp chứa các hình ảnh đặt song song (trái - phải) để tối đa hóa kích thước hình, giúp người xem dễ đọc nội dung bên trong hình (ví dụ: giao diện phần mềm, bảng biểu).
  - Tự động thiết lập cấu hình `\setbeamertemplate{caption}{\raggedright\insertcaption\par}` để xóa bỏ chữ "Figure:" tiếng Anh dư thừa.
  - Tận dụng `\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]` để ảnh luôn vừa vặn mà không bị bóp méo hay tràn lề.

## 3. Nhật ký thực hiện (Hoàn thành)
- [x] Soạn bản nháp kiến trúc kịch bản tạo slide (Python script `v4_generate_slides_ch02.py`).
- [x] Trích xuất nội dung từ `chuong_02.md`.
- [x] Ghép nối 20 hình ảnh tương ứng.
- [x] Chạy lệnh `xelatex` để biên dịch PDF.
- [x] Đã xuất thành công file PDF 51 trang hoàn chỉnh (`Slide_ML_Chap02.pdf`).
- [x] **Cập nhật (Refinement):** Tách slide "Môi trường Google Colab" thành 2 slide (1 text, 1 chứa 2 hình song song) để hình ảnh to rõ, dễ nhìn, đồng thời cập nhật quy tắc này vào kế hoạch.
