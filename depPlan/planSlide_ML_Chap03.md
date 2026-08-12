# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 3 - Phân loại

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **~50 Frames (Slides)** (Nằm trong khoảng 40 - 60 slides).  
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_03.md`).  
**Tích hợp hình ảnh:** Phân bổ toàn bộ 13 hình ảnh (`Hinh_3-1` đến `Hinh_3-13` trong thư mục `Figures/CH03/`) vào các slide tương ứng.

---

## 1. Bố cục Phân chương Tiết học (Sections & TOC)

Bộ slide Chương 3 được chia thành **3 Tiết học (3 Sections)** rõ ràng với tổng cộng khoảng 50 Frames:

### Tiết 1 (18 Frames): Giới thiệu Phân loại, MNIST & Đánh giá Hiệu suất cơ bản
- **Mục tiêu:** Nắm được khái niệm phân loại, làm quen bộ dữ liệu MNIST và các thước đo hiệu suất cốt lõi (Độ chính xác, Ma trận nhầm lẫn, Precision, Recall).
- **Nội dung các slide thực tế:**
  1. Trang bìa (`\titlepage`)
  2. Mục lục nội dung chương (`\tableofcontents`)
  3. Mục tiêu bài học
  4. Slide chuyển ý (Giới thiệu & Bộ dữ liệu MNIST)
  5. Giới thiệu Bộ dữ liệu MNIST ("Hello World" của Học máy)
  6. Hình ảnh MNIST (Hình 3-1)
  7. Sự đa dạng của chữ số MNIST (Hình 3-2)
  8. Phân chia tập huấn luyện và tập kiểm thử
  9. Slide chuyển ý (Huấn luyện bộ phân loại nhị phân)
  10. Bài toán: Bộ phát hiện chữ số 5 (Binary Classification)
  11. Thuật toán Stochastic Gradient Descent (SGD)
  12. Slide chuyển ý (Các thước đo hiệu suất)
  13. Đo độ chính xác bằng kiểm định chéo (Cross-Validation)
  14. Sự đánh lừa của Độ chính xác (Accuracy) trên dữ liệu lệch
  15. Ma trận nhầm lẫn (Confusion Matrix)
  16. Cấu trúc của Ma trận nhầm lẫn (Hình 3-3)
  17. Độ chính xác (Precision) và Độ nhạy (Recall / TPR)
  18. Sự kết hợp: Điểm F1 (F1 Score)

### Tiết 2 (16 Frames): Sự đánh đổi Hiệu suất & Phân loại Đa lớp
- **Mục tiêu:** Hiểu sâu về sự đánh đổi giữa Precision và Recall, đọc hiểu đường cong ROC, và mở rộng bài toán sang phân loại nhiều lớp.
- **Nội dung các slide thực tế:**
  19. Slide chuyển ý (Sự đánh đổi Độ chính xác và Độ nhạy)
  20. Ngưỡng quyết định (Decision Threshold) trong SGDClassifier
  21. Minh họa sự đánh đổi (Hình 3-4)
  22. Các biểu đồ phân tích Precision/Recall (Chỉ chứa Text - *Theo quy tắc tách slide*)
  23. Minh họa: Đường cong Precision/Recall (Hình 3-5 & Hình 3-6 song song - *Theo quy tắc tách slide*)
  24. Đường cong ROC (Receiver Operating Characteristic)
  25. ROC AUC: So sánh các bộ phân loại (Chỉ chứa Text - *Theo quy tắc tách slide*)
  26. Minh họa: Đường cong ROC của SGD và Random Forest (Hình 3-7 & Hình 3-8 song song - *Theo quy tắc tách slide*)
  27. Slide chuyển ý (Phân loại đa lớp)
  28. Chiến lược Một-đối-phần-còn-lại (OvR)
  29. Chiến lược Một-đối-một (OvO)
  30. Phân loại đa lớp tự động trong Scikit-Learn (với SVC)
  31. Phân loại đa lớp với thuật toán gốc (Random Forest, SGD)
  32. Điểm số quyết định cho dự đoán đa lớp
  33. Đánh giá bộ phân loại đa lớp
  34. Cải thiện mô hình với Co giãn đặc trưng (Feature Scaling)

### Tiết 3 (16 Frames): Phân tích lỗi, Đa nhãn & Đa đầu ra
- **Mục tiêu:** Phân tích các lỗi mô hình phân loại thường gặp, hiểu các khái niệm phân loại đa nhãn và phân loại đa đầu ra (multi-output).
- **Nội dung các slide thực tế:**
  35. Slide chuyển ý (Phân tích lỗi)
  36. Trực quan hóa Ma trận nhầm lẫn đa lớp (Chỉ chứa Text - *Theo quy tắc tách slide*)
  37. Minh họa: Phân tích Ma trận nhầm lẫn (Hình 3-9 & Hình 3-10 song song - *Theo quy tắc tách slide*)
  38. Phân tích lỗi riêng lẻ và cách khắc phục
  39. So sánh lỗi giữa 3 và 5 (Hình 3-11)
  40. Tăng cường dữ liệu (Data Augmentation)
  41. Slide chuyển ý (Phân loại đa nhãn)
  42. Phân loại đa nhãn (Multilabel Classification) là gì?
  43. Ví dụ về hệ thống đa nhãn (Số lớn & Số lẻ)
  44. Đánh giá hệ thống phân loại đa nhãn
  45. Phân loại đa nhãn với ClassifierChain
  46. Slide chuyển ý (Phân loại đa đầu ra)
  47. Phân loại đa đầu ra (Multioutput Classification) là gì?
  48. Bài toán khử nhiễu hình ảnh (Chỉ chứa Text - *Theo quy tắc tách slide*)
  49. Minh họa: Quá trình khử nhiễu (Hình 3-12 & Hình 3-13 song song - *Theo quy tắc tách slide*)
  50. Tổng kết Chương 3

---

## 2. Kiến trúc Kỹ thuật (Kịch bản sinh mã)
- Viết file mã nguồn Python (VD: `v5_generate_slides_ch03.py`) theo các nguyên tắc đã chốt từ Chương 1 và 2:
  - **Sử dụng môi trường chia cột:** `\begin{columns}` (tỷ lệ 0.5 - 0.5) cho slide có 1 hình ảnh.
  - **Quy tắc cho slide nhiều hình (Bắt buộc):** Nếu một chủ đề có 2 hình ảnh trở lên, bắt buộc tách thành 2 slide: một slide chỉ chứa văn bản giải thích, một slide kế tiếp chứa các hình ảnh đặt song song (trái - phải) để tối đa hóa kích thước hình, giúp người xem dễ đọc nội dung. (Áp dụng cho các cặp hình: Hình 3-5/3-6, Hình 3-7/3-8, Hình 3-9/3-10, Hình 3-12/3-13).
  - Tự động thiết lập cấu hình `\setbeamertemplate{caption}{\raggedright\insertcaption\par}` để xóa bỏ chữ "Figure:" tiếng Anh dư thừa.
  - Tận dụng `\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]` để ảnh luôn vừa vặn mà không bị bóp méo hay tràn lề.
  - **Quy tắc biên dịch (Bắt buộc):** Phải chạy lệnh `xelatex` ít nhất 2 lần liên tiếp để LaTeX có thể tạo đúng Mục lục (Table of Contents) cho slide "Nội dung Chương trình".

## 3. Nhật ký thực hiện (Hoàn thành)
- [x] Soạn bản nháp kiến trúc kịch bản tạo slide (Python script `v5_generate_slides_ch03.py`).
- [x] Trích xuất nội dung từ `chuong_03.md`.
- [x] Ghép nối 13 hình ảnh tương ứng (Chú ý áp dụng đúng quy tắc tách slide cho các cặp hình ảnh).
- [x] Chạy lệnh `xelatex` 2 lần để biên dịch PDF và hiển thị đúng Mục lục.
- [x] Đã xuất thành công file PDF 49 trang hoàn chỉnh (`Slide_ML_Chap03.pdf`).
- [x] Kiểm tra và đối soát lỗi hiển thị hình ảnh (nếu có).
