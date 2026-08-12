# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 5 - Máy học Véc-tơ hỗ trợ (SVM)

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **~32 Frames (Slides)**  
**Nguồn nội dung chữ:** Trích xuất từ tài liệu trang web của chương (`machineLearningWeb/docs/chuong_05.md`).  
**Tích hợp hình ảnh:** Phân bổ toàn bộ 13 hình ảnh (`Hinh_5-1` đến `Hinh_5-13` trong thư mục `Figures/CH05/`) vào các slide.

---

## 1. Bố cục Phân chương Tiết học (Sections & TOC)

Bộ slide Chương 5 tập trung vào mô hình học máy rất mạnh mẽ là Support Vector Machine, bao gồm cả phân loại, hồi quy và cơ chế hạt nhân (Kernel). Chương này được chia thành **2 Tiết học (2 Sections)** rõ ràng, kế thừa triệt để quy tắc **Tách slide hình ảnh**:

### Tiết 1: Phân loại SVM tuyến tính & phi tuyến
- **Mục tiêu:** Hiểu nguyên lý phân loại biên lớn (Large margin), phân loại biên mềm, và cách sử dụng các Kernel để xử lý dữ liệu phi tuyến.
- **Nội dung:**
  1. Trang bìa (`\titlepage`)
  2. Mục lục nội dung chương (`\tableofcontents`)
  3. Mục tiêu bài học
  4. Slide chuyển ý (Phân loại SVM)
  5. Phân loại SVM tuyến tính (Phân loại biên lớn)
  6. Minh họa: Phân loại biên lớn (Hình 5-1)
  7. Độ nhạy của mô hình SVM (Chỉ chứa Text - *Quy tắc tách slide*)
  8. Minh họa: Độ nhạy với thang đo đặc trưng (Hình 5-2)
  9. Minh họa: Độ nhạy của biên cứng với ngoại lệ (Hình 5-3)
  10. Phân loại biên mềm
  11. Minh họa: Biên lớn so với ít vi phạm biên (Hình 5-4)
  12. Phân loại SVM phi tuyến
  13. Minh họa: Thêm đặc trưng để phân tách tuyến tính (Hình 5-5)
  14. Minh họa: SVM tuyến tính sử dụng đặc trưng đa thức (Hình 5-6)
  15. Kernel đa thức (Polynomial Kernel)
  16. Minh họa: Bộ phân loại SVM với Kernel đa thức (Hình 5-7)
  17. Các đặc trưng tương tự (Similarity features)
  18. Minh họa: Đặc trưng tương tự sử dụng Gaussian RBF (Hình 5-8)
  19. Kernel Gaussian RBF
  20. Minh họa: Bộ phân loại SVM sử dụng Kernel RBF (Hình 5-9)

### Tiết 2: Các lớp SVM, Hồi quy SVM \& Cơ chế hoạt động
- **Mục tiêu:** Nắm rõ sự khác biệt giữa các lớp thuật toán, ứng dụng SVM cho hồi quy và hiểu sâu toán học bên trong SVM.
- **Nội dung:**
  21. Các lớp SVM và độ phức tạp tính toán
  22. Hồi quy SVM (SVM Regression) (Chỉ chứa Text - *Quy tắc tách slide*)
  23. Minh họa: Hồi quy tuyến tính SVM (Hình 5-10)
  24. Minh họa: Hồi quy đa thức SVM bậc hai (Hình 5-11)
  25. Bên trong bộ phân loại SVM tuyến tính
  26. Minh họa: Vector trọng số nhỏ hơn dẫn đến biên lớn hơn (Hình 5-12)
  27. Tối ưu hóa: Hàm mất mát bản lề (Hinge loss)
  28. Minh họa: Hàm mất mát bản lề và bản lề bình phương (Hình 5-13)
  29. Bài toán đối ngẫu (Dual problem)
  30. Từ giải pháp đối ngẫu đến giải pháp gốc
  31. SVM sử dụng Kernel (Kernelized SVMs) \& Kernel Trick
  32. Các Kernel thông dụng
  33. Đưa ra dự đoán bằng Kernelized SVM
  34. Tổng kết Chương 5

---

## 2. Kiến trúc Kỹ thuật (Kịch bản sinh mã)
- Viết file mã nguồn Python (VD: `v7_generate_slides_ch05.py`) theo các nguyên tắc đã chốt từ Chương 1-4:
  - **Sử dụng môi trường chia cột:** `\begin{columns}` (tỷ lệ 0.5 - 0.5) cho các slide có 1 hình ảnh đi kèm chữ, hoặc hiển thị full hình.
  - **Quy tắc tách slide nhiều hình (Cập nhật):** Đối với các hình dài theo chiều ngang (như Hình 5-2, 5-3, 5-10, 5-11), không ghép song song mà tách thành từng slide độc lập để đảm bảo hình ảnh to và rõ ràng.
  - **Thông tin tác giả (Cập nhật):** Bắt buộc cấu hình `\author{Giảng viên: TS. Trần Thành Thắng}` cho trang bìa.
  - **Quy tắc biên dịch (Bắt buộc):** Chạy lệnh `xelatex` **2 lần liên tiếp** để đảm bảo Mục lục hiển thị đầy đủ.
  - Bố trí cấu hình ảnh: `\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]`.

## 3. Nhật ký thực hiện (Hoàn thành)
- [x] Soạn bản nháp kiến trúc kịch bản tạo slide (Python script `v7_generate_slides_ch05.py`).
- [x] Trích xuất nội dung từ `chuong_05.md`.
- [x] Ghép nối 13 hình ảnh tương ứng (Kế thừa quy tắc tách slide song song).
- [x] Chạy lệnh `xelatex` 2 lần liên tiếp để biên dịch PDF.
- [x] Kiểm tra và đối soát lỗi hiển thị hình ảnh.
