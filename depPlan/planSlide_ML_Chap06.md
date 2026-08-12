# Kế hoạch Tạo Slide PDF - Chương 6: Cây quyết định

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **~27 Frames (Slides)**  
**Nguồn nội dung chữ:** Trích xuất từ tài liệu trang web của chương (`machineLearningWeb/docs/chuong_06.md`).  
**Tích hợp hình ảnh:** Phân bổ toàn bộ 9 hình ảnh (`Hinh_6-1` đến `Hinh_6-9` trong thư mục `Figures/CH06/`) vào các slide.

---
## 1. Phân tích Nội dung & Cấu trúc Slide
Dựa trên tệp `chuong_06.md` và 9 hình ảnh (Hình 6-1 đến Hình 6-9), cấu trúc bài giảng được chia thành các phần sau:

### Tiết 1: Cơ bản về Cây quyết định & Phân loại
- **Mục tiêu:** Hiểu nguyên lý hoạt động, cách huấn luyện, đưa ra dự đoán và đo lường độ không tinh khiết.
- **Nội dung:**
  1. Trang bìa (Tiêu đề, Giảng viên: TS. Trần Thành Thắng)
  2. Nội dung Chương trình (Mục lục)
  3. Giới thiệu chung về Cây quyết định
  4. Huấn luyện và trực quan hóa cây quyết định
  5. Minh họa: Cây quyết định Iris (Hình 6-1 - Hiển thị full hình vì cây phân nhánh rộng)
  6. Đưa ra dự đoán
  7. Độ không tinh khiết Gini (Phương trình 6-1)
  8. Minh họa: Đường biên quyết định của cây quyết định (Hình 6-2)
  9. Ước tính xác suất lớp
  10. Thuật toán huấn luyện CART & Hàm chi phí (Phương trình 6-2)
  11. Độ phức tạp tính toán
  12. Độ không tinh khiết Gini hay Entropy? (Phương trình 6-3)

### Tiết 2: Chính quy hóa, Hồi quy & Các hạn chế
- **Mục tiêu:** Nắm bắt cách chống quá khớp, áp dụng cây quyết định cho hồi quy và hiểu các hạn chế của mô hình.
- **Nội dung:**
  13. Siêu tham số chính quy hóa
  14. Các siêu tham số chính quy hóa phổ biến
  15. Minh họa: Cây không chính quy hóa và có chính quy hóa (Hình 6-3 - Hiển thị độc lập để đảm bảo rõ ràng do hình có 2 đồ thị con)
  16. Hồi quy bằng Cây quyết định
  17. Minh họa: Một cây quyết định cho hồi quy (Hình 6-4)
  18. Minh họa: Dự đoán của hai mô hình hồi quy (Hình 6-5 - Hiển thị độc lập)
  19. Hàm chi phí CART cho hồi quy
  20. Minh họa: Chính quy hóa trong cây hồi quy (Hình 6-6 - Hiển thị độc lập)
  21. Độ nhạy với hướng trục
  22. Minh họa: Độ nhạy với việc xoay tập huấn luyện (Hình 6-7 - Hiển thị độc lập)
  23. Khắc phục nhạy cảm hướng trục bằng PCA
  24. Minh họa: Đường biên quyết định sau PCA (Hình 6-8)
  25. Cây quyết định có phương sai cao
  26. Minh họa: Sự thay đổi mô hình khi huấn luyện lại (Hình 6-9)
  27. Tổng kết Chương 6

---

## 2. Kiến trúc Kỹ thuật (Kịch bản sinh mã)
- Viết file mã nguồn Python (VD: `v8_generate_slides_ch06.py`) theo các nguyên tắc đã chốt từ Chương 1-5:
  - **Sử dụng môi trường chia cột:** `\begin{columns}` (tỷ lệ 0.5 - 0.5) cho các slide có 1 hình ảnh nhỏ đi kèm chữ.
  - **Quy tắc tách slide nhiều hình/Hình ngang to:** Đối với các hình dài theo chiều ngang (như Hình 6-1, 6-3, 6-5, 6-6, 6-7), không dùng chia cột mà tách thành slide độc lập hoặc dùng `\begin{center}` với tỷ lệ width/height lớn để đảm bảo hình ảnh to và rõ ràng.
  - **Thông tin tác giả:** Bắt buộc cấu hình `\author{Giảng viên: TS. Trần Thành Thắng}` cho trang bìa.
  - **Quy tắc biên dịch (Bắt buộc):** Chạy lệnh `xelatex` **2 lần liên tiếp** để đảm bảo Mục lục hiển thị đầy đủ.
  - Bố trí cấu hình ảnh: `\includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]` cho slide độc lập, hoặc `0.6\textheight` cho slide chia cột.

## 3. Nhật ký thực hiện (Hoàn thành)
- [x] Soạn bản nháp kiến trúc kịch bản tạo slide (Python script `v8_generate_slides_ch06.py`).
- [x] Trích xuất nội dung từ `chuong_06.md`.
- [x] Ghép nối 9 hình ảnh tương ứng.
- [x] Chạy lệnh `xelatex` 2 lần liên tiếp để biên dịch PDF.
- [x] Kiểm tra và đối soát lỗi hiển thị hình ảnh.
