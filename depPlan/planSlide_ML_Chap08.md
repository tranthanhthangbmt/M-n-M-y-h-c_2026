# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 8 - Học máy (Chuẩn Đại học)

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **~55 Frames (Slides)** (Nằm trong khoảng 40 - 60 slides theo yêu cầu).  
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_08.md`).  
**Tích hợp hình ảnh:** Phân bổ các hình ảnh (`Hinh_8-1` đến `Hinh_8-7` trong thư mục `Figures/CH08/`) vào các slide tương ứng. *(Lưu ý: Các hình từ 8-8 đến 8-11 chưa có sẵn trong thư mục nên sẽ chỉ trình bày bằng chữ/code)*.

---

## 1. Bố cục Phân chương Tiết học (Sections & TOC)

Bộ slide Chương 8 được chia thành **3 Tiết học (3 Sections)** rõ ràng với tổng cộng 55 Frames:

### Tiết 1 (18 Frames): Giới thiệu Giảm chiều dữ liệu & Các cách tiếp cận chính
- **Mục tiêu:** Hiểu rõ vì sao cần giảm chiều dữ liệu, "lời nguyền của số chiều" và 2 cách tiếp cận: Phép chiếu & Học đa tạp.
- **Nội dung các slide thực tế:**
  1. Trang bìa (`\titlepage`)
  2. Mục lục nội dung chương (`\tableofcontents`)
  3. Giới thiệu Giảm chiều dữ liệu
  4. Lợi ích 1: Tăng tốc quá trình huấn luyện
  5. Lợi ích 2: Trực quan hóa dữ liệu
  6. Lời nguyền của số chiều (The Curse of Dimensionality)
  7. Hình ảnh: Lời nguyền số chiều (Minh họa không gian 4D) -> Hình 8-1
  8. Sự khác biệt khoảng cách trong không gian chiều cao
  9. Nguy cơ quá khớp (Overfitting) do thưa thớt dữ liệu
  10. Các cách tiếp cận chính để giảm chiều
  11. Cách tiếp cận 1: Phép chiếu (Projection)
  12. Hình ảnh: Tập dữ liệu 3D gần không gian con 2D -> Hình 8-2
  13. Hình ảnh: Tập dữ liệu 2D mới sau phép chiếu -> Hình 8-3
  14. Cách tiếp cận 2: Học đa tạp (Manifold Learning)
  15. Hình ảnh: Tập dữ liệu Swiss roll -> Hình 8-4
  16. Hình ảnh: Làm dẹt vs Mở cuộn Swiss roll -> Hình 8-5
  17. Giả định đa tạp (Manifold Hypothesis)
  18. Hình ảnh: Đường biên quyết định trong không gian giảm chiều -> Hình 8-6

### Tiết 2 (24 Frames): Phân tích thành phần chính (PCA)
- **Mục tiêu:** Nắm vững toán học và cách sử dụng PCA, lựa chọn số chiều, PCA ngẫu nhiên và PCA tăng dần.
- **Nội dung các slide thực tế:**
  19. Giới thiệu PCA (Principal Component Analysis)
  20. Ý tưởng PCA: Bảo toàn phương sai (Preserving Variance)
  21. Hình ảnh: Lựa chọn không gian con để chiếu -> Hình 8-7
  22. Các thành phần chính (Principal Components - PC)
  23. Toán học PCA: Phân tách giá trị số ít (SVD)
  24. Mã nguồn: Tìm thành phần chính bằng NumPy (SVD)
  25. Chiếu dữ liệu xuống d chiều
  26. Mã nguồn: Thực hiện phép chiếu bằng NumPy
  27. Sử dụng Scikit-Learn cho PCA
  28. Tỷ lệ phương sai giải thích (Explained Variance Ratio)
  29. Đánh giá tỷ lệ phương sai giải thích
  30. Chọn số chiều phù hợp như thế nào?
  31. Mã nguồn: Tìm số chiều để giữ 95% phương sai
  32. Đặt `n_components` là tỷ lệ phần trăm (0.95)
  33. Vẽ biểu đồ phương sai giải thích theo số chiều
  34. Điều chỉnh số chiều như một siêu tham số (Hyperparameter Tuning)
  35. Mã nguồn: Sử dụng `RandomizedSearchCV` cho PCA
  36. PCA dùng để nén dữ liệu (Compression)
  37. Tái tạo dữ liệu (Giải nén)
  38. Mã nguồn: Phép biến đổi ngược (`inverse_transform`)
  39. PCA ngẫu nhiên (Randomized PCA)
  40. Mã nguồn: Cài đặt Randomized PCA
  41. PCA tăng dần (Incremental PCA) cho dữ liệu lớn
  42. Mã nguồn: Cài đặt Incremental PCA với `array_split`
  43. Tối ưu Incremental PCA với tệp bộ nhớ (memmap)

### Tiết 3 (12 Frames): Phép chiếu ngẫu nhiên & LLE
- **Mục tiêu:** Biết thêm các kỹ thuật giảm chiều nâng cao như Random Projection, LLE, t-SNE, MDS.
- **Nội dung các slide thực tế:**
  44. Phép chiếu ngẫu nhiên (Random Projection)
  45. Định lý Johnson-Lindenstrauss
  46. Mã nguồn: Tìm số chiều tối thiểu bảo toàn khoảng cách
  47. Mã nguồn: Phép chiếu ngẫu nhiên thủ công
  48. `GaussianRandomProjection` và `SparseRandomProjection`
  49. Nhúng tuyến tính cục bộ (LLE)
  50. Mã nguồn: Sử dụng `LocallyLinearEmbedding` cho Swiss roll
  51. Thuật toán LLE: Bước 1 (Mô hình hóa cục bộ)
  52. Thuật toán LLE: Bước 2 (Giảm chiều bảo toàn khoảng cách)
  53. Các kỹ thuật giảm chiều khác: MDS & Isomap
  54. Các kỹ thuật giảm chiều khác: t-SNE & LDA
  55. Tổng kết Chương 8

---

## 2. Tiêu chuẩn Kỹ thuật và Nhận diện (Format)
- Bố cục Slide chia làm **2 cột** khi cần thiết để chèn hình ảnh minh họa không bị che mất chữ.
- Chèn code Python vào Block `lstlisting` với highlight màu sắc dễ nhìn.
- Các hình ảnh sẽ được scale bằng `[width=\textwidth]` hoặc `[height=0.7\textheight]` để vừa khung Widescreen 16:9.
- Dùng `\usepackage{fontspec}` và `\setmainfont{Times New Roman}`, `\setsansfont{Arial}` để tránh lỗi mất chữ tiếng Việt khi dùng `xelatex`.
- **Thông tin tác giả:** `\author{Giảng viên: TS. Trần Thành Thắng}`.
- **Biên dịch:** Chạy `xelatex` 2 lần liên tiếp.

## 3. Nhật ký thực hiện (Hoàn thành)
- [x] Soạn kịch bản Python tạo slide (`v10_generate_slides_ch08.py`).
- [x] Trích xuất nội dung từ `chuong_08.md`.
- [x] Chèn các đoạn code Python và công thức toán học.
- [x] Cấu hình đúng `fontspec` để không bị lỗi phông Mục lục.
- [x] Chạy lệnh `xelatex` 2 lần và kiểm tra PDF.

*Ghi chú: Đã biên dịch thành công `Slide_ML_Chap08.pdf` với tổng số 56 slides. Mã nguồn Python được lưu ở `v10_generate_slides_ch08.py`.*
