# Kế hoạch Tạo Slide PDF - Chương 7: Học tổ hợp và Rừng ngẫu nhiên

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **~45-60 Frames (Slides)**  
**Nguồn nội dung chữ:** Trích xuất từ tài liệu trang web của chương (`machineLearningWeb/docs/chuong_07.md`).  
**Tích hợp hình ảnh:** Phân bổ toàn bộ 13 hình ảnh (`Hinh_7-1` đến `Hinh_7-13` trong thư mục `Figures/CH07/`) vào các slide.

---

## 1. Phân tích Nội dung & Cấu trúc Slide
Dựa trên tệp `chuong_07.md`, cấu trúc bài giảng được chia thành 2 Tiết học như sau:

### Tiết 1: Học tổ hợp cơ bản & Túi hóa (Bagging)
- **Mục tiêu:** Hiểu khái niệm "sự khôn ngoan của đám đông", bộ phân loại biểu quyết và phương pháp kết hợp các bộ dự đoán cùng loại qua Bagging/Pasting.
- **Nội dung dự kiến:**
  1. Trang bìa (Tiêu đề, Giảng viên: TS. Trần Thành Thắng)
  2. Nội dung Chương trình (Mục lục)
  3. Giới thiệu học tổ hợp (Ensemble Learning)
  4. Sự khôn ngoan của đám đông
  5. Bộ phân loại biểu quyết (Voting Classifiers)
  6. Minh họa: Huấn luyện các bộ phân loại đa dạng (Hình 7-1)
  7. Bộ phân loại bỏ phiếu cứng (Hard Voting)
  8. Minh họa: Dự đoán bỏ phiếu cứng (Hình 7-2)
  9. Nguyên lý đằng sau bộ phân loại biểu quyết
  10. Luật số lớn (Hình 7-3)
  11. Ví dụ mã nguồn: Bộ phân loại bỏ phiếu cứng (Phần 1)
  12. Ví dụ mã nguồn: Bộ phân loại bỏ phiếu cứng (Phần 2)
  13. Bỏ phiếu mềm (Soft voting)
  14. Ví dụ mã nguồn: Bỏ phiếu mềm
  15. Túi hóa (Bagging) và Dán nhãn (Pasting)
  16. Đặc điểm của Bagging và Pasting
  17. Minh họa: Quá trình lấy mẫu (Hình 7-4)
  18. Bagging và Pasting trong Scikit-Learn
  19. Ví dụ mã nguồn: BaggingClassifier
  20. Phân tích kết quả của Bagging
  21. Minh họa: So sánh Cây đơn lẻ và Tập hợp túi hóa (Hình 7-5 - Hiển thị độc lập vì chứa 2 đồ thị con)
  22. Đánh giá ngoài mẫu (Out-of-Bag Evaluation)
  23. Ví dụ mã nguồn: Đánh giá OOB
  24. Random Patches và Random Subspaces

### Tiết 2: Rừng ngẫu nhiên, Tăng cường (Boosting) & Xếp chồng (Stacking)
- **Mục tiêu:** Khám phá sức mạnh của Rừng ngẫu nhiên, các thuật toán Boosting (AdaBoost, Gradient Boosting) và kỹ thuật Stacking.
- **Nội dung dự kiến:**
  25. Rừng ngẫu nhiên (Random Forests)
  26. Ví dụ mã nguồn: RandomForestClassifier
  27. So sánh BaggingClassifier và RandomForestClassifier
  28. Cây ngẫu nhiên cực đại (Extra-Trees)
  29. Cài đặt Extra-Trees trong Scikit-Learn
  30. Tầm quan trọng của đặc trưng
  31. Ví dụ mã nguồn: Tính toán Feature Importances
  32. Minh họa: Tầm quan trọng pixel MNIST (Hình 7-6)
  33. Tăng cường (Boosting)
  34. AdaBoost (Adaptive Boosting)
  35. Thuật toán huấn luyện AdaBoost
  36. Minh họa: Cập nhật trọng số trong AdaBoost (Hình 7-7)
  37. Phân tích trọng số trong SVM
  38. Minh họa: Đường biên quyết định liên tiếp (Hình 7-8)
  39. Toán học AdaBoost: Tỷ lệ lỗi và Trọng số bộ dự đoán
  40. Toán học AdaBoost: Cập nhật trọng số trường hợp
  41. Ví dụ mã nguồn: AdaBoostClassifier
  42. Gradient Boosting
  43. Nguyên lý Gradient Boosting
  44. Ví dụ mã nguồn: Gradient Tree Boosting thủ công (Phần 1)
  45. Ví dụ mã nguồn: Gradient Tree Boosting thủ công (Phần 2)
  46. Minh họa: Quá trình huấn luyện GBRT (Hình 7-9 - Hiển thị độc lập vì là lưới 3x2 rất lớn)
  47. GradientBoostingRegressor trong Scikit-Learn
  48. Vấn đề quá khớp và Shrinkage
  49. Minh họa: Các tập hợp GBRT không đủ và vừa đủ cây (Hình 7-10 - Hiển thị độc lập vì chứa 2 đồ thị con)
  50. Dừng sớm (Early Stopping) trong GBRT
  51. Tăng cường Gradient dựa trên Biểu đồ (HGB)
  52. Ví dụ mã nguồn: HistGradientBoostingRegressor
  53. Phân lớp xếp chồng (Stacking)
  54. Khái niệm Stacking và bộ trộn (Blender)
  55. Minh họa: Tổng hợp dự đoán bằng bộ trộn (Hình 7-11)
  56. Minh họa: Huấn luyện bộ trộn trong tập hợp (Hình 7-12)
  57. Kiến trúc xếp chồng phức tạp
  58. Minh họa: Tập hợp xếp chồng đa lớp (Hình 7-13)
  59. Cài đặt Stacking trong Scikit-Learn (StackingClassifier)
  60. Tổng kết Chương 7

---

## 2. Kiến trúc Kỹ thuật (Kịch bản sinh mã)
- Viết file mã nguồn Python (VD: `v9_generate_slides_ch07.py`).
- **Gói Font chữ chuẩn xác (Quan trọng):** Sử dụng `fontspec` để gọi trực tiếp font Windows, tránh lỗi mất ký tự tiếng Việt với `xelatex`:
  ```latex
  \usepackage{fontspec}
  \setmainfont{Times New Roman}
  \setsansfont{Arial}
  ```
- **Sử dụng môi trường chia cột:** `\begin{columns}` (tỷ lệ 0.5 - 0.5) cho slide hình ảnh đơn đi kèm chữ.
- **Quy tắc tách slide:** Đối với các hình ghép, hình to nhiều đồ thị con (Hình 7-5, 7-9, 7-10), cần tách thành slide hiển thị độc lập dùng `\begin{center}` để hình bung to nhất có thể.
- **Thông tin tác giả:** `\author{Giảng viên: TS. Trần Thành Thắng}`.
- **Biên dịch:** Chạy `xelatex` 2 lần liên tiếp.

## 3. Nhật ký thực hiện (Hoàn thành)
- [x] Soạn kịch bản Python tạo slide (`v9_generate_slides_ch07.py`).
- [x] Trích xuất nội dung từ `chuong_07.md`.
- [x] Xử lý tách các slide hình lớn độc lập.
- [x] Cấu hình đúng `fontspec` để không bị lỗi phông Mục lục.
- [x] Chạy lệnh `xelatex` 2 lần và kiểm tra PDF.

*Ghi chú: Đã biên dịch thành công `Slide_ML_Chap07.pdf` với tổng số 58 slides. Mã nguồn Python được lưu ở `v9_generate_slides_ch07.py`.*
