# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 12 - Mô hình tùy chỉnh và Huấn luyện với TensorFlow

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **$\geq 40$ Frames (Slides)**  
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_12.md` / `CHƯƠNG 12.htm`).  
**Tích hợp hình ảnh:** Phân bổ 04 hình ảnh cốt lõi (từ `Hinh_12-1` đến `Hinh_12-4` trong thư mục `Figures/CH12/`) vào các slide tương ứng. Đặc thù của chương 12 là mã nguồn (code) chiếm tỷ trọng rất cao nên slide sẽ tập trung hiển thị code highlight đẹp mắt.

---

## 1. Bố cục Phân chương Tiết học (Sections & TOC)

Bộ slide Chương 12 sẽ được chia thành **4 Tiết học (4 Sections)** bao phủ từ các thao tác tensor cơ bản đến thiết kế vòng lặp huấn luyện tùy chỉnh:

### Tiết 1 (12 Frames): Giới thiệu TensorFlow \& Sử dụng TensorFlow như NumPy
- **Mục tiêu:** Hiểu kiến trúc cơ bản của TensorFlow và cách thao tác với Tensors tương tự như Numpy arrays.
- **Nội dung các slide thực tế:**
  1. Trang bìa (`\titlepage`)
  2. Mục lục nội dung chương (`\tableofcontents`)
  3. Giới thiệu nhanh về TensorFlow (Kiến trúc hạ tầng, TPU, GPU)
  4. Hệ sinh thái TensorFlow (TF Hub, TFX, TensorBoard)
  5. Tensor và Phép toán (Tensors and Operations) - Khởi tạo `tf.constant`
  6. Các hàm toán học cơ bản (`tf.add`, `tf.square`, `tf.exp`)
  7. Tensors và NumPy: Tương tác qua lại (`np.array(tensor)`)
  8. Chuyển đổi kiểu dữ liệu (Type Conversions) - Không tự ép kiểu ngầm định
  9. Biến số (Variables) - Sự khác biệt với hằng số `tf.Variable`
  10. Cập nhật Biến (`assign`, `assign_add`)
  11. Các cấu trúc dữ liệu khác: Sparse tensor, Tensor arrays, Ragged tensors
  12. Chuỗi (String tensors) và Tập hợp (Sets)

### Tiết 2 (12 Frames): Tùy chỉnh Các thành phần trong Keras (Loss, Layers, Models)
- **Mục tiêu:** Thoát khỏi các API có sẵn của Keras để viết các hàm tùy chỉnh đặc thù (Ví dụ: Hàm mất mát Huber, Lớp tự định nghĩa).
- **Nội dung các slide thực tế:**
  13. Tại sao cần tùy chỉnh mô hình?
  14. Tùy chỉnh Hàm mất mát (Custom Loss Function) - Ví dụ: Huber Loss
  15. Triển khai Huber Loss bằng mã nguồn Keras
  16. Lưu và tải các mô hình có thành phần tùy chỉnh (`custom_objects`)
  17. Viết Hàm mất mát dưới dạng Subclassing (Kế thừa `keras.losses.Loss`)
  18. Hàm tùy chỉnh: Metric (Thước đo)
  19. Lớp tùy chỉnh (Custom Layers) - Tại sao cần lớp riêng?
  20. Viết một Lớp không có trọng số (Layer without weights)
  21. Kế thừa `keras.layers.Layer` để xây dựng Lớp có trọng số (Custom Weights)
  22. Phương thức `build()`, `call()` và `compute_output_shape()`
  23. Tùy chỉnh toàn bộ Mô hình (Custom Models) - Kế thừa `keras.Model`
  24. Ví dụ: Xây dựng Mô hình với Residual Block

### Tiết 3 (10 Frames): Hàm và Biểu đồ (Functions \& Graphs) - AutoGraph
- **Mục tiêu:** Hiểu cơ chế chuyển đổi mã Python thông thường thành Biểu đồ (Graph) TensorFlow tĩnh, tối ưu hóa tốc độ tính toán siêu việt.
- **Nội dung các slide thực tế:**
  25. Hàm TensorFlow so với Hàm Python thông thường
  26. `@tf.function`: Phép màu của AutoGraph
  27. Cơ chế hoạt động: Từ Python AST đến TF Graph
  28. Ưu điểm của việc thực thi bằng Graph (Tốc độ, tối ưu hóa, loại bỏ biến trung gian)
  29. Giới hạn của AutoGraph (Không dùng được các hàm Python I/O, `print`...)
  30. Cách xem Graph sinh ra (`get_concrete_function()`)
  31. Tracing (Theo dõi) và Poly-morphism trong TF Functions
  32. Xử lý logic nhánh và vòng lặp trong TF Function (`tf.cond`, `tf.while_loop`)
  33. Quy tắc AutoGraph an toàn (Ưu tiên các toán tử TF như `tf.range`)
  34. Khi nào không nên dùng `@tf.function`?

### Tiết 4 (10 Frames): Vòng lặp Huấn luyện Tùy chỉnh (Custom Training Loop)
- **Mục tiêu:** Thay thế hàm `model.fit()` bằng vòng lặp tùy chỉnh thủ công hoàn toàn, giúp can thiệp sâu vào gradient và cập nhật trọng số.
- **Nội dung các slide thực tế:**
  35. Hạn chế của hàm `fit()` mặc định
  36. Tổng quan thuật toán Vòng lặp huấn luyện thủ công
  37. `tf.GradientTape()`: Băng từ tự động tính đạo hàm
  38. Sử dụng `GradientTape` để ghi lại phép toán
  39. Tính Gradient và Tối ưu hóa trọng số
  40. Triển khai Vòng lặp Huấn luyện - Phần 1: Khởi tạo Dữ liệu \& Biến
  41. Triển khai Vòng lặp Huấn luyện - Phần 2: Cập nhật Gradient (`optimizer.apply_gradients`)
  42. Triển khai Vòng lặp Huấn luyện - Phần 3: Tính toán Metrics
  43. Xử lý Gradient Clipping trong vòng lặp thủ công
  44. Tóm tắt Chương 12 \& Hỏi Đáp (Q\&A)

---

## 2. Tiêu chuẩn Kỹ thuật và Nhận diện (Format)
- Bố cục Slide chia làm **2 cột** (`\begin{columns}`) khi cần thiết để hiển thị 04 hình ảnh (VD: Hình minh họa Tensor, Graph).
- Do trọng tâm chương này là lập trình, môi trường `\begin{lstlisting}[language=Python]` với tag `[fragile]` sẽ được tối ưu hóa tối đa, không để code bị tràn lề. Trang trí caption ảnh thủ công `\vspace{0.2cm}\textit{Hình 12-X: ...}` để tránh lặp từ "Hình".
- Dùng `\usepackage{fontspec}` thiết lập font tiếng Việt.
- **Thông tin tác giả:** `\author{Giảng viên: TS. Trần Thành Thắng}`.
- **Biên dịch:** Chạy `xelatex` 2 lần liên tiếp để tạo chỉ mục và TOC chính xác.

## 3. Quy trình thực hiện dự kiến
1. **Giai đoạn 1:** Cập nhật script `gen_chap12.py` với nội dung mở rộng ($\geq 40$ slide) và sử dụng 04 ảnh `Hình_12-X`.
2. **Giai đoạn 2:** Viết kịch bản `v14_generate_slides_ch12.py` (chứa đầy đủ các cấu hình `fragile` từ kinh nghiệm Chương 11) để sinh mã LaTeX cho 4 Tiết học.
3. **Giai đoạn 3:** Chạy `xelatex` để xuất ra PDF `Slide_ML_Chap12.pdf`.
4. **Giai đoạn 4:** Review lại mã nguồn Python trên slide để bảo đảm bố cục thụt lề chuẩn.
