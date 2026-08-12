# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 10 - Giới thiệu về mạng nơ-ron nhân tạo với Keras

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **~55-60 Frames (Slides)** (Nằm trong khoảng 40 - 60 slides theo yêu cầu).  
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_10.md`).  
**Tích hợp hình ảnh:** Phân bổ các hình ảnh (từ `Hinh_10-1` đến `Hinh_10-66` trong thư mục `Figures/CH10/`) vào các slide tương ứng. *(Do chương 10 có lượng hình ảnh lớn, một số ảnh code/biểu đồ sẽ được gộp chung hoặc lược bỏ bớt các hình không quá quan trọng).*

---

## 1. Bố cục Phân chương Tiết học (Sections & TOC)

Bộ slide Chương 10 được chia thành **5 Tiết học (5 Sections)** rõ ràng với tổng cộng khoảng 60 Frames:

### Tiết 1 (12 Frames): Từ Nơ-ron Sinh học đến Nơ-ron Nhân tạo
- **Mục tiêu:** Hiểu bối cảnh ra đời của mạng nơ-ron nhân tạo (ANN), khám phá cơ chế hoạt động của nơ-ron sinh học và kiến trúc Perceptron.
- **Nội dung các slide thực tế:**
  1. Trang bìa (`\titlepage`)
  2. Mục lục nội dung chương (`\tableofcontents`)
  3. Giới thiệu Mạng Nơ-ron nhân tạo (ANN)
  4. Lịch sử và Sự trỗi dậy của Deep Learning
  5. Từ Nơ-ron Sinh học đến Nhân tạo
  6. Mạng nơ-ron sinh học
  7. Tính toán logic với Nơ-ron (Mô hình McCulloch và Pitts)
  8. Perceptron là gì?
  9. Kiến trúc của Perceptron (TLU - Threshold Logic Unit)
  10. Huấn luyện Perceptron: Quy tắc học Hebbian
  11. Hạn chế của Perceptron (Bài toán XOR)
  12. Giải quyết XOR với Mạng Perceptron đa lớp

### Tiết 2 (12 Frames): Mạng Perceptron đa lớp (MLP) & Backpropagation
- **Mục tiêu:** Nắm vững cấu trúc Mạng nơ-ron tiến (FNN) và giải thuật cốt lõi: Lan truyền ngược (Backpropagation). Hiểu kiến trúc cho Hồi quy và Phân loại.
- **Nội dung các slide thực tế:**
  13. Mạng Perceptron đa lớp (MLP)
  14. Mạng nơ-ron tiến (Feedforward Neural Network - FNN)
  15. Backpropagation (Lan truyền ngược) là gì?
  16. Cơ chế hoạt động của Backpropagation
  17. Các hàm kích hoạt (Activation functions)
  18. Trực quan hóa các hàm kích hoạt (Sigmoid, Tanh, ReLU)
  19. MLP cho Hồi quy (Regression)
  20. Cấu trúc MLP Hồi quy
  21. MLP cho Phân loại (Classification)
  22. Phân loại nhị phân và đa nhãn
  23. Hàm kích hoạt Softmax cho phân loại đa lớp
  24. So sánh cấu trúc MLP Hồi quy và Phân loại

### Tiết 3 (12 Frames): Triển khai MLP với Keras & API Tuần tự
- **Mục tiêu:** Giới thiệu thư viện Keras, TensorFlow và cách xây dựng bộ phân loại hình ảnh (Fashion MNIST) bằng API tuần tự (Sequential API).
- **Nội dung các slide thực tế:**
  25. Triển khai MLP với TensorFlow và Keras
  26. Keras là gì?
  27. Tải bộ dữ liệu Fashion MNIST
  28. Xây dựng bộ phân loại hình ảnh bằng API Tuần tự (Sequential API)
  29. Xem cấu trúc mô hình (model.summary())
  30. Truy cập các lớp và trọng số
  31. Biên dịch (Compile) mô hình (Loss, Optimizer, Metrics)
  32. Huấn luyện và Đánh giá mô hình (fit)
  33. Phân tích lịch sử huấn luyện (Learning curves)
  34. Hiện tượng Overfitting và Underfitting trên biểu đồ
  35. Đánh giá trên tập Kiểm thử (evaluate)
  36. Đưa ra dự đoán trên dữ liệu mới (predict)

### Tiết 4 (12 Frames): Xây dựng mô hình phức tạp với Functional API & Subclassing
- **Mục tiêu:** Hiểu giới hạn của API tuần tự và làm quen với API chức năng (Functional API) để xây dựng các mô hình phức tạp (Wide & Deep).
- **Nội dung các slide thực tế:**
  37. Xây dựng mô hình bằng API Chức năng (Functional API)
  38. Kiến trúc Wide & Deep
  39. Mã nguồn: Xây dựng mô hình Wide & Deep
  40. Mô hình với nhiều đầu vào (Multiple Inputs)
  41. Mô hình với nhiều đầu ra (Multiple Outputs)
  42. Ứng dụng của kiến trúc nhiều đầu ra (Regularization, Multi-task)
  43. Xây dựng mô hình Động với Subclassing API
  44. Cách kế thừa từ tf.keras.Model
  45. So sánh: Sequential, Functional, và Subclassing
  46. Lưu và Khôi phục Mô hình
  47. Cách lưu toàn bộ mô hình (HDF5 / SavedModel)
  48. Sử dụng Callbacks (ModelCheckpoint, EarlyStopping)

### Tiết 5 (12 Frames): Sử dụng TensorBoard & Tinh chỉnh Siêu tham số
- **Mục tiêu:** Giám sát quá trình huấn luyện bằng TensorBoard và học cách tinh chỉnh các siêu tham số (Hyperparameters) của mạng nơ-ron.
- **Nội dung các slide thực tế:**
  49. Giám sát bằng TensorBoard
  50. Cách sử dụng TensorBoard callback
  51. Tinh chỉnh các siêu tham số mạng nơ-ron
  52. Khó khăn trong việc chọn siêu tham số
  53. KerasRegressor / KerasClassifier wrapper
  54. Tìm kiếm siêu tham số bằng RandomizedSearchCV
  55. Keras Tuner
  56. Số lượng lớp ẩn (Hidden Layers)
  57. Số lượng nơ-ron mỗi lớp
  58. Learning Rate, Batch Size, và Optimizer
  59. Tổng kết Chương 10
  60. Hỏi & Đáp (Q&A)

---

## 2. Tiêu chuẩn Kỹ thuật và Nhận diện (Format)
- Bố cục Slide chia làm **2 cột** (`\begin{columns}`) khi cần thiết để chèn hình ảnh minh họa bên phải và chữ bên trái, đặc biệt quan trọng với chương có đến 66 hình.
- Các hình ảnh code / kết quả tensorboard có thể được scale nhỏ bằng `\includegraphics[width=\textwidth,height=0.65\textheight,keepaspectratio]` hoặc ghép 2 ảnh song song để tiết kiệm diện tích và tăng trực quan.
- Chèn code Python vào Block `lstlisting` với highlight màu sắc dễ nhìn cho các thao tác Keras (Xây dựng, Compile, Fit).
- Dùng `\usepackage{fontspec}` với các thiết lập font mặc định để biên dịch tốt với **XeLaTeX** (không hardcode Times New Roman / Arial để tránh lỗi mất chữ như đã từng gặp ở các chương trước).
- **Thông tin tác giả:** `\author{Giảng viên: TS. Trần Thành Thắng}`.
- **Biên dịch:** Chạy `xelatex` 2 lần liên tiếp.

## 3. Quy trình thực hiện dự kiến
1. **Giai đoạn 1:** Soạn kịch bản Python tạo slide (`v12_generate_slides_ch10.py` hoặc tạo thủ công tex file). Trích xuất nội dung từ `CHƯƠNG 10.htm` / `chuong_10.md`.
2. **Giai đoạn 2:** Ánh xạ các hình ảnh `Hinh_10-X` vào các slide. Với các hình ảnh code hoặc hình không quá quan trọng có thể bỏ qua để tối ưu không gian slide.
3. **Giai đoạn 3:** Chạy lệnh `xelatex` 2 lần để biên dịch PDF và rà soát chất lượng dàn trang.
4. **Giai đoạn 4:** Sửa lỗi tràn viền hoặc bố cục hình ảnh nếu có.

*Kế hoạch này bám sát quy chuẩn của Chương 1-9 và tập trung giải quyết bài toán khối lượng hình ảnh lớn của Chương 10.*
