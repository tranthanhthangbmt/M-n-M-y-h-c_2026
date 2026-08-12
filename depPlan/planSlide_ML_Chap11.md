# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 11 - Huấn luyện mạng nơ-ron sâu

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **$\geq 40$ Frames (Slides)**  
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_11.md` / `CHƯƠNG 11.htm`).  
**Tích hợp hình ảnh:** Phân bổ toàn bộ 10 hình ảnh cốt lõi (từ `Hình_11-1` đến `Hình_11-10` trong thư mục `Figures/CH11/`) vào các slide tương ứng. Các ảnh dư thừa từ Word export (biểu thức toán học dạng ảnh) sẽ được gõ lại bằng mã LaTeX `\amsmath` hoặc loại bỏ.

---

## 1. Bố cục Phân chương Tiết học (Sections & TOC)

Bộ slide Chương 11 sẽ được chia thành **5 Tiết học (5 Sections)** bao phủ các kỹ thuật huấn luyện Deep Learning tiên tiến:

### Tiết 1 (12 Frames): Vấn đề Gradient biến mất/bùng nổ & Khởi tạo trọng số
- **Mục tiêu:** Hiểu rõ tại sao mạng quá sâu lại khó huấn luyện và cách giải quyết thông qua kỹ thuật khởi tạo trọng số và hàm kích hoạt mới.
- **Nội dung các slide thực tế:**
  1. Trang bìa (`\titlepage`)
  2. Mục lục nội dung chương (`\tableofcontents`)
  3. Mở đầu: Khó khăn khi huấn luyện Mạng nơ-ron sâu
  4. Vấn đề Gradient biến mất (Vanishing Gradients)
  5. Vấn đề Gradient bùng nổ (Exploding Gradients)
  6. Nguyên nhân sâu xa: Hàm Sigmoid và Khởi tạo trọng số
  7. Giải pháp: Khởi tạo Glorot (Xavier Initialization)
  8. Giải pháp: Khởi tạo He (He Initialization)
  9. Các hàm kích hoạt tốt hơn: Hạn chế của ReLU (Dying ReLU)
  10. Hàm Leaky ReLU và PReLU
  11. Hàm ELU (Exponential Linear Unit) và SELU
  12. Hàm GELU, Swish và Mish

### Tiết 2 (10 Frames): Chuẩn hóa Batch (Batch Normalization) & Gradient Clipping
- **Mục tiêu:** Khám phá kỹ thuật Batch Normalization (BN) - một trong những đột phá quan trọng nhất của Deep Learning để ổn định hóa quá trình huấn luyện.
- **Nội dung các slide thực tế:**
  13. Chuẩn hóa Batch (Batch Normalization - BN) là gì?
  14. Cơ chế hoạt động của thuật toán Batch Normalization
  15. Ưu điểm của Batch Normalization (Giảm vanish gradient, tăng learning rate)
  16. Triển khai Batch Normalization với Keras (`keras.layers.BatchNormalization`)
  17. Cấu hình BN trước hay sau hàm kích hoạt?
  18. Tham số `momentum` và `axis` trong BN
  19. Giải quyết vấn đề Gradient bùng nổ: Cắt xén Gradient (Gradient Clipping)
  20. Triển khai Gradient Clipping trong Keras (`clipvalue`, `clipnorm`)
  21. So sánh BN và Gradient Clipping
  22. Tóm tắt các kỹ thuật ổn định gradient

### Tiết 3 (12 Frames): Tái sử dụng mô hình (Transfer Learning) & Tiền huấn luyện
- **Mục tiêu:** Cách tận dụng các mạng nơ-ron đã được huấn luyện sẵn cho các tác vụ mới để tiết kiệm thời gian và dữ liệu.
- **Nội dung các slide thực tế:**
  23. Tái sử dụng các lớp đã huấn luyện (Transfer Learning)
  24. Lợi ích của Transfer Learning (Học chuyển giao)
  25. Kiến trúc khi Transfer Learning (Giữ lại Lower layers, thay thế Upper layers)
  26. Đóng băng trọng số (Freezing layers) bằng `layer.trainable = False`
  27. Fine-tuning (Tinh chỉnh mô hình)
  28. Tốc độ học (Learning rate) trong Fine-tuning
  29. Triển khai Transfer Learning với Keras
  30. Tiền huấn luyện không giám sát (Unsupervised Pretraining)
  31. Ứng dụng của Autoencoders và GANs trong Pretraining
  32. Tiền huấn luyện trên một tác vụ phụ trợ (Auxiliary Task)
  33. Cách tạo nhãn tự động cho tác vụ phụ trợ (Self-supervised learning)
  34. Tổng kết chiến lược tái sử dụng mô hình

### Tiết 4 (12 Frames): Các Trình tối ưu hóa nhanh hơn & Lập lịch Tốc độ học
- **Mục tiêu:** Vượt qua SGD truyền thống bằng các thuật toán tối ưu tiên tiến (Momentum, Adam) và các chiến lược giảm tốc độ học.
- **Nội dung các slide thực tế:**
  35. Các Trình tối ưu hóa (Optimizers) vượt trội hơn SGD
  36. Tối ưu hóa Động lượng (Momentum Optimization)
  37. Tối ưu hóa Nesterov Accelerated Gradient (NAG)
  38. Thuật toán AdaGrad
  39. Thuật toán RMSProp
  40. Tối ưu hóa Adam (Adaptive Moment Estimation)
  41. AdamW và Nadam
  42. So sánh các Trình tối ưu hóa (Bảng tổng hợp)
  43. Lập lịch tốc độ học (Learning Rate Scheduling)
  44. Power Scheduling và Exponential Scheduling
  45. Piecewise Constant Scheduling
  46. Triển khai Lập lịch tốc độ học bằng Keras Callbacks

### Tiết 5 (12 Frames): Tránh Overfitting thông qua Chính quy hóa
- **Mục tiêu:** Áp dụng các kỹ thuật Regularization ($\ell_1$, $\ell_2$, Dropout) để tăng khả năng khái quát hóa của mạng.
- **Nội dung các slide thực tế:**
  47. Tránh Overfitting trong Mạng nơ-ron sâu
  48. Chính quy hóa $\ell_1$ và $\ell_2$ (L1/L2 Regularization)
  49. Triển khai $\ell_1$ / $\ell_2$ trong Keras (`keras.regularizers`)
  50. Kỹ thuật Dropout: Khái niệm cơ bản
  51. Tại sao Dropout lại hiệu quả? (Học biểu diễn mạnh mẽ hơn)
  52. Triển khai Dropout trong Keras (`keras.layers.Dropout`)
  53. Monte Carlo (MC) Dropout: Đánh giá độ không chắc chắn
  54. Chính quy hóa Max-Norm (Max-Norm Regularization)
  55. Triển khai Max-Norm (`keras.constraints.max_norm`)
  56. Bảng cấu hình mạng DNN mặc định (Tóm tắt thực hành)
  57. Tóm tắt Chương 11
  58. Hỏi \& Đáp (Q\&A)

---

## 2. Tiêu chuẩn Kỹ thuật và Nhận diện (Format)
- Bố cục Slide chia làm **2 cột** (`\begin{columns}`) khi cần thiết để hiển thị 10 hình ảnh, như đồ thị so sánh gradient descent hoặc mô phỏng AutoGraph.
- Code Python minh họa sẽ dùng môi trường `\begin{lstlisting}[language=Python]` với cấu hình `[fragile]`, giúp hiển thị rõ ràng và đẹp mắt.
- Chú thích ảnh sẽ dùng chuẩn `\vspace{0.2cm}\textit{Hình 11-X: ...}` để tránh việc bị lặp từ "Hình".
- Chèn code Python vào Block `lstlisting` với highlight màu sắc đồng bộ như Chương 10 cho các đoạn code triển khai Batch Normalization, Dropout, và Optimizers.
- Dùng `\usepackage{fontspec}` thiết lập font tiếng Việt.
- **Thông tin tác giả:** `\author{Giảng viên: TS. Trần Thành Thắng}`.
- **Biên dịch:** Chạy `xelatex` 2 lần liên tiếp để tạo chỉ mục và TOC chính xác.

---

## 3. Quy trình thực hiện dự kiến
1. **Giai đoạn 1:** Soạn kịch bản Python tạo slide (`rename_images_ch11.py`) để lọc và chuẩn hóa tên 13 hình ảnh (`Hinh_11-1.png` đến `Hinh_11-13.png`).
2. **Giai đoạn 2:** Viết kịch bản `v13_generate_slides_ch11.py` để sinh mã LaTeX cho 5 Tiết học dựa trên kịch bản chi tiết này.
3. **Giai đoạn 3:** Chạy `xelatex` lần 1 và lần 2 để xuất file `Slide_ML_Chap11.pdf`.
4. **Giai đoạn 4:** Kiểm thử (Review) hiển thị của các phương trình toán học và highlight syntax Python. Tinh chỉnh (nếu có).
