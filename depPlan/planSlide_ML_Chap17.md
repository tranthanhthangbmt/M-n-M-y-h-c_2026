# Kế hoạch chi tiết tạo Slide bài giảng: Chương 17 - Autoencoder, Mạng Đối Kháng Sinh (GAN) và Mô hình Khuếch tán

**Thông tin chung:**
- **Thư mục mục tiêu:** `slideML/`
- **File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)
- **Thời lượng dự kiến:** Khoảng 45-55 Frames (Tối thiểu 40 frames - đáp ứng chuẩn đại học)
- **Số lượng hình ảnh minh họa:** 22 hình minh họa (từ Hình 17-1 đến Hình 17-22).
- **Theme & Color Theme:** Madrid theme, default colortheme
- **Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_17.md` / `CHƯƠNG 17.htm`).

---

## Cấu trúc Slide dự kiến (Total: ~55 slides)

### 0. Slide mở đầu và Mục tiêu (Slides 1-3)
- **Slide 1:** Tiêu đề "Chương 17: Autoencoder, GAN và Mô hình Khuếch tán".
- **Slide 2:** Mục tiêu bài giảng (Khái niệm biểu diễn dữ liệu ẩn, Autoencoder, kiến trúc Mạng sinh đối kháng GAN, cơ chế Mô hình Khuếch tán).
- **Slide 3:** Mục lục tổng quan.

---

### 1. Autoencoder Cơ bản và Biểu diễn Dữ liệu (Slides 4-13)
- **Slide 4:** Biểu diễn dữ liệu hiệu quả: Thí nghiệm trí nhớ cờ vua và nguyên lý nén/mã hóa.
- **Slide 5:** Kiến trúc Autoencoder đơn giản (Encoder & Decoder). *(Sử dụng Hình 17-1)*
- **Slide 6:** Nút thắt cổ chai (Bottleneck) và Autoencoder thiếu hoàn chỉnh (Undercomplete).
- **Slide 7:** Thực hiện PCA bằng Autoencoder tuyến tính.
- **Slide 8:** So sánh kết quả chiếu (Projection) của PCA và Autoencoder. *(Sử dụng Hình 17-2)*
- **Slide 9:** Autoencoder xếp chồng (Stacked Autoencoders) - Cấu trúc sâu hơn.
- **Slide 10:** Kiến trúc mạng Stacked Autoencoder. *(Sử dụng Hình 17-3)*
- **Slide 11:** Triển khai bằng Keras (Ví dụ code ngắn gọn về xây dựng lớp mã hóa và giải mã).
- **Slide 12:** Đánh giá chất lượng tái tạo (Reconstruction Loss). 
- **Slide 13:** Hình ảnh gốc và bản tái tạo trên Fashion MNIST. *(Sử dụng Hình 17-4)*

---

### 2. Kỹ thuật nâng cao với Autoencoder (Slides 14-25)
- **Slide 14:** Trực quan hóa dữ liệu nhiều chiều. Sử dụng t-SNE kết hợp với Autoencoder.
- **Slide 15:** Biểu đồ phân tán của Fashion MNIST qua Autoencoder + t-SNE. *(Sử dụng Hình 17-5)*
- **Slide 16:** Tiền huấn luyện không giám sát (Unsupervised Pretraining).
- **Slide 17:** Các tầng tham gia tiền huấn luyện. *(Sử dụng Hình 17-6)*
- **Slide 18:** Ràng buộc trọng số (Tying Weights) - Kỹ thuật giảm tham số giúp chống quá khớp.
- **Slide 19:** Huấn luyện từng Autoencoder một (Tham lam / Greedy Layer-wise). 
- **Slide 20:** Minh họa phương pháp huấn luyện từng lớp. *(Sử dụng Hình 17-7)*
- **Slide 21:** Autoencoder tích chập (Convolutional Autoencoders) dành cho hình ảnh.
- **Slide 22:** Autoencoder khử nhiễu (Denoising Autoencoders). Nguyên lý ép mô hình phục hồi tính hiệu gốc.
- **Slide 23:** Mô hình với nhiễu Gaussian hoặc Dropout. *(Sử dụng Hình 17-8)*
- **Slide 24:** Kết quả phục hồi ảnh bị nhiễu. *(Sử dụng Hình 17-9)*
- **Slide 25:** Autoencoder thưa (Sparse Autoencoders). Sử dụng độ đo Kullback-Leibler để ép sự thưa thớt. *(Sử dụng Hình 17-10)*

---

### 3. Autoencoder Biến phân (VAEs) (Slides 26-32)
- **Slide 26:** Giới thiệu Mô hình Sinh (Generative Models) và hạn chế của AE truyền thống.
- **Slide 27:** Autoencoder biến phân (Variational Autoencoder) - Sinh dữ liệu qua phân bố xác suất.
- **Slide 28:** Kiến trúc và toán học của VAEs (Sự lấy mẫu ngẫu nhiên). *(Sử dụng Hình 17-11)*
- **Slide 29:** Định lý tái tham số hóa (Reparameterization Trick).
- **Slide 30:** Kết quả tạo ảnh mới (Fashion MNIST) bằng VAEs. *(Sử dụng Hình 17-12)*
- **Slide 31:** Nội suy ngữ nghĩa (Semantic Interpolation) trong không gian ẩn.
- **Slide 32:** Quá trình biến đổi hình ảnh qua việc di chuyển trong không gian ẩn. *(Sử dụng Hình 17-13)*

---

### 4. Mạng Đối kháng Sinh (GANs) (Slides 33-45)
- **Slide 33:** Mạng Đối kháng Sinh (GAN) là gì? Khái niệm Kẻ làm giả (Generator) và Cảnh sát (Discriminator).
- **Slide 34:** Kiến trúc tổng quan của GAN. *(Sử dụng Hình 17-14)*
- **Slide 35:** Vòng lặp huấn luyện GAN (Cập nhật D, cập nhật G). Khái niệm hàm mất mát Minimax.
- **Slide 36:** Những khó khăn khi huấn luyện GAN: Sụp đổ chế độ (Mode Collapse), mất cân bằng.
- **Slide 37:** Kết quả tạo ảnh với Vanilla GAN trên Fashion MNIST. *(Sử dụng Hình 17-15)*
- **Slide 38:** GAN Tích chập Sâu (DCGANs) - Đột phá đầu tiên trong sinh ảnh thực tế.
- **Slide 39:** Hướng dẫn cấu hình DCGAN (Bỏ pooling layer, dùng Batch Normalization, LeakyReLU).
- **Slide 40:** Hình ảnh tạo bởi DCGAN sau 50 epochs. *(Sử dụng Hình 17-16)*
- **Slide 41:** Phép toán Véc-tơ trên Khái niệm Hình ảnh (Người đàn ông đeo kính - Người đàn ông = Kính). *(Sử dụng Hình 17-17)*
- **Slide 42:** Sự phát triển tăng dần của GANs (ProGANs). Tránh cú sốc từ ảnh độ phân giải cao.
- **Slide 43:** Tiến trình từ 4x4 lên 1024x1024. *(Sử dụng Hình 17-18)*
- **Slide 44:** StyleGANs - Quản lý phong cách hình ảnh ở nhiều độ phân giải.
- **Slide 45:** Cấu trúc mạng tạo (Generator) của StyleGAN. *(Sử dụng Hình 17-19)*

---

### 5. Mô hình Khuếch tán (Diffusion Models) (Slides 46-52)
- **Slide 46:** Tổng quan về Mô hình Khuếch tán (Khái niệm hủy hoại dữ liệu và phục hồi).
- **Slide 47:** Quá trình Tiến (Forward/Diffusion Process). Thêm nhiễu Gaussian liên tục.
- **Slide 48:** Quá trình Ngược (Reverse Process). Mạng nơ-ron khử nhiễu dần. *(Sử dụng Hình 17-20)*
- **Slide 49:** Toán học cốt lõi: Chuỗi Markov và lịch trình phương sai (Variance Schedule).
- **Slide 50:** Biểu đồ lịch phương sai nhiễu và tín hiệu còn lại. *(Sử dụng Hình 17-21)*
- **Slide 51:** Ưu điểm của Diffusion Models so với GAN (Tính đa dạng cao, dễ huấn luyện). Denoising Diffusion Probabilistic Models (DDPM).
- **Slide 52:** Hình ảnh chất lượng cao tạo ra bởi DDPM (DALL-E, MidJourney). *(Sử dụng Hình 17-22)*

---

### 6. Tổng kết và Bài tập (Slides 53-55)
- **Slide 53:** Tổng kết: Khi nào dùng Autoencoder, GAN, hay Diffusion?
- **Slide 54:** Hướng dẫn Bài tập và Câu hỏi ôn tập.
- **Slide 55:** Hỏi & Đáp (Q&A).

---

## 3. Kế hoạch Tích hợp Hình ảnh (Hình_17-1 đến Hình_17-22)

- **Thư mục lưu ảnh:** `machineLearningWeb/Figures/CH17`
- **Số lượng:** 22 bức hình (gồm .png và .jpg).
- **Nhiệm vụ:**
  - Lồng ghép đủ 22 hình vào bài giảng.
  - Mỗi hình sẽ có ít nhất 1 slide riêng biệt hoặc kết hợp với layout chia cột để giải thích ý nghĩa kiến trúc (đặc biệt là sơ đồ GAN và mô hình khuếch tán).

---

## 4. Các bước Triển khai thực tế

1. **Bước 1 (Đổi tên & Chuẩn hóa ảnh):** Đảm bảo tất cả 22 ảnh trong `Figures/CH17` có tên đúng định dạng `Hình_17-X.png/.jpg`. (Đã hoàn tất đổi tên `Hình_17-21_13.png` thành `Hình_17-21.png`).
2. **Bước 2 (Viết kịch bản Python):** Tạo script `gen_chap17.py` để tự động chèn nội dung LaTeX (text, code, hình ảnh) dựa theo cấu trúc đã lên kế hoạch. (Lưu ý xử lý linh hoạt extension .png và .jpg).
3. **Bước 3 (Biên dịch LaTeX):** Chạy `xelatex Slide_ML_Chap17.tex` 2 lần để cập nhật mục lục và tham chiếu.
4. **Bước 4 (Kiểm tra & Tinh chỉnh):** Duyệt qua file PDF (dự kiến ~55 trang) để đảm bảo không tràn chữ, không tràn code, và hình ảnh sắc nét.
