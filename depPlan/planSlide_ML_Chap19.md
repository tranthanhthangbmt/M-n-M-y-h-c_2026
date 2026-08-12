# Kế hoạch chi tiết tạo Slide bài giảng: Chương 19 - Huấn luyện và Triển khai Mô hình TensorFlow Quy mô Lớn

**Thông tin chung:**
- **Thư mục mục tiêu:** `slideML/`
- **File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)
- **Thời lượng dự kiến:** Khoảng 45-55 Frames (Tối thiểu 40 frames - đáp ứng chuẩn đại học)
- **Số lượng hình ảnh minh họa:** 18 hình minh họa (từ Hình 19-1 đến Hình 19-18).
- **Theme & Color Theme:** Madrid theme, default colortheme
- **Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_19.md` / `chương 19.htm`).

---

## 1. Cấu trúc Slide dự kiến (Total: ~50 slides)

### 1.1 Slide mở đầu và Mục tiêu (Slides 1-3)
- **Slide 1:** Tiêu đề "Chương 19: Huấn luyện và Triển khai Mô hình TensorFlow Quy mô Lớn".
- **Slide 2:** Mục tiêu bài giảng (Triển khai mô hình lên production với TF Serving; Đưa AI lên Mobile/Web với TFLite \& TF.js; Phân tán tính toán với Multi-GPU và TPU).
- **Slide 3:** Mục lục tổng quan.

---

### 1.2 Phục vụ Mô hình TensorFlow (Slides 4-13)
- **Slide 4:** Tổng quan vòng đời một dự án Machine Learning (MLOps): Từ huấn luyện (Train) đến phục vụ (Serving).
- **Slide 5:** Các khó khăn khi đưa AI vào môi trường Production thực tế (Độ trễ, Khả năng mở rộng, Cập nhật mô hình).
- **Slide 6:** Giới thiệu TensorFlow Serving: Công cụ chuyên dụng của Google để phục vụ mô hình ML hiệu năng cao. *(Sử dụng Hình 19-1)*
- **Slide 7:** Kiến trúc của TensorFlow Serving: Hỗ trợ nạp mô hình mới mà không làm gián đoạn hệ thống.
- **Slide 8:** Xuất mô hình (Export Model): Sử dụng định dạng `SavedModel`. 
- **Slide 9:** API REST vs gRPC trong TF Serving: Khi nào dùng cái nào? *(Sử dụng Hình 19-2)*
- **Slide 10:** Tạo dịch vụ dự đoán (Prediction Service) trên nền tảng đám mây Google Cloud Vertex AI. *(Sử dụng Hình 19-3)*
- **Slide 11:** Khởi tạo Endpoint và triển khai mô hình lên Vertex AI bằng Python SDK.
- **Slide 12:** Chạy các công việc Dự đoán Hàng loạt (Batch Predictions) trên Vertex AI khi không cần độ trễ thấp.
- **Slide 13:** Lợi ích của việc dùng Cloud (Vertex AI) so với việc tự xây máy chủ cục bộ.

---

### 1.3 Triển khai Mô hình lên Thiết bị Di động (Mobile) \& Web (Slides 14-22)
- **Slide 14:** Tại sao cần đưa AI lên thiết bị người dùng (Edge AI)? (Bảo mật, Offline, Tiết kiệm băng thông, Độ trễ cực thấp). *(Sử dụng Hình 19-4)*
- **Slide 15:** Giới thiệu TensorFlow Lite (TFLite): Bộ công cụ cho Mobile (Android/iOS) và Thiết bị nhúng (IoT/Raspberry Pi).
- **Slide 16:** Các bước sử dụng TFLite: Train $\rightarrow$ Convert $\rightarrow$ Deploy. *(Sử dụng Hình 19-5)*
- **Slide 17:** Tối ưu hóa mô hình (Model Optimization): Quantization (Lượng tử hóa từ Float32 xuống Int8).
- **Slide 18:** Tối ưu hóa mô hình: Pruning (Cắt tỉa mạng) và Clustering. *(Sử dụng Hình 19-6)*
- **Slide 19:** Kiến trúc TFLite Interpreter trên thiết bị di động.
- **Slide 20:** Chạy Mô hình trực tiếp trong Trang Web: Giới thiệu TensorFlow.js (TF.js). *(Sử dụng Hình 19-7)*
- **Slide 21:** Ưu điểm của TF.js: Không cần cài đặt, tận dụng WebGL/WebGPU để tăng tốc ngay trên trình duyệt.
- **Slide 22:** Luồng xử lý chuyển đổi mô hình từ Keras/SavedModel sang định dạng của TF.js.

---

### 1.4 Sử dụng GPU để Tăng tốc Tính toán (Slides 23-32)
- **Slide 23:** Sự cần thiết của phần cứng chuyên dụng (GPU, TPU) trong Deep Learning.
- **Slide 24:** CPU vs GPU vs TPU: Sự khác biệt trong cấu trúc xử lý (Ít nhân siêu mạnh vs Hàng ngàn nhân yếu hơn). *(Sử dụng Hình 19-8)*
- **Slide 25:** Cách có được GPU của riêng bạn (Mua cục bộ, Thuê Cloud VM, dùng Google Colab/Kaggle miễn phí).
- **Slide 26:** Quản lý RAM GPU trong TensorFlow: Ngăn TF chiếm dụng toàn bộ VRAM của máy (`memory_growth`). *(Sử dụng Hình 19-9)*
- **Slide 27:** Đặt các phép toán (Operations) và Biến (Variables) lên thiết bị cụ thể (`tf.device('/GPU:0')`).
- **Slide 28:** Thực thi song song trên nhiều thiết bị: Làm sao TensorFlow chạy nhiều phép tính cùng lúc?
- **Slide 29:** Huấn luyện mô hình trên nhiều thiết bị (Multi-Device Training): Các chiến lược mở rộng. *(Sử dụng Hình 19-10)*
- **Slide 30:** Mô hình song song (Model Parallelism): Chia một mạng nơ-ron khổng lồ ra nhiều GPU. *(Sử dụng Hình 19-11)*
- **Slide 31:** Hạn chế của Mô hình song song (Chi phí truyền tải dữ liệu giữa các GPU rất lớn).
- **Slide 32:** Song song Dữ liệu (Data Parallelism): Copy y hệt mô hình lên các GPU, mỗi GPU học một phần nhỏ dữ liệu. *(Sử dụng Hình 19-12)*

---

### 1.5 Cập nhật Đồng bộ và Cụm TensorFlow (Slides 33-46)
- **Slide 33:** Song song Dữ liệu Đồng bộ (Synchronous Data Parallelism): Bắt buộc các GPU phải đợi nhau để tính tổng Gradient (All-Reduce). *(Sử dụng Hình 19-13)*
- **Slide 34:** Song song Dữ liệu Bất đồng bộ (Asynchronous Data Parallelism): Dùng máy chủ tham số (Parameter Servers). *(Sử dụng Hình 19-14)*
- **Slide 35:** So sánh Đồng bộ và Bất đồng bộ (Cái nào hội tụ nhanh hơn, cái nào ít bị "cổ chai" hơn).
- **Slide 36:** Huấn luyện Quy mô lớn bằng API \texttt{tf.distribute.Strategy}: Viết code 1 lần, chạy trên 1 GPU hay 100 TPU đều được!
- **Slide 37:** Cụ thể về `MirroredStrategy`: Copy mô hình, đồng bộ Gradient. *(Sử dụng Hình 19-15)*
- **Slide 38:** Cụ thể về `MultiWorkerMirroredStrategy`: Huấn luyện trên nhiều máy tính khác nhau qua mạng LAN.
- **Slide 39:** Huấn luyện Mô hình trên Cụm TensorFlow (TensorFlow Cluster). *(Sử dụng Hình 19-16)*
- **Slide 40:** Cấu hình biến môi trường `TF_CONFIG` để thiết lập mạng máy tính huấn luyện AI.
- **Slide 41:** Chạy các tác vụ huấn luyện LỚN trên nền tảng Vertex AI.
- **Slide 42:** Tạo Docker Image chứa code huấn luyện và đẩy lên Google Container Registry (GCR). *(Sử dụng Hình 19-17)*
- **Slide 43:** Kích hoạt Custom Training Job trên Google Cloud.
- **Slide 44:** Điều chỉnh Siêu tham số (Hyperparameter Tuning) tự động trên Vertex AI. *(Sử dụng Hình 19-18)*
- **Slide 45:** Dịch vụ AutoML: Tự động hóa hoàn toàn quy trình tạo mô hình.
- **Slide 46:** Xu hướng tương lai của việc mở rộng quy mô Huấn luyện AI.

---

### 1.6 Tổng kết và Bài tập (Slides 47-50)
- **Slide 47:** Bảng tổng hợp: Chọn giải pháp Deploy nào (Serving, Lite, Web) tùy vào bài toán thực tế.
- **Slide 48:** Bảng tổng hợp: Khi nào cần Multi-GPU, khi nào cần Multi-Worker, và khi nào nên dùng TPU.
- **Slide 49:** Hướng dẫn Bài tập thực hành: Triển khai mô hình REST API với FastAPI/TF Serving.
- **Slide 50:** Hỏi & Đáp (Q&A).

---

## 2. Kỹ thuật triển khai

- **Script sử dụng:** `gen_chap19.py`
- **File TeX kết quả:** `Slide_ML_Chap19.tex`
- **Tránh lỗi Unicode và Dấu câu TeX:** Toàn bộ kịch bản sinh mã TeX sẽ được bọc bởi `r''' ... '''` trong script và dùng `xelatex` để hỗ trợ font Arial tiếng Việt.
- **Xử lý Code:** Sử dụng `[fragile]` để tích hợp các đoạn mã huấn luyện API phân tán và TensorFlow Serving minh họa.

---

## 3. Kế hoạch Tích hợp Hình ảnh (Hình_19-1 đến Hình_19-18)

- **Thư mục lưu ảnh:** `machineLearningWeb/Figures/CH19`
- **Số lượng:** 18 bức hình (tất cả định dạng .png). Đã được đổi tên tự động hóa và chuẩn hóa thứ tự từ Hình 1 đến Hình 18.
- **Nhiệm vụ:**
  - Chèn toàn bộ 18 hình vào các phân đoạn giới thiệu kiến trúc TensorFlow Serving, Cụm phần cứng GPU/TPU, và sơ đồ Distribution Strategies.
  - Sử dụng layout chia cột `\begin{columns}` để giúp người học vừa đọc lý thuyết vừa quan sát sơ đồ.

---

## 4. Các bước Triển khai thực tế

1. **Bước 1 (Đổi tên & Chuẩn hóa ảnh):** Đảm bảo tất cả 18 ảnh trong `Figures/CH19` có tên đúng định dạng `Hình_19-X.png/.jpg`. (Đã hoàn tất đổi tên và chuẩn hóa thứ tự từ Hình 1 đến 18).
2. **Bước 2 (Viết kịch bản Python):** Tạo script `gen_chap19.py` để tự động chèn nội dung LaTeX (text, code, hình ảnh) dựa theo cấu trúc đã lên kế hoạch. (Lưu ý xử lý linh hoạt extension .png và .jpg).
3. **Bước 3 (Biên dịch LaTeX):** Chạy `xelatex Slide_ML_Chap19.tex` 2 lần để cập nhật mục lục và tham chiếu.
4. **Bước 4 (Kiểm tra & Tinh chỉnh):** Duyệt qua file PDF (dự kiến ~50 trang) để đảm bảo không tràn chữ, không tràn code, và hình ảnh sắc nét.
