# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 14 - Thị giác Máy tính Chuyên sâu sử dụng Mạng Nơ-ron Tích chập (CNN)

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Thời lượng dự kiến:** 4 Tiết học (khoảng 45-60 frames)
**Số lượng hình ảnh minh họa:** Chính xác 29 hình cốt lõi (Từ Hình 14-1 đến Hình 14-29)
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_14.md` / `CHƯƠNG 14.htm`).

---

## 1. Cấu trúc Nội dung (Dàn ý chi tiết)

Bộ slide sẽ được chia thành **7 phần chính** bám sát theo đúng tài liệu gốc:

### Phần 1: Kiến trúc của vỏ não thị giác (14.1)
- Giới thiệu về vỏ não thị giác và sự tương đồng với kiến trúc mạng lưới máy tính.
- Tại sao lại cần Mạng nơ-ron Tích chập (CNN).
- Bài toán nhận diện hình ảnh và những hạn chế của DNN thông thường.

### Phần 2: Các lớp tích chập & Lớp gộp (14.2)
- Khái niệm về các lớp tích chập (Convolutional Layers).
- Filters (Bộ lọc) và Feature Maps (Bản đồ đặc trưng).
- Các tham số cốt lõi: Padding (Đệm) và Strides (Bước nhảy).
- Lớp gộp (Pooling layers): Max Pooling, Average Pooling, Global Average Pooling.

### Phần 3: Kiến trúc CNN tiêu biểu (14.3)
- Lịch sử và sự phát triển của các kiến trúc CNN (LeNet-5, AlexNet, GoogLeNet, ResNet,...).
- Triển khai một CNN ResNet-34 bằng Keras từ đầu (Code chi tiết).
- Sử dụng các mô hình đã được huấn luyện sẵn (Pretrained models) từ Keras (Transfer Learning).

### Phần 4: Phân loại và định vị (14.4)
- Phân loại hình ảnh (Classification) kết hợp với định vị đối tượng (Localization).
- Dự đoán bounding box (Khung giới hạn).
- Hàm mất mát cho bài toán định vị (MSE, GIoU).

### Phần 5: Phát hiện đối tượng & Theo dõi (14.5)
- Phát hiện nhiều đối tượng (Object Detection): Giới thiệu về YOLO (You Only Look Once).
- Chỉ số đánh giá: Intersection over Union (IoU), Mean Average Precision (mAP).
- Kỹ thuật Non-Max Suppression.
- Theo dõi đối tượng (Object Tracking) qua các khung hình.

### Phần 6: Phân đoạn ngữ nghĩa (14.6)
- Bài toán Semantic Segmentation (Gán nhãn cho từng pixel).
- Kiến trúc Fully Convolutional Networks (FCN).
- Các ứng dụng thực tế của phân đoạn ảnh (Y tế, xe tự lái).

### Phần 7: Tổng kết & Bài tập (14.7)
- Ôn tập những khái niệm cốt lõi.
- Bài tập ứng dụng thực hành phân loại hình ảnh hoặc dùng mô hình pretrained.

---

## 2. Kế hoạch Cắt & Trích xuất Hình ảnh (Hình_14-...)

Chương 14 đặc thù có rất nhiều hình ảnh liên quan đến thị giác máy tính, kiến trúc mô hình. Các hình ảnh đã được chuẩn bị:

- **Thư mục lưu ảnh:** `machineLearningWeb/Figures/CH14`
- **Kết quả:** Đã có chính xác 29 ảnh từ `Hình_14-1` đến `Hình_14-29`.
- **Chèn vào slide:** Các hình ảnh quá rộng sẽ được tách sang slide riêng để đảm bảo sinh viên ngồi xa vẫn thấy được chi tiết (kế thừa kinh nghiệm từ Chương 13, 15). Đặc biệt sử dụng `\vspace{0.2cm}\textit{Hình 14-X: ...}` để tránh bị lặp chữ "Hình".

---

## 3. Kế hoạch Code LaTeX & Beamer Cụ thể

- **Tệp nguồn:** `Slide_ML_Chap14.tex`
- **Theme:** Madrid (kế thừa từ các chương trước), với aspect ratio 16:9 widescreen.
- **Bố cục (Layout):**
  - Sử dụng block, alertblock cho các định nghĩa (Ví dụ: Định nghĩa Convolution, Pooling, IoU).
  - Sử dụng `columns` để chia đôi màn hình: Một bên mô tả các tham số của CNN, một bên hiện ảnh minh họa/kiến trúc mạng.
- **Mã nguồn (Code Snippets):**
  - Sử dụng môi trường `\begin{lstlisting}[language=Python]` kèm tuỳ chọn `[fragile]` cho khung chứa code.
  - Các code tiêu biểu: Code khởi tạo lớp `Conv2D`, `MaxPool2D`, xây dựng kiến trúc `ResNet-34` và cách load `ResNet50` pretrained từ Keras.

---

## 4. Các bước Triển khai thực tế (Quy trình 3 bước)

1. **Bước 1 (Xử lý dữ liệu & Ảnh):** 
   - Code và chạy script `v16_generate_slides_ch14.py` để rút trích toàn bộ ảnh có trong Chương 14.htm. Đổi tên đúng định dạng `Hình_14-x.jpg/png` và lưu vào `Figures/CH14`.
   
2. **Bước 2 (Viết LaTeX Source):**
   - Soạn thảo nội dung chi tiết cho `Slide_ML_Chap14.tex`. Đưa nội dung text vào, đặt mã code Python vào block `lstlisting` và link đúng đường dẫn ảnh tương ứng từ thư mục `Figures/CH14`.
   - Lưu ý đặc biệt các slide chứa hình kiến trúc mạng (như GoogLeNet, ResNet) sẽ được để trên một slide độc lập để ảnh to và dễ nhìn nhất.
   
3. **Bước 3 (Biên dịch & Khắc phục lỗi):**
   - Chạy lệnh biên dịch `xelatex Slide_ML_Chap14.tex` (chạy 2 lần để mục lục và đánh trang được khớp).
   - Kiểm tra PDF, khắc phục lỗi tràn chữ, ảnh bị bóp méo, hoặc lỗi biên dịch liên quan đến font/ký tự UTF-8.

---

## 5. Tiêu chuẩn Đánh giá (Nghiệm thu)

- File `Slide_ML_Chap14.pdf` hiển thị hoàn hảo ở tỷ lệ 16:9.
- Mục lục click được, các slide có số trang.
- Đầy đủ kiến thức về CNN (Convolution, Pooling) và các mô hình thị giác máy tính kinh điển.
- Code xây dựng ResNet-34 và sử dụng Pretrained model hiển thị có màu (syntax highlighting), không bị lỗi font hay tràn dòng.
- Không có trang trắng, hình ảnh kiến trúc mô hình to, rõ, dễ đọc.
