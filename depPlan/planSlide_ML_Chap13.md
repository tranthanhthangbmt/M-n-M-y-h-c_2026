# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 13 - Tải và Tiền xử lý dữ liệu với TensorFlow

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Thời lượng dự kiến:** 4 Tiết học (khoảng 40-45 frames)
**Số lượng hình ảnh minh họa:** Chính xác 07 hình cốt lõi (Từ Hình 13-1 đến Hình 13-7)
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_13.md` / `CHƯƠNG 13.htm`).  
**Tích hợp hình ảnh:** Phân bổ 7 hình ảnh cốt lõi (từ `Hinh_13-1` đến `Hinh_13-7` trong thư mục `Figures/CH13/`) vào các slide tương ứng. 

---

## 1. Bố cục Phân chương Tiết học (Sections & TOC)

Bộ slide Chương 13 sẽ được chia thành **4 Tiết học (4 Sections)** bao phủ toàn bộ các phương thức nạp và tiền xử lý dữ liệu quy mô lớn của TensorFlow:

### Tiết 1 (12 Frames): API `tf.data` và Tiền xử lý Luồng dữ liệu (Data Pipeline)
- **Mục tiêu:** Hiểu rõ cách `tf.data.Dataset` hoạt động, cách tạo và thao tác chuỗi (chaining) các phương thức biến đổi dữ liệu.
- **Nội dung các slide thực tế:**
  1. Trang bìa (`\titlepage`)
  2. Mục lục nội dung chương (`\tableofcontents`)
  3. Tại sao cần API `tf.data`? (Vượt qua giới hạn RAM, xử lý song song, nạp dữ liệu khổng lồ)
  4. Khởi tạo Dataset từ RAM: `tf.data.Dataset.from_tensor_slices`
  5. Chuỗi biến đổi (Chaining Transformations): `map()`, `filter()`
  6. Các phương thức trộn và gom nhóm: `shuffle()`, `batch()`, `repeat()`
  7. Ví dụ: Xây dựng Pipeline tiền xử lý hoàn chỉnh
  8. Cải thiện hiệu suất với Tìm nạp trước (Prefetching)
  9. Minh họa cơ chế Prefetch (Kết hợp Hình minh họa GPU/CPU song song)
  10. Nạp dữ liệu từ nhiều tệp (Interleaving)
  11. Đọc và phân tích cú pháp tệp CSV với `tf.data`
  12. Sử dụng Dataset trực tiếp với Keras (`model.fit`)

### Tiết 2 (12 Frames): Định dạng TFRecord và TensorFlow Protobufs
- **Mục tiêu:** Nắm bắt định dạng lưu trữ nhị phân tối ưu nhất của TensorFlow để tăng tốc độ I/O (Input/Output).
- **Nội dung các slide thực tế:**
  13. Giới thiệu TFRecord: Định dạng nhị phân nhẹ và cực nhanh
  14. Viết dữ liệu vào tệp TFRecord (`tf.io.TFRecordWriter`)
  15. Đọc dữ liệu từ tệp TFRecord (`tf.data.TFRecordDataset`)
  16. Nén tệp TFRecord (Tùy chọn nén GZIP)
  17. Giới thiệu ngắn về Protocol Buffers (Protobuf)
  18. Cấu trúc Protobuf đơn giản
  19. Protobufs của TensorFlow (Tập trung vào `tf.train.Example`)
  20. Cấu trúc `tf.train.Example` (Gồm Features: BytesList, FloatList, Int64List)
  21. Phân tích cú pháp (Parsing) chuỗi Protobuf (`tf.io.parse_single_example`)
  22. Phân tích hàng loạt (Batch Parsing)
  23. Xử lý dữ liệu ảnh với TFRecord
  24. Ví dụ: Nạp và giải mã ảnh Jpeg từ TFRecord

### Tiết 3 (10 Frames): Các Lớp Tiền xử lý của Keras (Keras Preprocessing Layers)
- **Mục tiêu:** Cách đính kèm trực tiếp các công đoạn tiền xử lý (chuẩn hóa, mã hóa) vào ngay bên trong Mô hình thay vì làm bên ngoài.
- **Nội dung các slide thực tế:**
  25. Lớp Tiền xử lý Keras là gì? (Tích hợp xử lý vào Model)
  26. Lớp chuẩn hóa (Normalization Layer)
  27. Cập nhật trạng thái bằng hàm `adapt()`
  28. Mã hóa đặc trưng phân loại (Categorical Features)
  29. Biến đổi dữ liệu phân loại thành One-Hot Vectors (`StringLookup`, `CategoryEncoding`)
  30. Xử lý đặc trưng chữ (Text Preprocessing)
  31. Mã hóa văn bản với `TextVectorization`
  32. Ứng dụng: Nhúng từ (Word Embeddings) với `Embedding` layer
  33. Xử lý dữ liệu ảnh (Image Preprocessing)
  34. Các lớp tăng cường dữ liệu ảnh (Data Augmentation)

### Tiết 4 (10 Frames): Dự án TensorFlow Datasets (TFDS)
- **Mục tiêu:** Khám phá bộ sưu tập dữ liệu công khai đồ sộ đã được định dạng sẵn cho TensorFlow.
- **Nội dung các slide thực tế:**
  35. Giới thiệu TensorFlow Datasets (TFDS)
  36. Cài đặt và sử dụng cơ bản (`tfds.load`)
  37. Tải và duyệt qua tập dữ liệu MNIST bằng TFDS
  38. Tham số `as_supervised=True` (Lấy trực tiếp Tuple `(features, label)`)
  39. Khám phá Meta-data (Thông tin mô tả tập dữ liệu)
  40. Tách tập train/test/validation nâng cao bằng chuỗi Slicing (e.g. `split='train[:75%]'`)
  41. Ví dụ: Tiền xử lý dữ liệu sau khi tải từ TFDS
  42. TFDS và Keras Models: Kết hợp hoàn hảo
  43. Hướng dẫn viết Custom TFDS (Sơ lược)
  44. Tóm tắt Chương 13 \& Hỏi Đáp (Q\&A)

---

## 2. Tiêu chuẩn Kỹ thuật và Nhận diện (Format)
- Bố cục Slide chia làm **2 cột** (`\begin{columns}`) để chèn vừa mã nguồn Python và 7 hình ảnh (chẳng hạn sơ đồ luồng dữ liệu, TFDS).
- Đặt thẻ `[fragile]` vào TẤT CẢ các slide chứa khối mã nguồn `lstlisting` nhằm hạn chế tối đa các lỗi ngớ ngẩn do ký tự đặc biệt gây ra với `xelatex`.
- Sử dụng lệnh `\texttt{}` cho các từ khóa Python/TensorFlow trong dòng văn bản thay vì dùng cặp dấu backtick (`).
- Dùng `\usepackage{fontspec}` thiết lập font tiếng Việt.
- **Thông tin tác giả:** `\author{Giảng viên: TS. Trần Thành Thắng}`.

## 3. Quy trình thực hiện dự kiến
1. **Giai đoạn 1:** Viết script Python (`rename_images_ch13.py`) để lấy và đổi tên 7 ảnh từ thư mục dữ liệu nguồn vào `machineLearningWeb/Figures/CH13/`.
2. **Giai đoạn 2:** Viết kịch bản `v15_generate_slides_ch13.py` sinh mã LaTeX cho 4 Tiết học dựa trên sườn nội dung trên.
3. **Giai đoạn 3:** Chạy `xelatex` 2 lần liên tiếp để xuất ra PDF `Slide_ML_Chap13.pdf` kèm mục lục.
4. **Giai đoạn 4:** Cung cấp thông báo hoàn thiện cho người dùng.
