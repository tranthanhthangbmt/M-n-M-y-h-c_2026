# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 16 - Xử lý Ngôn ngữ Tự nhiên với RNN và Cơ chế Chú ý

**Thư mục mục tiêu:** slideML/
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (\documentclass[aspectratio=169]{beamer})
**Thời lượng dự kiến:** Khoảng 45-55 Frames (Tối thiểu 40 frames - đáp ứng chuẩn đại học)
**Số lượng hình ảnh minh họa:** 12 hình cốt lõi (Từ Hình 16-1 đến Hình 16-12)
**Theme & Color Theme:** Madrid theme, default colortheme
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (machineLearningWeb/docs/chuong_16.md / CHƯƠNG 16.htm).

---

## 1. Cấu trúc Nội dung (Dàn ý chi tiết)

Bộ slide sẽ được chia thành các phần chính bám sát theo tài liệu gốc (Chương 16):

### Phần 1: Tạo văn bản kiểu Shakespeare với RNN ký tự
- Giới thiệu bài toán mô hình hóa ngôn ngữ.
- Chuẩn bị dữ liệu và xây dựng tập dữ liệu (Dataset).
- Xây dựng và huấn luyện mô hình RNN ký tự (Char-RNN).
- Sinh văn bản giả (Fake text generation).

### Phần 2: Phân tích cảm xúc (Sentiment Analysis)
- Giới thiệu bài toán phân tích cảm xúc trên tập dữ liệu IMDb.
- Xử lý dữ liệu văn bản: Tokenization, Embedding.
- Sử dụng RNN/LSTM/GRU cho phân tích cảm xúc.
- Các kỹ thuật tối ưu hóa (Masking, Padding).

### Phần 3: Mạng Mã hóa - Giải mã cho Dịch máy (Encoder-Decoder cho NMT)
- Kiến trúc Encoder - Decoder cơ bản.
- Cách hoạt động của dịch máy thần kinh (Neural Machine Translation).
- Hạn chế của cấu trúc truyền thống khi gặp chuỗi dài.

### Phần 4: Cơ chế Chú ý (Attention Mechanisms)
- Vấn đề cổ chai (bottleneck) của Encoder-Decoder.
- Cơ chế Attention (Bahdanau / Luong Attention).
- Kiến trúc Transformer gốc: "Attention Is All You Need".
- Multi-Head Attention và Positional Encoding.

### Phần 5: Sự bùng nổ của Transformer & Hugging Face
- Kỷ nguyên của các mô hình Transformer (BERT, GPT, T5, ...).
- Vision Transformers (ViT) - Ứng dụng Attention vào hình ảnh.
- Sử dụng thư viện Transformers của Hugging Face để tải và sử dụng các mô hình pre-trained.

---

## 2. Tiêu chuẩn Thiết kế (Kế thừa từ Chương 11 - 15)

Để đảm bảo chất lượng hiển thị xuất sắc, slide Chương 16 sẽ tuân thủ nghiêm ngặt các quy tắc:

1. **Sử dụng gói ontspec:** Biên dịch bằng xelatex để hiển thị tiếng Việt mượt mà, không bị lỗi rớt font.
2. **Quản lý Hình ảnh (12 hình):**
   - Mọi lệnh \includegraphics phải có giới hạn width=0.85\textwidth,height=0.65\textheight,keepaspectratio.
   - Tiền tố chú thích rõ ràng: \vspace{0.2cm}\textit{Hình 16-X: Tên hình} (không dùng \caption để tránh lặp từ "Hình").
3. **Bố cục Chia cột (\begin{columns}):** Đối với các sơ đồ kiến trúc lớn (như kiến trúc Transformer), sử dụng chia cột: 1 cột cho hình ảnh, 1 cột cho text giải thích.
4. **Mã nguồn (Code Snippets):** Dùng \begin{lstlisting}[language=Python] kèm tuỳ chọn [fragile] trên \begin{frame}. Đảm bảo code gọn gàng, không bị tràn viền.

---

## 3. Kế hoạch Tích hợp Hình ảnh (Hình_16-1 đến Hình_16-12)

- **Thư mục lưu ảnh:** machineLearningWeb/Figures/CH16
- **Số lượng:** 12 bức hình (gồm .png và .jpg).
- **Nhiệm vụ:**
  - Lồng ghép đủ 12 hình vào bài giảng.
  - Mỗi hình sẽ có ít nhất 1 slide riêng biệt hoặc kết hợp với layout chia cột để giải thích ý nghĩa kiến trúc (đặc biệt là sơ đồ Attention và Transformer).

---

## 4. Các bước Triển khai thực tế

1. **Bước 1 (Đổi tên & Chuẩn hóa ảnh):** Đảm bảo tất cả 12 ảnh trong Figures/CH16 có tên đúng định dạng Hình_16-X.png/.jpg.
2. **Bước 2 (Viết kịch bản Python):** Tạo script gen_chap16.py để tự động chèn nội dung LaTeX (text, code, hình ảnh) dựa theo cấu trúc đã lên kế hoạch.
3. **Bước 3 (Biên dịch LaTeX):** Chạy xelatex Slide_ML_Chap16.tex 2 lần để cập nhật mục lục và tham chiếu.
4. **Bước 4 (Kiểm tra & Tinh chỉnh):** Duyệt qua file PDF (dự kiến ~50 trang) để đảm bảo không tràn chữ, không tràn code, và hình ảnh sắc nét.

