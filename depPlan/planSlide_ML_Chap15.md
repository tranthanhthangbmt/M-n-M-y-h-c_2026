# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 15 - Xử lý Chuỗi bằng RNN và CNN

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Thời lượng dự kiến:** 4 Tiết học (Tối thiểu 40 frames - chi tiết hóa nội dung)
**Số lượng hình ảnh minh họa:** 14 hình cốt lõi (Từ Hình 15-1 đến Hình 15-14)
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_15.md` / `CHƯƠNG 15.htm`).

---

## 1. Cấu trúc Nội dung (Dàn ý chi tiết)

Bộ slide sẽ được chia thành các phần chính bám sát theo đúng tài liệu gốc:

### Phần 1: Mạng nơ-ron hồi quy cơ bản (RNN)
- Khái niệm về dữ liệu chuỗi (Sequence data).
- Mạng RNN cơ bản và Lớp ô (Cell layer).
- Cách RNN duy trì trạng thái (State) và xử lý chuỗi thời gian.

### Phần 2: Huấn luyện RNN
- Backpropagation Through Time (BPTT - Lan truyền ngược qua thời gian).
- Khó khăn trong huấn luyện RNN: Vanishing / Exploding Gradients.

### Phần 3: Dự báo chuỗi thời gian (Time Series Forecasting)
- Ứng dụng RNN vào bài toán dự báo.
- Baseline metrics.
- Dự báo nhiều bước (Multi-step forecasting).

### Phần 4: RNN Sâu (Deep RNNs)
- Xếp chồng nhiều lớp RNN.
- Kiến trúc mạng RNN sâu để học các đặc trưng phức tạp hơn.

### Phần 5: Các tế bào LSTM và GRU
- Vấn đề trí nhớ ngắn hạn của RNN cơ bản.
- Tế bào LSTM (Long Short-Term Memory): Cấu trúc, các cổng quên, cổng cập nhật, cổng xuất.
- Tế bào GRU (Gated Recurrent Unit): Phiên bản thu gọn và hiệu quả của LSTM.

### Phần 6: WaveNet (CNN 1D)
- Áp dụng CNN 1D cho dữ liệu chuỗi.
- Tích chập giãn nở (Dilated Convolutions) trong WaveNet giúp mở rộng trường thụ cảm.

---

## 2. Cập nhật các Tiêu chuẩn Mới (Kế thừa từ kinh nghiệm Chương 13, 14)

Để đảm bảo chất lượng hiển thị xuất sắc, slide Chương 15 sẽ tuân thủ nghiêm ngặt các quy tắc mới sau đây:

1. **Sử dụng gói `fontspec` thay vì `fontenc`:** Để sửa triệt để lỗi font tiếng Việt khi dùng trình biên dịch `xelatex`.
2. **Kích thước hình ảnh tối ưu:** Mọi lệnh `\includegraphics` phải có giới hạn `width=0.85\textwidth,height=0.6\textheight,keepaspectratio` để không bị tràn khung hình.
3. **Tiền tố chú thích rõ ràng:** Các chú thích hình ảnh phải được thêm tiền tố "Hình 15-x: " một cách chuẩn xác (Ví dụ: `\textit{Hình 15-1: Sơ đồ mạng RNN}`).
4. **Chia cột cho hình dọc/lớn:** Với các slide có hình ảnh quá cao (như sơ đồ LSTM, cấu trúc WaveNet), sẽ sử dụng môi trường `\begin{columns}` để chia slide: 
   - Cột trái (`0.65\textwidth`) dành cho hình ảnh, đẩy kích thước ảnh lên lớn nhất có thể.
   - Cột phải (`0.35\textwidth`) dành cho tiêu đề phụ hoặc chú thích.
5. **Tách Slide nếu cần:** Tách văn bản dài và hình ảnh to ra thành 2 slide riêng biệt nếu việc dùng cột vẫn không đủ không gian.

---

## 3. Kế hoạch Cắt & Trích xuất Hình ảnh (Hình_15-...)

- **Công cụ:** Đã sử dụng script Python tự động đổi tên thành công.
- **Thư mục lưu ảnh:** `machineLearningWeb/Figures/CH15`
- **Kết quả:** Có chính xác 14 bức hình chính (từ `Hình_15-1.png` đến `Hình_15-14.png`).
- **Nhiệm vụ Slide:** Tích hợp cả 14 hình này vào bài giảng, sử dụng cột hoặc tách trang để đảm bảo hình to rõ, có mô tả như giảng viên giảng dạy trên lớp. Đảm bảo tổng số slide > 40.

---

## 4. Kế hoạch Code LaTeX & Beamer Cụ thể

- **Tệp nguồn:** `Slide_ML_Chap15.tex`
- **Theme:** Madrid, aspect ratio 16:9.
- **Mã nguồn (Code Snippets):**
  - Sử dụng môi trường `\begin{lstlisting}[language=Python]` kèm tuỳ chọn `[fragile]` cho khung chứa code (VD: Code `SimpleRNN`, `LSTM`, `GRU`, `Conv1D`).
  - Hỗ trợ syntax highlight.

---

## 5. Các bước Triển khai thực tế

1. **Bước 1 (Trích xuất ảnh):** Viết và chạy script python để rút trích toàn bộ ảnh của Chương 15, lưu vào thư mục `Figures/CH15`.
2. **Bước 2 (Viết LaTeX Source):** Dựng khung `Slide_ML_Chap15.tex`, chèn nội dung, code và áp dụng triệt để 5 tiêu chuẩn mới (font, resize ảnh, chia cột,...).
3. **Bước 3 (Biên dịch & Khắc phục lỗi):** Chạy `xelatex Slide_ML_Chap15.tex` hai lần, kiểm tra PDF xem có trang nào bị lỗi hiển thị hay không, tinh chỉnh.

---

## 6. Tiêu chuẩn Đánh giá (Nghiệm thu)

- Khắc phục hoàn toàn lỗi rớt font tiếng Việt.
- Bố cục hình ảnh tuyệt đẹp, chữ rõ ràng bên phải và hình to bên trái ở các sơ đồ kiến trúc phức tạp.
- Chú thích hình ảnh đầy đủ định dạng `Hình 15-x: Tên hình`.
- Code Python syntax màu chuẩn xác, không bị lỗi tràn dòng do `xelatex`.
