# Kế hoạch thiết kế Tab Thuật ngữ cho Chương 1

Mục tiêu của kế hoạch này là áp dụng nghiêm ngặt Quy trình chuẩn hóa (Blueprint) để thiết kế Tab "Thuật ngữ & Khái niệm cốt lõi" cho Chương 1. Toàn bộ nội dung sẽ được lấy từ `glossaryChapter_01.md` và chèn tự động vào `docs/chuong_01.md`, đồng thời tuân thủ yêu cầu: **Tuyệt đối giữ lại trọn vẹn các tab trước đó ở bên cạnh (Lý thuyết, Tiếng Việt, Video...).**

## User Review Required
Bạn vui lòng xem kỹ các bước xử lý kỹ thuật dưới đây, đặc biệt là sự cẩn trọng với các Regex đã được nâng cấp sau bài học ở Chương 2. Nếu bạn đồng ý, hãy nhấn **Proceed** để tôi tiến hành viết script và thực thi ngay lập tức.

## Proposed Changes

Tôi sẽ tạo một kịch bản tự động hóa mang tên `rebuild_glossary_ch1.py`. Kịch bản này kế thừa toàn bộ thành tựu sửa lỗi từ Chương 2:

### 1. Dọn dẹp văn bản (Clean-up)
- Quét và xóa toàn bộ các dấu trích dẫn dạng `[cite: XXX]` do AI sinh ra bằng Regex an toàn: `r'(?:\\\\)*[ \t]*\[cite: \d+\]'`. (Điều này đảm bảo không để lại ký tự dư thừa làm crash bộ dịch KaTeX như đã từng gặp ở chương trước).

### 2. Xử lý Cấu trúc Tab & Accordion
- Chuyển đổi toàn bộ các dòng tiêu đề như `# PHẦN 1: ...` thành khối HTML Accordion chuẩn:
  ```html
  <details>
  <summary><b style="font-size:1.2em">PHẦN 1: ...</b></summary>
  <br>
  ... (Nội dung) ...
  </details>
  ```

### 3. Xử lý Hình ảnh bằng thẻ `<span>` (Bảo toàn Markdown)
- Quét tìm và xử lý định dạng hình ảnh dành riêng cho Chương 1 (ví dụ `Hình 1-X`).
- Trích xuất thành thẻ hình ảnh trỏ tới `../Figures/CH01/Hinh_1-X.png`.
- Chèn caption bằng thẻ nội tuyến **`<span>`** với đầy đủ thụt lề để danh sách Markdown không bị sập:
  ```html
  <span style="display: block; color: #333; font-style: italic; text-align: center; margin-top: 5px; margin-bottom: 15px;"><b>Hình 1-X: Tiêu đề</b></span>
  ```

### 4. Xử lý Toán học (Tích hợp KaTeX)
- Chuyển đổi toàn bộ khối công thức `\\[ ... \\]` thành `$$ ... $$`.
- Chuyển đổi công thức nội tuyến `\\( ... \\)` thành `$ ... $`.
- Xử lý mượt mà ngay cả khi công thức bị ngắt thành nhiều dòng nhờ cờ `re.DOTALL`.

### 5. Tiêm (Inject) an toàn vào `docs/chuong_01.md`
- Quét tìm định danh `<!-- tabs:start -->` trong file `chuong_01.md`.
- Tìm và gỡ bỏ Tab Thuật ngữ cũ (nếu có tồn tại ở các phiên bản trước).
- Chèn Tab mới vào với tư cách là tab ưu tiên số 1: `#### ** 📚 Thuật ngữ & Khái niệm **`.
- **Quan trọng:** Thuật toán tiêm sẽ tính toán vị trí nối chuỗi chính xác sao cho nó trượt ngay phía trên tab `#### ** 📖 Lý thuyết **`, **bảo tồn nguyên vẹn 100% các tab sẵn có còn lại.**

## Verification Plan

### Automated Tests
- Chạy script `rebuild_glossary_ch1.py` và theo dõi log báo cáo không có lỗi.

### Manual Verification
- Bạn và tôi sẽ tải lại trang và kiểm tra thủ công nội dung Chương 1.
- **Tiêu chí đạt chuẩn:**
  1. Tab "Thuật ngữ" xuất hiện ở vị trí đầu.
  2. Toàn bộ các Tab khác (Lý thuyết, Slide, Video...) nằm yên vị trí, không bị ăn mất như sự cố cũ.
  3. Giao diện Drag-to-scroll hoạt động trơn tru.
  4. Các hình ảnh (Ví dụ Hình 1-1, 1-2) hiển thị với chú thích đẹp mắt, không làm đứt đoạn danh sách văn bản.
