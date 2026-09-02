# 📘 Quy trình chuẩn hóa & Xây dựng Tab Thuật ngữ (Glossary Blueprint)

*Tài liệu này đóng vai trò là "Bản thiết kế mẫu" (Blueprint/SOP) được đúc kết từ quá trình giải quyết thành công các lỗi ở Chương 3. Khi bạn có bất kỳ file `glossaryChapter_XX.md` nào trong tương lai, AI sẽ đọc tài liệu này để tự động thiết lập một bản kế hoạch hoàn hảo và xây dựng Tab Thuật ngữ & Khái niệm sắc nét, không lỗi lầm cho chương đó.*

## 1. Mục tiêu (Objective)
- Tích hợp toàn bộ nội dung từ file thô `glossaryChapter_XX.md` (bao gồm chữ, hình ảnh, mã nguồn, công thức) vào đầu file bài học tương ứng (VD: `docs/chuong_XX.md`).
- Đảm bảo giao diện gọn gàng, không phá vỡ cấu trúc tab gốc, không gây lỗi cú pháp Markdown và mang lại trải nghiệm UX cao cấp.

## 2. Quy tắc kỹ thuật bắt buộc (Strict Technical Rules)

### 2.1. Cấu trúc Tab & Accordion
- **Cấm dùng Heading Markdown (`#`, `##`)** bên trong nội dung Tab vì plugin `docsify-tabs` của hệ thống sẽ quét chúng và cắt vụn trang web thành hàng chục tab nhỏ lỗi.
- **Sử dụng Accordion HTML:** Bắt buộc chuyển đổi các dấu `# PHẦN X` thành các khối đóng/mở HTML (`<details>` và `<summary>`).
- **Cấu trúc giao diện chuẩn cho Summary:**
  ```html
  <details>
  <summary><b style="font-size:1.2em">PHẦN X: TÊN PHẦN</b></summary>
  <br>
  ... Nội dung chi tiết ...
  </details>
  ```
- **Vị trí chèn (Injection Point):** Tab Thuật ngữ phải luôn nằm ở vị trí **ĐẦU TIÊN** bên trong khối `<!-- tabs:start -->`. Script tự động phải có chức năng: Tìm và **xóa sạch Tab Thuật ngữ cũ** (nếu có) trước khi chèn cái mới để tránh bị trùng lặp, đồng thời phải **bảo toàn tuyệt đối** các Tab Bài học khác phía dưới.

### 2.2. Quy tắc xử lý Hình ảnh (Image Processing)
- **Nguy cơ lỗi cú pháp:** Hình ảnh trong bản nháp thường bị nhúng dưới dạng công thức toán block (VD: `\\[\text{Hình X-Y: Tiêu đề}\\]`). Phải chạy Regex xử lý hình ảnh **trước** Regex xử lý Toán học.
- **Nguy cơ phá vỡ Markdown:** Để tránh việc parser Markdown hiểu nhầm caption ảnh thành Code Block (do thụt lề) hoặc làm gãy cấu trúc danh sách (List bullets), tuyệt đối **KHÔNG dùng thẻ HTML cấp khối (block-level)** như `<p>` hay `<div>`. Thay vào đó, sử dụng thẻ nội tuyến `<span>` kết hợp CSS.
- **Cấu trúc HTML an toàn 100%:**
  ```markdown
  ![Hình X-Y: Tiêu đề](../Figures/CHXX/Hinh_X-Y.png)
  <span style="display: block; color: #333; font-style: italic; text-align: center; margin-top: 5px; margin-bottom: 15px;"><b>Hình X-Y: Tiêu đề</b></span>
  ```
  *(Kỹ thuật quan trọng: Regex phải bảo lưu toàn bộ khoảng trắng thụt lề đầu dòng `([ \t]*)` của ảnh và áp đặt chính xác y hệt cho dòng thẻ `<span>` bên dưới).*
- **CSS Hỗ trợ (Global):** Hình ảnh đã được cấu hình trong `index.html` với `max-height: 350px`, bo góc, bóng đổ và tương thích hoàn toàn với plugin Zoom của Docsify.

### 2.3. Quy tắc xử lý Toán học (Math & KaTeX)
- Ký hiệu toán học `\\[ ... \\]` (dạng khối) và `\\( ... \\)` (dạng nội tuyến) trong file gốc phải được regex biến đổi chuẩn xác thành `$$ ... $$` và `$ ... $`.
- Luôn sử dụng cờ `re.DOTALL` trong Regex Python để có thể nắm bắt được các công thức toán học dài, bị ngắt xuống nhiều dòng.

### 2.4. Trải nghiệm người dùng (UX) độc quyền
Mã JS và CSS đã được cấy vĩnh viễn vào `index.html` (áp dụng chung cho toàn bộ web). Kể từ các chương sau, các tính năng này sẽ tự động chạy:
1. **Exclusive Accordion:** Nhấn mở phần này, các phần khác tự động thu lại.
2. **Auto-Scroll to Top:** Cuộn tự động mượt mà (smooth scroll) đưa tiêu đề phần đang xem lên sát mép trên màn hình.

## 3. Chu trình tự động hóa tương lai (Future Workflow)
Khi tác giả yêu cầu xử lý một chương mới (VD: Chương 4):
1. **Input:** Cung cấp `glossaryChapter_04.md`.
2. **AI Planning:** AI đọc bản thiết kế (Blueprint) này và phản hồi lại bằng bản kế hoạch áp dụng chuẩn.
3. **Scripting:** Tự động sinh file python (VD: `rebuild_glossary_ch4.py`) áp dụng đúng 100% các regex an toàn và logic chèn mục 2.
4. **Execution & QA:** Chạy script và hoàn tất. Giao diện chương mới sẽ đẹp và mượt mà ngay từ lần chạy đầu tiên.
