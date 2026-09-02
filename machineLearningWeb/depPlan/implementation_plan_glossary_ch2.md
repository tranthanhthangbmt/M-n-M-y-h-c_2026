# Kế hoạch thiết kế Tab Thuật ngữ cho Chương 2

Mục tiêu của kế hoạch này là áp dụng nghiêm ngặt Quy trình chuẩn hóa (Blueprint) vừa thiết lập để thiết kế Tab "Thuật ngữ & Khái niệm cốt lõi" cho Chương 2. Toàn bộ nội dung sẽ được lấy từ `glossaryChapter_02.md` và chèn tự động vào `docs/chuong_02.md`.

## User Review Required
Bạn vui lòng xem kỹ các bước xử lý kỹ thuật dưới đây để đảm bảo mọi quy tắc của Blueprint (Toán học, Hình ảnh, Accordion) đều đã được liệt kê chính xác. Nếu đồng ý, hãy nhấn **Proceed** để tôi tiến hành viết script và thực thi ngay lập tức.

## Proposed Changes

Tôi sẽ tạo một kịch bản tự động hóa mang tên `rebuild_glossary_ch2.py`.

### 1. Xử lý Cấu trúc Tab & Accordion (Tránh vỡ giao diện)
- Chuyển đổi toàn bộ các dòng tiêu đề như `# PHẦN 1: ...` thành khối HTML Accordion:
  ```html
  <details>
  <summary><b style="font-size:1.2em">PHẦN 1: ...</b></summary>
  <br>
  ... (Nội dung bên trong) ...
  </details>
  ```
- **Lưu ý:** Đóng thẻ `</details>` gọn gàng ở cuối mỗi phần. Tuyệt đối không dùng dấu `#` bên trong nội dung để plugin của Docsify không bị lỗi quét nhầm tab.

### 2. Xử lý Hình ảnh bằng thẻ `<span>` (Bảo toàn Markdown)
- **Bước này phải chạy trước khi xử lý Toán học.**
- Tìm kiếm các cụm như `\\[\text{Hình 2-X: Tiêu đề}\\]` (có thể bị ngắt dòng hoặc chứa ký tự đặc biệt).
- Biến đổi thành thẻ hình ảnh Markdown chuẩn hướng tới thư mục `../Figures/CH02/Hinh_2-X.png`.
- Quan trọng nhất: Chèn tiêu đề hình ảnh bên dưới bằng thẻ nội tuyến **`<span>`** thay vì `<p>`:
  ```html
  <span style="display: block; color: #333; font-style: italic; text-align: center; margin-top: 5px; margin-bottom: 15px;"><b>Hình 2-X: Tiêu đề</b></span>
  ```
- Kịch bản sẽ tự động trích xuất và **bảo toàn nguyên vẹn số khoảng trắng thụt lề** ở đầu dòng để danh sách gạch đầu dòng (bullet points) không bị sập cấu trúc.

### 3. Xử lý Toán học (Tích hợp KaTeX)
- Chuyển đổi toàn bộ các khối công thức `\\[ ... \\]` thành `$$ ... $$`.
- Chuyển đổi các công thức nội tuyến `\\( ... \\)` thành `$ ... $`.
- Bật cờ quét nhiều dòng (`re.DOTALL`) để đảm bảo không bỏ sót bất kỳ phương trình nhiều dòng nào (ví dụ công thức RMSE, Norm).

### 4. Tiêm (Inject) thông minh vào `docs/chuong_02.md`
- Quét tìm định danh `<!-- tabs:start -->` trong file `chuong_02.md`.
- Tìm kiếm và **xóa sạch toàn bộ Tab "Thuật ngữ" cũ** (nếu có) nằm trước Tab "Lý thuyết", đảm bảo không làm tổn hại cấu trúc của các tab bài học còn lại.
- Chèn khối văn bản đã được biên dịch xong vào làm tab ưu tiên đầu tiên: `#### ** 📚 Thuật ngữ & Khái niệm **`.

## Verification Plan

### Automated Tests
- Chạy script `rebuild_glossary_ch2.py` và theo dõi log báo cáo số lượng phương trình, hình ảnh và phần (sections) đã được thay thế thành công.

### Manual Verification
- Bạn và tôi sẽ tải lại trang và kiểm tra thủ công nội dung Chương 2.
- **Tiêu chí đạt chuẩn:**
  1. Hình ảnh không bị vỡ Markdown, tiêu đề căn giữa ngay ngắn.
  2. Toán học hiển thị đúng công thức RMSE, MAE đồ họa sắc nét.
  3. Kéo thanh tab mượt mà bằng JavaScript (Drag-to-scroll vừa làm) và click hoạt động tốt.
  4. Mở một Phần Accordion thì các Phần khác tự đóng lại, cuộn mượt (Exclusive Accordion).
