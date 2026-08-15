# Lập kế hoạch: Thêm Tab Bài tập Trắc nghiệm (Interactive Quiz) cho Chương 19

Dựa trên yêu cầu, hệ thống bài tập trắc nghiệm Chương 19 ("CHƯƠNG 19. HUẤN LUYỆN VÀ TRIỂN KHAI MÔ") sẽ được xây dựng gồm tối thiểu 30 câu hỏi. Toàn bộ nội dung kiến thức để soạn câu hỏi sẽ được trích xuất **chính xác và trực tiếp** từ tài liệu Tiếng Việt của Chương 19 như sau:

## Nguồn tài liệu tham khảo chính
1. **Nội dung lý thuyết Tiếng Việt:** `docs/chuong_19.md` (Tab "Lý thuyết")

## Phạm vi kiến thức bao phủ (từ nguồn trên)
1. **Phục vụ một mô hình TensorFlow:** Các khái niệm, định nghĩa và đặc điểm liên quan.
2. **Sử dụng GPU để Tăng tốc Tính toán:** Các khái niệm, định nghĩa và đặc điểm liên quan.
3. **Huấn luyện mô hình trên nhiều thiết bị:** Các khái niệm, định nghĩa và đặc điểm liên quan.
4. **Huấn luyện Quy mô lớn bằng API Chiến lược:** Các khái niệm, định nghĩa và đặc điểm liên quan.
5. **Chạy các Tác vụ Huấn luyện Lớn trên:** Các khái niệm, định nghĩa và đặc điểm liên quan.
6. **Điều chỉnh Siêu Tham số trên Vertex AI:** Các khái niệm, định nghĩa và đặc điểm liên quan.
7. **Bài tập:** Các khái niệm, định nghĩa và đặc điểm liên quan.

## Proposed Changes

Tôi sẽ tạo một trang HTML chứa tối thiểu 30 câu hỏi trắc nghiệm và nhúng nó vào file `docs/chuong_19.md`.

### Khởi tạo thư mục và file Quiz
#### [NEW] `quizzes/Chapter19/index.html`
- **Thiết kế giao diện:** Tái sử dụng form giao diện, màu sắc, và cấu trúc điều khiển (HTML/CSS/JS) chuẩn như đã áp dụng cho các học phần khác (ví dụ: AI Kế toán) để đảm bảo tính nhất quán và chuyên nghiệp.
- **Biên soạn câu hỏi:** Dựa vào nội dung `docs/chuong_19.md`, sinh tối thiểu 30 câu hỏi bám sát các mục lý thuyết kể trên. Đảm bảo đa dạng các loại câu hỏi như môn AI cho Kế toán (gồm: Trắc nghiệm đa lựa chọn - MCQ, Ghép nối - Matching, Sắp xếp thứ tự - Sorting, Điền từ vào chỗ trống/Kéo thả - Drag & Drop). Đồng thời, mỗi câu hỏi phải được phân loại và ghi rõ mức độ khó (Dễ, Trung bình, Khó).

### Cập nhật File Markdown
#### [MODIFY] `docs/chuong_19.md`
Thêm tab mới vào cuối file, ngay trước `<!-- tabs:end -->`:

```markdown
#### ** 📝 Bài tập Trắc nghiệm **

<iframe src="quizzes/Chapter19/index.html" style="width: 100%; min-height: 700px; border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></iframe>
```

## Verification Plan
- Viết nội dung mã tạo câu hỏi và giao diện `quizzes/Chapter19/index.html`.
- Mở và cập nhật file `docs/chuong_19.md`.
- Tải lại trang web chính trên trình duyệt tại đường dẫn `/#/docs/chuong_19` và chuyển sang tab **Bài tập Trắc nghiệm**.
- Kiểm tra tính năng tương tác (chọn đáp án, chuyển câu, nộp bài, xem kết quả đúng/sai).
