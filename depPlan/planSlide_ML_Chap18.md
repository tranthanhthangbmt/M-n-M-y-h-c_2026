# Kế hoạch chi tiết tạo Slide bài giảng: Chương 18 - Học Tăng Cường (Reinforcement Learning)

**Thông tin chung:**
- **Thư mục mục tiêu:** `slideML/`
- **File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)
- **Thời lượng dự kiến:** Khoảng 40-50 Frames (Tối thiểu 40 frames - đáp ứng chuẩn đại học)
- **Số lượng hình ảnh minh họa:** 10 hình minh họa (từ Hình 18-1 đến Hình 18-10).
- **Theme & Color Theme:** Madrid theme, default colortheme
- **Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_18.md` / `chương 18.htm`).

---

## 1. Cấu trúc Slide dự kiến (Total: ~45 slides)

### 1.1 Slide mở đầu và Mục tiêu (Slides 1-3)
- **Slide 1:** Tiêu đề "Chương 18: Học Tăng Cường (Reinforcement Learning)".
- **Slide 2:** Mục tiêu bài giảng (Khái niệm Agent, Environment, Reward; Thuật toán Q-Learning, Deep Q-Learning; Policy Gradients).
- **Slide 3:** Mục lục tổng quan.

---

### 1.2 Học để tối ưu hóa phần thưởng & Tìm kiếm chính sách (Slides 4-12)
- **Slide 4:** Giới thiệu Học Tăng Cường: Học thông qua thử và sai, lấy cảm hứng từ tâm lý học hành vi.
- **Slide 5:** Các thành phần cốt lõi: Tác nhân (Agent), Môi trường (Environment), Hành động (Action), Trạng thái (State), Phần thưởng (Reward). *(Sử dụng Hình 18-1)*
- **Slide 6:** Ví dụ thực tế về RL (Chơi cờ, Robot đi bộ, Xe tự lái).
- **Slide 7:** Tìm kiếm chính sách (Policy Search): Thế nào là một chính sách tốt?
- **Slide 8:** Khám phá không gian chính sách (Thêm nhiễu ngẫu nhiên, Thuật toán di truyền). *(Sử dụng Hình 18-2)*
- **Slide 9:** Môi trường OpenAI Gym: Giới thiệu thư viện chuẩn để huấn luyện RL.
- **Slide 10:** Bài toán CartPole: Giữ thăng bằng cây gậy trên xe đẩy. *(Sử dụng Hình 18-3)*
- **Slide 11:** Các chính sách mạng nơ-ron: Sử dụng Deep Learning để đưa ra quyết định thay vì bảng tĩnh.
- **Slide 12:** Đánh giá hành động: Bài toán phân bổ tín nhiệm (Credit Assignment Problem) - Làm sao biết hành động ở quá khứ gây ra hậu quả ở hiện tại?

---

### 1.3 Markov Decision Processes (MDP) và Học Q (Q-Learning) (Slides 13-25)
- **Slide 13:** Các quá trình quyết định Markov (MDP): Khung toán học của Học tăng cường. *(Sử dụng Hình 18-4)*
- **Slide 14:** Tính chất không nhớ (Memoryless) của MDP và phương trình Bellman.
- **Slide 15:** Hệ số chiết khấu (Discount Factor - $\gamma$): Cân nhắc giữa phần thưởng trước mắt và phần thưởng tương lai.
- **Slide 16:** Đánh giá trạng thái (State-Value Function $V(s)$) và Giá trị hành động (Action-Value Function $Q(s, a)$).
- **Slide 17:** Thuật toán Q-Learning cơ bản: Học thông qua bảng (Q-Table).
- **Slide 18:** Cập nhật giá trị Q dựa trên Chênh lệch thời gian (Temporal Difference - TD Learning).
- **Slide 19:** Khai thác và Khám phá (Exploration vs. Exploitation): Sự đánh đổi kinh điển.
- **Slide 20:** Chiến lược $\epsilon$-greedy: Lựa chọn ngẫu nhiên để không bị kẹt ở cực tiểu cục bộ. *(Sử dụng Hình 18-5)*
- **Slide 21:** Hạn chế của Q-Learning cơ bản: Không gian trạng thái quá lớn (Ví dụ: Số lượng điểm ảnh trong trò chơi Atari).
- **Slide 22:** Giới thiệu Học Q Xấp xỉ (Approximate Q-Learning): Dùng hàm thay vì dùng bảng.
- **Slide 23:** Cấu trúc Học Q sâu (Deep Q-Learning): Dùng Mạng Nơ-ron (DQN) để xấp xỉ hàm Q. *(Sử dụng Hình 18-6)*
- **Slide 24:** Vấn đề нестабиль (Instability) khi huấn luyện DQN.
- **Slide 25:** Trải nghiệm Phát lại (Replay Buffer): Phá vỡ tính tương quan của chuỗi dữ liệu.

---

### 1.4 Triển khai Học Q Sâu và Các biến thể (Slides 26-34)
- **Slide 26:** Kiến trúc Mạng Mục tiêu (Target Network): Giữ cố định mục tiêu để mạng chính dễ dàng hội tụ. *(Sử dụng Hình 18-7)*
- **Slide 27:** Triển khai mã nguồn DQN bằng Keras (Tổng quan các bước).
- **Slide 28:** Huấn luyện DQN trên CartPole và Atari.
- **Slide 29:** Các biến thể của Học Q Sâu: Double DQN - Khắc phục lỗi đánh giá quá cao (Overestimation).
- **Slide 30:** Dueling DQN: Tách biệt giá trị trạng thái và lợi thế hành động. *(Sử dụng Hình 18-8)*
- **Slide 31:** Ưu tiên Trải nghiệm (Prioritized Experience Replay - PER): Học từ những sai lầm đáng nhớ nhất.

---

### 1.5 Độ dốc chính sách (Policy Gradients) (Slides 35-42)
- **Slide 35:** Nhược điểm của DQN: Khó xử lý không gian hành động liên tục (Continuous Actions).
- **Slide 36:** Giới thiệu Độ dốc chính sách (Policy Gradients - PG): Tối ưu hóa trực tiếp mạng chính sách thay vì học giá trị Q.
- **Slide 37:** Thuật toán REINFORCE: Cập nhật trọng số mạng dựa trên tổng phần thưởng đạt được ở cuối mỗi ván (Episode). *(Sử dụng Hình 18-9)*
- **Slide 38:** Vấn đề Phương sai cao của thuật toán PG cơ bản.
- **Slide 39:** Sử dụng Đường cơ sở (Baseline) để giảm phương sai.
- **Slide 40:** Actor-Critic: Sự kết hợp hoàn hảo giữa Policy Gradients (Diễn viên) và Q-Learning (Nhà phê bình). *(Sử dụng Hình 18-10)*
- **Slide 41:** Thuật toán Asynchronous Advantage Actor-Critic (A3C).
- **Slide 42:** Ứng dụng thực tế: AlphaGo, AlphaStar, huấn luyện Robot mô phỏng.

---

### 1.6 Tổng kết và Bài tập (Slides 43-45)
- **Slide 43:** Bảng tổng hợp: Khi nào dùng Q-Learning, khi nào dùng Policy Gradients?
- **Slide 44:** Hướng dẫn Bài tập thực hành với thư viện OpenAI Gym.
- **Slide 45:** Hỏi & Đáp (Q&A).

---

## 2. Kỹ thuật triển khai

- **Script sử dụng:** `gen_chap18.py`
- **File TeX kết quả:** `Slide_ML_Chap18.tex`
- **Tránh lỗi Unicode và Dấu câu TeX:** Toàn bộ kịch bản sinh mã TeX sẽ được bọc bởi `r''' ... '''` trong script và dùng `xelatex` để hỗ trợ font Arial tiếng Việt.
- **Xử lý Code:** Sử dụng `[fragile]` để tích hợp các đoạn mã huấn luyện RL minh họa.

---

## 3. Kế hoạch Tích hợp Hình ảnh (Hình_18-1 đến Hình_18-10)

- **Thư mục lưu ảnh:** `machineLearningWeb/Figures/CH18`
- **Số lượng:** 10 bức hình (tất cả định dạng .png). Đã được đổi tên tự động hóa và chuẩn hóa thứ tự từ Hình 1 đến 10.
- **Nhiệm vụ:**
  - Chèn 10 hình vào các khung giải thích thuật toán, đặc biệt là sơ đồ MDP và mô hình Actor-Critic.
  - Sử dụng layout chia cột `\begin{columns}` để giúp người học vừa đọc mô tả vừa xem sơ đồ.

---

## 4. Các bước Triển khai thực tế

1. **Bước 1 (Đổi tên & Chuẩn hóa ảnh):** Đảm bảo tất cả 10 ảnh trong `Figures/CH18` có tên đúng định dạng `Hình_18-X.png/.jpg`. (Đã hoàn tất đổi tên và chuẩn hóa thứ tự từ Hình 1 đến 10).
2. **Bước 2 (Viết kịch bản Python):** Tạo script `gen_chap18.py` để tự động chèn nội dung LaTeX (text, code, hình ảnh) dựa theo cấu trúc đã lên kế hoạch. (Lưu ý xử lý linh hoạt extension .png và .jpg).
3. **Bước 3 (Biên dịch LaTeX):** Chạy `xelatex Slide_ML_Chap18.tex` 2 lần để cập nhật mục lục và tham chiếu.
4. **Bước 4 (Kiểm tra & Tinh chỉnh):** Duyệt qua file PDF (dự kiến ~45 trang) để đảm bảo không tràn chữ, không tràn code, và hình ảnh sắc nét.
