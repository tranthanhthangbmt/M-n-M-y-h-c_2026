Để giúp bạn hệ thống hóa toàn bộ kiến thức cốt lõi của **Chương 4: Huấn luyện mô hình (Training Models)** một cách chi tiết và dễ hiểu nhất, tôi đề xuất chia nội dung chương này thành **5 phần chuyên sâu** đi từ cơ bản đến nâng cao:

* **Phần 1: Hồi quy tuyến tính & Các phương pháp tối ưu hóa giải tích (Normal Equation, SVD)** *(Chúng ta sẽ hoàn thành phần đầu tiên này trước)*.
* **Phần 2: Thuật toán hạ Gradient (Gradient Descent) & Các biến thể (Batch GD, Stochastic GD, Mini-batch GD)**.
* **Phần 3: Hồi quy đa thức & Đánh giá mô hình qua Đường cong học tập (Learning Curves)**.
* **Phần 4: Các mô hình tuyến tính được chính quy hóa (Ridge, Lasso, Elastic Net, Early Stopping)**.
* **Phần 5: Hồi quy phân loại (Hồi quy Logistic & Hồi quy Softmax)**.

Dưới đây là chi tiết **Phần 1** được thiết kế theo đúng cấu trúc chuẩn của tệp mục lục thuật ngữ học thuật, bao gồm định nghĩa bản chất, ví dụ thực tế trong tài liệu, giải thích trực quan dựa trên sơ đồ hình ảnh và mã nguồn Python đi kèm.

---

### PHẦN 1: HỒI QUY TUYẾN TÍNH & CÁC PHƯƠNG PHÁP TỐI ƯU HÓA GIẢI TÍCH

##### 1. Hồi quy tuyến tính (Linear Regression)
*   **Giải thích bản chất:** 
    *   **Hồi quy tuyến tính** là một trong những mô hình cơ bản và đơn giản nhất trong học máy. Mô hình này đưa ra dự đoán bằng cách tính **tổng có trọng số của các đặc trưng đầu vào**, cộng thêm một hằng số gọi là **hệ số chệch (bias term)** (hay hệ số chặn - intercept term).
    *   **Công thức dự đoán (Phương trình 4-1):**
        \\[\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n\\]
        *(Trong đó: \\(\hat{y}\\) là giá trị dự đoán, \\(n\\) là số lượng đặc trưng, \\(x_i\\) là giá trị đặc trưng thứ \\(i\\), \\(\theta_0\\) là hệ số chệch, và \\(\theta_j\\) là trọng số đặc trưng thứ \\(j\\)).*
    *   **Dạng vector hóa (Phương trình 4-2):**
        \\[\hat{y} = h_{\theta}(x) = \theta \cdot x\\]
        *(Trong đó: \\(\theta\\) là vector tham số chứa cả hệ số chệch \\(\theta_0\\) và các trọng số đặc trưng từ \\(\theta_1\\) đến \\(\theta_n\\); \\(x\\) là vector đặc trưng của trường hợp dữ liệu với đặc trưng giả \\(x_0\\) luôn bằng 1; và \\(\theta \cdot x\\) đại diện cho tích vô hướng của hai vector).*
*   **Ví dụ thực tế trong tài liệu:** Mô hình đo lường chỉ số sự hài lòng cuộc sống dựa trên thu nhập GDP đầu người: \\(\text{life\_satisfaction} = \theta_0 + \theta_1 \times \text{GDP\_per\_capita}\\). Tài liệu cũng tạo ra một tập dữ liệu giả lập tuyến tính ngẫu nhiên chứa 100 mẫu dữ liệu dựa trên phương thức \\(y = 4 + 3x_1 + \text{Gaussian noise}\\) để tiến hành huấn luyện mô hình.
*   **Giải thích trực quan dựa trên hình ảnh:**
    *   **Trực quan hóa tập dữ liệu giả lập (Hình 4-1):** Tập dữ liệu gồm 100 điểm dữ liệu màu xanh phân bố rải rác nhưng có xu hướng tuyến tính rõ rệt. Nhiễu Gaussian được thêm vào khiến các điểm này không nằm thẳng hàng hoàn hảo mà dao động xung quanh một đường trục tuyến tính tưởng tượng.
    *   **Trực quan hóa dự đoán của mô hình (Hình 4-2):** Sau khi huấn luyện mô hình, dự đoán được vẽ bằng một đường thẳng màu đỏ. Đường thẳng này đi qua trung tâm phân bổ của các điểm dữ liệu thực tế màu xanh, thể hiện khả năng nắm bắt xu hướng tuyến tính của dữ liệu một cách tối ưu nhất.
*   **Mã nguồn Python minh họa:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Tạo tập dữ liệu tuyến tính ngẫu nhiên (Hình 4-1)
np.random.seed(42)
m = 100  # số lượng mẫu dữ liệu
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# 2. Huấn luyện mô hình hồi quy tuyến tính bằng Scikit-Learn
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Xem các tham số mô hình ước lượng được
print("Hệ số chệch (theta_0):", lin_reg.intercept_)  # Gần giá trị thực tế là 4
print("Trọng số đặc trưng (theta_1):", lin_reg.coef_)  # Gần giá trị thực tế là 3

# 3. Dự đoán trên dữ liệu mới và trực quan hóa (Hình 4-2)
X_new = np.array([,])
y_predict = lin_reg.predict(X_new)

plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.", label="Dữ liệu thực tế")
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Dự đoán")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis()
plt.grid(True)
plt.legend(loc="upper left")
plt.title("Hồi quy tuyến tính trên dữ liệu ngẫu nhiên")
plt.show()
```

##### 2. Hàm chi phí lỗi trung bình bình phương (Mean Squared Error - MSE)
*   **Giải thích bản chất:** Để huấn luyện mô hình, chúng ta cần một thước đo đánh giá mức độ phù hợp (hoặc không phù hợp) của mô hình với dữ liệu. Mặc dù lỗi trung bình bình phương gốc (RMSE) là thước đo hiệu suất phổ biến, việc tối thiểu hóa **MSE** đơn giản hơn về mặt toán học và dẫn đến cùng một kết quả tối ưu (vì hàm số đạt cực tiểu tại điểm nào thì căn bậc hai của nó cũng đạt cực tiểu tại điểm đó).
    *   **Công thức tính toán (Phương trình 4-3):**
        \\[MSE(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \theta^T x^{(i)} - y^{(i)} \right)^2\\]
*   **Ví dụ thực tế trong tài liệu:** Hàm chi phí MSE đóng vai trò là "la bàn" giúp các thuật toán tối ưu hóa (như Phương trình chuẩn tắc hay hạ Gradient) điều chỉnh vector tham số \\(\theta\\) hướng tới giá trị tốt nhất.
*   **Mã nguồn Python minh họa:**
```python
from sklearn.metrics import mean_squared_error

# Dự đoán trên toàn bộ tập dữ liệu X
y_pred_train = lin_reg.predict(X)

# Tính toán giá trị MSE
mse_score = mean_squared_error(y, y_pred_train)
print(f"Giá trị lỗi MSE trên tập huấn luyện: {mse_score:.4f}")
```

##### 3. Phương trình chuẩn tắc (Normal Equation)
*   **Giải thích bản chất:** Để tìm giá trị \\(\theta\\) làm tối thiểu hóa hàm chi phí MSE, tồn tại một nghiệm dạng đóng (closed-form solution) — nghĩa là một phương trình toán học cho ra kết quả tối ưu trực tiếp mà không cần lặp.
    *   **Công thức toán học (Công thức 4-4):**
        \\[\theta = (X^T X)^{-1} X^T y\\]
        *(Trong đó: \\(\theta\\) là vector tham số tối ưu, \\(X\\) là ma trận đặc trưng chứa thêm một cột giả \\(x_0 = 1\\), và \\(y\\) là vector các giá trị mục tiêu).*
*   **Ví dụ thực tế trong tài liệu:** Tài liệu sử dụng hàm nghịch đảo ma trận `np.linalg.inv()` và toán tử nhân ma trận `@` của NumPy để tìm nghiệm tối ưu trực tiếp trên tập dữ liệu giả lập. Kết quả thu được là \\(\theta_0 \approx 4.215\\) và \\(\theta_1 \approx 2.770\\). Sự sai khác nhỏ so với hàm gốc gốc (\\(\theta_0 = 4, \theta_1 = 3\\)) là do tác động của nhiễu Gaussian trong tập dữ liệu nhỏ.
*   **Mã nguồn Python minh họa:**
```python
from sklearn.preprocessing import add_dummy_feature

# Thêm đặc trưng giả x0 = 1 vào mỗi trường hợp để tính hệ số chệch theta_0
X_b = add_dummy_feature(X)

# Tính toán tham số tối ưu theo công thức toán học trực tiếp
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print("Tham số tối ưu tính bằng Phương trình chuẩn tắc:")
print(theta_best)
# Kết quả:
# [[4.21509616]
#  [2.77011339]]
```

##### 4. Phân tách giá trị suy biến (SVD) & Nghịch đảo giả (Pseudoinverse)
*   **Giải thích bản chất:** 
    *   Lớp `LinearRegression` của Scikit-Learn dựa trên hàm `scipy.linalg.lstsq()` (least squares) sử dụng kỹ thuật phân tách ma trận tiêu chuẩn gọi là **Phân tách giá trị suy biến (Singular Value Decomposition - SVD)** để phân tích ma trận dữ liệu huấn luyện \\(X\\) thành tích của ba ma trận: \\(U \Sigma V^T\\).
    *   Từ đó, nghiệm tối ưu được tính trực tiếp thông qua **nghịch đảo giả Moore-Penrose (Pseudoinverse)** \\(X^+\\) bằng công thức:
        \\[\theta = X^+ y \quad \text{với} \quad X^+ = V \Sigma^+ U^T\\]
    *   **Ưu điểm vượt trội so với Phương trình chuẩn tắc:**
        1.  **Độ phức tạp tính toán tối ưu hơn:** Phương trình chuẩn tắc yêu cầu tính nghịch đảo của ma trận \\((X^T X)\\), có độ phức tạp khoảng \\(O(n^{2.4})\\) đến \\(O(n^3)\\) (với \\(n\\) là số lượng đặc trưng). Trong khi đó, cách tiếp cận SVD chỉ có độ phức tạp khoảng \\(O(n^2)\\). Do đó, khi số đặc trưng tăng gấp đôi, SVD chỉ tốn thời gian gấp 4 lần, còn Phương trình chuẩn tắc tốn gấp 5.3 đến 8 lần.
        2.  **Khả năng xử lý các trường hợp đặc biệt:** Phương trình chuẩn tắc sẽ bị lỗi và không thể hoạt động nếu ma trận \\(X^T X\\) không khả nghịch (chẳng hạn khi số lượng mẫu dữ liệu ít hơn số đặc trưng \\(m < n\\) hoặc có các đặc trưng bị trùng lặp/thừa). Trong khi đó, nghịch đảo giả \\(X^+\\) luôn luôn được xác định một cách ổn định trong mọi trường hợp dữ liệu.
*   **Ví dụ thực tế trong tài liệu:** Scikit-Learn tự động áp dụng giải pháp SVD này bên dưới lớp `LinearRegression`. Tài liệu cũng minh họa việc gọi trực tiếp hàm nghịch đảo giả `np.linalg.pinv()` để cho ra kết quả tương đương.
*   **Mã nguồn Python minh họa:**
```python
# Cách 1: Tính gián tiếp qua hàm tối thiểu bình phương lstsq của SciPy
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print("Tham số tối ưu tính bằng SVD (lstsq):\n", theta_best_svd)

# Cách 2: Tính trực tiếp bằng hàm nghịch đảo giả Moore-Penrose (pinv)
theta_best_pinv = np.linalg.pinv(X_b) @ y
print("Tham số tối ưu tính bằng Nghịch đảo giả (pinv):\n", theta_best_pinv)

# Cả hai cách đều cho ra kết quả chính xác tương đồng:
# [[4.21509616]
#  [2.77011339]]
```

---

### PHẦN 2: THUẬT TOÁN HẠ GRADIENT & CÁC BIẾN THỂ (BATCH GD, STOCHASTIC GD, MINI-BATCH GD)

---

##### 1. Thuật toán hạ Gradient (Gradient Descent - GD)
*   **Giải thích bản chất:** 
    *   **Hạ Gradient** là một thuật toán tối ưu hóa tổng quát có khả năng tìm ra giải pháp tối ưu cho nhiều bài toán khác nhau bằng cách điều chỉnh các tham số một cách lặp đi lặp lại để giảm thiểu hàm chi phí.
    *   **Phép ẩn dụ thực tế:** Hãy tưởng tượng bạn đang bị lạc trên một vùng núi trong một lớp sương mù dày đặc và chỉ có thể cảm nhận được độ dốc của mặt đất dưới chân mình. Một chiến lược khôn ngoan để xuống thung lũng (cực tiểu của hàm chi phí) nhanh nhất là đi xuống dốc theo hướng dốc nhất. Thuật toán hạ Gradient hoạt động y hệt như vậy: nó đo lường độ dốc cục bộ (gradient) của hàm lỗi đối với vector tham số \\(\theta\\), rồi đi theo hướng dốc giảm dần. Khi độ dốc này bằng 0, bạn đã đạt đến điểm cực tiểu!
    *   **Quá trình hoạt động:** Thuật toán bắt đầu bằng việc gán các giá trị ngẫu nhiên cho vector tham số \\(\theta\\) (được gọi là **khởi tạo ngẫu nhiên - random initialization**). Sau đó, nó cải thiện từng bước nhỏ, mỗi bước cố gắng giảm hàm chi phí (như MSE) cho đến khi thuật toán hội tụ đến điểm cực tiểu.
    *   **Tính chất của hàm chi phí MSE:** May mắn thay, hàm chi phí MSE của mô hình Hồi quy tuyến tính là một **hàm lồi (convex function)**. Điều này có nghĩa là nếu chọn bất kỳ hai điểm nào trên đường cong, đoạn thẳng nối chúng sẽ không bao giờ nằm dưới đường cong. Hệ quả cực kỳ quan trọng là hàm số này **không có cực tiểu cục bộ nào, chỉ có duy nhất một cực tiểu toàn cục**. Nó cũng là một hàm liên tục với độ dốc không bao giờ thay đổi đột ngột, đảm bảo thuật toán sẽ tiến sát đến cực tiểu toàn cục nếu được thiết lập tốc độ học phù hợp.
    *   **Tầm quan trọng của chuẩn hóa đặc trưng (Feature Scaling):** Nếu các đặc trưng đầu vào có thang đo (scale) rất khác nhau, hàm chi phí MSE sẽ có hình dạng một cái bát bị kéo giãn rất dài. Khi đó, thuật toán sẽ mất rất nhiều thời gian vì nó phải đi ngoằn ngoèo qua một thung lũng gần như phẳng trước khi tới được đích. Chuẩn hóa dữ liệu giúp cái bát trở nên tròn trịa và cân đối hơn, giúp các bước hạ gradient đi thẳng về phía cực tiểu toàn cục một cách nhanh chóng.
*   **Ví dụ thực tế trong tài liệu:** Tài liệu sử dụng thuật toán này để tìm vector tham số tối ưu \\(\theta\\) cho mô hình hồi quy tuyến tính thay thế cho phương pháp giải tích (như Phương trình chuẩn tắc) khi số lượng đặc trưng quá lớn không thể tính toán nghịch đảo ma trận một cách hiệu quả.
*   **Giải thích trực quan dựa trên hình ảnh:**
    *   **Sơ đồ bước học hướng tới cực tiểu (Hình 4-3):** Thể hiện một đường cong chi phí hình chén lồi đều đặn. Từ điểm khởi tạo ngẫu nhiên ở sườn dốc bên trái, thuật toán thực hiện các bước nhảy (learning steps) đi xuống. Kích thước bước đi tỉ lệ thuận với độ dốc của hàm chi phí, do đó các bước đi sẽ lớn khi ở trên cao và nhỏ dần khi tiến sát về phía đáy cực tiểu (Minimum).
    *   **Tốc độ học quá nhỏ (Hình 4-4) & Tốc độ học quá cao (Hình 4-5):** 
        *   Nếu **tốc độ học quá nhỏ**, thuật toán sẽ di chuyển với các bước cực kỳ li ti, đòi hỏi hàng ngàn lần lặp (epochs) mới có thể hội tụ, gây lãng phí tài nguyên tính toán.
        *   Nếu **tốc độ học quá lớn**, thuật toán sẽ nhảy vọt qua thung lũng sang sườn dốc đối diện, thậm chí điểm sau còn cao hơn điểm trước. Điều này khiến thuật toán bị phân kỳ (diverge) và không bao giờ tìm được lời giải tốt.
    *   **Cạm bẫy địa hình không đều (Hình 4-6):** Đối với các hàm chi phí phức tạp phi tuyến (như trong mạng nơ-ron), địa hình có thể xuất hiện các hố sâu (cực tiểu cục bộ - local minimum) hoặc các vùng bằng phẳng (cao nguyên - plateau). Nếu khởi tạo ngẫu nhiên nằm ở vùng không thuận lợi, thuật toán có thể bị kẹt lại ở cực tiểu cục bộ (không tốt bằng cực tiểu toàn cục) hoặc tốn quá nhiều thời gian để bò qua vùng cao nguyên rồi bị dừng sớm giữa chừng.
    *   **Đường đồng mức có và không chuẩn hóa đặc trưng (Hình 4-7):** 
        *   Biểu đồ bên trái (đã chuẩn hóa): Các đường đồng mức là các vòng tròn đồng tâm hoàn hảo, thuật toán đi thẳng một mạch từ ngoài vào tâm cực tiểu toàn cục.
        *   Biểu đồ bên phải (chưa chuẩn hóa): Đường đồng mức là các hình elip dẹt bị kéo dài theo trục đặc trưng có thang đo lớn. Các bước hạ gradient di chuyển vuông góc với đường đồng mức nên bị dao động mạnh, đi đường vòng rất dài mới tới được đáy.

---

##### 2. Hạ Gradient theo lô (Batch Gradient Descent - BGD)
*   **Giải thích bản chất:** 
    *   **Hạ Gradient theo lô** tính toán đạo hàm riêng của hàm chi phí đối với từng tham số mô hình \\(\theta_j\\) bằng cách sử dụng **toàn bộ dữ liệu huấn luyện** tại mỗi bước lặp. Điều này giống như việc bạn đứng ở một điểm trên núi, nhìn xung quanh mọi hướng và tính toán độ dốc chính xác của toàn bộ ngọn núi trước khi quyết định bước tiếp một bước.
    *   **Công thức Toán học:**
        *   Đạo hàm riêng của MSE đối với từng tham số \\(\theta_j\\) (Công thức 4-5):
            \\[\frac{\partial}{\partial\theta_j} \text{MSE}(\theta) = \frac{2}{m} \sum_{i=1}^{m} \left( \theta^T x^{(i)} - y^{(i)} \right) x_j^{(i)}\\]
        *   Đóng gói dưới dạng vector Gradient chứa tất cả các đạo hàm riêng (Công thức 4-6):
            \\[\nabla_{\theta} \text{MSE}(\theta) = \begin{pmatrix} \frac{\partial}{\partial\theta_0} \text{MSE}(\theta) \\ \frac{\partial}{\partial\theta_1} \text{MSE}(\theta) \\ \vdots \\ \frac{\partial}{\partial\theta_n} \text{MSE}(\theta) \end{pmatrix} = \frac{2}{m} X^T (X\theta - y)\\]
        *   Công thức cập nhật tham số (Công thức 4-7):
            \\[\theta^{\text{(bước tiếp theo)}} = \theta - \eta \nabla_{\theta} \text{MSE}(\theta)\\]
            *(Trong đó: \\(\eta\\) là tốc độ học - learning rate).*
*   **Ưu và nhược điểm:**
    *   *Ưu điểm:* Đường đi cực kỳ ổn định, mượt mà và được đảm bảo hội tụ chính xác đến cực tiểu toàn cục đối với các hàm lồi.
    *   *Nhược điểm:* Vì phải duyệt qua toàn bộ \\(m\\) mẫu dữ liệu huấn luyện chỉ để thực hiện một bước cập nhật duy nhất, thuật toán trở nên **cực kỳ chậm chạp và tốn bộ nhớ** khi tập huấn luyện có quy mô lớn.
*   **Giải thích trực quan dựa trên hình ảnh:**
    *   **So sánh tốc độ học khác nhau (Hình 4-8):** Thể hiện 20 bước huấn luyện đầu tiên của 3 mô hình tuyến tính với các mức \\(\eta\\) khác nhau:
        *   \\(\eta = 0.02\\) (quá thấp): Các đường dự đoán (màu đỏ) di chuyển rất chậm từ đáy lên, cần nhiều thời gian để khớp dữ liệu.
        *   \\(\eta = 0.1\\) (tối ưu): Chỉ sau vài bước lặp, đường dự đoán đã nhanh chóng áp sát và khớp hoàn hảo với các điểm dữ liệu thực tế.
        *   \\(\eta = 0.5\\) (quá cao): Các đường dự đoán nhảy loạn xạ, phân kỳ hoàn toàn ra khỏi vùng dữ liệu.
*   **Mã nguồn Python minh họa:**
```python
import numpy as np

# Giả thiết X_b (đã thêm đặc trưng dummy x0=1) và y đã được định nghĩa từ Phần 1
X_b = np.c_[np.ones((100, 1)), X]  # Thêm cột x0 = 1 vào ma trận đặc trưng

# Thiết lập siêu tham số
eta = 0.1          # Tốc độ học (learning rate)
n_epochs = 1000    # Số vòng lặp huấn luyện
m = len(X_b)       # Số lượng mẫu dữ liệu

# Khởi tạo tham số ngẫu nhiên
np.random.seed(42)
theta = np.random.randn(2, 1)

# Vòng lặp huấn luyện Batch Gradient Descent
for epoch in range(n_epochs):
    # Tính vector gradient cho toàn bộ tập dữ liệu (Công thức 4-6)
    gradients = 2/m * X_b.T @ (X_b @ theta - y)
    
    # Cập nhật tham số theo hướng ngược chiều gradient (Công thức 4-7)
    theta = theta - eta * gradients

print("Tham số tối ưu tìm được bằng Batch GD:\n", theta)
# Kết quả: [[4.21509616], [2.77011339]] - Trùng khớp hoàn toàn với phương trình giải tích!
```

---

##### 3. Hạ Gradient ngẫu nhiên (Stochastic Gradient Descent - SGD)
*   **Giải thích bản chất:** 
    *   Trái ngược hoàn toàn với Batch GD, **Hạ Gradient ngẫu nhiên** chọn ngẫu nhiên **chỉ một mẫu dữ liệu duy nhất** tại mỗi bước lặp và tính toán gradient dựa trên mẫu đó để cập nhật tham số ngay lập tức.
    *   **Ưu điểm:** Thuật toán chạy cực kỳ nhanh và tốn rất ít bộ nhớ vì tại một thời điểm chỉ cần xử lý đúng một mẫu dữ liệu. Nó cho phép huấn luyện trên các tập dữ liệu khổng lồ vượt quá dung lượng RAM vật lý (học ngoài lõi - **out-of-core learning**).
    *   **Đặc tính dao động và Khả năng vượt bẫy:** Do chỉ dựa vào một mẫu ngẫu nhiên, hướng đi của SGD không mượt mà mà liên tục dao động lên xuống thất thường. Tuy nhiên, sự dao động mạnh mẽ này lại là một lợi thế giúp mô hình **dễ dàng nhảy thoát khỏi các cực tiểu cục bộ** để tìm kiếm cực tiểu toàn cục tốt hơn.
    *   **Lịch trình học (Learning Schedule):** Để khắc phục nhược điểm mô hình không bao giờ đứng yên tại cực tiểu mà liên tục nảy xung quanh nó khi hội tụ, người ta áp dụng kỹ thuật giảm dần tốc độ học theo thời gian (giống thuật toán **ủ kim loại** - simulated annealing). Tốc độ học ban đầu lớn để di chuyển nhanh và thoát bẫy cực tiểu cục bộ, sau đó nhỏ dần để mô hình ổn định tại cực tiểu toàn cục.
*   **Ví dụ thực tế trong tài liệu:** Scikit-Learn cung cấp lớp `SGDRegressor` để thực hiện hồi quy tuyến tính bằng SGD một cách tối ưu.
*   **Giải thích trực quan dựa trên hình ảnh:**
    *   **Sự biến thiên thất thường của SGD (Hình 4-9):** Bản đồ đồng mức của hàm chi phí cho thấy đường đi của SGD như một nét vẽ ngoằn ngoèo, nhảy liên tục từ sườn này sang sườn khác. Khi đến vùng trung tâm cực tiểu, nó tiếp tục quay vòng và nhảy xung quanh đáy mà không bao giờ hội tụ tĩnh tại một điểm duy nhất.
    *   **20 bước đầu tiên của SGD (Hình 4-10):** Biểu diễn các đường thẳng dự đoán đầu tiên của thuật toán. Các đường thẳng này thay đổi góc dốc liên tục một cách rất đột ngột và mất trật tự, phản ánh tính chất ngẫu nhiên cao độ trong từng bước cập nhật.
*   **Mã nguồn Python minh họa:**
```python
from sklearn.linear_model import SGDRegressor

# 1. Tự triển khai SGD thủ công với Lịch trình học (Learning Schedule)
n_epochs = 50
t0, t1 = 5, 50  # Siêu tham số của lịch trình học

def learning_schedule(t):
    return t0 / (t + t1)

np.random.seed(42)
theta_sgd = np.random.randn(2, 1)  # Khởi tạo tham số ngẫu nhiên

for epoch in range(n_epochs):
    # Xáo trộn dữ liệu ngẫu nhiên tại mỗi epoch để tăng tính ngẫu nhiên chuẩn xác
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    for i in range(m):
        xi = X_b_shuffled[i:i+1]
        yi = y_shuffled[i:i+1]
        
        # Tính toán gradient cho 1 mẫu duy nhất (không chia cho m)
        gradients = 2 * xi.T @ (xi @ theta_sgd - yi)
        
        # Cập nhật tốc độ học giảm dần theo thời gian
        eta = learning_schedule(epoch * m + i)
        
        # Cập nhật tham số
        theta_sgd = theta_sgd - eta * gradients

print("Tham số tối ưu tính thủ công bằng SGD:\n", theta_sgd)

# 2. Sử dụng thư viện Scikit-Learn qua SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01, 
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())

print("Tham số tối ưu bằng SGDRegressor Scikit-Learn:")
print("Hệ số chệch (bias):", sgd_reg.intercept_)
print("Trọng số đặc trưng:", sgd_reg.coef_)
```

---

##### 4. Hạ Gradient theo lô nhỏ (Mini-batch Gradient Descent - MBGD)
*   **Giải thích bản chất:** 
    *   **Hạ Gradient theo lô nhỏ** là sự kết hợp hài hòa giữa Batch GD và Stochastic GD. Tại mỗi bước lặp, thay vì dùng toàn bộ tập dữ liệu hay chỉ dùng một mẫu duy nhất, thuật toán tính toán gradient trên một tập hợp con ngẫu nhiên có kích thước nhỏ gọi là **mini-batch** (thường từ 32 đến 256 mẫu).
    *   **Ưu điểm phần cứng vượt trội:** Lợi ích lớn nhất của Mini-batch GD là cho phép tận dụng tối đa sức mạnh tính toán song song của các kiến trúc phần cứng hiện đại như **card đồ họa GPU** thông qua các phép toán ma trận được tối ưu hóa.
    *   **Đặc tính đường đi:** Đường đi của Mini-batch GD trong không gian tham số ít biến động và mượt mà hơn SGD (đặc biệt khi tăng kích thước lô nhỏ). Nó di chuyển ổn định hơn và có xu hướng tiến sát đến cực tiểu hơn so với SGD, tuy nhiên khả năng thoát khỏi các bẫy cực tiểu cục bộ sẽ kém hơn một chút.
*   **Ví dụ thực tế trong tài liệu:** Kỹ thuật này được áp dụng xuyên suốt và là cốt lõi cho việc huấn luyện các mạng nơ-ron sâu (Deep Learning) trong các chương sau của tài liệu.
*   **Giải thích trực quan dựa trên hình ảnh:**
    *   **Đường đi của 3 thuật toán trong không gian tham số (Hình 4-11):** 
        *   **Batch GD (Màu xanh dương nét trơn):** Đi một đường cong hoàn hảo, mượt mà và dừng lại chính xác tại tâm điểm cực tiểu toàn cục.
        *   **Stochastic GD (Màu đỏ nét răng cưa lớn):** Đường đi cực kỳ hỗn loạn, nhảy nhót liên tục và dao động rất mạnh quanh tâm cực tiểu.
        *   **Mini-batch GD (Màu xanh lá nét răng cưa nhỏ):** Đường đi có dao động nhưng biên độ nhỏ hơn nhiều so với SGD, tiến về rất sát cực tiểu toàn cục và dao động ổn định quanh một vùng không gian nhỏ hẹp ở trung tâm.

---

##### 5. Bảng so sánh tổng hợp các thuật toán huấn luyện Hồi quy tuyến tính (Bảng 4-1)

| Thuật toán | Quy mô tập dữ liệu (\\(m\\) lớn) | Hỗ trợ học ngoài lõi (Out-of-core) | Số lượng đặc trưng (\\(n\\) lớn) | Số lượng siêu tham số cần tinh chỉnh | Yêu cầu chuẩn hóa đặc trưng | Hỗ trợ trong thư viện Scikit-Learn |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phương trình chuẩn tắc** | Nhanh | Không | Chậm | 0 | Không | N/A (Không trực tiếp) |
| **Phân tách SVD** | Nhanh | Không | Chậm | 0 | Không | `LinearRegression` |
| **Hạ Gradient theo lô** | Chậm | Không | Nhanh | 2 | Có | N/A |
| **Hạ Gradient ngẫu nhiên** | Nhanh | Có | Nhanh | \\(\ge 2\\) | Có | `SGDRegressor` |
| **Hạ Gradient theo lô nhỏ** | Nhanh | Có | Nhanh | \\(\ge 2\\) | Có | N/A (Có thể dùng `partial_fit`) |

---

*Sau khi huấn luyện xong, sự khác biệt giữa các thuật toán biến mất hoàn toàn: tất cả đều tạo ra các mô hình có tham số tương tự nhau và đưa ra dự đoán với tốc độ nhanh tương đương nhau.*

---

Dưới đây là chi tiết **Phần 3: Hồi quy đa thức & Đánh giá mô hình qua Đường cong học tập (Learning Curves)** được xây dựng theo cấu trúc chuẩn hóa, kết hợp chặt chẽ giữa lý thuyết thống kê, mã nguồn thực thi và phân tích trực quan dựa trên các sơ đồ hình ảnh có trong tài liệu của chương.

---

### PHẦN 3: HỒI QUY ĐA THỨC & ĐÁNH GIÁ MÔ HÌNH QUA ĐƯỜNG CONG HỌC TẬP (LEARNING CURVES)

##### 1. Hồi quy đa thức (Polynomial Regression)
*   **Giải thích bản chất:**
    *   Trong thực tế, dữ liệu thường phức tạp và không tuân theo một đường thẳng tuyến tính đơn giản. Tuy nhiên, chúng ta hoàn toàn có thể sử dụng một mô hình tuyến tính để khớp với dữ liệu phi tuyến bằng một kỹ thuật gọi là **Hồi quy đa thức**.
    *   **Cách thức hoạt động:** Thêm các lũy thừa của mỗi đặc trưng gốc để tạo thành các đặc trưng mới, sau đó tiến hành huấn luyện mô hình Hồi quy tuyến tính thông thường trên tập đặc trưng mở rộng này.
    *   **Khả năng học mối quan hệ giữa các đặc trưng:** Khi dữ liệu có nhiều đặc trưng đầu vào, Hồi quy đa thức vượt trội hơn hồi quy tuyến tính ở chỗ nó có khả năng tìm ra mối quan hệ tương hỗ giữa các đặc trưng. Điều này đạt được là do lớp biến đổi `PolynomialFeatures` của Scikit-Learn sẽ tự động thêm tất cả các sự kết hợp của các đặc trưng lên đến bậc đa thức được chỉ định. 
        *   *Ví dụ:* Nếu đầu vào có hai đặc trưng \\(a\\) và \\(b\\), việc sử dụng bậc đa thức `degree=3` sẽ không chỉ tạo ra các đặc trưng bậc cao đơn lẻ \\(a^2, a^3, b^2, b^3\\), mà còn tạo ra các đặc trưng kết hợp tích chéo như \\(ab, a^2b, ab^2\\).
*   **Ví dụ thực tế trong tài liệu:** Tài liệu tạo ra một tập dữ liệu phi tuyến tính gồm 100 mẫu dựa trên phương trình bậc hai đơn giản có thêm nhiễu ngẫu nhiên: \\(y = 0.5x_1^2 + x_1 + 2 + \text{Gaussian noise}\\). Sau đó, áp dụng `PolynomialFeatures(degree=2, include_bias=False)` để biến đổi đặc trưng \\(x_1\\) ban đầu thành bộ đôi đặc trưng \\([x_1, x_1^2]\\) rồi đưa vào huấn luyện.
*   **Giải thích trực quan dựa trên hình ảnh trong tài liệu:**
    *   **Trực quan hóa tập dữ liệu phi tuyến (Hình 4-12):** Tập dữ liệu phân bổ theo hình dạng một đường cong parabol hướng lên (đường cong chữ U). Rõ ràng, một đường thẳng tuyến tính thuần túy sẽ không bao giờ có thể biểu diễn chính xác xu hướng này.
    *   **Dự đoán của mô hình hồi quy đa thức bậc 2 (Hình 4-13):** Sau khi biến đổi đặc trưng và khớp mô hình hồi quy tuyến tính, đường dự đoán thu được là một đường cong parabol màu đỏ rất mượt mà. Mô hình ước tính phương trình có dạng \\(\hat{y} = 0.56x_1^2 + 0.93x_1 + 1.78\\). Kết quả này cực kỳ sát với hàm gốc lý thuyết ban đầu là \\(y = 0.5x_1^2 + 1.0x_1 + 2.0\\).
    *   **Hồi quy đa thức bậc cao (Hình 4-14):** Đồ thị so sánh kết quả khớp dữ liệu giữa 3 mô hình: đường thẳng tuyến tính thông thường (bậc 1), đường cong bậc 2 và đường uốn lượn của mô hình đa thức bậc 300. Trong khi mô hình bậc 1 quá đơn giản (đường thẳng đâm xuyên qua parabol), thì mô hình bậc 300 uốn lượn dữ dội, cố gắng bám sát từng điểm dữ liệu huấn luyện đơn lẻ bất chấp các dao động nhiễu.
*   **Mã nguồn Python minh họa:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Tạo tập dữ liệu phi tuyến tính bậc hai (Hình 4-12)
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# 2. Biến đổi dữ liệu bằng cách thêm đặc trưng bậc hai (x^2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# 3. Huấn luyện mô hình Linear Regression trên tập dữ liệu mở rộng (Hình 4-13)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Xem các hệ số ước lượng được
print("Hệ số chặn (intercept_):", lin_reg.intercept_)
print("Các trọng số đặc trưng (coef_):", lin_reg.coef_)

# 4. Trực quan hóa kết quả dự đoán của mô hình
X_new = np.linspace(-3, 3, 100).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.", label="Dữ liệu thực tế")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Dự đoán (Bậc 2)")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([-3, 3, 0, 10])
plt.grid(True)
plt.legend(loc="upper left")
plt.title("Hồi quy đa thức bậc hai")
plt.show()
```

##### 2. Hiện tượng Quá khớp (Overfitting) và Dưới khớp (Underfitting)
*   **Giải thích bản chất:**
    *   **Dưới khớp (Underfitting):** Xảy ra khi mô hình quá đơn giản để có thể nắm bắt được cấu trúc ẩn sâu của dữ liệu. 
        *   *Ví dụ trong tài liệu:* Việc sử dụng một đường thẳng tuyến tính (bậc 1) để cố gắng khớp với tập dữ liệu parabol bậc hai. Mô hình này hoạt động kém trên cả tập dữ liệu huấn luyện lẫn dữ liệu xác thực thực tế.
    *   **Quá khớp (Overfitting):** Xảy ra khi mô hình quá phức tạp và nhạy cảm, cố gắng ghi nhớ cả các dao động nhiễu ngẫu nhiên trong dữ liệu huấn luyện thay vì học quy luật tổng quát.
        *   *Ví dụ trong tài liệu:* Sử dụng mô hình đa thức bậc 300. Mô hình uốn lượn cực kỳ phức tạp để đi qua sát nhất mọi điểm huấn luyện. Kết quả là nó đạt độ chính xác hoàn hảo trên tập huấn luyện nhưng khi đem dự báo trên dữ liệu mới (dữ liệu xác thực) thì sai số cực kỳ lớn.
*   **Cách phát hiện thông qua hiệu năng huấn luyện và kiểm định:**
    *   Nếu một mô hình hoạt động cực tốt trên dữ liệu huấn luyện nhưng lại mang lại kết quả tổng quát hóa tồi tệ trên các tập xác thực hoặc kiểm định chéo \\(\to\\) Mô hình đang bị **quá khớp**.
    *   Nếu mô hình hoạt động kém cỏi trên cả tập huấn luyện lẫn kiểm định chéo \\(\to\\) Mô hình đang bị **dưới khớp**.

##### 3. Đường cong học tập (Learning Curves)
*   **Giải thích bản chất:** 
    *   **Đường cong học tập** là đồ thị biểu diễn sai số hiệu suất của mô hình (thường dùng lỗi RMSE) trên cả tập huấn luyện và tập xác thực dưới dạng một hàm số phụ thuộc vào **quy mô của tập huấn luyện (hoặc số lần lặp/epoch)**.
    *   Để vẽ đồ thị này, mô hình được huấn luyện lặp đi lặp lại trên các tập con có kích thước tăng dần của tập huấn luyện gốc, sau đó đánh giá sai số trên chính tập con đó và trên toàn bộ tập xác thực.
*   **Giải thích trực quan dựa trên hình ảnh trong tài liệu:**
    *   **Đường cong học tập điển hình cho mô hình Dưới khớp (Hình 4-15):**
        *   *Đường lỗi huấn luyện (Đường nét liền màu đỏ có dấu cộng):* Khi tập huấn luyện chỉ có 1 hoặc 2 mẫu dữ liệu, mô hình tuyến tính dễ dàng khớp hoàn hảo, khiến lỗi huấn luyện xuất phát từ điểm 0. Khi thêm nhiều mẫu dữ liệu mới, do dữ liệu có chứa nhiễu và bản chất là phi tuyến tính, đường thẳng không thể khớp hoàn hảo nữa, khiến lỗi huấn luyện tăng vọt lên. Sau đó, lỗi đạt đến một **cao nguyên phẳng (plateau)** và việc thêm dữ liệu mới cũng không làm lỗi thay đổi nhiều.
        *   *Đường lỗi xác thực (Đường nét liền màu xanh dương):* Khi huấn luyện trên quá ít mẫu dữ liệu, mô hình không thể tổng quát hóa đúng, dẫn đến lỗi xác thực ban đầu cực kỳ lớn. Khi dữ liệu huấn luyện tăng lên, mô hình học được nhiều quy luật hơn và lỗi xác thực giảm dần. Tuy nhiên, vì đường thẳng không thể khớp tốt dữ liệu cong, đường cong này cũng nhanh chóng chạm cao nguyên và dừng lại ở một mức lỗi khá cao, nằm sát sạt đường lỗi huấn luyện.
        *   *Hệ quả rút ra:* Đối với một mô hình đang bị dưới khớp, việc **cố gắng thu thập thêm nhiều dữ liệu huấn luyện sẽ hoàn toàn vô ích** (lỗi sẽ không thể giảm xuống nữa). Giải pháp duy nhất là chọn mô hình phức tạp hơn hoặc bổ sung các đặc trưng tốt hơn.
    *   **Đường cong học tập điển hình cho mô hình Quá khớp (Hình 4-16 - Mô hình đa thức bậc 10):**
        *   *Sự khác biệt cốt lõi thứ nhất:* Lỗi trên dữ liệu huấn luyện (đường màu đỏ) duy trì ở mức **thấp hơn rất nhiều** so với mô hình tuyến tính dưới khớp.
        *   *Sự khác biệt cốt lõi thứ hai:* Xuất hiện một **khoảng cách rõ rệt (gap)** giữa đường lỗi huấn luyện và đường lỗi xác thực (đường màu xanh dương). Lỗi huấn luyện luôn thấp hơn nhiều so với lỗi xác thực, chứng tỏ mô hình hoạt động rất tốt trên tập huấn luyện nhưng tổng quát hóa kém trên tập dữ liệu chưa biết.
        *   *Hệ quả rút ra:* Nếu chúng ta cung cấp thêm một tập dữ liệu huấn luyện với quy mô lớn hơn nhiều, khoảng cách giữa hai đường cong này sẽ tiếp tục thu hẹp lại và tiến sát về nhau. Do đó, **thêm dữ liệu huấn luyện là giải pháp hữu hiệu để chữa trị lỗi quá khớp**.
*   **Mã nguồn Python minh họa vẽ đường cong học tập:**
```python
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression

# Vẽ đường cong học tập cho mô hình Hồi quy tuyến tính (Hình 4-15)
train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error"
)

# Chuyển đổi điểm số âm RMSE của Scikit-Learn thành giá trị lỗi dương thực tế
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.figure(figsize=(6, 4))
plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="Tập huấn luyện (train)")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="Tập xác thực (valid)")
plt.xlabel("Quy mô tập huấn luyện (Training set size)")
plt.ylabel("RMSE")
plt.axis([0, 80, 0, 2.5])
plt.grid(True)
plt.legend(loc="upper right")
plt.title("Đường cong học tập của mô hình bị dưới khớp")
plt.show()
```

##### 4. Sự đánh đổi giữa Độ chệch và Phương sai (Bias / Variance Trade-off)
*   **Giải thích bản chất:** 
    *   Một kết quả lý thuyết vô cùng quan trọng của thống kê toán học và học máy chứng minh rằng: **Lỗi tổng quát hóa (generalization error) của một mô hình học máy luôn là tổng hợp của ba loại sai số hoàn toàn khác nhau**:
        1.  **Độ chệch (Bias):** Sai số phát sinh từ những giả định sai lầm của con người về bản chất thực tế của dữ liệu (ví dụ giả định dữ liệu phân bổ tuyến tính phẳng trong khi thực tế nó là đường cong parabol bậc hai). Mô hình có độ chệch cao sẽ **gây ra hiện tượng dưới khớp** dữ liệu huấn luyện.
        2.  **Phương sai (Variance):** Sai số do mô hình quá nhạy cảm và phản ứng thái quá với các biến động nhỏ cục bộ của dữ liệu huấn luyện. Các mô hình có quá nhiều bậc tự do (như đa thức bậc cao uốn lượn) có xu hướng có phương sai cao và **gây ra hiện tượng quá khớp** dữ liệu huấn luyện.
        3.  **Lỗi không thể giảm (Irreducible Error):** Sai số tự nhiên phát sinh do tính chất nhiễu bộc phát của chính dữ liệu đầu vào. Cách duy nhất để giảm thiểu phần lỗi này là làm sạch nguồn dữ liệu (như sửa chữa cảm biến bị hỏng, loại bỏ ngoại lai) chứ không thể khắc phục bằng cách thay đổi mô hình toán học.
*   **Mối quan hệ đánh đổi (Trade-off):**
    *   Có một quy luật bất biến: việc **tăng độ phức tạp** của mô hình (tăng bậc tự do) sẽ làm **tăng phương sai** nhưng giúp **giảm độ chệch** của nó. 
    *   Ngược lại, việc **giảm độ phức tạp** của mô hình (thắt chặt các ràng buộc hoặc giảm bậc tự do) sẽ làm **tăng độ chệch** nhưng giúp **giảm phương sai**.
    *   Nhiệm vụ cốt lõi của kỹ sư học máy là tìm ra điểm cân bằng tối ưu giữa hai thái cực này để mô hình đạt hiệu năng tổng quát hóa tốt nhất trên dữ liệu thực tế.

---

*Lưu ý: Do hệ thống của chúng ta đang hoạt động ở chế độ đọc (Read-only), bạn không thể lưu trữ trực tiếp các tệp tin mới vào notebook. Nếu muốn thực thi trực tiếp các đoạn mã vẽ đồ thị này hoặc lưu trữ chúng thành các tệp tin báo cáo, bạn vui lòng tạo một bản sao cá nhân (personal copy) của notebook này.*

---

### PHẦN 4: CÁC MÔ HÌNH TUYẾN TÍNH ĐƯỢC CHÍNH QUY HÓA

Một cách hiệu quả để giảm thiểu hiện tượng **quá khớp (overfitting)** là thực hiện **chính quy hóa (regularization)** mô hình (tức là áp đặt thêm các ràng buộc đối với nó). Mô hình càng có ít bậc tự do thì càng khó để quá khớp dữ liệu. Đối với mô hình tuyến tính, việc chính quy hóa thường đạt được bằng cách thắt chặt và ràng buộc các trọng số của mô hình. 

Dưới đây là phân tích chi tiết về ba phương pháp ràng buộc trọng số phổ biến nhất: **Hồi quy Ridge**, **Hồi quy Lasso**, **Hồi quy Elastic Net** và kỹ thuật tối ưu hóa động **Dừng sớm (Early Stopping)**.

---

##### 1. Hồi quy Ridge (Ridge Regression / Chuẩn hóa Tikhonov)
*   **Giải thích bản chất & Tối ưu hóa hàm chi phí:**
    *   **Hồi quy Ridge** là một phiên bản được chính quy hóa của hồi quy tuyến tính. Trong quá trình huấn luyện, thuật toán không chỉ tìm cách khớp với dữ liệu huấn luyện mà còn bị bắt buộc phải giữ cho các trọng số của mô hình nhỏ nhất có thể.
    *   **Số hạng chính quy hóa** (số hạng phạt) được cộng trực tiếp vào hàm chi phí trong quá trình huấn luyện. Sau khi mô hình đã được huấn luyện xong, chúng ta chỉ sử dụng các thước đo lỗi không chuẩn hóa như MSE hoặc RMSE để đánh giá hiệu suất thực tế của mô hình.
    *   **Hàm chi phí hồi quy Ridge (Công thức 4-8):**
        \\[J(\theta) = \text{MSE}(\theta) + \alpha \frac{1}{m} \sum_{i=1}^{n} \theta_i^2\\]
    *   **Phân tích toán học:**
        *   Số hạng chệch \\(\theta_0\\) **không được đưa vào chính quy hóa** (tổng bắt đầu từ \\(i = 1\\), không phải \\(0\\)).
        *   Nếu định nghĩa \\(w\\) là vector trọng số đặc trưng (từ \\(\theta_1\\) đến \\(\theta_n\\)), số hạng chính quy hóa có thể viết gọn dưới dạng \\(\alpha \frac{\| w \|_2^2}{m}\\). Trong đó, \\(\| w \|_2\\) đại diện cho **chuẩn \\(l_2\\)** của vector trọng số.
        *   **Siêu tham số \\(\alpha\\)** kiểm soát mức độ chặt chẽ của việc phạt trọng số. Nếu \\(\alpha = 0\\), mô hình quay trở lại làm Hồi quy tuyến tính thông thường. Nếu \\(\alpha\\) rất lớn, tất cả các trọng số của đặc trưng sẽ bị nén về gần bằng 0. Kết quả là mô hình dự đoán ra một đường thẳng nằm ngang đi qua giá trị trung bình của dữ liệu.
        *   Đối với thuật toán **Hạ Gradient theo lô (Batch GD)**, chúng ta chỉ cần cộng thêm vector đạo hàm riêng \\(2\alpha w / m\\) vào phần vector gradient tương ứng với các trọng số đặc trưng (và giữ nguyên phần gradient của hệ số chệch \\(\theta_0\\)).
*   **Giải thích trực quan dựa trên hình ảnh trong tài liệu (Hình 4-17):**
    *   **Đồ thị bên trái (Mô hình tuyến tính với Ridge):** Khi dữ liệu bị nhiễu mạnh, việc tăng dần siêu tham số \\(\alpha\\) từ \\(0\\) (hồi quy tuyến tính thường - nét chấm màu xanh) lên \\(10\\) (nét đứt màu xanh lá) rồi đến \\(100\\) (nét liền màu đỏ) khiến đường dự đoán phẳng hơn (ít cực đoan hơn). Điều này giúp làm giảm phương sai nhưng làm tăng độ chệch của mô hình.
    *   **Đồ thị bên phải (Mô hình đa thức bậc 10 với Ridge):** Thể hiện hồi quy đa thức kết hợp chính quy hóa Ridge. Nếu không có chính quy hóa (\\(\alpha = 0\\)), đường dự báo uốn lượn dữ dội để ôm khít dữ liệu (quá khớp). Nhưng khi tăng dần \\(\alpha\\), đường đa thức bậc 10 dần ổn định, mượt mà và hợp lý hơn rất nhiều.
*   **Nghiệm dạng đóng (Closed-form Solution - Công thức 4-9):**
    \\[\theta = (X^T X + \alpha A)^{-1} X^T y\\]
    *(Trong đó, \\(A\\) là ma trận đơn vị kích thước \\((n+1) \times (n+1)\\) nhưng có ô trên cùng bên trái bằng 0, tương ứng với việc không áp đặt hình phạt lên số hạng chệch \\(\theta_0\\)).*
*   **Mã nguồn Python minh họa:**
```python
import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor

# 1. Giải bằng nghiệm dạng đóng (sử dụng phép phân tách Cholesky)
ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
print("Dự đoán của Ridge dạng đóng:", ridge_reg.predict([[1.5]])) # Output khoảng [[1.5532]] [cite: 59]

# 2. Giải bằng Stochastic Gradient Descent (SGD) với hình phạt l2
# l2_penalty tương đương hồi quy Ridge. Vì SGD của Scikit-Learn không chia cho m 
# đối với số hạng phạt, ta truyền alpha = 0.1 / m để đồng nhất kết quả với Ridge(alpha=0.1)
m = len(X)
sgd_reg = SGDRegressor(penalty="l2", alpha=0.1 / m, tol=None, 
                       max_iter=1000, eta0=0.01, random_state=42)
sgd_reg.fit(X, y.ravel())
print("Dự đoán của Ridge bằng SGD:", sgd_reg.predict([[1.5]])) # Output khoảng [1.5530] [cite: 59]
```

---

##### 2. Hồi quy Lasso (Lasso Regression)
*   **Giải thích bản chất & Tối ưu hóa hàm chi phí:**
    *   **Hồi quy Lasso** (Least Absolute Shrinkage and Selection Operator) là một phương pháp chính quy hóa tuyến tính khác. Nó cũng thêm một số hạng phạt vào hàm chi phí nhưng sử dụng **chuẩn \\(l_1\\)** của vector trọng số thay vì bình phương chuẩn \\(l_2\\).
    *   **Hàm chi phí hồi quy Lasso (Công thức 4-10):**
        \\[J(\theta) = \text{MSE}(\theta) + 2\alpha \sum_{i=1}^{n} |\theta_i|\\]
    *   **Đặc tính tự động lựa chọn đặc trưng (Feature Selection):** 
        *   Một tính chất cực kỳ quan trọng và khác biệt của Lasso là nó có xu hướng **loại bỏ hoàn toàn trọng số của các đặc trưng ít quan trọng nhất** (tức là đặt chúng bằng đúng giá trị 0).
        *   Hệ quả là hồi quy Lasso sẽ tự động thực hiện lựa chọn đặc trưng và cho ra một **mô hình thưa (sparse model)** chỉ giữ lại một vài trọng số đặc trưng khác không.
*   **Giải thích chi tiết về biểu đồ co hẹp trọng số (Hình 4-19):**
    *   **Biểu đồ góc trên (Hình phạt \\(l_1\\) và Lasso):** 
        *   Hàm phạt \\(l_1\\) (\\(\| \theta \|_1\\)) có đường đồng mức dạng hình thoi. Khi chạy thuật toán hạ Gradient từ một điểm khởi tạo ngẫu nhiên bất kỳ (đường đứt nét màu vàng), quỹ đạo di chuyển sẽ nhanh chóng chạm vào trục tọa độ \\(\theta_2 = 0\\) trước (vì hình phạt \\(l_1\\) giảm tuyến tính khi tiến sát về phía trục). 
        *   Sau đó, gradient tiếp tục "lăn" dọc theo máng xối (gutter) của trục hoành \\(\theta_2 = 0\\) để hội tụ về điểm tối ưu toàn cục (hình vuông màu đỏ). Kỹ thuật này giúp loại bỏ trực tiếp các đặc trưng dư thừa ra khỏi mô hình.
    *   **Biểu đồ góc dưới (Hình phạt \\(l_2\\) và Ridge):**
        *   Hàm phạt \\(l_2\\) (\\(\| \theta \|_2\\)) có các đường đồng mức là hình tròn đồng tâm hoàn hảo. Các gradient sẽ nhỏ dần khi tiến sát về gốc tọa độ, giúp thuật toán di chuyển thẳng, chậm dần và hạn chế tối đa sự dao động.
        *   Nhờ vậy, Ridge hội tụ nhanh và ổn định hơn Lasso. Tuy nhiên, các tham số tối ưu (hình vuông màu đỏ) chỉ bị nén lại gần gốc tọa độ hơn khi tăng \\(\alpha\\) chứ **không bao giờ bị triệt tiêu về đúng 0** hoàn toàn.
*   **Cận Gradient (Subgradient Vector):**
    *   Hàm chi phí Lasso không khả vi (không tính được đạo hàm) tại các điểm \\(\theta_i = 0\\) (do tính chất góc nhọn của hàm trị tuyệt đối).
    *   Tuy nhiên, thuật toán hạ Gradient vẫn hoạt động hoàn hảo nếu ta thay thế gradient thông thường bằng một **vector cận gradient (subgradient vector)** \\(g\\) bất cứ khi nào xuất hiện một trọng số \\(\theta_i = 0\\). Vector cận gradient này được định nghĩa chi tiết theo **Công thức 4-11** dựa trên hàm dấu (sign).
*   **Mã nguồn Python minh họa:**
```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
print("Dự đoán của Lasso:", lasso_reg.predict([[1.5]])) # Output: array([1.5378]) [cite: 65]
```

---

##### 3. Hồi quy Elastic Net
*   **Giải thích bản chất & Tối ưu hóa hàm chi phí:**
    *   **Hồi quy Elastic Net** là giải pháp trung hòa hoàn hảo, kết hợp cả hai cơ chế chính quy hóa của Ridge và Lasso.
    *   **Hàm chi phí của Elastic Net (Công thức 4-12):**
        \\[J(\theta) = \text{MSE}(\theta) + r \left( 2\alpha \sum_{i=1}^{n} |\theta_i| \right) + (1 - r) \left( \alpha \frac{1}{m} \sum_{i=1}^{n} \theta_i^2 \right)\\]
    *   **Tham số kiểm soát tỷ lệ pha trộn \\(r\\) (l1_ratio):**
        *   Khi \\(r = 0\\), mô hình tương đương với hồi quy Ridge.
        *   Khi \\(r = 1\\), mô hình tương đương với hồi quy Lasso.
*   **Lời khuyên lựa chọn mô hình trong thực tế:**
    *   Chúng ta **luôn luôn nên áp đặt một chút chính quy hóa** cho mô hình và tránh sử dụng Hồi quy tuyến tính thông thường. Do đó, **Ridge** là một lựa chọn mặc định rất tốt.
    *   Nếu bạn nghi ngờ rằng trong tập dữ liệu chỉ có một vài đặc trưng thực sự hữu ích, hãy ưu tiên dùng **Lasso** hoặc **Elastic Net** vì khả năng đưa các đặc trưng vô ích về 0.
    *   Nhìn chung, **Elastic Net được ưa chuộng hơn Lasso**. Lý do là Lasso có xu hướng hoạt động rất thất thường và thiếu ổn định khi số lượng đặc trưng lớn hơn số lượng mẫu huấn luyện (\\(n > m\\)), hoặc khi xuất hiện hiện tượng đa cộng tuyến (các đặc trưng đầu vào có sự tương quan cực kỳ mạnh mẽ với nhau).
*   **Mã nguồn Python minh họa:**
```python
from sklearn.linear_model import ElasticNet

# Khởi tạo Elastic Net với tỷ lệ trộn l1_ratio (r) = 0.5
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
print("Dự đoán của Elastic Net:", elastic_net.predict([[1.5]])) # Output: array([1.5433]) [cite: 68]
```

---

##### 4. Dừng sớm (Early Stopping) - "Bữa trưa miễn phí tuyệt vời"
*   **Giải thích bản chất & Cơ chế hoạt động:**
    *   **Dừng sớm** là một phương pháp chính quy hóa rất khác biệt dành cho các thuật toán tối ưu hóa lặp như Gradient Descent. Cơ chế của nó là: **Dừng quá trình huấn luyện ngay khi sai số trên tập xác thực đạt giá trị cực tiểu**.
    *   Nhà khoa học máy tính nổi tiếng Geoffrey Hinton đã gọi đây là một **"bữa trưa miễn phí tuyệt vời" (great free lunch)** bởi vì nó vừa cực kỳ đơn giản để triển khai, vừa mang lại hiệu quả vượt trội trong việc kiểm soát hiện tượng quá khớp mà không cần can thiệp sâu vào toán học của mô hình.
*   **Giải thích trực quan dựa trên đồ thị (Hình 4-20):**
    *   Đồ thị biểu thị sai số RMSE theo số lượng epoch huấn luyện.
    *   **Đường lỗi huấn luyện (Đường đứt nét màu đỏ):** Liên tục đi xuống đều đặn qua từng epoch khi mô hình học và ghi nhớ dữ liệu huấn luyện.
    *   **Đường lỗi xác thực (Đường nét liền màu xanh dương):** Ban đầu giảm xuống theo đường lỗi huấn luyện. Tuy nhiên, sau một số lượng epoch nhất định (khoảng epoch thứ 250), lỗi xác thực dừng giảm và bắt đầu có xu hướng **quay đầu tăng ngược trở lại**.
    *   Sự phân kỳ này là minh chứng rõ ràng cho thấy mô hình đã bắt đầu quá khớp với tập dữ liệu huấn luyện. Bằng cách áp dụng **Dừng sớm**, thuật toán sẽ tự động cắt tỉa và dừng ngay tại thời điểm lỗi xác thực chạm đáy (điểm **Best model** màu xanh dương), giúp giữ lại trạng thái tổng quát hóa tối ưu nhất của mô hình.
*   **Kỹ thuật triển khai chi tiết:**
    *   Trong thực tế, khi huấn luyện với hạ Gradient ngẫu nhiên bằng `SGDRegressor`, chúng ta sử dụng phương thức **`partial_fit()`** thay vì `fit()` thông thường để thực hiện học tăng dần qua từng epoch đơn lẻ.
    *   Tại mỗi epoch, ta tính toán lỗi RMSE trên tập xác thực. Nếu lỗi này nhỏ hơn lỗi nhỏ nhất từng ghi nhận trước đó, ta sẽ lưu lại một bản sao sâu của mô hình bằng hàm **`copy.deepcopy()`**.
    *   *Tại sao phải dùng `copy.deepcopy()`?* Hàm này giúp sao chép toàn bộ bao gồm cả siêu tham số và các trọng số thực tế đã học của mô hình tại thời điểm tối ưu đó. Ngược lại, hàm `clone()` của Scikit-Learn chỉ sao chép phần khung siêu tham số rỗng mà không giữ lại trọng số đã học.
*   **Mã nguồn Python minh họa:**
```python
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor

# 1. Pipeline biến đổi chuẩn hóa dữ liệu đa thức bậc cao
preprocessing = make_pipeline(
    PolynomialFeatures(degree=90, include_bias=False),
    StandardScaler()
)
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)

# 2. Khởi tạo mô hình SGDRegressor không chính quy hóa (penalty=None)
sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)

n_epochs = 500
best_valid_rmse = float('inf')
best_model = None

# 3. Vòng lặp huấn luyện tìm điểm Dừng sớm lý tưởng (Hình 4-20)
for epoch in range(n_epochs):
    # Huấn luyện tăng dần từng epoch
    sgd_reg.partial_fit(X_train_prep, y_train.ravel())
    
    # Dự đoán và tính toán lỗi RMSE trên tập xác thực
    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = mean_squared_error(y_valid, y_valid_predict, squared=False)
    
    # Nếu tìm thấy lỗi xác thực thấp hơn, tiến hành lưu trữ mô hình tốt nhất
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg) # Sao chép sâu cả cấu trúc trọng số tối ưu

print(f"Lỗi RMSE tốt nhất trên tập xác thực đạt được: {best_valid_rmse:.4f}")
```

---

*Lưu ý: Do hệ thống của chúng ta đang hoạt động ở chế độ đọc (Read-only), bạn không thể lưu trữ trực tiếp các tệp tin mới vào notebook. Nếu muốn thực thi trực tiếp các đoạn mã vẽ đồ thị này hoặc lưu trữ chúng thành các tệp tin báo cáo, bạn vui lòng tạo một bản sao cá nhân (personal copy) của notebook này.*

---

### PHẦN 5: HỒI QUY PHÂN LOẠI (HỒI QUY LOGISTIC & HỒI QUY SOFTMAX)

---

##### 1. Hồi quy Logistic (Logistic Regression)
*   **Giải thích bản chất:** 
    *   **Hồi quy Logistic** (hay hồi quy logit) là một mô hình phân loại thường được sử dụng để ước tính xác suất một trường hợp dữ liệu thuộc về một lớp cụ thể. 
    *   **Ước tính xác suất và Đưa ra dự đoán:** Thay vì xuất trực tiếp kết quả tuyến tính như hồi quy tuyến tính, mô hình hồi quy Logistic tính tổng trọng số của các đặc trưng đầu vào (cộng với hệ số chệch) rồi đưa kết quả này qua một hàm phi tuyến gọi là **hàm logistic**.
    *   **Công thức Toán học (Công thức 4-13 & 4-14):**
        \\[\hat{p} = h_{\theta}(x) = \sigma(\theta^T x)\\]
        \\[\sigma(t) = \frac{1}{1 + \exp(-t)}\\]
        *(Trong đó: \\(\sigma(t)\\) là hàm sigmoid cho ra giá trị nằm trong khoảng từ 0 đến 1).*
    *   **Ngưỡng quyết định mặc định (Công thức 4-15):** Nếu xác suất ước tính \\(\hat{p} \ge 0.5\\) (tương ứng với điểm số tuyến tính \\(\theta^T x \ge 0\\)), mô hình sẽ dự đoán nhãn dương tính "1". Ngược lại, nếu \\(\hat{p} < 0.5\\) (tương ứng với \\(\theta^T x < 0\\)), mô hình dự đoán nhãn âm tính "0". Do đó, nó hoạt động như một bộ phân loại nhị phân.
*   **Ví dụ thực tế trong tài liệu:** Xây dựng bộ phân loại phát hiện loài hoa diên vĩ **Iris virginica** dựa trên đặc trưng chiều rộng cánh hoa (petal width) từ tập dữ liệu Iris nổi tiếng gồm 150 bông hoa của ba loài khác nhau.
*   **Giải thích trực quan dựa trên hình ảnh trong tài liệu:**
    *   **Sơ đồ Hàm logistic (Hình 4-21):** Đồ thị minh họa hàm sigmoid \\(\sigma(t)\\) có hình chữ S đặc trưng. Hàm số này bị giới hạn tiệm cận trong khoảng \\((0, 1)\\) ở trục tung, đi qua điểm xác suất \\(0.5\\) tại hoành độ \\(t = 0\\). Khi \\(t\\) tiến tới dương vô cùng, xác suất tiến sát về \\(1\\); ngược lại khi \\(t\\) tiến tới âm vô cùng, xác suất tiến dần về \\(0\\).
    *   **Xác suất ước tính và Đường ranh giới quyết định (Hình 4-23):** 
        *   Chiều rộng cánh hoa thực tế của loài *Iris virginica* (được biểu thị bằng các hình tam giác màu xanh lá ở phía trên) dao động từ \\(1.4\\) cm đến \\(2.5\\) cm. Các loài diên vĩ khác (biểu thị bằng các hình vuông màu xanh dương ở phía dưới) dao động từ \\(0.1\\) cm đến \\(1.8\\) cm. 
        *   Đường màu xanh lá thể hiện xác suất mô hình dự đoán là *Iris virginica*, đường nét đứt màu xanh dương thể hiện xác suất "Không phải Iris virginica". Có một khoảng chồng lấn nhỏ giữa hai nhóm hoa từ \\(1.4\\) cm đến \\(1.8\\) cm. 
        *   **Đường ranh giới quyết định (Decision boundary)** được vẽ bằng đường chấm đứng màu đen tại điểm giao nhau của hai đường xác suất (đạt mức 50%), tương ứng với chiều rộng cánh hoa khoảng **\\(1.6\\) cm**. Nếu cánh hoa lớn hơn \\(1.6\\) cm, mô hình tự tin dự đoán đó là *Iris virginica*.
    *   **Đường ranh giới quyết định tuyến tính trên hai đặc trưng (Hình 4-24):** Khi huấn luyện mô hình dựa trên cả hai đặc trưng chiều dài và chiều rộng cánh hoa, đường ranh giới quyết định 50% (đường nét đứt màu đen) là một **đường thẳng phân tách tuyến tính**. Các đường thẳng song song xung quanh biểu thị các mức xác suất dự đoán cụ thể (từ 15% dưới cùng bên trái đến 90% ở góc trên cùng bên phải). Bất kỳ bông hoa nào nằm phía trên đường 90% đều có cơ hội cực kỳ cao là hoa *Iris virginica*.
*   **Mã nguồn Python minh họa:**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Tải tập dữ liệu Iris
iris = load_iris(as_frame=True)
X = iris.data[["petal width (cm)"]].values
# Gán nhãn nhị phân: True nếu là Iris virginica, False nếu là loài khác
y = (iris.target_names[iris.target] == 'virginica')

# 2. Phân chia tập dữ liệu và huấn luyện mô hình
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# 3. Dự đoán xác suất cho các giá trị chiều rộng cánh hoa từ 0 đến 3cm (Hình 4-23)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

# Xác định điểm ranh giới quyết định tại xác suất >= 50%
decision_boundary = X_new[y_proba[:, 1] >= 0.5]
print(f"Ranh giới quyết định tại: {decision_boundary:.2f} cm") # Khoảng 1.65 cm

# Thử nghiệm dự đoán thực tế
print("Dự đoán cho cánh hoa rộng 1.7cm và 1.5cm:", log_reg.predict([[1.7], [1.5]]))
# Kết quả: [True, False]
```

---

##### 2. Hàm chi phí Hồi quy Logistic (Log Loss)
*   **Giải thích bản chất & Toán học:**
    *   Mục tiêu huấn luyện của hồi quy Logistic là thiết lập vector tham số \\(\theta\\) sao cho mô hình gán xác suất cao cho các mẫu dương tính thực tế (\\(y = 1\\)) và xác suất thấp cho các mẫu âm tính thực tế (\\(y = 0\\)).
    *   **Hàm chi phí trên một mẫu dữ liệu đơn lẻ (Công thức 4-16):**
        \\[c(\theta) = \begin{cases} -\log(\hat{p}) & \text{nếu } y = 1 \\ -\log(1 - \hat{p}) & \text{nếu } y = 0 \end{cases}\\]
        *(Giải thích vật lý: Hàm toán học \\(-\log(t)\\) tăng vọt tiệm cận lên vô cùng khi \\(t\\) tiến sát về 0. Do đó, nếu mô hình dự đoán xác suất gần bằng 0 cho một mẫu thực tế dương tính (hoặc dự đoán xác suất gần bằng 1 cho một mẫu âm tính), nó sẽ bị phạt cực kỳ nặng).*
    *   **Hàm chi phí trên toàn bộ tập dữ liệu - Log Loss (Công thức 4-17):** Là trung bình cộng sai số trên tất cả \\(m\\) trường hợp huấn luyện:
        \\[J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(p^{(i)}) + (1 - y^{(i)}) \log(1 - p^{(i)}) \right]\\]
    *   **Độ hội tụ:** Không tồn tại nghiệm dạng đóng giải tích trực tiếp để tối thiểu hóa hàm số này, nhưng **Log loss là một hàm lồi (convex function)**. Nhờ vậy, các thuật toán tối ưu hóa lặp như Gradient Descent được đảm bảo chắc chắn sẽ tìm ra điểm cực tiểu toàn cục tối ưu.
    *   **Vector Gradient của Log Loss (Công thức 4-18):** Đạo hàm riêng đối với tham số \\(\theta_j\\) được tính bằng:
        \\[\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \sigma(\theta^T x^{(i)}) - y^{(i)} \right) x_j^{(i)}\\]
*   **Mã nguồn Python minh họa:**
```python
from sklearn.metrics import log_loss

# Dự đoán xác suất lớp dương tính trên tập kiểm thử
y_pred_proba = log_reg.predict_proba(X_test)

# Tính toán giá trị lỗi Log Loss thực tế
test_loss = log_loss(y_test, y_pred_proba)
print(f"Log Loss trên tập kiểm thử: {test_loss:.4f}")
```

---

##### 3. Hồi quy Softmax (Softmax Regression / Multinomial Logistic Regression)
*   **Giải thích bản chất:** 
    *   **Hồi quy Softmax** là sự tổng quát hóa trực tiếp của Hồi quy Logistic để hỗ trợ phân loại đa lớp (phân biệt nhiều hơn hai lớp khác nhau) mà không cần huấn luyện và kết hợp nhiều bộ phân loại nhị phân độc lập (như chiến lược OvR hay OvO).
    *   **Cơ chế hoạt động:** 
        1.  Với một trường hợp \\(x\\), mô hình tính toán điểm số \\(s_k(x)\\) cho từng lớp \\(k\\) bằng phương trình tuyến tính chuẩn (tương tự như hồi quy tuyến tính):
            \\[s_k(x) = (\theta^{(k)})^T x\\]
            *(Lưu ý: mỗi lớp \\(k\\) sở hữu một vector tham số riêng biệt \\(\theta^{(k)}\\), lưu trữ dưới dạng một hàng trong ma trận tham số lớn \\(\Theta\\)).*
        2.  Sau khi có điểm số của tất cả các lớp, mô hình chạy chúng qua **hàm softmax** để chuẩn hóa và đưa ra xác suất dự đoán \\(p_k\\) của từng lớp:
            \\[p_k = \sigma(s(x))_k = \frac{\exp(s_k(x))}{\sum_{j=1}^{K} \exp(s_j(x))}\\]
            *(Trong đó: \\(K\\) là tổng số lượng các lớp phân loại).*
        3.  Bộ phân loại Softmax đưa ra dự đoán cuối cùng là lớp có xác suất ước lượng cao nhất (tương ứng với lớp có điểm số \\(s_k(x)\\) cao nhất) bằng toán tử **argmax**:
            \\[\hat{y} = \text{argmax}_k s_k(x) = \text{argmax}_k \left( (\theta^{(k)})^T x \right)\\]
*   **Ví dụ thực tế trong tài liệu:** Áp dụng Hồi quy Softmax phân loại hoa diên vĩ thành cả 3 nhóm khác nhau (*Iris setosa, Iris versicolor, và Iris virginica*) dựa trên hai đặc trưng là chiều dài cánh hoa (petal length) và chiều rộng cánh hoa (petal width).
*   **Giải thích trực quan dựa trên hình ảnh trong tài liệu (Hình 4-25):**
    *   Đồ thị biểu diễn các đường ranh giới quyết định thu được bằng màu nền tương ứng: vùng màu vàng là loài *Setosa*, màu xanh dương đại diện cho *Versicolor*, và màu xanh lá cây là *Virginica*.
    *   Đường ranh giới quyết định phân tách giữa bất kỳ cặp hai lớp nào đều là một **đường thẳng tuyến tính**.
    *   Các đường cong đồng mức thể hiện giá trị xác suất cụ thể được ước lượng riêng cho lớp *Iris versicolor*. Ví dụ: đường đồng mức mang nhãn \\(0.30\\) biểu thị tập hợp các điểm mà mô hình ước lượng hoa có đúng 30% khả năng là loài *Versicolor*.
    *   Tại giao điểm trung tâm nơi cả ba đường ranh giới quyết định gặp nhau, mô hình hoàn toàn phân vân khi ước lượng xác suất của cả ba lớp bằng nhau và cùng bằng \\(33.3\%\\).
*   **Mã nguồn Python minh họa:**
```python
from sklearn.linear_model import LogisticRegression

# 1. Trích xuất cả hai đặc trưng: Chiều dài và Chiều rộng cánh hoa
X_multi = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_multi = iris["target"] # Chứa cả 3 nhãn phân loại (0, 1, 2)

# 2. Phân chia tập dữ liệu đa lớp
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, random_state=42)

# 3. LogisticRegression tự động kích hoạt hồi quy Softmax đa lớp khi y có nhiều hơn 2 lớp
# Siêu tham số C kiểm soát mức độ chính quy hóa l2 (C càng lớn chính quy hóa càng giảm)
softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train_m, y_train_m)

# 4. Đưa ra dự đoán cho một bông hoa mới (Dài 5cm, Rộng 2cm)
print("Loài hoa dự đoán (Lớp):", softmax_reg.predict([])) # Kết quả: Lớp 2 (Iris virginica)
print("Xác suất phân bổ chi tiết cho 3 lớp:\n", softmax_reg.predict_proba([]).round(2))
# Kết quả: [[0.0, 0.04, 0.96]] -> 96% tự tin là Iris virginica
```

---

##### 4. Hàm chi phí Hàm mất mát chéo (Cross Entropy)
*   **Giải thích bản chất & Toán học:**
    *   Mục tiêu huấn luyện của hồi quy Softmax là phạt nặng mô hình khi nó ước lượng xác suất thấp cho lớp mục tiêu thực tế. Chỉ số đo lường mức độ phù hợp giữa phân phối xác suất dự đoán và nhãn thực tế là **Cross entropy** (Hàm mất mát chéo).
    *   **Công thức Toán học (Công thức 4-22):**
        \\[J(\Theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(p_k^{(i)})\\]
        *(Trong đó: \\(y_k^{(i)}\\) bằng 1 nếu mẫu thứ \\(i\\) thuộc về lớp \\(k\\), ngược lại bằng 0).*
    *   **Sự tương đồng toán học:** Khi số lượng lớp \\(K = 2\\), hàm chi phí Cross entropy này thu gọn về dạng tương đương hoàn hảo với hàm chi phí Log loss của hồi quy Logistic.
    *   **Vector Gradient Cross Entropy của lớp k (Công thức 4-23):** 
        \\[\nabla_{\theta^{(k)}} J(\Theta) = \frac{1}{m} \sum_{i=1}^{m} \left( p_k^{(i)} - y_k^{(i)} \right) x^{(i)}\\]
        *Sau khi tính được vector gradient này cho từng lớp, chúng ta có thể áp dụng các thuật toán tối ưu hóa (như Gradient Descent) để tìm kiếm ma trận tham số \\(\Theta\\) tối ưu nhất.*
*   **Mã nguồn Python minh họa:**
```python
from sklearn.metrics import log_loss

# Dự đoán phân bổ xác suất đa lớp trên tập kiểm thử
y_pred_proba_m = softmax_reg.predict_proba(X_test_m)

# Tính toán giá trị lỗi Cross Entropy thực tế (sử dụng hàm log_loss đa lớp)
test_cross_entropy = log_loss(y_test_m, y_pred_proba_m)
print(f"Lỗi Cross Entropy trên tập kiểm thử: {test_cross_entropy:.4f}")
```

---

*Lưu ý: Do hệ thống của chúng ta đang hoạt động ở chế độ đọc (Read-only) trong notebook này, bạn không thể tạo trực tiếp các tệp tin mới hay lưu trữ trực tiếp các đồ thị này vào hệ thống của mình. Nếu muốn chạy thử mã nguồn vẽ đồ thị, hãy tạo một bản sao cá nhân (personal copy) của notebook để bắt đầu thực hành nhé.*

---