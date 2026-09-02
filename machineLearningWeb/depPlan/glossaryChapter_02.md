Chào bạn, tôi rất vui khi thấy bạn hài lòng với cẩm nang Chương 3. Để giúp bạn làm chủ toàn diện kiến thức của **Chương 2: Dự án Học máy từ đầu đến cuối (End-to-End Machine Learning Project)** — một chương cốt lõi mô phỏng quy trình làm việc thực tế của một Kỹ sư AI trên bộ dữ liệu bất động sản California — tôi đề xuất chia nội dung chương này thành **5 phần chi tiết**:

*   **Phần 1: Khởi động Dự án, Thước đo hiệu suất (RMSE, MAE) & Khởi tạo Tập kiểm thử (Train-Test Split)** *(Chúng ta sẽ thực hiện chi tiết phần đầu tiên này trước)*.
*   **Phần 2: Lấy mẫu phân tầng (Stratified Sampling) & Phân tích tương quan dữ liệu (Correlation Analysis)**.
*   **Phần 3: Tiền xử lý dữ liệu - Làm sạch dữ liệu khuyết & Mã hóa đặc trưng phân loại**.
*   **Phần 4: Thiết kế Đặc trưng (Feature Engineering), RBF Kernel & Chuỗi đường ống ColumnTransformer**.
*   **Phần 5: Huấn luyện mô hình (Linear, Decision Tree, Random Forest) & Tinh chỉnh siêu tham số nâng cao**.

Dưới đây là chi tiết **Phần 1** được biên soạn chuyên sâu theo cấu trúc chuẩn hóa, kết hợp toán học, sơ đồ trực quan và mã nguồn Python thực chiến.

---

# PHẦN 1: KHỞI ĐỘNG DỰ ÁN, THƯỚC ĐO HIỆU SUẤT & KHỞI TẠO TẬP KIỂM THỬ

### 1. Hệ thống ống dẫn dữ liệu (Data Pipeline)

*   **Giải thích bản chất:**
    Trong môi trường sản xuất thực tế, một hệ thống Học máy không hoạt động đơn độc. Dữ liệu thường được xử lý thông qua một **Hệ thống ống dẫn (Data Pipeline)**. Đó là một chuỗi các thành phần xử lý dữ liệu chạy độc lập, bất đồng bộ. Đầu ra của một thành phần này sẽ là đầu vào (tín hiệu) cung cấp cho một thành phần tiếp theo trong chuỗi. 
    
    Cơ chế thiết kế bất đồng bộ này giúp hệ thống cực kỳ linh hoạt: nếu một thành phần bị lỗi, các thành phần phía sau vẫn có thể tạm thời hoạt động bằng dữ liệu cũ hoặc hệ thống dự phòng, tránh việc làm sập toàn bộ chuỗi vận hành của doanh nghiệp.
*   **Ví dụ thực tế trong tài liệu:**
    Nhiệm vụ của chúng ta là xây dựng mô hình Học máy để dự đoán giá nhà trung bình của một quận ở California dựa trên dữ liệu điều tra dân số.
*   **Giải thích trực quan dựa trên sơ đồ hệ thống (Hình 2-2):**
    
    \\[\text{Hình 2-2: Sơ đồ hệ thống ống dẫn dữ liệu trong bài toán định giá bất động sản}\\]
    \\[\text{[District data]} \rightarrow \mathbf{\text{[District pricing (Thành phần của chúng ta)]}} \rightarrow \text{[District prices] + [Other signals]} \rightarrow \text{[Investment analysis]} \rightarrow \text{[Investments]} \text{}\\]
    
    Dựa trên sơ đồ **Hình 2-2** trong tài liệu:
    *   **Thành phần trung tâm (District pricing):** Đây chính là mô hình Học máy dự đoán giá nhà trung bình mà chúng ta sẽ xây dựng. Nó nhận đầu vào là các đặc trưng địa lý và dân số (**District data**).
    *   **Thành phần hạ nguồn (Investment analysis):** Nhận giá nhà dự đoán từ mô hình của chúng ta (**District prices**), kết hợp với các luồng thông tin khác (**Other signals**) để đưa ra quyết định đầu tư cuối cùng (**Investments**). Sự chính xác của mô hình định giá sẽ trực tiếp quyết định đến doanh thu và lợi nhuận của toàn bộ hệ thống đầu tư này.

---

### 2. Thước đo hiệu suất: Sai số toàn phương trung bình (RMSE)

*   **Giải thích bản chất:**
    **RMSE (Root Mean Squared Error)** là thước đo tiêu chuẩn và được ưa chuộng nhất để đánh giá hiệu năng của các bài toán hồi quy (đặc biệt là hồi quy đa biến đơn trị). RMSE đo lường khoảng cách trung bình giữa các giá trị dự đoán của mô hình và các giá trị thực tế. 
    
    Vì sai số giữa dự đoán và thực tế được bình phương trước khi lấy trung bình, **RMSE sẽ phạt rất nặng các lỗi dự đoán lớn (gán trọng số lớn hơn cho các sai số cực đoan)**. Do đó, RMSE phản ánh chân thực mức độ lỗi mà hệ thống thường mắc phải trong các kịch bản thực tế.
*   **Công thức Toán học (Công thức 2-1):**
    
    \\[RMSE(\mathbf{X}, h) = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( h(x^{(i)}) - y^{(i)} \right)^2} \quad \text{}\\]
    
    *Giải thích ký hiệu dựa trên tài liệu (Hình &):*
    *   \\(m\\): Số lượng mẫu dữ liệu (ví dụ: \\(m = 2.000\\) quận trong tập xác thực).
    *   \\(x^{(i)}\\): Vector chứa tất cả các giá trị đặc trưng (không bao gồm nhãn) của mẫu thứ \\(i\\). Ví dụ (Hình):
        
        \\[x^{(1)} = \begin{pmatrix} -118.29 \\ 33.91 \\ 1.416 \\ 38.372 \end{pmatrix} \quad (\text{Kinh độ, vĩ độ, dân số, thu nhập trung bình của quận đầu tiên}) \quad \text{}\\]
        
    *   \\(y^{(i)}\\): Nhãn thực tế (giá trị đầu ra mong muốn) của mẫu thứ \\(i\\) (ví dụ: \\(y^{(1)} = 156.400\\) USD).
    *   \\(\mathbf{X}\\): Ma trận chứa toàn bộ đặc trưng của mọi mẫu dữ liệu (Hình). Mỗi hàng trong ma trận \\(\mathbf{X}\\) tương ứng với ma trận chuyển vị của một vector đặc trưng \\((x^{(i)})^T\\):
        
        \\[\mathbf{X} = \begin{pmatrix} (x^{(1)})^T \\ (x^{(2)})^T \\ \vdots \\ (x^{(m)})^T \end{pmatrix} = \begin{pmatrix} -118.29 & 33.91 & 1.416 & 38.372 \\ \vdots & \vdots & \vdots & \vdots \end{pmatrix} \quad \text{}\\]
        
    *   \\(h\\): Hàm giả thuyết (hypothesis) dự đoán của hệ thống. Khi nhận vào vector \\(x^{(1)}\\), mô hình trả về giá trị dự đoán là \\(\hat{y}^{(1)} = h(x^{(1)})\\). Sai số dự đoán cho mẫu này là \\(h(x^{(1)}) - y^{(1)}\\).

*   **Mã nguồn Python minh họa:**
    ```python
    import numpy as np
    from sklearn.metrics import mean_squared_error

    # Giả sử y_true là nhãn thực tế, y_pred là dự đoán của mô hình
    y_true = np.array([458300., 483800., 101700., 96100., 361800.])
    y_pred = np.array([243700., 372400., 128800., 94400., 328300.])

    # CẢNH BÁO TRONG TÀI LIỆU:
    # Trong các phiên bản Scikit-Learn mới, bạn nên sử dụng root_mean_squared_error()
    # thay vì mean_squared_error(..., squared=False) để tránh các cảnh báo lỗi thời.
    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:
        # Định nghĩa hàm dự phòng nếu thư viện Scikit-Learn của bạn ở phiên bản cũ
        def root_mean_squared_error(labels, predictions):
            return mean_squared_error(labels, predictions, squared=False)

    rmse = root_mean_squared_error(y_true, y_pred)
    print(f"Sai số RMSE của mô hình: {rmse:.2f} USD")
    ```

---

### 3. Thước đo hiệu suất: Sai số tuyệt đối trung bình (MAE)

*   **Giải thích bản chất:**
    **MAE (Mean Absolute Error)** là một thước đo khoảng cách khác giữa vector dự đoán và vector mục tiêu thực tế. Khác với RMSE, MAE chỉ tính trung bình cộng của các giá trị sai số tuyệt đối mà không bình phương chúng lên. 
    
    Chính vì đặc tính này, **MAE hoạt động ổn định và ít nhạy cảm với các giá trị ngoại lai (outliers) hơn RMSE**. Nếu trong tập dữ liệu của bạn xuất hiện nhiều khu vực quận có giá nhà dị biệt cực đoan (nhiễu dữ liệu hoặc lỗi hệ thống), MAE sẽ giúp phản ánh sai số trung bình một cách khách quan nhất.
*   **Công thức Toán học (Công thức 2-2):**
    
    \\[MAE(\mathbf{X}, h) = \frac{1}{m} \sum_{i=1}^{m} |h(x^{(i)}) - y^{(i)}| \quad \text{}\\]
*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.metrics import mean_absolute_error

    mae = mean_absolute_error(y_true, y_pred)
    print(f"Sai số MAE của mô hình: {mae:.2f} USD")
    ```

---

### 4. Khái niệm chuẩn khoảng cách (Norms): Chuẩn \\(l_1\\) vs Chuẩn \\(l_2\\)

*   **Giải thích bản chất:**
    Trong Học máy, các thước đo khoảng cách như RMSE và MAE được chuẩn hóa dưới dạng các **Chuẩn toán học (Norms)** của vector sai số \\(v\\):
    *   **Chuẩn Euclid (Chuẩn \\(l_2\\), ký hiệu \\(\|\cdot\|_2\\)):** Tương ứng với phép tính RMSE. Đây là khái niệm khoảng cách đường chim bay hình học phẳng thông thường mà chúng ta vẫn sử dụng.
    *   **Chuẩn Manhattan (Chuẩn \\(l_1\\), ký hiệu \\(\|\cdot\|_1\\)):** Tương ứng với phép tính MAE. Nó đo khoảng cách giữa hai điểm trong một mạng lưới ô bàn cờ (giống như việc bạn chỉ có thể di chuyển dọc theo các con phố vuông góc ở quận Manhattan).
    *   **Chuẩn tổng quát \\(l_k\\) (Chuẩn Minkowski):**
        
        \\[\|v\|_k = \left( |v_1|^k + |v_2|^k + \dots + |v_n|^k \right)^{1/k} \quad \text{}\\]
        
    *   **Quy luật bất biến (Nhận xét quan trọng):**
        **Chỉ số chuẩn \\(k\\) càng cao, nó càng tập trung vào các giá trị lớn và bỏ qua các giá trị nhỏ**. Đây chính là lý do RMSE (\\(l_2\\)) cực kỳ nhạy cảm với các giá trị ngoại lai so với MAE (\\(l_1\\)). Tuy nhiên, trong các bài toán mà sai số tuân theo phân phối chuẩn hình chuông (Gaussian), RMSE hoạt động vô cùng hiệu quả và luôn là sự lựa chọn ưu tiên hàng đầu.

---

### 5. Sai lệch do rò rỉ dữ liệu (Data Snooping Bias)

*   **Giải thích bản chất:**
    Bộ não của con người là một hệ thống phát hiện mẫu cực kỳ xuất sắc nhưng cũng rất dễ bị đánh lừa. Nếu bạn tò mò khám phá tập kiểm thử (test set) quá sớm trước khi chọn mô hình, bạn sẽ vô thức phát hiện ra các mẫu hoặc cấu trúc đặc thù của tập kiểm thử đó. 
    
    Điều này dẫn dắt bạn đến việc thiết kế đặc trưng hoặc lựa chọn một loại mô hình Học máy thiên vị và quá khớp (overfit) với tập kiểm thử. Khi bạn đo lường sai số trên tập kiểm thử này, kết quả sẽ cực kỳ tốt (quá lạc quan), nhưng hệ thống sẽ **thất bại thảm hại khi đối mặt với dữ liệu thực tế**. Hiện tượng này được gọi là **Sai lệch do rò rỉ dữ liệu (Data Snooping Bias)**.
*   **Giải pháp xử lý:**
    Luôn chủ động gạt riêng **20% dữ liệu** làm tập kiểm thử ngay từ đầu và tuyệt đối không phân tích, trực quan hóa hay can thiệp vào nó cho đến khi mô hình đã được tinh chỉnh hoàn chỉnh.

---

### 6. Phân tách tập dữ liệu ổn định bằng Băm định danh (Identifier Hashing)

*   **Giải thích bản chất:**
    Nếu chúng ta phân chia tập huấn luyện và tập kiểm thử bằng cách xáo trộn ngẫu nhiên truyền thống, mỗi khi chạy lại chương trình, một tập kiểm thử mới sẽ được tạo ra. Qua nhiều lần chạy thử nghiệm, mô hình của bạn (hoặc bản thân bạn) sẽ vô tình được "nhìn thấy" toàn bộ dữ liệu, vi phạm nguyên tắc bảo mật tập kiểm thử. Sử dụng hạt giống ngẫu nhiên `np.random.seed(42)` có thể giải quyết được việc này, nhưng sẽ **bị hỏng ngay lập tức nếu bạn cập nhật thêm dữ liệu mới**.
    
    Để phân tách dữ liệu một cách ổn định lâu dài (ngay cả khi liên tục cập nhật thêm mẫu mới), phương pháp tối ưu là sử dụng **Băm định danh (Identifier Hashing)**:
    *   Tính toán giá trị băm (hash) của cột ID duy nhất của từng mẫu dữ liệu.
    *   Đưa mẫu dữ liệu đó vào tập kiểm thử nếu giá trị băm nhỏ hơn hoặc bằng 20% giá trị băm tối đa (ví dụ: băm CRC32 trả về số nguyên 32-bit không âm, ta so sánh với ngưỡng \\(0.2 \times 2^{32}\\)).
    *   *Kết quả:* Tập kiểm thử sẽ luôn nhất quán qua nhiều lần chạy. Dữ liệu mới được thêm vào sẽ được phân chia chính xác mà không bao giờ bị rò rỉ hay thay đổi các phần tử cũ trong tập huấn luyện.

*   **Ví dụ thực tế trong tài liệu:**
    Tập dữ liệu California ban đầu không có cột ID duy nhất. Tài liệu hướng dẫn hai cách giải quyết:
    1.  Dùng chỉ mục hàng (`index`) làm định danh duy nhất (Yêu cầu nghiêm ngặt: dữ liệu mới chỉ được thêm vào cuối và không được xóa hàng cũ).
    2.  Tự thiết lập một ID kết hợp từ các đặc trưng vật lý địa lý cực kỳ ổn định qua hàng triệu năm (vĩ độ và kinh độ của quận):
        
        \\[\text{ID} = \text{longitude} \times 1000 + \text{latitude} \quad \text{}\\]

*   **Mã nguồn Python minh họa:**
    ```python
    from zlib import crc32
    import numpy as np
    import pandas as pd

    # 1. Định nghĩa hàm kiểm tra băm định danh ổn định (Hình )
    def is_id_in_test_set(identifier, test_ratio):
        # np.int64 giúp ép kiểu dữ liệu đồng nhất
        return crc32(np.int64(identifier)) < test_ratio * 2**32 [cite: 116]

    # 2. Hàm phân tách dữ liệu dựa trên cột ID băm (Hình )
    def split_data_with_id_hash(data, test_ratio, id_column):
        ids = data[id_column] [cite: 116]
        in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio)) [cite: 116]
        return data.loc[~in_test_set], data.loc[in_test_set] [cite: 116]

    # Giả sử tạo một DataFrame mô phỏng dữ liệu California
    np.random.seed(42)
    df_mock = pd.DataFrame({
        "longitude": np.random.uniform(-124, -114, 100),
        "latitude": np.random.uniform(32, 42, 100),
        "median_house_value": np.random.randint(100000, 500000, 100)
    })

    # Tạo cột ID kết hợp ổn định địa chất
    df_mock["id"] = df_mock["longitude"] * 1000 + df_mock["latitude"] [cite: 117]

    # Thực hiện phân tách tập huấn luyện và tập kiểm thử ổn định
    train_set, test_set = split_data_with_id_hash(df_mock, 0.2, "id")
    print(f"Kích thước tập huấn luyện: {len(train_set)} dòng")
    print(f"Kích thước tập kiểm thử: {len(test_set)} dòng")
    ```

---

# PHẦN 2: LẤY MẪU PHÂN TẦNG & TRỰC QUAN HÓA DỮ LIỆU ĐỊA LÝ

---

### 1. Sai lệch lấy mẫu & Lấy mẫu phân tầng (Sampling Bias & Stratified Sampling)

*   **Giải thích bản chất:** 
    Khi phân chia dữ liệu thành tập huấn luyện và tập kiểm thử, phương pháp lấy mẫu hoàn toàn ngẫu nhiên (Random Sampling) thường hoạt động tốt trên các tập dữ liệu lớn. Tuy nhiên, nếu tập dữ liệu nhỏ hoặc một thuộc tính quan trọng bị phân bổ không đều, việc lấy mẫu ngẫu nhiên sẽ có nguy cơ cao tạo ra **Sai lệch lấy mẫu (Sampling Bias)**. Điều này nghĩa là tập kiểm thử được chọn ra không đại diện đúng cho đặc tính của toàn bộ quần thể dữ liệu.
    
    Để giải quyết vấn đề này, kỹ thuật **Lấy mẫu phân tầng (Stratified Sampling)** được áp dụng. Bản chất của phương pháp này là chia toàn bộ quần thể dữ liệu thành các nhóm con đồng nhất gọi là **Tầng (Strata)**. Sau đó, một số lượng mẫu chính xác được rút ra từ mỗi tầng để đảm bảo tỷ lệ đại diện của từng tầng trong tập kiểm thử hoàn toàn tương đồng với tỷ lệ đại diện của tầng đó trong toàn bộ quần thể.
*   **Ví dụ thực tế trong tài liệu:** 
    Tài liệu đưa ra một ví dụ đời thường về khảo sát dân số. Dân số Hoa Kỳ gồm **51.1% nữ** và **48.9% nam**. Một cuộc khảo sát chất lượng cao gồm 1.000 người sẽ không chọn ngẫu nhiên hoàn toàn mà chủ động gom đúng **511 nữ** và **489 nam** để duy trì tỷ lệ vàng này. Nếu lấy mẫu ngẫu nhiên hoàn toàn, có tới **10.7% khả năng** mẫu thu được bị lệch nghiêm trọng (dưới 48.5% hoặc trên 53.5% nữ), dẫn đến kết quả khảo sát bị sai lệch nặng nề. 
    
    Trong dự án California, các chuyên gia nhận định rằng **Thu nhập trung vị (Median Income)** là đặc trưng quan trọng bậc nhất để dự báo giá nhà. Do đó, chúng ta cần lấy mẫu phân tầng dựa trên đặc trưng này để tập kiểm thử phản ánh chính xác mọi phân khúc thu nhập của người dân California.

---

### 2. Phân tầng đặc trưng liên tục bằng `pd.cut` (Continuous Feature Stratification)

*   **Giải thích bản chất:** 
    Đặc trưng thu nhập trung vị (`median_income`) ban đầu là một biến số liên tục. Để có thể thực hiện lấy mẫu phân tầng, chúng ta bắt buộc phải chuyển đổi nó thành một **thuộc tính phân loại (categorical attribute)** để định hình các tầng.
    
    *Quy tắc thiết kế tầng trong Học máy:*
    1.  Không nên chia quá nhiều tầng để tránh việc mỗi tầng có quá ít dữ liệu.
    2.  Mỗi tầng phải có kích thước đủ lớn để ước tính thống kê đạt độ tin cậy cao và không bị sai lệch.
    
    Công cụ đắc lực nhất để làm việc này là hàm **`pd.cut()`** của thư viện Pandas, cho phép chia nhỏ các khoảng giá trị liên tục thành các nhóm danh mục rời rạc có giới hạn rõ ràng.
*   **Ví dụ thực tế trong tài liệu:**
    Dựa trên biểu đồ tần suất thu nhập, hầu hết các giá trị tập trung từ 1.5 đến 6 (tương đương \$15.000 – \$60.000), nhưng có một số ít trường hợp cực cao kéo dài sang bên phải. Tài liệu thực hiện phân chia thuộc tính `median_income` thành 5 nhóm danh mục mới (được dán nhãn từ 1 đến 5):
    *   **Nhóm 1:** Thu nhập từ `0` đến `1.5` (Dưới \$15.000).
    *   **Nhóm 2:** Thu nhập từ `1.5` đến `3.0`.
    *   **Nhóm 3:** Thu nhập từ `3.0` đến `4.5`.
    *   **Nhóm 4:** Thu nhập từ `4.5` đến `6.0`.
    *   **Nhóm 5:** Thu nhập từ `6.0` đến vô cùng (`np.inf`).

*   **Giải thích trực quan dựa trên hình ảnh (Hình 2-9):**
    
    \\[\text{Hình 2-9: Biểu đồ phân bổ số lượng quận theo phân nhóm thu nhập (Income Category)}\\]
    
    Dựa trên **Hình 2-9** trong tài liệu:
    *   **Trục hoành (X-axis):** Biểu thị 5 phân nhóm thu nhập (`Income category`) rời rạc.
    *   **Trục tung (Y-axis):** Biểu thị số lượng quận (`Number of districts`) thuộc về từng nhóm.
    *   **Phân tích phân phối:** Biểu đồ dạng cột cột cho thấy rõ nhóm thu nhập 3 (từ 3.0 đến 4.5) và nhóm thu nhập 2 là chiếm số lượng áp đảo (lần lượt hơn 7.000 và 6.000 quận). Trong khi đó nhóm thu nhập cực thấp (Nhóm 1) chỉ chiếm một phần rất nhỏ dưới 1.000 quận. Biểu đồ này giải thích trực quan tại sao ta cần lấy mẫu phân tầng: nếu dùng ngẫu nhiên, ta rất dễ vô tình rút ra một tập kiểm thử hoàn toàn thiếu vắng các mẫu của Nhóm 1 hoặc Nhóm 5.

*   **Mã nguồn Python minh họa:**
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    # 1. Nạp dữ liệu giả lập (hoặc thực tế)
    # housing = load_housing_data()

    # 2. Phân nhóm thu nhập thành biến danh mục bằng pd.cut (Hình 2-9)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=) [cite: 139]

    # 3. Vẽ biểu đồ cột để kiểm tra trực quan phân bổ (Hình 2-9)
    housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True) [cite: 139]
    plt.xlabel("Income category") [cite: 139]
    plt.ylabel("Number of districts") [cite: 139]
    plt.show() [cite: 139]
    ```

---

### 3. Đánh giá sai số phân tầng (Stratified vs. Random Sampling Error)

*   **Giải thích bản chất:**
    Để chứng minh tính hiệu quả vượt trội của kỹ thuật Lấy mẫu phân tầng so với Lấy mẫu ngẫu nhiên truyền thống, tài liệu so sánh **Tỷ lệ phần trăm đại diện** của từng nhóm thu nhập trên 3 tập dữ liệu: Toàn bộ dữ liệu gốc (`Overall`), tập kiểm thử phân tầng (`Stratified`) và tập kiểm thử ngẫu nhiên (`Random`). 
    
    Sau đó, chúng ta tính toán sai số tỷ lệ (sampling error) bằng công thức:
    
    \\[\text{Sai số \%} = \left( \frac{\text{Tỷ lệ mẫu kiểm thử}}{\text{Tỷ lệ gốc của quần thể}} - 1 \right) \times 100\\]
    
*   **Giải thích trực quan dựa trên hình ảnh (Hình 2-10):**
    
    \\[\text{Hình 2-10: Bảng đối chiếu sai số phân tách giữa Lấy mẫu phân tầng và Lấy mẫu ngẫu nhiên}\\]
    
    Dựa trên bảng số liệu thực tế **Hình 2-10** trong tài liệu:
    *   **Cột tỷ lệ Stratified % so với Overall %:** Tỷ lệ đại diện của 5 nhóm thu nhập trong tập kiểm thử phân tầng gần như trùng khớp hoàn hảo với tập dữ liệu gốc (ví dụ: nhóm 3 chiếm 35.05% trong tập phân tầng so với 35.06% của gốc). **Sai số phân tầng (Strat. Error %)** cực kỳ thấp, chỉ dao động quanh mức không đáng kể từ **-0.01% đến 0.36%**.
    *   **Cột tỷ lệ Random % so với Overall %:** Tập kiểm thử ngẫu nhiên bị lệch phân bổ rõ rệt (ví dụ: nhóm 1 tăng vọt lên 4.24% so với mức 3.98% gốc). **Sai số ngẫu nhiên (Rand. Error %)** rất lớn, dao động mạnh từ **-3.59% đến 6.45%**.
    *   *Kết luận:* Lấy mẫu phân tầng loại bỏ gần như hoàn toàn sai lệch phân tách, bảo vệ mô hình khỏi hiện tượng quá khớp thiên vị.

*   **Mã nguồn Python minh họa:**
    ```python
    # 1. Phân chia tập kiểm thử theo 2 phương pháp khác nhau
    # Cách A: Lấy mẫu ngẫu nhiên hoàn toàn
    train_rand, test_rand = train_test_split(housing, test_size=0.2, random_state=42) [cite: 10]

    # Cách B: Lấy mẫu phân tầng dựa trên cột income_cat
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42) [cite: 141]

    # 2. Hàm tính toán tỷ lệ đại diện của từng nhóm
    def income_cat_proportions(data):
        return data["income_cat"].value_counts() / len(data) [cite: 9]

    # 3. Tạo bảng so sánh đối chiếu sai số (Tái lập Hình 2-10)
    compare_props = pd.DataFrame({
        "Overall %": income_cat_proportions(housing),
        "Stratified %": income_cat_proportions(strat_test_set),
        "Random %": income_cat_proportions(test_rand),
    }).sort_index() [cite: 10]

    compare_props["Strat. Error %"] = (compare_props["Stratified %"] / compare_props["Overall %"] - 1) [cite: 10]
    compare_props["Rand. Error %"] = (compare_props["Random %"] / compare_props["Overall %"] - 1) [cite: 10]

    print((compare_props * 100).round(2)) [cite: 10]

    # CẢNH BÁO QUAN TRỌNG TRONG TÀI LIỆU:
    # Sau khi phân tách xong, phải xóa thuộc tính phụ "income_cat" khỏi các tập dữ liệu
    # để đưa dữ liệu trở lại trạng thái gốc ban đầu cho quá trình huấn luyện!
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True) [cite: 11]
    ```

---

### 4. Bản đồ phân tán địa lý (Geographical Density Scatter Plot)

*   **Giải thích bản chất:** 
    Khi làm việc với dữ liệu có chứa tọa độ địa lý (kinh độ - `longitude` và vĩ độ - `latitude`), việc đầu tiên cần làm là vẽ biểu đồ phân tán (Scatter Plot) để trực quan hóa dữ liệu trên không gian phẳng. 
    
    Tuy nhiên, nếu chỉ vẽ các điểm thô, các khu vực đông dân cư sẽ có hàng ngàn điểm dữ liệu đè chặt lên nhau, tạo thành những khối màu đặc đặc nghẹt khiến ta không thể nhận diện được mật độ phân bổ thực tế. Bằng cách thiết lập tham số **bán trong suốt `alpha`**, chúng ta cho phép ánh sáng xuyên qua các điểm dữ liệu. Những khu vực có các điểm đè lên nhau càng dày đặc sẽ tự động có màu đậm hơn, giúp các "điểm nóng" dân cư nổi bật lên một cách tự nhiên.
*   **Ví dụ thực tế trong tài liệu:**
    *   **Trực quan hóa thô (Hình 2-11):** Vẽ trực tiếp tọa độ vĩ độ và kinh độ của tất cả các quận. Mặc dù tạo ra hình dáng dải đất bang California, nhưng nó quá dày đặc và khó thấy bất kỳ mẫu cụ thể nào.
    *   **Trực quan hóa mật độ (Hình 2-12):** Thêm tham số `alpha=0.2` vào biểu đồ.

*   **Giải thích trực quan dựa trên hình ảnh:**
    *   **Hình 2-12 (Mật độ cao nổi bật):**
        
        \\[\text{Hình 2-12: Bản đồ mật độ dân cư California với alpha = 0.2}\\]
        
        Nhờ điều chỉnh `alpha=0.2`, các cụm dữ liệu phân bổ địa lý hiển thị cực kỳ rõ nét:
        *   **Vùng vịnh San Francisco (Phía Tây trung tâm):** Một quầng đen đậm rực rỡ thể hiện mật độ quận cực kỳ cao.
        *   **Khu vực Los Angeles và San Diego (Phía Nam):** Khối mật độ rậm rạp nhất toàn bang California.
        *   **Thung lũng Trung tâm (Central Valley):** Tạo thành một dải dài chạy dọc từ Bắc xuống Nam, nổi bật nhất là các đô thị vệ tinh xung quanh Sacramento và Fresno.

*   **Mã nguồn Python minh họa:**
    ```python
    # Bản sao sạch của tập huấn luyện để khám phá
    housing = strat_train_set.copy() [cite: 11]

    # 1. Vẽ bản đồ phân tán thô (Hình 2-11)
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True) [cite: 143]
    plt.title("Biểu đồ phân tán địa lý thô")
    plt.show() [cite: 143]

    # 2. Vẽ bản đồ mật độ bán trong suốt (Hình 2-12)
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2) [cite: 143]
    plt.title("Biểu đồ mật độ địa lý (alpha = 0.2)")
    plt.show() [cite: 143]
    ```

---

### 5. Trực quan hóa địa lý đa biến (Multi-dimensional Geographical Plot)

*   **Giải thích bản chất:**
    Một trong những kỹ thuật trực quan hóa đỉnh cao là tích hợp nhiều chiều thông tin (kinh độ, vĩ độ, dân số, giá nhà) vào một biểu đồ phẳng hai chiều duy nhất. Để làm được điều này, chúng ta sử dụng hai tham số điều hướng mạnh mẽ của Matplotlib:
    *   **Kích thước điểm (`s` - Size):** Đại diện cho một biến số (ví dụ: dân số). Vòng tròn càng lớn biểu thị khu vực đó càng đông dân cư.
    *   **Màu sắc điểm (`c` - Color):** Đại diện cho một biến số khác (ví dụ: giá trị nhà trung vị). Màu sắc được ánh xạ qua một bảng màu cầu vồng **`cmap` (Colormap)**.
*   **Ví dụ thực tế trong tài liệu:**
    Tài liệu thiết lập kích thước vòng tròn bằng `population / 100`. Ánh xạ màu sắc `c` cho thuộc tính giá nhà trung vị `median_house_value` sử dụng bản đồ màu chuẩn **`jet`** (chuyển đổi dải màu từ xanh dương đại diện cho giá nhà rẻ nhất sang màu đỏ rực đại diện cho phân khúc đắt đỏ nhất).

*   **Giải thích trực quan dựa trên hình ảnh (Hình 2-13 & Bản đồ California lồng nền):**
    
    \\[\text{Hình 2-13: Bản đồ giá nhà California tích hợp đa biến (Dân số & Giá nhà)}\\ [cite: 146]\\]
    
    Từ biểu đồ tuyệt đẹp **Hình 2-13** (và bản đồ có lồng ảnh vật lý bang California **Hình 2-13/Bản đồ**), chúng ta rút ra những hiểu biết sâu sắc mang tính quyết định cho mô hình:
    1.  **Quy luật vùng ven biển (Coastal rule):** Các đốm màu đỏ rực (giá nhà cực cao tiệm cận mức giới hạn trên \$500.000) tập trung kéo dài dọc theo bờ biển mỏng phía Tây từ Vùng Vịnh xuống Los Angeles. Càng đi sâu vào đất liền (Inland) về phía Đông, màu sắc nhanh chóng chuyển sang màu xanh dương (giá nhà rất rẻ).
    2.  **Mối quan hệ Dân số & Giá trị kinh tế:** Các khu vực có vòng tròn lớn nhất (mật độ dân số cực kỳ đông đúc) cũng chính là những nơi có màu đỏ đậm nhất. Điều này chỉ ra rằng giá nhà có sự tương quan tuyến tính chặt chẽ với mật độ dân cư và vị trí địa lý.
    3.  *Gợi ý xây dựng đặc trưng:* Quy luật địa lý ven biển không đơn thuần là một đường thẳng dốc (ở Bắc California giá nhà ven biển vẫn có những chỗ rẻ). Do đó, chúng ta nên thiết kế thêm đặc trưng **Tính toán khoảng cách đến các tâm cụm đô thị lớn** bằng các thuật toán phân cụm (như K-Means) kết hợp với hàm khoảng cách tương đồng RBF.

*   **Mã nguồn Python minh họa:**
    ```python
    # Vẽ biểu đồ đa biến địa lý trực quan hóa giá nhà và dân số (Hình 2-13)
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                 s=housing["population"] / 100, label="population", # Kích thước vòng tròn tỉ lệ với dân số
                 c="median_house_value", cmap="jet", colorbar=True, # Màu sắc đại diện cho giá nhà (jet cmap)
                 legend=True, sharex=False, figsize=(10, 7)) [cite: 145]
    plt.title("Giá nhà California: Màu đỏ là đắt, Màu xanh là rẻ")
    plt.show() [cite: 145]
    ```

---

# PHẦN 3: TIỀN XỬ LÝ DỮ LIỆU – LÀM SẠCH DỮ LIỆU KHUYẾT & MÃ HÓA ĐẶC TRƯNG PHÂN LOẠI

---

### 1. Làm sạch dữ liệu khuyết (Data Cleaning - Missing Values)

*   **Giải thích bản chất:** 
    Trong các dự án dữ liệu thực tế, việc thiếu hụt giá trị (khuyết thiếu dữ liệu) xảy ra rất phổ biến. Hầu hết các thuật toán Học máy không thể hoạt động hiệu quả nếu dữ liệu đầu vào chứa các giá trị trống (`NaN` - Not a Number). Chúng ta có **3 lựa chọn cơ bản** để xử lý các giá trị khuyết này:
    *   **Lựa chọn 1 (dropna):** Loại bỏ hoàn toàn các hàng dữ liệu chứa giá trị khuyết. Phương pháp này hữu hiệu khi lượng hàng bị khuyết rất nhỏ và không làm mất đi tính đại diện của tập dữ liệu.
    *   **Lựa chọn 2 (drop):** Loại bỏ toàn bộ thuộc tính (cột) chứa giá trị khuyết. Áp dụng khi thuộc tính đó chứa quá nhiều dữ liệu khuyết và không đóng vai trò quan trọng trong việc dự đoán.
    *   **Lựa chọn 3 (fillna):** Điền các giá trị khuyết bằng một giá trị thay thế cụ thể (như số 0, giá trị trung bình, hoặc giá trị trung vị). Đây được gọi là kỹ thuật **Điền khuyết (Imputation)**.

*   **Ví dụ thực tế trong tài liệu:** 
    Trong tập huấn luyện dự án California, thuộc tính **`total_bedrooms`** chỉ có **20.433 dòng** không rỗng trên tổng số **20.640 dòng** (thiếu mất 207 giá trị). Tài liệu phân tích chi tiết kết quả của cả 3 lựa chọn làm sạch này trên tập dữ liệu.

*   **Giải thích trực quan dựa trên hình ảnh (`multimodal_41`):**
    
    \\[\text{Hình ảnh trong mã nguồn: Minh họa ba lựa chọn làm sạch dữ liệu của thuộc tính total\_bedrooms}\\]
    
    Dựa trên kết quả in ra của các dòng dữ liệu bị khuyết ban đầu (các chỉ mục dòng 14452, 18217, 11889, 20325, 14360) tại ảnh **`multimodal_41`**:
    *   **Bảng dữ liệu gốc (Bảng đầu tiên):** Cho thấy cột `total_bedrooms` của các dòng này đều mang giá trị rỗng `NaN`.
    *   **Kết quả Lựa chọn 1 (Bảng thứ hai):** Sau khi chạy lệnh `dropna(subset=["total_bedrooms"])`, các dòng bị khuyết (như dòng 14452, 18217...) bị **xóa hoàn toàn** khỏi DataFrame, bảng kết quả rỗng không hiển thị dòng nào.
    *   **Kết quả Lựa chọn 2 (Bảng thứ ba):** Sau khi chạy lệnh `drop("total_bedrooms", axis=1)`, cột thuộc tính `total_bedrooms` đã **biến mất hoàn toàn** khỏi DataFrame của chúng ta.
    *   **Kết quả Lựa chọn 3 (Bảng thứ tư):** Sau khi chạy lệnh `fillna(median)`, các ô rỗng `NaN` trước đó đã được thay thế đồng loạt bằng giá trị trung vị **`434.0`**. Đây là phương án ít phá hủy cấu trúc dữ liệu nhất và được ưu tiên lựa chọn.

*   **Mã nguồn Python minh họa:**
    ```python
    import pandas as pd

    # Giả sử chúng ta có tập dữ liệu "housing" bị khuyết
    # null_rows_idx lọc ra các dòng chứa giá trị khuyết để theo dõi
    null_rows_idx = housing.isnull().any(axis=1)

    # Lựa chọn 1: Loại bỏ các hàng bị khuyết thuộc tính total_bedrooms
    housing_option1 = housing.copy()
    housing_option1.dropna(subset=["total_bedrooms"], inplace=True)

    # Lựa chọn 2: Loại bỏ hoàn toàn cột total_bedrooms
    housing_option2 = housing.copy()
    housing_option2.drop("total_bedrooms", axis=1, inplace=True)

    # Lựa chọn 3: Điền khuyết bằng giá trị trung vị (Median) của cột
    housing_option3 = housing.copy()
    median = housing["total_bedrooms"].median()
    housing_option3["total_bedrooms"].fillna(median, inplace=True)
    ```

---

### 2. Bộ điền khuyết tự động `SimpleImputer`

*   **Giải thích bản chất:**
    Mặc dù có thể điền khuyết thủ công bằng Pandas, nhưng sử dụng lớp **`SimpleImputer`** của Scikit-Learn đem lại lợi thế vượt trội trong lập trình Học máy. `SimpleImputer` hoạt động như một **Estimator** và **Transformer**:
    1.  Nó học các tham số thống kê (như giá trị trung vị) từ tập huấn luyện thông qua phương thức `.fit()` và lưu vào thuộc tính `statistics_`.
    2.  Nó áp dụng giá trị đã học này để điền khuyết nhất quán cho tập huấn luyện, tập kiểm thử và bất kỳ luồng dữ liệu mới nào trong tương lai thông qua phương thức `.transform()`, tránh hiện tượng rò rỉ dữ liệu.
    
    *Lưu ý:* Vì giá trị trung vị chỉ tính được trên các đặc trưng số, ta bắt buộc phải tách riêng dữ liệu số trước khi áp dụng `SimpleImputer` với chiến lược `"median"`.

*   **Ví dụ thực tế trong tài liệu:**
    Khởi tạo `SimpleImputer(strategy="median")` và áp dụng cho tập thuộc tính số `housing_num`.

*   **Giải thích trực quan dựa trên hình ảnh (`multimodal_25`):**
    
    \\[\text{Hình ảnh trong mã nguồn: Quy trình huấn luyện và áp dụng SimpleImputer}\\]
    
    Dựa trên dòng chảy thực thi tại ảnh **`multimodal_25`**:
    *   **Fit mô hình:** Lệnh `imputer.fit(housing_num)` đã tính toán ra mảng giá trị trung vị lưu trong `imputer.statistics_`. Giá trị trung vị của cột `total_bedrooms` là `434.0`.
    *   **Transform dữ liệu:** Lệnh `imputer.transform(housing_num)` trả về một mảng NumPy thô không chứa tên cột.
    *   **Khôi phục DataFrame:** Để dễ dàng phân tích, mảng này được bọc lại thành DataFrame `housing_tr`. Khi hiển thị lại các dòng bị khuyết ban đầu (ví dụ dòng 14452), giá trị rỗng tại cột `total_bedrooms` đã được điền khuyết chính xác và sạch sẽ bằng giá trị **`434.0`**.

*   **Mã nguồn Python minh họa:**
    ```python
    import numpy as np
    from sklearn.impute import SimpleImputer

    # 1. Khởi tạo imputer với chiến lược trung vị (median)
    imputer = SimpleImputer(strategy="median")

    # 2. Lọc ra các cột dữ liệu số (loại bỏ cột phân loại ocean_proximity)
    housing_num = housing.select_dtypes(include=[np.number])

    # 3. Huấn luyện imputer trên tập dữ liệu số
    imputer.fit(housing_num)

    # Kiểm tra mảng trung vị đã học được
    print("Giá trị trung vị học được:", imputer.statistics_)

    # 4. Biến đổi dữ liệu (điền khuyết) - kết quả trả về là mảng NumPy
    X = imputer.transform(housing_num)

    # 5. Khôi phục lại cấu trúc DataFrame đẹp mắt của Pandas
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    ```

---

### 3. Mã hóa có thứ tự (Ordinal Encoding)

*   **Giải thích bản chất:**
    Hầu hết các thuật toán Học máy chỉ làm việc hiệu quả với các con số, không thể xử lý trực tiếp văn bản. **`OrdinalEncoder`** là bộ mã hóa giúp chuyển đổi các giá trị phân loại dạng văn bản thành các giá trị số nguyên có thứ tự (0, 1, 2...).
    
    *Hạn chế cốt lõi:* `OrdinalEncoder` gán các nhãn số nguyên dựa trên thứ tự bảng chữ cái của danh mục. Thuật toán Học máy sẽ mặc định hiểu rằng các số nằm gần nhau (như 0 và 1) có tính tương đồng cao hơn các số nằm xa nhau (như 0 và 4). Điều này hoàn toàn sai lệch đối với các thuộc tính không mang tính thứ tự tự nhiên (ví dụ danh mục vị trí địa lý).

*   **Ví dụ thực tế trong tài liệu:**
    Mã hóa thuộc tính phân loại **`ocean_proximity`** (gồm các nhãn `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`).

*   **Giải thích trực quan dựa trên hình ảnh (`multimodal_26` / `multimodal_43`):**
    
    \\[\text{Hình ảnh trong mã nguồn: Mã hóa thuộc tính ocean\_proximity bằng OrdinalEncoder}\\]
    
    Dựa trên kết quả hiển thị tại ảnh **`multimodal_26`**:
    *   **Dữ liệu phân loại gốc (Bảng phía trên):** Hiển thị các chuỗi văn bản lặp đi lặp lại như `NEAR BAY`, `<1H OCEAN`, `INLAND`.
    *   **Kết quả mã hóa `housing_cat_encoded[:8]`:** Chuyển đổi thành mảng số thực 1 chiều: dòng chứa `NEAR BAY` biến thành `[3.]`, dòng chứa `INLAND` biến thành `[1.]`, dòng chứa `<1H OCEAN` biến thành `[0.]`.
    *   **Tra cứu thuộc tính danh mục (`ordinal_encoder.categories_`):** Hiển thị rõ danh sách ánh xạ nhãn số từ 0 đến 4 theo đúng thứ tự bảng chữ cái tương ứng: `['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']`. Ta nhận thấy nhãn `0` (`<1H OCEAN`) và nhãn `4` (`NEAR OCEAN`) thực chất có tính chất địa lý tương đồng rất cao, nhưng bộ mã hóa này lại đẩy khoảng cách của chúng xa nhau nhất (0 và 4), gây khó khăn cho mô hình.

*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.preprocessing import OrdinalEncoder

    # Tách riêng cột dữ liệu phân loại ocean_proximity
    housing_cat = housing[["ocean_proximity"]]

    # Khởi tạo và thực hiện mã hóa có thứ tự
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

    print("8 dòng mã hóa đầu tiên:\n", housing_cat_encoded[:8])
    print("Danh mục học được:", ordinal_encoder.categories_)
    ```

---

### 4. Mã hóa một-nóng (One-Hot Encoding)

*   **Giải thích bản chất:**
    Để khắc phục nhược điểm của mã hóa thứ tự, phương pháp **Mã hóa một-nóng (One-Hot Encoding)** được áp dụng phổ biến. Thuật toán sẽ tạo ra một thuộc tính nhị phân (chỉ nhận giá trị `0` hoặc `1`) cho mỗi danh mục có sẵn của biến phân loại. Với mỗi hàng dữ liệu, chỉ có duy nhất một thuộc tính tương ứng với danh mục thực tế của hàng đó mang giá trị `1` (trạng thái "hot"), trong khi tất cả các thuộc tính danh mục khác đều mang giá trị `0` (trạng thái "cold").

*   **Khái niệm Ma trận thưa (Sparse Matrix):**
    Khi thuộc tính phân loại có hàng trăm danh mục, mã hóa một-nóng sẽ tạo ra hàng trăm cột mới chứa toàn số `0` và chỉ có một số `1` duy nhất ở mỗi hàng. Để tối ưu hóa tài nguyên hệ thống, `OneHotEncoder` của Scikit-Learn mặc định trả về một **Ma trận thưa SciPy (Sparse Matrix)**. Ma trận này chỉ lưu trữ vị trí của các phần tử khác `0`, giúp tiết kiệm tối đa bộ nhớ RAM và tăng tốc độ tính toán. Ta có thể chuyển đổi nó về mảng dày đặc (Dense Array) thông thường bằng phương thức `.toarray()` hoặc thiết lập tham số `sparse_output=False` khi khởi tạo.

*   **Sự ưu việt của `OneHotEncoder` so với hàm `pd.get_dummies()` của Pandas:**
    Mặc dù hàm `pd.get_dummies()` của Pandas cũng hỗ trợ chuyển đổi một-nóng rất nhanh chóng, nhưng **`OneHotEncoder` mới là giải pháp chuẩn mực cho môi trường sản xuất** nhờ khả năng hoạt động như một ước lượng viên chuyên nghiệp:
    1.  **Ghi nhớ nhất quán:** `OneHotEncoder` ghi nhớ chính xác danh sách và thứ tự các cột danh mục đã học từ tập huấn luyện (qua thuộc tính `categories_` và `feature_names_in_`). Trong khi đó, `pd.get_dummies()` chỉ hoạt động cục bộ trên tập dữ liệu hiện tại, dễ dẫn đến hiện tượng lệch số lượng cột khi nạp dữ liệu kiểm thử mới.
    2.  **Xử lý an toàn nhãn lạ (`handle_unknown="ignore"`):** Nếu dữ liệu mới xuất hiện một danh mục lạ chưa từng có trong tập huấn luyện, `pd.get_dummies()` sẽ tự ý tạo thêm cột mới, trong khi `OneHotEncoder` mặc định sẽ báo lỗi cảnh báo an toàn. Nếu thiết lập `handle_unknown="ignore"`, bộ mã hóa sẽ thông minh bỏ qua lỗi này bằng cách gán toàn bộ các giá trị cột mã hóa của nhãn lạ đó bằng `0`.

*   **Giải thích trực quan dựa trên hình ảnh (`multimodal_27` / `multimodal_44`):**
    
    \\[\text{Hình ảnh trong mã nguồn: So sánh cơ chế hoạt động giữa get\_dummies và OneHotEncoder}\\]
    
    Dựa trên so sánh thực tế hiển thị tại ảnh **`multimodal_27`**:
    *   Khi nạp dữ liệu kiểm thử mới chứa danh mục lạ `df_test_unknown` (gồm 2 dòng: `"<2H OCEAN"`, `"ISLAND"`):
        *   `pd.get_dummies()` vui vẻ xuất ra bảng dữ liệu mới chỉ có đúng 2 cột: `ocean_proximity_<2H OCEAN` và `ocean_proximity_ISLAND`. Điều này sẽ trực tiếp làm lỗi sập hệ thống Học máy của bạn vì mô hình đang mong đợi nhận vào đúng 5 cột đặc trưng địa lý như khi huấn luyện.
        *   `OneHotEncoder` (với `handle_unknown="ignore"`) giải quyết hoàn hảo: Dòng chứa nhãn lạ `<2H OCEAN` được biểu diễn an toàn bằng vector toàn số 0: **`[0., 0., 0., 0., 0.]`**; dòng chứa nhãn quen thuộc `ISLAND` được mã hóa chính xác về đúng vị trí cột số 3 đã học: **`[0., 0., 1., 0., 0.]`**. Sự đồng bộ 5 cột được duy trì tuyệt đối.

*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd

    # 1. Khởi tạo OneHotEncoder trả về mảng NumPy dày đặc và bỏ qua các nhãn lạ
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # 2. Huấn luyện và biến đổi cột ocean_proximity
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print("Mảng mã hóa một-nóng:\n", housing_cat_1hot[:3])

    # 3. Sử dụng tên thuộc tính đầu ra get_feature_names_out() để khôi phục DataFrame đẹp
    feature_names = cat_encoder.get_feature_names_out()
    df_output = pd.DataFrame(housing_cat_1hot, columns=feature_names, index=housing_cat.index)
    print("\nDataFrame mã hóa một-nóng:\n", df_output.head(3))
    ```

---

# PHẦN 4: THIẾT KẾ ĐẶC TRƯNG, BIẾN ĐỔI PHÂN PHỐI & CHUỖI ĐƯỜNG ỐNG COLUMNTRANSFORMER

---

### 1. Thiết kế Đặc trưng tuyến tính / Tỷ lệ đặc trưng (Ratio Features)

*   **Giải thích bản chất:** 
    Trong các dự án dữ liệu thực tế, các biến số thô thu thập được chưa chắc đã phản ánh đúng bản chất hành vi hay tính chất kinh tế của đối tượng. Ví dụ, tổng số phòng trong một quận (`total_rooms`) không có nhiều ý nghĩa nếu chúng ta không biết có bao nhiêu hộ gia đình (`households`) đang chia sẻ số phòng đó. 
    
    Bằng cách thực hiện **Thiết kế Đặc trưng (Feature Engineering)** để tạo ra các cột tỷ số mới, chúng ta cung cấp cho mô hình Học máy những thông tin mang tính ngữ nghĩa sâu sắc hơn, giúp các thuật toán (đặc biệt là mô hình tuyến tính) dễ dàng phát hiện ra các mối tương quan mạnh mẽ hơn với nhãn mục tiêu.
*   **Ví dụ thực tế trong tài liệu:** 
    Tài liệu tạo ra 3 đặc trưng tỷ lệ đột phá giúp cải thiện đáng kể hệ số tương quan với giá nhà trung vị:
    1.  **Số phòng trung bình trên mỗi hộ gia đình (`rooms_per_house`):** 
        
        \\[\text{rooms\_per\_house} = \frac{\text{total\_rooms}}{\text{households}}\\]
        
    2.  **Tỷ lệ phòng ngủ trên tổng số phòng (`bedrooms_per_room`):** 
        
        \\[\text{bedrooms\_per\_room} = \frac{\text{total\_bedrooms}}{\text{total\_rooms}}\\]
        
    3.  **Số người trung bình trên mỗi hộ gia đình (`people_per_house`):** 
        
        \\[\text{people\_per\_house} = \frac{\text{population}}{\text{households}}\\]
        
    *Hệ quả nghiên cứu:* Khi tính toán lại ma trận tương quan, thuộc tính mới `bedrooms_per_room` đạt hệ số tương quan **-0.26** với giá nhà. Con số này cao hơn nhiều so với các thuộc tính thô ban đầu như `total_rooms` hay `total_bedrooms`. Nó cho thấy một quy luật thực tế: nhà có tỷ lệ phòng ngủ trên tổng số phòng càng thấp (tức là nhà có nhiều phòng chức năng khác như phòng khách, phòng làm việc) thì giá trị căn nhà đó càng đắt đỏ.

---

### 2. Xử lý phân phối lệch bằng phép biến đổi Logarithm (Log Transformation)

*   **Giải thích bản chất:** 
    Rất nhiều thuật toán Học máy hoạt động tối ưu nhất khi các đặc trưng đầu vào tuân theo **Phân phối chuẩn hình chuông (Gaussian / Normal Distribution)**. Tuy nhiên, các biến số thực tế về dân số hay tài chính thường có xu hướng bị lệch nặng về một phía — cụ thể là **Lệch phải (Right-skewed / Heavy-tailed Distribution)**. Điều này nghĩa là phần lớn dữ liệu tập trung ở vùng giá trị thấp, trong khi có một số ít giá trị cực kỳ lớn kéo dài sang bên phải.
    
    Để giải quyết vấn đề này, phép biến đổi **Logarithm (Log) hoặc căn bậc hai** được áp dụng. Phép biến đổi này hoạt động bằng cách "nén" các giá trị cực lớn ở đuôi phải lại gần trung tâm hơn và kéo giãn các giá trị nhỏ ra, giúp phân phối của đặc trưng trở nên cân đối và đối xứng hơn, tiệm cận với phân phối chuẩn.
*   **Ví dụ thực tế trong tài liệu:**
    Các đặc trưng như `population`, `total_rooms`, `total_bedrooms`, và `households` đều bị lệch phải nghiêm trọng. Tài liệu sử dụng lớp `FunctionTransformer` của Scikit-Learn lồng ghép hàm toán học `np.log` để tự động hóa quy trình biến đổi này trong chuỗi đường ống tiền xử lý.

*   **Giải thích trực quan dựa trên hình ảnh (Hình 2-17):**
    
    \\[\text{Hình 2-17: Đối chiếu phân bổ của thuộc tính trước và sau khi áp dụng phép biến đổi Logarithm}\\]
    
    Nhìn vào sơ đồ biến đổi trực quan **Hình 2-17** trong tài liệu:
    *   **Biểu đồ bên trái (Trước khi biến đổi Log):** Phân bổ của đặc trưng dân số hiển thị một cột dốc đứng ở sát vách trái (vùng dưới 2.000 dân) rồi thoải dần một dải cực dài và mỏng về phía bên phải. Đây là dạng phân phối lệch phải điển hình, gây khó khăn lớn cho các trình tối ưu hóa dựa trên Gradient Descent.
    *   **Biểu đồ bên phải (Sau khi biến đổi Log):** Phân phối của biến số mới đã được kéo giãn hoàn hảo thành một hình chuông đối xứng tuyệt đẹp với đỉnh nằm ngay trung tâm. Mô hình giờ đây có thể học các mối quan hệ tuyến tính và phi tuyến dễ dàng hơn gấp nhiều lần.

---

### 3. Hàm tương đồng Gaussian RBF để tháo gỡ phân bổ đa đỉnh (Multimodal Distribution & RBF Kernel)

*   **Giải thích bản chất:** 
    Khi một đặc trưng có **phân bổ đa đỉnh (Multimodal Distribution)** — nghĩa là đồ thị của nó xuất hiện nhiều đỉnh núi cao độc lập — các phép biến đổi đơn giản như Logarithm sẽ hoàn toàn bất lực. Ví dụ tiêu biểu là tọa độ địa lý hay khoảng cách: giá nhà sẽ tăng vọt khi nằm gần một số "điểm nóng" (như trung tâm kinh tế Los Angeles hoặc San Francisco) và sụt giảm khi nằm ở xa.
    
    Để tháo gỡ cấu trúc đa đỉnh phức tạp này, chúng ta sử dụng **Hàm cơ sở xuyên tâm Gaussian RBF (Gaussian Radial Basis Function - RBF)**. Hàm này đo lường **Độ tương đồng (Similarity)** giữa mỗi điểm dữ liệu \\(\mathbf{x}\\) với một mốc định sẵn (landmark) \\(\boldsymbol{\mu}\\) theo công thức:
    
    \\[\phi(\mathbf{x}, \boldsymbol{\gamma}) = \exp\left(-\gamma \|\mathbf{x} - \boldsymbol{\mu}\|^2\right)\\]
    
    *Cơ chế hoạt động:*
    *   Khi điểm dữ liệu nằm **trùng khít** với mốc (\\(\mathbf{x} = \boldsymbol{\mu}\\)), giá trị tương đồng đạt mức tối đa bằng **`1.0`**.
    *   Khi điểm dữ liệu **dịch chuyển ra xa** mốc, giá trị tương đồng sụt giảm theo hàm mũ và tiệm cận dần về **`0.0`**.
    *   Siêu tham số \\(\gamma\\) (gamma) kiểm soát tốc độ sụt giảm của dải chuông theo khoảng cách.

*   **Ví dụ thực tế trong tài liệu:**
    Tài liệu thiết lập hai tâm mốc địa lý quan trọng là San Francisco và Los Angeles. Với mỗi quận, thuật toán sẽ tính toán giá trị tương đồng địa lý RBF so với hai tâm mốc này để làm đặc trưng huấn luyện mới.

*   **Giải thích trực quan dựa trên đồ thị phân bổ (Hình 2-18):**
    
    \\[\text{Hình 2-18: Chuyển đổi tọa độ vĩ độ địa lý thành các đặc trưng tương đồng RBF}\\]
    
    Dựa trên đồ thị mô phỏng **Hình 2-18** của tài liệu:
    *   **Biểu đồ phía trên (Vĩ độ thô):** Cho thấy phân bổ của các quận có hai đỉnh núi rạch ròi ở vĩ độ 34° (khu vực LA) và vĩ độ 38° (khu vực SF). 
    *   **Biểu đồ phía dưới (Đặc trưng tương đồng RBF):** Sau khi biến đổi, tọa độ thô được chuyển hóa thành hai đường cong hình chuông trơn tru. Một đường chuông ôm trọn vùng LA và đường còn lại ôm trọn vùng SF. Các giá trị nằm ngoài vùng ảnh hưởng của hai thành phố lớn này đều có điểm số tương đồng kéo sát về 0, giúp mô hình phân loại địa lý cực kỳ nhạy bén.

---

### 4. Chuỗi đường ống tích hợp `ColumnTransformer` (Hình 2-19)

*   **Giải thích bản chất:**
    Chúng ta có rất nhiều kiểu dữ liệu khác nhau (dữ liệu số thô, dữ liệu số lệch cần log, tọa độ cần tính toán RBF, dữ liệu phân loại văn bản cần mã hóa một-nóng). Việc xử lý thủ công từng cột riêng biệt cực kỳ tốn thời gian, dễ sai sót và không thể đóng gói để triển khai thực tế.
    
    Lớp **`ColumnTransformer`** của Scikit-Learn chính là giải pháp tối thượng. Nó hoạt động như một bộ điều phối trung tâm: cho phép chúng ta định nghĩa các nhánh đường ống biến đổi độc lập (pipeline) cho từng nhóm cột cụ thể, chạy song song tất cả các nhánh đó, rồi **ghép nối (concatenate) kết quả đầu ra dọc theo trục cột** để tạo thành một ma trận đặc trưng dày đặc duy nhất sẵn sàng nạp vào mô hình Học máy.

*   **Giải thích trực quan dựa trên sơ đồ luồng dữ liệu (Hình 2-19):**
    
    \\[\text{Hình 2-19: Sơ đồ cấu trúc luồng xử lý dữ liệu song song của ColumnTransformer}\]
    
    Sơ đồ **Hình 2-19** mô tả trọn vẹn "nhà máy chế biến đặc trưng" tự động của chúng ta với 4 nhánh song song:
    1.  **Nhánh Số cơ bản (Numerical Pipeline):** Tiếp nhận các cột số thông thường \\(\rightarrow\\) đi qua `SimpleImputer` để điền khuyết trung vị \\(\rightarrow\\) đi qua `StandardScaler` để đưa về cùng một thang đo.
    2.  **Nhánh Logarithm (Log Pipeline):** Tiếp nhận các cột bị lệch phải \\(\rightarrow\\) đi qua `FunctionTransformer(np.log)` \\(\rightarrow\\) đi qua `StandardScaler`.
    3.  **Nhánh Địa lý RBF (RBF Cluster Pipeline):** Tiếp nhận tọa độ kinh - vĩ độ \\(\rightarrow\\) đi qua một bộ biến đổi tùy chỉnh để tính điểm tương đồng địa lý dựa trên khoảng cách RBF đến các trung tâm đô thị.
    4.  **Nhánh Danh mục (Categorical Pipeline):** Tiếp nhận cột chữ `ocean_proximity` \\(\rightarrow\\) đi qua `OneHotEncoder` để chuyển hóa thành các cột nhị phân 0 và 1.
    
    *Điểm hội tụ:* Toàn bộ đầu ra của 4 nhánh này được kết nối trực tiếp vào khối liên kết cuối cùng để xuất ra một ma trận dữ liệu hợp nhất sạch sẽ.

---

### 5. Mã nguồn Python minh họa chi tiết

Dưới đây là mã nguồn xây dựng toàn bộ hệ thống tiền xử lý đặc trưng phức hợp sử dụng `ColumnTransformer` và các lớp tùy chỉnh chuyên nghiệp theo đúng tài liệu:

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

# ==========================================
# 1. Định nghĩa bộ biến đổi tùy chỉnh tính khoảng cách địa lý RBF
# ==========================================
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        # Sử dụng KMeans để tự động tìm ra n_clusters điểm mốc đô thị quan trọng nhất
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X)
        return self

    def transform(self, X):
        # Tính toán ma trận khoảng cách tương đồng RBF từ các điểm dữ liệu đến các tâm cụm học được
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster similarity {i}" for i in range(self.n_clusters)]

# ==========================================
# 2. Định nghĩa các nhóm cột cho từng nhánh xử lý độc lập
# ==========================================
# Các cột số cơ bản cần điền khuyết và chuẩn hóa
num_cols = ["housing_median_age", "total_rooms", "total_bedrooms", 
            "population", "households", "median_income"]

# Các cột bị lệch phải nghiêm trọng cần được biến đổi Logarithm
log_cols = ["total_rooms", "total_bedrooms", "population", "households", "median_income"]

# Cột địa lý cần tính toán tương đồng RBF
geo_cols = ["latitude", "longitude"]

# Cột phân loại dạng chữ
cat_cols = ["ocean_proximity"]

# ==========================================
# 3. Xây dựng các đường ống biến đổi con (Sub-pipelines)
# ==========================================
# Đường ống xử lý số cơ bản
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Đường ống xử lý biến đổi Log
log_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
    ("scaler", StandardScaler())
])

# Đường ống xử lý địa lý RBF cụm (Hình 2-18)
geo_pipeline = Pipeline([
    ("cluster_sim", ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)),
    ("scaler", StandardScaler())
])

# Đường ống xử lý danh mục chữ
cat_pipeline = Pipeline([
    ("one_hot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

# ==========================================
# 4. Hợp nhất tất cả vào ColumnTransformer trung tâm (Hình 2-19)
# ==========================================
preprocessing_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("log", log_pipeline, log_cols),
    ("geo", geo_pipeline, geo_cols),
    ("cat", cat_pipeline, cat_cols)
], remainder="drop") # Loại bỏ mọi cột không được chỉ định để bảo mật dữ liệu

# ==========================================
# 5. Chạy thử nghiệm trên tập dữ liệu huấn luyện
# ==========================================
# X_train_prepared = preprocessing_pipeline.fit_transform(X_train)
# print("Kích thước ma trận đặc trưng sau tiền xử lý:", X_train_prepared.shape)
```

---

# PHẦN 5: HUẤN LUYỆN MÔ HÌNH, KIỂM ĐỊNH CHÉO K-FOLD & TINH CHỈNH SIÊU THAM SỐ CHUYÊN SÂU

---

### 1. Huấn luyện Mô hình cơ bản & Các hiện tượng Lệch pha (Model Training & Basic Evaluation)

*   **Giải thích bản chất:** 
    Sau khi đã hoàn thành chuỗi đường ống tiền xử lý dữ liệu phức hợp, chúng ta bước vào giai đoạn cốt lõi: huấn luyện các thuật toán Học máy [cite: 114]. Tài liệu giới thiệu việc thử nghiệm nhanh ba thuật toán có tính chất hoàn toàn khác nhau để tìm ra giải pháp hứa hẹn:
    *   **Hồi quy tuyến tính (Linear Regression):** Mô hình tuyến tính cơ bản, phân bổ trọng số thô trên các đặc trưng đầu vào [cite: 172].
    *   **Cây quyết định (Decision Tree Regressor):** Mô hình phi tham số mạnh mẽ, có khả năng học các mối quan hệ phi tuyến phức tạp dựa trên việc phân chia các vùng dữ liệu [cite: 175].
    *   **Rừng ngẫu nhiên (Random Forest Regressor):** Mô hình học tổ hợp (Ensemble Learning) hoạt động bằng cách huấn luyện nhiều cây quyết định trên các tập con ngẫu nhiên của dữ liệu rồi lấy trung bình dự đoán, giúp cải thiện hiệu năng và giảm thiểu phương sai [cite: 179].
*   **Hiện tượng Dưới khớp (Underfitting) & Quá khớp (Overfitting) thực tế:**
    *   **Hồi quy tuyến tính dưới khớp (Underfitting):** Khi đo lường sai số RMSE của mô hình trên tập huấn luyện, kết quả đạt tới **\$68.687** [cite: 23, 173]. Trong khi đó, giá trị nhà trung bình ở hầu hết các quận chỉ nằm trong khoảng \$120.000 đến \$265.000 [cite: 174]. Sai số xấp xỉ \$68.000 cho thấy mô hình quá đơn giản và các đặc trưng hiện tại chưa cung cấp đủ thông tin phi tuyến cho một bộ hồi quy tuyến tính [cite: 174].
    *   **Cây quyết định quá khớp nghiêm trọng (Overfitting):** Khi đánh giá trên cùng tập huấn luyện, sai số RMSE trả về kết quả bằng **`0.0`** [cite: 24, 175]! Đây không phải là mô hình hoàn hảo mà thực chất là lỗi "học vẹt". Mô hình đã học thuộc lòng toàn bộ tập huấn luyện và không còn khả năng tổng quát hóa khi gặp dữ liệu lạ [cite: 176, 179].
    *   **Rừng ngẫu nhiên:** Đạt sai số trên tập huấn luyện là **\$17.474** [cite: 26, 181]. Kết quả này thấp hơn nhiều so với mô hình tuyến tính nhưng vẫn cao hơn so với điểm số kiểm định chéo thực tế (khoảng \$47.019) [cite: 26, 181]. Điều này cho thấy hiện tượng quá khớp vẫn diễn ra nhẹ và cần được chính quy hóa hoặc tinh chỉnh [cite: 181].

*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline

    # 1. Huấn luyện Hồi quy tuyến tính cơ bản
    lin_reg = make_pipeline(preprocessing, LinearRegression())
    lin_reg.fit(housing, housing_labels) [cite: 20, 172]

    # Dự đoán thử 5 mẫu đầu tiên để kiểm tra thực tế (Tái lập kết quả [cite: 21, 173])
    some_predictions = lin_reg.predict(housing)
    print("5 Dự đoán đầu tiên:", some_predictions[:5].round(-2)) [cite: 21, 173]
    print("5 Giá trị thực tế: ", housing_labels.iloc[:5].values) [cite: 21, 173]

    # 2. Huấn luyện Cây quyết định (gây ra hiện tượng quá khớp nghiêm trọng RMSE = 0)
    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42)) [cite: 23, 175]
    tree_reg.fit(housing, housing_labels) [cite: 23, 175]
    tree_predictions = tree_reg.predict(housing)
    tree_rmse = root_mean_squared_error(housing_labels, tree_predictions) [cite: 24, 175]
    print(f"RMSE của Cây quyết định trên tập huấn luyện: {tree_rmse:.2f} USD") # Kết quả: 0.0 [cite: 24, 175]
    ```

---

### 2. Đánh giá tốt hơn bằng Kiểm định chéo K-Fold (K-Fold Cross-Validation)

*   **Giải thích bản chất:**
    Để đo lường khả năng tổng quát hóa của mô hình một cách khách quan nhất mà không cần chạm vào tập kiểm thử bảo mật, chúng ta sử dụng phương pháp **Kiểm định chéo K-Fold** [cite: 176, 177]. 
    
    Ý tưởng là chia tập huấn luyện thành \\(K\\) phần không chồng lấn (gọi là các folds) [cite: 177]. Mô hình sẽ được huấn luyện và đánh giá chéo \\(K\\) lần độc lập; tại mỗi lần, một fold riêng biệt sẽ được giữ lại làm tập xác thực (validation set) để tính điểm, còn \\(K-1\\) folds còn lại được gộp chung làm tập huấn luyện [cite: 177].
*   **Bản chất toán học của Điểm số âm trong Scikit-Learn:**
    API kiểm định chéo `cross_val_score` của Scikit-Learn yêu cầu sử dụng một **Hàm tiện ích (Utility function - càng lớn càng tốt)** thay vì một Hàm chi phí (Cost function - càng nhỏ càng tốt) [cite: 24, 177]. 
    
    Do đó, khi tính toán RMSE, chúng ta phải chỉ định tham số `scoring="neg_root_mean_squared_error"` để nhận về các giá trị âm [cite: 24, 177]. Để thu được giá trị RMSE dương thực tế, lập trình viên cần chủ động **thêm dấu âm (`-`)** phía trước kết quả trả về [cite: 24, 177].
*   **Số liệu cụ thể đối chiếu hiệu năng thực tế (10-Fold CV):**
    Sau khi chạy kiểm định chéo 10 lần, chúng ta thu được bức tranh thực sự về hiệu suất của các mô hình [cite: 24, 25, 26, 177, 180]:

| Thuật toán | Điểm RMSE trung bình (Mean RMSE) | Độ lệch chuẩn (Standard Deviation) | Nhận xét chi tiết từ tài liệu |
| :--- | :---: | :---: | :--- |
| **Linear Regression** | **`69.858`** USD [cite: 25, 179] | \\(\pm 4.182\\) USD [cite: 25, 179] | Mô hình hoạt động kém, lỗi dao động lớn [cite: 174, 179]. |
| **Decision Tree** | **`66.868`** USD [cite: 24, 178] | \\(\pm 2.061\\) USD [cite: 24, 178] | **RMSE thực sự hiển hiện**. Điểm số tệ tương đương mô hình tuyến tính, chứng minh RMSE = 0.0 trước đó chỉ là ảo ảnh quá khớp [cite: 178, 179]. |
| **Random Forest** | **`47.019`** USD [cite: 26, 180] | \\(\pm 1.033\\) USD [cite: 26, 180] | **Mô hình vượt trội hoàn toàn**, độ lệch chuẩn rất nhỏ, cực kỳ ổn định [cite: 181]. |

*   **Mã nguồn Python minh họa:**
    ```python
    import pandas as pd
    from sklearn.model_selection import cross_val_score

    # Đánh giá Cây quyết định bằng 10-Fold Cross-Validation (Hình ảnh trong [cite: 24, 177])
    tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                                  scoring="neg_root_mean_squared_error", cv=10) [cite: 24, 177]
                                  
    # In thống kê mô tả (describe) để có điểm số trung bình và độ lệch chuẩn (Tái lập [cite: 24, 178])
    print(pd.Series(tree_rmses).describe())
    ```

---

### 3. Tinh chỉnh siêu tham số bằng Tìm kiếm theo lưới (GridSearchCV)

*   **Giải thích bản chất:**
    Khi đã chọn được Random Forest làm mô hình hứa hẹn nhất, chúng ta cần tinh chỉnh các siêu tham số của nó [cite: 181, 182]. **`GridSearchCV`** giúp tự động hóa quá trình này bằng cách duyệt qua toàn bộ các tổ hợp siêu tham số có thể có từ một lưới (Grid) định sẵn, đánh giá hiệu năng của chúng bằng kiểm định chéo và giữ lại cấu hình tối ưu nhất [cite: 182, 186].
*   **Ý nghĩa của ký hiệu dấu gạch dưới kép `__`:**
    Trong một Pipeline phức tạp gồm nhiều thành phần lồng nhau, `GridSearchCV` cho phép chúng ta cấu hình trực tiếp các siêu tham số nằm sâu bên trong bằng cách sử dụng cú pháp: **`{tên_thành_phần_pipeline}__{tên_siêu_tham_số}`** [cite: 184].
    *   Ví dụ: `'preprocessing__geo__n_clusters'` [cite: 27, 183]. Hệ thống sẽ phân tách chuỗi tại vị trí `__`: tìm khối tiền xử lý `'preprocessing'` [cite: 184] \\(\rightarrow\\) tìm nhánh địa lý `'geo'` bên trong [cite: 184] \\(\rightarrow\\) định cấu hình tham số `n_clusters` của lớp `ClusterSimilarity` [cite: 184].
*   **Ví dụ thực tế trong tài liệu:**
    Lưới tham số `param_grid` được thiết lập gồm hai từ điển độc lập để thử nghiệm song song [cite: 27, 183, 185]:
    *   **Từ điển 1:** Thử nghiệm 3 giá trị `n_clusters` kết hợp với 3 giá trị `max_features` \\(\rightarrow\\) \\(3 \times 3 = 9\\) tổ hợp [cite: 27, 183, 185].
    *   **Từ điển 2:** Thử nghiệm 2 giá trị `n_clusters` kết hợp với 3 giá trị `max_features` \\(\rightarrow\\) \\(2 \times 3 = 6\\) tổ hợp [cite: 27, 183, 185].
    
    Tổng cộng lưới khám phá **15 tổ hợp** siêu tham số [cite: 185]. Với kiểm định chéo 3-fold (`cv=3`), hệ thống sẽ thực hiện huấn luyện tổng cộng **\\(15 \times 3 = 45\\) vòng độc lập** [cite: 27, 183, 185].

*   **Giải thích trực quan dựa trên hình ảnh kết quả (`multimodal_19`):**
    
    \\[\text{Hình ảnh trong mã nguồn: Bảng thống kê chi tiết điểm số cv\_results\_ của GridSearchCV}\\]
    
    Dựa trên kết quả chạy thực tế hiển thị tại ảnh **`multimodal_19`**:
    *   Bảng kết quả tra cứu cho thấy sự cải thiện rõ rệt của mô hình khi cấu hình thay đổi [cite: 31, 187].
    *   Khi tăng số lượng cụm địa lý `n_clusters` lên `15` và đặt `max_features` bằng `6` (Dòng chỉ mục 12), điểm sai số **mean_test_rmse đạt mức thấp kỷ lục là `44.042` USD** [cite: 29, 31, 187]. Điểm số này vượt trội hoàn toàn so với mức lỗi mặc định ban đầu là \$47.019 [cite: 26, 180, 188].

*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.model_selection import GridSearchCV

    # 1. Định nghĩa pipeline hoàn chỉnh gồm tiền xử lý và mô hình Random Forest
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ]) [cite: 27, 183]

    # 2. Thiết lập lưới tham số chi tiết (Tái lập [cite: 27, 183])
    param_grid = [
        {'preprocessing__geo__n_clusters':,
         'random_forest__max_features':},
        {'preprocessing__geo__n_clusters':,
         'random_forest__max_features':},
    ] [cite: 27, 183]

    # 3. Chạy tìm kiếm lưới chéo với cv = 3
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                               scoring='neg_root_mean_squared_error') [cite: 27, 183]
    grid_search.fit(housing, housing_labels) [cite: 27, 183]

    # Xuất ra tổ hợp tốt nhất (Tái lập [cite: 29, 186])
    print("Tổ hợp tối ưu nhất:", grid_search.best_params_)
    # Kết quả: {'preprocessing__geo__n_clusters': 15, 'random_forest__max_features': 6}
    ```

---

### 4. Tinh chỉnh siêu tham số bằng Tìm kiếm ngẫu nhiên (RandomizedSearchCV)

*   **Giải thích bản chất:**
    Khi không gian tìm kiếm siêu tham số mở rộng quá lớn (nhiều siêu tham số liên tục), việc sử dụng Tìm kiếm lưới sẽ trở nên bất khả thi do bùng nổ số lượng tổ hợp cần huấn luyện [cite: 188, 189]. **`RandomizedSearchCV`** khắc phục triệt để bằng cách không thử nghiệm tất cả các điểm cố định [cite: 188]. Thay vào đó, tại mỗi lượt lặp (`n_iter`), nó sẽ rút ngẫu nhiên một giá trị cho mỗi siêu tham số từ một phân phối xác suất được chỉ định trước [cite: 188, 190].
    
    *Ưu thế tuyệt đối:*
    1.  Cho phép khám phá hàng ngàn giá trị siêu tham số liên tục khác nhau thay vì chỉ bó hẹp trong vài giá trị khai báo cứng của Grid Search [cite: 189].
    2.  Không lãng phí tài nguyên tính toán vào các siêu tham số ít quan trọng [cite: 189].
    3.  Kiểm soát hoàn toàn thời gian huấn luyện và ngân sách thông qua tham số số lượt lặp `n_iter` [cite: 189].

*   **Giải thích trực quan dựa trên phân phối xác suất (Hình ảnh trong `multimodal_21` - top):**
    
    \\[\text{Hình ảnh trong mã nguồn: Đồ thị mô phỏng phân phối expon và loguniform của SciPy}\\]
    
    Khi cấu hình cho tìm kiếm ngẫu nhiên, việc lựa chọn phân phối xác suất phù hợp đóng vai trò quyết định hiệu năng:
    *   **Phân phối hàm mũ `expon(scale=1.0)` (Đồ thị góc trên bên trái):** Được sử dụng khi chúng ta **đã phỏng đoán được thang đo tối ưu** của tham số (ví dụ: gamma thích hợp nằm quanh khoảng 1.0) [cite: 48]. Đồ thị cho thấy mật độ xác suất tập trung dày đặc ở vùng giá trị nhỏ và dốc dần về phía bên phải; khoảng 80% mẫu ngẫu nhiên rút ra sẽ tập trung ổn định trong khoảng từ 0.1 đến 2.3 [cite: 48].
    *   **Phân phối `loguniform` / `reciprocal` (Đồ thị góc dưới bên trái):** Được áp dụng khi chúng ta **hoàn toàn mơ hồ, không có manh mối nào về thang đo tối ưu** của tham số (ví dụ: C có thể là 20 hoặc 200.000) [cite: 45, 48]. Đồ thị phân phối log cho thấy cơ hội lấy mẫu được phân bổ đều trên các thang bậc lũy thừa cơ số 10 (ví dụ: khoảng xác suất rút ra từ 20 đến 200 tương đương với từ 2.000 đến 20.000) [cite: 48].

*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint

    # 1. Khai báo các phân phối xác suất cho từng siêu tham số
    param_distribs = {
        'preprocessing__geo__n_clusters': randint(low=3, high=50), [cite: 190]
        'random_forest__max_features': randint(low=2, high=20) [cite: 190]
    }

    # 2. Khởi tạo RandomizedSearchCV chạy thử 10 tổ hợp ngẫu nhiên
    rnd_search = RandomizedSearchCV(
        full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
        scoring='neg_root_mean_squared_error', random_state=42) [cite: 32, 190]
        
    rnd_search.fit(housing, housing_labels) [cite: 33, 191]
    ```

---

### 5. Phân tích các mô hình tốt nhất & Tầm quan trọng của Đặc trưng (Feature Importances)

*   **Giải thích bản chất:**
    Một trong những ưu điểm vượt trội của thuật toán Rừng ngẫu nhiên là khả năng lượng hóa mức độ đóng góp của từng đặc trưng đầu vào cho việc giảm thiểu sai số dự báo của hệ thống, thông qua thuộc tính **`feature_importances_`** [cite: 35, 193]. 
    
    Bằng cách sắp xếp độ quan trọng này theo thứ tự giảm dần và ghép nối với tên đặc trưng được trích xuất từ chuỗi tiền xử lý (qua phương thức `get_feature_names_out()`), chúng ta có được những góc nhìn sâu sắc để tối ưu hóa đặc trưng [cite: 36, 194, 195].

*   **Giải thích trực quan dựa trên kết quả (`multimodal_21` - bottom):**
    
    \\[\text{Hình ảnh trong mã nguồn: Danh sách các đặc trưng được sắp xếp theo độ quan trọng}\\]
    
    Dựa trên mảng dữ liệu thực tế in ra từ ảnh **`multimodal_21`**:
    1.  **`log__median_income` (18.69%):** Đóng vai trò thống trị tuyệt đối [cite: 36, 194]. Thu nhập trung vị của hộ gia đình (sau khi log) là yếu tố quyết định hàng đầu đến giá nhà [cite: 143, 194].
    2.  **`cat__ocean_proximity_INLAND` (7.48%):** Là đặc trưng vị trí quan trọng thứ hai [cite: 36, 194]. Nó xác nhận quy luật địa lý: việc một ngôi nhà nằm sâu trong đất liền (Inland) có tác động rất lớn (thường kéo giá nhà giảm mạnh) [cite: 143].
    3.  **`bedrooms__ratio` (6.93%):** Đặc trưng tỷ số phòng ngủ tự thiết kế ở Phần 4 tỏ ra cực kỳ hiệu quả, vượt trội hơn nhiều so với các đặc trưng gốc [cite: 36, 194].
    
    *Hành động thực tế:* Dựa trên danh sách này, chúng ta có thể chủ động drop bỏ các cột danh mục ocean_proximity kém quan trọng khác (như `NEAR BAY` chỉ đạt 0.015%) để đơn giản hóa kiến trúc mô hình và tăng tốc độ xử lý của hệ thống [cite: 37, 194, 195].

*   **Mã nguồn Python minh họa:**
    ```python
    # 1. Trích xuất mô hình tối ưu nhất và đo đạc độ quan trọng của đặc trưng
    final_model = rnd_search.best_estimator_ [cite: 35, 193]
    feature_importances = final_model["random_forest"].feature_importances_ [cite: 35, 193]

    # 2. Lấy tên các đặc trưng đầu ra từ ColumnTransformer tiền xử lý
    feature_names = final_model["preprocessing"].get_feature_names_out() [cite: 36, 194]

    # 3. Sắp xếp giảm dần và in ra màn hình (Tái lập [cite: 36, 194])
    sorted_features = sorted(zip(feature_importances, feature_names), reverse=True) [cite: 36, 194]
    for importance, name in sorted_features[:5]:
        print(f"Đặc trưng: {name:<30} | Độ quan trọng: {importance:.2%}")
    ```

---

### 6. Đánh giá hệ thống trên Tập kiểm thử (Evaluating on the Test Set)

*   **Giải thích bản chất:**
    Sau nhiều tuần thử nghiệm và tinh chỉnh, khi đã có được một mô hình tối ưu mà chúng ta hoàn toàn tự tin, bước cuối cùng là đánh giá hiệu năng thực tế của hệ thống trên **Tập kiểm thử (Test Set)** gạt riêng từ đầu chương [cite: 196]. 
    
    Chúng ta tiến hành dự đoán và tính toán khoảng tin cậy 95% (95% Confidence Interval) cho lỗi RMSE [cite: 38, 196, 197]. Khoảng tin cậy này giúp đo lường độ chính xác của ước tính sai số: cho biết sai số thực tế khi triển khai mô hình vào đời thực sẽ nằm trong khoảng nào với độ tin cậy 95% [cite: 197].
*   **Kết quả định lượng cụ thể từ tài liệu:**
    *   **Lỗi RMSE cuối cùng đạt được:** **`41.424.40` USD** [cite: 38, 196].
    *   **Khoảng tin cậy 95% cho RMSE:** Dao động từ **`39.275` đến `43.467` USD** (được tính toán bằng kiểm định giả thuyết t-test dựa trên mảng sai số bình phương) [cite: 38, 197]. Khoảng dao động hẹp này khẳng định mô hình có độ ổn định cực cao và sẵn sàng đưa vào môi trường sản xuất [cite: 198].

*   **Mã nguồn Python minh họa:**
    ```python
    from scipy import stats

    # 1. Tách đặc trưng và nhãn từ tập kiểm thử bảo mật
    X_test = strat_test_set.drop("median_house_value", axis=1) [cite: 37, 196]
    y_test = strat_test_set["median_house_value"].copy() [cite: 37, 196]

    # 2. Chạy dự báo trực tiếp (Pipeline tự động hóa toàn bộ khâu tiền xử lý X_test)
    final_predictions = final_model.predict(X_test) [cite: 38, 196]

    # 3. Tính toán sai số RMSE tập kiểm thử
    final_rmse = root_mean_squared_error(y_test, final_predictions) [cite: 38, 196]
    print(f"RMSE trên tập kiểm thử: {final_rmse:.2f} USD") # Kết quả: ~ 41424.40 USD [cite: 38, 196]

    # 4. Tính khoảng tin cậy 95% cho RMSE (Tái lập [cite: 38, 197])
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2 [cite: 38, 197]
    ci = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                  loc=squared_errors.mean(),
                                  scale=stats.sem(squared_errors))) [cite: 38, 197]
    print(f"Khoảng tin cậy 95% của RMSE: {ci} USD")
    ```

---

### 7. Lưu trữ mô hình bằng `joblib` (Model Persistence)

*   **Giải thích bản chất:**
    Để có thể triển khai mô hình hoạt động trực tiếp trong các ứng dụng thực tế (như một dịch vụ web API chạy trên đám mây Vertex AI) mà không cần phải thực hiện lại quy trình huấn luyện tốn kém, chúng ta cần lưu trữ (serialize) mô hình đã tối ưu xuống đĩa cứng [cite: 198, 199]. 
    
    Thư viện **`joblib`** được ưu tiên lựa chọn thay thế cho thư viện `pickle` mặc định của Python vì nó được tối ưu hóa đặc biệt để xử lý cực kỳ nhanh các đối tượng Học máy chứa các ma trận dữ liệu NumPy lớn (như hàng ngàn cây quyết định trong mô hình Rừng ngẫu nhiên) [cite: 40].
*   **Mã nguồn Python minh họa:**
    ```python
    import joblib

    # 1. Lưu trữ toàn bộ mô hình (gồm cả preprocessing pipeline bên trong) thành 1 tệp tin duy nhất
    joblib.dump(final_model, "my_california_housing_model.pkl") [cite: 39, 198]

    # 2. Quy trình nạp lại mô hình trong script chạy môi trường sản xuất (Tái lập [cite: 40, 198])
    final_model_reloaded = joblib.load("my_california_housing_model.pkl") [cite: 40, 198]

    # Dự đoán tức thời cho dữ liệu quận mới đổ về
    # new_predictions = final_model_reloaded.predict(new_data)
    ```

---

### KẾT LUẬN TOÀN DIỆN CHƯƠNG 2

Chúng ta đã cùng nhau hoàn thành xuất sắc toàn bộ hành trình xây dựng một **Dự án Học máy từ đầu đến cuối (End-to-End)** [cite: 114]:
1.  **Phần 1:** Định hình bài toán kinh doanh, lựa chọn các thước đo khoảng cách chuẩn mực (RMSE, MAE) và thiết lập tập kiểm thử bảo mật [cite: 117, 121, 124, 132].
2.  **Phần 2:** Ngăn ngừa sai lệch lấy mẫu bằng phương pháp Lấy mẫu phân tầng và thấu hiểu dữ liệu qua trực quan hóa địa lý đa biến [cite: 135, 141, 145].
3.  **Phần 3:** Làm sạch dữ liệu rỗng và chuẩn hóa mã hóa các cột phân loại (One-Hot Encoding) an toàn cho sản xuất [cite: 146, 151].
4.  **Phần 4:** Sáng tạo các đặc trưng tỷ số mới, nén đuôi lệch bằng Log và thiết lập "nhà máy biến đổi" tự động `ColumnTransformer` [cite: 153, 168, 169].
5.  **Phần 5:** Huấn luyện, đánh giá K-Fold khách quan, thám hiểm không gian siêu tham số và đóng gói mô hình sẵn sàng phục vụ thực tế [cite: 177, 182, 188, 198].

---