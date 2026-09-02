Để giúp bạn dễ dàng làm quen và làm chủ các kiến thức cốt lõi của **Chương 3: Phân loại (Classification)** trong học phần Máy học, tôi đề xuất chia nội dung chương này thành **5 phần chi tiết** đi từ cơ bản đến nâng cao:

*   **Phần 1: Giới thiệu Bộ dữ liệu MNIST & Khái niệm Phân loại nhị phân** *(Chúng ta sẽ hoàn thành phần đầu tiên này trước)*.
*   **Phần 2: Ma trận nhầm lẫn (Confusion Matrix) & Các chỉ số đánh giá cơ bản (Precision, Recall, F1-Score)**.
*   **Phần 3: Đánh đổi Precision/Recall & Ngưỡng quyết định (Decision Threshold)**.
*   **Phần 4: Đường cong ROC (Receiver Operating Characteristic) & AUC**.
*   **Phần 5: Phân loại đa lớp (Multiclass), Phân loại đa nhãn (Multilabel) & Phân loại đa đầu ra (Multioutput)**.

Dưới đây là chi tiết **Phần 1** với đầy đủ định nghĩa, giải thích bản chất, trích dẫn trực quan từ hình ảnh trong tài liệu và mã nguồn Python minh họa cụ thể.

---

# PHẦN 1: BỘ DỮ LIỆU MNIST & PHÂN LOẠI NHỊ PHÂN

### 1. Bộ dữ liệu MNIST (MNIST Dataset)

*   **Giải thích bản chất:** 
    MNIST là một tập hợp gồm **70.000 hình ảnh nhỏ** chụp các chữ số viết tay bởi học sinh trung học và nhân viên Cục Thống kê Hoa Kỳ. Mỗi hình ảnh được gắn nhãn trước với chữ số mà nó đại diện. MNIST phổ biến tới mức nó thường được gọi là **"Hello World" của ngành Học máy**.
    
    Về mặt kỹ thuật, mỗi hình ảnh có kích thước gốc là **\\(28 \times 28\\) pixel**. Khi chuyển đổi thành dữ liệu đầu vào cho mô hình, hình ảnh được trải phẳng thành một vector **784 đặc trưng**. Mỗi đặc trưng đại diện cho cường độ sáng của một pixel duy nhất, nhận giá trị từ `0` (trắng hoàn toàn) đến `255` (đen hoàn toàn).
*   **Ví dụ thực tế trong tài liệu:** 
    Sử dụng hàm `fetch_openml('mnist_784', as_frame=False)` của Scikit-Learn để tải trực tuyến toàn bộ dữ liệu này về máy. Sử dụng tham số `as_frame=False` vì dữ liệu hình ảnh được xử lý hiệu quả nhất dưới dạng mảng NumPy thay vì cấu trúc bảng Pandas DataFrame.
*   **Giải thích trực quan dựa trên hình ảnh:**
    *   **Trực quan hóa một chữ số đơn lẻ (Hình 3-1):** 
        Khi lấy phần tử dữ liệu đầu tiên \\(X\\) (có nhãn là `'5'`), chúng ta định hình lại nó từ vector 784 chiều về ma trận \\(28 \times 28\\) và vẽ bằng thư viện Matplotlib. Kết quả cho ra hình ảnh trực quan của số 5 viết tay như dưới đây:
        
        \\[\text{Hình 3-1: Ví dụ về hình ảnh MNIST (Số 5 viết tay)}\\]
        \\[\text{[some\_digit\_plot - Số 5 viết tay nét đen trên nền trắng]}\\]
    *   **Trực quan hóa sự đa dạng của tập dữ liệu (Hình 3-2):**
        Để thấy mức độ phức tạp của tác vụ này, tài liệu vẽ một lưới \\(10 \times 10\\) gồm 100 chữ số đầu tiên trong bộ dữ liệu:
        
        \\[\text{Hình 3-2: Các chữ số từ bộ dữ liệu MNIST}\\]
        \\[\text{[more\_digits\_plot - Lưới 100 chữ số viết tay với nhiều hình dáng, nét chữ khác nhau]}\\]
        
        Dựa vào hình ảnh này, ta thấy cùng một chữ số (ví dụ số 5 hay số 3) nhưng mỗi người lại có một cách viết khác nhau (nét nghiêng, thẳng, viết liền hoặc đứt đoạn). Đây chính là lý do các hệ thống lập trình truyền thống (dùng luật cứng) bất khả thi trong việc nhận diện, đòi hỏi phải sử dụng Học máy để học các mẫu đặc trưng từ dữ liệu.

*   **Mã nguồn Python minh họa:**
    ```python
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml

    # 1. Tải bộ dữ liệu MNIST dưới dạng mảng NumPy
    mnist = fetch_openml('mnist_784', as_frame=False)
    X, y = mnist.data, mnist.target

    # 2. Kiểm tra kích thước của dữ liệu
    print("Kích thước đặc trưng (X):", X.shape)  # Kết quả: (70000, 784)
    print("Kích thước nhãn mục tiêu (y):", y.shape)  # Kết quả: (70000,)

    # 3. Hàm vẽ một chữ số dựa trên vector 784 đặc trưng
    def plot_digit(image_data):
        image = image_data.reshape(28, 28)  # Định hình lại thành ma trận 28x28 pixel
        plt.imshow(image, cmap="binary")    # Vẽ ảnh xám (0 là trắng, 255 là đen)
        plt.axis("off")                     # Ẩn trục tọa độ

    # 4. Vẽ thử phần tử đầu tiên (Hình 3-1)
    some_digit = X
    plot_digit(some_digit)
    plt.show()

    # Kiểm tra nhãn thực tế đi kèm
    print("Nhãn của some_digit là:", y)  # Kết quả: '5'
    ```

---

### 2. Bộ phân loại nhị phân & Thuật toán SGD (Binary Classifier & SGD)

*   **Giải thích bản chất:**
    **Bộ phân loại nhị phân** là một hệ thống có khả năng phân biệt dữ liệu thành **chỉ hai lớp duy nhất** (ví dụ: lớp dương tính và lớp âm tính, True và False, hoặc Đạt và Không đạt). 
    
    Để xây dựng bộ phân loại nhị phân này, tài liệu giới thiệu thuật toán **Xuống dốc ngẫu nhiên (Stochastic Gradient Descent - SGD)** thông qua lớp `SGDClassifier`. Điểm mạnh vượt trội của SGD là khả năng xử lý các tập dữ liệu cực kỳ lớn một cách hiệu quả. Điều này là do SGD xử lý các mẫu dữ liệu huấn luyện một cách độc lập và tuần tự, từng mẫu một. Đặc điểm này cũng làm cho SGD trở nên cực kỳ phù hợp cho các tác vụ **Học trực tuyến (Online Learning)**.
*   **Ví dụ thực tế trong tài liệu:**
    Tài liệu thiết lập một bộ phân loại nhị phân gọi là **"Bộ phát hiện số 5" (5-detector)**. Nhiệm vụ của nó là nhận vào một hình ảnh bất kỳ và trả về kết quả `True` nếu hình ảnh đó là số 5, ngược lại trả về `False` (đối với mọi chữ số khác từ 0 đến 9).
*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.linear_model import SGDClassifier

    # 1. Phân chia tập huấn luyện và tập kiểm thử (MNIST đã được xáo trộn sẵn)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # 2. Tạo nhãn nhị phân: True đối với số 5, False đối với tất cả các số khác
    y_train_5 = (y_train == '5')
    y_test_5 = (y_test == '5')

    # 3. Khởi tạo mô hình SGDClassifier với random_state để kết quả có thể lặp lại
    sgd_clf = SGDClassifier(random_state=42)

    # 4. Huấn luyện mô hình trên tập dữ liệu nhị phân
    sgd_clf.fit(X_train, y_train_5)

    # 5. Thử dự đoán hình ảnh some_digit (số 5) ở trên
    prediction = sgd_clf.predict([some_digit])
    print("Dự đoán của mô hình cho some_digit:", prediction)  # Kết quả: [True]
    ```

---

### 3. Kiểm định chéo phân tầng (Stratified K-Fold Cross-Validation)

*   **Giải thích bản chất:**
    **Kiểm định chéo k-fold (k-fold Cross-Validation)** chia tập huấn luyện thành \\(k\\) phần (gọi là các folds). Mô hình sẽ được huấn luyện và đánh giá chéo \\(k\\) lần độc lập; tại mỗi lần, một fold riêng biệt sẽ được giữ lại để làm tập kiểm thử để tính điểm, còn \\(k-1\\) folds còn lại được dùng làm tập huấn luyện.
    
    Đối với các tác vụ phân loại, việc chia ngẫu nhiên thông thường có thể khiến một fold bị lệch (ví dụ fold đó hoàn toàn thiếu vắng chữ số 5). Để giải quyết điều này, kỹ thuật **Stratified K-Fold (Lấy mẫu phân tầng)** được áp dụng. Phương pháp này chia các fold sao cho **tỷ lệ đại diện của từng lớp trong mỗi fold luôn tương đương với tỷ lệ đại diện của lớp đó trong toàn bộ tập dữ liệu gốc**.
*   **Ví dụ thực tế trong tài liệu:**
    Tài liệu hướng dẫn cách tự triển khai quy trình kiểm định chéo phân tầng bằng cách sử dụng lớp `StratifiedKFold` của Scikit-Learn kết hợp với hàm sao chép mô hình `clone()`. Điều này giúp lập trình viên kiểm soát chi tiết từng bước huấn luyện và đánh giá trên từng fold, thay vì chỉ nhận về điểm số cuối cùng như khi dùng hàm đóng gói sẵn `cross_val_score()`.
*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    # Thiết lập kiểm định chéo phân tầng với 3 folds
    skfolds = StratifiedKFold(n_splits=3)

    # Vòng lặp duyệt qua từng fold huấn luyện và kiểm thử chéo
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        # 1. Tạo một bản sao sạch của mô hình ban đầu để tránh rò rỉ thông tin giữa các fold
        clone_clf = clone(sgd_clf)
        
        # 2. Phân tách dữ liệu folds dựa trên các chỉ mục (indexes)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train_5[test_index]
        
        # 3. Huấn luyện mô hình nhân bản trên fold huấn luyện
        clone_clf.fit(X_train_folds, y_train_folds)
        
        # 4. Dự đoán trên fold kiểm định tương ứng
        y_pred = clone_clf.predict(X_test_fold)
        
        # 5. Tính toán và in ra tỷ lệ dự đoán đúng (Accuracy) của fold này
        n_correct = sum(y_pred == y_test_fold)
        print("Tỷ lệ dự đoán đúng của fold:", n_correct / len(y_pred))
        
    # Kết quả in ra lần lượt sẽ tương đương với: 0.95035, 0.96035, và 0.9604
    ```

---

### 4. Tập dữ liệu lệch & Sự hạn chế của Độ chính xác (Skewed Dataset & Accuracy Limit)

*   **Giải thích bản chất:**
    *   **Độ chính xác (Accuracy):** Chỉ đơn giản là tỷ lệ số lượng mẫu dự đoán đúng trên tổng số lượng mẫu dự đoán.
    *   **Tập dữ liệu lệch (Skewed Dataset):** Là tập dữ liệu mà trong đó một vài lớp có số lượng mẫu vượt trội hoàn toàn so với các lớp còn lại. (Trong ví dụ MNIST nhị phân, số lượng số 5 chỉ chiếm khoảng 10% tập dữ liệu, 90% còn lại là các chữ số khác).
    *   **Hạn chế của Accuracy:** Trên một tập dữ liệu lệch, **Accuracy không còn là một thước đo đáng tin cậy để đánh giá hiệu suất**. Nếu một mô hình không học bất kỳ thứ gì, chỉ đơn thuần đoán mọi hình ảnh đều thuộc về lớp chiếm đa số (lớp phủ định), nó vẫn sẽ đạt được độ chính xác rất cao (lên tới 90%)! Điều này tạo ra một ảo giác sai lệch rằng mô hình hoạt động hiệu quả.
*   **Ví dụ thực tế trong tài liệu:**
    Để chứng minh sự hạn chế này, tài liệu xây dựng một **bộ phân loại giả (Dummy Classifier)** bằng lớp `DummyClassifier`. Bộ phân loại này vô cùng ngây ngô: nó không thèm xem hình ảnh chữ số chứa gì, nó chỉ luôn luôn dự đoán mọi hình ảnh đều là `False` (không phải là số 5). Khi chạy kiểm định chéo, mô hình giả này vẫn đạt được độ chính xác tuyệt đối là **90,9%** trên cả 3 folds!
*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import cross_val_score

    # 1. Khởi tạo một Dummy Classifier (mặc định sẽ luôn dự đoán lớp phổ biến nhất)
    dummy_clf = DummyClassifier()
    dummy_clf.fit(X_train, y_train_5)

    # 2. Kiểm tra xem mô hình giả này có phát hiện được bất kỳ số 5 nào không
    print("Có số 5 nào được phát hiện không?:", any(dummy_clf.predict(X_train))) 
    # Kết quả in ra: False (Mô hình luôn đoán là "không phải số 5")

    # 3. Đánh giá độ chính xác (Accuracy) của bộ phân loại giả này bằng 3-fold cross validation
    dummy_scores = cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    print("Độ chính xác của Dummy Classifier trên mỗi fold:", dummy_scores)
    # Kết quả in ra: array([0.90965, 0.90965, 0.90965])
    ```

---

# PHẦN 2: MA TRẬN NHẦM LẪN & CÁC CHỈ SỐ ĐÁNH GIÁ CƠ BẢN (PRECISION, RECALL, F1-SCORE)

---

### 1. Ma trận nhầm lẫn (Confusion Matrix)

*   **Giải thích bản chất:** 
    Một phương pháp đánh giá hiệu năng của bộ phân loại tốt hơn nhiều so với việc chỉ nhìn vào độ chính xác (Accuracy) là phân tích **Ma trận nhầm lẫn (Confusion Matrix)**. Ý tưởng cốt lõi của ma trận nhầm lẫn là **đếm số lần các thực thể thuộc lớp A bị phân loại nhầm thành lớp B**, áp dụng cho tất cả các cặp lớp A và B.
    
    Cấu trúc của ma trận nhầm lẫn nhị phân gồm có:
    *   **Hàng (Rows):** Biểu thị các **lớp thực tế (Actual classes)**.
    *   **Cột (Columns):** Biểu thị các **lớp được dự đoán (Predicted classes)**.
    
    Từ đó, ma trận chia dữ liệu thành 4 nhóm cụ thể:
    1.  **True Negatives (TN - Âm tính đúng):** Các trường hợp âm tính thực tế và được mô hình phân loại đúng là âm tính.
    2.  **False Positives (FP - Dương tính giả / Lỗi loại I):** Các trường hợp âm tính thực tế nhưng bị mô hình phân loại sai thành dương tính.
    3.  **False Negatives (FN - Âm tính giả / Lỗi loại II):** Các trường hợp dương tính thực tế nhưng bị mô hình phân loại sai thành âm tính.
    4.  **True Positives (TP - Dương tính đúng):** Các trường hợp dương tính thực tế và được mô hình phân loại đúng là dương tính.

*   **Giải thích trực quan dựa trên hình ảnh (Hình 3-3):**
    
    \\[\text{Hình 3-3: Sơ đồ minh họa Ma trận nhầm lẫn}\\]
    
    Để hiểu trực quan, tài liệu cung cấp **Hình 3-3** phân tích cụ thể các phân vùng dự đoán trên tập dữ liệu số viết tay:
    *   **Nửa hàng trên (Actual Negative):** Là các chữ số thực tế **không phải là số 5** (gồm hình các chữ số 8, 7, 3, 9, 2).
        *   **Góc trên bên trái (TN):** Các chữ số 8, 7, 3, 9, 2 được dự đoán đúng là "Không phải 5".
        *   **Góc trên bên phải (FP - Lỗi loại I):** Một chữ số 6 viết ngoằn ngoèo bị mô hình đoán sai thành "Số 5" (nằm trong vùng Dương tính giả).
    *   **Nửa hàng dưới (Actual Positive):** Là các chữ số thực tế **là số 5**.
        *   **Góc dưới bên trái (FN - Lỗi loại II):** Hai chữ số 5 viết mờ hoặc xấu bị mô hình bỏ sót và dự đoán nhầm thành "Không phải 5".
        *   **Góc dưới bên phải (TP):** Ba chữ số 5 viết tương đối rõ ràng được mô hình nhận diện chính xác là "Số 5".

*   **Ví dụ thực tế trong tài liệu:**
    Để tính ma trận nhầm lẫn mà không làm ảnh hưởng đến tập kiểm thử (test set), tài liệu sử dụng hàm `cross_val_predict()` để tạo ra các dự đoán "sạch" (out-of-sample) trên tập huấn luyện. Đối với "Bộ phát hiện số 5" dùng mô hình `SGDClassifier`, kết quả ma trận nhầm lẫn thu được là:
    *   **53.892** ảnh không phải số 5 được phân loại đúng là không phải số 5 (**TN**).
    *   **687** ảnh không phải số 5 bị đoán sai là số 5 (**FP**).
    *   **1.891** ảnh số 5 bị bỏ sót và đoán sai thành không phải số 5 (**FN**).
    *   **3.530** ảnh số 5 được nhận diện chính xác (**TP**).
    
    *Lưu ý:* Một bộ phân loại hoàn hảo sẽ có đường chéo phụ bằng 0 (tức FP = FN = 0).

*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix

    # 1. Tạo các dự đoán chéo sạch (out-of-sample) trên tập huấn luyện
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

    # 2. Xuất ma trận nhầm lẫn
    cm = confusion_matrix(y_train_5, y_train_pred)
    print("Ma trận nhầm lẫn thực tế:\n", cm)
    # Kết quả:
    # [
    #  [ 1891,  3530]]

    # 3. Minh họa ma trận nhầm lẫn của một mô hình hoàn hảo giả định
    y_train_perfect_predictions = y_train_5
    perfect_cm = confusion_matrix(y_train_5, y_train_perfect_predictions)
    print("Ma trận nhầm lẫn hoàn hảo giả định:\n", perfect_cm)
    # Kết quả:
    # [
    #  [    0,  5421]]
    ```

---

### 2. Độ chính xác trên dự đoán dương tính (Precision)

*   **Giải thích bản chất:**
    **Precision (Độ chính xác của các dự đoán dương tính)** đo lường mức độ tin cậy khi mô hình đưa ra quyết định dự báo một mẫu thuộc lớp tích cực. Nó trả lời cho câu hỏi: *"Trong số tất cả các trường hợp mô hình gán nhãn là Dương tính (Positive), có bao nhiêu trường hợp thực sự đúng?"*
*   **Công thức Toán học (Phương trình 3-1):**
    
    \\[\text{Precision} = \frac{TP}{TP + FP}\\]
    
    *Trong đó: TP là số lượng dương tính đúng, FP là số lượng dương tính giả.*
*   **Ví dụ thực tế trong tài liệu:**
    Với bộ phát hiện số 5 ở trên, tỷ lệ dương tính đúng thực tế là: 
    
    \\[\text{Precision} = \frac{3530}{3530 + 687} \approx 83.7\% \quad\\]
    
    Điều này nghĩa là mỗi khi bộ phát hiện số 5 thông báo một hình ảnh là số 5, nó chỉ chính xác khoảng **83.7%** thời gian.
*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.metrics import precision_score

    # Tính toán điểm Precision trực tiếp từ Scikit-Learn
    precision = precision_score(y_train_5, y_train_pred)
    print("Precision của mô hình:", precision)
    # Kết quả: 0.8370879772350012
    ```

---

### 3. Độ nhạy / Độ triệu hồi (Recall / Sensitivity)

*   **Giải thích bản chất:**
    **Recall (Độ nhạy hay Tỷ lệ dương tính đúng - TPR)** đo lường khả năng tìm kiếm và bao phủ toàn bộ các mẫu dương tính thực tế của mô hình. Nó trả lời cho câu hỏi: *"Trong số tất cả các mẫu thực sự là Dương tính (Positive) có trong dữ liệu, mô hình đã tìm ra và phát hiện được bao nhiêu phần trăm?"*
*   **Công thức Toán học (Phương trình 3-2):**
    
    \\[\text{Recall} = \frac{TP}{TP + FN}\\]
    
    *Trong đó: TP là số lượng dương tính đúng, FN là số lượng âm tính giả (bỏ sót thực tế).*
*   **Ví dụ thực tế trong tài liệu:**
    Với bộ phát hiện số 5 ở trên, tỷ lệ bao phủ thực tế là:
    
    \\[\text{Recall} = \frac{3530}{3530 + 1891} \approx 65.1\% \quad\\]
    
    Nói cách khác, mô hình chỉ nhận diện ra được **65.1%** tổng số lượng chữ số 5 viết tay có trong toàn bộ tập dữ liệu huấn luyện, bỏ sót mất gần 35% còn lại.
*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.metrics import recall_score

    # Tính toán điểm Recall trực tiếp từ Scikit-Learn
    recall = recall_score(y_train_5, y_train_pred)
    print("Recall của mô hình:", recall)
    # Kết quả: 0.6511713705958311
    ```

---

### 4. Điểm F1 (F1-Score)

*   **Giải thích bản chất:**
    Để thuận tiện so sánh giữa các bộ phân loại khác nhau, chúng ta thường kết hợp Precision và Recall vào một chỉ số đánh giá duy nhất gọi là **Điểm F1 (F1-Score)**. 
    
    Điểm F1 được định nghĩa là **Trung bình điều hòa (Harmonic Mean)** của Precision và Recall. Trong khi trung bình cộng thông thường coi tất cả các giá trị như nhau, trung bình điều hòa lại **dành sự ưu tiên và kéo điểm số về phía giá trị thấp hơn**. Kết quả là, mô hình sẽ chỉ đạt được điểm F1 cao nếu **cả Precision và Recall đều đồng thời cao**.
*   **Công thức Toán học (Phương trình 3-3):**
    
    \\[F_1 = \frac{2}{\frac{1}{\text{Precision}} + \frac{1}{\text{Recall}}} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{TP}{TP + \frac{FN + FP}{2}}\\]
*   **Ví dụ thực tế trong tài liệu:**
    Trong ví dụ nhận diện số 5 của chúng ta, điểm F1-score đạt được là:
    
    \\[F_1 = 2 \times \frac{0.8370 \times 0.6511}{0.8370 + 0.6511} \approx 73.2\% \quad\\]
    
    *Lưu ý về ứng dụng thực tế:* Điểm F1 cao thiên vị các mô hình có Precision và Recall cân bằng. Tuy nhiên, tùy thuộc vào ngữ cảnh dự án thực tế, bạn không phải lúc nào cũng cần sự cân bằng này:
    *   **Ưu tiên Precision:** Ví dụ hệ thống lọc video an toàn cho trẻ em. Thà chấp nhận lọc nhầm nhiều video tốt (Recall thấp) còn hơn là để lọt một video độc hại (yêu cầu Precision cực cao).
    *   **Ưu tiên Recall:** Ví dụ hệ thống camera giám sát phát hiện kẻ trộm. Chấp nhận hệ thống báo động nhầm vài lần do gió thổi (Precision thấp) để đảm bảo không bỏ sót bất cứ một kẻ trộm thực sự nào đột nhập (yêu cầu Recall cực cao).
*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.metrics import f1_score

    # Tính toán điểm F1 trực tiếp từ Scikit-Learn
    f1 = f1_score(y_train_5, y_train_pred)
    print("Điểm F1-Score của mô hình:", f1)
    # Kết quả: 0.7325171197343846
    ```

---

# PHẦN 3: SỰ ĐÁNH ĐỔI PRECISION/RECALL & NGƯỠNG QUYẾT ĐỊNH

---

### 1. Điểm quyết định (Decision Score) & Ngưỡng quyết định (Decision Threshold)

*   **Giải thích bản chất:** 
    Để đưa ra quyết định phân loại nhị phân, mô hình `SGDClassifier` không trực tiếp gán nhãn ngay lập tức. Thay vào đó, đối với mỗi mẫu dữ liệu đầu vào, mô hình sẽ tính toán một giá trị điểm số thô gọi là **Điểm quyết định (Decision Score)** thông qua một **Hàm quyết định (Decision Function)**.
    
    Sau đó, mô hình sẽ so sánh điểm số này với một thước đo gọi là **Ngưỡng quyết định (Decision Threshold)**:
    *   Nếu **Điểm quyết định > Ngưỡng quyết định**: Mẫu được phân loại vào **Lớp dương tính (Positive Class)**.
    *   Nếu **Điểm quyết định \\(\le\\) Ngưỡng quyết định**: Mẫu được phân loại vào **Lớp âm tính (Negative Class)**.
    
    Theo mặc định, bộ phân loại `SGDClassifier` thiết lập ngưỡng quyết định bằng **`0`**.
*   **Ví dụ thực tế trong tài liệu:** 
    Scikit-Learn không cho phép người dùng thay đổi trực tiếp giá trị ngưỡng quyết định bên trong phương thức `.predict()`. Tuy nhiên, lập trình viên có thể truy cập điểm số thô này bằng cách gọi phương thức **`.decision_function()`**. 
    *   Với mẫu số 5 đầu tiên (`some_digit`), hàm quyết định trả về điểm số là **`2164.22`**.
    *   Nếu ta đặt ngưỡng quyết định bằng `0` (mặc định): Điểm \\(2164.22 > 0 \rightarrow\\) mô hình trả về dự đoán `True` (là số 5).
    *   Nếu ta chủ động tăng ngưỡng quyết định lên thành `3000`: Điểm \\(2164.22 \le 3000 \rightarrow\\) mô hình trả về dự đoán `False` (bỏ sót số 5 này). Điều này chứng minh rằng việc tăng ngưỡng quyết định sẽ làm giảm độ nhạy (Recall).

---

### 2. Sự đánh đổi giữa Precision và Recall (Precision/Recall Trade-off)

*   **Giải thích bản chất:** 
    Trong các bài toán phân loại, chúng ta luôn đối mặt với một quy luật bất biến: **bạn không thể đồng thời tối đa hóa cả Precision và Recall**. Khi bạn cố gắng điều chỉnh ngưỡng quyết định để tăng chỉ số này, chỉ số kia sẽ tự động giảm xuống. Hiện tượng này được gọi là **Sự đánh đổi Precision/Recall**.

*   **Giải thích trực quan dựa trên sơ đồ phân bổ (Hình 3-4):**
    
    \\[\text{Hình 3-4: Sơ đồ minh họa Sự đánh đổi Precision/Recall qua các ngưỡng quyết định}\\]
    
    Trong **Hình 3-4** của tài liệu, 12 chữ số được sắp xếp theo thứ tự tuyến tính tăng dần từ trái sang phải dựa trên điểm quyết định của chúng. Các chữ số thực tế là số 5 nằm rải rác. Tài liệu phân tích 3 kịch bản ngưỡng quyết định cụ thể để làm rõ sự đánh đổi này:
    
    1.  **Ngưỡng quyết định thấp (Mũi tên bên trái - giữa số 9 và số 5):**
        *   **Vùng dương tính dự đoán (bên phải ngưỡng):** Chứa 6 chữ số 5 thực tế, 1 chữ số 6 sai và 1 chữ số 2 sai (tổng cộng 8 hình).
        *   **Precision:** \\(\frac{6 \text{ (đúng)}}{8 \text{ (dự đoán positive)}} = \mathbf{75\%}\\).
        *   **Recall:** Mô hình tìm ra toàn bộ 6 chữ số 5 thực tế \\(\rightarrow \frac{6}{6} = \mathbf{100\%}\\).
    2.  **Ngưỡng quyết định trung tâm (Mũi tên ở giữa - giữa số 2 và số 5):**
        *   **Vùng dương tính dự đoán:** Chứa 4 chữ số 5 thực tế và 1 chữ số 6 sai (tổng cộng 5 hình).
        *   **Precision:** \\(\frac{4 \text{ (đúng)}}{5 \text{ (dự đoán positive)}} = \mathbf{80\%}\\).
        *   **Recall:** Mô hình phát hiện được 4 trên tổng số 6 chữ số 5 thực tế \\(\rightarrow \frac{4}{6} \approx \mathbf{67\%}\\).
    3.  **Ngưỡng quyết định cao (Mũi tên bên phải - giữa số 6 và số 5):**
        *   **Vùng dương tính dự đoán:** Chỉ chứa 3 chữ số 5 thực tế (tổng cộng 3 hình).
        *   **Precision:** Không mắc lỗi dương tính giả nào \\(\rightarrow \frac{3}{3} = \mathbf{100\%}\\).
        *   **Recall:** Chỉ phát hiện được một nửa số lượng số 5 thực tế \\(\rightarrow \frac{3}{6} = \mathbf{50\%}\\).
        
    *Nhận xét:* Khi dịch chuyển ngưỡng quyết định từ trái sang phải (tăng ngưỡng), **Precision tăng dần (từ 75% lên 100%)** nhưng **Recall sụt giảm nghiêm trọng (từ 100% xuống 50%)**.

---

### 3. Đồ thị Precision và Recall theo Ngưỡng quyết định (Hình 3-5)

*   **Giải thích bản chất:** 
    Để lựa chọn ngưỡng quyết định tối ưu cho từng dự án, chúng ta cần vẽ đồ thị biểu diễn giá trị của cả Precision và Recall dưới dạng các hàm số phụ thuộc vào giá trị Ngưỡng (Threshold).
    
*   **Giải thích trực quan dựa trên đồ thị (Hình 3-5):**
    
    \\[\text{Hình 3-5: Đồ thị đường cong Precision và Recall biến thiên theo Ngưỡng quyết định}\\]
    
    Từ **Hình 3-5** trong tài liệu, chúng ta quan sát thấy hai đường đặc trưng rất khác nhau:
    *   **Đường Recall (Nét liền màu xanh lá):** Luôn là một đường cong **mượt mà đi xuống** khi ngưỡng tăng. Điều này cực kỳ dễ hiểu vì ngưỡng càng cao thì điều kiện để được duyệt vào lớp dương tính càng khắt khe, dẫn đến việc bỏ sót mẫu tăng lên (Recall giảm liên tục).
    *   **Đường Precision (Đường đứt nét màu xanh dương):** Có xu hướng đi lên khi ngưỡng tăng, nhưng **đôi khi có những điểm răng cưa nhấp nhô (bumpy)**. 
        *   *Tại sao lại có hiện tượng nhấp nhô này?* Bản chất toán học là do khi ta tăng ngưỡng lên một chút, chúng ta có thể vô tình loại bỏ một mẫu Dương tính đúng (TP) trước khi loại bỏ được mẫu Dương tính giả (FP). Lúc này, tử số (TP) giảm nhanh hơn mẫu số (TP + FP), khiến điểm Precision bị sụt giảm tạm thời tại điểm đó, mặc dù xu hướng chung của nó vẫn là tăng.

---

### 4. Cách thiết lập mô hình đạt Precision mục tiêu (Ví dụ: Precision đạt 90%)

*   **Giải thích bản chất:**
    Nếu dự án của bạn yêu cầu một độ tin cậy cụ thể (ví dụ: bộ lọc thư rác cần **Precision đạt tối thiểu 90%** để tránh xóa nhầm thư quan trọng của người dùng), bạn có thể chủ động tìm ra ngưỡng tối thiểu đáp ứng yêu cầu này bằng cách sử dụng hàm `argmax()` của NumPy. 
    
    Hàm `argmax()` sẽ quét qua mảng điều kiện Boolean và trả về **chỉ mục (index) đầu tiên chứa giá trị `True`** (tương ứng với vị trí đầu tiên mà Precision vượt qua mốc 90%). Sau đó, ta dùng chỉ mục này để tra cứu ra giá trị Ngưỡng tương ứng.
*   **Ví dụ thực tế trong tài liệu:**
    Quá trình tìm kiếm tự động xác định ngưỡng tối thiểu để đạt 90% Precision là **`3370.02`**. Khi áp dụng ngưỡng này cho tập dữ liệu huấn luyện:
    *   **Precision đạt được:** **`90.00%`** (thỏa mãn mục tiêu).
    *   **Recall bị đánh đổi:** Sụt giảm xuống chỉ còn **`48.00%`**.

---

### 5. Mã nguồn Python minh họa chi tiết

Dưới đây là đoạn mã đầy đủ giúp bạn tính toán điểm quyết định, vẽ đường cong đánh đổi và tự định nghĩa ngưỡng quyết định tùy chỉnh:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

# 1. Lấy điểm quyết định (decision scores) "sạch" thay vì nhãn dự đoán trực tiếp
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, 
                             method="decision_function")

# 2. Tính toán các giá trị Precision, Recall tương ứng với mọi ngưỡng quyết định có thể
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# 3. Tìm chỉ mục và giá trị ngưỡng nhỏ nhất để Precision đạt ít nhất 90%
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]

print("Ngưỡng quyết định để đạt 90% Precision là:", threshold_for_90_precision)
# Kết quả: ~ 3370.02

# 4. Áp dụng ngưỡng mới để đưa ra dự đoán thủ công
y_train_pred_90 = (y_scores >= threshold_for_90_precision)

# 5. Kiểm tra thực tế các chỉ số đánh giá sau khi áp dụng ngưỡng tùy chỉnh
new_precision = precision_score(y_train_5, y_train_pred_90)
new_recall = recall_score(y_train_5, y_train_pred_90)

print(f"Precision mới: {new_precision:.2%}")  # Kết quả: 90.00%
print(f"Recall mới: {new_recall:.2%}")        # Kết quả: 48.00%

# 6. Mã vẽ đồ thị biến thiên Precision và Recall theo Ngưỡng (Hình 3-5)
plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold_for_90_precision, 0, 1.0, "k", "dotted", label="Ngưỡng đạt 90% Precision")
plt.xlabel("Ngưỡng quyết định (Threshold)")
plt.ylabel("Giá trị chỉ số")
plt.axis([-50000, 50000, 0, 1])
plt.grid(True)
plt.legend(loc="center right")
plt.show()
```

---

# PHẦN 4: ĐƯỜNG CONG ROC (RECEIVER OPERATING CHARACTERISTIC) & AUC

---

### 1. Đường cong ROC (Receiver Operating Characteristic Curve)

*   **Giải thích bản chất:** 
    **Đường cong đặc trưng hoạt động của bộ thu (ROC)** là một công cụ đồ họa phổ biến khác được thiết lập để đánh giá và lựa chọn bộ phân loại nhị phân. Đường cong này hoạt động rất giống với đường cong Precision/Recall, nhưng thay vì đặt mối quan hệ giữa Precision và Recall, đường cong ROC vẽ biểu diễn **Tỷ lệ dương tính đúng (True Positive Rate - TPR)** đối chiếu với **Tỷ lệ dương tính giả (False Positive Rate - FPR)**.
    
    Các thông số toán học cốt lõi cấu thành nên đường cong này bao gồm:
    *   **True Positive Rate (TPR):** Là tỷ lệ các mẫu dương tính thực tế được mô hình phát hiện chính xác. TPR thực chất là tên gọi khác của **Recall** (Độ nhạy - Sensitivity).
        
        \\[\text{TPR (Recall)} = \frac{TP}{TP + FN}\\]
        
    *   **False Positive Rate (FPR / Fall-out):** Là tỷ lệ các mẫu âm tính thực tế nhưng bị mô hình phân loại sai thành dương tính. FPR được tính bằng hiệu số của 1 trừ đi **Tỷ lệ âm tính đúng (True Negative Rate - TNR)**.
        
        \\[\text{FPR} = \frac{FP}{TN + FP} = 1 - \text{TNR}\\]
        
    *   **True Negative Rate (TNR / Specificity):** Là tỷ lệ các mẫu âm tính thực tế được mô hình gán nhãn chính xác là âm tính. Chỉ số này còn được gọi là **Độ đặc hiệu (Specificity)**.
    
    Vì lý do đó, đồ thị đường cong ROC phản ánh trực quan mối tương quan giữa **Độ nhạy (Recall/Sensitivity) ở trục tung** so với **\\(1 - \text{Độ đặc hiệu (1 - Specificity)}\\) ở trục hoành**.

*   **Giải thích trực quan dựa trên hình ảnh (Hình 3-7):**
    
    \\[\text{Hình 3-7: Đường cong ROC của bộ phân loại SGD trên bài toán phát hiện số 5}\\]
    
    Dựa trên **Hình 3-7** trong tài liệu, chúng ta thu được các phân tích trực quan sau:
    *   **Đường cong ROC thực tế (Đường nét liền màu xanh dương):** Thể hiện mối quan hệ đánh đổi thực tế của mô hình `SGDClassifier`. Khi ta cố gắng điều chỉnh để mô hình nhạy hơn (tăng TPR), mô hình sẽ tự động tạo ra nhiều lỗi dương tính giả hơn (tăng FPR).
    *   **Đường phân loại ngẫu nhiên (Đường chấm chéo màu đen từ góc dưới trái lên góc trên phải):** Đại diện cho một **bộ phân loại hoàn toàn ngẫu nhiên** (tung đồng xu). Một mô hình học máy hoạt động tốt phải có đường cong ROC **nằm càng xa đường chấm chéo này càng tốt**, hướng sát về phía góc trên cùng bên trái.
    *   **Vòng tròn đen nổi bật (Điểm ngưỡng đạt Precision 90%):** Điểm này tương ứng với ngưỡng quyết định đã chọn ở phần trước (khoảng `3370.02`), giúp mô hình đạt được Precision 90% và Recall 48%. Vị trí của điểm này trên đồ thị cho thấy tại đây, mô hình giữ được FPR ở mức cực kỳ thấp (tiệm cận sát trục tung), nghĩa là rất ít chữ số khác bị nhận nhầm thành số 5, đổi lại Recall của mô hình chỉ đạt dưới mức trung bình.

---

### 2. Diện tích dưới đường cong AUC (Area Under Curve)

*   **Giải thích bản chất:**
    Để định lượng và so sánh trực tiếp hiệu năng giữa các bộ phân loại khác nhau một cách nhanh chóng, chúng ta sử dụng số đo **Diện tích dưới đường cong ROC (ROC AUC Score)**. Chỉ số này tính toán toàn bộ phần diện tích nằm bên dưới đường cong ROC.
*   **Ý nghĩa điểm số:**
    *   **ROC AUC = \\(1.0\\):** Bộ phân loại **hoàn hảo**, đường cong ROC đi vuông góc lên sát góc trên bên trái.
    *   **ROC AUC = \\(0.5\\):** Bộ phân loại **hoàn toàn ngẫu nhiên**, đường cong ROC trùng khít với đường chấm chéo mặc định.
*   **Ví dụ thực tế trong tài liệu:**
    Khi tiến hành đo lường hiệu năng của bộ phát hiện số 5 dùng thuật toán SGD, mô hình đạt được điểm số ROC AUC tương đối ấn tượng là **`0.9605` (96.05%)**.

---

### 3. So sánh hiệu năng: SGDClassifier vs. RandomForestClassifier

*   **Sự khác biệt về phương thức dự đoán điểm số:**
    *   Mô hình tuyến tính `SGDClassifier` sử dụng điểm số thô được tính từ hàm quyết định `decision_function()` để so sánh với ngưỡng.
    *   Mô hình cây tổ hợp `RandomForestClassifier` **không có phương thức `decision_function()`** do cơ chế hoạt động đặc thù. Thay vào đó, lớp này cung cấp phương thức **`predict_proba()`**. Phương thức này trả về một ma trận chứa xác suất ước tính của từng mẫu thuộc về mỗi lớp. 
    *   *Cách giải quyết:* Chúng ta có thể trích xuất cột thứ hai (xác suất ước tính của lớp dương tính - tức là khả năng hình ảnh là số 5) để sử dụng làm điểm số quyết định thay thế và truyền vào các hàm đánh giá.

*   **Giải thích trực quan dựa trên sơ đồ so sánh (Hình 3-8):**
    
    \\[\text{Hình 3-8: So sánh đường cong PR (Precision/Recall) giữa Random Forest và SGD}\\]
    
    Để so sánh hai mô hình một cách khách quan nhất trên tập dữ liệu lệch, tài liệu vẽ đồng thời đường cong Precision/Recall của cả hai lên **Hình 3-8**:
    *   **Đường cong Random Forest (Nét liền màu xanh dương):** Nằm cao hơn, vượt trội hoàn toàn và ôm sát góc trên cùng bên phải hơn hẳn so với đường của SGD (đường nét đứt). Điều này chứng minh trực quan rằng Random Forest duy trì được độ tin cậy dự đoán (Precision) cực cao ngay cả khi ta yêu cầu mô hình truy quét và bao phủ phần lớn số 5 thực tế (Recall cao).

*   **Kết quả định lượng đối chiếu:**
    Sử dụng ngưỡng xác suất mặc định là **50%** để đưa ra dự đoán cho mô hình Random Forest, chúng ta có bảng so sánh hiệu năng vượt trội như sau:

| Chỉ số đánh giá | Bộ phân loại SGD (`SGDClassifier`) | Bộ phân loại Rừng ngẫu nhiên (`RandomForestClassifier`) |
| :--- | :---: | :---: |
| **Độ chính xác (Precision)** | ~ 83.7% | **~ 99.1%** |
| **Độ nhạy (Recall)** | ~ 65.1% | **~ 86.6%** |
| **Điểm F1 (F1-Score)** | ~ 73.25% | **~ 92.42%** |
| **ROC AUC Score** | ~ 96.05% | **~ 99.83%** |

---

### 4. Mã nguồn Python minh họa chi tiết

Đoạn mã dưới đây thực hiện huấn luyện mô hình `RandomForestClassifier`, trích xuất xác suất dự đoán, tính toán ROC AUC và vẽ đồ thị đối chiếu hiệu năng:

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score

# 1. Lấy điểm quyết định của SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
y_scores_sgd = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

# 2. Huấn luyện RandomForestClassifier và dự đoán mảng xác suất
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

# Trích xuất xác suất thuộc lớp dương tính (cột 1) để làm điểm quyết định
y_scores_forest = y_probas_forest[:, 1]

# 3. Tính toán các chỉ số cho đường cong ROC
fpr_sgd, tpr_sgd, _ = roc_curve(y_train_5, y_scores_sgd)
fpr_forest, tpr_forest, _ = roc_curve(y_train_5, y_scores_forest)

# 4. Tính toán điểm ROC AUC
auc_sgd = roc_auc_score(y_train_5, y_scores_sgd)
auc_forest = roc_auc_score(y_train_5, y_scores_forest)
print(f"ROC AUC của SGD Classifier: {auc_sgd:.4f}")       # Kết quả: ~ 0.9605
print(f"ROC AUC của Random Forest: {auc_forest:.4f}")      # Kết quả: ~ 0.9983

# 5. Đánh giá chi tiết Random Forest tại ngưỡng xác suất mặc định >= 50%
y_train_pred_forest = (y_scores_forest >= 0.5)
f1_forest = f1_score(y_train_5, y_train_pred_forest)
print(f"F1-Score của Random Forest: {f1_forest:.4f}")      # Kết quả: ~ 0.9242

# 6. Mã vẽ so sánh đường cong PR (Hình 3-8)
precisions_sgd, recalls_sgd, _ = precision_recall_curve(y_train_5, y_scores_sgd)
precisions_forest, recalls_forest, _ = precision_recall_curve(y_train_5, y_scores_forest)

plt.figure(figsize=(6, 5))
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label="Random Forest")
plt.plot(recalls_sgd, precisions_sgd, "g--", linewidth=2, label="SGD")
plt.xlabel("Recall (Độ nhạy)")
plt.ylabel("Precision (Độ tin cậy)")
plt.axis()
plt.grid(True)
plt.legend(loc="lower left")
plt.title("So sánh đường cong PR giữa Random Forest và SGD")
plt.show()
```

---
Rất vui được tiếp tục đồng hành cùng bạn để hoàn thiện chương này. Dưới đây là **Phần 5**, phần cuối cùng của cẩm nang chuyên sâu về **Chương 3: Phân loại (Classification)**, tập trung vào các chiến lược mở rộng phân loại nâng cao, phân tích lỗi sâu và các cấu trúc nhãn phức tạp.

---

# PHẦN 5: PHÂN LOẠI ĐA LỚP, PHÂN LOẠI ĐA NHÃN & PHÂN LOẠI ĐA ĐẦU RA

### 1. Phân loại đa lớp (Multiclass Classification)

*   **Giải thích bản chất:**
    Trong khi các bộ phân loại nhị phân chỉ phân biệt giữa hai lớp (như số 5 và không phải số 5), **bộ phân loại đa lớp** (hoặc bộ phân loại đa thức) có khả năng phân biệt giữa nhiều hơn hai lớp khác nhau. 
    
    Một số thuật toán hỗ trợ phân loại đa lớp một cách tự nhiên (như `RandomForestClassifier`, `LogisticRegression` hay `GaussianNB`). Ngược lại, một số thuật toán khác lại là bộ phân loại nhị phân nghiêm ngặt (như `SGDClassifier` hay `SVC` - Máy vector hỗ trợ). Để giải quyết các tác vụ đa lớp bằng thuật toán nhị phân, chúng ta sử dụng hai chiến lược chính:
    *   **Một-đối-phần-còn-lại (One-versus-Rest - OvR hoặc One-versus-All - OvA):** Huấn luyện \\(N\\) bộ phân loại nhị phân độc lập cho \\(N\\) lớp (ví dụ: bộ phát hiện số 0, bộ phát hiện số 1... bộ phát hiện số 9). Khi phân loại một mẫu mới, ta chạy mẫu đó qua toàn bộ \\(N\\) bộ phân loại, lấy điểm quyết định từ từng bộ và **chọn lớp có điểm số cao nhất**. Hầu hết các thuật toán phân loại nhị phân đều ưu tiên chiến lược này.
    *   **Một-đối-một (One-versus-One - OvO):** Huấn luyện một bộ phân loại nhị phân cho **mỗi cặp lớp** (ví dụ: bộ phân biệt 0 và 1, bộ phân biệt 0 và 2...). Nếu có \\(N\\) lớp, hệ thống cần huấn luyện tổng cộng **\\(\frac{N \times (N - 1)}{2}\\) bộ phân loại**. Với bài toán MNIST (10 lớp), điều này nghĩa là chúng ta phải huấn luyện tới **45 bộ phân loại** nhị phân! Khi dự đoán, mẫu dữ liệu sẽ được chạy qua tất cả 45 bộ phân loại để xem lớp nào giành được nhiều "chiến thắng" nhất.
        *   *Tại sao lại dùng OvO?* Điểm mạnh của OvO là mỗi bộ phân loại nhị phân chỉ cần huấn luyện trên phần dữ liệu nhỏ thuộc hai lớp mà nó phân biệt. Chiến lược này cực kỳ ưu việt đối với các thuật toán mở rộng kém với kích thước tập dữ liệu huấn luyện (như SVM).

*   **Cơ chế tự động của Scikit-Learn:**
    Scikit-Learn sẽ tự động nhận diện khi bạn truyền dữ liệu đa lớp vào một thuật toán nhị phân thuần túy, và **tự động áp dụng OvR hoặc OvO tùy thuộc vào đặc thù thuật toán**.
    *   Khi sử dụng `SVC` (SVM), Scikit-Learn tự động chạy chiến lược **OvO** bên dưới.
    *   Khi sử dụng `SGDClassifier`, Scikit-Learn tự động áp dụng chiến lược **OvR**.

*   **Mã nguồn Python minh họa:**
    ```python
    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    # 1. Huấn luyện SVM đa lớp (Scikit-Learn tự chạy OvO với 45 bộ phân loại nhị phân)
    # Để tối ưu thời gian chạy, ta chỉ huấn luyện trên 2.000 mẫu đầu tiên
    svm_clf = SVC(random_state=42)
    svm_clf.fit(X_train[:2000], y_train[:2000]) [cite: 46]

    # Dự đoán một mẫu cụ thể
    print("Dự đoán lớp của some_digit:", svm_clf.predict([some_digit])) # Kết quả: ['5'] [cite: 46, 47]

    # Xem 10 điểm quyết định tương ứng với 10 lớp mục tiêu
    some_digit_scores = svm_clf.decision_function([some_digit])
    print("Điểm số quyết định của các lớp:\n", some_digit_scores.round(2)) [cite: 47]
    # Lớp thắng nhiều trận đấu nhất sẽ có điểm cao nhất (~9.3 thuộc về lớp '5') [cite: 154, 155]

    # Xem danh sách các lớp lưu trong mô hình
    print("Danh sách lớp:", svm_clf.classes_) # Kết quả: ['0' '1' '2' '3' '4' '5' '6' '7' '8' '9'] [cite: 155]

    # 2. Buộc Scikit-Learn sử dụng một chiến lược cụ thể (ví dụ: Ép SVM chạy OvR thay vì OvO)
    ovr_clf = OneVsRestClassifier(SVC(random_state=42))
    ovr_clf.fit(X_train[:2000], y_train[:2000]) [cite: 49]
    print("Số lượng bộ ước lượng được huấn luyện dưới OvR:", len(ovr_clf.estimators_)) # Kết quả: 10 [cite: 49]

    # 3. Huấn luyện SGDClassifier đa lớp (mặc định dùng OvR) và áp dụng tỷ lệ đầu vào (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype("float64")) [cite: 50]
    
    # Kiểm định chéo để đánh giá độ chính xác (Accuracy tăng từ ~85.8% lên trên 89.1% nhờ chuẩn hóa)
    sgd_acc = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy") [cite: 50]
    print("Độ chính xác của SGD sau khi chuẩn hóa qua các folds:", sgd_acc)
    ```

---

### 2. Phân tích lỗi trực quan (Error Analysis)

*   **Giải thích bản chất:**
    Khi xây dựng một mô hình học máy thực tế, việc phân tích chi tiết các loại sai lệch (lỗi) mà mô hình mắc phải là bước đi quan trọng nhất để tìm hướng cải thiện hệ thống. Chúng ta thực hiện điều này bằng cách trực quan hóa nâng cao ma trận nhầm lẫn.

*   **Giải thích trực quan dựa trên các hình ảnh trong tài liệu:**

    *   **Ma trận nhầm lẫn thô và Ma trận nhầm lẫn chuẩn hóa theo hàng (Hình 3-9):**
        
        \\[\text{Hình 3-9: Ma trận nhầm lẫn thô (trái) và Ma trận nhầm lẫn chuẩn hóa theo hàng (phải)}\\]
        
        Trong **Hình 3-9**, biểu đồ bên trái hiển thị số lượng dự đoán thô. Đường chéo chính sáng rực rỡ thể hiện phần lớn các chữ số được phân loại chính xác. Tuy nhiên, để đánh giá khách quan, ta cần chuẩn hóa bằng tham số `normalize="true"` (chia mỗi ô cho tổng số mẫu thực tế của hàng đó) nhằm loại bỏ sự chênh lệch về kích cỡ mẫu giữa các lớp.
        
        Ở biểu đồ bên phải, ô giao lộ dòng 5 và cột 5 tối màu hơn rõ rệt so với các số khác. Tài liệu chỉ ra rằng **chỉ có 82% hình ảnh chữ số 5 thực tế được phân loại đúng**. Sai lầm lớn nhất của mô hình đối với số 5 là **nhận nhầm nó thành số 8** (chiếm tới 10% tổng số chữ số 5 thực tế).

    *   **Ma trận tập trung hiển thị lỗi (Hình 3-10):**
        
        \\[\text{Hình 3-10: Biểu đồ lỗi chuẩn hóa theo hàng (trái) và theo cột (phải)}\\]
        
        Để các lỗi phân loại hiển thị một cách nổi bật nhất, tài liệu sử dụng kỹ thuật đặt trọng số bằng `0` cho toàn bộ các dự đoán đúng (để trống đường chéo chính). 
        
        Nhìn vào **Hình 3-10**, cột số 8 sáng rực rỡ từ trên xuống dưới. Điều này xác nhận rằng **lỗi phổ biến nhất của hầu hết các lớp là bị phân loại sai thành số 8**. Biểu đồ bên phải (chuẩn hóa theo cột) còn cho thấy rõ một điểm nghẽn khác: có tới **56% số 7 bị phân loại sai thực chất lại là số 9**.
        
        *Hướng cải thiện:* Thu thập thêm dữ liệu của các chữ số dễ nhầm lẫn, hoặc thiết kế thêm các đặc trưng kỹ thuật mới như đếm số vòng lặp kín (số 8 có hai vòng, số 6 có một, số 5 không có).

    *   **Phân tích lỗi đơn lẻ của Số 3 và Số 5 (Hình 3-11):**
        
        \\[\text{Hình 3-11: Lưới hiển thị các chữ số 3 và 5 phân loại đúng và sai}\\]
        
        Trong **Hình 3-11**, tài liệu trực quan hóa một lưới gồm 4 phân vùng giao chéo giữa lớp thực tế và lớp dự đoán của số 3 và số 5:
        *   **Góc trên trái (Thực tế là 5, dự đoán là 3):** Những nét chữ 5 viết cẩu thả bị gán nhầm.
        *   **Góc dưới phải (Thực tế là 5, dự đoán là 5):** Nhận diện đúng.
        *   **Góc dưới trái (Thực tế là 3, dự đoán là 3):** Nhận diện đúng.
        *   **Góc trên phải (Thực tế là 3, dự đoán là 5):** Những số 3 viết lệch nét bị nhận nhầm.
        
        *Bản chất lỗi:* Vì thuật toán `SGDClassifier` chỉ là một mô hình tuyến tính đơn giản (phân bổ trọng số thô trên từng pixel rồi cộng dồn), nó cực kỳ **nhạy cảm với việc chữ số bị dịch chuyển hoặc xoay nhẹ**. Điểm khác biệt mấu chốt giữa số 3 và số 5 nằm ở vị trí của nét gạch nối nhỏ nối nét ngang trên cùng với cung tròn bên dưới. Nếu người viết vẽ nét nối này hơi lệch sang trái, mô hình tuyến tính sẽ nhầm số 3 thành số 5 ngay lập tức.
        
        *Giải pháp xử lý:* Thực hiện **Tăng cường dữ liệu (Data Augmentation)** bằng cách dịch chuyển và xoay nhẹ các hình ảnh huấn luyện gốc để dạy mô hình tính chống chịu với các biến thể chữ viết.

*   **Mã nguồn Python minh họa vẽ lỗi:**
    ```python
    from sklearn.metrics import ConfusionMatrixDisplay

    # 1. Vẽ ma trận lỗi chuẩn hóa theo hàng (Hình 3-10 bên trái)
    # Gán trọng số 0 cho các dự đoán chính xác để làm nổi bật lỗi
    sample_weight = (y_train_pred != y_train) [cite: 51]
    
    plt.rc('font', size=10)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, 
                                            sample_weight=sample_weight, 
                                            normalize="true", 
                                            values_format=\".0%\") [cite: 51]
    plt.title("Lỗi chuẩn hóa theo hàng (Sáng nhất nghĩa là sai nhiều nhất)")
    plt.show()
    ```

---

### 3. Phân loại đa nhãn (Multilabel Classification)

*   **Giải thích bản chất:**
    Là một hệ thống phân loại mà trong đó mô hình không chỉ gán một nhãn duy nhất cho mỗi thực thể, mà sẽ **xuất ra một tập hợp nhiều thẻ (nhãn) nhị phân đồng thời** cho mỗi trường hợp. 
*   **Ví dụ thực tế trong tài liệu:**
    Xây dựng một bộ phân loại đa nhãn nhận vào hình ảnh chữ số MNIST và dự đoán đồng thời hai thuộc tính:
    1.  **Chữ số đó có phải là số lớn không? (Lớn gồm 7, 8, 9)**
    2.  **Chữ số đó có phải là số lẻ không?**
    
    Khi đưa vào ảnh số 5, bộ phân loại đa nhãn sẽ trả về kết quả là mảng nhị phân hai phần tử: `[False, True]` (Không lớn nhưng là số lẻ).
*   **Đánh giá hiệu năng:**
    Chúng ta có thể tính điểm F1-score riêng cho từng nhãn nhị phân rồi lấy trung bình cộng. Sử dụng tham số `average="macro"` nếu coi các nhãn có tầm quan trọng ngang nhau, hoặc `average="weighted"` để tính trọng số đóng góp dựa trên tần suất xuất hiện (hỗ trợ) của từng lớp nhãn trong dữ liệu thực tế.
*   **Xử lý phụ thuộc nhãn (Classifier Chains):**
    Nếu sử dụng các mô hình không hỗ trợ đa nhãn tự nhiên (như SVM), bạn có thể huấn luyện các bộ phân loại nhị phân riêng biệt cho từng nhãn. Tuy nhiên, cách này bỏ qua mối quan hệ tương hỗ (ví dụ: số lớn thì khả năng lẻ cao hơn). Để khắc phục, ta sử dụng lớp **`ClassifierChain`** của Scikit-Learn để xếp các mô hình thành một chuỗi liên kết: khi đưa ra quyết định, mô hình phía sau sẽ tận dụng kết quả dự đoán của các mô hình phía trước làm đặc trưng đầu vào bổ sung.

*   **Mã nguồn Python minh họa:**
    ```python
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score
    from sklearn.multioutput import ClassifierChain

    # 1. Tạo tập nhãn đa mục tiêu (y_multilabel)
    y_train_large = (y_train >= '7') [cite: 55]
    y_train_odd = (y_train.astype('int8') % 2 == 1) [cite: 55]
    y_multilabel = np.c_[y_train_large, y_train_odd] # Ghép cột dữ liệu [cite: 55]

    # 2. Huấn luyện mô hình KNeighborsClassifier (Hỗ trợ đa nhãn tự nhiên)
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel) [cite: 55]

    # Kiểm tra dự đoán trên some_digit (số 5)
    print("Dự đoán đa nhãn cho some_digit:", knn_clf.predict([some_digit]))
    # Kết quả: array([[False,  True]]) [cite: 56]

    # 3. Đánh giá điểm F1 trung bình macro trên toàn bộ tập dữ liệu đa nhãn
    # (Lưu ý: Quá trình tính toán kiểm định chéo có thể mất vài phút)
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3) [cite: 56]
    macro_f1 = f1_score(y_multilabel, y_train_knn_pred, average="macro") [cite: 56]
    print(f"Điểm F1-Score (Macro) của mô hình đa nhãn: {macro_f1:.4f}") # Kết quả: ~ 0.9764 [cite: 56]

    # 4. Sử dụng chuỗi phân loại ClassifierChain với mô hình SVM làm nền tảng
    chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
    chain_clf.fit(X_train[:2000], y_multilabel[:2000]) [cite: 57]
    print("Dự đoán chuỗi nhãn cho some_digit:", chain_clf.predict([some_digit])) # Kết quả: [[0., 1.]] [cite: 57]
    ```

---

### 4. Phân loại đa đầu ra (Multioutput Classification)

*   **Giải thích bản chất:**
    Là dạng tổng quát hóa cao nhất của phân loại đa nhãn, trong đó **mỗi nhãn trong tập hợp nhãn đa mục tiêu không còn là nhị phân (True/False) mà là một biến đa lớp** (tức là có thể nhận nhiều hơn hai giá trị khác nhau).
*   **Ví dụ thực tế trong tài liệu:**
    Hệ thống **loại bỏ nhiễu cho hình ảnh MNIST (Denoising System)**.
    *   **Đầu vào:** Một chữ số bị làm mờ bởi nhiễu hạt ngẫu nhiên.
    *   **Đầu ra:** Một hình ảnh chữ số được khôi phục sạch sẽ.
    *   *Tại sao là đa đầu ra?* Đầu ra là tập hợp của **784 nhãn (tương ứng với 784 pixel)** trong bức ảnh \\(28 \times 28\\). Mỗi nhãn (mỗi pixel) lại nhận giá trị cường độ sáng chạy từ `0` đến `255` (một biến đa lớp với 256 giá trị có thể).

*   **Giải thích trực quan dựa trên hình ảnh (Hình 3-12 & Hình 3-13):**
    
    \\[\text{Hình 3-12: Ảnh đầu vào bị nhiễu (trái) và Ảnh mục tiêu sạch cần phục hồi (phải)}\\]
    
    \\[\text{Hình 3-13: Kết quả thực tế sau khi được mô hình KNN làm sạch}\\]
    
    Trong **Hình 3-12**, ảnh bên trái chứa số 7 bị phủ một lớp nhiễu thô rậm rạp được tạo ra từ hàm `np.random.randint()`. Ảnh bên phải là ảnh gốc sạch sẽ làm mục tiêu. 
    
    Khi đưa ảnh nhiễu này qua mô hình `KNeighborsClassifier` đa đầu ra, mô hình phân tích mối tương quan láng giềng pixel để đưa ra quyết định cường độ mới cho cả 784 pixel đồng thời. Kết quả đầu ra ở **Hình 3-13** cho thấy chữ số 7 được tái tạo sắc nét và hoàn toàn sạch nhiễu, gần như tương đồng hoàn hảo với ảnh mục tiêu gốc!

*   **Mã nguồn Python minh họa:**
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Tạo tập dữ liệu nhiễu (X_train_mod) và nhãn mục tiêu sạch (y_train_mod)
    np.random.seed(42)
    noise_train = np.random.randint(0, 100, (len(X_train), 784)) [cite: 57]
    noise_test = np.random.randint(0, 100, (len(X_test), 784)) [cite: 57]
    
    X_train_mod = X_train + noise_train [cite: 57]
    X_test_mod = X_test + noise_test [cite: 57]
    y_train_mod = X_train [cite: 57]
    y_test_mod = X_test [cite: 57]

    # 2. Huấn luyện bộ phân loại KNN đa đầu ra
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_mod, y_train_mod) [cite: 58]

    # 3. Làm sạch một bức ảnh bị nhiễu từ tập kiểm thử
    some_noisy_digit = X_test_mod
    cleaned_digit = knn_clf.predict([some_noisy_digit]) [cite: 58]

    # 4. Trực quan hóa đối chiếu kết quả (Hình 3-12 & Hình 3-13)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    
    # Vẽ ảnh nhiễu đầu vào
    axs.imshow(some_noisy_digit.reshape(28, 28), cmap="binary")
    axs.set_title("Ảnh bị nhiễu")
    axs.axis("off")
    
    # Vẽ ảnh mục tiêu sạch gốc
    axs.imshow(y_test_mod.reshape(28, 28), cmap="binary")
    axs.set_title("Mục tiêu sạch")
    axs.axis("off")
    
    # Vẽ ảnh do mô hình làm sạch
    axs.imshow(cleaned_digit.reshape(28, 28), cmap="binary")
    axs.set_title("Ảnh mô hình khôi phục")
    axs.axis("off")
    
    plt.show()
    ```

---

### KẾT LUẬN CHƯƠNG 3

Thông qua 5 phần chi tiết của cẩm nang, chúng ta đã đi trọn vẹn hành trình của bài toán Phân loại (Classification):
1.  **Xây dựng nền tảng** với bộ dữ liệu chuẩn mực MNIST và mô hình phân loại nhị phân `SGDClassifier`.
2.  **Làm chủ các thước đo hiệu năng thực tế** (Confusion Matrix, Precision, Recall, F1-score) để không bao giờ bị đánh lừa bởi chỉ số Accuracy trên các tập dữ liệu lệch.
3.  **Thấu hiểu bản chất sự đánh đổi Precision/Recall** để linh hoạt cấu trúc Ngưỡng quyết định tùy thuộc vào mục tiêu nghiệp vụ.
4.  **Sử dụng đường cong ROC & chỉ số AUC** làm bệ phóng so sánh hiệu năng các mô hình nhị phân một cách chuẩn xác.
5.  **Mở rộng biên giới phân loại** sang đa lớp, đa nhãn và đa đầu ra để xử lý các bài toán phức tạp trong thế giới thực.

---
