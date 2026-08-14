<!-- tabs:start -->

#### ** 📖 Lý thuyết **
# PHỤ LỤC A. DANH SÁCH KIỂM TRA DỰ ÁN HỌC
MÁY

Danh sách kiểm tra này có thể hướng dẫn bạn thực hiện các dự án học
máy của mình. Có tám bước chính:


1.     
Xác định vấn đề và nhìn vào bức
tranh lớn.


2.     
Lấy dữ liệu.


3.     
Khám phá dữ liệu để hiểu rõ
hơn.


4.     
Chuẩn bị dữ liệu để làm nổi bật
các mẫu dữ liệu tiềm ẩn cho thuật toán học máy.


5.     
Khám phá nhiều mô hình khác
nhau và lập danh sách rút gọn những mô hình tốt nhất.


6.     
Tinh chỉnh các mô hình của bạn
và kết hợp chúng thành một giải pháp tuyệt vời.


7.     
Trình bày giải pháp của bạn.


8.     
Khởi chạy, giám sát và bảo trì
hệ thống của bạn.


Rõ ràng, bạn nên thoải mái điều chỉnh danh sách
kiểm tra này cho phù hợp với nhu cầu của mình.



### Xác định Vấn đề và Nhìn vào Bức tranh lớn

·        
Xác định mục tiêu bằng các thuật
ngữ kinh doanh.


·        
Giải pháp của bạn sẽ được sử dụng
như thế nào?


·        
Các giải pháp/cách giải quyết
hiện tại (nếu có) là gì?


·        
Bạn nên xác định vấn đề này như
thế nào (có giám sát/không giám sát, trực tuyến/ngoại tuyến, v.v.)?


·        
Hiệu suất nên được đo lường như
thế nào?


·        
Biện pháp hiệu suất có phù hợp
với mục tiêu kinh doanh không?


·        
Hiệu suất tối thiểu cần thiết để
đạt được mục tiêu kinh doanh là gì?


·        
Có những vấn đề nào tương tự? Bạn
có thể tái sử dụng kinh nghiệm hoặc công cụ không?


·        
Có chuyên môn của con người
không?


·        
Bạn sẽ giải quyết vấn đề bằng
cách thủ công như thế nào?


·        
Liệt kê các giả định bạn (hoặc
người khác) đã đưa ra cho đến nay.


·        
Xác minh các giả định nếu có thể.



### Lấy Dữ liệu

Lưu ý: tự động hóa càng nhiều càng tốt để bạn có thể dễ dàng lấy dữ
liệu mới.


·        
Liệt kê dữ liệu bạn cần và số
lượng bạn cần.


·        
Tìm và ghi lại nơi bạn có thể lấy
dữ liệu đó.


·        
Kiểm tra dung lượng nó sẽ chiếm.


·        
Kiểm tra các nghĩa vụ pháp lý
và xin cấp phép nếu cần.


·        
Nhận quyền truy cập.


·        
Tạo không gian làm việc (với đủ
dung lượng lưu trữ).


·        
Lấy dữ liệu.


·        
Chuyển đổi dữ liệu sang định dạng
bạn có thể dễ dàng thao tác (mà không làm thay đổi dữ liệu).


·        
Đảm bảo thông tin nhạy cảm bị
xóa hoặc bảo vệ (ví dụ: ẩn danh).


·        
Kiểm tra kích thước và loại dữ
liệu (chuỗi thời gian, mẫu, địa lý, v.v.).


·        
Lấy mẫu một tập hợp thử nghiệm,
đặt nó sang một bên và không bao giờ nhìn vào nó (không rình mò dữ liệu!).



### Khám phá Dữ liệu

Lưu ý: cố gắng thu thập thông tin chuyên sâu từ một chuyên gia trong
lĩnh vực này cho các bước này.


·        
Tạo một bản sao của dữ liệu để
khám phá (lấy mẫu xuống kích thước dễ quản lý nếu cần).


·        
Tạo một sổ tay Jupyter để ghi lại
quá trình khám phá dữ liệu của bạn.


·        
Nghiên cứu từng thuộc tính và
các đặc điểm của nó:


o  
Tên


o  
Loại (phân loại, int/float, có
giới hạn/không giới hạn, văn bản, có cấu trúc, v.v.)


o  
% giá trị bị thiếu


o  
Độ nhiễu và loại nhiễu (ngẫu
nhiên, ngoại lệ, lỗi làm tròn, v.v.)


o  
Mức độ hữu ích cho tác vụ


o  
Loại phân phối (Gaussian, đồng
nhất, logarit, v.v.)


·        
Đối với các tác vụ học có giám
sát, xác định thuộc tính mục tiêu.


·        
Trực quan hóa dữ liệu.


·        
Nghiên cứu các mối tương quan
giữa các thuộc tính.


·        
Nghiên cứu cách bạn sẽ giải quyết
vấn đề bằng cách thủ công.


·        
Xác định các phép biến đổi tiềm
năng mà bạn có thể muốn áp dụng.


·        
Xác định dữ liệu bổ sung sẽ hữu
ích (quay lại phần “Lấy Dữ liệu”).


·        
Ghi lại những gì bạn đã học được.



### Chuẩn bị Dữ liệu

Lưu ý:


·        
Làm việc trên các bản sao của dữ
liệu (giữ nguyên tập dữ liệu gốc).


·        
Viết hàm cho tất cả các phép biến
đổi dữ liệu bạn áp dụng, vì năm lý do:


o  
Để bạn có thể dễ dàng chuẩn bị
dữ liệu vào lần tới khi bạn nhận được một tập dữ liệu mới.


o  
Để bạn có thể áp dụng các phép
biến đổi này trong các dự án tương lai.


o  
Để làm sạch và chuẩn bị tập kiểm
tra.


o  
Để làm sạch và chuẩn bị các
instance dữ liệu mới sau khi giải pháp của bạn hoạt động trực tiếp.


o  
Để dễ dàng coi các lựa chọn chuẩn
bị của bạn là các siêu tham số.


·        
Làm sạch dữ liệu:


o  
Sửa hoặc loại bỏ các giá trị
ngoại lệ (tùy chọn).


o  
Điền vào các giá trị bị thiếu
(ví dụ: bằng 0, trung bình, trung vị…) hoặc loại bỏ các hàng (hoặc cột) của
chúng.


·        
Thực hiện chọn tính năng (tùy
chọn):


o  
Loại bỏ các thuộc tính không
cung cấp thông tin hữu ích cho tác vụ.


·        
Thực hiện kỹ thuật tính năng
(feature engineering), khi thích hợp:


o  
Rời rạc hóa các tính năng liên
tục.


o  
Phân tích các tính năng (ví dụ:
phân loại, ngày/giờ, v.v.).


o   
Thêm các phép biến đổi tiềm
năng của tính năng (ví dụ: 

 , 

 , 

 , v.v.).


o  
Tổng hợp các tính năng thành
các tính năng mới đầy hứa hẹn.


·        
Thực hiện chia tỷ lệ tính năng
(feature scaling):


o  
Chuẩn hóa hoặc bình thường hóa
các tính năng.



### Lập danh sách rút gọn các Mô hình Tiềm
năng

Lưu ý:


·        
Nếu dữ liệu rất lớn, bạn có thể
muốn lấy mẫu các tập huấn luyện nhỏ hơn để bạn có thể huấn luyện nhiều mô hình
khác nhau trong một thời gian hợp lý (lưu ý rằng điều này làm giảm hiệu suất của
các mô hình phức tạp như mạng nơ-ron lớn hoặc rừng ngẫu nhiên).


·        
Một lần nữa, cố gắng tự động
hóa các bước này càng nhiều càng tốt.


·        
Huấn luyện nhiều mô hình “nhanh
chóng và sơ sài” từ các danh mục khác nhau (ví dụ: tuyến tính, naive Bayes,
SVM, rừng ngẫu nhiên, mạng nơ-ron, v.v.) bằng cách sử dụng các tham số tiêu chuẩn.


·        
Đo lường và so sánh hiệu suất của
chúng:


o  
Đối với mỗi mô hình, sử dụng
xác thực chéo N lần (N-fold cross-validation) và tính trung bình và độ lệch chuẩn
của biện pháp hiệu suất trên N lần.


·        
Phân tích các biến quan trọng
nhất cho mỗi thuật toán.


·        
Phân tích các loại lỗi mà các
mô hình mắc phải:


o  
Dữ liệu nào mà con người đã sử
dụng để tránh những lỗi này?


·        
Thực hiện một vòng chọn tính
năng và kỹ thuật tính năng nhanh chóng.


·        
Thực hiện thêm một hoặc hai lần
lặp nhanh chóng các bước trước đó.


·        
Lập danh sách rút gọn ba đến
năm mô hình tiềm năng nhất, ưu tiên các mô hình mắc các loại lỗi khác nhau.



### Tinh chỉnh Hệ thống

Lưu ý:


·        
Bạn sẽ muốn sử dụng càng nhiều
dữ liệu càng tốt cho bước này, đặc biệt là khi bạn tiến gần đến cuối quá trình
tinh chỉnh.


·        
Như mọi khi, tự động hóa những
gì bạn có thể.


·        
Tinh chỉnh các siêu tham số bằng
cách sử dụng xác thực chéo:


o  
Coi các lựa chọn biến đổi dữ liệu
của bạn là các siêu tham số, đặc biệt khi bạn không chắc chắn về chúng (ví dụ:
nếu bạn không chắc chắn nên thay thế các giá trị bị thiếu bằng số 0 hay bằng
giá trị trung vị, hoặc chỉ đơn giản là loại bỏ các hàng).


o  
Trừ khi có rất ít giá trị siêu
tham số để khám phá, hãy ưu tiên tìm kiếm ngẫu nhiên (random search) hơn tìm kiếm
lưới (grid search). Nếu quá trình huấn luyện rất dài, bạn có thể ưu tiên phương
pháp tối ưu hóa Bayes (ví dụ: sử dụng Gaussian process priors, như được mô tả bởi
Jasper Snoek et al.).


·        
Thử các phương pháp tập hợp
(ensemble methods). Kết hợp các mô hình tốt nhất của bạn thường sẽ mang lại hiệu
suất tốt hơn so với việc chạy chúng riêng lẻ.


·        
Khi bạn tin tưởng vào mô hình
cuối cùng của mình, hãy đo lường hiệu suất của nó trên tập kiểm tra để ước tính
lỗi khái quát hóa (generalization error).



### Trình bày Giải pháp của bạn

·        
Ghi lại những gì bạn đã làm.


·        
Tạo một bài thuyết trình đẹp mắt:


o  
Đảm bảo bạn làm nổi bật bức
tranh lớn trước tiên.


·        
Giải thích tại sao giải pháp của
bạn đạt được mục tiêu kinh doanh.


·        
Đừng quên trình bày những điểm
thú vị bạn nhận thấy trong quá trình:


o  
Mô tả những gì đã hoạt động và
những gì không.


o  
Liệt kê các giả định và các hạn
chế của hệ thống của bạn.


·        
Đảm bảo các phát hiện chính của
bạn được truyền đạt thông qua các hình ảnh trực quan đẹp mắt hoặc các câu dễ nhớ
(ví dụ: “thu nhập trung bình là yếu tố dự đoán số một về giá nhà ở”).



### Khởi chạy!

·        
Chuẩn bị giải pháp của bạn để sản
xuất (cắm vào đầu vào dữ liệu sản xuất, viết kiểm thử đơn vị, v.v.).


·        
Viết mã giám sát để kiểm tra hiệu
suất trực tiếp của hệ thống của bạn theo định kỳ và kích hoạt cảnh báo khi nó
giảm:


o  
Cẩn thận với sự xuống cấp chậm:
các mô hình có xu hướng “thối rữa” khi dữ liệu phát triển.


o  
Việc đo lường hiệu suất có thể
yêu cầu một quy trình của con người (ví dụ: thông qua dịch vụ crowdsourcing).


o  
Cũng giám sát chất lượng đầu
vào của bạn (ví dụ: một cảm biến trục trặc gửi các giá trị ngẫu nhiên, hoặc đầu
ra của một nhóm khác trở nên lỗi thời). Điều này đặc biệt quan trọng đối với
các hệ thống học trực tuyến (online learning systems).


·        
Huấn luyện lại các mô hình của
bạn một cách thường xuyên trên dữ liệu mới (tự động hóa càng nhiều càng tốt).


Phán đoán và Công thức Toán:


Phụ lục này là một danh sách kiểm tra dự án học máy, tập trung vào
quy trình và các bước thực hành hơn là các công thức toán học cụ thể. Do đó,
không có công thức toán học tường minh nào trong phần này.


Tuy nhiên, các khái niệm được đề cập có liên quan đến các nguyên tắc
toán học và thống kê cơ bản trong học máy:


·        
Đo lường Hiệu suất
(Performance Measurement):


o  
“How should performance be
measured?” và “Is the performance measure aligned with the business
objective?”: Điều này liên quan đến việc chọn các metrics phù hợp (ví dụ:
độ chính xác, độ chính xác, độ thu hồi, F1-score, RMSE, MAE, R-squared). Việc lựa
chọn metrics là một vấn đề thống kê và tối ưu hóa, đảm bảo rằng việc tối ưu hóa
mô hình cũng tối ưu hóa mục tiêu kinh doanh.


o  
“For each model, use N-fold
cross-validation and compute the mean and standard deviation of the performance
measure on the N folds”:


§  Cross-validation (Xác thực chéo): Một kỹ
thuật thống kê để ước tính hiệu suất của mô hình.


§        
Mean (Giá trị trung bình):


§        
Standard Deviation (Độ lệch
chuẩn): 

 Việc tính toán trung bình và
độ lệch chuẩn của các metrics trên các fold giúp đánh giá sự ổn định và độ tin
cậy của hiệu suất mô hình.


·        
Kỹ thuật Tính năng (Feature
Engineering):


o   
“Add promising transformations
of features (e.g., 

 , 

 , 

 , etc.)”: Đây là các phép biến
đổi toán học cơ bản áp dụng lên các đặc trưng để giúp thuật toán học máy học tốt
hơn từ dữ liệu. Ví dụ, 

 có thể được dùng để chuẩn hóa
các phân phối bị lệch, 

 tạo ra các đặc trưng bậc hai
để mô hình có thể học các mối quan hệ phi tuyến.


·        
Chia tỷ lệ Tính năng
(Feature Scaling):


o  
“Standardize or normalize
features”:


§        
Standardization (Chuẩn hóa
Z-score): 

 (trong đó 

 là trung bình, 

 là độ lệch chuẩn).


§        
Normalization (Min-Max
Scaling): 

 (chuẩn hóa về khoảng 

 ). Các phép biến đổi này là
các công thức toán học giúp đưa các đặc trưng về cùng một phạm vi, cải thiện hiệu
suất của nhiều thuật toán học máy.


·        
Tối ưu hóa Siêu Tham số
(Hyperparameter Tuning):


o  
“Prefer random search over grid
search. If training is very long, you may prefer a Bayesian optimization
approach…”: Đây là các chiến lược tìm kiếm siêu tham số, với Bayesian
optimization sử dụng các mô hình xác suất (ví dụ: Gaussian processes) để ước
tính hàm hiệu suất của siêu tham số, từ đó đưa ra lựa chọn thông minh hơn về
các siêu tham số để thử tiếp theo. Mặc dù không có công thức cụ thể, nó dựa
trên lý thuyết xác suất và tối ưu hóa.


·        
Lỗi Khái quát hóa
(Generalization Error):


o  
“measure its performance on the
test set to estimate the generalization error”: Đây là ước tính hiệu suất của
mô hình trên dữ liệu chưa từng thấy. Mục tiêu của học máy là tối thiểu hóa lỗi
khái quát hóa, thường được đo bằng các metrics đã chọn.


·        
Giám sát Hệ thống
(Monitoring):


o  
“Beware of slow degradation:
models tend to “rot” as data evolves.” (Khái niệm drift dữ liệu/mô hình)


o  
“Also monitor your inputs’
quality…”: Các khái niệm này liên quan đến việc giám sát các phân phối thống kê
của dữ liệu đầu vào và đầu ra theo thời gian để phát hiện sự thay đổi, điều này
thường yêu cầu các phép đo thống kê liên tục.


Tóm lại, mặc dù không có công thức toán học phức
tạp, toàn bộ phụ lục này được xây dựng dựa trên các nguyên tắc toán học, thống
kê và tối ưu hóa cốt lõi của học máy.

#### ** 🎦 Slide Bài Giảng **
<object data="TaiLieu/slideML/Slide_ML_Chap20.pdf#view=FitH" type="application/pdf" class="pdf-container">
    <p>Trình duyệt của bạn không hỗ trợ xem PDF nhúng. <a href="TaiLieu/slideML/Slide_ML_Chap20.pdf" target="_blank">Nhấn vào đây để tải Slide Bài Giảng</a>.</p>
</object>
<p style="text-align: right;"><a href="TaiLieu/slideML/Slide_ML_Chap20.pdf" target="_blank" style="font-weight: bold; color: #1a73e8;">📥 Tải về Slide Bài Giảng (PDF)</a></p>

#### ** 🎥 Video **
*Đang cập nhật...*

#### ** 📝 Trắc nghiệm **
*Đang cập nhật...*

#### ** 💻 Thực hành **

<div class="practice-container" style="background: #f8faff; border: 1px solid #cce0ff; border-radius: 8px; padding: 20px; margin-top: 15px;">
  <h3 style="margin-top:0; color: #1a73e8; display:flex; align-items:center; gap:8px;">🚀 Bài tập Thực hành Jupyter Notebook</h3>
  <p>Dưới đây là các sổ tay (notebook) chứa mã nguồn Python thực hành cho chương này. Bạn có thể mở trực tiếp trên Google Colab để chạy thử nghiệm, hoặc tải file về máy.</p>
  <p><em>Chưa có bài thực hành cụ thể cho chương này.</em></p>
  <div style="margin-top: 20px; border-top: 1px dashed #cce0ff; padding-top: 15px;">
    <strong>Hoặc truy cập toàn bộ kho tài liệu:</strong> <a href="https://drive.google.com/drive/folders/1nRV7W748VkSldg-BaKdcejBV-sBP47_M?usp=sharing" target="_blank" style="color: #1a73e8; font-weight: bold;">Thư mục Google Drive Thực hành</a>
  </div>
</div>

<!-- tabs:end -->
