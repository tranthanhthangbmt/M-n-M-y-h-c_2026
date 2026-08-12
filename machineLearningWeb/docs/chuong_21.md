<!-- tabs:start -->

#### ** 📖 Lý thuyết **
# PHỤ LỤC B. TỰ ĐỘNG VI PHÂN (AUTODIFF)

Phụ lục này giải thích cách hoạt động của tính năng tự động vi phân
(autodiff) của TensorFlow, và cách nó so sánh với các giải pháp khác.


Giả sử bạn định nghĩa một hàm 

 , và bạn cần các đạo hàm
riêng 

 và 

 , thường là để thực hiện tối
ưu hóa gradient descent (hoặc một thuật toán tối ưu hóa khác). Các lựa chọn
chính của bạn là vi phân thủ công, xấp xỉ sai phân hữu hạn, autodiff chế độ
xuôi (forward-mode autodiff), và autodiff chế độ ngược (reverse-mode autodiff).
TensorFlow triển khai autodiff chế độ ngược, nhưng để hiểu nó, hữu ích khi xem
xét các lựa chọn khác trước. Vì vậy, hãy đi qua từng lựa chọn, bắt đầu với vi
phân thủ công.



### Vi Phân Thủ Công (Manual Differentiation)

Cách tiếp cận đầu tiên để tính đạo hàm là cầm bút chì và một tờ giấy
và sử dụng kiến thức giải tích của bạn để tìm ra phương trình thích hợp. Đối với
hàm 

 vừa định nghĩa, không quá
khó; bạn chỉ cần sử dụng năm quy tắc:


·        
Đạo hàm của một hằng số là 0.


·    
Đạo hàm của 

 là 

 (trong đó 

 là một hằng số).


·        
Đạo hàm của 

 là 

 , vì vậy đạo hàm của 

 là 

 .


·        
Đạo hàm của tổng các hàm là tổng
các đạo hàm của các hàm đó.


·    
Đạo hàm của 

 nhân với một hàm là 

 nhân với đạo hàm của nó.


Từ các quy tắc này, bạn có thể tìm ra Phương
trình B-1.


Phương trình B-1. Các đạo hàm riêng của


Cách tiếp cận này có thể trở nên rất tẻ nhạt đối
với các hàm phức tạp hơn, và bạn có nguy cơ mắc lỗi. May mắn thay, có những lựa
chọn khác. Bây giờ chúng ta hãy xem xét xấp xỉ sai phân hữu hạn.



### Xấp Xỉ Sai Phân Hữu Hạn (Finite
Difference Approximation)

Nhắc lại rằng đạo hàm 

 của một hàm 

 tại một điểm 

 là độ dốc của hàm tại điểm
đó. Chính xác hơn, đạo hàm được định nghĩa là giới hạn của độ dốc của một đường
thẳng đi qua điểm 

 này và một điểm 

 khác trên hàm, khi 

 tiến gần vô hạn đến 

 (xem Phương trình B-2).


Phương trình B-2. Định nghĩa đạo hàm của một hàm 

 tại điểm


Vì vậy, nếu chúng ta muốn tính đạo hàm riêng của 

 đối với 

 tại 

 và 

 , chúng ta có thể tính 

 và chia kết quả cho 

 , sử dụng một giá trị rất nhỏ
cho 

 . Loại xấp xỉ số học này của
đạo hàm được gọi là xấp xỉ sai phân hữu hạn (finite difference
approximation), và phương trình cụ thể này được gọi là thương số sai
phân của Newton (Newton’s difference quotient). Đó chính xác là những gì mã
sau đây thực hiện:



```python
def f(x, y):
    return x**2
* y + y + 2

def derivative(f, x, y, x_eps, y_eps):
    # Đây là một
hàm helper để minh họa ý tưởng, không phải cách tính đạo hàm chính xác
    # Nếu x_eps
và y_eps đều khác 0, nó tính đạo hàm theo hướng chéo, không phải đạo hàm riêng
    # Để tính đạo
hàm riêng, một trong x_eps hoặc y_eps phải là 0
    return (f(x
+ x_eps, y + y_eps) - f(x, y)) / (x_eps + y_eps)

# Để tính đạo hàm riêng theo x:
# x_eps = một số rất nhỏ, y_eps = 0
df_dx = (f(3 + 0.00001, 4) - f(3, 4)) / 0.00001
# Để tính đạo hàm riêng theo y:
# x_eps = 0, y_eps = một số rất nhỏ
df_dy = (f(3, 4 + 0.00001) - f(3, 4)) / 0.00001
```

Thật không may, kết quả không chính xác (và nó trở
nên tệ hơn đối với các hàm phức tạp hơn). Các kết quả chính xác lần lượt là 24
và 10, nhưng thay vào đó chúng ta nhận được:



```python
>>> df_dx
24.000039999805264
>>> df_dy 10.000000000331966
```

Lưu ý rằng để tính cả hai đạo hàm riêng, chúng ta
phải gọi hàm f() ít nhất ba lần (chúng ta đã gọi nó bốn
lần trong mã trước, nhưng có thể tối ưu hóa). Nếu có 1.000 tham số, chúng ta sẽ
cần gọi f() ít nhất 1.001 lần. Khi bạn đang xử
lý các mạng nơ-ron lớn, điều này làm cho xấp xỉ sai phân hữu hạn trở nên quá
không hiệu quả.


Tuy nhiên, phương pháp này rất đơn giản để triển khai nên nó là một
công cụ tuyệt vời để kiểm tra xem các phương pháp khác có được triển khai đúng
cách hay không. Ví dụ, nếu nó không khớp với hàm bạn đã đạo hàm thủ công, thì
hàm của bạn có thể chứa lỗi.


Cho đến nay, chúng ta đã xem xét hai cách để tính gradient: sử dụng
vi phân thủ công và sử dụng xấp xỉ sai phân hữu hạn. Thật không may, cả hai đều
có những hạn chế nghiêm trọng khi huấn luyện một mạng nơ-ron quy mô lớn. Vì vậy,
hãy chuyển sang autodiff, bắt đầu với chế độ xuôi.



### Autodiff Chế Độ Xuôi (Forward-Mode
Autodiff)

Hình B-1 cho thấy cách hoạt động của autodiff chế độ xuôi trên một
hàm thậm chí còn đơn giản hơn, 

 . Đồ thị cho hàm đó được biểu
diễn ở bên trái. Sau autodiff chế độ xuôi, chúng ta có đồ thị ở bên phải, biểu
diễn đạo hàm riêng 

 (chúng ta cũng có thể tìm đạo
hàm riêng đối với 

 một cách tương tự).


Thuật toán sẽ đi qua đồ thị tính toán từ đầu vào đến đầu ra (do đó
có tên “chế độ xuôi”). Nó bắt đầu bằng cách lấy các đạo hàm riêng của các nút
lá. Nút hằng số (5) trả về hằng số 0, vì đạo hàm của một hằng số luôn là 0. Biến


 trả về hằng số 1 vì 

 , và biến 

 trả về hằng số 0 vì 

 (nếu chúng ta đang tìm đạo
hàm riêng đối với 

 , nó sẽ ngược lại).


Bây giờ chúng ta có tất cả những gì cần thiết để di chuyển lên đồ thị
đến nút nhân trong hàm 

 . Giải tích cho chúng ta biết
rằng đạo hàm của tích của hai hàm 

 và 

 là 

 . Do đó, chúng ta có thể xây
dựng một phần lớn của đồ thị ở bên phải, biểu diễn 

 .


Cuối cùng, chúng ta có thể đi lên nút cộng trong hàm 

 . Như đã đề cập, đạo hàm của
tổng các hàm là tổng các đạo hàm của các hàm đó, vì vậy chúng ta chỉ cần tạo một
nút cộng và kết nối nó với các phần của đồ thị mà chúng ta đã tính toán. Chúng
ta nhận được đạo hàm riêng chính xác: 

 .


Hình B-1. Autodiff chế độ
xuôi.


Tuy nhiên, phương trình này có thể được đơn giản hóa (rất nhiều). Bằng
cách áp dụng một vài bước cắt tỉa đồ thị tính toán để loại bỏ tất cả các phép
toán không cần thiết, chúng ta có được một đồ thị nhỏ hơn nhiều chỉ với một
nút: 

 . Trong trường hợp này, việc
đơn giản hóa khá dễ dàng, nhưng đối với một hàm phức tạp hơn, autodiff chế độ
xuôi có thể tạo ra một đồ thị khổng lồ mà có thể khó đơn giản hóa và dẫn đến hiệu
suất không tối ưu.


Lưu ý rằng chúng ta bắt đầu với một đồ thị tính toán, và autodiff chế
độ xuôi tạo ra một đồ thị tính toán khác. Đây được gọi là vi phân ký hiệu
(symbolic differentiation), và nó có hai tính năng hay: thứ nhất, một khi đồ
thị tính toán của đạo hàm đã được tạo ra, chúng ta có thể sử dụng nó bao nhiêu
lần tùy thích để tính đạo hàm của hàm đã cho với bất kỳ giá trị nào của 

 và 

 ; thứ hai, chúng ta có thể chạy
lại autodiff chế độ xuôi trên đồ thị kết quả để nhận được đạo hàm bậc hai nếu
chúng ta cần (tức là đạo hàm của đạo hàm). Chúng ta thậm chí có thể tính đạo
hàm bậc ba, v.v.


Nhưng cũng có thể chạy autodiff chế độ xuôi mà không cần xây dựng đồ
thị (tức là theo số học, không phải ký hiệu), chỉ bằng cách tính toán các kết
quả trung gian ngay lập tức. Một cách để làm điều này là sử dụng số kép
(dual numbers), là những số kỳ lạ nhưng hấp dẫn có dạng 

 , trong đó 

 và 

 là số thực và 

 là một số vô cùng nhỏ sao cho


 (nhưng 

 ). Bạn có thể nghĩ số kép 

 như một cái gì đó tương tự 

 với vô số số 0 (nhưng tất
nhiên đây chỉ là đơn giản hóa để bạn có ý tưởng về số kép). Một số kép được biểu
diễn trong bộ nhớ dưới dạng một cặp số float. Ví dụ, 

 được biểu diễn bởi cặp 

 .


Số kép có thể được cộng, nhân, v.v., như được thể hiện trong Phương
trình B-3.


Phương trình B-3. Một vài phép toán với số kép


Công thức B-3: Một vài phép toán với số kép


·


Quan trọng nhất, có thể chứng minh rằng 

 . Do đó, việc tính 

 sẽ cho bạn cả 

 và đạo hàm 

 chỉ trong một lần tính.


Điều này cho
phép tính đạo hàm riêng của một hàm. Ví dụ, để tính đạo hàm riêng của 

 theo 

 tại 

 và 

 (ký hiệu là 

 ), chúng ta có thể sử dụng số kép. Tất cả những
gì chúng ta cần làm là tính 

 ; kết quả sẽ là một số kép có thành phần đầu
tiên bằng với 

 và thành phần thứ hai bằng với 

 .


Hình B-2. Autodiff chế độ
xuôi sử dụng số kép.


Để tính 

 , chúng ta sẽ phải đi qua đồ
thị một lần nữa, nhưng lần này với 

 và 

 .


Vì vậy, autodiff chế độ xuôi chính xác hơn nhiều so với xấp xỉ sai
phân hữu hạn, nhưng nó mắc phải cùng một lỗi lớn, ít nhất là khi có nhiều đầu
vào và ít đầu ra (như trường hợp xử lý mạng nơ-ron): nếu có 1.000 tham số, nó sẽ
yêu cầu 1.000 lần đi qua đồ thị để tính tất cả các đạo hàm riêng. Đây là nơi
autodiff chế độ ngược phát huy tác dụng: nó có thể tính tất cả chúng chỉ trong
hai lần đi qua đồ thị. Hãy xem cách thực hiện.



### Autodiff Chế Độ Ngược (Reverse-Mode
Autodiff)

Autodiff chế độ ngược là giải pháp được TensorFlow triển khai. Nó đầu
tiên đi qua đồ thị theo hướng xuôi (tức là từ đầu vào đến đầu ra) để tính giá
trị của mỗi nút. Sau đó, nó thực hiện một lần đi qua thứ hai, lần này theo hướng
ngược lại (tức là từ đầu ra đến đầu vào), để tính tất cả các đạo hàm riêng. Tên
“chế độ ngược” xuất phát từ lần đi qua thứ hai này qua đồ thị, nơi các gradient
chảy theo hướng ngược lại. Hình B-3 biểu diễn lần đi qua thứ hai. Trong lần đi
qua đầu tiên, tất cả các giá trị nút đã được tính toán, bắt đầu từ 

 và 

 . Bạn có thể thấy các giá trị
đó ở phía dưới bên phải của mỗi nút (ví dụ: 

 ). Các nút được dán nhãn 

 đến 

 để rõ ràng. Nút đầu ra là 

 : 

 .


Hình B-3. Autodiff chế độ
ngược.


Ý tưởng là dần dần đi xuống đồ thị, tính đạo hàm riêng của 

 đối với mỗi nút liên tiếp,
cho đến khi chúng ta đạt đến các nút biến. Đối với điều này, autodiff chế độ
ngược dựa rất nhiều vào quy tắc chuỗi (chain rule), được thể hiện trong
Phương trình B-4.


Công thức B-4: Quy tắc chuỗi


Vì 

 là nút đầu ra, nên 

 , do đó 

 .


Hãy tiếp tục đi xuống đồ thị: 

 thay đổi như thế nào khi 

 thay đổi? Câu trả lời là 

 . Chúng ta đã biết 

 . Vì 

 chỉ đơn giản là thực hiện
phép tổng 

 , nên ta thấy 

 . Do đó, 

 .


Bây giờ chúng ta có thể tiếp tục đến nút 

 : 

 thay đổi bao nhiêu khi 

 thay đổi? Câu trả lời là 

 . Vì 

 , nên ta thấy 

 . Do đó, 

 .


Quá trình này tiếp tục cho đến khi chúng ta đi đến đáy của đồ thị. Tại
thời điểm đó, chúng ta đã tính tất cả các đạo hàm riêng của 

 tại điểm 

 và 

 . Trong ví dụ này, chúng ta
tìm thấy 

 và 

 . Nghe có vẻ đúng!


Quá trình này tiếp tục cho đến khi chúng ta đạt đến đáy của đồ thị.
Tại thời điểm đó, chúng ta sẽ đã tính toán tất cả các đạo hàm riêng của 

 tại điểm 

 và 

 . Trong ví dụ này, chúng ta
tìm thấy 

 và 

 . Nghe có vẻ đúng!


Autodiff chế độ ngược là một kỹ thuật rất mạnh mẽ và chính xác, đặc
biệt khi có nhiều đầu vào và ít đầu ra, vì nó chỉ yêu cầu một lần đi qua xuôi cộng
với một lần đi qua ngược cho mỗi đầu ra để tính tất cả các đạo hàm riêng cho tất
cả các đầu ra đối với tất cả các đầu vào. Khi huấn luyện mạng nơ-ron, chúng ta
thường muốn tối thiểu hóa hàm mất mát (loss), vì vậy chỉ có một đầu ra (hàm mất
mát), và do đó chỉ cần hai lần đi qua đồ thị để tính gradient. Autodiff chế độ
ngược cũng có thể xử lý các hàm không hoàn toàn khả vi, miễn là bạn yêu cầu nó
tính đạo hàm riêng tại các điểm khả vi.


Trong Hình B-3, các kết quả số học được tính toán ngay lập tức, tại
mỗi nút. Tuy nhiên, đó không phải chính xác những gì TensorFlow làm: thay vào
đó, nó tạo ra một đồ thị tính toán mới. Nói cách khác, nó triển khai autodiff
chế độ ngược ký hiệu (symbolic reverse-mode autodiff). Bằng cách này, đồ thị
tính toán để tính gradient của hàm mất mát đối với tất cả các tham số trong mạng
nơ-ron chỉ cần được tạo một lần, và sau đó nó có thể được thực thi lặp đi lặp lại,
bất cứ khi nào bộ tối ưu hóa cần tính gradient. Hơn nữa, điều này giúp có thể
tính toán các đạo hàm bậc cao hơn nếu cần.

#### ** 🎦 Slide Bài Giảng **
<object data="TaiLieu/slideML/Slide_ML_Chap21.pdf#view=FitH" type="application/pdf" class="pdf-container">
    <p>Trình duyệt của bạn không hỗ trợ xem PDF nhúng. <a href="TaiLieu/slideML/Slide_ML_Chap21.pdf" target="_blank">Nhấn vào đây để tải Slide Bài Giảng</a>.</p>
</object>
<p style="text-align: right;"><a href="TaiLieu/slideML/Slide_ML_Chap21.pdf" target="_blank" style="font-weight: bold; color: #1a73e8;">📥 Tải về Slide Bài Giảng (PDF)</a></p>

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