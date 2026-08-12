\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}
\usepackage{fontspec}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{xcolor}

\usetheme{Madrid}
\usecolortheme{default}

% Configure listings for Python
\lstset{
    language=Python,
    basicstyle=\ttfamily\footnotesize,
    keywordstyle=\color{blue}\bfseries,
    stringstyle=\color{red},
    commentstyle=\color{green!60!black},
    numbers=left,
    numberstyle=\tiny\color{gray},
    stepnumber=1,
    frame=single,
    breaklines=true,
    breakatwhitespace=true,
    showstringspaces=false,
    tabsize=4,
    captionpos=b
}

\title[Chương 12]{Chương 12: Mô hình tùy chỉnh và Huấn luyện với TensorFlow}
\author{Giảng viên: TS. Trần Thành Thắng}
\institute{Đại học Đông Á}
\date{Năm học 2024-2025}

\begin{document}

% Slide 1
\begin{frame}
    \titlepage
\end{frame}

% Slide 2
\begin{frame}{Nội dung chương}
    \tableofcontents
\end{frame}

\section{Giới thiệu TensorFlow \& Sử dụng TensorFlow như NumPy}

% Slide 3
\begin{frame}{Giới thiệu nhanh về TensorFlow}
    \begin{itemize}
        \item TensorFlow là thư viện tính toán số học mãnh mẽ mã nguồn mở, đặc biệt phù hợp cho học máy.
        \item Cốt lõi của nó tương tự NumPy, nhưng thao tác với \textbf{Tensor} thay vì mảng.
        \item Lợi thế tuyệt đối: Hỗ trợ tính toán hiệu năng cao trên \textbf{GPU} (Graphics Processing Unit) và \textbf{TPU} (Tensor Processing Unit).
        \item Hỗ trợ tính toán phân tán trên hàng ngàn thiết bị và server.
        \item Cho phép biên dịch tự động các biểu thức tính toán Python thành Biểu đồ (Graph) C++ hiệu suất cao.
    \end{itemize}
\end{frame}

% Slide 4
\begin{frame}{Hệ sinh thái TensorFlow}
    \begin{itemize}
        \item TensorFlow không chỉ là một thư viện độc lập mà là trọng tâm của một hệ sinh thái lớn:
        \item \textbf{TensorBoard}: Trực quan hóa dữ liệu, đồ thị mô hình.
        \item \textbf{TensorFlow Hub}: Nơi tải xuống và sử dụng lại các mô hình đã huấn luyện (Pretrained models).
        \item \textbf{TensorFlow Extended (TFX)}: Bộ công cụ triển khai mô hình lên môi trường sản xuất (Production).
        \item \textbf{TF Lite \& TF.js}: Triển khai mô hình trên thiết bị di động, phần cứng nhúng và trình duyệt Web.
    \end{itemize}
\end{frame}

% Slide 5
\begin{frame}[fragile]{Tensor và Phép toán (Tensors and Operations)}
    \begin{itemize}
        \item API của TensorFlow chủ yếu xoay quanh các \textit{tensors} (cáp số liệu đa chiều).
        \item Cách dễ nhất để tạo tensor là dùng \texttt{tf.constant()}.
    \end{itemize}
    \begin{lstlisting}[language=Python]
import tensorflow as tf

# Tao tensor tu ma tran
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t.shape) # TensorShape([2, 3])
print(t.dtype) # <dtype: 'float32'>
    \end{lstlisting}
\end{frame}

% Slide 6
\begin{frame}[fragile]{Các hàm toán học cơ bản}
    \begin{itemize}
        \item TensorFlow cung cấp hàng loạt hàm toán học hoạt động trên tensors.
        \item Các toán tử chuẩn Python (\texttt{+}, \texttt{-}, \texttt{*}, \texttt{**}) được nạp chồng.
    \end{itemize}
    \begin{lstlisting}[language=Python]
# Tinh toan co ban
t1 = t + 10                 # Phep cong broadcasting
t2 = tf.square(t)           # Binh phuong tung phan tu
t3 = t @ tf.transpose(t)    # Nhan ma tran (hoac dung tf.matmul)
    \end{lstlisting}
    \begin{itemize}
        \item Ký hiệu \texttt{@} xuất hiện từ Python 3.5 để thay thế cho phép nhân ma trận.
    \end{itemize}
\end{frame}

% Slide 7
\begin{frame}[fragile]{Tensors và NumPy: Tương tác qua lại}
    \begin{itemize}
        \item Tensors tương tác rất thân thiện với thư viện NumPy.
        \item Bạn có thể tạo tensor từ NumPy array và ngược lại.
    \end{itemize}
    \begin{lstlisting}[language=Python]
import numpy as np

# Numpy -> Tensor
a = np.array([2., 4., 5.])
t_a = tf.constant(a)

# Tensor -> Numpy
a_back = t_a.numpy()        # Hoac dung np.array(t_a)

# Tinh toan cheo
t_b = tf.square(a)          # TF dung du lieu Numpy
a_b = np.square(t_a)        # Numpy dung du lieu TF
    \end{lstlisting}
\end{frame}

% Slide 8
\begin{frame}[fragile]{Chuyển đổi kiểu dữ liệu (Type Conversions)}
    \begin{itemize}
        \item Trái với NumPy, TensorFlow \textbf{không tự động} ép kiểu (type conversion) để tránh làm giảm hiệu năng ẩn.
        \item Nếu cộng \texttt{float32} và \texttt{float64}, hoặc \texttt{float32} và \texttt{int32}, TensorFlow sẽ văng lỗi thay vì ép kiểu ngầm.
    \end{itemize}
    \begin{lstlisting}[language=Python]
# tf.constant(2.0) + tf.constant(40) # --> Loi TypeError

# Can phai ep kieu ro rang bang tf.cast:
t2 = tf.constant(40., dtype=tf.float64)
result = tf.constant(2.0) + tf.cast(t2, tf.float32)
    \end{lstlisting}
\end{frame}

% Slide 9
\begin{frame}[fragile]{Biến số (Variables)}
    \begin{itemize}
        \item Các giá trị \texttt{tf.constant()} không thể bị thay đổi. Tuy nhiên, trọng số (weights) của mô hình thì cần được cập nhật qua mỗi vòng lặp.
        \item Ta cần cấu trúc \texttt{tf.Variable}.
    \end{itemize}
    \begin{lstlisting}[language=Python]
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])

# Cap nhat toan bo
v.assign(2 * v)

# Cap nhat phan tu
v[0, 1].assign(42)

# Cap nhat bang phep tinh rieng
v.assign_add([[1., 1., 1.], [1., 1., 1.]])
    \end{lstlisting}
\end{frame}

% Slide 10
\begin{frame}{Các cấu trúc dữ liệu khác}
    \begin{itemize}
        \item \textbf{Sparse tensors (\texttt{tf.SparseTensor}):} Đại diện tối ưu cho Tensors chứa phần lớn giá trị là 0. Chứa 3 mảng: \textit{indices}, \textit{values}, \textit{dense\_shape}.
        \item \textbf{Tensor arrays (\texttt{tf.TensorArray}):} Danh sách các Tensors. Kích thước mặc định là cố định nhưng có thể cho phép mở rộng động (dynamic size).
        \item \textbf{Ragged tensors (\texttt{tf.RaggedTensor}):} Đại diện cho tập hợp danh sách các tensor có độ dài không bằng nhau (ví dụ: mảng các câu có số từ khác nhau).
        \item \textbf{String tensors:} Tensor chứa các chuỗi byte.
        \item \textbf{Sets:} Trình diễn các tập hợp dưới dạng Sparse tensor hoặc thông qua \texttt{tf.sets}.
    \end{itemize}
\end{frame}

\section{Tùy chỉnh mô hình và Thuật toán huấn luyện}

% Slide 11
\begin{frame}{Tại sao cần tùy chỉnh mô hình trong Keras?}
    \begin{itemize}
        \item Keras API cung cấp sẵn hàng chục hàm \textit{losses, metrics, layers, optimizers}. Tuy nhiên, đôi khi nó là không đủ.
        \item Bạn có thể gặp một bài toán đòi hỏi hàm tính sai số riêng (ví dụ: mô hình dự báo thời tiết cần phạt nặng lỗi dự đoán bão sai).
        \item Cần một kiến trúc tầng Neural đặc biệt vừa mới được công bố trong bài báo khoa học.
        \item \textbf{Giải pháp:} TensorFlow/Keras cho phép bạn tạo Custom Component bằng mã Python, sau đó hệ thống sẽ tự động tổng hợp (compile) chúng cùng với phần còn lại của mạng.
    \end{itemize}
\end{frame}

% Slide 12
\begin{frame}[fragile]{Tùy chỉnh Hàm mất mát (Custom Loss) - Huber Loss}
    \begin{itemize}
        \item Huber Loss là hàm phạt mềm dẻo: Giống MSE khi lỗi nhỏ, giống MAE khi lỗi lớn. Phù hợp cho tập dữ liệu nhiều nhiễu ngoại lai.
    \end{itemize}
    \begin{lstlisting}[language=Python]
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

# Su dung
model.compile(loss=huber_fn, optimizer="nadam")
model.fit(X_train, y_train, [...])
    \end{lstlisting}
\end{frame}

% Slide 13
\begin{frame}[fragile]{Lưu và tải mô hình có thành phần tùy chỉnh}
    \begin{itemize}
        \item Khi lưu mô hình bằng Keras, tên của hàm tùy chỉnh được lưu lại. Nhưng khi load, ta phải chỉ rõ hàm đó để Keras biên dịch (map objects).
    \end{itemize}
    \begin{lstlisting}[language=Python]
# Luu nhu binh thuong
model.save("my_model_with_a_custom_loss.h5")

# Tai lai (load)
model = keras.models.load_model(
    "my_model_with_a_custom_loss.h5",
    custom_objects={"huber_fn": huber_fn}
)
    \end{lstlisting}
    \begin{itemize}
        \item Lời khuyên: Để dễ quản lý siêu tham số (hyperparameters), có thể định nghĩa Custom Loss thông qua class kế thừa \texttt{keras.losses.Loss}.
    \end{itemize}
\end{frame}

% Slide 14
\begin{frame}[fragile]{Tùy chỉnh Hàm Loss theo lớp (Subclassing)}
    \begin{lstlisting}[language=Python]
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
        
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
    \end{lstlisting}
\end{frame}

% Slide 15
\begin{frame}[fragile]{Tùy chỉnh Lớp ẩn (Custom Layers)}
    \begin{itemize}
        \item Nếu bạn muốn xây dựng một loại nơ-ron mới (không phải Dense hay Conv2D).
        \item Lớp không có trọng số (vd: Flatten, ReLU): Xây dựng dễ dàng qua \texttt{keras.layers.Lambda}.
    \end{itemize}
    \begin{lstlisting}[language=Python]
# Tao tang mu (exponential layer)
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))

# Tang Lambda thich hop neu code don gian hoac dung lam
# ham hoat hoa cho lop truoc.
    \end{lstlisting}
\end{frame}

% Slide 16
\begin{frame}[fragile]{Lớp tùy chỉnh có chứa trọng số (Stateful Custom Layer)}
    \begin{itemize}
        \item Đối với tầng chứa trọng số, phải kế thừa \texttt{keras.layers.Layer}.
    \end{itemize}
    \begin{lstlisting}[language=Python]
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros")
        super().build(batch_input_shape)

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)
    \end{lstlisting}
\end{frame}

% Slide 17
\begin{frame}[fragile]{Tùy chỉnh toàn bộ Mô hình (Custom Models)}
    \begin{itemize}
        \item Phù hợp với cấu trúc phức tạp: Có nhánh vòng lặp (Residual Network), skip-connections...
        \item Ta kế thừa \texttt{keras.Model}, khai báo Layer trong \texttt{\_\_init\_\_}, và quy định cách dữ liệu truyền đi trong phương thức \texttt{call()}.
    \end{itemize}
    \begin{lstlisting}[language=Python]
class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation="elu")
                       for _ in range(n_layers)]
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z  # Phep cong Skip connection dac trung
    \end{lstlisting}
\end{frame}

\section{Hàm và biểu đồ TensorFlow (AutoGraph)}

% Slide 18
\begin{frame}{Hàm TensorFlow và đồ thị (Graphs)}
    \begin{itemize}
        \item Điểm sáng giá lớn nhất của TensorFlow 2.x là sự phân tách rõ ràng giữa chế độ chạy Tức thời (Eager Execution) và chế độ Biểu đồ tĩnh (Graph Mode).
        \item \textbf{Eager Execution:} Mã chạy dòng nào tính dòng đó. Rất tuyệt để Debug. Nhưng chậm!
        \item \textbf{Graph Execution:} TensorFlow chuyển mã Python thành Đồ thị C++ nội bộ để loại bỏ chi phí biên dịch, có thể chạy song song và tối ưu bộ nhớ cực tốt. Mạng chạy nhanh hơn đáng kể.
    \end{itemize}
\end{frame}

% Slide 19
\begin{frame}[fragile]{\texttt{@tf.function}: Phép màu của AutoGraph}
    \begin{itemize}
        \item Làm thế nào chuyển đổi code Python thành Graph? Cực kỳ đơn giản! 
    \end{itemize}
    \begin{lstlisting}[language=Python]
def cube(x):
    return x ** 3

# Chuyen ham Python sang ham TensorFlow (TF Function)
tf_cube = tf.function(cube)

# Hoac dung decorator truc tiep (pho bien hon):
@tf.function
def tf_cube(x):
    return x ** 3
    \end{lstlisting}
    \begin{itemize}
        \item \texttt{tf\_cube()} giờ đã là một Graph có thể tính toán với hiệu suất tối ưu.
    \end{itemize}
\end{frame}

% Slide 20
\begin{frame}{Cơ chế của AutoGraph}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Bước 1: TF phân tích Cây cú pháp trừu tượng Python (AST).
                \item Bước 2: Dịch các vòng lặp \texttt{for}, lệnh \texttt{if} thành toán tử TensorFlow (\texttt{tf.while\_loop}, \texttt{tf.cond}).
                \item Bước 3: Tạo Graph trung gian.
                \item Bước 4: Tối ưu Graph (xóa toán tử dư, ghép nhóm) để chạy nhanh nhất trên GPU/TPU.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH12/Hình12-2}\\
                \textit{Hình 12-2. Sơ đồ chuyển đổi của AutoGraph}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 21
\begin{frame}[fragile]{Giới hạn và Quy tắc của TF Functions}
    \begin{itemize}
        \item TF Function không thể chứa các mã Python gây hiệu ứng phụ (side effects).
    \end{itemize}
    \begin{lstlisting}[language=Python]
@tf.function
def my_func(x):
    print("Ham dang duoc trace:", x) # Chi in 1 lan trong luc tao Graph!
    return x + 1

# Loi the: TF se dung Poly-morphism (Tao ra Graph rieng cho tung Kieu du lieu).
my_func(tf.constant([1, 2]))   # Trace cho tensor int
my_func(tf.constant([1., 2.])) # Trace cho tensor float (Graph moi!)
    \end{lstlisting}
    \begin{itemize}
        \item Lời khuyên: Đừng dùng \texttt{print()} hay \texttt{open(file)} bên trong \texttt{@tf.function}. Hãy dùng \texttt{tf.print()} nếu cần log.
    \end{itemize}
\end{frame}

\section{Vòng lặp Huấn luyện Tùy chỉnh (Custom Training Loop)}

% Slide 22
\begin{frame}{Vòng lặp huấn luyện tùy chỉnh}
    \begin{itemize}
        \item Mặc dù \texttt{model.fit()} cực kỳ tiện lợi, trong nhiều nghiên cứu mới, đôi khi ta muốn làm chủ 100\% quá trình học (chẳng hạn như dùng nhiều loại Optimizer cùng lúc cho nhiều nhánh khác nhau).
        \item Viết vòng lặp tùy chỉnh (Custom Training Loop) mất công sức, dễ sinh lỗi (bugs), nhưng cho phép tùy biến không giới hạn.
    \end{itemize}
\end{frame}

% Slide 23
\begin{frame}[fragile]{\texttt{tf.GradientTape()}: Tính đạo hàm tự động}
    \begin{itemize}
        \item Để huấn luyện, ta phải tính đạo hàm (Gradient). \texttt{GradientTape} là một chiếc "băng ghi âm", ghi lại mọi phép tính bên trong ngữ cảnh (block) của nó.
    \end{itemize}
    \begin{lstlisting}[language=Python]
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = tf.Variable(5.), tf.Variable(3.)

with tf.GradientTape() as tape:
    z = f(w1, w2)

# Yeu cau Tinh Gradient cua z theo w1 va w2
gradients = tape.gradient(z, [w1, w2])

print(gradients) # [tf.Tensor(36., shape=(), dtype=float32), tf.Tensor(10., shape=(), dtype=float32)]
    \end{lstlisting}
\end{frame}

% Slide 24
\begin{frame}[fragile]{Viết Vòng lặp huấn luyện - Khởi tạo}
    \begin{lstlisting}[language=Python]
l2_reg = keras.regularizers.l2(0.05)
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
                       kernel_regularizer=l2_reg),
    keras.layers.Dense(1, kernel_regularizer=l2_reg)
])

# Tu tao tap du lieu (Dataset)
batch_size = 32
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_set = train_set.shuffle(len(X_train)).batch(batch_size)

optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error
    \end{lstlisting}
\end{frame}

% Slide 25
\begin{frame}[fragile]{Viết Vòng lặp huấn luyện - Thực thi (Epochs \& Steps)}
    \begin{lstlisting}[language=Python]
epochs = 5
for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch + 1, epochs))
    for step, (X_batch, y_batch) in enumerate(train_set):
        
        # 1. Ghi lai qua trinh Forward pass
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
            
        # 2. Tinh Gradient cua Loss theo tung trong so model
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # 3. Cap nhat trong so (Optimizer apply)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    \end{lstlisting}
\end{frame}

% Slide 26
\begin{frame}{Cấu trúc hoàn chỉnh và Những lưu ý}
    \begin{itemize}
        \item Khác với \texttt{model.fit()}, trong Vòng lặp tùy chỉnh bạn phải chịu trách nhiệm:
        \begin{enumerate}
            \item Thêm logic theo dõi Metric thủ công (\texttt{mean\_loss.update\_state()}).
            \item Truyền cờ \texttt{training=True} vào lệnh gọi model, vì một số Lớp như BatchNormalization hay Dropout hoạt động khác nhau giữa Training và Testing.
            \item Nếu có biến không thuộc diện trainable nhưng cần cập nhật (như trung bình trượt của BN), bạn phải tự gán lại.
        \end{enumerate}
        \item Hãy bao bọc vòng lặp train-step vào \texttt{@tf.function} để biến nó thành Graph, tăng tốc độ xử lý!
    \end{itemize}
\end{frame}

% Slide 27
\begin{frame}{Tóm tắt}
    \begin{itemize}
        \item TensorFlow cung cấp khả năng điều khiển rất gần gũi với NumPy cho Tensors.
        \item Với kiến trúc linh hoạt, ta có thể tạo ra Loss Function, Metrics, Layers, Models bất kỳ tùy theo dự án.
        \item AutoGraph (\texttt{@tf.function}) là công nghệ tăng tốc đáng kinh ngạc, biên dịch Python thành mô hình toán học (Graph) tối ưu.
        \item Vòng lặp huấn luyện tùy chỉnh kết hợp với \texttt{GradientTape} giúp tùy biến sâu luồng làm việc cho những bài toán học máy phi tiêu chuẩn.
    \end{itemize}
\end{frame}

\end{document}
