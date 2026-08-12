import os

def generate_slides():
    tex_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap11.tex"
    
    latex_code = r"""\documentclass[aspectratio=169]{beamer}
\usepackage{fontspec}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{listings}
\usepackage{xcolor}

% Define colors for code listings
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}
\lstset{style=mystyle}

\usetheme{Madrid}
\usecolortheme{default}
\setbeamertemplate{caption}{\raggedright\insertcaption\par}

\title[Chương 11: Huấn luyện DNN]{Học Máy (Machine Learning)\\Chương 11: Huấn luyện Mạng Nơ-ron Sâu}
\author{Giảng viên: TS. Trần Thành Thắng}
\institute{Đại học Đông Á}
\date{\today}

\begin{document}

% Slide 1
\begin{frame}
    \titlepage
\end{frame}

% Slide 2
\begin{frame}{Nội dung Chương trình}
    \tableofcontents
\end{frame}

\section{Vấn đề Gradient biến mất/bùng nổ \& Khởi tạo trọng số}

% Slide 3
\begin{frame}{Khó khăn khi huấn luyện Mạng nơ-ron sâu}
    \begin{itemize}
        \item Mạng nơ-ron sâu (DNN) có hàng chục, thậm chí hàng trăm tầng ẩn.
        \item Lợi thế: Học được các đặc trưng từ thấp đến cao cực kỳ phức tạp.
        \item Khó khăn gặp phải:
        \begin{itemize}
            \item Vấn đề Gradient biến mất (Vanishing Gradients).
            \item Vấn đề Gradient bùng nổ (Exploding Gradients).
            \item Thiếu dữ liệu hoặc tốn chi phí gán nhãn lớn.
            \item Tốc độ huấn luyện cực kỳ chậm.
            \item Nguy cơ Overfitting nghiêm trọng do hàng triệu tham số.
        \end{itemize}
    \end{itemize}
\end{frame}

% Slide 4
\begin{frame}{Vấn đề Gradient biến mất (Vanishing Gradients)}
    \begin{itemize}
        \item Khi thuật toán Backpropagation truyền lỗi từ đầu ra ngược về đầu vào, nó tính gradient bằng Quy tắc chuỗi (Chain Rule).
        \item Nếu các gradient này càng lúc càng nhỏ khi đi xuống các tầng thấp hơn (Lower layers), trọng số ở các tầng này sẽ gần như không được cập nhật.
        \item Kết quả: Các tầng đầu tiên không thể hội tụ $\rightarrow$ Toàn bộ mạng không thể học được các đặc trưng cơ bản.
    \end{itemize}
\end{frame}

% Slide 5
\begin{frame}{Vấn đề Gradient bùng nổ (Exploding Gradients)}
    \begin{itemize}
        \item Ngược lại với Gradient biến mất, trong một số mạng (đặc biệt là Mạng nơ-ron hồi quy - RNN), gradient có thể ngày càng lớn.
        \item Các trọng số nhận cập nhật khổng lồ, khiến thuật toán tối ưu phân kỳ (Diverge).
        \item Quá trình huấn luyện trở nên cực kỳ mất ổn định và bị hỏng.
    \end{itemize}
\end{frame}

% Slide 6
\begin{frame}{Nguyên nhân: Hàm Sigmoid và Khởi tạo trọng số}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Năm 2010, Glorot và Bengio phát hiện nguyên nhân gốc rễ:
                \begin{enumerate}
                    \item Hàm kích hoạt \textit{Logistic (Sigmoid)} bảo hòa ở 0 hoặc 1, đạo hàm gần như bằng 0.
                    \item Phương pháp khởi tạo ngẫu nhiên theo phân phối chuẩn với trung bình 0, phương sai 1.
                \end{enumerate}
                \item Phương sai của tín hiệu tăng dần khi đi qua các tầng, làm kích hoạt rơi vào vùng bão hòa.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH11/Hình11-1}\\
                \textit{Hình 11-1. Hàm kích hoạt logistic bão hòa}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 7
\begin{frame}{Giải pháp: Khởi tạo Glorot (Xavier Initialization)}
    \begin{itemize}
        \item Để tín hiệu lưu thông tốt, phương sai đầu ra của mỗi tầng phải xấp xỉ phương sai đầu vào: $fan_{in} \approx fan_{out}$.
        \item Khởi tạo Glorot sử dụng trung bình của đầu vào và đầu ra $fan_{avg} = (fan_{in} + fan_{out}) / 2$.
        \item Phân phối chuẩn: $\mu = 0$, $\sigma^2 = \frac{1}{fan_{avg}}$.
        \item Phân phối đều: Kéo dài từ $-r$ đến $+r$ với $r = \sqrt{\frac{3}{fan_{avg}}}$.
    \end{itemize}
\end{frame}

% Slide 8
\begin{frame}[fragile]{Giải pháp: Khởi tạo He (He Initialization)}
    \begin{itemize}
        \item Khởi tạo Glorot hoạt động tốt với Sigmoid/Tanh. Nhưng với ReLU, ta cần Khởi tạo He.
        \item Keras mặc định sử dụng Glorot uniform. Để đổi sang He, ta làm như sau:
    \end{itemize}
    \begin{lstlisting}[language=Python]
import tensorflow as tf
from tensorflow import keras

# Su dung Khoi tao He voi ReLU
keras.layers.Dense(10, activation="relu", 
                   kernel_initializer="he_normal")
    \end{lstlisting}
\end{frame}

% Slide 9
\begin{frame}{Các hàm kích hoạt tốt hơn: Hạn chế của ReLU}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item ReLU rất tuyệt nhưng gặp vấn đề \textbf{"Dying ReLU"}.
                \item Khi tính tổng có trọng số là âm, đầu ra ReLU là 0 và gradient cũng là 0.
                \item Nơ-ron bị "chết", vĩnh viễn không bao giờ cập nhật nữa vì tín hiệu gradient bị chặn lại hoàn toàn.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH11/Hình_11-2}\\
                \textit{Hình 11-2. Hàm Leaky ReLU}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 10
\begin{frame}[fragile]{Hàm Leaky ReLU và PReLU}
    \begin{itemize}
        \item \textbf{Leaky ReLU:} Định nghĩa bằng $LeakyReLU_\alpha(z) = \max(\alpha z, z)$.
        \item $\alpha$ (vd 0.01) tạo một độ dốc nhỏ (leak) khi $z < 0$, giữ cho nơ-ron không bao giờ "chết".
        \item \textbf{PReLU (Parametric Leaky ReLU):} $\alpha$ được học trong quá trình huấn luyện như một tham số thông thường. Rất tốt cho bộ dữ liệu lớn.
    \end{itemize}
    \begin{lstlisting}[language=Python]
model = keras.models.Sequential([
    keras.layers.Dense(10, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2)
])
    \end{lstlisting}
\end{frame}

% Slide 11
\begin{frame}{Hàm ELU (Exponential Linear Unit) và SELU}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item \textbf{ELU} (Clevert et al. 2015): Tốt hơn tất cả các biến thể ReLU.
                $$ ELU_\alpha(z) = \begin{cases} z & \text{khi } z \ge 0 \\ \alpha(\exp(z) - 1) & \text{khi } z < 0 \end{cases} $$
                \item \textbf{SELU:} Biến thể của ELU, cho phép mạng tự chuẩn hóa (self-normalize). Cực kỳ mạnh mẽ cho MLP nhiều tầng.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH11/Hình_11-3}\\
                \textit{Hình 11-3. Hàm kích hoạt ELU}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

\section{Chuẩn hóa Batch (Batch Normalization) \& Gradient Clipping}

% Slide 12
\begin{frame}{Chuẩn hóa Batch (Batch Normalization - BN) là gì?}
    \begin{itemize}
        \item Do Ioffe và Szegedy đề xuất (2015).
        \item BN giảm thiểu tình trạng \textit{Dịch chuyển hợp điểm (Internal Covariate Shift)}.
        \item Ý tưởng: Thêm một thao tác \textbf{ngay trước hoặc sau hàm kích hoạt}, tập trung hóa trung bình (zero-mean) và chuẩn hóa dữ liệu đầu vào.
        \item Tác dụng: Cực kỳ ổn định quá trình học, giảm Gradient biến mất, cho phép tăng Learning Rate lớn hơn nhiều.
    \end{itemize}
\end{frame}

% Slide 13
\begin{frame}{Cơ chế hoạt động của thuật toán Batch Normalization}
    \begin{itemize}
        \item Thuật toán đo lường $\mu_B$ (trung bình batch) và $\sigma_B^2$ (phương sai batch).
        \item Chuẩn hóa: $\hat{\mathbf{x}}^{(i)} = \frac{\mathbf{x}^{(i)} - \boldsymbol{\mu}_B}{\sqrt{\boldsymbol{\sigma}_B^2 + \epsilon}}$.
        \item Scale và shift: $\mathbf{z}^{(i)} = \boldsymbol{\gamma} \otimes \hat{\mathbf{x}}^{(i)} + \boldsymbol{\beta}$.
        \item $\gamma$ (tham số tỷ lệ) và $\beta$ (tham số dịch chuyển) được mô hình học thông qua Backpropagation.
        \item Khi Inference (Dự đoán), BN sử dụng trung bình trượt động (moving average) đã học toàn cục.
    \end{itemize}
\end{frame}

% Slide 14
\begin{frame}[fragile]{Triển khai Batch Normalization với Keras}
    \begin{itemize}
        \item Triển khai BN trong Keras rất trực quan:
    \end{itemize}
    \begin{lstlisting}[language=Python]
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    
    keras.layers.Dense(300, activation="relu"),
    keras.layers.BatchNormalization(),
    
    keras.layers.Dense(100, activation="relu"),
    keras.layers.BatchNormalization(),
    
    keras.layers.Dense(10, activation="softmax")
])
    \end{lstlisting}
\end{frame}

% Slide 15
\begin{frame}[fragile]{BN trước hay sau hàm kích hoạt?}
    \begin{itemize}
        \item Các tác giả BN ban đầu gợi ý đặt BN \textit{trước} hàm kích hoạt.
        \item Keras cho phép tách hàm kích hoạt để chèn BN vào giữa:
    \end{itemize}
    \begin{lstlisting}[language=Python]
model.add(keras.layers.Dense(300, use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
    \end{lstlisting}
    \begin{itemize}
        \item Lưu ý: \texttt{use\_bias=False} vì BN đã cung cấp tham số dịch chuyển $\beta$.
    \end{itemize}
\end{frame}

% Slide 16
\begin{frame}[fragile]{Cắt xén Gradient (Gradient Clipping)}
    \begin{itemize}
        \item Giải pháp trực tiếp và đơn giản nhất cho \textbf{Gradient bùng nổ}.
        \item Trong lúc backpropagation, nếu gradient vượt quá một ngưỡng nhất định, ta "cắt xén" (clip) chúng xuống.
        \item Ứng dụng mạnh nhất trong mạng RNN (nơi BN khó triển khai).
    \end{itemize}
    \begin{lstlisting}[language=Python]
# Clip theo gia tri
optimizer = keras.optimizers.SGD(clipvalue=1.0)

# Clip theo norm (chuan) - thuong dung hon
optimizer = keras.optimizers.SGD(clipnorm=1.0)
    \end{lstlisting}
\end{frame}

\section{Tái sử dụng mô hình (Transfer Learning) \& Tiền huấn luyện}

% Slide 17
\begin{frame}{Tái sử dụng các lớp đã huấn luyện (Transfer Learning)}
    \begin{itemize}
        \item Không bao giờ nên huấn luyện một DNN từ đầu (từ số không) nếu có thể tìm thấy một mạng đã học tác vụ tương tự!
        \item \textbf{Transfer Learning:} Lấy các lớp dưới (Lower layers) của một mô hình đã học tốt, lắp ghép vào mô hình mới của bạn.
        \item Lợi ích: Tăng tốc độ hội tụ siêu nhanh, đòi hỏi lượng dữ liệu gán nhãn ít hơn hẳn so với học từ đầu.
    \end{itemize}
\end{frame}

% Slide 18
\begin{frame}{Kiến trúc khi Transfer Learning}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Lower layers: Có xu hướng học các đặc trưng cấp thấp (cạnh, góc, kết cấu). Dễ dàng chuyển giao.
                \item Upper layers: Học các khái niệm đặc thù. Cần loại bỏ và thay bằng tầng mới cho tác vụ hiện tại.
                \item Hình bên: Giữ lại Layer 1-3 từ Mạng A, thay thế Output cho Mạng B.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH11/Hình_11-4}\\
                \textit{Hình 11-4. Sử dụng lại mô hình}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 19
\begin{frame}[fragile]{Đóng băng trọng số (Freezing layers)}
    \begin{itemize}
        \item Để trọng số cũ không bị phá hỏng trong lần huấn luyện đầu, ta phải \textbf{đóng băng} (freeze) các lớp cũ.
    \end{itemize}
    \begin{lstlisting}[language=Python]
model_A = keras.models.load_model("model_A.h5")
# Loai bo tang dau ra
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

# Dong bang cac tang cu
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(loss="binary_crossentropy", 
                     optimizer=keras.optimizers.SGD(lr=1e-3))
    \end{lstlisting}
\end{frame}

% Slide 20
\begin{frame}{Tiền huấn luyện không giám sát (Unsupervised Pretraining)}
    \begin{itemize}
        \item Nếu bạn thiếu dữ liệu nhãn (Labeled data) nhưng có rất nhiều dữ liệu không nhãn (Unlabeled data):
        \item Huấn luyện mô hình không giám sát (Autoencoders hoặc GAN) trên dữ liệu không nhãn.
        \item Sau đó, dùng các lớp tái tạo đặc trưng đó làm nền tảng cho mạng giám sát mới.
        \item Rộng rãi sử dụng từ thời phục hưng Deep Learning (Geoffrey Hinton, 2006).
    \end{itemize}
\end{frame}

\section{Các Trình tối ưu hóa nhanh hơn \& Lập lịch Tốc độ học}

% Slide 21
\begin{frame}{Các Trình tối ưu hóa (Optimizers) vượt trội hơn SGD}
    \begin{itemize}
        \item \textbf{SGD} (Stochastic Gradient Descent) cơ bản hoạt động tốt nhưng rất \textbf{chậm}.
        \item Các giải pháp tăng tốc quá trình học:
        \begin{enumerate}
            \item Tối ưu hóa Động lượng (Momentum).
            \item Tối ưu hóa Nesterov Accelerated Gradient (NAG).
            \item AdaGrad.
            \item RMSProp.
            \item Tối ưu hóa Adam (Adaptive Moment Estimation) \& Nadam.
        \end{enumerate}
    \end{itemize}
\end{frame}

% Slide 22
\begin{frame}{Tối ưu hóa Động lượng (Momentum Optimization)}
    \begin{itemize}
        \item Lấy cảm hứng từ một quả bóng lăn xuống dốc: Tích lũy đà (momentum) để lăn ngày càng nhanh!
        \item Cập nhật vec-tơ động lượng $\mathbf{m}$:
        $$\mathbf{m} \leftarrow \beta \mathbf{m} - \eta \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$$
        $$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \mathbf{m}$$
        \item Tham số động lượng $\beta$ thường bằng $0.9$.
        \item Triển khai Keras: \texttt{optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9)}
    \end{itemize}
\end{frame}

% Slide 23
\begin{frame}[fragile]{Tối ưu hóa Adam (Adaptive Moment Estimation)}
    \begin{itemize}
        \item **Adam** kết hợp ý tưởng của Momentum (đà) và RMSProp (tỷ lệ chuẩn).
        \item Là một trong những Optimizer mạnh mẽ, mặc định tốt nhất cho đa số dự án Deep Learning hiện nay.
        \item Rất ít cần phải tinh chỉnh tốc độ học.
    \end{itemize}
    \begin{lstlisting}[language=Python]
optimizer = keras.optimizers.Adam(lr=0.001, 
                                  beta_1=0.9, 
                                  beta_2=0.999)
    \end{lstlisting}
    \begin{itemize}
        \item \textit{Nadam} và \textit{AdamW} là các bản nâng cấp xuất sắc hơn của Adam!
    \end{itemize}
\end{frame}

% Slide 24
\begin{frame}{Lập lịch tốc độ học (Learning Rate Scheduling)}
    \begin{itemize}
        \item Tốc độ học ($\eta$) quan trọng nhất. 
        \item Nếu $\eta$ cố định lớn: Khó hội tụ tối ưu cục bộ. Nếu $\eta$ cố định nhỏ: Học quá chậm.
        \item Giải pháp: Bắt đầu lớn, sau đó giảm dần $\eta$ theo lịch (Schedules).
        \begin{itemize}
            \item Power Scheduling.
            \item Exponential Scheduling.
            \item Piecewise Constant Scheduling.
            \item Performance Scheduling.
        \end{itemize}
    \end{itemize}
\end{frame}

% Slide 25
\begin{frame}[fragile]{Triển khai Lập lịch trong Keras (Exponential)}
    \begin{itemize}
        \item Sử dụng \texttt{keras.callbacks.LearningRateScheduler} hoặc thiết lập trong Optimizer.
    \end{itemize}
    \begin{lstlisting}[language=Python]
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
lr_scheduler = keras.callbacks.LearningRateScheduler(
    exponential_decay_fn
)
history = model.fit(X_train, y_train, [...], 
                    callbacks=[lr_scheduler])
    \end{lstlisting}
\end{frame}

\section{Tránh Overfitting thông qua Chính quy hóa}

% Slide 26
\begin{frame}{Chính quy hóa trong Mạng nơ-ron sâu}
    \begin{itemize}
        \item Với hàng chục triệu tham số, mạng DNN có thể ghi nhớ toàn bộ tập Train (Overfitting cực độ).
        \item Các kỹ thuật Regularization tiêu chuẩn:
        \begin{itemize}
            \item Early Stopping (Dừng sớm).
            \item $\ell_1$ và $\ell_2$ Regularization.
            \item \textbf{Dropout}.
            \item Max-Norm Regularization.
        \end{itemize}
    \end{itemize}
\end{frame}

% Slide 27
\begin{frame}[fragile]{Chính quy hóa $\ell_1$ và $\ell_2$ (L1/L2 Regularization)}
    \begin{itemize}
        \item $\ell_2$ giới hạn độ lớn của các trọng số mạng nơ-ron, $\ell_1$ giúp tạo ra một mô hình thưa thớt (loại bỏ các trọng số không quan trọng).
    \end{itemize}
    \begin{lstlisting}[language=Python]
layer = keras.layers.Dense(100, activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))
    \end{lstlisting}
    \begin{itemize}
        \item Việc viết mã lặp lại nhiều lần rất rườm rà. Lời khuyên: Sử dụng hàm \texttt{functools.partial()} để tạo lớp cấu hình chung!
    \end{itemize}
\end{frame}

% Slide 28
\begin{frame}{Kỹ thuật Dropout}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Khái niệm đỉnh cao từ Geoffrey Hinton (2012).
                \item Tại mỗi bước huấn luyện, mọi nơ-ron (kể cả tầng vào, ngoại trừ tầng ra) có xác suất $p$ bị bỏ qua ("dropped out").
                \item Mạng bị buộc phải học cách không quá phụ thuộc vào bất kỳ nơ-ron riêng lẻ nào $\rightarrow$ Tính khái quát cực cao.
                \item Tham số chuẩn: $p = 0.5$.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH11/Hình_11-9}\\
                \textit{Hình 11-9. Kỹ thuật Dropout}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 29
\begin{frame}{Tóm tắt cấu hình mạng DNN mặc định}
    \begin{center}
    \begin{tabular}{ll}
        \toprule
        \textbf{Siêu tham số} & \textbf{Cấu hình mặc định gợi ý} \\
        \midrule
        Khởi tạo Kernel & Khởi tạo He \\
        Hàm kích hoạt & ELU (hoặc ReLU cho đơn giản) \\
        Chuẩn hóa & Không nếu là mạng nông; Batch Norm nếu mạng sâu \\
        Chính quy hóa & Early Stopping (+ L2 nếu cần) \\
        Trình tối ưu hóa & Nadam, AdamW hoặc Adam \\
        Lập lịch Tốc độ học & 1-cycle hoặc Exponential decay \\
        \bottomrule
    \end{tabular}
    \end{center}
    \vspace{0.5cm}
    \textit{Lưu ý: Bảng này áp dụng chung, nhưng có những kiến trúc riêng như CNN (sẽ dùng ReLU) hay Mạng Tự chuẩn hóa SELU.}
\end{frame}

% Slide 30
\begin{frame}
    \begin{center}
        \Huge \textbf{Hỏi \& Đáp (Q\&A)}
    \end{center}
\end{frame}

\end{document}
"""

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    
    print(f"Da tao thanh cong: {tex_path}")

if __name__ == "__main__":
    generate_slides()
