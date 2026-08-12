import os

tex_content = r"""\documentclass[aspectratio=169]{beamer}
\usepackage{fontspec}
\usepackage[utf8]{inputenc}
\usepackage[vietnamese]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{amsmath}

\usetheme{Madrid}
\usecolortheme{default}

% Cấu hình code Python
\lstset{
    language=Python,
    basicstyle=\ttfamily\footnotesize,
    keywordstyle=\color{blue},
    stringstyle=\color{red},
    commentstyle=\color{green!50!black},
    showstringspaces=false,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10}
}

\title[Máy học - Chương 15]{Chương 15: Xử lý Chuỗi bằng RNN và CNN}
\author[Trần Thành Thắng]{Trần Thành Thắng, Nguyễn Văn A}
\institute[UDA]{Khoa Công nghệ thông tin - Đại học Đông Á}
\date{Năm học: 2024 - 2025}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}{Nội dung chính}
    \tableofcontents
\end{frame}

% ==============================================
% PHẦN 1: MẠNG NƠ-RON HỒI QUY CƠ BẢN
% ==============================================
\section{Mạng nơ-ron hồi quy cơ bản (RNN)}

\begin{frame}{15.1. Giới thiệu Dữ liệu chuỗi (Sequence Data)}
    \begin{itemize}
        \item Mạng nơ-ron truyền thẳng (Feedforward Neural Networks) chỉ nhận dữ liệu đầu vào có kích thước cố định và tạo ra dự đoán không quan tâm đến thứ tự hoặc lịch sử.
        \item Dữ liệu chuỗi xuất hiện khắp nơi trong thực tế:
        \begin{itemize}
            \item \textbf{Chuỗi thời gian (Time series):} Giá cổ phiếu, thời tiết, doanh thu theo ngày.
            \item \textbf{Văn bản (Text):} Một câu là một chuỗi các từ hoặc ký tự.
            \item \textbf{Âm thanh (Audio):} Tín hiệu sóng âm biến đổi theo thời gian.
        \end{itemize}
        \item Cần một kiến trúc mạng có khả năng "nhớ" thông tin quá khứ $\rightarrow$ Mạng nơ-ron hồi quy (RNN - Recurrent Neural Networks).
    \end{itemize}
\end{frame}

\begin{frame}{Đặc điểm của Mạng Nơ-ron Hồi quy}
    \begin{itemize}
        \item RNN khác với Feedforward NN ở chỗ nó có các \textbf{kết nối chéo ngược lại (recurrent connections)}.
        \item Một nơ-ron hồi quy nhận hai đầu vào tại mỗi bước thời gian $t$:
        \begin{enumerate}
            \item Đầu vào hiện tại $x_{(t)}$
            \item Trạng thái (đầu ra) từ bước thời gian trước đó $h_{(t-1)}$
        \end{enumerate}
        \item Do đó, RNN có một "bộ nhớ" ẩn bên trong duy trì trạng thái của hệ thống qua từng bước thời gian.
    \end{itemize}
\end{frame}

\begin{frame}{Cấu trúc nơ-ron hồi quy cơ bản}
    \begin{columns}
        \begin{column}{0.65\textwidth}
            \includegraphics[width=\textwidth,height=0.85\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH15/Hình_15-1}
        \end{column}
        \begin{column}{0.35\textwidth}
            \textbf{Trải phẳng theo thời gian (Unrolling through time):}\\
            \vspace{0.5em}
            Hình ảnh bên trái là một nơ-ron RNN với kết nối vòng.\\
            Hình ảnh bên phải thể hiện việc "trải phẳng" vòng lặp này qua trục thời gian.
        \end{column}
    \end{columns}
\end{frame}

\begin{frame}{Lớp nơ-ron hồi quy (Cell Layer)}
    \begin{itemize}
        \item Chúng ta có thể kết hợp nhiều nơ-ron hồi quy thành một \textbf{lớp (layer)}.
        \item Tại mỗi bước thời gian $t$, mỗi nơ-ron nhận đầu vào là vector $x_{(t)}$ và vector đầu ra của cả lớp từ bước thời gian trước $h_{(t-1)}$.
        \item Các trọng số của lớp RNN bao gồm:
        \begin{itemize}
            \item $W_x$: Ma trận trọng số cho các đầu vào $x_{(t)}$.
            \item $W_h$: Ma trận trọng số cho các đầu ra từ bước trước $h_{(t-1)}$.
            \item $b$: Vector độ lệch (bias).
        \end{itemize}
    \end{itemize}
\end{frame}

\begin{frame}{Công thức tính toán của RNN}
    \begin{block}{Tính toán đầu ra của một lớp RNN}
    Đầu ra tại bước thời gian $t$ (đồng thời cũng là trạng thái mới) được tính bằng công thức:
    \[ h_{(t)} = \phi \left( W_x^T x_{(t)} + W_h^T h_{(t-1)} + b \right) \]
    \end{block}
    \vspace{1em}
    Trong đó:
    \begin{itemize}
        \item $x_{(t)}$ là vector đầu vào có kích thước $(n_{features} \times 1)$.
        \item $W_x$ có kích thước $(n_{features} \times n_{neurons})$.
        \item $W_h$ có kích thước $(n_{neurons} \times n_{neurons})$.
        \item $h_{(t-1)}$ là trạng thái ẩn từ bước trước $(n_{neurons} \times 1)$.
        \item $\phi$ là hàm kích hoạt (thường dùng $\tanh$ hoặc $ReLU$).
    \end{itemize}
\end{frame}

\begin{frame}{Biểu diễn qua Mini-batch}
    \begin{itemize}
        \item Để tăng tốc độ huấn luyện, ta thường đưa vào một \textbf{mini-batch} gồm $m$ chuỗi dữ liệu thay vì từng chuỗi riêng lẻ.
        \item Công thức được vector hóa cho một mini-batch $X_{(t)}$:
    \end{itemize}
    \begin{block}{Vector hóa đầu ra RNN}
    \[ H_{(t)} = \phi \left( X_{(t)} W_x + H_{(t-1)} W_h + b \right) \]
    \end{block}
    \begin{itemize}
        \item $X_{(t)}$: Ma trận kích thước $(m \times n_{features})$
        \item $H_{(t-1)}$: Ma trận kích thước $(m \times n_{neurons})$
        \item $H_{(t)}$: Ma trận kết quả kích thước $(m \times n_{neurons})$
    \end{itemize}
\end{frame}

\begin{frame}{Các loại cấu trúc RNN}
    \begin{center}
        \includegraphics[width=0.85\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH15/Hình_15-3}
        
        \textit{Hình 15-3: Các biến thể của RNN (Seq2Seq, Seq2Vector, Vector2Seq, Seq2Seq trễ)}
    \end{center}
\end{frame}

\begin{frame}{Giải thích các biến thể RNN}
    \begin{itemize}
        \item \textbf{Sequence-to-Sequence (Seq2Seq):} Đầu vào là chuỗi, đầu ra là chuỗi. (Ví dụ: Dự báo chuỗi thời gian, dự báo giá trị tương lai).
        \item \textbf{Sequence-to-Vector (Seq2Vec):} Đầu vào là chuỗi, nhưng mạng chỉ lấy đầu ra ở bước thời gian cuối cùng. (Ví dụ: Phân loại cảm xúc văn bản).
        \item \textbf{Vector-to-Sequence (Vec2Seq):} Nhận đầu vào là một vector (chỉ ở bước $t=0$) và tạo ra một chuỗi ở đầu ra. (Ví dụ: Image captioning).
        \item \textbf{Encoder-Decoder (Seq2Vec $\rightarrow$ Vec2Seq):} Sử dụng hai mạng. Mạng thứ nhất nén chuỗi thành vector, mạng thứ hai giải mã vector thành chuỗi. (Ví dụ: Dịch máy - Machine Translation).
    \end{itemize}
\end{frame}


% ==============================================
% PHẦN 2: HUẤN LUYỆN RNN
% ==============================================
\section{Huấn luyện RNN}

\begin{frame}{15.2. Huấn luyện RNN: BPTT}
    \begin{itemize}
        \item Làm thế nào để huấn luyện mạng nơ-ron hồi quy? Kỹ thuật được sử dụng là \textbf{Backpropagation Through Time (BPTT)}.
        \item Các bước thực hiện:
        \begin{enumerate}
            \item \textbf{Lan truyền xuôi (Forward pass):} Mạng được "trải phẳng" qua thời gian và dữ liệu đi qua toàn bộ chuỗi để tính toán dự đoán.
            \item \textbf{Tính toán Loss:} Tính hàm mất mát $L$ dựa trên các dự đoán (có thể bỏ qua một số dự đoán trung gian nếu là Seq2Vec).
            \item \textbf{Lan truyền ngược (Backward pass):} Gradient của Loss được tính ngược lại qua từng bước thời gian của mạng đã được trải phẳng.
            \item \textbf{Cập nhật trọng số:} Sử dụng Gradient Descent để cập nhật $W_x, W_h, b$.
        \end{enumerate}
    \end{itemize}
\end{frame}

\begin{frame}{Vấn đề cốt lõi khi huấn luyện RNN}
    \begin{itemize}
        \item Khi chuỗi dữ liệu quá dài (vd: 100 bước thời gian), mạng RNN được trải phẳng thành một mạng rất sâu có 100 lớp.
        \item Dẫn đến hai vấn đề nghiêm trọng:
        \begin{itemize}
            \item \textbf{Vanishing Gradients (Gradient suy biến):} Gradient giảm theo cấp số nhân trong quá trình lan truyền ngược, khiến các lớp đầu tiên (các bước thời gian cũ) không thể học được.
            \item \textbf{Exploding Gradients (Gradient bùng nổ):} Gradient tăng vọt, gây mất ổn định và phá vỡ trọng số mô hình.
        \end{itemize}
    \end{itemize}
\end{frame}

\begin{frame}{Giải pháp cho Exploding Gradients}
    \begin{itemize}
        \item Hiện tượng bùng nổ gradient thường dễ phát hiện (mô hình trả về giá trị NaN).
        \item \textbf{Giải pháp:} Cắt xén gradient (\textit{Gradient Clipping}).
    \end{itemize}
    \begin{block}{Gradient Clipping}
    Giới hạn giá trị của Gradient trong một khoảng $[-C, C]$ (ví dụ: cắt tại $1.0$). Nếu gradient tính được lớn hơn $1.0$, nó sẽ bị ép về $1.0$.
    \end{block}
    \begin{itemize}
        \item Trong Keras, rất dễ dàng để áp dụng bằng cách truyền tham số `clipnorm` hoặc `clipvalue` vào Optimizer.
    \end{itemize}
\end{frame}

\begin{frame}[fragile]{Sử dụng Gradient Clipping trong Keras}
    \begin{lstlisting}[language=Python]
from tensorflow import keras

# Cắt gradient để không vượt quá giá trị tuyệt đối 1.0
optimizer = keras.optimizers.SGD(learning_rate=0.01, clipvalue=1.0)

# Hoặc cắt gradient dựa trên L2-norm (Bảo toàn hướng vector gradient)
optimizer = keras.optimizers.SGD(learning_rate=0.01, clipnorm=1.0)

model.compile(loss="mse", optimizer=optimizer)
    \end{lstlisting}
\end{frame}

\begin{frame}{Giải pháp cho Vanishing Gradients (1)}
    \begin{itemize}
        \item Giải quyết Vanishing Gradients khó khăn hơn nhiều vì bộ nhớ dài hạn của RNN từ từ phai nhạt dần.
        \item \textbf{Khởi tạo trọng số hợp lý:} Không thể dùng khởi tạo He cho RNN cơ bản một cách ngây thơ vì nó có thể dẫn đến bùng nổ. Nên dùng các phương pháp khởi tạo trực giao (\textit{Orthogonal Initialization}).
        \item \textbf{Hàm kích hoạt (Activation Functions):} 
        \begin{itemize}
            \item Dùng $ReLU$ có thể gây bùng nổ gradient trong RNN.
            \item Khác biệt với mạng CNN/DNN, RNN thường ưu tiên $\tanh$ hơn vì nó có giới hạn giá trị đầu ra (từ -1 đến 1), giúp tránh trạng thái ẩn bùng nổ.
        \end{itemize}
    \end{itemize}
\end{frame}

\begin{frame}{Giải pháp cho Vanishing Gradients (2)}
    \begin{itemize}
        \item \textbf{Batch Normalization (BN):} 
        \begin{itemize}
            \item Hoạt động không tốt nếu đặt giữa các bước thời gian (nghĩa là tác động lên $H_{(t)}$), do phân phối dữ liệu ở mỗi bước có thể khác nhau.
        \end{itemize}
        \item \textbf{Layer Normalization (Chuẩn hóa Lớp):}
        \begin{itemize}
            \item Thay vì chuẩn hóa theo chiều batch, LayerNorm chuẩn hóa theo chiều đặc trưng (feature dimension).
            \item Giúp ổn định rất tốt trạng thái ẩn của RNN tại từng bước thời gian độc lập với các bước khác.
        \end{itemize}
        \item \textbf{Thiết kế lại cấu trúc tế bào:} Cách triệt để nhất là thay thế nơ-ron cơ bản bằng các ô (cells) có cấu trúc cổng (Gating mechanisms) như LSTM và GRU.
    \end{itemize}
\end{frame}

% ==============================================
% PHẦN 3: DỰ BÁO CHUỖI THỜI GIAN
% ==============================================
\section{Dự báo chuỗi thời gian}

\begin{frame}{15.3. Bài toán Dự báo chuỗi thời gian}
    \begin{itemize}
        \item **Mục tiêu:** Cho một chuỗi các giá trị trong quá khứ $X = \{x_1, x_2, ..., x_t\}$, nhiệm vụ là dự đoán giá trị tại tương lai $x_{t+1}$ hoặc $x_{t+n}$.
        \item Các ứng dụng: 
        \begin{itemize}
            \item Dự báo giá cổ phiếu.
            \item Dự báo nhu cầu điện năng.
            \item Số lượng hành khách trên chuyến bay, ...
        \end{itemize}
        \item Trước khi dùng RNN sâu, ta luôn cần xây dựng các chỉ số đánh giá cơ sở (Baselines).
    \end{itemize}
\end{frame}

\begin{frame}{Tạo dữ liệu cho dự báo chuỗi thời gian}
    \begin{center}
        \includegraphics[width=0.85\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH15/Hình_15-5}
        
        \textit{Hình 15-5: Ví dụ tạo ra chuỗi thời gian giả lập bằng cách kết hợp các chuỗi sóng sin (hàm lượng giác) cộng thêm nhiễu ngẫu nhiên.}
    \end{center}
\end{frame}

\begin{frame}{Các phương pháp Baseline}
    \begin{itemize}
        \item \textbf{Naive Forecasting (Dự báo ngây thơ):}
        \begin{itemize}
            \item Đơn giản là dự đoán giá trị tiếp theo bằng đúng giá trị ở bước cuối cùng: $\hat{x}_{t+1} = x_t$.
            \item Mặc dù cực kỳ đơn giản nhưng rất khó đánh bại trong dữ liệu nhiễu cao (như giá cổ phiếu - random walk).
        \end{itemize}
        \item \textbf{Linear Regression (Hồi quy tuyến tính):}
        \begin{itemize}
            \item Sử dụng các bước thời gian $n$ gần nhất như là $n$ đặc trưng (features) độc lập và huấn luyện một mô hình tuyến tính đơn giản.
            \item Cung cấp một ngưỡng cơ sở để đánh giá xem RNN phức tạp có thực sự hữu ích hay không.
        \end{itemize}
    \end{itemize}
\end{frame}

\begin{frame}[fragile]{Triển khai một Simple RNN cơ bản}
    \begin{itemize}
        \item Xây dựng mạng RNN đơn giản nhất bằng Keras: Chỉ 1 lớp `SimpleRNN` chứa 1 nơ-ron duy nhất.
    \end{itemize}
    \begin{lstlisting}[language=Python]
# X_train có shape: (batch_size, n_steps, 1)
model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1])
])

model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, epochs=20)
    \end{lstlisting}
    \begin{itemize}
        \item Mạng này sẽ học một tổ hợp tuyến tính phức tạp hơn qua thời gian. Kích thước chuỗi đầu vào (`n_steps`) được đặt là `None` để mạng có thể xử lý độ dài chuỗi tuỳ ý.
    \end{itemize}
\end{frame}

\begin{frame}[fragile]{Dự báo nhiều bước thời gian (Multi-step)}
    \begin{itemize}
        \item **Cách 1: Tái sử dụng dự đoán (Autoregressive).**
        \begin{itemize}
            \item Dự báo bước $t+1$, sau đó thêm kết quả đó vào chuỗi để dự báo bước $t+2$, ... lặp lại liên tục.
            \item \textit{Nhược điểm:} Lỗi sẽ tích luỹ qua từng bước, dự đoán xa sẽ rất thiếu chính xác.
        \end{itemize}
    \end{itemize}
    \begin{lstlisting}[language=Python]
series = X_valid[0:1]  # lấy chuỗi gốc
for step_ahead in range(10): # Dự báo 10 bước
    y_pred_one = model.predict(series[:, step_ahead:])
    # Nối dự đoán mới vào cuối chuỗi
    series = np.concatenate([series, y_pred_one.reshape(1,1,1)], axis=1)
    \end{lstlisting}
\end{frame}

\begin{frame}[fragile]{Dự báo nhiều bước (Seq2Vector nhiều đầu ra)}
    \begin{itemize}
        \item **Cách 2: Dự đoán 10 giá trị cùng một lúc.**
        \begin{itemize}
            \item Thay vì xuất ra 1 giá trị, lớp Dense cuối cùng sẽ xuất ra 10 giá trị tương ứng với 10 bước thời gian trong tương lai.
        \end{itemize}
    \end{itemize}
    \begin{lstlisting}[language=Python]
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10) # 10 nơ ron cho 10 bước thời gian
])
    \end{lstlisting}
    \begin{itemize}
        \item Cần tạo `y_train` thành các vector gồm 10 giá trị (từ $t+1$ đến $t+10$).
    \end{itemize}
\end{frame}

\begin{frame}{Kiến trúc Seq2Seq cho chuỗi thời gian}
    \begin{itemize}
        \item **Cách 3: Output tại mỗi bước thời gian.**
        \begin{itemize}
            \item Tại bước $t$, dự đoán giá trị $t+1, ..., t+10$.
            \item Tại bước $t+1$, dự đoán giá trị $t+2, ..., t+11$.
        \end{itemize}
        \item Điều này cung cấp rất nhiều tín hiệu gradient cho RNN, giúp tăng tốc độ hội tụ và giảm thiểu Vanishing Gradients.
        \item \textbf{Cài đặt Keras:} Dùng `return_sequences=True` ở tất cả lớp RNN và bọc lớp Dense trong `TimeDistributed`.
    \end{itemize}
\end{frame}

% ==============================================
% PHẦN 4: RNN SÂU (DEEP RNNs)
% ==============================================
\section{Mạng RNN Sâu (Deep RNNs)}

\begin{frame}{15.4. Khái niệm về Deep RNN}
    \begin{columns}
        \begin{column}{0.65\textwidth}
            \includegraphics[width=\textwidth,height=0.85\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH15/Hình_15-10}
        \end{column}
        \begin{column}{0.35\textwidth}
            \textit{Hình 15-10: Cấu trúc của Deep RNN.}\\
            \vspace{1em}
            - Xếp chồng nhiều lớp RNN lên nhau.\\
            - Đầu ra của lớp RNN dưới ở bước thời gian $t$ sẽ trở thành đầu vào của lớp RNN trên ở cùng bước thời gian $t$.
        \end{column}
    \end{columns}
\end{frame}

\begin{frame}[fragile]{Triển khai Deep RNN bằng Keras}
    \begin{itemize}
        \item Để xếp chồng nhiều lớp RNN, các lớp ở bên dưới \textbf{phải} trả về toàn bộ chuỗi đầu ra (sử dụng `return_sequences=True`).
    \end{itemize}
    \begin{lstlisting}[language=Python]
model = keras.models.Sequential([
    # Lớp 1: return_sequences=True để truyền cả chuỗi lên lớp 2
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    
    # Lớp 2: return_sequences=True để truyền lên lớp 3
    keras.layers.SimpleRNN(20, return_sequences=True),
    
    # Lớp 3: return_sequences=False vì là lớp cuối, chỉ lấy giá trị cuối
    keras.layers.SimpleRNN(20),
    
    keras.layers.Dense(1)
])
    \end{lstlisting}
\end{frame}

\begin{frame}{Lưu ý khi dùng Deep RNN}
    \begin{itemize}
        \item RNN không nhất thiết phải sâu như CNN (ví dụ ResNet có 152 lớp). Thông thường RNN chỉ cần từ 2 đến 3 lớp.
        \item Thời gian huấn luyện của mạng Deep RNN tăng lên rất đáng kể.
        \item RNN rất dễ bị quá khớp (Overfitting), vì vậy thường cần phải sử dụng Dropout.
        \item \textbf{Dropout trong RNN:} Không nên áp dụng giữa các bước thời gian. Keras hỗ trợ `dropout` (áp dụng cho đầu vào) và `recurrent_dropout` (áp dụng cho trạng thái hồi quy).
    \end{itemize}
\end{frame}


% ==============================================
% PHẦN 5: TẾ BÀO LSTM VÀ GRU
% ==============================================
\section{Các Tế Bào Bộ Nhớ (LSTM \& GRU)}

\begin{frame}{15.5. Trí nhớ ngắn hạn của Simple RNN}
    \begin{itemize}
        \item Simple RNN gặp phải hiện tượng quên thông tin rất nhanh. 
        \item Mặc dù lý thuyết nó có thể kết nối thông tin xa, nhưng thực tế với dữ liệu dài (ví dụ: một câu văn 100 từ), khi đọc đến từ thứ 100, Simple RNN gần như quên mất từ thứ 1.
        \item Trọng số $W_h$ liên tục tác động nhân lên trạng thái ẩn, làm phai nhòa ký ức xa.
        \item Năm 1997, Sepp Hochreiter và Jürgen Schmidhuber giới thiệu tế bào **LSTM (Long Short-Term Memory)** giải quyết xuất sắc vấn đề này.
    \end{itemize}
\end{frame}

\begin{frame}{Cấu trúc tế bào LSTM}
    \begin{columns}
        \begin{column}{0.65\textwidth}
            \includegraphics[width=\textwidth,height=0.85\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH15/Hình_15-12}
        \end{column}
        \begin{column}{0.35\textwidth}
            \textit{Hình 15-12: Cấu trúc bên trong của một tế bào LSTM.}\\
            \vspace{1em}
            Tách thành 2 đường trạng thái:\\
            - $c_{(t)}$: Cell state (dài hạn)\\
            - $h_{(t)}$: Hidden state (ngắn hạn)
        \end{column}
    \end{columns}
\end{frame}

\begin{frame}{Nguyên lý hoạt động của LSTM (Các Cổng - Gates)}
    \begin{itemize}
        \item Trạng thái dài hạn $c_{(t-1)}$ chạy xuyên qua mạng thẳng từ trái sang phải, chỉ đi qua một phép nhân cơ bản và một phép cộng. Điều này giúp bộ nhớ dài hạn được bảo toàn qua hàng ngàn bước thời gian (Tránh Vanishing Gradients).
        \item LSTM điều khiển thông tin thông qua 3 "Cổng" (Gating mechanisms):
        \begin{enumerate}
            \item \textbf{Forget gate (Cổng quên):} Quyết định phần nào của bộ nhớ dài hạn $c_{(t-1)}$ nên bị xóa đi.
            \item \textbf{Input gate (Cổng đầu vào):} Quyết định phần thông tin mới nào nên được thêm vào bộ nhớ dài hạn.
            \item \textbf{Output gate (Cổng xuất):} Quyết định phần nào của bộ nhớ dài hạn hiện tại $c_{(t)}$ nên được xuất ra làm trạng thái ngắn hạn $h_{(t)}$.
        \end{enumerate}
    \end{itemize}
\end{frame}

\begin{frame}[fragile]{Triển khai LSTM trong Keras}
    \begin{itemize}
        \item Việc sử dụng LSTM trong Keras cực kỳ dễ dàng, chỉ cần thay `SimpleRNN` bằng `LSTM`.
        \item Các thông số kỹ thuật bên trong đã được Keras tối ưu hoá bằng C++ và GPU (như CuDNN).
    \end{itemize}
    \begin{lstlisting}[language=Python]
model = keras.models.Sequential([
    # LSTM sẽ tự động hiểu cách xử lý chuỗi dài
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20),
    keras.layers.Dense(1)
])
    \end{lstlisting}
\end{frame}

\begin{frame}{Tế bào GRU (Gated Recurrent Unit)}
    \begin{columns}
        \begin{column}{0.65\textwidth}
            \includegraphics[width=\textwidth,height=0.85\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH15/Hình_15-13}
        \end{column}
        \begin{column}{0.35\textwidth}
            \textit{Hình 15-13: Cấu trúc của GRU.}\\
            \vspace{1em}
            Do Kyunghyun Cho et al. đề xuất (2014).\\
            Là phiên bản "rút gọn" của LSTM.
        \end{column}
    \end{columns}
\end{frame}

\begin{frame}{Điểm khác biệt của GRU so với LSTM}
    \begin{itemize}
        \item \textbf{Trạng thái gộp:} GRU kết hợp trạng thái dài hạn $c_{(t)}$ và trạng thái ngắn hạn $h_{(t)}$ thành một trạng thái duy nhất $h_{(t)}$.
        \item \textbf{Chỉ có 2 cổng:}
        \begin{itemize}
            \item \textbf{Update gate (Cổng cập nhật):} Đóng vai trò vừa là cổng quên, vừa là cổng đầu vào. Nếu cổng cập nhật bằng 1, mô hình nhớ thông tin cũ và bỏ qua thông tin mới. Nếu bằng 0, mô hình quên thông tin cũ và ghi đè thông tin mới.
            \item \textbf{Reset gate (Cổng thiết lập lại):} Quyết định xem phần thông tin quá khứ nào nên được phớt lờ đi khi tính toán ứng viên trạng thái hiện tại.
        \end{itemize}
        \item \textbf{Ưu điểm:} Ít tham số hơn LSTM, tính toán nhanh hơn mà hiệu suất phân tích ngữ cảnh dài không thua kém đáng kể.
    \end{itemize}
\end{frame}


% ==============================================
% PHẦN 6: WAVENET VÀ CNN 1D
% ==============================================
\section{Xử lý chuỗi với CNN (WaveNet)}

\begin{frame}{15.6. Tại sao dùng CNN cho Dữ liệu chuỗi?}
    \begin{itemize}
        \item RNN có một giới hạn cốt lõi: Việc tính toán bắt buộc phải diễn ra tuần tự từ $t=0, 1, 2, ...$ Do đó \textbf{không thể song song hóa} tốt trên GPU.
        \item CNN có thể chạy đồng thời trên toàn bộ bức ảnh (hoặc chuỗi). Lớp tích chập 1D (\textbf{1D Convolution}) trượt dọc theo dữ liệu chuỗi, tương tự như trượt trên ảnh ngang.
        \item Ưu điểm của 1D CNN:
        \begin{itemize}
            \item Tốc độ huấn luyện cực kỳ nhanh.
            \item Có khả năng nhận diện các mô thức cục bộ (local patterns) trong chuỗi.
        \end{itemize}
    \end{itemize}
\end{frame}

\begin{frame}[fragile]{Sử dụng Conv1D làm tiền xử lý cho RNN}
    \begin{itemize}
        \item Một kỹ thuật phổ biến là dùng `Conv1D` để rút gọn chuỗi, sau đó đưa kết quả vào lớp RNN.
    \end{itemize}
    \begin{lstlisting}[language=Python]
model = keras.models.Sequential([
    # Giảm chiều dài chuỗi đi một nửa với strides=2
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                        input_shape=[None, 1]),
    
    # Truyền đặc trưng đã rút gọn vào GRU
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20),
    keras.layers.Dense(1)
])
    \end{lstlisting}
\end{frame}

\begin{frame}{WaveNet và Tích chập giãn nở (Dilated Convolution)}
    \begin{columns}
        \begin{column}{0.65\textwidth}
            \includegraphics[width=\textwidth,height=0.85\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH15/Hình_15-14}
        \end{column}
        \begin{column}{0.35\textwidth}
            \textit{Hình 15-14: Kiến trúc WaveNet.}\\
            \vspace{1em}
            Tích chập 1D với "dilation rate" tăng dần (1, 2, 4, 8...).
        \end{column}
    \end{columns}
\end{frame}

\begin{frame}{Nguyên lý sức mạnh của WaveNet}
    \begin{itemize}
        \item WaveNet được giới thiệu bởi DeepMind (2016) để tổng hợp giọng nói cực kỳ chân thực.
        \item \textbf{Dilated Convolution:} Là bộ lọc tích chập có các "khoảng trống" ở giữa.
        \item Nếu `dilation_rate=2`, bộ lọc sẽ bỏ qua 1 giá trị đầu vào ở giữa các trọng số của nó.
        \item Bằng cách tăng gấp đôi `dilation_rate` ở mỗi lớp ($1, 2, 4, 8, ...$), \textbf{Trường thụ cảm (Receptive Field)} của mạng mở rộng theo cấp số nhân.
        \item Mạng thấp học đặc trưng ngắn (phát âm), mạng cao học đặc trưng dài hạn (ngữ điệu, từ ngữ) mà chỉ cần một số lượng lớp rất ít so với RNN thông thường.
    \end{itemize}
\end{frame}

\section{Tổng kết}

\begin{frame}{15.7. Tổng kết Chương}
    \begin{itemize}
        \item \textbf{RNN cơ bản:} Giải quyết bài toán dữ liệu chuỗi nhờ trạng thái hồi quy. Dễ gặp vấn đề về Vanishing Gradients với chuỗi dài.
        \item \textbf{Dự báo chuỗi thời gian:} Cần xây dựng baseline (Linear Regression, Naive), có thể dùng RNN để dự báo Seq2Vector hoặc Seq2Seq.
        \item \textbf{Bộ nhớ cải tiến (LSTM, GRU):} Cấu trúc Cổng (Gates) giải quyết triệt để sự lãng quên thông tin xa, bảo toàn gradient dài hạn.
        \item \textbf{CNN 1D \& WaveNet:} Cung cấp tốc độ xử lý song song vượt trội và mở rộng khả năng tiếp nhận chuỗi dài theo cấp số nhân thông qua Dilated Convolutions.
    \end{itemize}
\end{frame}

\begin{frame}
    \begin{center}
        \Huge \textbf{Hết Chương 15}\\
        \vspace{1em}
        \Large Chúc các bạn học tốt!
    \end{center}
\end{frame}

\end{document}
"""

with open(r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap15.tex', 'w', encoding='utf-8') as f:
    f.write(tex_content)
