import os

def generate_slides():
    tex_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap10.tex"
    
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

\title[Chương 10: Giới thiệu Mạng Nơ-ron]{Học Máy (Machine Learning)\\Chương 10: Giới thiệu về Mạng Nơ-ron Nhân tạo với Keras}
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

\section{Từ Nơ-ron Sinh học đến Nơ-ron Nhân tạo}

% Slide 3
\begin{frame}{Giới thiệu Mạng Nơ-ron nhân tạo (ANN)}
    \begin{itemize}
        \item Mạng Nơ-ron Nhân tạo (Artificial Neural Networks - ANN) là cốt lõi của Học sâu (Deep Learning).
        \item Rất đa dạng, mạnh mẽ và có khả năng mở rộng quy mô.
        \item Lý tưởng để giải quyết các bài toán học máy phức tạp:
        \begin{itemize}
            \item Phân loại hình ảnh (Image Classification).
            \item Nhận dạng giọng nói (Speech Recognition).
            \item Dịch máy (Machine Translation).
            \item Lái xe tự hành và AI chơi game (AlphaGo).
        \end{itemize}
    \end{itemize}
\end{frame}

% Slide 4
\begin{frame}{Lịch sử và Sự trỗi dậy của Deep Learning}
    \begin{itemize}
        \item ANN được giới thiệu lần đầu vào năm 1943 (McCulloch và Pitts).
        \item Trải qua nhiều giai đoạn thịnh vượng và suy thoái ("Mùa đông AI").
        \item Sự trở lại mạnh mẽ từ năm 2012 nhờ:
        \begin{itemize}
            \item Lượng dữ liệu khổng lồ (Big Data).
            \item Sức mạnh tính toán tăng vọt (Định luật Moore, GPU).
            \item Cải tiến trong các thuật toán huấn luyện.
            \item Sự hỗ trợ của cộng đồng và nguồn quỹ đầu tư.
        \end{itemize}
    \end{itemize}
\end{frame}

% Slide 5
\begin{frame}{Từ Nơ-ron Sinh học đến Nhân tạo}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Nơ-ron sinh học cấu tạo gồm: Thân tế bào, sợi nhánh (dendrites), và sợi trục (axon).
                \item Giao tiếp qua các khớp thần kinh (synapses) bằng tín hiệu hóa học/điện.
                \item Mạng lưới sinh học chứa hàng tỷ nơ-ron hoạt động song song.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH10/Hinh_10-1}\\
                \textit{Hình 10-1. Cấu tạo nơ-ron sinh học}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 6
\begin{frame}{Tính toán logic với Nơ-ron}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Mô hình nơ-ron nhân tạo đầu tiên của McCulloch và Pitts (1943).
                \item Một nơ-ron có thể kích hoạt nếu nhận đủ số tín hiệu đầu vào.
                \item Có thể xây dựng các phép toán logic đơn giản: C, OR, AND.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH10/Hinh_10-3}\\
                \textit{Hình 10-3. Phép tính logic bằng mạng nơ-ron}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 7
\begin{frame}{Perceptron là gì?}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Do Frank Rosenblatt phát minh năm 1957.
                \item Dựa trên Đơn vị Logic Ngưỡng (TLU - Threshold Logic Unit).
                \item Tính tổng có trọng số của các đầu vào: $z = w_1 x_1 + w_2 x_2 + ... + w_n x_n = \mathbf{w}^T \mathbf{x}$.
                \item Áp dụng hàm bước (step function) lên tổng đó: $h_w(x) = step(z)$.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH10/Hinh_10-4}\\
                \textit{Hình 10-4. Đơn vị Logic Ngưỡng (TLU)}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 8
\begin{frame}{Kiến trúc của Perceptron}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Một Perceptron gồm một tầng các TLU kết nối với tất cả các đầu vào.
                \item Mỗi kết nối có một trọng số (weight).
                \item Nơ-ron xu hướng (bias neuron) luôn xuất giá trị 1.
                \item Có thể sử dụng để phân loại đa lớp (Multi-class classification).
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH10/Hinh_10-5}\\
                \textit{Hình 10-5. Kiến trúc Perceptron}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 9
\begin{frame}{Huấn luyện Perceptron: Quy tắc học Hebbian}
    \begin{itemize}
        \item Hebb (1949): "Các tế bào kích hoạt cùng nhau thì liên kết với nhau."
        \item Quy tắc cập nhật trọng số của Perceptron:
        $$w_{i,j}^{(next step)} = w_{i,j} + \eta (y_j - \hat{y}_j) x_i$$
        \begin{itemize}
            \item $w_{i,j}$: Trọng số kết nối giữa nơ-ron vào $i$ và nơ-ron ra $j$.
            \item $x_i$: Giá trị đầu vào thứ $i$.
            \item $\hat{y}_j$: Dự đoán của nơ-ron đầu ra $j$.
            \item $y_j$: Nhãn thực tế.
            \item $\eta$: Tốc độ học (Learning rate).
        \end{itemize}
    \end{itemize}
\end{frame}

% Slide 10
\begin{frame}{Hạn chế của Perceptron (Bài toán XOR)}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Minsky và Papert (1969) chỉ ra điểm yếu nghiêm trọng: Perceptron không thể học các bài toán không tuyến tính cơ bản, ví dụ: Phép toán XOR (Độc quyền OR).
                \item Tuy nhiên, có thể khắc phục bằng cách xếp chồng các Perceptron lên nhau.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH10/Hinh_10-6}\\
                \textit{Hình 10-6. Bài toán XOR và Mạng nơ-ron đa lớp}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

\section{Mạng Perceptron đa lớp (MLP) \& Backpropagation}

% Slide 11
\begin{frame}{Mạng Perceptron đa lớp (MLP)}
    \begin{itemize}
        \item MLP cấu tạo từ:
        \begin{itemize}
            \item Một tầng đầu vào (Input layer).
            \item Một hoặc nhiều tầng ẩn (Hidden layers) gồm các TLU.
            \item Một tầng đầu ra (Output layer).
        \end{itemize}
        \item Được gọi là Mạng nơ-ron tiến (Feedforward Neural Network - FNN) vì tín hiệu chỉ truyền theo một hướng (từ đầu vào tới đầu ra).
        \item Mạng có từ 2 tầng ẩn trở lên được gọi là Mạng nơ-ron sâu (Deep Neural Network - DNN).
    \end{itemize}
\end{frame}

% Slide 12
\begin{frame}{Backpropagation (Lan truyền ngược)}
    \begin{itemize}
        \item Do Rumelhart, Hinton, và Williams giới thiệu (1986).
        \item Là thuật toán huấn luyện cốt lõi của Học sâu.
        \item Quá trình:
        \begin{enumerate}
            \item \textbf{Truyền tiến (Forward pass):} Truyền dữ liệu qua mạng để tính kết quả dự đoán.
            \item \textbf{Tính lỗi:} Đo lường sai số so với kết quả thực tế.
            \item \textbf{Truyền ngược (Backward pass):} Đi ngược qua các tầng để tính độ dốc lỗi (Gradient) của từng kết nối bằng Quy tắc chuỗi (Chain Rule).
            \item \textbf{Cập nhật trọng số (Gradient Descent):} Điều chỉnh trọng số để giảm lỗi.
        \end{enumerate}
    \end{itemize}
\end{frame}

% Slide 13
\begin{frame}{Các hàm kích hoạt (Activation functions)}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Backpropagation yêu cầu hàm kích hoạt có đạo hàm khác 0 (không giống hàm bước).
                \item Các hàm phổ biến:
                \begin{itemize}
                    \item Hàm Sigmoid: $\sigma(z) = 1 / (1 + \exp(-z))$.
                    \item Hàm Hyperbolic Tangent (Tanh): Đầu ra từ -1 đến 1.
                    \item Hàm ReLU (Rectified Linear Unit): $ReLU(z) = \max(0, z)$. Đơn giản và phổ biến nhất hiện nay.
                \end{itemize}
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH10/Hinh_10-8}\\
                \textit{Hình 10-8. Các hàm kích hoạt và đạo hàm}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 14
\begin{frame}{MLP cho Hồi quy (Regression)}
    \begin{itemize}
        \item Đầu ra: 1 nơ-ron cho bài toán dự đoán 1 giá trị liên tục (vd: giá nhà).
        \item Không sử dụng hàm kích hoạt ở tầng đầu ra (để xuất giá trị tự do).
        \item Hàm mất mát (Loss function): Thường dùng Mean Squared Error (MSE) hoặc Mean Absolute Error (MAE).
        \item Có thể dự đoán nhiều giá trị cùng lúc (Multi-output regression) bằng cách dùng nhiều nơ-ron đầu ra.
    \end{itemize}
\end{frame}

% Slide 15
\begin{frame}{MLP cho Phân loại (Classification)}
    \begin{itemize}
        \item \textbf{Phân loại nhị phân (Binary):} 1 nơ-ron đầu ra với hàm kích hoạt Sigmoid. Hàm mất mát: \textit{binary\_crossentropy}.
        \item \textbf{Phân loại đa nhãn (Multilabel):} $N$ nơ-ron đầu ra (một cho mỗi nhãn) với hàm kích hoạt Sigmoid.
        \item \textbf{Phân loại đa lớp (Multiclass):} $N$ nơ-ron đầu ra với hàm kích hoạt \textbf{Softmax} (để tổng xác suất các lớp bằng 1). Hàm mất mát: \textit{categorical\_crossentropy}.
    \end{itemize}
\end{frame}

\section{Triển khai MLP với Keras \& API Tuần tự}

% Slide 16
\begin{frame}{Triển khai MLP với TensorFlow và Keras}
    \begin{itemize}
        \item \textbf{TensorFlow:} Thư viện học sâu mạnh mẽ của Google, hỗ trợ tính toán song song trên GPU/TPU.
    \item \textbf{Keras:} API cấp cao được tích hợp sẵn trong TensorFlow (\texttt{tf.keras}).
        \item Keras cung cấp giao diện trực quan, nhất quán để xây dựng, huấn luyện và đánh giá mô hình.
    \end{itemize}
\end{frame}

% Slide 17
\begin{frame}[fragile]{Xây dựng bộ phân loại hình ảnh bằng API Tuần tự (Sequential API)}
    \begin{lstlisting}[language=Python]
import tensorflow as tf
from tensorflow import keras

# Khởi tạo mô hình
model = keras.models.Sequential()
# Tầng đầu vào: làm phẳng hình ảnh 28x28 thành vector 784
model.add(keras.layers.Flatten(input_shape=[28, 28]))
# Tầng ẩn 1 với 300 nơ-ron và hàm ReLU
model.add(keras.layers.Dense(300, activation="relu"))
# Tầng ẩn 2 với 100 nơ-ron và hàm ReLU
model.add(keras.layers.Dense(100, activation="relu"))
# Tầng đầu ra với 10 nơ-ron và hàm Softmax
model.add(keras.layers.Dense(10, activation="softmax"))
    \end{lstlisting}
\end{frame}

% Slide 18
\begin{frame}[fragile]{Xem cấu trúc mô hình}
    \begin{itemize}
    \item Lệnh \texttt{model.summary()} hiển thị tất cả các tầng, số lượng tham số (trọng số và bias).
    \end{itemize}
    \begin{lstlisting}[language=Python]
model.summary()
# Layer (type)                 Output Shape              Param #   
# =================================================================
# flatten (Flatten)            (None, 784)               0         
# _________________________________________________________________
# dense (Dense)                (None, 300)               235500    
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               30100     
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1010      
# =================================================================
# Total params: 266,610
    \end{lstlisting}
\end{frame}

% Slide 19
\begin{frame}[fragile]{Biên dịch (Compile) mô hình}
    \begin{itemize}
    \item Trước khi huấn luyện, cần cấu hình mô hình bằng \texttt{compile()}.
    \end{itemize}
    \begin{lstlisting}[language=Python]
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
    \end{lstlisting}
    \begin{itemize}
        \item \texttt{loss}: Đo sai số (ví dụ: dùng sparse cho các nhãn chỉ số 0-9).
        \item \texttt{optimizer}: Cập nhật trọng số (vd: "sgd" - Stochastic Gradient Descent).
        \item \texttt{metrics}: Thước đo đánh giá hiệu suất.
    \end{itemize}
\end{frame}

% Slide 20
\begin{frame}[fragile]{Huấn luyện và Đánh giá}
    \begin{itemize}
    \item Sử dụng hàm \texttt{fit()} để huấn luyện và \texttt{evaluate()} để kiểm thử.
    \end{itemize}
    \begin{lstlisting}[language=Python]
# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_valid, y_valid))

# Đánh giá trên tập kiểm thử
model.evaluate(X_test, y_test)

# Dự đoán
X_new = X_test[:3]
y_pred = np.argmax(model.predict(X_new), axis=-1)
    \end{lstlisting}
\end{frame}

% Slide 21
\begin{frame}{Phân tích Lịch sử Huấn luyện (Learning Curves)}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
    \item Object \texttt{history.history} lưu giữ loss và metrics qua các epochs.
                \item Có thể dùng Pandas và Matplotlib để trực quan hóa.
                \item Nếu tập validation đi ngang trong khi tập train tiếp tục giảm $\rightarrow$ Dấu hiệu Overfitting.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH10/Hinh_10-12}\\
                \textit{Hình 10-12. Learning curves}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

\section{Xây dựng mô hình phức tạp với Functional API}

% Slide 22
\begin{frame}{Xây dựng mô hình bằng API Chức năng (Functional API)}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Không phải mạng nơ-ron nào cũng là một luồng dữ liệu tuyến tính.
                \item Kiến trúc **Wide \& Deep** (Được Google giới thiệu năm 2016): Kết nối một phần (hoặc toàn bộ) dữ liệu đầu vào thẳng với tầng đầu ra.
                \item Giúp mô hình học được cả các mẫu sâu phức tạp (đường deep) lẫn các quy tắc đơn giản (đường wide).
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH10/Hinh_10-14}\\
                \textit{Hình 10-14. Kiến trúc Wide \& Deep}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 23
\begin{frame}[fragile]{Mã nguồn: API Chức năng (Functional API)}
    \begin{lstlisting}[language=Python]
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

# Nối đầu vào trực tiếp với đầu ra của hidden2
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs=[input_], outputs=[output])
    \end{lstlisting}
\end{frame}

% Slide 24
\begin{frame}{Xây dựng mô hình Động với Subclassing API}
    \begin{itemize}
        \item Sequential và Functional API có tính khai báo (Declarative): Mô tả trước cấu trúc mạng, dễ dàng lưu trữ và phân tích.
        \item Tuy nhiên, với một số kiến trúc có điều kiện rẽ nhánh động (như vòng lặp \texttt{for}, \texttt{if/else}), cần sử dụng \textbf{Subclassing API}.
        \item Cách làm: Kế thừa lớp \texttt{keras.Model}, định nghĩa các tầng trong \texttt{\_\_init\_\_()} và điều hướng luồng tính toán trong phương thức \texttt{call()}.
    \end{itemize}
\end{frame}

\section{Lưu khôi phục, Callback \& Tinh chỉnh}

% Slide 25
\begin{frame}{Lưu, Khôi phục Mô hình và Callbacks}
    \begin{itemize}
        \item \textbf{Lưu mô hình:} \texttt{model.save("my\_keras\_model.h5")} (lưu cả kiến trúc, trọng số và trạng thái optimizer).
        \item \textbf{Khôi phục:} \texttt{model = keras.models.load\_model("my\_keras\_model.h5")}.
        \item \textbf{Callbacks:} Các hàm được gọi trong quá trình huấn luyện để can thiệp:
        \begin{itemize}
            \item `ModelCheckpoint`: Lưu trạng thái mô hình tốt nhất.
            \item `EarlyStopping`: Dừng sớm nếu không còn cải thiện trên tập validation (tránh Overfitting).
        \end{itemize}
    \end{itemize}
\end{frame}

% Slide 26
\begin{frame}{Giám sát bằng TensorBoard}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item TensorBoard là công cụ trực quan hóa tương tác tuyệt vời.
                \item Giúp so sánh learning curves giữa các lần chạy, xem đồ thị cấu trúc mạng, phân tích profiling.
    \item Sử dụng qua \texttt{keras.callbacks.TensorBoard}.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH10/Hinh_10-16}\\
                \textit{Hình 10-16. Giao diện TensorBoard}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 27
\begin{frame}{Tinh chỉnh các siêu tham số mạng nơ-ron}
    \begin{itemize}
        \item Mạng nơ-ron có vô số siêu tham số: Số tầng, Số nơ-ron mỗi tầng, Hàm kích hoạt, Tốc độ học (Learning rate), Batch size...
        \item Làm sao để tìm ra cấu hình tốt nhất?
        \begin{itemize}
    \item \textbf{Bọc (Wrap)} mô hình Keras trong \texttt{KerasRegressor} hoặc \texttt{KerasClassifier} để dùng chung với Scikit-Learn.
    \item Sử dụng \texttt{RandomizedSearchCV} hoặc \texttt{GridSearchCV}.
            \item Tốt hơn: Sử dụng các thư viện chuyên dụng như \textbf{Keras Tuner}, Optuna hoặc Hyperopt.
        \end{itemize}
    \end{itemize}
\end{frame}

% Slide 28
\begin{frame}{Các lưu ý về Siêu tham số cốt lõi}
    \begin{itemize}
        \item \textbf{Số lượng tầng ẩn:} Thường nên bắt đầu với 1-2 tầng. Các mạng sâu giúp học các đặc trưng phân cấp, nhưng dễ gặp vấn đề vanishing gradients.
        \item \textbf{Số nơ-ron:} Thường dùng cấu trúc hình phễu (giảm dần số lượng nơ-ron ở các tầng sâu). Hiện nay thường giữ số nơ-ron bằng nhau ở tất cả các tầng (để đơn giản hóa việc tinh chỉnh).
        \item \textbf{Tốc độ học (Learning rate):} Siêu tham số quan trọng nhất. Nếu quá nhỏ, mô hình học chậm; nếu quá lớn, quá trình hội tụ sẽ phân kỳ.
        \item \textbf{Kích thước lô (Batch size):} Batch lớn giúp tận dụng phần cứng (GPU) nhưng có thể làm giảm khả năng khái quát hóa. Thường thử nghiệm với batch size $\leq 32$.
    \end{itemize}
\end{frame}

% Slide 29
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
