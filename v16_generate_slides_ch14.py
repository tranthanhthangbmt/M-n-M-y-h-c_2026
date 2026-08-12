import os

tex_content = r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}
\usepackage[T5]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{xcolor}

\usetheme{Madrid}
\usecolortheme{default}

% Setup for code listings
\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    stringstyle=\color{red},
    commentstyle=\color{green!60!black},
    showstringspaces=false,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10}
}

\title[Thị giác Máy tính Chuyên sâu với CNN]{Chương 14. Thị giác Máy tính Chuyên sâu\\sử dụng Mạng Nơ-ron Tích chập}
\author{Giảng viên: TS. Trần Thành Thắng}
\institute{Đại học Đông Á}
\date{\today}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}{Mục lục}
    \tableofcontents
\end{frame}

\section{Kiến trúc của vỏ não thị giác}

\begin{frame}{14.1. Kiến trúc của vỏ não thị giác}
    \begin{itemize}
        \item Năm 1958, David Hubel và Torsten Wiesel thực hiện loạt thí nghiệm trên mèo (và khỉ).
        \item Khám phá: Nhiều nơ-ron trong vỏ não thị giác có \textbf{trường tiếp nhận cục bộ} (local receptive field).
        \item Các nơ-ron chỉ phản ứng với kích thích hình ảnh trong một khu vực nhỏ của trường nhìn.
        \item Một số nơ-ron phản ứng với các đường thẳng nằm ngang, một số khác với đường chéo.
    \end{itemize}
\end{frame}

\begin{frame}{Sơ đồ kiến trúc vỏ não thị giác}
    \begin{center}
        \includegraphics[height=0.7\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH14/Hình_14-1}\\
        \textit{Các nơ-ron kết nối tạo thành các mẫu phức tạp hơn}
    \end{center}
\end{frame}

\section{Các lớp tích chập}

\begin{frame}{14.2. Các lớp tích chập (Convolutional layers)}
    \begin{itemize}
        \item Yếu tố quan trọng nhất của CNN là lớp \textbf{tích chập} (convolutional layer).
        \item Thay vì kết nối với toàn bộ nơ-ron ở lớp trước (như DNN), mỗi nơ-ron ở lớp tích chập chỉ kết nối với các nơ-ron trong trường tiếp nhận của nó.
        \item Kiến trúc này giúp tập trung vào các đặc trưng cục bộ mức thấp (đoạn thẳng, góc), rồi ghép nối thành các đặc trưng mức cao (khuôn mặt, ô tô).
    \end{itemize}
\end{frame}

\begin{frame}{Sự kết nối ở các lớp CNN}
    \begin{center}
        \includegraphics[width=0.9\textwidth,height=0.8\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH14/Hình_14-2}
    \end{center}
\end{frame}

\begin{frame}{Bộ lọc (Filters)}
    \begin{itemize}
        \item Một bộ trọng số của một nơ-ron có thể được biểu diễn như một hình ảnh nhỏ, gọi là \textbf{bộ lọc} (filter) hoặc nhân (kernel).
        \item Hai bộ lọc cơ bản:
        \begin{itemize}
            \item Bộ lọc dọc (chỉ cho phép đường dọc đi qua).
            \item Bộ lọc ngang (chỉ cho phép đường ngang đi qua).
        \end{itemize}
        \item Nếu áp dụng bộ lọc dọc lên hình ảnh, ta sẽ làm nổi bật các đường thẳng dọc và làm mờ phần còn lại.
    \end{itemize}
\end{frame}

\begin{frame}{Bản đồ đặc trưng (Feature Map)}
    \begin{center}
        \includegraphics[width=0.9\textwidth,height=0.8\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH14/Hình_14-5}
    \end{center}
\end{frame}

\begin{frame}[fragile]{Triển khai bằng Keras}
    \begin{itemize}
        \item Khởi tạo lớp Conv2D trong Keras rất đơn giản:
    \end{itemize}
    \begin{lstlisting}
from tensorflow import keras

conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME", activation="relu")
    \end{lstlisting}
    \begin{itemize}
        \item \texttt{filters}: số lượng feature maps cần tạo.
        \item \texttt{kernel\_size}: kích thước bộ lọc (ví dụ: $3 \times 3$).
        \item \texttt{strides}: bước nhảy của bộ lọc khi quét qua hình ảnh.
        \item \texttt{padding}: "SAME" (giữ nguyên kích thước) hoặc "VALID" (không đệm).
    \end{itemize}
\end{frame}

\begin{frame}{Lớp gộp (Pooling layers)}
    \begin{itemize}
        \item Mục tiêu: Trích xuất các đặc trưng nổi bật (subsample), thu nhỏ hình ảnh để giảm gánh nặng tính toán, giảm dung lượng bộ nhớ.
        \item Các loại:
        \begin{itemize}
            \item \textbf{Max Pooling:} Lấy giá trị lớn nhất trong vùng gộp. Giữ lại chi tiết sáng nhất.
            \item \textbf{Average Pooling:} Lấy giá trị trung bình. Thường ít dùng hơn Max Pooling vì giữ lại quá nhiều chi tiết mờ.
            \item \textbf{Global Average Pooling:} Lấy trung bình trên TOÀN BỘ feature map. Thường dùng ở lớp cuối trước khi đưa vào Softmax.
        \end{itemize}
    \end{itemize}
\end{frame}

\begin{frame}{Sơ đồ Max Pooling}
    \begin{center}
        \includegraphics[width=0.9\textwidth,height=0.8\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH14/Hình_14-8}
    \end{center}
\end{frame}

\section{Kiến trúc CNN}

\begin{frame}{14.3. Các kiến trúc CNN kinh điển}
    \begin{itemize}
        \item Kiến trúc CNN thông thường bao gồm chuỗi các cặp \texttt{Conv2D} và \texttt{Pooling}, sau đó trải phẳng (Flatten) và đưa vào các lớp Dense.
        \item Các kiến trúc nổi bật trong lịch sử:
        \begin{itemize}
            \item \textbf{LeNet-5 (1998):} Ứng dụng nhận dạng chữ số viết tay.
            \item \textbf{AlexNet (2012):} Kiến trúc mạng lớn đầu tiên giành chiến thắng vang dội tại ImageNet, sử dụng ReLU.
            \item \textbf{GoogLeNet (2014):} Khởi xướng mô-đun Inception. Đạt tỷ lệ lỗi cực thấp.
            \item \textbf{ResNet (2015):} Khởi xướng mô-đun Residual (bỏ qua kết nối). Mạng sâu tới 152 lớp.
        \end{itemize}
    \end{itemize}
\end{frame}

\begin{frame}{Mô-đun Inception (GoogLeNet)}
    \begin{center}
        \includegraphics[height=0.7\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH14/Hình_14-13}\\
        \textit{Mô-đun Inception thực hiện nhiều kích thước bộ lọc cùng lúc rồi nối kết quả lại.}
    \end{center}
\end{frame}

\begin{frame}{Kết nối dư thừa - Skip Connection (ResNet)}
    \begin{center}
        \includegraphics[height=0.7\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH14/Hình_14-15}\\
        \textit{Mô-đun Residual cộng thêm đầu vào ban đầu vào kết quả sau khi qua các lớp Conv, giúp giải quyết bài toán vanishing gradient cho mạng siêu sâu.}
    \end{center}
\end{frame}

\begin{frame}[fragile]{Triển khai ResNet-34 bằng Keras}
    \begin{itemize}
        \item Định nghĩa Khối Dư thừa (Residual Unit):
    \end{itemize}
    \begin{lstlisting}
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()]
    \end{lstlisting}
\end{frame}

\begin{frame}[fragile]{Triển khai ResNet-34 bằng Keras (tiếp)}
    \begin{lstlisting}
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    \end{lstlisting}
\end{frame}

\begin{frame}[fragile]{Sử dụng Mô hình Pretrained từ Keras}
    \begin{itemize}
        \item Keras cung cấp sẵn rất nhiều mô hình đã được huấn luyện với tập dữ liệu khổng lồ (ImageNet).
    \end{itemize}
    \begin{lstlisting}
# Khởi tạo mô hình ResNet50
model = keras.applications.resnet50.ResNet50(weights="imagenet")

# Tiền xử lý hình ảnh đầu vào (kích thước chuẩn 224x224)
images_resized = tf.image.resize(images, [224, 224])
inputs = keras.applications.resnet50.preprocess_input(images_resized)

# Dự đoán
Y_proba = model.predict(inputs)
    \end{lstlisting}
\end{frame}

\section{Phân loại và định vị}

\begin{frame}{14.4. Phân loại và Định vị}
    \begin{itemize}
        \item Phân loại hình ảnh (Classification): Trong hình có con mèo không?
        \item Định vị (Localization): Con mèo ở đâu? (Dự đoán bounding box)
        \item Mô hình thường có 2 nhánh đầu ra (heads):
        \begin{itemize}
            \item Nhánh Classification: Hàm mất mát \texttt{sparse\_categorical\_crossentropy}.
            \item Nhánh Bounding Box (4 giá trị: $y, x, height, width$): Hàm mất mát \texttt{MSE} hoặc \texttt{GIoU}.
        \end{itemize}
    \end{itemize}
\end{frame}

\begin{frame}{Chỉ số IoU (Intersection over Union)}
    \begin{center}
        \includegraphics[height=0.7\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH14/Hình_14-19}\\
        \textit{IoU = Diện tích phần giao nhau chia cho Diện tích phần hợp nhau của 2 bounding boxes.}
    \end{center}
\end{frame}

\section{Phát hiện đối tượng}

\begin{frame}{14.5. Phát hiện đối tượng (Object Detection)}
    \begin{itemize}
        \item Trong một ảnh có THỂ CÓ nhiều đối tượng. Chúng ta cần tìm TẤT CẢ các bounding boxes của chúng.
        \item \textbf{Cách 1 (Truyền thống):} Sliding Window. Quét một cửa sổ qua toàn bộ ảnh, kiểm tra xem có đối tượng nào không. (Rất chậm).
        \item \textbf{Cách 2 (FCN - Fully Convolutional Networks):} Biến Dense Layers cuối cùng thành Convolution Layers, giúp mô hình có thể quét ảnh trong 1 lượt (Single shot).
        \item YOLO (You Only Look Once): Một mô hình cực kỳ nhanh và phổ biến cho bài toán này.
    \end{itemize}
\end{frame}

\begin{frame}{Non-Max Suppression}
    \begin{itemize}
        \item Một đối tượng có thể được mô hình dự đoán ra nhiều bounding box (do quét nhiều lưới lân cận).
        \item Kỹ thuật **Non-Max Suppression** (Khử tối đa) được sử dụng để loại bỏ các bounding box trùng lặp:
        \begin{enumerate}
            \item Xóa tất cả các box có độ tự tin (confidence score) thấp hơn một ngưỡng.
            \item Chọn box có độ tự tin cao nhất.
            \item Tính IoU của các box khác với box đã chọn, xóa những box có IoU lớn (do bị đè lên).
            \item Lặp lại bước 2-3 cho đến khi hết box.
        \end{enumerate}
    \end{itemize}
\end{frame}

\section{Phân đoạn ngữ nghĩa}

\begin{frame}{14.6. Phân đoạn ngữ nghĩa (Semantic Segmentation)}
    \begin{itemize}
        \item Bài toán khó nhất: Phân loại \textbf{từng pixel} trong hình ảnh xem nó thuộc về đối tượng (class) nào.
        \item Ứng dụng: Xe tự lái (phân biệt đường, xe, vỉa hè), y tế (phát hiện khối u trong ảnh X-Quang).
        \item Kiến trúc FCN được mở rộng bằng cách thêm các lớp \textbf{Upsampling} (tăng kích thước ảnh) ở phía sau mạng (Ví dụ: kiến trúc U-Net).
    \end{itemize}
\end{frame}

\begin{frame}{Phân đoạn đối tượng}
    \begin{center}
        \includegraphics[height=0.7\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH14/Hình_14-26}\\
        \textit{Từng pixel được tô màu tương ứng với nhãn của đối tượng đó (màu xanh lá cho chó, màu vàng cho xe, v.v.).}
    \end{center}
\end{frame}

\section{Bài tập}

\begin{frame}{14.7. Bài tập}
    \begin{itemize}
        \item 1. Sự khác biệt giữa mạng nơ-ron truyền thẳng (DNN) và mạng CNN. Ưu thế của CNN là gì?
        \item 2. Tính kích thước Feature map đầu ra nếu đầu vào là $100 \times 100$, kernel size là $3 \times 3$, padding = "VALID", strides = 2.
        \item 3. Triển khai một ứng dụng phát hiện đối tượng nhỏ bằng YOLO (hoặc mô hình tương đương) trên dữ liệu tự thu thập.
        \item 4. Thử nghiệm kiến trúc U-Net cho bài toán phân đoạn ngữ nghĩa ảnh y tế đơn giản.
    \end{itemize}
\end{frame}

\begin{frame}
    \begin{center}
        \Huge \textbf{Hết Chương 14}\\
        \vspace{1em}
        \Large Chúc các bạn học tốt!
    \end{center}
\end{frame}

\end{document}
"""

with open(r'd:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap14.tex', 'w', encoding='utf-8') as f:
    f.write(tex_content)

print("Created Slide_ML_Chap14.tex successfully.")
