\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}
\usepackage{fontspec}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{graphicx}

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

\title[Chương 13]{Chương 13: Tải và Tiền xử lý dữ liệu với TensorFlow}
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

\section{API tf.data và Tiền xử lý Luồng dữ liệu}

% Slide 3
\begin{frame}{Tại sao cần API \texttt{tf.data}?}
    \begin{itemize}
        \item Xử lý một lượng dữ liệu khổng lồ (vài chục GB hoặc TB) là thách thức cực lớn vì bộ nhớ RAM không bao giờ đủ chứa.
        \item \textbf{API \texttt{tf.data}} ra đời để giải quyết vấn đề này. Nó nạp dữ liệu một cách thông minh:
        \begin{enumerate}
            \item Đọc tuần tự dữ liệu từ đĩa cứng (thậm chí từ đám mây).
            \item Tiến hành tiền xử lý (giải mã, chuẩn hóa kích thước, cắt ghép).
            \item Chia theo batch và đưa vào GPU để huấn luyện liên tục.
        \end{enumerate}
        \item Hoạt động cực kỳ hiệu quả bằng cách chạy xử lý dữ liệu trên CPU đa luồng, \textit{song song} với GPU.
    \end{itemize}
\end{frame}

% Slide 4
\begin{frame}[fragile]{Khởi tạo Dataset từ RAM}
    \begin{itemize}
        \item Cách đơn giản nhất để tạo Dataset là nạp một tensor có sẵn trong RAM.
    \end{itemize}
    \begin{lstlisting}[language=Python]
import tensorflow as tf

X = tf.range(10) # Tensor [0, 1, ..., 9]
dataset = tf.data.Dataset.from_tensor_slices(X)

# Duyet qua Dataset
for item in dataset:
    print(item) # Output lan luot: tf.Tensor(0, ...), tf.Tensor(1, ...), ...
    \end{lstlisting}
    \begin{itemize}
        \item Hàm \texttt{from\_tensor\_slices()} cắt tensor dọc theo chiều thứ nhất (first dimension).
    \end{itemize}
\end{frame}

% Slide 5
\begin{frame}[fragile]{Chuỗi biến đổi (Chaining Transformations)}
    \begin{itemize}
        \item Dataset API được thiết kế dạng Fluent API: Cho phép "chuỗi" liên kết các thao tác với nhau thành luồng dữ liệu (Data Pipeline).
    \end{itemize}
    \begin{lstlisting}[language=Python]
dataset = tf.data.Dataset.range(10)
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

# Ket qua:
# tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int64)
# tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int64)
# tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int64)
# tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int64)
# tf.Tensor([8 9], shape=(2,), dtype=int64)
    \end{lstlisting}
\end{frame}

% Slide 6
\begin{frame}[fragile]{Các phương thức trộn (Shuffle) và Biến đổi}
    \begin{itemize}
        \item \textbf{map()}: Biến đổi từng phần tử dữ liệu bằng một hàm tùy chỉnh. (Rất hay dùng để nạp ảnh và resize).
        \item \textbf{filter()}: Lọc dữ liệu theo điều kiện.
        \item \textbf{shuffle(buffer\_size)}: Lấy ngẫu nhiên các phần tử từ một bộ đệm tạm thời để xáo trộn tập dữ liệu, chống over-fitting.
    \end{itemize}
    \begin{lstlisting}[language=Python]
dataset = tf.data.Dataset.range(10)

# Xao tron, bien doi va loc
dataset = dataset.shuffle(buffer_size=5, seed=42).batch(2)

dataset = dataset.map(lambda x: x * 2) # Nhan doi

# dataset = dataset.filter(lambda x: x < 10)
    \end{lstlisting}
\end{frame}

% Slide 7
\begin{frame}[fragile]{Tìm nạp trước (Prefetching)}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Gọi hàm \texttt{dataset.prefetch(1)} luôn đặt ở cuối luồng dữ liệu.
                \item Cơ chế: CPU sẽ chuẩn bị batch dữ liệu tiếp theo (batch n+1) và giữ trong bộ nhớ đệm, trong lúc GPU đang miệt mài huấn luyện với batch n.
                \item Loại bỏ hoàn toàn thời gian "GPU đói" (starving), nâng tốc độ lên gấp đôi.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH13/Hình_13-1}\\
                \textit{Hình 13-1. Xử lý song song CPU và GPU bằng Prefetch}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 8
\begin{frame}[fragile]{Nạp dữ liệu đa luồng từ nhiều tệp (Interleaving)}
    \begin{itemize}
        \item Khi dữ liệu cực lớn chia ra hàng chục file CSV riêng rẽ, ta dùng hàm \texttt{interleave()} để đọc nhiều tệp cùng một lúc theo các luồng (threads).
    \end{itemize}
    \begin{lstlisting}[language=Python]
filepaths_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)

# Tao dataset gom cac dong tu cac tep CSV xen ke nhau
n_readers = 5
dataset = filepaths_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1), # Bo qua Header
    cycle_length=n_readers
)

# Chuyen kieu string sang so thuc bang tf.io.decode_csv
    \end{lstlisting}
\end{frame}

% Slide 9
\begin{frame}[fragile]{Đọc cú pháp tệp CSV với \texttt{tf.data}}
    \begin{lstlisting}[language=Python]
record_defaults=[0, np.nan, tf.constant(np.nan, dtype=tf.float64), "Unknown"]
# Cot 1 (int), cot 2 (float), cot 3 (float64), cot 4 (chuoi string)

parsed_fields = tf.io.decode_csv(line, record_defaults)
# tf.io.decode_csv() nhan vao 1 string (dong cua tep) va tra ra
# List cac tensor tuc la cac cot duoc boc tach.
    \end{lstlisting}
    \begin{itemize}
        \item Ta sẽ gộp \texttt{decode\_csv} vào bên trong hàm \texttt{map()} của dataset.
        \item Keras có sẵn \texttt{model.fit(dataset, epochs=10)} nhận vào luồng Dataset trực tiếp rất tương thích!
    \end{itemize}
\end{frame}

\section{Định dạng TFRecord và TensorFlow Protobufs}

% Slide 10
\begin{frame}{Định dạng TFRecord}
    \begin{itemize}
        \item CSV khá đơn giản nhưng rất tốn kém chi phí I/O (chậm chạp vì là dạng Text) và chiếm không gian lưu trữ lớn.
        \item **TFRecord** là định dạng nhị phân nhẹ, linh hoạt và được thiết kế chuẩn để lưu trữ dữ liệu với tốc độ đọc cực cao cho TensorFlow.
        \item Khi dữ liệu huấn luyện > 100GB, sử dụng TFRecord gần như là điều bắt buộc.
    \end{itemize}
\end{frame}

% Slide 11
\begin{frame}[fragile]{Viết và Đọc tệp TFRecord}
    \begin{itemize}
        \item TFRecord là một chuỗi các bản ghi. Mỗi bản ghi bao gồm Độ dài (Length), Kiểm tra lỗi (CRC) và dữ liệu thực tế.
    \end{itemize}
    \begin{lstlisting}[language=Python]
# Tao tep TFRecord
with tf.io.TFRecordWriter("my_data.tfrecord") as f:
    f.write(b"Day la record dau tien")
    f.write(b"Day la record thu hai")

# Doc tep TFRecord
filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item) # Output dang kieu bytes (b'...')
    \end{lstlisting}
\end{frame}

% Slide 12
\begin{frame}[fragile]{Nén tệp TFRecord}
    \begin{itemize}
        \item Bạn có thể nén file TFRecord với GZIP để tiết kiệm đĩa cứng và tăng tốc độ truyền tải trên mạng.
    \end{itemize}
    \begin{lstlisting}[language=Python]
options = tf.io.TFRecordOptions(compression_type="GZIP")

# Ghi co nen
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
    f.write(b"Du lieu nen")
    
# Doc co nen
dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"], 
                                  compression_type="GZIP")
    \end{lstlisting}
\end{frame}

% Slide 13
\begin{frame}{Protocol Buffers (Protobuf) của TensorFlow}
    \begin{itemize}
        \item Bạn không lưu các chuỗi ngẫu nhiên (string) vào TFRecord. Bạn phải lưu \textbf{Protocol Buffers (Protobuf)}.
        \item Protobuf là định dạng Serialization cấu trúc do Google phát triển (nhẹ, nhanh và cấu trúc tĩnh mạnh mẽ hơn JSON/XML).
        \item TensorFlow đã định nghĩa sẵn Protobuf chuẩn mang tên \texttt{tf.train.Example}.
    \end{itemize}
\end{frame}

% Slide 14
\begin{frame}[fragile]{Cấu trúc \texttt{tf.train.Example}}
    \begin{itemize}
        \item Cấu trúc \texttt{Example} chứa đối tượng \texttt{Features}. 
        \item Mỗi thuộc tính dữ liệu chứa một kiểu list: \texttt{BytesList}, \texttt{FloatList}, \texttt{Int64List}.
    \end{itemize}
    \begin{lstlisting}[language=Python]
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
        }
    ))
# Ghi ra chuoi bytes de luu vao TFRecord
serialized = person_example.SerializeToString()
    \end{lstlisting}
\end{frame}

% Slide 15
\begin{frame}[fragile]{Phân tích cú pháp chuỗi Protobuf (Parsing)}
    \begin{itemize}
        \item Khi đọc chuỗi Serialized từ TFRecord, bạn cần dùng \texttt{tf.io.parse\_single\_example()} để chuyển ngược thành dữ liệu Tensor.
    \end{itemize}
    \begin{lstlisting}[language=Python]
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string), # Do dai thay doi -> SparseTensor
}

parsed_example = tf.io.parse_single_example(serialized, feature_description)
print(parsed_example["emails"].values) # Output cac email dang bytes
    \end{lstlisting}
\end{frame}

% Slide 16
\begin{frame}[fragile]{Lưu Ảnh (Images) vào TFRecord}
    \begin{itemize}
        \item Rất phổ biến trong Computer Vision! Ta mã hóa ảnh JPEG bằng \texttt{tf.io.encode\_jpeg()} rồi biến nó thành BytesList.
    \end{itemize}
    \begin{lstlisting}[language=Python]
import matplotlib.pyplot as plt

image = plt.imread("my_dog.jpg") # Doc hinh vao ma tran NumPy
encoded_img = tf.io.encode_jpeg(image)

example = Example(features=Features(feature={
    "image": Feature(bytes_list=BytesList(value=[encoded_img.numpy()]))
}))

# Luu vao tep
with tf.io.TFRecordWriter("image_data.tfrecord") as f:
    f.write(example.SerializeToString())
    \end{lstlisting}
\end{frame}

\section{Các Lớp Tiền xử lý của Keras}

% Slide 17
\begin{frame}{Các Lớp Tiền xử lý Keras (Preprocessing Layers)}
    \begin{itemize}
        \item Khái niệm: Đóng gói quá trình xử lý đặc trưng (Chuẩn hóa số, Mã hóa chữ One-hot...) thành \textbf{Lớp (Layer)} trực tiếp bên trong Kiến trúc Mô hình.
        \item Lợi ích to lớn: Bất kỳ ai sử dụng Mô hình tải về không cần biết dữ liệu của bạn trước đó đã được Normalize hay Encode ra sao! Họ truyền trực tiếp Data thô vào model.
        \item Bao gồm 3 nhóm chính: Tiền xử lý Đặc trưng Số, Phân loại/Văn bản, và Dữ liệu Ảnh.
    \end{itemize}
\end{frame}

% Slide 18
\begin{frame}[fragile]{Lớp chuẩn hóa \texttt{Normalization}}
    \begin{itemize}
        \item Cực kỳ hữu ích, thay thế việc tính thủ công trung bình (mean) và phương sai (variance) bằng tay (StandardScaler).
    \end{itemize}
    \begin{lstlisting}[language=Python]
normalization = keras.layers.Normalization()

# adapt() de tinh Trung binh va Phuong sai tren tap Train
normalization.adapt(X_train) 

# Dat vao mo hinh nhu mot tang binh thuong
model = keras.models.Sequential([
    normalization,
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(1)
])
    \end{lstlisting}
\end{frame}

% Slide 19
\begin{frame}[fragile]{Mã hóa biến Phân loại \& Rời rạc hóa}
    \begin{itemize}
        \item \textbf{Discretization:} Chia dữ liệu liên tục thành các thùng (Bins) phân loại. Ví dụ: Từ Độ tuổi (liên tục) $\rightarrow$ Khoảng Nhóm Tuổi (Thanh niên, Trung niên).
        \item \textbf{CategoryEncoding:} Chuyển giá trị nhãn rời rạc thành dạng Multi-hot.
    \end{itemize}
    \begin{lstlisting}[language=Python]
# Tao ma hoa Multi-hot hoac One-hot
cat_layer = keras.layers.CategoryEncoding(num_tokens=4, output_mode="one_hot")

print(cat_layer([1, 2, 0])) 
# Kq:
# [[0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [1. 0. 0. 0.]]
    \end{lstlisting}
\end{frame}

% Slide 20
\begin{frame}[fragile]{Tiền xử lý Dữ liệu Chuỗi Chữ (TextVectorization)}
    \begin{itemize}
        \item Trước khi mạng nơ-ron nhận văn bản, bạn cần tách câu thành các từ (tokenization) và gán cho mỗi từ 1 mã ID (số nguyên).
    \end{itemize}
    \begin{lstlisting}[language=Python]
text_layer = keras.layers.TextVectorization()

# adapt() de lay tu vung (Vocabulary)
text_layer.adapt(["I love machine learning", "Deep learning is great"])

# Xu ly
print(text_layer(["I love deep learning"])) 
# Output ID cua cac tu: [2 3 5 4] (Mo phong)
    \end{lstlisting}
    \begin{itemize}
        \item Output của \texttt{TextVectorization} có thể đi thẳng vào lớp \texttt{Embedding} (Nhúng từ) cực kỳ phổ biến trong Xử lý ngôn ngữ tự nhiên (NLP).
    \end{itemize}
\end{frame}

% Slide 21
\begin{frame}{Các lớp Tăng cường Ảnh (Image Augmentation)}
    \begin{itemize}
        \item \texttt{Resizing}: Thay đổi kích thước (crop, pad).
        \item \texttt{Rescaling}: Chia giá trị pixel cho 255.
        \item \texttt{RandomFlip}, \texttt{RandomRotation}, \texttt{RandomZoom}: Lật, xoay, phóng to ảnh tự động giúp mạng học được nhiều trường hợp dị thường chống over-fitting.
        \item Ghi chú: Keras sẽ \textbf{chỉ tự động kích hoạt} Data Augmentation khi Huấn luyện (\texttt{training=True}) và bỏ qua nó khi Inference (Dự đoán).
    \end{itemize}
\end{frame}

\section{Dự án TensorFlow Datasets (TFDS)}

% Slide 22
\begin{frame}{Dự án TensorFlow Datasets (TFDS)}
    \begin{itemize}
        \item Bạn mệt mỏi với việc tìm, tải tệp, tạo pipeline TFRecord cho các tập dữ liệu máy học phổ biến?
        \item \textbf{TFDS} sinh ra để tải bộ dữ liệu hoàn chỉnh thông qua duy nhất một dòng lệnh.
        \item Bao gồm rất nhiều tác vụ: Nhận diện ảnh (CIFAR, ImageNet), Xử lý Ngôn ngữ (IMDB, WMT), Audio...
        \item Được tối ưu mặc định cấu trúc \texttt{tf.data.Dataset}.
    \end{itemize}
\end{frame}

% Slide 23
\begin{frame}[fragile]{Sử dụng \texttt{tfds.load}}
    \begin{itemize}
        \item Hãy import thư viện \texttt{tensorflow\_datasets} (có thể cần pip install).
    \end{itemize}
    \begin{lstlisting}[language=Python]
import tensorflow_datasets as tfds

# Tai tap MNIST (Tu dong tao folder tensorflow_datasets)
dataset = tfds.load(name="mnist", split="train")

for item in dataset.take(1):
    image = item["image"]
    label = item["label"]
    print(image.shape, label) # (28, 28, 1) , tf.Tensor(4, ...)
    \end{lstlisting}
\end{frame}

% Slide 24
\begin{frame}[fragile]{Tiện ích Slicing (Cắt chia) và \texttt{as\_supervised}}
    \begin{itemize}
        \item TFDS cung cấp Slicing linh hoạt qua phần trăm (Percentage Slicing) như \texttt{'train[:75\%]'}.
        \item Bật \texttt{as\_supervised=True} để TFDS trả về trực tiếp Dữ liệu cặp Tuple \texttt{(features, label)} để lắp khít với hàm \texttt{fit()} của Keras!
    \end{itemize}
    \begin{lstlisting}[language=Python]
# Lay 75% du lieu train cho tap Train thuc te, 25% cho tap Validation
train_set, val_set = tfds.load(
    name="mnist", 
    split=["train[:75%]", "train[75%:]"],
    as_supervised=True
)

train_set = train_set.shuffle(10000).batch(32).prefetch(1)
model.fit(train_set, epochs=5, validation_data=val_set)
    \end{lstlisting}
\end{frame}

% Slide 25
\begin{frame}{Tổng hợp ứng dụng (Pipeline hoàn thiện)}
    \begin{columns}
        \begin{column}{0.6\textwidth}
            \begin{itemize}
                \item Bước 1: Nạp từ TFDS.
                \item Bước 2: Dùng \texttt{map()} chuyển đổi kích thước và chia 255.
                \item Bước 3: Cache, Shuffle, Batch, Prefetch.
                \item Bước 4: Đẩy vào mô hình có sẵn các Preprocessing Layers.
                \item Bước 5: Cấu trúc mô hình tự động kết nối hoàn hảo với API Dữ liệu luồng siêu lớn.
            \end{itemize}
        \end{column}
        \begin{column}{0.4\textwidth}
            \begin{center}
                \includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH13/Hình_13-5}\\
                \textit{Sơ đồ luồng (Pipeline)}
            \end{center}
        \end{column}
    \end{columns}
\end{frame}

% Slide 26
\begin{frame}{Tóm tắt}
    \begin{itemize}
        \item \textbf{API \texttt{tf.data}} là cách tối ưu nhất để nạp dữ liệu không vừa bộ nhớ, hỗ trợ Thread/Process song song.
        \item Sử dụng \textbf{TFRecord} và \textbf{Protobuf} để lưu trữ dữ liệu dạng nhị phân I/O hiệu năng cực cao.
        \item Xây dựng các khối biến đổi dữ liệu trực tiếp trong Mô hình bằng \textbf{Keras Preprocessing Layers} giúp đơn giản hóa khâu triển khai API ngoài thực tế.
        \item Sử dụng thư viện \textbf{TFDS} để gọi cực nhanh hàng ngàn tập dữ liệu kinh điển.
    \end{itemize}
\end{frame}

\end{document}
