import os

def generate_slides():
    tex_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap09.tex"
    
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

\title[Chương 9: Các kỹ thuật Học không giám sát]{Học Máy (Machine Learning)\\Chương 9: Các kỹ thuật Học không giám sát}
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

\section{Giới thiệu Học không giám sát \& K-Means}

% Slide 3
\begin{frame}{Giới thiệu Học không giám sát}
    \begin{itemize}
        \item Mặc dù đa số các ứng dụng học máy ngày nay dựa trên Học có giám sát, nhưng phần lớn dữ liệu trong thực tế là \textbf{không được gắn nhãn}.
        \item Chúng ta có các đặc trưng đầu vào $X$, nhưng không có nhãn $y$.
        \item Yann LeCun (Giám đốc AI của Meta): "Nếu trí thông minh là một chiếc bánh, thì học không giám sát sẽ là chiếc bánh, học có giám sát sẽ là lớp kem trên bánh."
        \item Tiềm năng của học không giám sát là vô cùng lớn nhưng chúng ta mới chỉ bắt đầu khai thác.
    \end{itemize}
\end{frame}

% Slide 4
\begin{frame}{Phân cụm (Clustering) là gì?}
    \begin{itemize}
        \item \textbf{Phân cụm:} Là tác vụ xác định các trường hợp tương tự nhau và gán chúng vào các "cụm" (clusters), hoặc các nhóm trường hợp tương tự.
        \item Giống như phân loại, mỗi trường hợp được gán vào một nhóm. Nhưng khác biệt ở chỗ đây là tác vụ \textit{không giám sát} (chúng ta không biết trước các nhóm là gì).
        \item Thuật toán tự động tìm ra cấu trúc ẩn bên trong dữ liệu dựa trên sự tương đồng của các đặc trưng.
    \end{itemize}
\end{frame}

% Slide 5
\begin{frame}{Phân loại so với phân cụm}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-1}\\
        \vspace{0.2cm}
        \textit{Hình 9-1. Phân loại (trái) - đã có nhãn lớp, so với Phân cụm (phải) - không có nhãn}
    \end{center}
\end{frame}

% Slide 6
\begin{frame}{Ứng dụng của Phân cụm}
    \begin{itemize}
        \item \textbf{Phân khúc khách hàng:} Phân cụm khách hàng dựa trên lịch sử mua hàng, hoạt động trên trang web để điều chỉnh chiến dịch tiếp thị (Hệ thống đề xuất).
        \item \textbf{Phân tích dữ liệu:} Phân cụm dữ liệu mới và phân tích từng cụm riêng biệt để hiểu sâu hơn.
        \item \textbf{Giảm chiều dữ liệu:} Thay vì dùng đặc trưng gốc, dùng "khoảng cách tới tâm các cụm" làm đặc trưng mới.
        \item \textbf{Phân đoạn hình ảnh (Image Segmentation):} Nhóm các điểm ảnh có màu sắc tương tự.
    \end{itemize}
\end{frame}

% Slide 7
\begin{frame}{Thuật toán K-Means}
    \begin{itemize}
        \item K-Means là một thuật toán phân cụm đơn giản, phổ biến và có khả năng phân cụm dữ liệu rất nhanh chóng, phân mảnh thành $k$ cụm.
        \item Được đề xuất đầu tiên vào năm 1957 bởi Stuart Lloyd tại Bell Labs.
        \item Người dùng phải định nghĩa trước số lượng cụm (tham số $k$).
    \end{itemize}
\end{frame}

% Slide 8
\begin{frame}{Tập dữ liệu chưa gán nhãn gồm 5 khối}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-2}\\
        \vspace{0.2cm}
        \textit{Hình 9-2. Một tập dữ liệu không được gắn nhãn bao gồm 5 khối trường hợp rõ rệt}
    \end{center}
\end{frame}

% Slide 9
\begin{frame}{Phép phân vùng Voronoi của K-Means}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-3}\\
        \vspace{0.2cm}
        \textit{Hình 9-3. K-Means chia không gian bằng phép phân vùng Voronoi (đường biên quyết định)}
    \end{center}
\end{frame}

% Slide 10
\begin{frame}{Cách thuật toán K-Means hoạt động}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-4}\\
        \vspace{0.2cm}
        \textit{Hình 9-4. Khởi tạo ngẫu nhiên tâm cụm $\rightarrow$ Gán điểm cho cụm $\rightarrow$ Cập nhật tâm cụm $\rightarrow$ Lặp lại}
    \end{center}
\end{frame}

% Slide 11
\begin{frame}{Vấn đề khởi tạo tâm cụm ngẫu nhiên}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-5}\\
        \vspace{0.2cm}
        \textit{Hình 9-5. K-Means có thể kẹt ở điểm tối ưu cục bộ nếu khởi tạo tâm cụm ngẫu nhiên không may mắn}
    \end{center}
\end{frame}

% Slide 12
\begin{frame}{Quán tính (Inertia) là gì?}
    \begin{itemize}
        \item Làm sao để biết một mô hình K-Means tốt hơn mô hình khác? (Do không có nhãn để tính độ chính xác).
        \item Dùng \textbf{Quán tính (Inertia)}: Tổng khoảng cách bình phương từ mỗi điểm dữ liệu đến tâm cụm gần nhất của nó.
        \item Mục tiêu của K-Means chính là tối thiểu hóa Quán tính. 
        \item K-Means++ (mặc định trong Scikit-Learn) dùng thuật toán khởi tạo thông minh để tránh bị kẹt ở điểm tối ưu cục bộ.
    \end{itemize}
\end{frame}

% Slide 13
\begin{frame}{Quán tính giảm theo các vòng lặp}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-6}\\
        \vspace{0.2cm}
        \textit{Hình 9-6. Quán tính giảm nhanh qua các bước lặp K-Means}
    \end{center}
\end{frame}

% Slide 14
\begin{frame}{Hậu quả của việc chọn số lượng cụm (k) sai}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-7}\\
        \vspace{0.2cm}
        \textit{Hình 9-7. Chọn k=3 (thiếu cụm) hoặc k=8 (chia cắt quá mức cụm thực)}
    \end{center}
\end{frame}

% Slide 15
\begin{frame}{Lựa chọn số cụm bằng phương pháp khuỷu tay (Elbow)}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-8}\\
        \vspace{0.2cm}
        \textit{Hình 9-8. Vẽ Quán tính theo số lượng cụm $k$. Chọn điểm "khuỷu tay" ($k=5$ hoặc $k=4$)}
    \end{center}
\end{frame}

\section{Điểm Silhouette, Ứng dụng K-Means \& DBSCAN}

% Slide 16
\begin{frame}{Đánh giá chất lượng bằng điểm Silhouette}
    \begin{itemize}
        \item Phương pháp "khuỷu tay" quá thô sơ. Đánh giá \textbf{điểm Silhouette (Silhouette Score)} chính xác hơn.
        \item Điểm Silhouette của mỗi điểm dữ liệu: $s = \frac{b - a}{\max(a, b)}$
        \begin{itemize}
            \item $a$: Khoảng cách trung bình đến các điểm trong cùng cụm.
            \item $b$: Khoảng cách trung bình đến các điểm trong cụm gần nhất tiếp theo.
        \end{itemize}
        \item Giá trị dao động từ -1 đến +1.
        \item Gần +1: Điểm nằm sâu trong cụm.
        \item Gần 0: Điểm nằm ở rìa cụm.
        \item Gần -1: Điểm có thể bị gán sai cụm.
    \end{itemize}
\end{frame}

% Slide 17
\begin{frame}{Biểu đồ điểm Silhouette theo k}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-9}\\
        \vspace{0.2cm}
        \textit{Hình 9-9. Chọn số lượng cụm $k$ tối ưu (đỉnh cao nhất, ví dụ $k=4$ hoặc $k=5$)}
    \end{center}
\end{frame}

% Slide 18
\begin{frame}{Phân tích biểu đồ dao (Silhouette plots)}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-10}\\
        \vspace{0.2cm}
        \textit{Hình 9-10. Hình dao cho mỗi cụm. Dao càng rộng, cụm càng tốt. Ranh giới đỏ là điểm Silhouette trung bình.}
    \end{center}
\end{frame}

% Slide 19
\begin{frame}{Giới hạn của K-Means}
    \begin{itemize}
        \item Cần chạy thuật toán nhiều lần để tránh tối ưu cục bộ.
        \item Phải chỉ định trước số cụm $k$.
        \item \textbf{Nhược điểm lớn nhất:} K-Means hoạt động rất kém khi các cụm có kích thước khác nhau nhiều, mật độ khác nhau, hoặc hình dạng không phải hình cầu (ví dụ: hình elip).
        \item Lý do: K-Means chỉ quan tâm đến \textbf{khoảng cách} tới tâm cụm.
    \end{itemize}
\end{frame}

% Slide 20
\begin{frame}{K-Means gặp khó khăn với các cụm hình elip}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-11}\\
        \vspace{0.2cm}
        \textit{Hình 9-11. K-Means phân cụm sai lầm khi cắt ngang khối hình elip tự nhiên}
    \end{center}
\end{frame}

% Slide 21
\begin{frame}{Ứng dụng: Phân đoạn hình ảnh (Image Segmentation)}
    \begin{itemize}
        \item Phân đoạn hình ảnh: Chia hình ảnh thành nhiều phân đoạn.
        \item Ứng dụng phổ biến: \textbf{Phân đoạn màu sắc}. Nhóm các pixel có màu giống nhau vào cùng một cụm K-Means.
        \item Thay thế giá trị màu của mọi pixel bằng màu của tâm cụm K-Means, ta có thể giảm số lượng màu xuống $k$ màu, tạo hiệu ứng ảnh như tranh vẽ (Posterization).
    \end{itemize}
\end{frame}

% Slide 22
\begin{frame}{Phân đoạn màu sắc hình ảnh bọ rùa}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-12}\\
        \vspace{0.2cm}
        \textit{Hình 9-12. Phân đoạn hình ảnh sử dụng K-means với các số lượng cụm màu khác nhau ($k=10, 8, 6, 4, 2$)}
    \end{center}
\end{frame}

% Slide 23
\begin{frame}{Ứng dụng: Tiền xử lý để học bán giám sát}
    \begin{itemize}
        \item Khi bạn có rất nhiều dữ liệu nhưng chỉ một phần rất nhỏ có nhãn (vd: chỉ gán nhãn được 50 tấm ảnh MNIST).
        \item Thay vì gán nhãn 50 ảnh ngẫu nhiên, ta dùng K-Means chia tập dữ liệu thành 50 cụm.
        \item Sau đó, tìm hình ảnh \textbf{gần tâm cụm nhất} (hình ảnh đại diện) và chỉ gán nhãn cho 50 hình ảnh này.
        \item Cuối cùng, có thể \textbf{truyền lan nhãn} (Label Propagation) sang tất cả các ảnh khác trong cùng cụm.
    \end{itemize}
\end{frame}

% Slide 24
\begin{frame}{Lấy 50 hình ảnh đại diện bằng K-Means}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-13}\\
        \vspace{0.2cm}
        \textit{Hình 9-13. 50 hình ảnh chữ số đại diện (một hình ảnh cho mỗi cụm). Việc gán nhãn chúng mang lại ý nghĩa cao hơn chọn ngẫu nhiên.}
    \end{center}
\end{frame}

% Slide 25
\begin{frame}{Thuật toán phân cụm DBSCAN}
    \begin{itemize}
        \item \textbf{DBSCAN (Density-Based Spatial Clustering of Applications with Noise):} Dựa trên ý tưởng các cụm là các khu vực có \textbf{mật độ cao} phân tách bởi khu vực có \textbf{mật độ thấp}.
        \item Không giống K-Means, DBSCAN xác định được các cụm có \textbf{hình dạng tùy ý} và có thể phát hiện điểm ngoại lai (nhiễu).
    \end{itemize}
\end{frame}

% Slide 26
\begin{frame}{Cơ chế hoạt động của DBSCAN}
    \begin{itemize}
        \item Nhìn vào bán kính khoảng cách $\varepsilon$ (\textbf{eps}) xung quanh mỗi điểm dữ liệu. Khu vực này gọi là \textit{$\varepsilon$-neighborhood}.
        \item Nếu khu vực này có chứa ít nhất \textbf{min\_samples} điểm, điểm đó được coi là một \textbf{điểm lõi (core instance)}.
        \item Bất kỳ điểm nào nằm trong vùng lân cận của một điểm lõi cũng thuộc cùng cụm. (Các điểm lõi gần nhau kết hợp lại thành cụm lớn).
        \item Nếu điểm không phải là lõi và không nằm trong vùng của điểm lõi nào $\rightarrow$ Nó bị coi là Dị thường (Anomaly / Noise).
    \end{itemize}
\end{frame}

% Slide 27
\begin{frame}{DBSCAN với 2 bán kính vùng lân cận khác nhau}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-14}\\
        \vspace{0.2cm}
        \textit{Hình 9-14. DBSCAN bắt được hình dạng "mặt trăng" cực tốt. eps nhỏ (trái) tạo nhiều cụm đứt gãy. eps lớn hơn (phải) tạo ra 2 cụm hoàn hảo.}
    \end{center}
\end{frame}

% Slide 28
\begin{frame}{Ưu điểm và nhược điểm của DBSCAN}
    \begin{itemize}
        \item \textbf{Ưu điểm:}
        \begin{itemize}
            \item Không cần khai báo số cụm $k$ trước.
            \item Kháng nhiễu cực tốt (tự phát hiện và loại bỏ điểm ngoại lai).
            \item Tìm được cụm có hình dạng hình học phức tạp (chữ U, hình vành khăn, mặt trăng...).
        \end{itemize}
        \item \textbf{Nhược điểm:}
        \begin{itemize}
            \item DBSCAN \textbf{không có} phương thức \texttt{predict()}, nó không thể tự gán cụm cho một trường hợp mới sinh ra.
            \item Kém hiệu quả nếu các cụm trong tập dữ liệu có \textbf{mật độ khác biệt nhau quá lớn}.
        \end{itemize}
    \end{itemize}
\end{frame}

% Slide 29
\begin{frame}[fragile]{Dự đoán với DBSCAN (Kết hợp KNN)}
    Vì DBSCAN không có \texttt{predict()}, ta có thể dùng thuật toán phân loại KNN để huấn luyện trên các lõi DBSCAN vừa tìm được.
    \begin{lstlisting}[language=Python]
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50)
# dbscan.components_ chua cac diem loi
# dbscan.labels_ chua nhan cua cac diem do
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

import numpy as np
X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
print(knn.predict(X_new))
# Output: [1 0 1 0]
    \end{lstlisting}
\end{frame}

% Slide 30
\begin{frame}{Đường biên quyết định của phân loại KNN trên tập DBSCAN}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-15}\\
        \vspace{0.2cm}
        \textit{Hình 9-15. Dùng KNN tạo đường biên phân loại mới dựa trên các cụm đã tìm thấy bởi DBSCAN}
    \end{center}
\end{frame}

\section{Mô hình Hỗn hợp Gaussian (GMM)}

% Slide 31
\begin{frame}{Mô hình Hỗn hợp Gaussian (GMM) là gì?}
    \begin{itemize}
        \item \textbf{Gaussian Mixture Model (GMM):} Là một mô hình xác suất giả định rằng tất cả các điểm dữ liệu được sinh ra từ một \textit{hỗn hợp (mixture)} của một số hữu hạn các phân phối Gaussian (Phân phối chuẩn) với các tham số chưa biết.
        \item Nó đại diện cho các cụm có hình dạng \textbf{hình elip} (có thể dẹt, nghiêng, xoay) với nhiều kích thước khác nhau.
        \item GMM là một kỹ thuật tuyệt vời để khắc phục nhược điểm của K-Means với các cụm hình elip.
    \end{itemize}
\end{frame}

% Slide 32
\begin{frame}{Thuật toán Cực đại hóa Kỳ vọng (EM)}
    \begin{itemize}
        \item Để khớp GMM với tập dữ liệu, người ta dùng thuật toán \textbf{Expectation-Maximization (EM)}.
        \item Có nhiều điểm tương đồng với K-Means: khởi tạo ngẫu nhiên tâm và lặp lại cho đến khi hội tụ.
        \item EM không chỉ tìm tâm cụm (trung bình $\mu$), mà còn tìm cả kích thước, hình dạng, và hướng của chúng (ma trận hiệp phương sai $\Sigma$), cũng như xác suất tương đối của chúng (trọng số $\phi$).
        \item Khác K-Means (gán cứng), EM thực hiện \textbf{gán mềm (soft assignments)}: tính \textit{xác suất} một điểm thuộc về mỗi cụm.
    \end{itemize}
\end{frame}

% Slide 33
\begin{frame}[fragile]{Khởi tạo GMM trong Scikit-Learn}
    Giống K-Means, bạn phải cung cấp số lượng cụm $k$ (biến \texttt{n\_components}):
    \begin{lstlisting}[language=Python]
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)

print(gm.weights_)
print(gm.means_)
print(gm.covariances_)
    \end{lstlisting}
    Thuật toán EM rất dễ kẹt ở điểm tối ưu cục bộ, nên ta chạy nhiều lần (\texttt{n\_init=10}) và giữ lại mô hình tốt nhất.
\end{frame}

% Slide 34
\begin{frame}{Các giá trị trung bình và đường biên quyết định của GMM}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-16}\\
        \vspace{0.2cm}
        \textit{Hình 9-16. GMM tìm ra các cụm hình elip bị kéo giãn và nghiêng cực kỳ chính xác}
    \end{center}
\end{frame}

% Slide 35
\begin{frame}{Hạn chế dạng ma trận hiệp phương sai}
    Nếu tập dữ liệu lớn với nhiều chiều, GMM cực kỳ khó huấn luyện vì ma trận hiệp phương sai bành trướng. Có thể hạn chế kiểu hình dạng bằng \texttt{covariance\_type}:
    \begin{itemize}
        \item \texttt{"full"} (mặc định): Hình elip tự do hướng theo bất kỳ góc nào.
        \item \texttt{"tied"}: Tất cả các cụm phải dùng chung kích thước, góc xoay, hình dáng.
        \item \texttt{"spherical"}: Cụm phải là hình cầu hoàn hảo (tương tự K-Means, nhưng bán kính có thể khác nhau).
        \item \texttt{"diag"}: Cụm hình elip nhưng trục phải song song với các trục tọa độ.
    \end{itemize}
\end{frame}

% Slide 36
\begin{frame}{GMM với các cụm liên kết (tied) và hình cầu (spherical)}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-17}\\
        \vspace{0.2cm}
        \textit{Hình 9-17. Bên trái: tied (elip giống hệt nhau). Bên phải: spherical (hình cầu hoàn hảo)}
    \end{center}
\end{frame}

% Slide 37
\begin{frame}{Ước tính mật độ sinh ngẫu nhiên (Generative process)}
    \begin{itemize}
        \item GMM là một **Mô hình sinh ngẫu nhiên (Generative model)**.
        \item Nó không chỉ phân cụm, mà nó mô hình hóa hoàn toàn hàm Mật độ xác suất (PDF) của hệ thống.
        \item Do đó, bạn có thể sinh ra (generate) các điểm dữ liệu \textbf{mới} hoàn toàn tự nhiên trông giống như phân phối gốc bằng hàm \texttt{gm.sample(50)}.
    \end{itemize}
\end{frame}

% Slide 38
\begin{frame}{Phát hiện dị thường (Anomaly Detection) bằng GMM}
    \begin{itemize}
        \item Bằng cách ước lượng hàm PDF tổng quát, bất kỳ điểm dữ liệu nào nằm trong khu vực có \textbf{mật độ cực thấp} (low density) đều bị tình nghi là Dị thường (Anomaly / Outlier).
        \item Dùng hàm \texttt{gm.score\_samples(X)} để đánh giá mức độ mật độ (log-likelihood) tại điểm đó. 
        \item Nếu mật độ thấp hơn một ngưỡng phân vị, ta đánh dấu điểm đó là dị thường.
    \end{itemize}
\end{frame}

% Slide 39
\begin{frame}[fragile]{Cách tính ngưỡng mật độ phát hiện dị thường}
    Ví dụ, ta quyết định $4\%$ số điểm có mật độ thấp nhất là dị thường:
    \begin{lstlisting}[language=Python]
import numpy as np

# score_samples tra ve mat do log (log PDF)
densities = gm.score_samples(X)

# Tim nguong mat do tuong ung voi percentile thu 4
density_threshold = np.percentile(densities, 4)

# Danh dau cac diem nam duoi nguong do la Di thuong (anomalies)
anomalies = X[densities < density_threshold]
    \end{lstlisting}
\end{frame}

% Slide 40
\begin{frame}{Phát hiện dị thường sử dụng GMM}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-18}\\
        \vspace{0.2cm}
        \textit{Hình 9-18. Các ngôi sao màu đỏ là các điểm dị thường (Anomalies) nằm ở khu vực mật độ thấp}
    \end{center}
\end{frame}

% Slide 41
\begin{frame}{Hàm tham số của mô hình}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-19}\\
        \vspace{0.2cm}
        \textit{Hình 9-19. Không gian hàm PDF của Mô hình hỗn hợp Gaussian từ một cụm đơn giản}
    \end{center}
\end{frame}

% Slide 42
\begin{frame}{Lựa chọn số lượng cụm trong GMM}
    \begin{itemize}
        \item Với K-Means ta dùng Quán tính hay Silhouette Score.
        \item Tuy nhiên, các số liệu này không đáng tin cậy với GMM (chúng thiên vị các cụm hình cầu).
        \item Với GMM, ta dùng các \textbf{Tiêu chí Thông tin (Information Criteria)}: \textbf{BIC} (Bayesian Information Criterion) hoặc \textbf{AIC} (Akaike Information Criterion).
    \end{itemize}
\end{frame}

% Slide 43
\begin{frame}{Công thức tính toán AIC và BIC}
    \begin{itemize}
        \item Cả BIC và AIC phạt (penalty) mô hình nếu có quá nhiều tham số phải học (ngăn chặn quá khớp), và thưởng cho mô hình khớp tốt dữ liệu.
        \item Mô hình tốt nhất là mô hình có AIC/BIC \textbf{thấp nhất}.
    \end{itemize}
    $$ BIC = \log(m)p - 2\log(\hat{L}) $$
    $$ AIC = 2p - 2\log(\hat{L}) $$
    Trong đó:
    \begin{itemize}
        \item $m$ là số mẫu dữ liệu, $p$ là số lượng tham số trong mô hình GMM.
        \item $\hat{L}$ là hàm giá trị khả năng cực đại (Likelihood) của mô hình.
    \end{itemize}
\end{frame}

% Slide 44
\begin{frame}{Biểu đồ AIC và BIC cho các k khác nhau}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-20}\\
        \vspace{0.2cm}
        \textit{Hình 9-20. BIC và AIC đạt giá trị cực tiểu rõ ràng tại $k=3$. Đây là số lượng cụm tối ưu nhất.}
    \end{center}
\end{frame}

\section{Mô hình Hỗn hợp Gaussian Bayes}

% Slide 45
\begin{frame}{Mô hình Hỗn hợp Gaussian Bayes (BGM)}
    \begin{itemize}
        \item Khá vất vả khi phải chạy thử thủ công một loạt các k để đo BIC/AIC.
        \item Thay vì vậy, \textbf{Mô hình Hỗn hợp Gaussian Bayes (Bayesian Gaussian Mixture)} ra đời.
        \item Bạn khởi tạo nó với một số lượng cụm $k$ \textbf{dư dả} (vd: lớn hơn thực tế bạn nghĩ).
        \item Thuật toán sẽ tự động gán trọng số (weights) của các cụm không cần thiết bằng $0$ hoặc xấp xỉ $0$.
        \item $\rightarrow$ BGM tự động khám phá ra số lượng cụm $k$ thực tế!
    \end{itemize}
\end{frame}

% Slide 46
\begin{frame}[fragile]{Sử dụng BGM bằng Scikit-Learn}
    \begin{lstlisting}[language=Python]
from sklearn.mixture import BayesianGaussianMixture

# Thiet lap k=10 du gia
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)

# Kiem tra trong so cac cum
np.round(bgm.weights_, 2)
# Output: array([0.4 , 0.21, 0.4 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ])
    \end{lstlisting}
    \begin{itemize}
        \item Chỉ có 3 cụm đầu tiên có trọng số lớn hơn $0$. BGM đã triệt tiêu 7 cụm dư thừa và tự nhận ra bài toán chỉ có 3 cụm.
    \end{itemize}
\end{frame}

% Slide 47
\begin{frame}{Tiên nghiệm Dirichlet (Dirichlet Prior)}
    \begin{itemize}
        \item BGM đạt được sự tự động này là nhờ áp dụng \textbf{phân phối Dirichlet} vào quá trình cập nhật Bayes.
        \item Thông qua tham số \texttt{weight\_concentration\_prior}, bạn có thể cho thuật toán biết niềm tin (Prior) của bạn:
        \begin{itemize}
            \item Nếu đặt bằng tỷ lệ nhỏ (mặc định \texttt{1/n\_components}), thuật toán sẽ mạnh tay loại bỏ cụm (tìm ít cụm).
            \item Nếu đặt số lớn, thuật toán sẽ cố gắng giữ lại tất cả các cụm (như GMM thường).
        \end{itemize}
    \end{itemize}
\end{frame}

% Slide 48
\begin{frame}{Lưu ý khi dùng BGM với dữ liệu hình dạng tùy ý}
    \begin{center}
        \includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH09/Hinh_9-21}\\
        \vspace{0.2cm}
        \textit{Hình 9-21. BGM phân cụm lỗi đối với tập dữ liệu Moons. GMM sinh ra các cục elip, nên một hình cong trăng khuyết sẽ bị nó chia làm nhiều cục elip nhỏ thay vì 1 cụm.}
    \end{center}
\end{frame}

% Slide 49
\begin{frame}{Các thuật toán phát hiện dị thường khác}
    Bên cạnh GMM, một số thuật toán chuyên biệt cho Phát hiện Dị thường (Anomaly Detection):
    \begin{itemize}
        \item \textbf{PCA (Phân tích thành phần chính):} Sai số tái tạo (Reconstruction Error) của ngoại lệ thường cao hơn nhiều điểm bình thường.
        \item \textbf{Fast-MCD (Minimum Covariance Determinant):} Rất giỏi trong việc dọn dẹp các điểm nhiễu ngoại lai trước khi chạy thuật toán học máy.
        \item \textbf{Isolation Forest:} Chuyên gia phát hiện ngoại lệ trong tập dữ liệu không gian nhiều chiều.
        \item \textbf{LOF (Local Outlier Factor):} Dựa trên sự khác biệt mật độ cục bộ.
    \end{itemize}
\end{frame}

% Slide 50
\begin{frame}{Tóm tắt Chương 9: K-Means \& Ứng dụng}
    \begin{itemize}
        \item Học không giám sát khai phá dữ liệu chưa gắn nhãn, hứa hẹn mở ra cuộc cách mạng AI.
        \item \textbf{K-Means} là ông vua tốc độ. Sử dụng quán tính, quy tắc khuỷu tay và điểm Silhouette để dò $k$. Thích ứng tốt nhất với cụm dạng cầu, kích thước đều.
        \item \textbf{Ứng dụng quan trọng:} Rất tuyệt vời để Phân đoạn hình ảnh và Tiền xử lý cho việc Gán nhãn Bán giám sát (Semi-supervised).
    \end{itemize}
\end{frame}

% Slide 51
\begin{frame}{Tóm tắt Chương 9: DBSCAN \& GMM}
    \begin{itemize}
        \item \textbf{DBSCAN} không quan tâm số cụm, dựa vào mật độ, có thể theo vết các cụm hình dạng phức tạp và kháng nhiễu tốt.
        \item \textbf{GMM (Hỗn hợp Gaussian)} khắc phục nhược điểm về hình elip, rất mạnh trong **Phát hiện dị thường**.
        \item \textbf{BGM (Hỗn hợp Gaussian Bayes)} giúp tự động hóa quá trình dò tìm số lượng cụm $k$ nhờ tiêu chuẩn học máy thống kê.
    \end{itemize}
\end{frame}

% Slide 52
\begin{frame}{Hỏi \& Đáp}
    \begin{center}
        \Large{\textbf{CẢM ƠN CÁC BẠN ĐÃ LẮNG NGHE!}}\\
        \vspace{1cm}
        Q \& A
    \end{center}
\end{frame}

\end{document}
"""

    # Ghi nội dung vào file tex
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"Da tao thanh cong: {tex_path}")

if __name__ == "__main__":
    generate_slides()
