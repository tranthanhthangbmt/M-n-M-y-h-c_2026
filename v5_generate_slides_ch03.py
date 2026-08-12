import os

tex_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap03.tex"

latex_code = r"""\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}
\usecolortheme{default}
\setbeamertemplate{caption}{\raggedright\insertcaption\par}
\usepackage{fontspec}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{xcolor}

\title[Chương 3: Phân loại]{DỰ ÁN HỌC MÁY TỪ ĐẦU ĐẾN CUỐI \\ \vspace{0.5cm} \Large Chương 3 - Phân loại (Classification)}
\author{TS. Trần Thành Thắng}
\institute{Đại học Đông Á}
\date{\today}

\begin{document}

\begin{frame}
\titlepage
\end{frame}

\begin{frame}{Nội dung Chương trình}
\tableofcontents[hideallsubsections]
\end{frame}

\begin{frame}{Mục tiêu bài học}
\begin{itemize}
    \item Nắm vững khái niệm Phân loại (Classification) trong Học máy.
    \item Hiểu rõ và thực hành với bộ dữ liệu MNIST ("Hello World" của học máy).
    \item Đánh giá mô hình bằng các thước đo hiệu suất (Ma trận nhầm lẫn, Precision, Recall, F1-score, ROC AUC).
    \item Phân biệt và thực hiện Phân loại Đa lớp (Multiclass), Đa nhãn (Multilabel) và Đa đầu ra (Multioutput).
    \item Kỹ năng phân tích lỗi để cải thiện hiệu suất mô hình.
\end{itemize}
\end{frame}

\section{1. Giới thiệu Phân loại, MNIST \& Đánh giá Hiệu suất cơ bản}

\begin{frame}
\vfill\centering\LARGE\textbf{1. Giới thiệu Phân loại, MNIST \& Đánh giá Hiệu suất cơ bản}\vfill
\end{frame}

\begin{frame}{Giới thiệu Bộ dữ liệu MNIST}
\begin{itemize}
    \item Các tác vụ học có giám sát phổ biến nhất là Hồi quy (Regression) và Phân loại (Classification).
    \item Chúng ta sẽ sử dụng bộ dữ liệu \textbf{MNIST}, gồm 70.000 hình ảnh nhỏ của các chữ số viết tay bởi học sinh và nhân viên Cục Thống kê Hoa Kỳ.
    \item MNIST được nghiên cứu nhiều đến mức nó được gọi là \textbf{"Hello World" của học máy}.
    \item Mục tiêu: Phân loại (nhận dạng) chữ số trong từng bức ảnh.
\end{itemize}
\end{frame}

\begin{frame}{Hình ảnh chữ số MNIST}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Mỗi hình ảnh có kích thước 28x28 pixel, tương đương 784 đặc trưng (pixels).
    \item Mỗi pixel đại diện cho cường độ xám, từ 0 (trắng) đến 255 (đen).
    \item Trong ảnh là một ví dụ trực quan hóa một chữ số 5 từ bộ dữ liệu.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-1.png}
\caption{Hình 3-1. Ví dụ về hình ảnh MNIST}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Sự đa dạng của chữ số MNIST}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Các chữ số được viết với nhiều phong cách, nét thanh, nét đậm khác nhau.
    \item Để minh họa độ phức tạp của bài toán phân loại, hình bên cạnh liệt kê một số chữ số mẫu từ tập dữ liệu.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-2.png}
\caption{Hình 3-2. Các chữ số từ bộ dữ liệu MNIST}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Phân chia tập huấn luyện và tập kiểm thử}
\begin{itemize}
    \item \textbf{Nguyên tắc vàng:} Luôn tạo một tập kiểm thử (Test set) và tách biệt nó trước khi kiểm tra kỹ lưỡng dữ liệu.
    \item May mắn là tập dữ liệu MNIST tải từ OpenML đã được chia sẵn:
    \begin{itemize}
        \item Tập huấn luyện (Train set): 60.000 hình ảnh đầu tiên.
        \item Tập kiểm thử (Test set): 10.000 hình ảnh cuối cùng.
    \end{itemize}
    \item Tập huấn luyện đã được xáo trộn (shuffled), đảm bảo các thuật toán học (đặc biệt là các thuật toán học tuần tự) không bị nhận nhiều trường hợp giống nhau liên tiếp.
\end{itemize}
\end{frame}

\begin{frame}
\vfill\centering\LARGE\textbf{Huấn luyện bộ phân loại nhị phân (Binary Classification)}\vfill
\end{frame}

\begin{frame}{Bài toán: Bộ phát hiện chữ số 5}
\begin{itemize}
    \item Hãy đơn giản hóa bài toán: Chúng ta chỉ cần phát hiện xem một chữ số có phải là "5" hay không.
    \item Đây là ví dụ về \textbf{Phân loại Nhị phân (Binary Classification)}: Phân biệt giữa 2 lớp (5 và không-phải-5).
    \item Mục tiêu (Label) mới: True (nếu là số 5), False (nếu là số khác).
\end{itemize}
\end{frame}

\begin{frame}{Thuật toán Stochastic Gradient Descent (SGD)}
\begin{itemize}
    \item SGD (Xuống dốc ngẫu nhiên) là một thuật toán tối ưu rất hiệu quả để xử lý các tập dữ liệu lớn.
    \item SGD xử lý các trường hợp huấn luyện độc lập, từng trường hợp một (phù hợp với học trực tuyến - online learning).
    \item Lớp \texttt{SGDClassifier} trong Scikit-Learn giúp ta dễ dàng huấn luyện mô hình này.
\end{itemize}
\end{frame}

\begin{frame}
\vfill\centering\LARGE\textbf{Các thước đo hiệu suất}\vfill
\end{frame}

\begin{frame}{Đo độ chính xác bằng kiểm định chéo}
\begin{itemize}
    \item Giống như hồi quy, ta có thể dùng kiểm định chéo (Cross-Validation) để đánh giá mô hình phân loại.
    \item \textbf{Kết quả:} \texttt{SGDClassifier} đạt độ chính xác (Accuracy) trên 95\% cho mọi fold!
    \item Nghe có vẻ tuyệt vời, nhưng hãy xem xét cẩn thận hơn.
\end{itemize}
\end{frame}

\begin{frame}{Sự đánh lừa của Độ chính xác (Accuracy)}
\begin{itemize}
    \item Nếu ta tạo một "bộ phân loại ngốc nghếch" (Dummy Classifier) luôn đoán "không-phải-5", thì độ chính xác của nó cũng trên 90\%!
    \item Tại sao? Vì chỉ có 10\% chữ số trong tập dữ liệu là số 5.
    \item \textbf{Kết luận:} Độ chính xác (Accuracy) không phải là thước đo tốt cho các \textbf{tập dữ liệu bị lệch (skewed datasets)}.
\end{itemize}
\end{frame}

\begin{frame}{Ma trận nhầm lẫn (Confusion Matrix)}
\begin{itemize}
    \item Ma trận nhầm lẫn (CM) là một cách tốt hơn nhiều để đánh giá hiệu suất.
    \item Ý tưởng chung: Đếm số lần các trường hợp của lớp A bị phân loại nhầm thành lớp B.
    \item Mỗi hàng là một lớp thực tế, mỗi cột là một lớp được dự đoán.
\end{itemize}
\end{frame}

\begin{frame}{Cấu trúc của Ma trận nhầm lẫn}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item \textbf{True Negative (TN):} Âm tính thật.
    \item \textbf{False Positive (FP):} Dương tính giả (Lỗi loại I).
    \item \textbf{False Negative (FN):} Âm tính giả (Lỗi loại II).
    \item \textbf{True Positive (TP):} Dương tính thật.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-3.png}
\caption{Hình 3-3. Ma trận nhầm lẫn minh họa}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Độ chính xác (Precision) và Độ nhạy (Recall)}
\begin{itemize}
    \item \textbf{Precision (Độ chính xác của lớp Dương tính):} Trong số các trường hợp mô hình dự đoán là Tích cực (5), có bao nhiêu phần trăm là đúng?
    \begin{equation*}
        Precision = \frac{TP}{TP + FP}
    \end{equation*}
    \item \textbf{Recall (Độ nhạy / Tỷ lệ dương tính đúng):} Trong số các trường hợp Tích cực thực sự, mô hình đã phát hiện được bao nhiêu phần trăm?
    \begin{equation*}
        Recall = \frac{TP}{TP + FN}
    \end{equation*}
\end{itemize}
\end{frame}

\begin{frame}{Sự kết hợp: Điểm F1 (F1 Score)}
\begin{itemize}
    \item Điểm F1 là \textbf{trung bình điều hòa (Harmonic Mean)} của Precision và Recall.
    \item Điểm F1 chỉ cao khi CẢ Precision và Recall đều cao.
    \begin{equation*}
        F1 = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}} = 2 \times \frac{Precision \times Recall}{Precision + Recall}
    \end{equation*}
    \item Điểm F1 ưu tiên các bộ phân loại có Precision và Recall tương tự nhau.
\end{itemize}
\end{frame}

\section{2. Sự đánh đổi Hiệu suất \& Phân loại Đa lớp}

\begin{frame}
\vfill\centering\LARGE\textbf{2. Sự đánh đổi Độ chính xác và Độ nhạy}\vfill
\end{frame}

\begin{frame}{Ngưỡng quyết định (Decision Threshold)}
\begin{itemize}
    \item Làm sao mô hình quyết định một mẫu là Tích cực hay Tiêu cực?
    \item Nó tính toán một \textbf{điểm số quyết định (decision score)}.
    \item Nếu điểm số > Ngưỡng (Threshold) $\rightarrow$ Tích cực.
    \item Nếu điểm số < Ngưỡng (Threshold) $\rightarrow$ Tiêu cực.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa sự đánh đổi (Trade-off)}
\begin{columns}
\column{0.4\textwidth}
\begin{itemize}
    \item Nếu \textbf{Tăng ngưỡng}: Giảm False Positive (Precision tăng), nhưng tăng False Negative (Recall giảm).
    \item Nếu \textbf{Giảm ngưỡng}: Giảm False Negative (Recall tăng), nhưng tăng False Positive (Precision giảm).
\end{itemize}
\column{0.6\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-4.png}
\caption{Hình 3-4. Sự đánh đổi độ chính xác/độ nhạy theo ngưỡng quyết định}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Các biểu đồ phân tích Precision/Recall}
\begin{itemize}
    \item Để chọn được một Threshold phù hợp nhất với bài toán thực tế, ta cần trực quan hóa Precision và Recall.
    \item \textbf{Đường cong số 1:} Vẽ Precision và Recall theo từng giá trị ngưỡng (Threshold).
    \item \textbf{Đường cong số 2:} Vẽ trực tiếp Precision theo Recall để xem đường tiệm cận.
    \item Ta nên chọn điểm ngưỡng ngay trước khi Precision bị sụt giảm quá mạnh (ví dụ: ở Recall khoảng 60\%-80\%).
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Đường cong Precision/Recall}
\begin{columns}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-5.png}
\caption{Hình 3-5. Precision và Recall so với Threshold}
\end{figure}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-6.png}
\caption{Hình 3-6. Precision so với Recall}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Đường cong ROC (Receiver Operating Characteristic)}
\begin{itemize}
    \item Rất phổ biến trong bộ phân loại nhị phân.
    \item Vẽ biểu đồ **Tỷ lệ dương tính đúng (TPR / Recall)** so với **Tỷ lệ dương tính giả (FPR / Fall-out)**.
    \item FPR là tỷ lệ các mẫu Âm tính bị nhận diện sai là Dương tính.
    \item Sự đánh đổi: TPR càng cao (mô hình càng nhạy), thì FPR càng cao (càng nhiều báo động giả).
\end{itemize}
\end{frame}

\begin{frame}{ROC AUC: So sánh các bộ phân loại}
\begin{itemize}
    \item Để so sánh 2 mô hình, ta đo \textbf{Diện tích dưới đường cong (AUC - Area Under the Curve)}.
    \item Bộ phân loại hoàn hảo: ROC AUC = 1.
    \item Bộ phân loại ngẫu nhiên: ROC AUC = 0.5.
    \item Mô hình có ROC gần góc trên bên trái nhất (AUC cao) là mô hình tốt nhất.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Đường cong ROC của SGD và Random Forest}
\begin{columns}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-7.png}
\caption{Hình 3-7. Đường cong ROC của mô hình SGD}
\end{figure}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-8.png}
\caption{Hình 3-8. So sánh ROC: SGD vs Random Forest}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}
\vfill\centering\LARGE\textbf{Phân loại Đa lớp (Multiclass Classification)}\vfill
\end{frame}

\begin{frame}{Chiến lược Một-đối-phần-còn-lại (OvR / OvA)}
\begin{itemize}
    \item Phân biệt giữa nhiều hơn hai lớp.
    \item \textbf{OvR (One-vs-Rest):} Huấn luyện 10 bộ phân loại nhị phân (từ chữ số 0 đến 9).
    \item Bộ phân loại "0": Số 0 vs Các số còn lại.
    \item Khi phân loại: Lấy điểm quyết định từ cả 10 mô hình và chọn lớp có điểm cao nhất.
\end{itemize}
\end{frame}

\begin{frame}{Chiến lược Một-đối-một (OvO)}
\begin{itemize}
    \item \textbf{OvO (One-vs-One):} Huấn luyện một bộ phân loại nhị phân cho mỗi CẶP chữ số.
    \item 0 vs 1, 0 vs 2, 1 vs 2...
    \item Tổng số mô hình cần huấn luyện: $N \times (N-1) / 2 = 45$ bộ phân loại!
    \item Phù hợp cho các thuật toán khó mở rộng với lượng dữ liệu lớn (như Support Vector Machine - SVC).
\end{itemize}
\end{frame}

\begin{frame}{Phân loại đa lớp tự động trong Scikit-Learn}
\begin{itemize}
    \item \texttt{Scikit-Learn} tự động phát hiện khi ta dùng mô hình nhị phân cho dữ liệu đa lớp.
    \item Nó tự động chọn OvO hoặc OvR tùy thuộc vào thuật toán.
    \item Ví dụ: Truyền toàn bộ các nhãn $y\_train$ (từ 0-9) vào mô hình Support Vector Machine (\texttt{SVC}), Scikit-Learn sẽ tự động huấn luyện bằng phương pháp OvO.
\end{itemize}
\end{frame}

\begin{frame}{Phân loại đa lớp với thuật toán gốc}
\begin{itemize}
    \item Một số mô hình sinh ra đã hỗ trợ Phân loại đa lớp (ví dụ: \texttt{RandomForestClassifier}, \texttt{SGDClassifier}, \texttt{GaussianNB}).
    \item Trong trường hợp này, Scikit-Learn không cần phải tạo ra các thuật toán nhị phân giả lập (OvO/OvR).
\end{itemize}
\end{frame}

\begin{frame}{Điểm số quyết định cho dự đoán đa lớp}
\begin{itemize}
    \item Khi chạy \texttt{decision\_function()} cho đa lớp, mô hình sẽ trả về 1 mảng điểm tương ứng với mọi lớp có thể.
    \item Điểm cao nhất nằm ở vị trí (index) nào thì mô hình sẽ dự đoán ảnh thuộc về lớp đó.
    \item Danh sách các lớp mục tiêu được lưu trong thuộc tính \texttt{classes\_}.
\end{itemize}
\end{frame}

\begin{frame}{Đánh giá bộ phân loại đa lớp}
\begin{itemize}
    \item Ta vẫn có thể sử dụng hàm \texttt{cross\_val\_score} để đánh giá độ chính xác như phân loại nhị phân.
    \item Tuy nhiên, để cải thiện kết quả, cần kết hợp với các bước \textbf{Chuẩn bị Dữ liệu} ở Chương 2.
    \item Ví dụ: Chuẩn hóa tỉ lệ các đặc trưng (Feature Scaling) bằng \texttt{StandardScaler}.
\end{itemize}
\end{frame}

\section{3. Phân tích lỗi, Đa nhãn \& Đa đầu ra}

\begin{frame}
\vfill\centering\LARGE\textbf{3. Phân tích Lỗi \& Các bài toán nâng cao}\vfill
\end{frame}

\begin{frame}{Trực quan hóa Ma trận nhầm lẫn đa lớp}
\begin{itemize}
    \item Đối với bài toán phân loại đa lớp (10 lớp), ma trận nhầm lẫn sẽ có kích thước $10 \times 10$. Rất khó để nhìn bằng mắt thường.
    \item Cách tốt nhất là trực quan hóa nó bằng biểu đồ màu (heat map).
    \item Ở biểu đồ chuẩn, phần lớn dữ liệu hội tụ trên đường chéo (nghĩa là dự đoán đúng).
    \item Để xem phần LỖI rõ hơn, ta tập trung trọng số vào các ô "dự đoán sai".
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Phân tích Ma trận nhầm lẫn}
\begin{columns}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-9.png}
\caption{Hình 3-9. Ma trận nhầm lẫn được chuẩn hóa}
\end{figure}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-10.png}
\caption{Hình 3-10. Ma trận nhầm lẫn chỉ hiển thị Lỗi}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Phân tích lỗi riêng lẻ và cách khắc phục}
\begin{itemize}
    \item Nhìn vào biểu đồ lỗi, ta có thể thấy số 8 thường bị nhầm lẫn nhiều nhất.
    \item \textbf{Cách khắc phục:}
    \begin{itemize}
        \item Thu thập thêm dữ liệu số 8 để mô hình học rõ hơn.
        \item Viết thuật toán tạo đặc trưng (ví dụ: đếm số vòng kín của số 8).
        \item Tiền xử lý hình ảnh, tăng độ tương phản...
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{So sánh lỗi giữa số 3 và 5}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item SGDClassifier là một mô hình tuyến tính, nó chỉ tổng hợp điểm số dựa trên trọng số pixel.
    \item Hai số 3 và 5 rất giống nhau (chỉ lệch vài pixel nối cung).
    \item Hình bên cạnh cho thấy các lỗi có thể do chữ viết tay quá xấu, dẫn đến mô hình bối rối.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-11.png}
\caption{Hình 3-11. Sự nhầm lẫn giữa chữ số 3 và 5}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Tăng cường dữ liệu (Data Augmentation)}
\begin{itemize}
    \item Mô hình máy học rất nhạy cảm với việc hình ảnh bị lệch (shift) hoặc xoay (rotate).
    \item Giải pháp hiệu quả: **Tăng cường dữ liệu (Data Augmentation)**.
    \item Biến đổi nhẹ các bức ảnh huấn luyện (dịch trái, dịch phải 1 pixel...) rồi thêm vào tập huấn luyện.
    \item Giúp mô hình có khả năng chịu đựng sai số (robust) tốt hơn!
\end{itemize}
\end{frame}

\begin{frame}
\vfill\centering\LARGE\textbf{Phân loại Đa nhãn (Multilabel Classification)}\vfill
\end{frame}

\begin{frame}{Phân loại Đa nhãn (Multilabel) là gì?}
\begin{itemize}
    \item Một mẫu (instance) duy nhất nhưng thuộc về \textbf{nhiều lớp} cùng lúc.
    \item Ví dụ: Nhận diện khuôn mặt. Một bức ảnh có chứa cả Alice và Charlie, nhưng không có Bob.
    \item Đầu ra của mô hình sẽ là một chuỗi cờ Nhị phân (Binary Tags), ví dụ: \texttt{[True, False, True]}.
\end{itemize}
\end{frame}

\begin{frame}{Ví dụ về hệ thống đa nhãn}
\begin{itemize}
    \item Áp dụng vào MNIST: Xây dựng bộ phân loại đánh giá 2 tiêu chí cùng lúc:
    \begin{itemize}
        \item Chữ số đó có lớn không? (7, 8, 9) $\rightarrow$ \texttt{y\_large}
        \item Chữ số đó có phải là số lẻ không? (1, 3, 5, 7, 9) $\rightarrow$ \texttt{y\_odd}
    \end{itemize}
    \item Kết hợp 2 nhãn này lại: \texttt{y\_multilabel}. Mô hình \texttt{KNeighborsClassifier} hỗ trợ xử lý việc này một cách tự nhiên.
\end{itemize}
\end{frame}

\begin{frame}{Đánh giá hệ thống phân loại đa nhãn}
\begin{itemize}
    \item Tính \textbf{F1-score} cho từng nhãn (từng tiêu chí) riêng biệt, sau đó lấy trung bình của tất cả các nhãn.
    \item Tùy chọn \texttt{average="macro"} giả định mọi nhãn quan trọng như nhau.
    \item Tùy chọn \texttt{average="weighted"} sẽ đánh trọng số dựa trên số lượng mẫu thực tế của từng nhãn (nếu tập dữ liệu bị mất cân bằng).
\end{itemize}
\end{frame}

\begin{frame}{Phân loại đa nhãn với ClassifierChain}
\begin{itemize}
    \item Nếu dùng mô hình như \texttt{SVC} (không hỗ trợ multilabel tự nhiên), ta có thể huấn luyện 1 mô hình cho mỗi nhãn.
    \item Tuy nhiên, các nhãn có thể phụ thuộc nhau (Ví dụ: số lớn > 7 thì có tỷ lệ số chẵn cao hơn).
    \item Lớp \texttt{ClassifierChain} giải quyết bằng cách tổ chức mô hình thành một \textbf{chuỗi (chain)}: Mô hình A dự đoán xong sẽ đưa kết quả làm đầu vào phụ cho Mô hình B dự đoán tiếp.
\end{itemize}
\end{frame}

\begin{frame}
\vfill\centering\LARGE\textbf{Phân loại Đa đầu ra (Multioutput Classification)}\vfill
\end{frame}

\begin{frame}{Phân loại Đa đầu ra là gì?}
\begin{itemize}
    \item Khái quát hóa của Phân loại đa nhãn: Đầu ra chứa nhiều nhãn, nhưng mỗi nhãn có thể là \textbf{đa lớp} (nhận nhiều giá trị khác nhau).
    \item Ứng dụng: \textbf{Khử nhiễu hình ảnh (Noise removal)}.
    \item Đầu vào: 1 bức ảnh MNIST bị làm nhiễu pixel.
    \item Đầu ra: 784 pixels, mỗi pixel (là 1 nhãn) nhận một giá trị từ 0 đến 255 (Đa lớp).
\end{itemize}
\end{frame}

\begin{frame}{Bài toán khử nhiễu hình ảnh}
\begin{itemize}
    \item Ta cố tình thêm nhiễu (noise) ngẫu nhiên vào dữ liệu huấn luyện. Mục tiêu mong muốn chính là hình ảnh MNIST ban đầu chưa nhiễu.
    \item Mô hình \texttt{KNeighborsClassifier} sẽ học cách so khớp bức ảnh nhiễu và tìm lại bức ảnh gần gốc nhất.
    \item Xem minh họa quá trình thực hiện ở Slide tiếp theo.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Quá trình khử nhiễu}
\begin{columns}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-12.png}
\caption{Hình 3-12. Ảnh bị nhiễu và Ảnh mục tiêu}
\end{figure}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH03/Hinh_3-13.png}
\caption{Hình 3-13. Ảnh đã được mô hình làm sạch}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Tổng kết Chương 3}
\begin{itemize}
    \item Hiểu rõ các chỉ số đánh giá cho Bài toán Phân loại (Precision, Recall, F1, ROC).
    \item Kiểm soát và điều chỉnh được sự đánh đổi Precision/Recall bằng Threshold.
    \item Nắm được quy trình Phân tích lỗi (Error Analysis) qua Ma trận nhầm lẫn.
    \item Áp dụng các kỹ thuật Phân loại Đa lớp (OvR/OvO), Đa nhãn và Đa đầu ra cho các bài toán phức tạp.
\end{itemize}
\end{frame}

\end{document}
"""

with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(latex_code)
print(f"Generated {tex_path}")
