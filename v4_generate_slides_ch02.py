import os

tex_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap02.tex"

latex_code = r"""\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}
\usecolortheme{default}
\setbeamertemplate{caption}{\raggedright\insertcaption\par}
\usepackage{fontspec}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{xcolor}

\title[Chương 2: Dự án Học máy Từ đầu đến cuối]{DỰ ÁN HỌC MÁY TỪ ĐẦU ĐẾN CUỐI \\ \vspace{0.5cm} \Large Chương 2 - End-to-End Machine Learning Project}
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
    \item Hiểu rõ toàn bộ quy trình phát triển một dự án học máy thực tế từ khi bắt đầu đến khi triển khai.
    \item Thực hành các kỹ thuật chuẩn bị, làm sạch, và biến đổi dữ liệu.
    \item Trực quan hóa dữ liệu để tìm ra các mẫu (patterns) và tương quan.
    \item Xây dựng pipeline dữ liệu chuẩn.
    \item Đánh giá, tinh chỉnh và lựa chọn mô hình học máy tốt nhất.
\end{itemize}
\end{frame}

\section{1. Giới thiệu Dự án \& Khởi tạo Dữ liệu}

\begin{frame}
\vfill\centering\LARGE\textbf{1. Giới thiệu Dự án \& Khởi tạo Dữ liệu}\vfill
\end{frame}

\begin{frame}{Làm việc với dữ liệu thực}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Học máy tốt nhất là thực hành trên dữ liệu thực.
    \item Trong chương này, chúng ta sử dụng tập dữ liệu \textbf{Nhà ở California (California Housing Prices)}.
    \item Mục tiêu: Dự đoán giá nhà trung bình tại một khu vực bất kỳ ở California dựa trên các đặc trưng như dân số, thu nhập trung bình, v.v.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-1.png}
\caption{Hình 2-1. Giá nhà California}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Nhìn vào bức tranh lớn: Xác định vấn đề}
\begin{itemize}
    \item Trước khi phân tích, cần hiểu mục đích kinh doanh là gì? Mô hình này sẽ được sử dụng như thế nào?
    \item Việc hiểu mục đích sẽ giúp quyết định: Thuật toán nào? Thước đo hiệu suất nào? Mức độ đầu tư công sức?
    \item \textbf{Bài toán của chúng ta:} Hệ thống nhận dữ liệu khu vực và dự đoán giá nhà. Đầu ra này sẽ được đưa vào một hệ thống máy học khác để quyết định có nên đầu tư vào khu vực đó hay không.
\end{itemize}
\end{frame}

\begin{frame}{Pipeline hệ thống học máy}
\begin{figure}
\includegraphics[width=0.8\textwidth,height=0.65\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-2.png}
\caption{Hình 2-2. Một pipeline máy học cho đầu tư bất động sản}
\end{figure}
\end{frame}

\begin{frame}{Chọn một Thước đo Hiệu suất}
\begin{itemize}
    \item Vì đây là bài toán hồi quy (Regression), chúng ta sử dụng \textbf{Căn bậc hai Sai số Toàn phương Trung bình (RMSE - Root Mean Square Error)}.
    \item RMSE cho biết sai số dự đoán điển hình. Càng nhỏ càng tốt.
    \item Nếu dữ liệu có nhiều ngoại lai (outliers), ta có thể sử dụng \textbf{Sai số Tuyệt đối Trung bình (MAE - Mean Absolute Error)}.
\end{itemize}
\end{frame}

\begin{frame}{Kiểm tra các Giả định}
\begin{itemize}
    \item Luôn liệt kê và xác minh các giả định đã đặt ra.
    \item Ví dụ: Hệ thống hạ nguồn có thực sự cần dự đoán giá trị cụ thể, hay chỉ cần các hạng mục (rẻ, trung bình, đắt)?
    \item Nếu họ chỉ cần hạng mục, thì chúng ta nên cấu trúc nó thành bài toán Phân loại (Classification) thay vì Hồi quy.
\end{itemize}
\end{frame}

\begin{frame}
\vfill\centering\LARGE\textbf{Lấy Dữ liệu \& Môi trường làm việc}\vfill
\end{frame}

\begin{frame}{Môi trường Google Colab}
\begin{itemize}
    \item Chúng ta sẽ sử dụng \textbf{Google Colab} - một môi trường Jupyter Notebook miễn phí trên đám mây.
    \item Không cần cài đặt phần mềm phức tạp trên máy tính cá nhân.
    \item Cung cấp GPU/TPU miễn phí để huấn luyện mô hình.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Môi trường Google Colab}
\begin{columns}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-3.png}
\caption{Hình 2-3. Giao diện Google Colab}
\end{figure}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-4.png}
\caption{Hình 2-4. Chạy code trên Colab}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Tải và Xem dữ liệu}
\begin{figure}
\includegraphics[width=0.9\textwidth,height=0.65\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-6.png}
\caption{Hình 2-6. Năm hàng đầu tiên trong tập dữ liệu}
\end{figure}
\end{frame}

\begin{frame}{Xem tóm tắt dữ liệu}
\begin{columns}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-7.png}
\caption{Hình 2-7. Tóm tắt của từng thuộc tính số}
\end{figure}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-8.png}
\caption{Hình 2-8. Biểu đồ tần suất (Histogram)}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Tạo tập kiểm thử (Test set): Tại sao cần thiết?}
\begin{itemize}
    \item Chúng ta phải tách một phần dữ liệu (khoảng 20\%) ngay lập tức và \textbf{không bao giờ nhìn vào nó}.
    \item Lý do: Tránh \textit{thiên kiến rình mò dữ liệu (Data Snooping Bias)}.
    \item Nếu chọn thuật toán dựa trên toàn bộ dữ liệu, mô hình có thể Overfit trên tập đó nhưng dự đoán kém trong thực tế.
\end{itemize}
\end{frame}

\begin{frame}{Phân phối Thu nhập}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Thu nhập trung bình là một thuộc tính rất quan trọng để dự đoán giá nhà.
    \item Nếu lấy mẫu ngẫu nhiên thuần túy, ta có thể vô tình chọn phải một tập kiểm thử thiếu tính đại diện.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-9.png}
\caption{Hình 2-9. Các loại thu nhập}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Lấy mẫu phân tầng (Stratified Sampling) so với Ngẫu nhiên}
\begin{figure}
\includegraphics[width=0.7\textwidth,height=0.65\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-10.png}
\caption{Hình 2-10. Tỷ lệ phân loại thu nhập trong tổng thể so với mẫu ngẫu nhiên và phân tầng}
\end{figure}
\end{frame}

\section{2. Khám phá \& Chuẩn bị Dữ liệu}

\begin{frame}
\vfill\centering\LARGE\textbf{2. Khám phá \& Chuẩn bị Dữ liệu}\vfill
\end{frame}

\begin{frame}{Trực quan hóa dữ liệu địa lý}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Dữ liệu có thuộc tính kinh độ (longitude) và vĩ độ (latitude).
    \item Việc vẽ biểu đồ phân tán (scatter plot) sẽ giúp ta nhìn thấy mô hình phân bố của các khu dân cư.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-11.png}
\caption{Hình 2-11. Trực quan hóa dữ liệu địa lý}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Phân tích mật độ địa lý}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Bằng cách chỉnh thông số \texttt{alpha}, ta có thể nhìn thấy rõ ràng các khu vực tập trung đông dân cư.
    \item Nổi bật lên khu vực Vùng Vịnh (Bay Area), Los Angeles và San Diego.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-12.png}
\caption{Hình 2-12. Trực quan hóa tốt hơn để thấy khu vực mật độ cao}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Giá nhà theo dân số \& Vị trí}
\begin{figure}
\includegraphics[width=0.7\textwidth,height=0.65\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-13.png}
\caption{Hình 2-13. Giá nhà khu vực California (Màu: Giá nhà, Vòng tròn: Dân số)}
\end{figure}
\end{frame}

\begin{frame}{Tìm kiếm các tương quan (Correlation)}
\begin{itemize}
    \item Hệ số tương quan Pearson ($r$) chạy từ -1 đến +1.
    \item Gần +1: Tương quan dương mạnh (ví dụ: thu nhập tăng, giá nhà tăng).
    \item Gần -1: Tương quan âm mạnh.
    \item Gần 0: Không có tương quan tuyến tính.
\end{itemize}
\end{frame}

\begin{frame}{Ma trận phân tán (Scatter Matrix)}
\begin{figure}
\includegraphics[width=0.7\textwidth,height=0.65\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-14.png}
\caption{Hình 2-14. Ma trận phân tán hiển thị tương quan giữa các đặc trưng}
\end{figure}
\end{frame}

\begin{frame}{Thu nhập trung bình vs Giá nhà trung bình}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Thu nhập trung bình là thuộc tính có tương quan mạnh nhất với giá nhà.
    \item Trên biểu đồ xuất hiện một số "đường ngang" (trần giá) ở mức \$500k, \$450k, và \$350k.
    \item Có thể cần phải loại bỏ các điểm dữ liệu này để thuật toán không học những mô hình sai lệch này.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-15.png}
\caption{Hình 2-15. Thu nhập so với giá trị nhà}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Phân phối và độ lệch chuẩn của dữ liệu}
\begin{figure}
\includegraphics[width=0.75\textwidth,height=0.65\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-16.jpg}
\caption{Hình 2-16. Độ lệch chuẩn của các tập dữ liệu khác nhau}
\end{figure}
\end{frame}

\begin{frame}{Thử nghiệm với các kết hợp đặc trưng}
\begin{itemize}
    \item Đôi khi, kết hợp các đặc trưng hiện có sẽ tạo ra đặc trưng mới ý nghĩa hơn.
    \item \textbf{Ví dụ:} 
    \begin{itemize}
        \item Số lượng phòng trong một quận không hữu ích nếu không biết số hộ gia đình. Thay vào đó, tạo đặc trưng \textit{Số phòng trên mỗi hộ}.
        \item Số lượng phòng ngủ trên tổng số phòng.
        \item Dân số trên mỗi hộ.
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}
\vfill\centering\LARGE\textbf{Chuẩn bị dữ liệu cho học máy}\vfill
\end{frame}

\begin{frame}{Làm sạch dữ liệu (Data Cleaning)}
\begin{itemize}
    \item Các thuật toán học máy không thể hoạt động với các giá trị bị thiếu (Missing Values).
    \item Có 3 lựa chọn để xử lý:
    \begin{enumerate}
        \item Loại bỏ các mẫu tương ứng (xóa hàng).
        \item Loại bỏ toàn bộ thuộc tính đó (xóa cột).
        \item Điền vào các giá trị bị thiếu bằng một giá trị nào đó (0, trung bình, trung vị...).
    \end{enumerate}
    \item Sử dụng \texttt{SimpleImputer} của Scikit-Learn.
\end{itemize}
\end{frame}

\begin{frame}{Xử lý các thuộc tính văn bản và phân loại}
\begin{itemize}
    \item Hầu hết các thuật toán ML thích làm việc với con số. Cần chuyển đổi văn bản sang số.
    \item Đặc trưng "ocean\_proximity" là văn bản.
    \item Phương pháp \textbf{One-Hot Encoding}:
    \begin{itemize}
        \item Tạo một cột nhị phân cho mỗi danh mục.
        \item Tránh việc thuật toán hiểu sai rằng danh mục "1" thì gần danh mục "2" hơn danh mục "4".
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{Co giãn Đặc trưng (Feature Scaling)}
\begin{itemize}
    \item Các thuật toán hoạt động không tốt khi các đặc trưng có thang đo rất khác nhau (ví dụ: dân số từ vài nghìn, nhưng thu nhập chỉ từ 0 - 15).
    \item \textbf{Min-Max Scaling (Chuẩn hóa Normalization):} Giá trị được chuyển về khoảng từ 0 đến 1.
    \item \textbf{Standardization (Chuẩn hóa Standardization):} Trừ đi giá trị trung bình và chia cho độ lệch chuẩn. Không bị ảnh hưởng nhiều bởi ngoại lai.
\end{itemize}
\end{frame}

\begin{frame}{Biến đổi Đặc trưng để gần với phân phối Gaussian}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Nhiều biểu đồ tần suất bị "nặng đuôi" (tail-heavy).
    \item Sử dụng các phép tính logarit (Log) hoặc logarit căn bản để biến đổi dữ liệu về dạng giống cái chuông (bell-shaped Gaussian).
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-17.png}
\caption{Hình 2-17. Biến đổi đặc trưng nặng đuôi}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Biến đổi Đặc trưng bằng RBF}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Đối với phân phối có nhiều đỉnh (multimodal).
    \item Dùng \textbf{Radial Basis Function (RBF)} - Hàm cơ sở xuyên tâm. Đo khoảng cách (sự tương đồng) giữa một giá trị và một "đỉnh" (landmark) cố định.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-18.png}
\caption{Hình 2-18. Đặc trưng RBF (Đo khoảng cách tới 35)}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Đặc trưng từ phân cụm bằng K-Means}
\begin{figure}
\includegraphics[width=0.7\textwidth,height=0.65\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-19.png}
\caption{Hình 2-19. Đặc trưng RBF với tâm cụm từ K-Means}
\end{figure}
\end{frame}

\begin{frame}{Bộ biến đổi tùy chỉnh (Custom Transformers)}
\begin{itemize}
    \item Mặc dù Scikit-Learn cung cấp nhiều bộ biến đổi, ta vẫn cần tự viết mã để thực hiện logic riêng (ví dụ: tính số phòng trên mỗi hộ).
    \item Tạo các lớp (Class) tùy chỉnh tuân theo API của Scikit-Learn (kế thừa từ \texttt{BaseEstimator} và \texttt{TransformerMixin}).
\end{itemize}
\end{frame}

\begin{frame}{Các pipeline biến đổi (Transformation Pipelines)}
\begin{itemize}
    \item Quá trình biến đổi dữ liệu yêu cầu thực hiện nhiều bước theo đúng thứ tự.
    \item \texttt{Pipeline} trong Scikit-Learn giúp liên kết các bước xử lý dữ liệu lại với nhau thành một luồng duy nhất.
    \item \texttt{ColumnTransformer}: Áp dụng các luồng xử lý khác nhau cho các loại cột khác nhau (VD: Cột số chạy qua StandardScaler, cột phân loại chạy qua OneHotEncoder).
\end{itemize}
\end{frame}

\section{3. Chọn, Huấn luyện, Tinh chỉnh \& Triển khai}

\begin{frame}
\vfill\centering\LARGE\textbf{3. Chọn, Huấn luyện, Tinh chỉnh \& Triển khai}\vfill
\end{frame}

\begin{frame}{Huấn luyện trên Tập huấn luyện}
\begin{itemize}
    \item Bắt đầu với một mô hình đơn giản như \textbf{Hồi quy tuyến tính (Linear Regression)}.
    \item Nếu RMSE quá cao (ví dụ sai số 68,000 USD), nghĩa là mô hình bị \textbf{Underfitting} (chưa khớp).
    \item Thử nghiệm với một mô hình phức tạp hơn: \textbf{Cây quyết định (Decision Tree Regressor)}. Mô hình mạnh mẽ, có thể tìm ra các mối quan hệ phi tuyến tính phức tạp.
\end{itemize}
\end{frame}

\begin{frame}{Đánh giá tốt hơn bằng kiểm định chéo (Cross-Validation)}
\begin{itemize}
    \item Sử dụng \textbf{K-fold Cross-Validation}. 
    \item Chia ngẫu nhiên tập huấn luyện thành K tập con (fold) khác biệt. 
    \item Huấn luyện và đánh giá mô hình K lần, mỗi lần chọn một fold khác nhau để đánh giá và K-1 fold còn lại để huấn luyện.
    \item Giúp biết chính xác mô hình tổng quát hóa tốt như thế nào mà không cần chạm vào tập kiểm thử thực (Test set).
\end{itemize}
\end{frame}

\begin{frame}{Thử nghiệm với Rừng ngẫu nhiên (Random Forest)}
\begin{itemize}
    \item Xây dựng nhiều Cây quyết định trên các tập hợp con ngẫu nhiên của các đặc trưng, sau đó lấy trung bình dự đoán của chúng.
    \item Đây là một mô hình \textbf{Học tập hợp (Ensemble Learning)}.
    \item Thường cho kết quả vượt trội và là một trong những mô hình máy học mạnh nhất.
\end{itemize}
\end{frame}

\begin{frame}
\vfill\centering\LARGE\textbf{Tinh chỉnh mô hình}\vfill
\end{frame}

\begin{frame}{Tìm kiếm theo lưới (Grid Search)}
\begin{itemize}
    \item Sử dụng \texttt{GridSearchCV} của Scikit-Learn để tự động thử nghiệm tất cả các tổ hợp có thể có của các siêu tham số.
    \item Rất hữu ích khi khám phá một vài kết hợp, nhưng cực kỳ tốn thời gian tính toán nếu không gian tìm kiếm lớn.
\end{itemize}
\end{frame}

\begin{frame}{Tìm kiếm ngẫu nhiên (Random Search)}
\begin{itemize}
    \item Dùng \texttt{RandomizedSearchCV}: thay vì thử nghiệm tất cả các kết hợp (như Grid Search), nó sẽ đánh giá một số lượng ngẫu nhiên nhất định.
    \item Cho phép điều chỉnh siêu tham số với hiệu suất tính toán tốt hơn nếu không gian tìm kiếm quá lớn (nhiều siêu tham số).
\end{itemize}
\end{frame}

\begin{frame}{Các phương pháp tập hợp (Ensemble Methods)}
\begin{itemize}
    \item Một cách khác để tinh chỉnh hệ thống là cố gắng kết hợp các mô hình hoạt động tốt nhất.
    \item Một nhóm (ensemble) thường sẽ hoạt động tốt hơn so với mô hình cá nhân tốt nhất. (Giống như quyết định của một hội đồng thường tốt hơn một cá nhân).
\end{itemize}
\end{frame}

\begin{frame}{Phân tích các mô hình tốt nhất và lỗi của chúng}
\begin{itemize}
    \item Có thể đạt được những cái nhìn sâu sắc bằng cách kiểm tra các mô hình tốt nhất.
    \item Mô hình RandomForest có thể cho biết mức độ quan trọng tương đối của từng đặc trưng.
    \item Dựa vào thông tin này, bạn có thể cân nhắc loại bỏ một số đặc trưng ít hữu ích nhất.
\end{itemize}
\end{frame}

\begin{frame}{Đánh giá hệ thống trên tập kiểm thử}
\begin{itemize}
    \item Khi đã hài lòng với mô hình cuối cùng, đây là lúc kiểm tra hiệu suất trên \textbf{Test Set}.
    \item Chạy tập kiểm thử qua \texttt{full\_pipeline} để chuyển đổi dữ liệu. \textbf{Lưu ý: Chỉ gọi \texttt{transform()}, không gọi \texttt{fit()}}.
    \item Hiệu suất thường sẽ hơi tệ hơn so với trên tập xác thực (do mô hình đã được tinh chỉnh cho tập xác thực). Đừng cố tình điều chỉnh siêu tham số nữa!
\end{itemize}
\end{frame}

\begin{frame}
\vfill\centering\LARGE\textbf{Triển khai, Giám sát và Bảo trì hệ thống}\vfill
\end{frame}

\begin{frame}{Triển khai mô hình (Model Deployment)}
\begin{itemize}
    \item Lưu trữ mô hình đã huấn luyện bằng cách sử dụng các thư viện như \texttt{joblib}.
    \item Tải mô hình lên môi trường sản xuất (Production).
    \item Có thể đóng gói bằng Container (Docker) và triển khai lên Cloud.
\end{itemize}
\end{frame}

\begin{frame}{Kiến trúc Triển khai (Ví dụ)}
\begin{figure}
\includegraphics[width=0.7\textwidth,height=0.65\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH02/Hinh_2-20.png}
\caption{Hình 2-20. Triển khai mô hình học máy thành dịch vụ web (Web API)}
\end{figure}
\end{frame}

\begin{frame}{Giám sát hệ thống (Monitoring)}
\begin{itemize}
    \item Theo thời gian, dữ liệu thực tế thay đổi dẫn đến hiện tượng suy giảm hiệu suất (Model Rot / Data Drift).
    \item Cần theo dõi sự chênh lệch giữa dự đoán của hệ thống và dữ liệu thực tế mới nhất.
    \item Thiết lập cảnh báo để đội ngũ xử lý khi độ chính xác giảm xuống dưới một ngưỡng nhất định.
\end{itemize}
\end{frame}

\begin{frame}{Tự động hóa quá trình bảo trì và cập nhật}
\begin{itemize}
    \item Thu thập dữ liệu mới thường xuyên và tự động gán nhãn cho nó (nếu có thể).
    \item Tự động đánh giá các mô hình mới. Nếu hiệu suất tốt hơn mô hình cũ, tự động đẩy (deploy) phiên bản mới lên máy chủ.
    \item Giữ lại các bản sao lưu để có thể quay lại (rollback) phiên bản trước nhanh chóng nếu mô hình mới gây ra lỗi nghiêm trọng.
\end{itemize}
\end{frame}

\begin{frame}{Tổng kết Chương 2}
\begin{itemize}
    \item Quá trình xây dựng ứng dụng học máy là một chu trình \textbf{khép kín (End-to-End)}: Từ dữ liệu thô $\rightarrow$ Phân tích \& Chuẩn bị $\rightarrow$ Huấn luyện $\rightarrow$ Triển khai $\rightarrow$ Bảo trì.
    \item Hầu hết công sức của chuyên gia dữ liệu là ở khâu \textbf{Chuẩn bị dữ liệu} và hiểu bài toán, thay vì chọn các thuật toán tinh vi.
    \item Một pipeline dữ liệu vững chắc là chìa khóa thành công.
\end{itemize}
\end{frame}

\end{document}
"""

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_code)
print(f"Generated {tex_path}")
