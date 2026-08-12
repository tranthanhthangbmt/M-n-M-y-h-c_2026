import os

tex_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap04.tex"

latex_code = r"""\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}
\usecolortheme{default}
\setbeamertemplate{caption}{\raggedright\insertcaption\par}
\usepackage{fontspec}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{amsmath}

\title[Chương 4: Huấn luyện Mô hình]{DỰ ÁN HỌC MÁY TỪ ĐẦU ĐẾN CUỐI \\ \vspace{0.5cm} \Large Chương 4 - Huấn luyện Mô hình (Training Models)}
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
    \item Hiểu rõ cách thức hoạt động bên trong của các mô hình học máy thay vì coi chúng là "hộp đen".
    \item Tìm hiểu và áp dụng các phương pháp giải Hồi quy tuyến tính (Phương trình chuẩn tắc, SVD, Gradient Descent).
    \item Phân biệt được các loại Gradient Descent (Batch, Stochastic, Mini-batch) và ảnh hưởng của Tốc độ học (Learning Rate).
    \item Hiểu cách xử lý dữ liệu phi tuyến tính bằng Hồi quy đa thức.
    \item Phát hiện Quá khớp (Overfitting) thông qua Đường cong học tập và các kỹ thuật Chính quy hóa (Ridge, Lasso, Elastic Net, Early Stopping).
    \item Tìm hiểu Hồi quy Logistic cho phân loại nhị phân và Hồi quy Softmax cho đa lớp.
\end{itemize}
\end{frame}

\section{1. Hồi quy tuyến tính \& Phương trình chuẩn tắc}

\begin{frame}
\vfill\centering\LARGE\textbf{1. Hồi quy tuyến tính \& Phương trình chuẩn tắc}\vfill
\end{frame}

\begin{frame}{Hồi quy tuyến tính (Linear Regression)}
\begin{itemize}
    \item Hồi quy tuyến tính là một trong những mô hình đơn giản nhất. Mô hình sẽ đưa ra dự đoán bằng cách tính tổng trọng số của các đặc trưng đầu vào, cộng với một hằng số gọi là số hạng độ lệch (bias term) hoặc hệ số chặn (intercept term).
    \item \textbf{Mô hình dự đoán hồi quy tuyến tính (Dạng toán học):}
    \begin{equation}
    \hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
    \end{equation}
    \item \textbf{Dạng Vector (Rút gọn):}
    \begin{equation}
    \hat{y} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \boldsymbol{\theta} \cdot \mathbf{x}
    \end{equation}
    \item Trong đó: $\boldsymbol{\theta}$ là vector tham số mô hình, $\mathbf{x}$ là vector đặc trưng của trường hợp.
\end{itemize}
\end{frame}

\begin{frame}{Hàm chi phí MSE (Mean Squared Error)}
\begin{itemize}
    \item Để huấn luyện mô hình Hồi quy tuyến tính, chúng ta cần tìm giá trị cho vector tham số $\boldsymbol{\theta}$ nhằm tối thiểu hóa sai số giữa giá trị dự đoán và giá trị thực tế.
    \item Thước đo phổ biến nhất là Hàm chi phí Lỗi Bình phương Trung bình (MSE - Mean Squared Error).
    \item \textbf{Công thức MSE cho mô hình hồi quy tuyến tính:}
    \begin{equation}
    MSE(\mathbf{X}, h_{\boldsymbol{\theta}}) = \frac{1}{m} \sum_{i=1}^{m} \left( \boldsymbol{\theta}^\top \mathbf{x}^{(i)} - y^{(i)} \right)^2
    \end{equation}
    \item Mục tiêu của quá trình huấn luyện là tìm $\boldsymbol{\theta}$ để MSE đạt giá trị nhỏ nhất.
\end{itemize}
\end{frame}

\begin{frame}{Phương trình chuẩn tắc (Normal Equation)}
\begin{itemize}
    \item Để tìm giá trị $\boldsymbol{\theta}$ làm tối thiểu hóa hàm chi phí, có một nghiệm dạng đóng (closed-form solution) – một phương trình toán học cho ra kết quả trực tiếp. Nó được gọi là Phương trình chuẩn tắc.
    \item \textbf{Công thức Phương trình chuẩn tắc:}
    \begin{equation}
    \hat{\boldsymbol{\theta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
    \end{equation}
    \item $\hat{\boldsymbol{\theta}}$ là giá trị $\boldsymbol{\theta}$ làm giảm thiểu hàm chi phí.
    \item $\mathbf{y}$ là vector của các giá trị mục tiêu (từ $y^{(1)}$ đến $y^{(m)}$).
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Tập dữ liệu tuyến tính ngẫu nhiên}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Để kiểm tra Phương trình chuẩn tắc, chúng ta có thể tạo một số dữ liệu nhìn có vẻ tuyến tính một cách ngẫu nhiên.
    \item Công thức sử dụng để tạo dữ liệu: $y = 4 + 3x + \text{nhiễu Gauss}$.
    \item Biểu đồ bên cạnh cho thấy các điểm dữ liệu được tạo ra. Dữ liệu này sẽ được dùng để khớp mô hình.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-1.png}
\caption{Hình 4-1. Dữ liệu tuyến tính được tạo ngẫu nhiên}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Minh họa: Dự đoán của Hồi quy tuyến tính}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Sau khi tính toán được $\hat{\boldsymbol{\theta}}$ từ Phương trình chuẩn tắc, chúng ta có thể vẽ đường thẳng dự đoán.
    \item Hình bên cạnh cho thấy các dự đoán của mô hình (đường thẳng màu đỏ) khớp với tập dữ liệu ngẫu nhiên của chúng ta như thế nào.
    \item Đường thẳng này đi qua trung tâm của các điểm dữ liệu, cho thấy mô hình đã học được xu hướng chung của dữ liệu.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-2.png}
\caption{Hình 4-2. Dự đoán của mô hình hồi quy tuyến tính}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Thực hiện với Scikit-Learn và Singular Value Decomposition}
\begin{itemize}
    \item Trong thực tế, Scikit-Learn sử dụng \textbf{Singular Value Decomposition (SVD)} thay vì tính trực tiếp Phương trình chuẩn tắc.
    \item \textbf{Lý do:} Ma trận $\mathbf{X}^\top \mathbf{X}$ có thể không khả nghịch (chẳng hạn khi số lượng đặc trưng lớn hơn số lượng trường hợp huấn luyện). Phương pháp SVD luôn xử lý được các trường hợp này bằng ma trận giả nghịch đảo (pseudoinverse).
    \item Đoạn mã Scikit-Learn rất đơn giản:
    \begin{quote}
    \texttt{lin\_reg = LinearRegression()} \\
    \texttt{lin\_reg.fit(X, y)}
    \end{quote}
\end{itemize}
\end{frame}

\begin{frame}{Độ phức tạp tính toán}
\begin{itemize}
    \item Cả phương trình chuẩn tắc và cách tiếp cận SVD đều trở nên \textbf{rất chậm} khi số lượng đặc trưng tăng lên đáng kể.
    \item Độ phức tạp tính toán của việc đảo ngược ma trận (như Phương trình chuẩn tắc) là khoảng $\mathcal{O}(n^{2.4})$ đến $\mathcal{O}(n^3)$.
    \item Phương pháp SVD được Scikit-Learn sử dụng là khoảng $\mathcal{O}(n^2)$. Tức là, nếu tăng gấp đôi số lượng đặc trưng, thời gian tính toán tăng gấp 4 lần.
    \item \textbf{Ưu điểm:} Cả hai phương pháp này lại là tuyến tính đối với số lượng trường hợp huấn luyện (độ phức tạp $\mathcal{O}(m)$). Chúng xử lý rất tốt tập huấn luyện lớn miễn là nó vừa với bộ nhớ.
\end{itemize}
\end{frame}

\section{2. Hạ Gradient (Gradient Descent)}

\begin{frame}
\vfill\centering\LARGE\textbf{2. Hạ Gradient (Gradient Descent)}\vfill
\end{frame}

\begin{frame}{Gradient Descent là gì?}
\begin{itemize}
    \item \textbf{Gradient Descent (Hạ Gradient)} là một thuật toán tối ưu hóa chung có khả năng tìm ra giải pháp tối ưu cho nhiều loại vấn đề.
    \item Ý tưởng chung: Tinh chỉnh các tham số theo vòng lặp để giảm thiểu hàm chi phí.
    \item Ví dụ trực quan: Tưởng tượng bạn bị lạc trên núi trong một màn sương mù dày đặc, và bạn chỉ có thể cảm nhận độ dốc của mặt đất dưới chân mình. Chiến lược tốt nhất để xuống đáy thung lũng là bước xuống dốc theo hướng dốc nhất.
    \item Nó đo lường gradient (đạo hàm) cục bộ của hàm lỗi so với vector tham số $\boldsymbol{\theta}$, và đi theo hướng giảm của gradient cho đến khi gradient bằng 0.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Nguyên lý hoạt động của Gradient Descent}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Kích thước bước nhảy được xác định bởi siêu tham số \textbf{tốc độ học (learning rate)}.
    \item Bắt đầu với một giá trị $\boldsymbol{\theta}$ ngẫu nhiên, thuật toán dần dần tiến gần đến điểm cực tiểu.
    \item Hình bên minh họa sự mô phỏng quá trình này: các bước đi dần hội tụ tại cực tiểu toàn cục.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-3.png}
\caption{Hình 4-3. Gradient Descent}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Phân tích ảnh hưởng của Tốc độ học (Learning Rate)}
\begin{itemize}
    \item Tốc độ học (Learning Rate - $\eta$) là một siêu tham số quan trọng nhất của thuật toán Gradient Descent.
    \item \textbf{Tốc độ học quá nhỏ:}
    \begin{itemize}
        \item Thuật toán sẽ phải trải qua rất nhiều vòng lặp để hội tụ.
        \item Mất rất nhiều thời gian.
    \end{itemize}
    \item \textbf{Tốc độ học quá cao:}
    \begin{itemize}
        \item Bạn có thể "nhảy cóc" qua thung lũng và kết thúc ở bên kia, thậm chí cao hơn trước.
        \item Làm cho thuật toán phân kỳ (diverge), giá trị hàm chi phí ngày càng tăng và không bao giờ tìm được điểm tối ưu.
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Tốc độ học quá nhỏ và quá cao}
\begin{columns}
\column{0.5\textwidth}
\begin{figure}
\centering
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-4.png}
\caption{Hình 4-4. Tốc độ học quá nhỏ}
\end{figure}
\column{0.5\textwidth}
\begin{figure}
\centering
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-5.png}
\caption{Hình 4-5. Tốc độ học quá lớn}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Các cạm bẫy của Gradient Descent}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Không phải tất cả các hàm chi phí đều trông giống như những cái bát hình parabol hoàn hảo.
    \item \textbf{Cực tiểu cục bộ (Local Minimum):} Nếu bạn bắt đầu từ một điểm khởi tạo ngẫu nhiên, thuật toán có thể hội tụ về điểm thấp cục bộ chứ không phải cực tiểu toàn cục (Global Minimum).
    \item \textbf{Cao nguyên (Plateau):} Hàm chi phí đi ngang, khiến thuật toán mất rất nhiều thời gian để đi qua và có thể bị dừng lại sớm.
    \item Rất may, hàm chi phí MSE của hồi quy tuyến tính là \textbf{hàm lồi (convex function)}, đảm bảo không có cực tiểu cục bộ.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-6.png}
\caption{Hình 4-6. Cạm bẫy cực tiểu cục bộ}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Tầm quan trọng của việc điều chỉnh đặc trưng (Scaling)}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Khi sử dụng Gradient Descent, bạn phải đảm bảo rằng tất cả các đặc trưng có thang đo (scale) tương tự nhau (VD: dùng \texttt{StandardScaler}).
    \item \textbf{Bên trái:} Các đặc trưng được chia tỷ lệ bằng nhau, hàm chi phí có dạng cái bát tròn, đường đi đến cực tiểu nhanh gọn và thẳng.
    \item \textbf{Bên phải:} Đặc trưng 1 nhỏ hơn Đặc trưng 2, hàm chi phí có dạng bát kéo dài. Đường đi của thuật toán ngoằn ngoèo, tốn nhiều thời gian hội tụ.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-7.png}
\caption{Hình 4-7. Gradient Descent với và không có Scaling}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Hạ Gradient theo lô (Batch Gradient Descent)}
\begin{itemize}
    \item Batch Gradient Descent (Hạ Gradient theo lô) tính toán đạo hàm riêng của hàm chi phí đối với mỗi tham số $\theta_j$ bằng cách sử dụng \textbf{toàn bộ tập dữ liệu huấn luyện} tại mỗi bước lặp.
    \item \textbf{Vector Gradient của hàm chi phí:}
    \begin{equation}
    \nabla_{\boldsymbol{\theta}} MSE(\boldsymbol{\theta}) = \frac{2}{m} \mathbf{X}^\top (\mathbf{X} \boldsymbol{\theta} - \mathbf{y})
    \end{equation}
    \item \textbf{Bước nhảy Gradient Descent:}
    \begin{equation}
    \boldsymbol{\theta}^{(\text{bước tiếp theo})} = \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} MSE(\boldsymbol{\theta})
    \end{equation}
    \item Mặc dù chậm trên tập dữ liệu rất lớn (do phải dùng toàn bộ dữ liệu), nhưng nó lại chia tỷ lệ tốt với số lượng đặc trưng lớn (tốt hơn Phương trình chuẩn tắc).
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Batch GD với các tốc độ học khác nhau}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Hình bên cho thấy 10 bước đầu tiên của Gradient Descent theo lô.
    \item \textbf{Trái ($\eta = 0.02$):} Tốc độ học quá thấp, sẽ đạt cực tiểu nhưng chậm.
    \item \textbf{Giữa ($\eta = 0.1$):} Tốc độ học rất tốt, thuật toán hội tụ nhanh chóng chỉ trong vài bước.
    \item \textbf{Phải ($\eta = 0.5$):} Tốc độ học quá cao, thuật toán nhảy loạn xạ và đang phân kỳ (diverging).
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-8.png}
\caption{Hình 4-8. Batch GD với tốc độ học khác nhau}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Hạ Gradient ngẫu nhiên (Stochastic Gradient Descent - SGD)}
\begin{itemize}
    \item Vấn đề chính của Batch GD là sử dụng toàn bộ tập dữ liệu huấn luyện để tính gradient tại mỗi bước, làm nó rất chậm với tập dữ liệu lớn.
    \item **Hạ Gradient ngẫu nhiên (SGD)** ngược lại, chọn ra một trường hợp ngẫu nhiên trong tập huấn luyện tại mỗi bước và chỉ tính gradient trên trường hợp duy nhất đó.
    \item Điều này làm cho thuật toán cực kỳ nhanh, và có khả năng thực hiện huấn luyện các mô hình rất lớn vì mỗi lần lặp chỉ cần 1 trường hợp trong bộ nhớ (hỗ trợ Out-of-core learning).
\end{itemize}
\end{frame}

\begin{frame}{Đặc điểm đường đi của Stochastic Gradient Descent}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Do tính chất ngẫu nhiên, SGD thất thường hơn nhiều so với Batch GD.
    \item Thay vì giảm nhẹ nhàng cho đến khi đạt mức tối thiểu, hàm chi phí sẽ dao động lên xuống, giảm dần về trung bình.
    \item Ngay cả khi kết thúc gần cực tiểu, các giá trị tham số sẽ tiếp tục dao động. Nghĩa là giá trị cuối cùng là tốt, nhưng chưa phải tối ưu tuyệt đối.
    \item Ưu điểm: Sự ngẫu nhiên giúp nó nhảy ra khỏi các cực tiểu cục bộ tốt hơn Batch GD.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-9.png}
\caption{Hình 4-9. Đặc điểm bất thường của SGD}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Minh họa: 20 bước đầu tiên của SGD}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Hình bên cho thấy 20 bước đầu tiên của SGD.
    \item Bạn có thể thấy các bước không đều đặn như Batch GD. Để SGD hội tụ sát hơn, ta phải áp dụng kỹ thuật \textbf{lịch học (learning schedule)}: Tốc độ học lớn lúc đầu (tránh kẹt cực tiểu cục bộ), và giảm dần tốc độ học khi tiến về gần đích để ổn định tại đáy thung lũng.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-10.png}
\caption{Hình 4-10. 20 bước đầu tiên của SGD}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Hạ Gradient theo Mini-Batch}
\begin{itemize}
    \item Tại mỗi bước, thay vì tính toán gradient dựa trên toàn bộ dữ liệu (như Batch GD) hoặc chỉ một trường hợp (như SGD), thuật toán tính gradient trên các tập con nhỏ ngẫu nhiên gọi là **mini-batch**.
    \item Ưu điểm lớn nhất là bạn có thể tăng hiệu suất từ việc tối ưu hóa phần cứng của các phép toán ma trận, đặc biệt khi sử dụng GPUs.
    \item Sự tiến bộ của thuật toán ít thất thường hơn SGD (với mini-batch khá lớn). Nó sẽ tiến gần cực tiểu hơn SGD một chút, nhưng có thể khó thoát cực tiểu cục bộ hơn.
\end{itemize}
\end{frame}

\begin{frame}{So sánh đường đi của 3 thuật toán Gradient Descent}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item \textbf{Batch GD (Đỏ):} Điểm đến thẳng cực tiểu và dừng lại, nhưng tốn nhiều thời gian mỗi bước.
    \item \textbf{SGD (Trắng):} Dao động mạnh xung quanh đích.
    \item \textbf{Mini-Batch GD (Xanh):} Ít dao động hơn SGD và tiến gần đích hơn, nhưng vẫn không dừng lại chính xác tại một điểm trừ khi có lịch học (learning schedule) tốt.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-11.png}
\caption{Hình 4-11. Đường đi của 3 thuật toán GD}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Bảng so sánh tổng hợp các thuật toán hồi quy tuyến tính}
\begin{table}[]
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lllllll@{}}
\toprule
\textbf{Thuật toán} & \textbf{Lớn m (trường hợp)} & \textbf{Ngoài lõi} & \textbf{Lớn n (đặc trưng)} & \textbf{Hyperparams} & \textbf{Scaling} & \textbf{Scikit-Learn} \\ \midrule
Phương trình chuẩn tắc & Nhanh & Không & Chậm & 0 & Không & N/A \\
SVD & Nhanh & Không & Chậm & 0 & Không & LinearRegression \\
Batch GD & Chậm & Không & Nhanh & 2 & Có & N/A \\
Stochastic GD & Nhanh & Có & Nhanh & >=2 & Có & SGDRegressor \\
Mini-batch GD & Nhanh & Có & Nhanh & >=2 & Có & N/A \\ \bottomrule
\end{tabular}%
}
\end{table}
\textit{Lưu ý: Sau khi huấn luyện, tất cả các thuật toán này đều tạo ra mô hình rất giống nhau và đưa ra dự đoán theo cùng một cách.}
\end{frame}

\section{3. Hồi quy đa thức, Đường cong học tập \& Chính quy hóa}

\begin{frame}
\vfill\centering\LARGE\textbf{3. Hồi quy đa thức, Đường cong học tập \& Chính quy hóa}\vfill
\end{frame}

\begin{frame}{Hồi quy đa thức (Polynomial Regression)}
\begin{itemize}
    \item Điều gì sẽ xảy ra nếu dữ liệu của bạn phức tạp hơn một đường thẳng?
    \item Thật đáng ngạc nhiên, bạn có thể sử dụng một mô hình tuyến tính để khớp với dữ liệu phi tuyến tính.
    \item Cách đơn giản: Thêm các lũy thừa của mỗi đặc trưng làm đặc trưng mới, sau đó huấn luyện mô hình tuyến tính trên tập đặc trưng mở rộng này.
    \item Kỹ thuật này được gọi là \textbf{Hồi quy đa thức}.
    \item \texttt{PolynomialFeatures} trong Scikit-Learn có thể dễ dàng tạo ra các biến bậc cao và kết hợp biến (tương tác giữa các đặc trưng).
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Khớp Hồi quy đa thức trên dữ liệu phi tuyến}
\begin{itemize}
    \item Giả sử chúng ta có một tập dữ liệu phân bố theo phương trình bậc hai ($y = ax^2 + bx + c$ + nhiễu). Một đường thẳng tuyến tính bình thường không bao giờ khớp được (Underfitting).
    \item Bằng cách dùng \texttt{PolynomialFeatures(degree=2)}, chúng ta thêm $x^2$ vào bộ đặc trưng. Sau đó dùng \texttt{LinearRegression} khớp dữ liệu mới.
    \item Kết quả thu được là một đường parabol bám rất sát dữ liệu (như Hình 4-13).
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Dữ liệu phi tuyến tính và Dự đoán Hồi quy Đa thức}
\begin{columns}
\column{0.5\textwidth}
\begin{figure}
\centering
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-12.png}
\caption{Hình 4-12. Tập dữ liệu phi tuyến tính ngẫu nhiên}
\end{figure}
\column{0.5\textwidth}
\begin{figure}
\centering
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-13.png}
\caption{Hình 4-13. Dự đoán mô hình hồi quy đa thức}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Hiện tượng Quá khớp (Overfitting) với đa thức bậc cao}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Nếu thực hiện hồi quy đa thức bậc quá cao (Ví dụ: bậc 300), mô hình sẽ khớp dữ liệu huấn luyện cực kỳ tốt.
    \item Hình bên cho thấy đường uốn lượn dữ dội cố gắng đi qua mọi điểm.
    \item Tuy nhiên, đây là \textbf{Quá khớp nghiêm trọng}. Mô hình bậc 2 (Quadratic) mới là mô hình tổng quát hóa tốt nhất (Generalize) cho bộ dữ liệu này.
    \item Làm sao nhận biết mô hình đang quá khớp hay dưới khớp? Hãy sử dụng Đường cong học tập.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-14.png}
\caption{Hình 4-14. Hồi quy đa thức bậc cao (Overfitting)}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Đánh giá mô hình bằng Đường cong học tập (Learning Curves)}
\begin{itemize}
    \item Đường cong học tập là các biểu đồ thể hiện lỗi trên tập Huấn luyện (Train) và tập Xác thực (Validation) của mô hình dưới dạng hàm của \textbf{kích thước tập huấn luyện}.
    \item \textbf{Dưới khớp (Underfitting):} Cả hai đường đều đạt cao nguyên, lỗi cao và sát nhau (không cải thiện thêm dù cho dữ liệu).
    \item \textbf{Quá khớp (Overfitting):} Lỗi trên tập huấn luyện thấp hơn nhiều, và có \textbf{khoảng cách (gap)} đáng kể giữa lỗi huấn luyện và lỗi xác thực (Mô hình hoạt động rất tốt trên tập huấn luyện nhưng kém trên dữ liệu mới).
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Đường cong học tập (Dưới khớp và Quá khớp)}
\begin{columns}
\column{0.5\textwidth}
\begin{figure}
\centering
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-15.png}
\caption{Hình 4-15. Dưới khớp (Đường cong sát nhau, lỗi cao)}
\end{figure}
\column{0.5\textwidth}
\begin{figure}
\centering
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-16.png}
\caption{Hình 4-16. Quá khớp (Khoảng cách rõ ràng giữa 2 đường cong)}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Sự đánh đổi giữa Độ chệch (Bias) và Phương sai (Variance)}
\begin{itemize}
    \item Lỗi tổng quát hóa có thể phân rã thành ba loại lỗi:
    \item \textbf{Độ chệch (Bias):} Lỗi do các giả định sai của mô hình (Vd: giả định dữ liệu tuyến tính khi thực tế là phi tuyến). Độ chệch cao thường gây ra underfitting.
    \item \textbf{Phương sai (Variance):} Lỗi do mô hình quá nhạy cảm với biến động nhỏ trong tập huấn luyện. Một mô hình nhiều bậc tự do (bậc cao) có phương sai cao, gây ra overfitting.
    \item \textbf{Lỗi không thể giảm (Irreducible error):} Do độ nhiễu của bản thân dữ liệu. Chỉ có thể giảm bằng cách làm sạch dữ liệu.
    \item \textbf{Sự đánh đổi:} Tăng độ phức tạp mô hình $\rightarrow$ Tăng variance, giảm bias. Giảm độ phức tạp $\rightarrow$ Tăng bias, giảm variance.
\end{itemize}
\end{frame}

\begin{frame}
\vfill\centering\LARGE\textbf{Chính quy hóa (Regularization) Mô hình Tuyến tính}\vfill
\end{frame}

\begin{frame}{Hồi quy Ridge (Chuẩn hóa Tikhonov)}
\begin{itemize}
    \item Một cách tốt để giảm Overfitting là \textbf{chính quy hóa (ràng buộc)} mô hình. Đối với mô hình tuyến tính, điều này thường thực hiện bằng cách ràng buộc kích thước của các trọng số $\boldsymbol{\theta}$.
    \item \textbf{Hồi quy Ridge} thêm một số hạng phạt $l_2$ (chuẩn $l_2$) vào hàm chi phí MSE:
    \begin{equation}
    J(\boldsymbol{\theta}) = MSE(\boldsymbol{\theta}) + \alpha \frac{1}{2} \sum_{i=1}^{n} \theta_i^2
    \end{equation}
    \item $\alpha$ kiểm soát mức độ chính quy hóa. Nếu $\alpha=0$, nó là hồi quy tuyến tính bình thường. Nếu $\alpha$ lớn, tất cả trọng số $\theta$ sẽ gần về 0, tạo thành đường thẳng phẳng cắt qua trung bình dữ liệu.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Hồi quy Ridge}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item \textbf{Bên trái:} Hồi quy Ridge áp dụng trên mô hình tuyến tính thông thường. $\alpha$ tăng làm cho đường dự đoán phẳng hơn, giảm độ chệch của cực đoan.
    \item \textbf{Bên phải:} Hồi quy Ridge áp dụng sau biến đổi PolynomialFeatures(degree=10). Dễ dàng thấy $\alpha=0$ thì dao động cực mạnh, nhưng $\alpha$ lớn thì đường cong trở nên êm hơn nhiều, giảm overfitting đáng kể.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-17.png}
\caption{Hình 4-17. Hồi quy Ridge trên mô hình tuyến tính và đa thức}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Hồi quy Lasso}
\begin{itemize}
    \item \textbf{Hồi quy Lasso} (Least Absolute Shrinkage and Selection Operator Regression) tương tự Ridge nhưng sử dụng chuẩn $l_1$ làm số hạng phạt:
    \begin{equation}
    J(\boldsymbol{\theta}) = MSE(\boldsymbol{\theta}) + \alpha \sum_{i=1}^{n} |\theta_i|
    \end{equation}
    \item Điểm đặc biệt nhất của Lasso: Nó có xu hướng hoàn toàn loại bỏ (đặt trọng số bằng 0) đối với các đặc trưng ít quan trọng nhất.
    \item Do đó, Lasso tự động thực hiện **lựa chọn đặc trưng (feature selection)**, tạo ra một mô hình thưa (sparse model).
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Hồi quy Lasso}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Hình bên minh họa Lasso trên dữ liệu tuyến tính (trái) và đa thức (phải).
    \item Với $\alpha = 1$, đối với mô hình đa thức, đường dự đoán gần như trở thành một đường thẳng bậc thấp, chứng tỏ các tham số bậc cao đã bị đưa về giá trị 0 hoàn toàn.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-18.png}
\caption{Hình 4-18. Hồi quy Lasso trên mô hình tuyến tính và đa thức}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Điểm khác biệt giữa Lasso và Ridge trong không gian tham số}
\begin{itemize}
    \item Ridge (chuẩn $l_2$) khiến các trọng số giảm dần khi tiến tới gốc tọa độ, gradient trở nên nhỏ nên việc đi đến tối ưu hội tụ êm và không dao động. Các tham số không bao giờ bị loại bỏ hoàn toàn (không về 0).
    \item Lasso (chuẩn $l_1$) thì hàm chi phí giảm tuyến tính khi tiến tới các trục (trục $\theta_1$, $\theta_2$). Hạ gradient trên Lasso sẽ nhanh chóng khiến một tham số về 0 (như $\theta_2=0$), sau đó trượt trên trục $\theta_1$ cho đến khi cả hai đạt cực tiểu. 
    \item Kết quả: Lasso tạo ra mô hình thưa và loại bỏ đặc trưng.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Lasso vs Ridge trong Không gian tham số}
\begin{figure}
\centering
\includegraphics[width=0.7\textwidth,height=0.7\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-19.png}
\caption{Hình 4-19. Chính quy hóa Lasso (Trên) và Ridge (Dưới)}
\end{figure}
\end{frame}

\begin{frame}{Hồi quy Elastic Net \& Khi nào nên dùng?}
\begin{itemize}
    \item \textbf{Elastic Net} là sự kết hợp trung gian giữa Ridge và Lasso. Hàm phạt của nó kết hợp cả chuẩn $l_1$ và $l_2$, được điều khiển bởi tỷ lệ trộn $r$.
    \item Khi $r=0$, nó là Ridge; Khi $r=1$, nó là Lasso.
    \item \textbf{Khi nào sử dụng loại nào?}
    \begin{itemize}
        \item Luôn ưu tiên dùng một ít chính quy hóa. \textbf{Ridge là lựa chọn mặc định tốt.}
        \item Nếu bạn nghi ngờ chỉ có một vài đặc trưng là hữu ích (Nhiễu nhiều), dùng \textbf{Lasso hoặc Elastic Net} để loại bỏ các đặc trưng vô ích.
        \item Nói chung, \textbf{Elastic Net được ưa chuộng hơn Lasso} vì Lasso có thể hoạt động thất thường khi số lượng đặc trưng lớn hơn số trường hợp, hoặc khi các đặc trưng có tương quan mạnh với nhau.
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{Dừng sớm (Early Stopping)}
\begin{itemize}
    \item Một cách rất khác (và đơn giản) để chính quy hóa các thuật toán học lặp (như GD) là dừng huấn luyện ngay khi Lỗi xác thực (Validation Error) đạt đến cực tiểu.
    \item Khi các epoch trôi qua, lỗi trên tập huấn luyện luôn giảm. Lỗi xác thực cũng giảm ban đầu, nhưng sau đó sẽ tăng ngược trở lại (Đó là lúc mô hình bắt đầu Overfitting).
    \item Chỉ cần dừng thuật toán ngay tại điểm lỗi xác thực chạm mức thấp nhất. Kỹ thuật này hiệu quả đến mức Geoffrey Hinton gọi nó là \textbf{"bữa trưa miễn phí tuyệt vời"}.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Chính quy hóa bằng Dừng sớm (Early Stopping)}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Mũi tên chỉ vào thời điểm tốt nhất để dừng huấn luyện. 
    \item Các bước sau đó chỉ làm mô hình tệ hơn khi đối mặt với dữ liệu chưa từng thấy (Mặc dù lỗi RMSE trên tập huấn luyện vẫn đang giảm).
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-20.png}
\caption{Hình 4-20. Early Stopping}
\end{figure}
\end{columns}
\end{frame}

\section{4. Hồi quy Logistic \& Softmax}

\begin{frame}
\vfill\centering\LARGE\textbf{4. Hồi quy Logistic \& Softmax}\vfill
\end{frame}

\begin{frame}{Hồi quy Logistic (Logistic Regression)}
\begin{itemize}
    \item Dù mang tên "Hồi quy", đây lại là mô hình thường được sử dụng cho \textbf{Phân loại} (Thường là nhị phân - 2 lớp).
    \item Mô hình ước tính xác suất một trường hợp thuộc về một lớp cụ thể (Vd: 70\% là thư rác). 
    \item Nếu xác suất ước tính $> 50\%$, nó dự đoán thuộc lớp "1" (dương). Ngược lại là lớp "0" (âm).
    \item Hoạt động giống Hồi quy tuyến tính, tính tổng trọng số các đặc trưng. Nhưng thay vì xuất kết quả trực tiếp, nó xuất ra \textbf{hàm logistic} của kết quả đó.
    \begin{equation}
    \hat{p} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \sigma(\boldsymbol{\theta}^\top \mathbf{x})
    \end{equation}
\end{itemize}
\end{frame}

\begin{frame}{Hàm Logistic (Sigmoid)}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Hàm logistic ($\sigma$) là một hàm sigmoid (có hình chữ S) lấy đầu vào là bất kỳ số thực nào và trả về kết quả trong khoảng từ 0 đến 1.
    \item $\sigma(t) = \frac{1}{1 + \exp(-t)}$
    \item Biểu đồ bên cho thấy giá trị $\sigma(t)$ tiến về 1 khi $t$ lớn và tiến về 0 khi $t$ là số âm lớn.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-21.png}
\caption{Hình 4-21. Đồ thị Hàm Logistic (Sigmoid)}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Hàm huấn luyện và Hàm chi phí (Log Loss)}
\begin{itemize}
    \item Hàm chi phí đối với một trường hợp huấn luyện đơn lẻ:
    \begin{equation}
    c(\boldsymbol{\theta}) = 
    \begin{cases}
    -\log(\hat{p}) & \text{nếu } y = 1 \\
    -\log(1 - \hat{p}) & \text{nếu } y = 0
    \end{cases}
    \end{equation}
    \item Có ý nghĩa: $-\log(t)$ rất lớn khi $t$ tiến tới 0. Lỗi rất lớn nếu mô hình dự đoán xác suất gần 0 cho lớp thực sự là 1, và ngược lại.
    \item \textbf{Hàm chi phí trên toàn bộ tập (Log Loss):}
    \begin{equation}
    J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]
    \end{equation}
    \item Không có phương trình dạng đóng để tính trực tiếp (như PT Chuẩn tắc), nhưng Log Loss là hàm lồi nên có thể dùng \textbf{Gradient Descent} để tối ưu.
\end{itemize}
\end{frame}

\begin{frame}{Tập dữ liệu hoa Diên vĩ (Iris Dataset)}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Để minh họa trực quan Logistic Regression, chúng ta dùng tập dữ liệu nổi tiếng Iris.
    \item Gồm thông số chiều dài/chiều rộng đài hoa và cánh hoa của 150 bông diên vĩ thuộc 3 loài: \textit{Setosa, Versicolor, Virginica}.
    \item Bài toán: Phân loại một bông hoa có phải loài \textit{Virginica} hay không dựa trên chiều rộng cánh hoa.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-22.png}
\caption{Hình 4-22. Ba loài hoa Diên vĩ}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Các đường ranh giới quyết định (Decision Boundaries)}
\begin{itemize}
    \item Khi mô hình đã được huấn luyện, chúng ta xem các xác suất ước tính của mô hình cho các bông hoa.
    \item Ranh giới quyết định là điểm mà tại đó mô hình thay đổi quyết định (xác suất $= 50\%$).
    \item Mô hình Hồi quy Logistic có ranh giới quyết định là tuyến tính. Nếu có 1 đặc trưng, nó là 1 điểm; có 2 đặc trưng, nó là 1 đường thẳng; $>2$ đặc trưng, nó là một siêu mặt phẳng.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Ranh giới quyết định Hồi quy Logistic}
\begin{columns}
\column{0.5\textwidth}
\begin{figure}
\centering
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-23.png}
\caption{Hình 4-23. Ranh giới quyết định (1 chiều - chiều rộng cánh hoa)}
\end{figure}
\column{0.5\textwidth}
\begin{figure}
\centering
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-24.png}
\caption{Hình 4-24. Đường ranh giới tuyến tính (2 chiều - rộng \& dài)}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}{Hồi quy Softmax (Hồi quy Logistic Đa thức)}
\begin{itemize}
    \item Hồi quy Logistic có thể tổng quát hóa hỗ trợ trực tiếp đa lớp mà không cần phải kết hợp nhiều bộ phân loại nhị phân (OvR). Được gọi là \textbf{Hồi quy Softmax}.
    \item Cách thức: 
    \begin{enumerate}
        \item Tính điểm số $s_k(\mathbf{x})$ cho từng lớp $k$: $s_k(\mathbf{x}) = (\boldsymbol{\theta}^{(k)})^\top \mathbf{x}$
        \item Biến đổi điểm số thành xác suất ước tính $\hat{p}_k$ thông qua \textbf{Hàm Softmax}.
    \end{enumerate}
    \item Hàm Softmax tính toán cấp số nhân (hàm mũ) của mọi điểm số rồi chuẩn hóa chúng.
    \begin{equation}
    \hat{p}_k = \sigma(\mathbf{s}(\mathbf{x}))_k = \frac{\exp(s_k(\mathbf{x}))}{\sum_{j=1}^{K} \exp(s_j(\mathbf{x}))}
    \end{equation}
\end{itemize}
\end{frame}

\begin{frame}{Dự đoán và Hàm chi phí Cross Entropy}
\begin{itemize}
    \item Mô hình Softmax sẽ dự đoán lớp có xác suất cao nhất.
    \item \textbf{Hàm chi phí Cross Entropy (Mất mát chéo):} 
    \begin{equation}
    J(\boldsymbol{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log\left(\hat{p}_k^{(i)}\right)
    \end{equation}
    \item Cross entropy rất hiệu quả vì nó phạt nặng các mô hình khi ước lượng một xác suất thấp cho lớp đúng (mục tiêu).
    \item Khi $K = 2$ (2 lớp), Cross Entropy tương đương chính xác với hàm Log Loss của Logistic Regression.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Ranh giới quyết định của Hồi quy Softmax}
\begin{columns}
\column{0.5\textwidth}
\begin{itemize}
    \item Chạy Softmax trên cả 3 lớp hoa Iris.
    \item Hình bên cho thấy ranh giới phân định giữa 3 lớp. 
    \item Các ranh giới quyết định giữa bất kỳ hai lớp nào đều là \textbf{tuyến tính} (Đường thẳng).
    \item Tại trung tâm, ranh giới 3 lớp giao nhau, xác suất các lớp đều bằng 33\%.
\end{itemize}
\column{0.5\textwidth}
\begin{figure}
\includegraphics[width=\textwidth,height=0.6\textheight,keepaspectratio]{../machineLearningWeb/docs/../Figures/CH04/Hinh_4-25.png}
\caption{Hình 4-25. Ranh giới quyết định hồi quy Softmax}
\end{figure}
\end{columns}
\end{frame}

\begin{frame}
\vfill\centering\LARGE\textbf{Tổng kết Chương 4}\vfill
\end{frame}

\begin{frame}{Tổng kết Chương 4}
\begin{itemize}
    \item Chúng ta đã khám phá bên trong "Hộp đen" của Hồi quy tuyến tính và giải bằng Phương trình chuẩn tắc, SVD.
    \item Hiểu rõ và áp dụng thuật toán tối ưu mạnh mẽ là \textbf{Gradient Descent} dưới nhiều hình thức (Batch, SGD, Mini-batch) cũng như tầm quan trọng của tốc độ học (Learning rate).
    \item Mở rộng khả năng của mô hình cho dữ liệu phức tạp bằng **Hồi quy đa thức**.
    \item Biết cách sử dụng **Đường cong học tập** để chẩn đoán Underfitting/Overfitting.
    \item Triển khai các phương pháp **Chính quy hóa** (Ridge, Lasso, Elastic Net, Early Stopping) để ngăn chặn Overfitting.
    \item Làm quen với các công cụ Phân loại xác suất thông qua **Hồi quy Logistic** và **Hồi quy Softmax**.
\end{itemize}
\end{frame}

\end{document}
"""

with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(latex_code)
print('Generated Slide_ML_Chap04.tex successfully.')
