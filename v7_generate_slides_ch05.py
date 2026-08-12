import os
import re

def generate_slides():
    tex_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap05.tex"
    
    latex_code = r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}
\usepackage[vietnamese]{babel}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{caption}

\usetheme{Madrid}
\usecolortheme{default}

% Hide "Figure:" prefix
\setbeamertemplate{caption}{\raggedright\insertcaption\par}

\title{Chương 5. Máy học Véc-tơ hỗ trợ (SVM)}
\author{Giảng viên: TS. Trần Thành Thắng}
\date{\today}

\begin{document}

\begin{frame}
\titlepage
\end{frame}

\begin{frame}{Nội dung Chương trình}
\tableofcontents
\end{frame}

\begin{frame}{Mục tiêu bài học}
\begin{itemize}
    \item Hiểu nguyên lý phân loại biên lớn (Large margin) của Máy học Véc-tơ hỗ trợ (SVM).
    \item Phân biệt giữa phân loại biên cứng và phân loại biên mềm.
    \item Hiểu cách SVM xử lý dữ liệu phi tuyến bằng các kỹ thuật như đặc trưng đa thức, đặc trưng tương tự, và Kernel Gaussian RBF.
    \item Tìm hiểu cách điều chỉnh các siêu tham số của SVM (như C, gamma, epsilon).
    \item Nắm rõ các lớp SVM khác nhau (LinearSVC, SVC, SGDClassifier) và sự khác biệt về độ phức tạp tính toán.
    \item Khám phá việc áp dụng SVM cho các bài toán hồi quy (Hồi quy SVM).
    \item Tìm hiểu chi tiết cơ chế hoạt động bên trong: bài toán đối ngẫu, kernel trick và hàm mất mát bản lề.
\end{itemize}
\end{frame}

\section{Phân loại SVM tuyến tính \& phi tuyến}

\begin{frame}{1. Phân loại SVM tuyến tính}
\begin{itemize}
    \item Một máy học véc-tơ hỗ trợ (SVM) là một mô hình học máy mạnh mẽ và linh hoạt.
    \item Đặc biệt hiệu quả với các tập dữ liệu phi tuyến có kích thước nhỏ đến trung bình.
    \item \textbf{Phân loại biên lớn (Large Margin Classification):} Ý tưởng cơ bản của SVM là khớp một “con đường” rộng nhất có thể giữa các lớp.
    \item Đường biên quyết định không chỉ phân tách hai lớp mà còn giữ khoảng cách xa nhất có thể so với các trường hợp huấn luyện gần nhất.
    \item Việc thêm nhiều trường hợp huấn luyện “ngoài lề” không ảnh hưởng đến đường biên quyết định.
    \item Đường này hoàn toàn được xác định (hoặc “hỗ trợ”) bởi các trường hợp nằm trên rìa của “con đường”, được gọi là \textbf{các véc-tơ hỗ trợ}.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Phân loại biên lớn}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-1.png}
    \captionof{figure}{Hình 5-1. Phân loại biên lớn}
\end{center}
\end{frame}

\begin{frame}{Độ nhạy của mô hình SVM}
\begin{itemize}
    \item Nếu áp đặt nghiêm ngặt rằng tất cả các trường hợp phải nằm ngoài “con đường” và ở đúng phía, ta gọi là \textbf{phân loại biên cứng (hard margin classification)}.
    \item Phân loại biên cứng có 2 vấn đề:
    \begin{enumerate}
        \item Chỉ hoạt động nếu dữ liệu có thể phân tách tuyến tính.
        \item Rất nhạy cảm với các \textbf{ngoại lệ (outliers)}.
    \end{enumerate}
    \item Hơn nữa, SVM cũng rất \textbf{nhạy cảm với thang đo đặc trưng}.
    \item Nếu một đặc trưng có thang đo lớn hơn nhiều so với đặc trưng khác, "con đường" sẽ gần như song song với trục có thang đo lớn. Do đó, việc chuẩn hóa dữ liệu (ví dụ bằng \texttt{StandardScaler}) là rất quan trọng.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Độ nhạy với thang đo đặc trưng}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-2.png}
    \captionof{figure}{Hình 5-2. Độ nhạy với thang đo đặc trưng}
\end{center}
\end{frame}

\begin{frame}{Minh họa: Độ nhạy của biên cứng với các ngoại lệ}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-3.png}
    \captionof{figure}{Hình 5-3. Độ nhạy của biên cứng với các ngoại lệ}
\end{center}
\end{frame}

\begin{frame}{Phân loại biên mềm}
\begin{itemize}
    \item Để tránh vấn đề của phân loại biên cứng, ta dùng \textbf{phân loại biên mềm (soft margin classification)}.
    \item Mục tiêu: Tìm sự cân bằng giữa việc giữ “con đường” lớn nhất có thể và hạn chế \textbf{vi phạm biên} (các trường hợp nằm ở giữa đường hoặc sai phía).
    \item Siêu tham số điều chỉnh là $C$:
    \begin{itemize}
        \item $C$ thấp: “con đường” lớn hơn, nhưng dẫn đến nhiều vi phạm biên hơn (ít rủi ro quá khớp).
        \item $C$ cao: “con đường” hẹp hơn, ít vi phạm biên hơn (có nguy cơ quá khớp nếu quá cao).
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Biên lớn so với ít vi phạm biên}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-4.png}
    \captionof{figure}{Hình 5-4. Biên lớn (trái) so với ít vi phạm biên hơn (phải)}
\end{center}
\end{frame}

\begin{frame}{2. Phân loại SVM phi tuyến}
\begin{itemize}
    \item Mặc dù SVM tuyến tính hiệu quả, nhiều tập dữ liệu không thể phân tách tuyến tính.
    \item Một cách tiếp cận là thêm các \textbf{đặc trưng đa thức} ($x_1^2, x_1^3, x_1x_2, \dots$).
    \item Trong một số trường hợp, việc thêm đặc trưng mới có thể biến một tập dữ liệu 1D hoặc 2D không thể phân tách tuyến tính thành tập dữ liệu có thể phân tách tuyến tính ở không gian chiều cao hơn.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Thêm đặc trưng để phân tách tuyến tính}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-5.png}
    \captionof{figure}{Hình 5-5. Thêm đặc trưng để làm cho một tập dữ liệu có thể phân tách tuyến tính}
\end{center}
\end{frame}

\begin{frame}{Minh họa: SVM tuyến tính sử dụng đặc trưng đa thức}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-6.png}
    \captionof{figure}{Hình 5-6. Bộ phân loại SVM tuyến tính sử dụng các đặc trưng đa thức}
\end{center}
\end{frame}

\begin{frame}{Kernel đa thức (Polynomial Kernel)}
\begin{itemize}
    \item Việc thêm nhiều đặc trưng đa thức thủ công bằng \texttt{PolynomialFeatures} có thể làm chậm mô hình khi bậc cao.
    \item SVM cung cấp một kỹ thuật toán học là \textbf{Kernel trick}.
    \item Kernel trick giúp đạt được kết quả tương tự như khi thêm nhiều đặc trưng đa thức mà không thực sự phải thêm chúng, ngăn chặn sự bùng nổ tổ hợp (combinatorial explosion) về số lượng đặc trưng.
    \item Trong Scikit-Learn: Sử dụng \texttt{SVC(kernel="poly", degree=3)}.
    \item Siêu tham số \texttt{coef0} kiểm soát mức độ ảnh hưởng của các số hạng bậc cao so với bậc thấp.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Bộ phân loại SVM với Kernel đa thức}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-7.png}
    \captionof{figure}{Hình 5-7. Các bộ phân loại SVM với kernel đa thức}
\end{center}
\end{frame}

\begin{frame}{Các đặc trưng tương tự (Similarity features)}
\begin{itemize}
    \item Kỹ thuật thứ hai để giải quyết dữ liệu phi tuyến là thêm các đặc trưng được tính toán bằng \textbf{hàm tương tự} (ví dụ: so sánh với một "điểm mốc").
    \item Hàm Gaussian RBF (Radial Basis Function):
    \begin{equation}
        \phi\gamma(\mathbf{x}, \ell) = \exp(-\gamma \|\mathbf{x} - \ell\|^2)
    \end{equation}
    \item Trong đó: $\ell$ là điểm mốc, $\gamma$ (gamma) điều chỉnh độ rộng của đường cong hình chuông.
    \item Phương pháp tạo điểm mốc đơn giản nhất: Đặt mỗi trường hợp huấn luyện là một điểm mốc, biến đổi $m$ điểm dữ liệu gốc thành $m$ đặc trưng RBF mới.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Đặc trưng tương tự sử dụng Gaussian RBF}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-8.png}
    \captionof{figure}{Hình 5-8. Các đặc trưng tương tự sử dụng Gaussian RBF}
\end{center}
\end{frame}

\begin{frame}{Kernel Gaussian RBF}
\begin{itemize}
    \item Tương tự đa thức, việc tính toán mọi đặc trưng tương tự RBF cho toàn tập dữ liệu cực kỳ tốn kém.
    \item \textbf{Kernel RBF} giúp mô phỏng việc thêm các đặc trưng này mà không thực sự tính toán chúng.
    \item Trong Scikit-Learn: \texttt{SVC(kernel="rbf", gamma=5, C=0.001)}.
    \item Tác dụng của $\gamma$ (gamma):
    \begin{itemize}
        \item $\gamma$ lớn: Phạm vi ảnh hưởng hẹp, đường biên quyết định uốn lượn mạnh (dễ quá khớp).
        \item $\gamma$ nhỏ: Phạm vi ảnh hưởng lớn, đường biên mượt mà (dễ dưới khớp).
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Bộ phân loại SVM sử dụng Kernel RBF}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-9.png}
    \captionof{figure}{Hình 5-9. Các bộ phân loại SVM sử dụng kernel RBF}
\end{center}
\end{frame}

\section{Các lớp SVM, Hồi quy SVM \& Cơ chế hoạt động}

\begin{frame}{3. Các lớp SVM và độ phức tạp tính toán}
\begin{itemize}
    \item \textbf{LinearSVC:} Triển khai bằng thư viện liblinear.
    \begin{itemize}
        \item Không hỗ trợ kernel trick.
        \item Mở rộng tuyến tính với tập dữ liệu lớn: $O(m \times n)$.
    \end{itemize}
    \item \textbf{SVC:} Triển khai bằng thư viện libsvm.
    \begin{itemize}
        \item Hỗ trợ kernel trick (poly, rbf, sigmoid).
        \item Chạy rất chậm khi số lượng trường hợp lớn: $O(m^2 \times n)$ đến $O(m^3 \times n)$.
        \item Lý tưởng cho dữ liệu phi tuyến vừa và nhỏ.
    \end{itemize}
    \item \textbf{SGDClassifier:} Sử dụng Gradient Descent.
    \begin{itemize}
        \item Mở rộng cực kỳ tốt ($O(m \times n)$) và hỗ trợ học ngoài lõi (out-of-core).
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{4. Hồi quy SVM (SVM Regression)}
\begin{itemize}
    \item SVM không chỉ dành cho phân loại mà còn có thể áp dụng cho \textbf{hồi quy}.
    \item Mục tiêu đảo ngược so với phân loại:
    \begin{itemize}
        \item Phân loại: Khớp con đường rộng nhất \textbf{không} chứa trường hợp (hạn chế vi phạm).
        \item Hồi quy: Khớp \textbf{càng nhiều trường hợp càng tốt} trên con đường, hạn chế các trường hợp rớt ra ngoài.
    \end{itemize}
    \item Chiều rộng của con đường được kiểm soát bằng siêu tham số $\epsilon$ (epsilon).
    \item \textbf{Tính không nhạy cảm với $\epsilon$:} Việc thêm các trường hợp bên trong lề sẽ không thay đổi dự đoán của mô hình.
    \item Scikit-Learn hỗ trợ hồi quy thông qua lớp \texttt{LinearSVR} và \texttt{SVR} (hỗ trợ kernel).
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Hồi quy tuyến tính SVM}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-10.png}
    \captionof{figure}{Hình 5-10. Hồi quy SVM}
\end{center}
\end{frame}

\begin{frame}{Minh họa: Hồi quy đa thức SVM bậc hai}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-11.png}
    \captionof{figure}{Hình 5-11. Hồi quy SVM sử dụng kernel đa thức bậc hai}
\end{center}
\end{frame}

\begin{frame}{5. Bên trong bộ phân loại SVM tuyến tính}
\begin{itemize}
    \item Bộ phân loại SVM tuyến tính dự đoán bằng hàm quyết định: $\boldsymbol{\theta}^T \mathbf{x} = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$.
    \item Nếu hàm quyết định dương, lớp là $1$; ngược lại, lớp là $0$.
    \item Biên quyết định là các điểm mà hàm bằng $0$. Các lề của "con đường" là các điểm hàm bằng $-1$ hoặc $+1$.
    \item \textbf{Cực tiểu hóa trọng số:} Trọng số $\boldsymbol{\theta}$ càng nhỏ thì khoảng cách giữa hai lề càng lớn. Để có biên lớn nhất, ta cần cực tiểu hóa vector trọng số.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Vector trọng số nhỏ hơn dẫn đến biên lớn hơn}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-12.png}
    \captionof{figure}{Hình 5-12. Một vector trọng số nhỏ hơn dẫn đến một biên lớn hơn}
\end{center}
\end{frame}

\begin{frame}{Hàm mất mát bản lề (Hinge loss)}
\begin{itemize}
    \item Ngoài việc dùng bộ giải phương trình bậc hai (Quadratic Programming - QP), SVM cũng có thể được huấn luyện bằng Gradient Descent.
    \item Khi đó, mô hình tối thiểu hóa \textbf{Hàm mất mát bản lề (Hinge loss)}.
    \item Hàm Hinge loss có dạng: $L(y, \hat{y}) = \max(0, 1 - y \hat{y})$.
    \item Tại những điểm đã vượt qua lề đúng phía (hàm quyết định $> 1$ cho lớp $1$), mất mát sẽ bằng $0$. Càng vi phạm xa, mất mát càng tăng tuyến tính (hoặc bậc hai nếu dùng Squared hinge loss).
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Hàm mất mát bản lề và bản lề bình phương}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH05/Hinh_5-13.png}
    \captionof{figure}{Hình 5-13. Hàm mất mát bản lề (trái) và hàm mất mát bản lề bình phương (phải)}
\end{center}
\end{frame}

\begin{frame}{Bài toán đối ngẫu (Dual problem)}
\begin{itemize}
    \item Bài toán tối ưu ban đầu của SVM gọi là \textbf{Bài toán gốc (Primal problem)}.
    \item Do đặc điểm toán học, bài toán gốc của SVM có thể chuyển sang \textbf{Bài toán đối ngẫu (Dual problem)}.
    \item Giải bài toán đối ngẫu sẽ cho kết quả hoàn toàn giống với bài toán gốc.
    \item Ưu điểm:
    \begin{itemize}
        \item Bài toán đối ngẫu giải nhanh hơn khi số lượng trường hợp nhỏ hơn số lượng đặc trưng.
        \item Quan trọng nhất, nó cho phép áp dụng \textbf{Kernel trick} - điều mà bài toán gốc không làm được.
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{6. Kernelized SVMs \& Kernel Trick}
\begin{itemize}
    \item Kernel Trick là việc thay thế tích vô hướng của các vector đặc trưng sau biến đổi $\phi(\mathbf{a})^T \phi(\mathbf{b})$ bằng một hàm số toán học (Kernel function) tính trực tiếp từ vector ban đầu $K(\mathbf{a}, \mathbf{b})$.
    \item Tức là, không cần phải thực sự thực hiện phép biến đổi chi phí cao $\phi(\mathbf{x})$, mà vẫn có được kết quả tương tự.
    \item Một số Kernel thông dụng:
    \begin{itemize}
        \item Tuyến tính: $K(\mathbf{a}, \mathbf{b}) = \mathbf{a}^T \mathbf{b}$
        \item Đa thức: $K(\mathbf{a}, \mathbf{b}) = (\gamma \mathbf{a}^T \mathbf{b} + r)^d$
        \item Gaussian RBF: $K(\mathbf{a}, \mathbf{b}) = \exp(-\gamma \|\mathbf{a} - \mathbf{b}\|^2)$
        \item Sigmoid: $K(\mathbf{a}, \mathbf{b}) = \tanh(\gamma \mathbf{a}^T \mathbf{b} + r)$
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{Tổng kết Chương 5}
\begin{itemize}
    \item \textbf{Máy học Véc-tơ hỗ trợ (SVM)} là một trong những mô hình phổ biến và mạnh mẽ nhất trong Machine Learning.
    \item SVM tuyến tính hoạt động theo cơ chế tối đa hóa biên (Margin) giữa các lớp, với sự hỗ trợ của các điểm rìa (Véc-tơ hỗ trợ).
    \item Với dữ liệu phức tạp phi tuyến, \textbf{Kernel Trick} cho phép SVM có thể tạo ra các ranh giới phân loại uyển chuyển bằng cách phóng chiếu ngầm dữ liệu lên không gian nhiều chiều.
    \item Việc cân bằng giữa số lượng vi phạm lề và độ rộng của lề (thông qua siêu tham số $C$) giúp hạn chế quá khớp.
    \item SVM không chỉ áp dụng cho phân loại (SVC) mà còn rất hiệu quả khi làm các bài toán hồi quy (SVR).
\end{itemize}
\end{frame}

\end{document}
"""
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print("Generated Slide_ML_Chap05.tex successfully.")

if __name__ == "__main__":
    generate_slides()
