import os

def generate_slides():
    tex_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap06.tex"
    
    latex_code = r"""\documentclass[aspectratio=169]{beamer}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setsansfont{Arial}
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

\title{Chương 6. Cây quyết định}
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
    \item Hiểu nguyên lý hoạt động, cách huấn luyện, đưa ra dự đoán và đo lường độ không tinh khiết.
    \item Nắm bắt thuật toán huấn luyện CART.
    \item Cách chống quá khớp bằng các siêu tham số chính quy hóa.
    \item Áp dụng cây quyết định cho bài toán phân loại và hồi quy.
    \item Hiểu các hạn chế của mô hình như sự nhạy cảm hướng trục và phương sai cao.
\end{itemize}
\end{frame}

\section{Cơ bản về Cây quyết định \& Phân loại}

\begin{frame}{1. Giới thiệu chung về Cây quyết định}
\begin{itemize}
    \item Cây quyết định là thuật toán học máy linh hoạt có thể thực hiện phân loại, hồi quy và đa đầu ra.
    \item Khả năng khớp các tập dữ liệu phức tạp mạnh mẽ.
    \item Là thành phần cơ bản của thuật toán Rừng ngẫu nhiên (Random Forest).
    \item Ưu điểm:
    \begin{itemize}
        \item Dễ hiểu, dễ diễn giải.
        \item Cần ít quá trình tiền xử lý dữ liệu.
        \item Tốc độ dự đoán rất nhanh.
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{2. Huấn luyện và trực quan hóa}
\begin{itemize}
    \item Trong Scikit-Learn, sử dụng lớp \texttt{DecisionTreeClassifier}.
    \item Trực quan hóa cây bằng hàm \texttt{export\_graphviz()} hoặc thư viện \texttt{graphviz}.
    \item Mỗi nút trong cây quyết định biểu diễn một câu hỏi hoặc bài kiểm tra về đặc trưng.
    \item Các nhánh thể hiện câu trả lời (Đúng/Sai), và nút lá (leaf node) cho biết dự đoán cuối cùng.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Cây quyết định Iris}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH06/Hinh_6-1.png}
    \captionof{figure}{Hình 6-1. Cây quyết định Iris}
\end{center}
\end{frame}

\begin{frame}{3. Đưa ra dự đoán}
\begin{itemize}
    \item Quá trình dự đoán bắt đầu từ \textbf{nút gốc} (root node).
    \item Tại mỗi nút, mô hình kiểm tra một điều kiện (ví dụ: chiều dài cánh hoa $\le$ 2.45 cm).
    \item Di chuyển sang nhánh \textbf{Trái} (Đúng) hoặc \textbf{Phải} (Sai) tùy theo kết quả kiểm tra.
    \item Thuộc tính \texttt{samples} của một nút đếm số lượng trường hợp huấn luyện đi qua nút đó.
    \item Thuộc tính \texttt{value} cho biết số lượng mẫu của mỗi lớp tại nút đó.
    \item Nút \textbf{lá} (leaf node) quyết định kết quả dự đoán.
\end{itemize}
\end{frame}

\begin{frame}{Độ không tinh khiết Gini}
\begin{itemize}
    \item Thuộc tính \texttt{gini} đo độ "không tinh khiết" (impurity) của một nút.
    \item Nút là "thuần khiết" (\texttt{gini=0}) nếu tất cả các mẫu thuộc cùng một lớp.
    \item Công thức tính độ không tinh khiết Gini của nút $i$:
    \begin{equation}
        G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2
    \end{equation}
    \item Trong đó, $p_{i,k}$ là tỷ lệ các mẫu thuộc lớp $k$ trong tổng số mẫu ở nút $i$.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Đường biên quyết định}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH06/Hinh_6-2.png}
    \captionof{figure}{Hình 6-2. Đường biên quyết định của cây quyết định}
\end{center}
\end{frame}

\begin{frame}{4. Ước tính xác suất lớp}
\begin{itemize}
    \item Cây quyết định ước tính xác suất một trường hợp thuộc về một lớp cụ thể.
    \item Xác suất bằng tỷ lệ các mẫu của lớp đó trên tổng số mẫu tại nút lá tương ứng.
    \item Ví dụ, tại một nút lá có 54 mẫu, gồm 49 mẫu lớp A và 5 mẫu lớp B:
    \begin{itemize}
        \item Xác suất lớp A: $49/54 \approx 90.7\%$
        \item Xác suất lớp B: $5/54 \approx 9.3\%$
    \end{itemize}
    \item Lớp có xác suất cao nhất sẽ là kết quả phân loại cuối cùng.
\end{itemize}
\end{frame}

\begin{frame}{5. Thuật toán huấn luyện CART}
\begin{itemize}
    \item Scikit-Learn dùng thuật toán CART (Classification and Regression Trees).
    \item Hàm chi phí CART cho phân loại tìm cách chia dữ liệu thành hai tập con sao cho tổng độ không tinh khiết là nhỏ nhất:
    \begin{equation}
        J(k, t_k) = \frac{m_{\text{left}}}{m} G_{\text{left}} + \frac{m_{\text{right}}}{m} G_{\text{right}}
    \end{equation}
    \item $G$: độ không tinh khiết; $m$: số lượng mẫu.
    \item Quá trình chia đệ quy dừng khi đạt độ sâu tối đa (\texttt{max\_depth}) hoặc không thể giảm thêm độ không tinh khiết.
\end{itemize}
\end{frame}

\begin{frame}{6. Độ phức tạp tính toán}
\begin{itemize}
    \item \textbf{Dự đoán:} Duyệt cây quyết định yêu cầu đi qua khoảng $O(\log_2(m))$ nút.
    \item Mỗi nút chỉ kiểm tra 1 đặc trưng nên độ phức tạp dự đoán là $O(\log_2(m))$, không phụ thuộc vào số lượng đặc trưng. Dự đoán rất nhanh!
    \item \textbf{Huấn luyện:} Cần so sánh tất cả các đặc trưng trên mọi mẫu ở từng nút. Độ phức tạp là $O(n \times m \log(m))$.
\end{itemize}
\end{frame}

\begin{frame}{7. Độ không tinh khiết Gini hay Entropy?}
\begin{itemize}
    \item Có thể đổi độ đo từ \texttt{gini} (mặc định) sang \texttt{entropy} bằng \texttt{criterion="entropy"}.
    \item Khái niệm Entropy xuất phát từ nhiệt động lực học và lý thuyết thông tin, đo lường sự "hỗn loạn".
    \item Công thức Entropy:
    \begin{equation}
        H_i = -\sum_{k=1, p_{i,k} \ne 0}^{n} p_{i,k} \log_2(p_{i,k})
    \end{equation}
    \item Không có nhiều sự khác biệt thực tế giữa Gini và Entropy. Gini tính toán nhanh hơn. Entropy thường tạo ra các cây cân bằng hơn một chút.
\end{itemize}
\end{frame}

\section{Chính quy hóa, Hồi quy \& Các hạn chế}

\begin{frame}{8. Siêu tham số chính quy hóa}
\begin{itemize}
    \item Cây quyết định mặc định là mô hình phi tham số (nonparametric).
    \item Nếu không bị ràng buộc, nó sẽ khớp rất chặt với dữ liệu huấn luyện, dễ dẫn đến \textbf{quá khớp} (overfitting).
    \item Để tránh quá khớp, ta cần \textbf{chính quy hóa} (regularization) mô hình.
    \item Siêu tham số quan trọng nhất là \texttt{max\_depth} (giới hạn độ sâu tối đa của cây).
\end{itemize}
\end{frame}

\begin{frame}{Các siêu tham số chính quy hóa khác}
\begin{itemize}
    \item \texttt{min\_samples\_split}: Số lượng mẫu tối thiểu một nút phải có để tiếp tục phân chia.
    \item \texttt{min\_samples\_leaf}: Số lượng mẫu tối thiểu tại một nút lá.
    \item \texttt{min\_weight\_fraction\_leaf}: Tương tự trên nhưng tính theo tỷ lệ phần trăm.
    \item \texttt{max\_leaf\_nodes}: Số lượng nút lá tối đa.
    \item \texttt{max\_features}: Số lượng đặc trưng tối đa được xem xét tại mỗi lần chia.
    \item \textbf{Quy tắc chung:} Tăng các tham số \texttt{min\_*} hoặc giảm các tham số \texttt{max\_*} sẽ giúp chính quy hóa mô hình tốt hơn.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Tác dụng của chính quy hóa}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH06/Hinh_6-3.png}
    \captionof{figure}{Hình 6-3. Cây không chính quy hóa (trái) và cây chính quy hóa (phải)}
\end{center}
\end{frame}

\begin{frame}{9. Hồi quy bằng Cây quyết định}
\begin{itemize}
    \item Cây quyết định áp dụng cho hồi quy dự đoán một \textbf{giá trị liên tục} tại mỗi nút lá, thay vì một lớp.
    \item Trong Scikit-Learn, sử dụng lớp \texttt{DecisionTreeRegressor}.
    \item Giá trị dự đoán cho mỗi vùng chính là \textbf{giá trị mục tiêu trung bình} của các trường hợp trong vùng đó.
    \item Thuật toán cố gắng chia vùng sao cho hầu hết các mẫu gần nhất với giá trị trung bình đó (giảm thiểu sai số).
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Một cây quyết định cho hồi quy}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH06/Hinh_6-4.png}
    \captionof{figure}{Hình 6-4. Một cây quyết định cho hồi quy}
\end{center}
\end{frame}

\begin{frame}{Minh họa: Dự đoán của hai mô hình hồi quy}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH06/Hinh_6-5.png}
    \captionof{figure}{Hình 6-5. Dự đoán của hai mô hình hồi quy cây quyết định}
\end{center}
\end{frame}

\begin{frame}{Hàm chi phí CART cho hồi quy}
\begin{itemize}
    \item Thay vì giảm thiểu độ không tinh khiết Gini, thuật toán CART chia vùng để giảm thiểu \textbf{sai số bình phương trung bình (MSE)}.
    \item Hàm chi phí CART cho hồi quy:
    \begin{equation}
        J(k, t_k) = \frac{m_{\text{left}}}{m} \text{MSE}_{\text{left}} + \frac{m_{\text{right}}}{m} \text{MSE}_{\text{right}}
    \end{equation}
    \item Giống phân loại, cây hồi quy cũng rất dễ bị quá khớp nếu không sử dụng các siêu tham số chính quy hóa.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Chính quy hóa trong cây hồi quy}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH06/Hinh_6-6.png}
    \captionof{figure}{Hình 6-6. Cây hồi quy không (trái) và có chính quy hóa (phải)}
\end{center}
\end{frame}

\begin{frame}{10. Độ nhạy với hướng trục}
\begin{itemize}
    \item Hạn chế: Cây quyết định ưu tiên các đường biên quyết định trực giao (vuông góc với trục đặc trưng).
    \item Điều này làm cho chúng rất \textbf{nhạy cảm với hướng của dữ liệu}.
    \item Cùng một bộ dữ liệu, nếu bị xoay một góc, đường biên sẽ trở nên phức tạp kiểu "bậc thang" và mô hình khó tổng quát hóa.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Độ nhạy với việc xoay tập huấn luyện}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH06/Hinh_6-7.png}
    \captionof{figure}{Hình 6-7. Độ nhạy với việc xoay tập huấn luyện}
\end{center}
\end{frame}

\begin{frame}{Khắc phục nhạy cảm hướng trục bằng PCA}
\begin{itemize}
    \item Phương pháp khắc phục: Chuẩn hóa dữ liệu rồi dùng phép biến đổi \textbf{PCA (Phân tích thành phần chính)}.
    \item PCA giúp xoay dữ liệu theo hướng làm giảm sự tương quan giữa các đặc trưng, giúp cây quyết định dễ dàng tìm ra đường chia hơn.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Đường biên quyết định sau PCA}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH06/Hinh_6-8.png}
    \captionof{figure}{Hình 6-8. Đường biên trên tập dữ liệu đã chuẩn hóa và xoay PCA}
\end{center}
\end{frame}

\begin{frame}{11. Cây quyết định có phương sai cao}
\begin{itemize}
    \item Vấn đề lớn nhất của cây quyết định là \textbf{phương sai cao} (high variance).
    \item Các thay đổi nhỏ trên dữ liệu hoặc siêu tham số có thể tạo ra các cây hoàn toàn khác biệt.
    \item Thuật toán Scikit-Learn bản chất là ngẫu nhiên, việc huấn luyện lại trên cùng tập dữ liệu cũng ra kết quả khác nhau (trừ khi đặt \texttt{random\_state}).
    \item \textbf{Giải pháp:} Sử dụng \textbf{Rừng ngẫu nhiên (Random Forest)} để tính trung bình các dự đoán từ nhiều cây, giảm mạnh phương sai.
\end{itemize}
\end{frame}

\begin{frame}{Minh họa: Sự thay đổi mô hình khi huấn luyện lại}
\begin{center}
    \includegraphics[width=\textwidth,height=0.75\textheight,keepaspectratio]{../machineLearningWeb/Figures/CH06/Hinh_6-9.png}
    \captionof{figure}{Hình 6-9. Mô hình khác biệt khi huấn luyện lại}
\end{center}
\end{frame}

\begin{frame}{Tổng kết Chương 6}
\begin{itemize}
    \item Cây quyết định là mô hình trực quan, linh hoạt, giải quyết tốt cả phân loại và hồi quy.
    \item Các siêu tham số (như \texttt{max\_depth}) rất quan trọng để điều chỉnh mức độ khớp dữ liệu, chống quá khớp.
    \item Nhược điểm chính: Dễ bị quá khớp, nhạy cảm với trục dữ liệu và có phương sai cao.
    \item Tuy nhiên, đây là khối xây dựng cốt lõi cho các thuật toán Ensemble cực mạnh như Random Forest.
\end{itemize}
\end{frame}

\end{document}
"""

    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"Generated {tex_path}")

if __name__ == "__main__":
    generate_slides()
