import os

def generate_slides():
    tex_path = r"d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\slideML\Slide_ML_Chap07.tex"
    
    latex_code = r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setsansfont{Arial}
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

\title[Chương 7: Học tổ hợp \& Rừng ngẫu nhiên]{Học Máy (Machine Learning)\\Chương 7: Học tổ hợp và Rừng ngẫu nhiên}
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

\section{Học tổ hợp (Ensemble Learning)}
% Slide 3
\begin{frame}{Giới thiệu Học tổ hợp}
    \begin{itemize}
        \item \textbf{Định nghĩa:} Một phương pháp kết hợp các dự đoán của một nhóm các bộ dự đoán (bộ phân loại hoặc bộ hồi quy) để đạt được kết quả dự đoán tốt hơn so với bất kỳ bộ dự đoán riêng lẻ nào.
        \item Một nhóm các bộ dự đoán được gọi là \textbf{một tập hợp (ensemble)}.
        \item Phương pháp học này được gọi là \textbf{học tổ hợp (ensemble learning)}.
        \item Thuật toán sử dụng kỹ thuật này gọi là \textbf{phương pháp tổ hợp (ensemble method)}.
    \end{itemize}
\end{frame}

% Slide 4
\begin{frame}{Sự khôn ngoan của đám đông}
    \begin{itemize}
        \item Giả sử bạn đặt một câu hỏi phức tạp cho hàng nghìn người ngẫu nhiên, sau đó tổng hợp câu trả lời của họ.
        \item Trong nhiều trường hợp, câu trả lời tổng hợp này tốt hơn câu trả lời của một chuyên gia $\rightarrow$ \textbf{Sự khôn ngoan của đám đông}.
        \item \textbf{Ví dụ:} Huấn luyện một nhóm các bộ phân loại cây quyết định trên các tập con ngẫu nhiên khác nhau của tập huấn luyện $\rightarrow$ \textbf{Rừng ngẫu nhiên (Random Forest)}.
        \item Thường được áp dụng gần cuối một dự án Machine Learning để đẩy tối đa độ chính xác của hệ thống.
    \end{itemize}
\end{frame}

\section{Bộ phân loại biểu quyết}
% Slide 5
\begin{frame}{Bộ phân loại biểu quyết (Voting Classifiers)}
    \begin{itemize}
        \item Bắt đầu bằng việc huấn luyện một vài bộ phân loại độc lập trên cùng một tập huấn luyện.
        \item Mỗi bộ phân loại có thể thuộc các loại khác nhau: SVM, Logistic Regression, Random Forest, K-Nearest Neighbors...
        \item Mỗi bộ phân loại đạt độ chính xác riêng lẻ, ví dụ khoảng 80\%.
        \item Kỹ thuật: Tổng hợp các dự đoán của tất cả các bộ phân loại và đưa ra dự đoán tập hợp.
    \end{itemize}
\end{frame}

% Slide 6
\begin{frame}{Huấn luyện các bộ phân loại đa dạng}
    \begin{center}
        \includegraphics[width=0.8\textwidth]{../machineLearningWeb/Figures/CH07/Hinh_7-1.png}\\
        \vspace{0.3cm}
        \textit{Hình 7-1. Huấn luyện các bộ phân loại đa dạng}
    \end{center}
\end{frame}

% Slide 7
\begin{frame}{Bộ phân loại bỏ phiếu cứng (Hard Voting)}
    \begin{itemize}
        \item \textbf{Bỏ phiếu cứng (Hard voting):} Dự đoán của tập hợp là lớp nhận được nhiều phiếu bầu đa số nhất từ các bộ phân loại riêng lẻ.
        \item Mặc dù đơn giản, bộ phân loại bỏ phiếu cứng thường đạt được độ chính xác cao hơn bộ phân loại tốt nhất trong tập hợp.
        \item Ngay cả khi mỗi bộ phân loại chỉ là một \textbf{bộ học yếu} (weak learner - chỉ tốt hơn đoán ngẫu nhiên một chút), tập hợp vẫn có thể trở thành một \textbf{bộ học mạnh} (strong learner) nếu có đủ số lượng bộ học đa dạng.
    \end{itemize}
\end{frame}

% Slide 8
\begin{frame}{Dự đoán bỏ phiếu cứng}
    \begin{center}
        \includegraphics[width=0.8\textwidth]{../machineLearningWeb/Figures/CH07/Hinh_7-2.png}\\
        \vspace{0.3cm}
        \textit{Hình 7-2. Dự đoán của bộ phân loại bỏ phiếu cứng}
    \end{center}
\end{frame}

% Slide 9
\begin{frame}{Nguyên lý đằng sau bộ phân loại biểu quyết}
    \begin{itemize}
        \item Làm sao để các bộ học yếu kết hợp lại thành bộ học mạnh?
        \item \textbf{Phép tương tự:} Tung một đồng xu hơi lệch (51\% sấp, 49\% ngửa).
        \item Nếu tung 1.000 lần, tỷ lệ mặt sấp sẽ tiến gần 51\%. Xác suất để đa số (trên 500 lần) là mặt sấp lên đến $\sim 75\%$.
        \item Tung 10.000 lần, xác suất đạt đa số mặt sấp tăng lên $>97\%$.
        \item Đây là ứng dụng của \textbf{Luật số lớn (Law of Large Numbers)}.
    \end{itemize}
\end{frame}

% Slide 10
\begin{frame}{Luật số lớn}
    \begin{center}
        \includegraphics[width=0.7\textwidth]{../machineLearningWeb/Figures/CH07/Hinh_7-3.png}\\
        \vspace{0.3cm}
        \textit{Hình 7-3. Luật số lớn - Tỷ lệ mặt sấp khi số lần tung tăng}
    \end{center}
    \begin{itemize}
        \item \textit{Lưu ý:} Tập hợp chỉ hoạt động hoàn hảo khi các bộ phân loại \textbf{hoàn toàn độc lập}, mắc lỗi không tương quan. Do cùng dùng một tập dữ liệu, chúng thường tương quan $\rightarrow$ làm giảm độ chính xác tập hợp so với kỳ vọng lý thuyết.
    \end{itemize}
\end{frame}

% Slide 11
\begin{frame}[fragile]{Mã nguồn: Bộ phân loại bỏ phiếu cứng (Phần 1)}
    Sử dụng \texttt{VotingClassifier} trong Scikit-Learn trên tập dữ liệu moons:
    \begin{lstlisting}[language=Python]
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    \end{lstlisting}
\end{frame}

% Slide 12
\begin{frame}[fragile]{Mã nguồn: Bộ phân loại bỏ phiếu cứng (Phần 2)}
    Định nghĩa và huấn luyện tập hợp:
    \begin{lstlisting}[language=Python]
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)

# Kiem tra ket qua:
voting_clf.score(X_test, y_test) # Output: 0.912
    \end{lstlisting}
    \begin{itemize}
        \item Độ chính xác 91.2\%, cao hơn độ chính xác của Logistic (86.4\%), Random Forest (89.6\%) và SVC (89.6\%).
    \end{itemize}
\end{frame}

% Slide 13
\begin{frame}{Bỏ phiếu mềm (Soft Voting)}
    \begin{itemize}
        \item Nếu tất cả các bộ phân loại có khả năng ước tính xác suất lớp (có phương thức \texttt{predict\_proba()}), tập hợp có thể đưa ra dự đoán dựa trên \textbf{trung bình xác suất} của từng lớp trên tất cả các bộ phân loại.
        \item \textbf{Bỏ phiếu mềm (Soft voting):} Chọn lớp có xác suất trung bình cao nhất.
        \item Thường đạt hiệu suất cao hơn bỏ phiếu cứng vì nó ưu tiên các phiếu bầu có độ tin cậy cao.
    \end{itemize}
\end{frame}

% Slide 14
\begin{frame}[fragile]{Mã nguồn: Bỏ phiếu mềm}
    Đặt siêu tham số \texttt{voting="soft"} và đảm bảo các mô hình (như SVC) có tính xác suất (\texttt{probability=True}):
    \begin{lstlisting}[language=Python]
voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)
# Output: 0.92
    \end{lstlisting}
    \begin{itemize}
        \item Độ chính xác tăng lên 92\% chỉ nhờ sử dụng bỏ phiếu mềm.
    \end{itemize}
\end{frame}

\section{Túi hóa (Bagging) và Dán nhãn (Pasting)}
% Slide 15
\begin{frame}{Túi hóa (Bagging) và Dán nhãn (Pasting)}
    \begin{itemize}
        \item Thay vì dùng nhiều thuật toán khác nhau, một cách tiếp cận khác là sử dụng \textbf{cùng một thuật toán} cho mỗi bộ dự đoán nhưng huấn luyện trên các \textbf{tập con ngẫu nhiên khác nhau} của tập huấn luyện.
        \item \textbf{Túi hóa (Bagging - Bootstrap Aggregating):} Lấy mẫu ngẫu nhiên \textit{có hoàn lại} (có thay thế). Một mẫu có thể được chọn nhiều lần cho cùng một bộ dự đoán.
        \item \textbf{Dán nhãn (Pasting):} Lấy mẫu ngẫu nhiên \textit{không hoàn lại} (không thay thế). Một mẫu chỉ được chọn nhiều nhất một lần cho mỗi bộ dự đoán.
    \end{itemize}
\end{frame}

% Slide 16
\begin{frame}{Đặc điểm của Bagging và Pasting}
    \begin{itemize}
        \item Hàm tổng hợp cuối cùng thường là \textbf{chế độ thống kê} (đối với phân loại) hoặc \textbf{giá trị trung bình} (đối với hồi quy).
        \item Mỗi bộ dự đoán riêng lẻ có \textbf{độ lệch (bias) cao hơn} so với khi huấn luyện trên toàn bộ dữ liệu.
        \item Nhưng kết hợp lại, tập hợp có độ lệch tương tự nhưng \textbf{phương sai (variance) thấp hơn} rất nhiều.
        \item \textbf{Mở rộng tốt:} Các bộ dự đoán độc lập nên có thể huấn luyện song song (tận dụng đa lõi CPU hoặc cụm máy chủ).
    \end{itemize}
\end{frame}

% Slide 17
\begin{frame}{Minh họa: Quá trình lấy mẫu Bagging/Pasting}
    \begin{center}
        \includegraphics[height=0.7\textheight]{../machineLearningWeb/Figures/CH07/Hinh_7-4.png}\\
        \vspace{0.3cm}
        \textit{Hình 7-4. Túi hóa và Dán nhãn trên các mẫu ngẫu nhiên}
    \end{center}
\end{frame}

% Slide 18
\begin{frame}{Bagging và Pasting trong Scikit-Learn}
    \begin{itemize}
        \item Scikit-Learn cung cấp lớp \texttt{BaggingClassifier} (hoặc \texttt{BaggingRegressor}).
        \item \textbf{Tham số quan trọng:}
            \begin{itemize}
                \item \texttt{n\_estimators}: Số lượng bộ dự đoán (ví dụ: 500 cây).
                \item \texttt{max\_samples}: Kích thước tập mẫu ngẫu nhiên cho mỗi bộ.
                \item \texttt{bootstrap=True}: Sử dụng Bagging (hoặc \texttt{False} cho Pasting).
                \item \texttt{n\_jobs=-1}: Sử dụng tất cả các lõi CPU để tăng tốc.
            \end{itemize}
    \end{itemize}
\end{frame}

% Slide 19
\begin{frame}[fragile]{Mã nguồn: BaggingClassifier}
    Huấn luyện tập hợp gồm 500 cây quyết định bằng Bagging:
    \begin{lstlisting}[language=Python]
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True, # Doi thanh False de dung Pasting
    n_jobs=-1, 
    random_state=42
)
bag_clf.fit(X_train, y_train)
    \end{lstlisting}
\end{frame}

% Slide 20
\begin{frame}{Phân tích kết quả của Bagging}
    \begin{itemize}
        \item Bagging mang lại nhiều đa dạng hơn trong các tập con huấn luyện, do đó Bagging có độ lệch nhỉnh hơn một chút so với Pasting.
        \item Sự đa dạng này giúp các bộ dự đoán ít tương quan hơn $\rightarrow$ \textbf{phương sai của tập hợp được giảm mạnh hơn}.
        \item Nhìn chung, Bagging thường mang lại mô hình tổng quát hóa tốt hơn và là lựa chọn ưu tiên mặc định.
    \end{itemize}
\end{frame}

% Slide 21
\begin{frame}{So sánh Cây đơn lẻ và Tập hợp túi hóa}
    \begin{center}
        \includegraphics[width=0.9\textwidth]{../machineLearningWeb/Figures/CH07/Hinh_7-5.png}\\
        \vspace{0.3cm}
        \textit{Hình 7-5. Cây quyết định đơn lẻ (trái) vs. Tập hợp Bagging 500 cây (phải)}
    \end{center}
\end{frame}

% Slide 22
\begin{frame}{Đánh giá ngoài mẫu (Out-of-Bag Evaluation)}
    \begin{itemize}
        \item Trong Bagging, việc lấy mẫu có hoàn lại (bootstrap) khiến một số trường hợp được lấy nhiều lần, số khác thì không được lấy.
        \item Thống kê cho thấy: mỗi bộ dự đoán chỉ lấy trung bình khoảng \textbf{63\%} số trường hợp.
        \item Khoảng \textbf{37\%} trường hợp không bao giờ được bộ dự đoán đó nhìn thấy gọi là các trường hợp \textbf{ngoài mẫu (Out-of-Bag - OOB)}.
        \item Có thể dùng 37\% OOB này để đánh giá (evaluate) tập hợp mà \textbf{không cần tập xác thực riêng biệt}.
    \end{itemize}
\end{frame}

% Slide 23
\begin{frame}[fragile]{Mã nguồn: Đánh giá OOB}
    Yêu cầu Scikit-Learn tính toán điểm OOB tự động bằng cách bật tham số \texttt{oob\_score=True}:
    \begin{lstlisting}[language=Python]
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    oob_score=True, n_jobs=-1, random_state=42
)
bag_clf.fit(X_train, y_train)

# Xem diem OOB
print(bag_clf.oob_score_) # Output: 0.896

# Kiem tra tren tap Test thuc te
from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred)) # Output: 0.92
    \end{lstlisting}
\end{frame}

% Slide 24
\begin{frame}{Random Patches và Random Subspaces}
    \begin{itemize}
        \item Lớp \texttt{BaggingClassifier} hỗ trợ lấy mẫu theo \textbf{đặc trưng (features)} thông qua \texttt{max\_features} và \texttt{bootstrap\_features}.
        \item \textbf{Random Patches:} Lấy mẫu cả \textit{trường hợp huấn luyện} và \textit{đặc trưng}. Rất hiệu quả cho dữ liệu nhiều chiều (chẳng hạn như hình ảnh).
        \item \textbf{Random Subspaces:} Giữ toàn bộ trường hợp huấn luyện (\texttt{bootstrap=False, max\_samples=1.0}) nhưng \textit{chỉ lấy mẫu ngẫu nhiên đặc trưng}.
        \item Đổi một chút độ lệch cao hơn để lấy phương sai thấp hơn.
    \end{itemize}
\end{frame}

\section{Rừng ngẫu nhiên (Random Forests)}
% Slide 25
\begin{frame}{Rừng ngẫu nhiên (Random Forests)}
    \begin{itemize}
        \item Rừng ngẫu nhiên là một tập hợp các cây quyết định, thường được huấn luyện bằng phương pháp \textbf{Bagging} với \texttt{max\_samples} bằng toàn bộ kích thước tập huấn luyện.
        \item Thuật toán Rừng ngẫu nhiên \textbf{đưa thêm sự ngẫu nhiên} vào việc phát triển cây: thay vì tìm kiếm đặc trưng tốt nhất trong toàn bộ đặc trưng, nó tìm kiếm \textbf{đặc trưng tốt nhất trong một tập con ngẫu nhiên} của các đặc trưng (mặc định là $\sqrt{n}$ đặc trưng).
        \item Dẫn đến sự đa dạng cây lớn hơn, đổi độ lệch cao hơn lấy phương sai thấp hơn, mang lại một mô hình tổng quát hóa tốt.
    \end{itemize}
\end{frame}

% Slide 26
\begin{frame}[fragile]{Mã nguồn: RandomForestClassifier}
    Sử dụng lớp \texttt{RandomForestClassifier} tiện lợi và tối ưu hơn so với dùng \texttt{BaggingClassifier} lồng \texttt{DecisionTreeClassifier}:
    \begin{lstlisting}[language=Python]
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    n_jobs=-1,
    random_state=42
)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
    \end{lstlisting}
\end{frame}

% Slide 27
\begin{frame}[fragile]{So sánh BaggingClassifier và RandomForestClassifier}
    Mã lệnh BaggingClassifier sau đây gần tương đương với \texttt{RandomForestClassifier} trước đó:
    \begin{lstlisting}[language=Python]
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, 
    n_jobs=-1, 
    random_state=42
)
    \end{lstlisting}
    \begin{itemize}
        \item \texttt{RandomForestClassifier} được tối ưu hiệu năng tốt hơn nhiều cho cây quyết định.
    \end{itemize}
\end{frame}

% Slide 28
\begin{frame}{Cây ngẫu nhiên cực đại (Extra-Trees)}
    \begin{itemize}
        \item Có thể làm cho cây ngẫu nhiên hơn nữa bằng cách sử dụng các \textbf{ngưỡng ngẫu nhiên} cho mỗi đặc trưng thay vì tìm ngưỡng chia cắt tốt nhất có thể.
        \item Tập hợp gồm các cây cực kỳ ngẫu nhiên này được gọi là \textbf{Extra-Trees} (Extremely Randomized Trees).
        \item Lợi ích: Tăng tốc độ huấn luyện đáng kể do không phải tốn kém thời gian tìm ngưỡng tốt nhất, đổi thêm một chút độ lệch lấy phương sai thấp hơn nữa.
    \end{itemize}
\end{frame}

% Slide 29
\begin{frame}[fragile]{Cài đặt Extra-Trees trong Scikit-Learn}
    Trong \texttt{DecisionTreeClassifier}, đặt tham số \texttt{splitter="random"} để tạo cây Extra-Tree đơn lẻ.
    Hoặc sử dụng tập hợp trực tiếp bằng \texttt{ExtraTreesClassifier}:
    \begin{lstlisting}[language=Python]
from sklearn.ensemble import ExtraTreesClassifier

extra_clf = ExtraTreesClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    n_jobs=-1,
    random_state=42
)
# API giong het voi RandomForestClassifier
    \end{lstlisting}
\end{frame}

% Slide 30
\begin{frame}{Tầm quan trọng của đặc trưng}
    \begin{itemize}
        \item \textbf{Feature Importances:} Rừng ngẫu nhiên rất tuyệt vời để đo lường tầm quan trọng tương đối của từng đặc trưng.
        \item Scikit-Learn đo lường bằng cách tính toán mức độ mà các nút cây sử dụng một đặc trưng để \textbf{làm giảm độ không tinh khiết} (impurity), tính trung bình trên toàn bộ rừng.
        \item Kết quả được chuẩn hóa để tổng độ quan trọng của tất cả các đặc trưng bằng 1.
        \item Thuộc tính chứa kết quả: \texttt{feature\_importances\_}.
    \end{itemize}
\end{frame}

% Slide 31
\begin{frame}[fragile]{Mã nguồn: Tính toán Feature Importances}
    \begin{lstlisting}[language=Python]
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)

for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
    print(round(score, 2), name)
    
# Output:
# 0.11 sepal length (cm)
# 0.02 sepal width (cm)
# 0.44 petal length (cm)  <-- Quan trong nhat
# 0.42 petal width (cm)   <-- Quan trong thu hai
    \end{lstlisting}
\end{frame}

% Slide 32
\begin{frame}{Tầm quan trọng pixel MNIST}
    \begin{center}
        \includegraphics[height=0.7\textheight]{../machineLearningWeb/Figures/CH07/Hinh_7-6.png}\\
        \vspace{0.3cm}
        \textit{Hình 7-6. Tầm quan trọng của pixel trên ảnh MNIST}
    \end{center}
\end{frame}

\section{Tăng cường (Boosting)}
% Slide 33
\begin{frame}{Tăng cường (Boosting)}
    \begin{itemize}
        \item \textbf{Boosting} (Tăng cường giả thuyết) là phương pháp tập hợp giúp biến các bộ học yếu thành bộ học mạnh.
        \item Ý tưởng cốt lõi: Huấn luyện các bộ dự đoán \textbf{theo trình tự}, mỗi bộ dự đoán sau sẽ cố gắng \textbf{sửa chữa lỗi} của bộ tiền nhiệm.
        \item Hai phương pháp phổ biến nhất:
            \begin{enumerate}
                \item AdaBoost (Adaptive Boosting)
                \item Gradient Boosting
            \end{enumerate}
    \end{itemize}
\end{frame}

% Slide 34
\begin{frame}{AdaBoost (Adaptive Boosting)}
    \begin{itemize}
        \item Cách bộ dự đoán mới sửa lỗi: \textbf{Tập trung nhiều hơn vào các trường hợp huấn luyện bị phân loại sai} bởi bộ tiền nhiệm.
        \item Quy trình:
            \begin{enumerate}
                \item Huấn luyện mô hình cơ sở (thường là Cây quyết định một nút - decision stump) và dự đoán.
                \item \textbf{Tăng trọng số} tương đối của các trường hợp bị sai.
                \item Huấn luyện mô hình thứ 2 dựa trên trọng số mới cập nhật, rồi lại dự đoán và tăng trọng số, v.v.
            \end{enumerate}
        \item Kỹ thuật này khá giống Gradient Descent, nhưng thay vì chỉnh tham số cho một mô hình, nó thêm mô hình mới vào tập hợp.
    \end{itemize}
\end{frame}

% Slide 35
\begin{frame}{Cập nhật trọng số trong AdaBoost}
    \begin{center}
        \includegraphics[height=0.7\textheight]{../machineLearningWeb/Figures/CH07/Hinh_7-7.png}\\
        \vspace{0.3cm}
        \textit{Hình 7-7. Huấn luyện AdaBoost tuần tự với cập nhật trọng số}
    \end{center}
\end{frame}

% Slide 36
\begin{frame}{Phân tích trọng số trong SVM}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Hình bên minh họa 5 bộ SVM (RBF kernel) liên tiếp trên dữ liệu moons.
                \item Biểu đồ trái: Learning rate lớn, bộ thứ 2 sửa rất mạnh các lỗi của bộ đầu.
                \item Biểu đồ phải: Learning rate bằng một nửa, các SVM sửa đổi một cách cẩn thận và dần dần hơn.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \includegraphics[width=\textwidth]{../machineLearningWeb/Figures/CH07/Hinh_7-8.png}
        \end{column}
    \end{columns}
\end{frame}

% Slide 37
\begin{frame}{Toán học AdaBoost: Tỷ lệ lỗi và Trọng số bộ dự đoán}
    \begin{itemize}
        \item Trọng số ban đầu của mỗi trường hợp $w^{(i)} = 1/m$.
        \item \textbf{Tỷ lệ lỗi có trọng số của bộ thứ $j$ ($r_j$):}
            \begin{equation}
                r_j = \frac{\sum_{i=1, \hat{y}_j^{(i)} \neq y^{(i)}}^m w^{(i)}}{\sum_{i=1}^m w^{(i)}}
            \end{equation}
        \item \textbf{Trọng số của bộ dự đoán thứ $j$ ($\alpha_j$):}
            \begin{equation}
                \alpha_j = \eta \log \frac{1 - r_j}{r_j}
            \end{equation}
            Trong đó $\eta$ là learning rate (mặc định 1). Bộ càng chính xác thì $\alpha_j$ càng cao. Nếu đoán bừa thì $\alpha_j \approx 0$.
    \end{itemize}
\end{frame}

% Slide 38
\begin{frame}{Toán học AdaBoost: Cập nhật trọng số trường hợp}
    \begin{itemize}
        \item \textbf{Quy tắc cập nhật trọng số trường hợp ($w^{(i)}$):}
            \begin{equation}
                w^{(i)} \leftarrow 
                \begin{cases} 
                w^{(i)} & \text{nếu } \hat{y}_j^{(i)} = y^{(i)} \\
                w^{(i)} \exp(\alpha_j) & \text{nếu } \hat{y}_j^{(i)} \neq y^{(i)}
                \end{cases}
            \end{equation}
        \item Sau đó, tất cả trọng số được chuẩn hóa (chia cho $\sum w^{(i)}$).
        \item \textbf{Dự đoán cuối cùng:}
            \begin{equation}
                \hat{y}(\mathbf{x}) = \underset{k}{\mathrm{argmax}} \sum_{j=1, \hat{y}_j(\mathbf{x}) = k}^N \alpha_j
            \end{equation}
    \end{itemize}
\end{frame}

% Slide 39
\begin{frame}[fragile]{Mã nguồn: AdaBoostClassifier}
    Scikit-Learn sử dụng biến thể đa lớp tên là SAMME (hoặc SAMME.R để tận dụng xác suất).
    \begin{lstlisting}[language=Python]
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), # Cay 1 nut (decision stump)
    n_estimators=30,
    learning_rate=0.5, 
    random_state=42
)
ada_clf.fit(X_train, y_train)
    \end{lstlisting}
\end{frame}

% Slide 40
\begin{frame}{Gradient Boosting}
    \begin{itemize}
        \item Tương tự AdaBoost: huấn luyện bộ sau sửa lỗi của bộ trước.
        \item Khác biệt: Thay vì thay đổi trọng số trường hợp, Gradient Boosting \textbf{khớp bộ dự đoán mới với các lỗi còn lại (residual errors)} do bộ dự đoán trước tạo ra.
        \item Gradient Tree Boosting (GBRT): Sử dụng Cây quyết định làm bộ cơ sở.
    \end{itemize}
\end{frame}

% Slide 41
\begin{frame}[fragile]{Mã nguồn: Gradient Tree Boosting thủ công (Phần 1)}
    Tạo dữ liệu và huấn luyện cây đầu tiên:
    \begin{lstlisting}[language=Python]
from sklearn.tree import DecisionTreeRegressor
import numpy as np

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
    \end{lstlisting}
\end{frame}

% Slide 42
\begin{frame}[fragile]{Mã nguồn: Gradient Tree Boosting thủ công (Phần 2)}
    Huấn luyện trên phần dư (residuals) và kết hợp:
    \begin{lstlisting}[language=Python]
y2 = y - tree_reg1.predict(X) # Phan du 1
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)

y3 = y2 - tree_reg2.predict(X) # Phan du 2
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)

# Du doan moi bang tong cac cay:
X_new = np.array([[-0.4], [0.], [0.5]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    \end{lstlisting}
\end{frame}

% Slide 43
\begin{frame}{Quá trình huấn luyện GBRT}
    \begin{center}
        \includegraphics[height=0.8\textheight]{../machineLearningWeb/Figures/CH07/Hinh_7-9.png}\\
        \vspace{0.1cm}
        \textit{Hình 7-9. Các dự đoán của từng cây (trái) và tổng hợp của tập hợp (phải)}
    \end{center}
\end{frame}

% Slide 44
\begin{frame}[fragile]{GradientBoostingRegressor trong Scikit-Learn}
    Sử dụng lớp có sẵn để tự động quá trình này:
    \begin{lstlisting}[language=Python]
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=3,
    learning_rate=1.0, 
    random_state=42
)
gbrt.fit(X, y)
    \end{lstlisting}
\end{frame}

% Slide 45
\begin{frame}{Vấn đề quá khớp và Shrinkage}
    \begin{itemize}
        \item \texttt{learning\_rate}: Kiểm soát mức độ đóng góp của từng cây.
        \item Nếu đặt giá trị thấp (ví dụ 0.05), sẽ cần nhiều cây hơn để khớp dữ liệu (Shrinkage - một phương pháp chính quy hóa).
        \item Nếu số lượng cây \texttt{n\_estimators} quá ít $\rightarrow$ Dưới khớp.
        \item Nếu số lượng cây \texttt{n\_estimators} quá nhiều $\rightarrow$ Quá khớp.
    \end{itemize}
\end{frame}

% Slide 46
\begin{frame}{Các tập hợp GBRT không đủ và vừa đủ cây}
    \begin{center}
        \includegraphics[width=0.9\textwidth]{../machineLearningWeb/Figures/CH07/Hinh_7-10.png}\\
        \vspace{0.3cm}
        \textit{Hình 7-10. Dưới khớp (trái) vs. Vừa vặn (phải)}
    \end{center}
\end{frame}

% Slide 47
\begin{frame}[fragile]{Dừng sớm (Early Stopping) trong GBRT}
    Tránh phải dùng GridSearchCV dò \texttt{n\_estimators}, ta có thể dùng Dừng sớm qua \texttt{n\_iter\_no\_change}:
    \begin{lstlisting}[language=Python]
gbrt_best = GradientBoostingRegressor(
    max_depth=2, 
    learning_rate=0.05, 
    n_estimators=500,
    n_iter_no_change=10, # Dung neu 10 lan lap lien tiep khong giam loi
    random_state=42
)
gbrt_best.fit(X, y)

print(gbrt_best.n_estimators_) # Chi can 92 cay (tu dung som) thay vi 500
    \end{lstlisting}
\end{frame}

% Slide 48
\begin{frame}{Tăng cường Gradient dựa trên Biểu đồ (HGB)}
    \begin{itemize}
        \item GBRT chậm trên dữ liệu lớn. Scikit-Learn cung cấp \textbf{Histogram-Based Gradient Boosting (HGB)} tối ưu cho dữ liệu hàng triệu dòng.
        \item \textbf{Cơ chế:} Nhóm các đặc trưng lại (binning) thành tối đa 255 nhóm (bins). Giảm đáng kể số ngưỡng phân chia.
        \item \textbf{Độ phức tạp:} Giảm từ $O(b \times m \times \log(m))$ xuống $O(b \times m)$, huấn luyện nhanh hơn \textbf{hàng trăm lần}.
        \item HGB xử lý luôn được cả Missing values và Categorical Features (không cần Imputer hay OneHotEncoder).
    \end{itemize}
\end{frame}

% Slide 49
\begin{frame}[fragile]{Mã nguồn: HistGradientBoostingRegressor}
    \begin{lstlisting}[language=Python]
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
                            remainder="passthrough"),
    HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
)
hgb_reg.fit(housing, housing_labels)
    \end{lstlisting}
\end{frame}

\section{Phân lớp xếp chồng (Stacking)}
% Slide 50
\begin{frame}{Phân lớp xếp chồng (Stacking)}
    \begin{itemize}
        \item \textbf{Stacking (Stacked Generalization):} Thay vì dùng hàm tầm thường như Bỏ phiếu đa số, tại sao không \textbf{huấn luyện một mô hình máy học khác} để tổng hợp các dự đoán?
        \item Bộ mô hình lớp cuối cùng được gọi là \textbf{bộ trộn (blender)} hay meta-learner.
        \item Lấy các dự đoán của các bộ học cơ sở làm đầu vào (input) và học cách tạo ra dự đoán chính xác nhất từ đó.
    \end{itemize}
\end{frame}

% Slide 51
\begin{frame}{Tổng hợp dự đoán bằng bộ trộn}
    \begin{center}
        \includegraphics[height=0.7\textheight]{../machineLearningWeb/Figures/CH07/Hinh_7-11.png}\\
        \vspace{0.3cm}
        \textit{Hình 7-11. Sử dụng bộ trộn để kết hợp 3 mô hình}
    \end{center}
\end{frame}

% Slide 52
\begin{frame}{Khái niệm bộ trộn (Blender)}
    \begin{itemize}
        \item Để huấn luyện bộ trộn, ta chia dữ liệu thành 2 phần:
        \item \textbf{Tập 1 (Tập huấn luyện):} Dùng để huấn luyện các mô hình cơ sở.
        \item \textbf{Tập 2 (Tập pha trộn - Blend/Hold-out):} Các mô hình cơ sở sẽ dự đoán trên tập này. Các dự đoán xuất ra sẽ đóng vai trò là \textbf{đặc trưng (features) mới}. Bộ trộn sẽ được huấn luyện trên tập đặc trưng mới này kết hợp với nhãn gốc.
        \item Scikit-Learn sử dụng \texttt{cross\_val\_predict} (K-Fold) để tránh làm lãng phí dữ liệu.
    \end{itemize}
\end{frame}

% Slide 53
\begin{frame}{Huấn luyện bộ trộn trong tập hợp}
    \begin{center}
        \includegraphics[height=0.7\textheight]{../machineLearningWeb/Figures/CH07/Hinh_7-12.png}\\
        \vspace{0.3cm}
        \textit{Hình 7-12. Tạo dữ liệu huấn luyện cho bộ trộn qua Out-of-Sample Predictions}
    \end{center}
\end{frame}

% Slide 54
\begin{frame}{Kiến trúc xếp chồng phức tạp}
    \begin{itemize}
        \item Không dừng lại ở 1 bộ trộn, ta có thể huấn luyện \textbf{nhiều bộ trộn khác nhau} (ví dụ: Linear, Random Forest) trên cùng một layer.
        \item Sau đó thêm 1 bộ trộn Tối cao lên trên cùng (Layer 3) để đưa ra dự đoán cuối.
        \item Càng nhiều lớp, mô hình càng tốn kém thời gian huấn luyện và độ phức tạp tính toán.
    \end{itemize}
\end{frame}

% Slide 55
\begin{frame}{Tập hợp xếp chồng đa lớp}
    \begin{center}
        \includegraphics[height=0.7\textheight]{../machineLearningWeb/Figures/CH07/Hinh_7-13.png}\\
        \vspace{0.3cm}
        \textit{Hình 7-13. Mạng Stacking đa tầng (Multi-layer Stacking Ensemble)}
    \end{center}
\end{frame}

% Slide 56
\begin{frame}[fragile]{Cài đặt StackingClassifier trong Scikit-Learn}
    \begin{lstlisting}[language=Python]
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5 # 5-Fold cross validation
)
stacking_clf.fit(X_train, y_train)
    \end{lstlisting}
\end{frame}

\section{Tổng kết Chương 7}
% Slide 57
\begin{frame}{Tổng kết Chương 7}
    \begin{itemize}
        \item \textbf{Học tổ hợp (Ensemble Learning)} là công cụ quyền lực đẩy hiệu năng mô hình đến giới hạn.
        \item \textbf{Voting \& Bagging/Pasting:} Đơn giản, dễ song song hóa, giảm phương sai cực tốt. Rừng ngẫu nhiên (Random Forest) là thuật toán tiêu biểu, hiệu năng ổn định.
        \item \textbf{Boosting (AdaBoost, GBRT, HGB):} Tập trung sửa sai, chuyển bộ học yếu thành mạnh. HGB là đột phá về tốc độ cho dữ liệu lớn.
        \item \textbf{Stacking:} Đẩy mạnh sức mạnh thông qua mô hình Meta-learner thay vì trung bình cứng nhắc.
        \item \textit{Rừng ngẫu nhiên và GBRT luôn là những mô hình đầu tiên nên thử nghiệm trong mọi dự án Machine Learning dạng bảng.}
    \end{itemize}
\end{frame}

% Slide 58
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
