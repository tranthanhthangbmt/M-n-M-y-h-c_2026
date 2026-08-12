# Kế hoạch Thiết kế và Xây dựng Bộ Slide Chương 9 - Học máy (Chuẩn Đại học)

**Thư mục mục tiêu:** `slideML/`  
**File định dạng mới:** LaTeX Beamer Widescreen 16:9 (`\documentclass[aspectratio=169]{beamer}`)  
**Theme & Color Theme:** `Madrid` theme, `default` colortheme  
**Quy mô dự kiến:** **~55 Frames (Slides)** (Nằm trong khoảng 40 - 60 slides theo yêu cầu).  
**Nguồn nội dung chữ:** Trích xuất và cô đọng trực tiếp từ nội dung tài liệu trang web của chương (`machineLearningWeb/docs/chuong_09.md`).  
**Tích hợp hình ảnh:** Phân bổ các hình ảnh (từ `Hinh_9-1` đến `Hinh_9-21` trong thư mục `Figures/CH09/`) vào các slide tương ứng. *(Lưu ý: Nếu chưa có file hình ảnh vật lý, script sẽ giả lập khối [Image Placeholder] kèm Caption để giữ cấu trúc bố cục).*

---

## 1. Bố cục Phân chương Tiết học (Sections & TOC)

Bộ slide Chương 9 được chia thành **4 Tiết học (4 Sections)** rõ ràng với tổng cộng 55 Frames:

### Tiết 1 (15 Frames): Giới thiệu Học không giám sát & Thuật toán K-Means
- **Mục tiêu:** Hiểu khái niệm học không giám sát, đặc biệt là phân cụm (Clustering). Nắm vững thuật toán K-Means, khái niệm quán tính (inertia) và phương pháp khuỷu tay.
- **Nội dung các slide thực tế:**
  1. Trang bìa (`\titlepage`)
  2. Mục lục nội dung chương (`\tableofcontents`)
  3. Giới thiệu Học không giám sát
  4. Phân cụm (Clustering) là gì?
  5. Hình ảnh: Phân loại so với phân cụm -> Hình 9-1
  6. Ứng dụng của Phân cụm (Phân khúc khách hàng, Phân tích dữ liệu, v.v.)
  7. Thuật toán K-Means
  8. Hình ảnh: Tập dữ liệu chưa gán nhãn gồm 5 khối -> Hình 9-2
  9. Hình ảnh: Phép phân vùng Voronoi của K-Means -> Hình 9-3
  10. Cách thuật toán K-Means hoạt động -> Hình 9-4
  11. Vấn đề khởi tạo tâm cụm ngẫu nhiên -> Hình 9-5
  12. Quán tính (Inertia) là gì?
  13. Hình ảnh: Quán tính giảm theo các vòng lặp -> Hình 9-6
  14. Hậu quả của việc chọn số lượng cụm (k) sai -> Hình 9-7
  15. Lựa chọn số cụm bằng phương pháp khuỷu tay (Elbow) -> Hình 9-8

### Tiết 2 (15 Frames): Đánh giá Silhouette, Ứng dụng K-Means & DBSCAN
- **Mục tiêu:** Biết cách đánh giá chất lượng cụm bằng điểm Silhouette. Hiểu các ứng dụng thực tế của K-Means (như phân đoạn hình ảnh, học bán giám sát) và tìm hiểu thuật toán phân cụm dựa trên mật độ DBSCAN.
- **Nội dung các slide thực tế:**
  16. Đánh giá chất lượng bằng điểm Silhouette
  17. Hình ảnh: Biểu đồ điểm Silhouette theo k -> Hình 9-9
  18. Hình ảnh: Phân tích biểu đồ dao (Silhouette plots) -> Hình 9-10
  19. Giới hạn của K-Means
  20. Hình ảnh: K-Means gặp khó khăn với các cụm hình elip/kích thước khác nhau -> Hình 9-11
  21. Ứng dụng: Phân đoạn hình ảnh (Image Segmentation) bằng K-Means
  22. Hình ảnh: Phân đoạn màu sắc hình ảnh bọ rùa -> Hình 9-12
  23. Ứng dụng: Tiền xử lý để học bán giám sát (Semi-supervised Learning)
  24. Hình ảnh: Lấy 50 hình ảnh đại diện (MNIST) bằng K-Means -> Hình 9-13
  25. Thuật toán phân cụm DBSCAN
  26. Cơ chế hoạt động của DBSCAN (Core instances, epsilon, min_samples)
  27. Hình ảnh: DBSCAN với 2 bán kính vùng lân cận khác nhau -> Hình 9-14
  28. Ưu điểm và nhược điểm của DBSCAN
  29. Dự đoán với DBSCAN (Kết hợp KNN)
  30. Hình ảnh: Đường biên quyết định của bộ phân loại KNN trên tập DBSCAN -> Hình 9-15

### Tiết 3 (15 Frames): Mô hình Hỗn hợp Gaussian (GMM)
- **Mục tiêu:** Hiểu Mô hình Hỗn hợp Gaussian (Gaussian Mixture Model) để giải quyết các trường hợp K-Means thất bại. Ứng dụng GMM vào phân cụm, ước tính mật độ và phát hiện dị thường.
- **Nội dung các slide thực tế:**
  31. Mô hình Hỗn hợp Gaussian (GMM) là gì?
  32. Thuật toán Cực đại hóa Kỳ vọng (Expectation-Maximization - EM)
  33. Khởi tạo GMM trong Scikit-Learn
  34. Hình ảnh: Các giá trị trung bình và đường biên quyết định của GMM -> Hình 9-16
  35. Hạn chế dạng ma trận hiệp phương sai (Covariance Types)
  36. Hình ảnh: GMM với các cụm liên kết (tied) và hình cầu (spherical) -> Hình 9-17
  37. Ước tính mật độ sinh ngẫu nhiên (Generative process)
  38. Phát hiện dị thường (Anomaly Detection) bằng GMM
  39. Cách tính ngưỡng mật độ phát hiện dị thường
  40. Hình ảnh: Phát hiện dị thường sử dụng GMM -> Hình 9-18
  41. Hình ảnh: Hàm tham số của mô hình -> Hình 9-19
  42. Lựa chọn số lượng cụm trong GMM
  43. Các tiêu chí thông tin lý thuyết: BIC và AIC
  44. Công thức tính toán AIC và BIC
  45. Hình ảnh: Biểu đồ AIC và BIC cho các k khác nhau -> Hình 9-20

### Tiết 4 (10 Frames): Mô hình Hỗn hợp Gaussian Bayes (BGM)
- **Mục tiêu:** Nâng cấp GMM với suy luận Bayes để mô hình tự động vô hiệu hóa các cụm không cần thiết, tự tìm ra số cụm tối ưu.
- **Nội dung các slide thực tế:**
  46. Mô hình Hỗn hợp Gaussian Bayes (Bayesian Gaussian Mixture)
  47. Cách BGM tự động loại bỏ các cụm dư thừa
  48. Tiên nghiệm Dirichlet (Dirichlet Prior) trên trọng số cụm
  49. Hình ảnh: Khớp BGM tự động với tập moons (không phải elip) -> Hình 9-21
  50. Lưu ý khi dùng BGM với dữ liệu hình dạng tùy ý
  51. Các thuật toán phát hiện dị thường khác (PCA, Fast-MCD, Isolation Forest, LOF)
  52. Tóm tắt Chương 9: K-Means & Ứng dụng
  53. Tóm tắt Chương 9: DBSCAN & GMM
  54. Kết luận về sức mạnh của Học không giám sát
  55. Hỏi & Đáp (Q&A)

---

## 2. Tiêu chuẩn Kỹ thuật và Nhận diện (Format)
- Bố cục Slide chia làm **2 cột** khi cần thiết để chèn hình ảnh minh họa không bị che mất chữ. Do Chương 9 có tới 21 hình ảnh, việc chia cột 50-50 (`\begin{columns}`) sẽ được sử dụng thường xuyên.
- Chèn code Python vào Block `lstlisting` với highlight màu sắc dễ nhìn (ví dụ: code K-Means, DBSCAN, GaussianMixture).
- Các hình ảnh sẽ được scale bằng `[width=0.9\textwidth]` hoặc `[height=0.65\textheight]` để vừa khung Widescreen 16:9.
- Dùng `\usepackage{fontspec}` và `\setmainfont{Times New Roman}`, `\setsansfont{Arial}` để tránh lỗi mất chữ tiếng Việt khi dùng `xelatex`.
- **Thông tin tác giả:** `\author{Giảng viên: TS. Trần Thành Thắng}`.
- **Biên dịch:** Chạy `xelatex` 2 lần liên tiếp.

## 3. Nhật ký thực hiện (Hoàn thành)
- [x] Soạn kịch bản Python tạo slide (`v11_generate_slides_ch09.py`).
- [x] Trích xuất nội dung từ `chuong_09.md`.
- [x] Chèn các đoạn code Python và công thức toán học.
- [x] Cấu hình đúng `fontspec` để không bị lỗi phông Mục lục.
- [x] Chạy lệnh `xelatex` 2 lần và kiểm tra PDF.

*Ghi chú: Đã biên dịch thành công `Slide_ML_Chap09.pdf` với tổng số 52 slides. Quá trình biên dịch đã áp dụng cơ chế Placeholder cho 21 hình ảnh chưa có sẵn. Mã nguồn Python được lưu ở `v11_generate_slides_ch09.py`.*
