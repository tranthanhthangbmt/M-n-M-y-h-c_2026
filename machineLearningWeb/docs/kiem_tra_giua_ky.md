# 📝 BÀI KIỂM TRA GIỮA KỲ - MÔN MÁY HỌC

> **Học phần:** Máy học (Machine Learning) — 3 Tín chỉ  
> **Đơn vị:** Khoa Công nghệ Thông tin & AI — Đại học Đông Á  
> **Giảng viên:** TS. Trần Thành Thắng  

---

### 📋 Thông Tin & Quy Chế Làm Bài

| Mục | Chi tiết |
| :--- | :--- |
| **Hình thức thi 1 (Trắc nghiệm)** | Trắc nghiệm trực tuyến 90 câu (45 phút, bốc ngẫu nhiên cân bằng 12 chương, làm 1 lần duy nhất) |
| **Hình thức thi 2 (Thực hành)** | Làm bài trên **Google Colab** (60–75 phút, chọn 1 trong 3 đề thi phân bổ theo ca) |
| **Quy định nộp bài thực hành** | Chạy toàn bộ lệnh (Runtime > Run all) ➔ **Kết xuất file PDF** ➔ **Nộp bài lên Canvas LMS** |
| **Mật khẩu làm bài** | Do Giảng viên cung cấp trực tiếp tại phòng thi |
| **Giám sát & An ninh** | Khóa Copy/Paste, khóa chuột phải, F12; Bắt buộc bật JavaScript; Tự động ghi nhận số lần chuyển tab |
| **Lưu trữ dữ liệu trắc nghiệm** | Tự động đồng bộ và lưu trữ an toàn vào cơ sở dữ liệu của Giảng viên |

<div style="margin: 16px 0 24px 0; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
  <a href="quizzes/MidtermExam/index.html" target="_blank" class="btn-fullscreen" style="display: inline-flex; align-items: center; gap: 8px; background: #1a73e8; color: white !important; padding: 10px 18px; border-radius: 8px; font-weight: 600; text-decoration: none; box-shadow: 0 4px 12px rgba(26, 115, 232, 0.3);">
    🚀 Mở Bài Thi Trắc Nghiệm Toàn Màn Hình
  </a>
  <a href="#/docs/kiem_tra_giua_ky?id=dethi-colab-canvas" style="display: inline-flex; align-items: center; gap: 8px; background: #7c3aed; color: white !important; padding: 10px 18px; border-radius: 8px; font-weight: 600; text-decoration: none; box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);">
    📑 Xem 3 Đề Thi Thực Hành (Colab & Canvas) ↓
  </a>
</div>

---

### 🖥️ Giao Diện Bài Thi Trực Tuyến

<iframe src="quizzes/MidtermExam/index.html" style="width: 100%; height: 92vh; min-height: 880px; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 6px 24px rgba(0,0,0,0.08); background: #ffffff;" allowfullscreen></iframe>

---

<span id="dethi-colab-canvas"></span>

### 📑 Đề Thi Giữa Kỳ (3 Đề Colab & Nộp Canvas)

> 📢 **DÀNH CHO SINH VIÊN:** Dưới đây là **3 Đề thi giữa kỳ môn Máy học** (thực hành trên Google Colab). Sinh viên xem ca thi / đề thi được Giảng viên phân công, nhấp vào tab đề thi tương ứng để mở đề trên Google Colab. Sau khi hoàn thành và chạy toàn bộ mã lệnh, sinh viên **kết xuất ra file PDF** và **nộp bài lên hệ thống Canvas**.

<!-- tabs:start -->

#### ** 📌 Hướng Dẫn Làm Bài & Nộp Canvas **

<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
  <h4 style="margin-top:0; color:#1e293b;">🎯 QUY TRÌNH 5 BƯỚC HOÀN THÀNH VÀ NỘP BÀI THI</h4>
  
  <ol style="line-height: 1.8; color: #334155;">
    <li>
      <strong>Bước 1: Mở đề thi và Tạo bản sao cá nhân</strong><br>
      - Nhấp nút <code>🚀 Mở Trên Google Colab</code> ở tab đề thi tương ứng (Đề 01, 02 hoặc 03).<br>
      - <span style="color:#dc2626; font-weight:bold;">BẮT BUỘC:</span> Vào menu <code>Tệp (File)</code> &rarr; Chọn <code>Lưu một bản sao trong Drive (Save a copy in Drive)</code> để có quyền viết mã nguồn và lưu trữ vào Google Drive cá nhân của bạn.
    </li>
    <li>
      <strong>Bước 2: Điền thông tin thí sinh</strong><br>
      - Điền chính xác <strong>Họ và tên</strong> và <strong>Mã số sinh viên (MSSV)</strong> vào ô thông tin đầu Notebook.
    </li>
    <li>
      <strong>Bước 3: Thực hiện các câu hỏi lập trình</strong><br>
      - Đọc kỹ yêu cầu từng câu từ Câu 1 đến Câu 4.<br>
      - Điền code vào các vị trí có chú thích <code># TODO: [Sinh viên hoàn thành]</code>.<br>
      - Chạy từng ô lệnh để kiểm tra kết quả (bảng dữ liệu, độ đo RMSE/Accuracy, biểu đồ trực quan).
    </li>
    <li>
      <strong>Bước 4: Chạy kiểm thử toàn bộ (Run All)</strong><br>
      - Trước khi nộp, vào menu <code>Thời gian chạy (Runtime)</code> &rarr; Chọn <code>Khởi động lại và chạy tất cả các ô (Restart and run all)</code>.<br>
      - Đảm bảo 100% các ô lệnh đều thực thi không báo lỗi và hiển thị đầy đủ kết quả đầu ra (Outputs).
    </li>
    <li>
      <strong>Bước 5: Kết xuất file PDF và Nộp lên Canvas</strong><br>
      - <strong>Cách xuất file PDF chuẩn đẹp:</strong> Bấm tổ hợp phím <code>Ctrl + P</code> (hoặc vào menu trình duyệt &rarr; <code>In / Print</code>) &rarr; Mục <em>Máy in đích (Destination)</em> chọn <strong>Lưu dưới dạng PDF (Save as PDF)</strong> &rarr; Bật tùy chọn <em>Đồ họa nền (Background graphics)</em> &rarr; Bấm <strong>Lưu (Save)</strong>.<br>
      - <strong>Quy cách đặt tên file:</strong> <code>[MSSV]_[HoVaTen]_De01.pdf</code> (Ví dụ: <code>1042301_NguyenVanAn_De01.pdf</code>).<br>
      - <strong>Nộp bài lên Canvas:</strong> Truy cập cổng Canvas <a href="https://canvas.donga.edu.vn/" target="_blank">canvas.donga.edu.vn</a> &rarr; Lớp học phần <strong>Máy học</strong> &rarr; Mục <strong>Assignments (Bài tập)</strong> &rarr; <strong>[Kiểm Tra Giữa Kỳ - Thực Hành Máy Học]</strong> &rarr; Nhấp <strong>Submit Assignment</strong> và tải file PDF lên.
    </li>
  </ol>

  <div style="margin-top: 16px; padding: 12px 16px; background: #eff6ff; border-left: 4px solid #1a73e8; border-radius: 4px;">
    💡 <strong>Mẹo nhỏ:</strong> Sinh viên nên tải thêm file mã nguồn gốc <code>.ipynb</code> về máy tính qua menu <code>Tệp &rarr; Tải xuống &rarr; Tải xuống .ipynb</code> để lưu trữ dự phòng đối chiếu khi Giảng viên yêu cầu.
  </div>
</div>

<div style="display: flex; gap: 12px; flex-wrap: wrap;">
  <a href="docs/exams/Bo_3_De_Thi_Giua_Ky_May_Hoc.zip" download style="display: inline-flex; align-items: center; gap: 8px; background: #0284c7; color: white !important; padding: 10px 18px; border-radius: 8px; font-weight: 600; text-decoration: none;">
    📦 Tải Trọn Bộ 3 Đề Thi (.ZIP)
  </a>
  <a href="https://canvas.donga.edu.vn/" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; background: #e11d48; color: white !important; padding: 10px 18px; border-radius: 8px; font-weight: 600; text-decoration: none;">
    🎓 Đăng Nhập Cổng Canvas UDA Để Nộp Bài
  </a>
</div>

#### ** 🔹 Đề Thi Số 01 (California Housing & MLP) **

<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 22px; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">
  <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
    <div>
      <h3 style="margin: 0; color: #1a73e8;">ĐỀ THI SỐ 01: BÀI TOÁN HỒI QUY GIÁ NHÀ & MLP</h3>
      <div style="font-size: 0.9rem; color: #64748b; margin-top: 4px;">Phạm vi: Chương 2 (Quy trình ML), Chương 4 (Hồi quy), Chương 6 (Cây quyết định), Chương 10 (Keras MLP)</div>
    </div>
    <div style="display: flex; gap: 8px;">
      <a href="https://colab.research.google.com/drive/179f0UQ5e2jxV6tH31bqwFjO-d2zfx1ez?usp=sharing" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: #f59e0b; color: #0f172a !important; padding: 8px 14px; border-radius: 6px; font-weight: 700; text-decoration: none; font-size: 0.88rem;">
        🚀 Mở Trên Google Colab
      </a>
      <a href="docs/exams/DE_THI_SO_01.ipynb" download style="display: inline-flex; align-items: center; gap: 6px; background: #f1f5f9; color: #334155 !important; border: 1px solid #cbd5e1; padding: 8px 14px; border-radius: 6px; font-weight: 600; text-decoration: none; font-size: 0.88rem;">
        💾 Tải file .ipynb
      </a>
    </div>
  </div>

  <table style="width:100%; border-collapse: collapse; margin-bottom: 16px; font-size: 0.9rem;">
    <tr>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600; width: 160px;">⏱️ Thời gian làm bài:</td>
      <td style="padding: 6px 12px;">60 – 75 phút</td>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600; width: 140px;">🎯 Thang điểm:</td>
      <td style="padding: 6px 12px;">10.0 điểm</td>
    </tr>
    <tr>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600;">💻 Bộ dữ liệu:</td>
      <td style="padding: 6px 12px;">California Housing (20.640 mẫu, 8 đặc trưng)</td>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600;">⚙️ Môi trường:</td>
      <td style="padding: 6px 12px;">Google Colab (CPU runtime)</td>
    </tr>
  </table>

  <h4>NỘI DUNG YÊU CẦU ĐỀ BÀI:</h4>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #10b981;">
    <strong>Câu 1 (Mức độ Dễ - 2.5 điểm): Khám phá & Phân chia dữ liệu (Chương 2)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Hiển thị 5 dòng đầu tiên của tập dữ liệu <code>X</code> (dùng <code>head()</code>).</li>
      <li>Sử dụng hàm <code>train_test_split</code> để phân chia dữ liệu thành tập huấn luyện và tập kiểm thử (<code>X_train, X_test, y_train, y_test</code>) với tỉ lệ <code>test_size=0.2</code> và <code>random_state=42</code>.</li>
      <li>In ra số lượng mẫu dữ liệu của <code>X_train</code> và <code>X_test</code>.</li>
    </ol>
  </div>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #10b981;">
    <strong>Câu 2 (Mức độ Dễ - 2.5 điểm): Chuẩn hóa dữ liệu & Hồi quy tuyến tính (Chương 2, 4)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Dùng <code>StandardScaler</code> để chuẩn hóa đặc trưng: fit trên <code>X_train</code> và transform trên cả <code>X_train</code> và <code>X_test</code>.</li>
      <li>Khởi tạo và huấn luyện mô hình <code>LinearRegression</code> trên tập huấn luyện đã chuẩn hóa.</li>
      <li>Dự đoán 5 mẫu đầu tiên của tập kiểm thử và in kết quả so sánh đối chiếu với giá trị thực tế <code>y_test[:5]</code>.</li>
    </ol>
  </div>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #3b82f6;">
    <strong>Câu 3 (Mức độ Trung bình - 3.0 điểm): Đánh giá Sai số RMSE & Cây quyết định Hồi quy (Chương 2, 6)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Viết hàm tính sai số bình phương trung bình căn bậc hai (<code>RMSE</code>) giữa nhãn thực tế và nhãn dự đoán.</li>
      <li>Tính và in điểm số RMSE của mô hình <code>LinearRegression</code> trên toàn bộ tập kiểm tra <code>X_test_scaled</code>.</li>
      <li>Khởi tạo mô hình cây quyết định <code>DecisionTreeRegressor(max_depth=5, random_state=42)</code>, huấn luyện và tính RMSE trên tập kiểm tra. Nhận xét mô hình nào cho sai số thấp hơn.</li>
    </ol>
  </div>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #8b5cf6;">
    <strong>Câu 4 (Mức độ Khá/Giỏi - 2.0 điểm): Xây dựng Mạng Nơ-ron MLP Hồi quy với Keras (Chương 10)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Sử dụng <code>tf.keras.Sequential</code> xây dựng mạng MLP gồm:
        <ul>
          <li>1 lớp ẩn <code>Dense</code> có 30 nơ-ron, hàm kích hoạt <code>relu</code>, <code>input_shape=[X_train.shape[1]]</code>.</li>
          <li>1 lớp đầu ra <code>Dense</code> có 1 nơ-ron (không dùng hàm kích hoạt vì là bài toán hồi quy liên tục).</li>
        </ul>
      </li>
      <li>Biên dịch mô hình với hàm mất mát <code>loss="mse"</code> và bộ tối ưu hóa <code>optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)</code>.</li>
      <li>Huấn luyện trong 10 epochs với <code>batch_size=32</code> và tính sai số RMSE của mô hình trên tập kiểm tra.</li>
    </ol>
  </div>
</div>

#### ** 🔹 Đề Thi Số 02 (Hoa Iris - PCA - K-Means) **

<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 22px; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">
  <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
    <div>
      <h3 style="margin: 0; color: #1a73e8;">ĐỀ THI SỐ 02: PHÂN LOẠI HOA IRIS, GIẢM CHIỀU & PHÂN CỤM</h3>
      <div style="font-size: 0.9rem; color: #64748b; margin-top: 4px;">Phạm vi: Chương 3 (Độ đo phân loại), Chương 5 (SVM), Chương 7 (Học tổ hợp), Chương 8 (PCA), Chương 9 (K-Means)</div>
    </div>
    <div style="display: flex; gap: 8px;">
      <a href="https://colab.research.google.com/drive/1KKslPSAtsTSjjiE9U4DrIi5WFP9XeZSh?usp=sharing" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: #f59e0b; color: #0f172a !important; padding: 8px 14px; border-radius: 6px; font-weight: 700; text-decoration: none; font-size: 0.88rem;">
        🚀 Mở Trên Google Colab
      </a>
      <a href="docs/exams/DE_THI_SO_02.ipynb" download style="display: inline-flex; align-items: center; gap: 6px; background: #f1f5f9; color: #334155 !important; border: 1px solid #cbd5e1; padding: 8px 14px; border-radius: 6px; font-weight: 600; text-decoration: none; font-size: 0.88rem;">
        💾 Tải file .ipynb
      </a>
    </div>
  </div>

  <table style="width:100%; border-collapse: collapse; margin-bottom: 16px; font-size: 0.9rem;">
    <tr>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600; width: 160px;">⏱️ Thời gian làm bài:</td>
      <td style="padding: 6px 12px;">60 – 75 phút</td>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600; width: 140px;">🎯 Thang điểm:</td>
      <td style="padding: 6px 12px;">10.0 điểm</td>
    </tr>
    <tr>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600;">💻 Bộ dữ liệu:</td>
      <td style="padding: 6px 12px;">Iris Dataset (150 mẫu hoa, 4 đặc trưng, 3 loài hoa)</td>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600;">⚙️ Môi trường:</td>
      <td style="padding: 6px 12px;">Google Colab (CPU runtime)</td>
    </tr>
  </table>

  <h4>NỘI DUNG YÊU CẦU ĐỀ BÀI:</h4>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #10b981;">
    <strong>Câu 1 (Mức độ Dễ - 2.5 điểm): Chuẩn bị dữ liệu & Huấn luyện phân loại tuyến tính (Chương 3, 5)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Phân chia tập dữ liệu <code>X, y</code> thành tập train và test theo tỉ lệ 80:20 (<code>test_size=0.2</code>, <code>random_state=42</code>).</li>
      <li>Chuẩn hóa dữ liệu bằng <code>StandardScaler</code>.</li>
      <li>Khởi tạo mô hình <code>LinearSVC(C=1.0, random_state=42)</code> và huấn luyện trên tập dữ liệu đã chuẩn hóa.</li>
    </ol>
  </div>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #10b981;">
    <strong>Câu 2 (Mức độ Dễ - 2.5 điểm): Ma trận nhầm lẫn & Báo cáo phân loại (Chương 3)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Dự đoán nhãn lớp cho toàn bộ tập kiểm tra <code>X_test_scaled</code>.</li>
      <li>Tính và in ra ma trận nhầm lẫn (<code>confusion_matrix</code>).</li>
      <li>Sử dụng hàm <code>classification_report</code> để hiển thị các chỉ số Precision, Recall và F1-score của 3 lớp hoa.</li>
    </ol>
  </div>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #3b82f6;">
    <strong>Câu 3 (Mức độ Trung bình - 3.0 điểm): Giảm chiều dữ liệu PCA & Phân cụm K-Means (Chương 8, 9)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Sử dụng kỹ thuật <code>PCA</code> để giảm 4 đặc trưng của hoa Iris xuống còn <strong>2 chiều</strong> (<code>n_components=2</code>).</li>
      <li>In ra tỉ lệ phương sai được giải thích (<code>explained_variance_ratio_</code>) của 2 thành phần chính.</li>
      <li>Áp dụng thuật toán <code>KMeans(n_clusters=3, random_state=42, n_init=10)</code> trên không gian 2 chiều này để tìm 3 tâm cụm.</li>
      <li>Dùng <code>plt.scatter</code> trực quan hóa dữ liệu 2D với màu sắc tương ứng theo nhãn cụm mà K-Means vừa dự đoán.</li>
    </ol>
  </div>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #8b5cf6;">
    <strong>Câu 4 (Mức độ Khá/Giỏi - 2.0 điểm): Rừng ngẫu nhiên & Tầm quan trọng của đặc trưng (Chương 7)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Khởi tạo mô hình <code>RandomForestClassifier(n_estimators=100, random_state=42)</code> và huấn luyện trên <code>X_train, y_train</code> (dữ liệu 4 đặc trưng gốc).</li>
      <li>Trích xuất thuộc tính <code>feature_importances_</code> của mô hình đã huấn luyện.</li>
      <li>In ra độ quan trọng của từng đặc trưng theo định dạng: <code>Tên đặc trưng: Tỉ lệ %</code>. Chỉ ra đâu là đặc trưng quan trọng nhất để nhận dạng loài hoa.</li>
    </ol>
  </div>
</div>

#### ** 🔹 Đề Thi Số 03 (Fashion MNIST - MLP & CNN) **

<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 22px; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">
  <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
    <div>
      <h3 style="margin: 0; color: #1a73e8;">ĐỀ THI SỐ 03: PHÂN LOẠI ẢNH FASHION MNIST VỚI MLP & CNN</h3>
      <div style="font-size: 0.9rem; color: #64748b; margin-top: 4px;">Phạm vi: Chương 3 (Phân loại), Chương 10 (Keras MLP), Chương 11 (Dropout chống Overfitting), Chương 14 (CNN)</div>
    </div>
    <div style="display: flex; gap: 8px;">
      <a href="https://colab.research.google.com/drive/1M8fQgx71l7Jxxpan548oWuwHcx1UVvve?usp=sharing" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: #f59e0b; color: #0f172a !important; padding: 8px 14px; border-radius: 6px; font-weight: 700; text-decoration: none; font-size: 0.88rem;">
        🚀 Mở Trên Google Colab
      </a>
      <a href="docs/exams/DE_THI_SO_03.ipynb" download style="display: inline-flex; align-items: center; gap: 6px; background: #f1f5f9; color: #334155 !important; border: 1px solid #cbd5e1; padding: 8px 14px; border-radius: 6px; font-weight: 600; text-decoration: none; font-size: 0.88rem;">
        💾 Tải file .ipynb
      </a>
    </div>
  </div>

  <table style="width:100%; border-collapse: collapse; margin-bottom: 16px; font-size: 0.9rem;">
    <tr>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600; width: 160px;">⏱️ Thời gian làm bài:</td>
      <td style="padding: 6px 12px;">60 – 75 phút</td>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600; width: 140px;">🎯 Thang điểm:</td>
      <td style="padding: 6px 12px;">10.0 điểm</td>
    </tr>
    <tr>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600;">💻 Bộ dữ liệu:</td>
      <td style="padding: 6px 12px;">Fashion MNIST (70.000 ảnh kích thước 28x28, 10 lớp thời trang)</td>
      <td style="padding: 6px 12px; background: #f8fafc; font-weight: 600;">⚙️ Môi trường:</td>
      <td style="padding: 6px 12px;">Google Colab (CPU hoặc GPU runtime)</td>
    </tr>
  </table>

  <h4>NỘI DUNG YÊU CẦU ĐỀ BÀI:</h4>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #10b981;">
    <strong>Câu 1 (Mức độ Dễ - 2.5 điểm): Tiền xử lý & Trực quan hóa dữ liệu ảnh (Chương 10)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Tách 5.000 mẫu cuối của tập huấn luyện để làm tập xác thực (<code>X_valid, y_valid</code>), các mẫu còn lại làm tập huấn luyện chính (<code>X_train, y_train</code>).</li>
      <li>Chuẩn hóa giá trị cường độ điểm ảnh từ khoảng <code>[0, 255]</code> về khoảng số thực <code>[0.0, 1.0]</code> bằng cách chia cho <code>255.0</code> cho cả 3 tập <code>X_train, X_valid, X_test</code>.</li>
      <li>Dùng <code>plt.imshow</code> vẽ hình ảnh của mẫu thứ hai (<code>X_train[1]</code>) với bản đồ màu <code>cmap="binary"</code> kèm tiêu đề là tên lớp tương ứng từ <code>class_names</code>.</li>
    </ol>
  </div>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #10b981;">
    <strong>Câu 2 (Mức độ Dễ - 2.5 điểm): Xây dựng Mạng Nơ-ron MLP cơ bản (Chương 10)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Sử dụng <code>tf.keras.Sequential</code> xây dựng mạng nơ-ron truyền thẳng MLP:
        <ul>
          <li>1 lớp <code>Flatten</code> với <code>input_shape=[28, 28]</code> làm phẳng ma trận ảnh 2D thành vector 1D.</li>
          <li>1 lớp ẩn <code>Dense</code> có 128 nơ-ron, hàm kích hoạt <code>relu</code>.</li>
          <li>1 lớp đầu ra <code>Dense</code> có 10 nơ-ron, hàm kích hoạt <code>softmax</code>.</li>
        </ul>
      </li>
      <li>Biên dịch mô hình với <code>loss="sparse_categorical_crossentropy"</code>, <code>optimizer="sgd"</code>, và <code>metrics=["accuracy"]</code>.</li>
      <li>In bảng tóm tắt kiến trúc mạng bằng lệnh <code>model.summary()</code>.</li>
    </ol>
  </div>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #3b82f6;">
    <strong>Câu 3 (Mức độ Trung bình - 3.0 điểm): Huấn luyện & Tối ưu hóa mô hình với Dropout (Chương 10, 11)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Huấn luyện mô hình <code>model_mlp</code> trong <strong>5 epochs</strong>, truyền tập dữ liệu xác thực <code>validation_data=(X_valid, y_valid)</code>.</li>
      <li>Đánh giá độ chính xác (Accuracy) của <code>model_mlp</code> trên tập kiểm tra <code>X_test, y_test</code>.</li>
      <li>Để giảm nguy cơ quá khớp (Overfitting), xây dựng mô hình mới bổ sung thêm 1 lớp <code>Dropout(rate=0.2)</code> ngay sau lớp <code>Dense(128, activation="relu")</code>.</li>
    </ol>
  </div>

  <div style="margin-bottom: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #8b5cf6;">
    <strong>Câu 4 (Mức độ Khá/Giỏi - 2.0 điểm): Xây dựng Mạng Nơ-ron Tích chập CNN Đơn giản (Chương 14)</strong>
    <ol style="margin-top: 6px; padding-left: 20px;">
      <li>Định hình lại dữ liệu ảnh đầu vào để có chiều kênh (channel): thêm 1 chiều kích thước cho ảnh đơn sắc (thành <code>[28, 28, 1]</code>) bằng <code>np.expand_dims</code>.</li>
      <li>Xây dựng mạng tích chập cơ bản gồm:
        <ul>
          <li>Lớp <code>Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[28, 28, 1])</code></li>
          <li>Lớp <code>MaxPool2D(pool_size=2)</code></li>
          <li>Lớp <code>Flatten()</code></li>
          <li>Lớp <code>Dense(64, activation="relu")</code></li>
          <li>Lớp đầu ra <code>Dense(10, activation="softmax")</code></li>
        </ul>
      </li>
      <li>Biên dịch mô hình với bộ tối ưu hóa <code>optimizer="adam"</code> và in bảng tóm tắt <code>model.summary()</code>.</li>
    </ol>
  </div>
</div>

<!-- tabs:end -->

---

*Lưu ý: Sinh viên cần tuân thủ nghiêm túc quy chế kiểm tra. Mã nguồn nộp lên Canvas sẽ được hệ thống kiểm tra đối chiếu tính trùng lặp tự động.*
