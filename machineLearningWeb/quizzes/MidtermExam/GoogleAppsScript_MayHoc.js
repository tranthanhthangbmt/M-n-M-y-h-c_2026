/**
 * ====================================================================
 * MÃ GOOGLE APPS SCRIPT: TỰ ĐỘNG LƯU ĐIỂM BÀI KIỂM TRA MÔN MÁY HỌC
 * Học phần: Máy học (Machine Learning)
 * Trường Đại học Đông Á
 * ====================================================================
 * 
 * HƯỚNG DẪN CÀI ĐẶT NHANH (MẤT 2 PHÚT):
 * --------------------------------------------------------------------
 * 1. Mở một trang tính Google Sheet mới tại https://sheets.google.com
 * 2. Đặt tên trang tính, ví dụ: "KẾT QUẢ KIỂM TRA GIỮA KỲ MÁY HỌC - 2026".
 * 3. Trên thanh Menu của Google Sheet, chọn:
 *    -> Tiện ích mở rộng (Extensions) -> Apps Script.
 * 4. Xóa toàn bộ mã mặc định (myFunction), dán toàn bộ đoạn mã này vào.
 * 5. Bấm nút "Lưu" (biểu tượng đĩa mềm 💾 hoặc Ctrl + S).
 * 6. Bấm vào nút "Triển khai" (Deploy) màu xanh ở góc trên bên phải:
 *    -> Chọn "Tùy chọn triển khai mới" (New deployment).
 *    -> Bấm vào biểu tượng Bánh răng (⚙️) bên cạnh "Chọn loại", chọn: "Ứng dụng web" (Web App).
 *    -> Mô tả: "Lưu điểm thi trắc nghiệm giữa kỳ môn Máy học".
 *    -> Thực thi dưới dạng (Execute as): "Tôi" (Me).
 *    -> Ai có quyền truy cập (Who has access): "Bất kỳ ai" (Anyone)  RẤT QUAN TRỌNG!
 * 7. Bấm nút "Triển khai" (Deploy):
 *    -> Google sẽ yêu cầu cấp quyền: Bấm "Ủy quyền truy cập" (Authorize access).
 *    -> Chọn tài khoản Google của Thầy/Cô.
 *    -> Nếu hiện màn hình cảnh báo "Google chưa xác minh ứng dụng này":
 *       Bấm vào "Nâng cao" (Advanced) ở góc dưới bên trái -> Bấm vào "Đi tới ... (không an toàn)".
 *       Bấm "Cho phép" (Allow).
 * 8. Sao chép "URL ứng dụng web" (Web App URL có dạng https://script.google.com/macros/s/.../exec).
 * 9. Dán URL này vào ô "Cấu hình Google Sheet URL" trên trang web bài kiểm tra giữa kỳ.
 */

function doPost(e) {
  var lock = LockService.getScriptLock();
  // Khóa script tối đa 15 giây để xếp hàng các sinh viên nộp bài cùng lúc, tránh ghi đè dòng
  lock.tryLock(15000);

  try {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = ss.getActiveSheet();

    // Tự động tạo dòng tiêu đề nếu trang tính còn mới
    if (sheet.getLastRow() === 0) {
      sheet.appendRow([
        "Thời gian nộp", 
        "Họ và tên sinh viên", 
        "Mã sinh viên (MSSV)", 
        "Lớp sinh hoạt", 
        "Lần thi",
        "Điểm (Thang 10)", 
        "Số câu đúng", 
        "Tổng số câu", 
        "Thời gian làm bài", 
        "Tỷ lệ hoàn thành",
        "Số lần rời màn hình (Vi phạm)"
      ]);
      
      // Định dạng dòng tiêu đề xanh dương công nghệ Máy học
      var headerRange = sheet.getRange(1, 1, 1, 11);
      headerRange.setFontWeight("bold");
      headerRange.setBackground("#1a73e8");
      headerRange.setFontColor("#ffffff");
      headerRange.setHorizontalAlignment("center");
      headerRange.setVerticalAlignment("middle");
      sheet.setRowHeight(1, 35);
      sheet.setFrozenRows(1);
    }

    // Đọc dữ liệu JSON gửi từ ứng dụng web
    var data = {};
    if (e.postData && e.postData.contents) {
      data = JSON.parse(e.postData.contents);
    }

    var row = [
      data.submitTime || Utilities.formatDate(new Date(), "GMT+7", "dd/MM/yyyy HH:mm:ss"),
      data.fullName || "Chưa có tên",
      data.studentId || "Chưa có MSSV",
      data.className || "Chưa có lớp",
      data.attemptNumber !== undefined ? "Lần " + data.attemptNumber : "Lần 1",
      data.score10 !== undefined ? Number(data.score10) : 0,
      data.correctCount !== undefined ? Number(data.correctCount) : 0,
      data.totalQuestions || 90,
      data.timeSpent || "",
      data.summaryText || "",
      data.tabSwitches !== undefined ? Number(data.tabSwitches) : 0
    ];

    sheet.appendRow(row);
    
    // Canh chỉnh giao diện các ô vừa thêm
    var lastRow = sheet.getLastRow();
    sheet.setRowHeight(lastRow, 26);
    sheet.getRange(lastRow, 1).setHorizontalAlignment("center");
    sheet.getRange(lastRow, 3).setHorizontalAlignment("center");
    sheet.getRange(lastRow, 4).setHorizontalAlignment("center");
    sheet.getRange(lastRow, 5).setHorizontalAlignment("center");
    sheet.getRange(lastRow, 6).setHorizontalAlignment("center").setFontWeight("bold").setFontColor("#1a73e8");
    sheet.getRange(lastRow, 7, 1, 5).setHorizontalAlignment("center");
    if (data.tabSwitches && Number(data.tabSwitches) > 0) {
      sheet.getRange(lastRow, 11).setFontColor("#dc2626").setFontWeight("bold");
    } else {
      sheet.getRange(lastRow, 11).setFontColor("#16a34a");
    }

    return ContentService.createTextOutput(JSON.stringify({ 
      "status": "success", 
      "message": "Đã lưu kết quả thi môn Máy học thành công vào Google Sheet!",
      "row": lastRow
    })).setMimeType(ContentService.MimeType.JSON);

  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({ 
      "status": "error", 
      "message": err.toString() 
    })).setMimeType(ContentService.MimeType.JSON);

  } finally {
    lock.releaseLock();
  }
}

function doGet(e) {
  return ContentService.createTextOutput("Hệ thống Web App chấm điểm môn Máy học (Đại học Đông Á) đang hoạt động bình thường!");
}

