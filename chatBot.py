import re
import pandas as pd
import os
from datetime import datetime

diem_danh_df = None

def get_bot_response(message):
    original_message = message.strip()
    message_lower = original_message.lower()

    greetings = ["hi", "hello", "hey", "xin chào", "chào bạn", "bot ơi", "yo"]
    usage = ["help", "giúp tôi", "giúp", "hỗ trợ", "hướng dẫn", "sử dụng", "dùng sao", "cách dùng"]
    thanks = ["thank you", "cảm ơn", "cảm ơn bạn", "cảm ơn bot", "cảm ơn nhiều", "cảm ơn rất nhiều"]
    bye = ["bye", "tạm biệt", "hẹn gặp lại", "tạm biệt bạn", "tạm biệt bot"]

    if message_lower in greetings:
        return "Xin chào! Tôi là bot hỗ trợ của bạn. Bạn cần gì hôm nay?"
    elif message_lower in usage:
        return (
            "Hướng dẫn sử dụng hệ thống điểm danh, bạn cần thực hiện các bước như sau:\n"
            "Bước 1: Nhập tên sinh viên vào ô tên. Lưu ý cần lưu tên không có ký tự đặc biệt hoặc dấu.\n"
            "Bước 2: Để chụp ảnh sinh viên, nhấn vào nút **Capture Faces**. Việc chụp ảnh sẽ mất một chút thời gian\n"
            "Lưu ý: Khi chụp ảnh, bạn hãy thực hiện những hành động như: Nhìn thẳng, nghiêng trái, nghiêng phải, cúi xuống, ngẩng lên, cười, nhắm mắt.\n"
            "Bước 3: Sau khi chụp ảnh, bạn có thể nhấn vào nút **Train System** để bắt đầu quá trình huấn luyện mô hình nhận diện khuôn mặt. Chờ hệ thống xử lý và training\n"
            "Bước 4: Sau khi thực hiện **Training** bạn có thể bắt đầu điểm danh bằng cách nhấn vào **Recognize Faces**\n"
            "Hệ thống sẽ điểm danh và đưa ra tên người đã được điểm danh bên khung phải\n"
            "Nếu bạn muốn điểm danh lại chỉ cần nhấn **Stop recognize** và nhấn **Recognize Faces** lại\n"
            "Lưu ý: Nếu bạn muốn điểm danh lại, bạn cần nhấn vào nút **Stop recognize** trước khi nhấn vào nút **Recognize Faces**\n"
            "Bước 5: Sau khi điểm danh hoàn tất bạn hãy chọn nút **Doawload File** để xem danh sách sinh viên đã điểm danh, nếu danh sách chưa đủ thì bạn bấm **Hủy** và điểm danh lại từ Bước 4. Mặc khác, nếu danh sách đầy đủ thì chọn **Download** để tải file EXCEL\n"
        )
    elif message_lower in thanks:
        return "Không có gì! Tôi luôn sẵn sàng giúp đỡ bạn."
    elif message_lower in bye:
        return "Tạm biệt! Hẹn gặp lại bạn sau."
    elif (
        "có điểm danh" in message_lower
        or "điểm danh ngày" in message_lower
        or "ai đã điểm danh" in message_lower
        or "ai vắng" in message_lower
    ):
        return process_attendance_query(original_message)
    else:
        return "Xin lỗi, tôi chưa hiểu bạn muốn gì. Bạn có thể hỏi về điểm danh hoặc gõ 'hướng dẫn' để biết cách sử dụng."


def clean_name(name):
    return name.strip().lower()

def process_attendance_query(message):
    try:
        if not os.path.exists("attendance.xlsx"):
            return "Không tìm thấy file điểm danh."
        df = pd.read_excel("attendance.xlsx")
        if not all(col in df.columns for col in ['Date', 'Name', 'Session']):
            return "File 'attendance.xlsx' thiếu cột bắt buộc: Date, Name, Session."
        # Tiền xử lý dữ liệu
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Day'] = df['Date'].dt.day.astype(str).str.zfill(2)
        df['Name_clean'] = df['Name'].astype(str).apply(clean_name)
        message_lower = message.lower()
        ngay_match = re.search(r"ngày\s+(\d{1,2})", message_lower)
        ngay = ngay_match.group(1).zfill(2) if ngay_match else None
        ten_match = re.search(r'bạn\s+[\"“”]?([^\"“”]+?)[\"“”]?\s+có\s+điểm\s+danh', message_lower)
        if ten_match and ngay:
            ten_raw = ten_match.group(1).strip()
            ten_clean = clean_name(ten_raw)
            matched = df[(df['Name_clean'] == ten_clean) & (df['Day'] == ngay)]
            if matched.empty:
                return f"Không tìm thấy thông tin điểm danh của bạn {ten_raw} ngày {ngay}"
            buois = ', '.join(matched['Session'].tolist())
            return f"Bạn {ten_raw} có điểm danh ngày {ngay} vào buổi: {buois}."

        # === 2. Ai đã điểm danh ngày đó ===
        if "ai đã điểm danh" in message_lower and ngay:
            matched = df[df['Day'] == ngay]
            if matched.empty:
                return f"📭 Không có ai điểm danh ngày {ngay}."
            danh_sach = '\n'.join(f"- {row['Name']} ({row['Session']})" for _, row in matched.iterrows())
            return f"Danh sách đã điểm danh ngày {ngay}:\n{danh_sach}"

        # === 3. Ai vắng mặt ngày đó ===
        if "có ai vắng" in message_lower and ngay:
            if not os.path.exists("student_list.xlsx"):
                return "Bạn cần cung cấp file 'student_list.xlsx' để kiểm tra ai vắng."

            student_df = pd.read_excel("student_list.xlsx")
            if 'Name' not in student_df.columns:
                return "File 'student_list.xlsx' không đúng định dạng. Thiếu cột 'Name'."

            student_df['Name_clean'] = student_df['Name'].astype(str).apply(clean_name)
            diem_danh_names = df[df['Day'] == ngay]['Name_clean'].unique()
            vang = student_df[~student_df['Name_clean'].isin(diem_danh_names)]

            if vang.empty:
                return f"Tất cả sinh viên đều đã điểm danh ngày {ngay}."
            danh_sach_vang = '\n'.join(f"- {row['Name']}" for _, row in vang.iterrows())
            return f"Danh sách sinh viên vắng mặt ngày {ngay}:\n{danh_sach_vang}"

        # === Không khớp mẫu nào ===
        return (
            "Bạn có thể hỏi:\n"
            "- Bạn \"Nguyễn Văn A\" có điểm danh ngày ?? không?\n"
            "Lưu ý: Tên hãy để trong dấu ngoặc kép (" ")\n"
            "- Ai đã điểm danh ngày ???\n"
            "- Ngày ?? có ai vắng không?"
        )

    except Exception as e:
        return f"⚠️ Đã xảy ra lỗi khi xử lý: {str(e)}"
