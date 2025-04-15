import base64
import threading
import cv2
import os
from flask import Flask, jsonify, request, send_file
from flask_restx import Api, Resource, fields, reqparse
from flask_cors import CORS
from datetime import datetime
from mtcnn import MTCNN
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from capture_faces import process_uploaded_image, capture_context
from training import load_facenet_model, process_data
from recognize import recognize_faces_from_image, save_attendance, attendance_data

app = Flask(__name__)
CORS(app)
api = Api(app, version="1.0", title="Face Recognition API", description="API điểm danh bằng nhận diện khuôn mặt")
ns = api.namespace("api", description="Chức năng điểm danh")

capture_model = api.model('CaptureInput', {
    'images': fields.List(fields.String, required=True, description='Danh sách ảnh base64'),
    'student_id': fields.String(required=True, description='Tên sinh viên'),
})

training_model = api.model('TrainingInput', {
    'start_training': fields.Boolean(required=True, description='Bắt đầu quá trình huấn luyện mô hình'),
})

recognize_model = api.model('RecognizeInput', {
    'image': fields.String(required=True, description='Ảnh base64 (jpg/png)')
})

# Đường dẫn đến các tệp tin
EMBEDDINGS_PATH = "embeddings.npz"
MODEL_PATH = "trained_facenet_model.pb"
DATASET_PATH = "dataset"
VIDEO_PATH = "videos"
training_status = {"status": "Chưa bắt đầu"}

# Khởi tạo model faceNet
def start_training_thread():
    global training_status
    try:
        training_status["status"] = "Đang huấn luyện..."
        print("[INFO] Kiểm tra các tệp tin và thư mục...")
        if not os.path.exists(MODEL_PATH):
            training_status["status"] = "Lỗi: Không tìm thấy model"
            print(f"[ERROR] Không tìm thấy mô hình tại: {MODEL_PATH}")
            return
        if not os.path.exists(DATASET_PATH):
            training_status["status"] = "Lỗi: Không tìm thấy dataset"
            print(f"[ERROR] Không tìm thấy thư mục dataset tại: {DATASET_PATH}")
            return
        if not os.path.exists(VIDEO_PATH):
            training_status["status"] = "Lỗi: Không tìm thấy videos"
            print(f"[ERROR] Không tìm thấy thư mục videos tại: {VIDEO_PATH}")
            return
        print("[INFO] Bắt đầu huấn luyện mô hình...")
        sess, input_tensor, output_tensor, phase_train, graph = load_facenet_model(MODEL_PATH)
        process_data(sess, input_tensor, output_tensor, phase_train, graph)
        sess.close()
        training_status["status"] = f"Hoàn tất training hệ thống điểm danh bằng khuôn mặt"
        print("[INFO] Quá trình huấn luyện hoàn tất.")
    except Exception as e:
        training_status["status"] = f"Lỗi: {e}"
        print(f"[ERROR] Lỗi khi huấn luyện mô hình: {e}")

@ns.route("/capture")
class CaptureFace(Resource):
    @ns.expect(capture_model)
    def post(self):
        try:
            data = request.get_json()
            images_base64 = data.get("images")
            student_id = data.get("student_id")

            if not isinstance(images_base64, list) or not student_id:
                return jsonify({"error": "Thieu anh hoac ma sinh vien"}), 400
            global capture_context
            if capture_context.get("student_id") != student_id or capture_context.get("done", False):
                print(f"[INFO] Đang chuyển sang sinh viên mới: {student_id}")
                capture_context = {"student_id": student_id}
            results = []
            for image_base64 in images_base64:
                result = process_uploaded_image(image_base64, student_id, capture_context)
                results.append(result)
                if result.get("done"):
                    break

            return {
                "student_id": student_id,
                "processed_count": len(results),
                "done": any(r.get("done") for r in results),
                "results": results}, 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@ns.route("/training")
class TrainModel(Resource):
    @ns.expect(training_model)
    def post(self):
        try:
            data = request.get_json()
            start_training = data.get("start_training", False)

            if start_training:
                # Thực hiện huấn luyện mô hình trong một thread riêng để không làm gián đoạn API
                threading.Thread(target=start_training_thread, daemon=True).start()
                return {"message": "Đã bắt đầu huấn luyện mô hình."}, 200
            else:
                return {"error": "Thông tin huấn luyện không hợp lệ."}, 400

        except Exception as e:
            return {"error": str(e)}, 500
        
@ns.route("/training_status")
class TrainingStatus(Resource):
    def get(self):
        return {"training_status": training_status["status"]}, 200 

def base64_to_image(base64_str):
    try:
        if "," in base64_str:
            header, encoded = base64_str.split(",", 1)
        else:
            encoded = base64_str
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] base64 decode failed: {e}")
        return None

@ns.route("/recognize")
class RecognizeFace(Resource):
    @ns.expect(recognize_model)
    def post(self):
        attendance_data.clear()
        data = request.get_json()
        if not data or "image" not in data:
            return {"error": "Missing image"}, 400
        frame = base64_to_image(data["image"])
        if frame is None:
            return {"error": "Invalid image"}, 400
        results = recognize_faces_from_image(frame)
        if results:
            names = [item["label"] for item in results if "label" in item]
            return {"message": f"Da nhan dien duoc: {', '.join(names)}", "results": results}
        else:
            return {"message": "Khong nhan dien duoc mat.", "results": []}

@ns.route("/save_attendance")
class SaveAttendance(Resource):
    def post(self):
        if not attendance_data:
            return {"error": "Chưa có ai được điểm danh!"}, 400
        try:
            save_attendance()
            attendance_data.clear()  
            return {"message": "Đã lưu điểm danh vào attendance.xlsx"}, 200
        except Exception as e:
            return {"error": str(e)}, 500

@ns.route("/download_attendance")
class DownloadAttendance(Resource):
    def get(self):
        path = "attendance.xlsx"
        if os.path.exists(path):
            return send_file(path, as_attachment=True)
        return {"error": "File không tìm thấy"}, 404

if __name__ == "__main__":
    app.run(debug=False)
