import os
import time
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from mtcnn import MTCNN
from datetime import datetime
from tensorflow.keras.models import load_model  # type: ignore
from openpyxl import load_workbook
import atexit

# === Cấu hình ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
EMBEDDINGS_PATH = "embeddings.npz"
MODEL_PATH = "trained_facenet_model.pb"
ATTENDANCE_EXCEL = "attendance.xlsx"
MLP_MODEL_PATH = "mlp_classifier.h5"
LABEL_MAP_PATH = "label_mapping.pkl"
MLP_PROB_THRESHOLD = 0.85      # Số ... đến 0.85
EMBED_DIST_THRESHOLD = 0.55    # Số 0.55 đến 0.7

# === Trạng thái toàn cục ===
attended_today = set()
detector = MTCNN()
attendance_data = []
ACTIVE_TRACKS = {}
TOTAL_TRACKS = {}
NEXT_ID = 1
RECOGNITION_COUNTS = {} 
RECOGNITION_THRESHOLD = 2

# === Load mô hình ===
def load_facenet_model():
    with tf.io.gfile.GFile(MODEL_PATH, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    sess = tf.compat.v1.Session(graph=graph)
    return sess, graph

sess, graph = load_facenet_model()
input_tensor = graph.get_tensor_by_name("input:0")
output_tensor = graph.get_tensor_by_name("embeddings:0")
phase_train = graph.get_tensor_by_name("phase_train:0")
atexit.register(sess.close) 

mlp_model = load_model(MLP_MODEL_PATH, compile=False)
with open(LABEL_MAP_PATH, "rb") as f:
    label_mapping = pickle.load(f)
inv_label_mapping = {v: k for k, v in label_mapping.items()}
data = np.load(EMBEDDINGS_PATH)
known_embeddings = data['embeddings']
known_labels = data['labels']

def preprocess_face(face_img):
    if face_img is None or face_img.shape[0] < 20 or face_img.shape[1] < 20:
        return None
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    face_img = cv2.filter2D(face_img, -1, kernel)
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm

def get_nearest_label_by_embedding(embedding, known_embeddings, known_labels, threshold=0.8):
    distances = np.linalg.norm(known_embeddings - embedding, axis=1)
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]

    if min_dist < threshold:
        label_idx = known_labels[min_idx]
        return label_idx, min_dist
    else:
        return "Unknown", min_dist

def get_face_embedding(sess, input_tensor, output_tensor, phase_train, face_img, graph):
    processed = preprocess_face(face_img)
    if processed is None:
        return None
    with graph.as_default():
        embedding = sess.run(output_tensor, feed_dict={
            input_tensor: processed,
            phase_train: False
        })[0]
    return embedding / np.linalg.norm(embedding)

def iou(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_session_period():
    now = datetime.now().hour
    if 5 <= now < 12:
        return "Sáng"
    elif 12 <= now < 17:
        return "Chiều"
    else:
        return "Tối"

def mark_attendance(label):
    global attended_today
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    session = get_session_period()
    for record in attendance_data:
        if record['Date'] == date_str and record['Session'] == session and record['Name'] == label:
            return False
    attendance_data.append({"Date": date_str, "Time": time_str, "Session": session, "Name": label})
    attended_today.add(label)
    print(f"Đã điểm danh: {label} ({session})")
    return True

def save_attendance():
    if not attendance_data:
        print("[INFO] Chưa có ai được điểm danh!")
        return
    df = pd.DataFrame(attendance_data)
    if os.path.exists(ATTENDANCE_EXCEL):
        with pd.ExcelWriter(ATTENDANCE_EXCEL, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            workbook = writer.book
            sheet = workbook.active
            start_row = sheet.max_row + 1 
            df.to_excel(writer, index=False, header=False, startrow=start_row)
    else:
        df.to_excel(ATTENDANCE_EXCEL, index=False, header=True)
    print("Đã lưu điểm danh thành công!")

# def recognize_faces():
#     global NEXT_ID, attended_today
#     detector = MTCNN()
#     cap = cv2.VideoCapture(0)
#     print("[INFO] Mở camera. Nhấn 'q' để bắt đầu hoặc reset điểm danh, 'f' để lưu, 'ESC' để thoát.")
#     attendance_mode = False
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[LỖI] Không thể đọc từ camera")
#             break
#         frame = preprocess_image(frame)
#         faces = detector.detect_faces(frame)
#         current_ids = set()
#         if attendance_mode:
#             for face in faces:
#                 x, y, w, h = face['box']
#                 face_img = frame[y:y + h, x:x + w]
#                 embedding = get_face_embedding(face_img)
#                 if embedding is None:
#                     continue
#                 probs = mlp_model.predict(np.expand_dims(embedding, axis=0), verbose=0)[0]
#                 max_prob = np.max(probs)
#                 label_idx = np.argmax(probs)
#                 label = inv_label_mapping[label_idx] if max_prob > THRESHOLD else "Unknown"
#                 assigned_id = None
#                 min_dist = 1e9
#                 for pid, info in ACTIVE_TRACKS.items():
#                     iou_score = iou(face['box'], info['bbox'])
#                     emb_dist = np.linalg.norm(embedding - info['embedding'])
#                     if iou_score > 0.3 or (label != "Unknown" and emb_dist < 0.6):
#                         if emb_dist < min_dist:
#                             min_dist = emb_dist
#                             assigned_id = pid
#                 if assigned_id is None:
#                     assigned_id = f"Person_{NEXT_ID}"
#                     NEXT_ID += 1
#                     TOTAL_TRACKS[assigned_id] = label
#                 ACTIVE_TRACKS[assigned_id] = {'bbox': face['box'], 'embedding': embedding}
#                 current_ids.add(assigned_id)
#                 # Vẽ thông tin lên ảnh
#                 color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                 cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#                 if label != "Unknown":
#                     mark_attendance(label)
#         for pid in set(ACTIVE_TRACKS.keys()) - current_ids:
#             del ACTIVE_TRACKS[pid]
#         # Thông tin hiển thị
#         now_str = datetime.now().strftime("%H:%M:%S")
#         cv2.putText(frame, f"People: {len(TOTAL_TRACKS)} | Present: {len(ACTIVE_TRACKS)} | Marked: {len(attended_today)}",
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(frame, f"Time: {now_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
#         cv2.imshow("Nhận diện khuôn mặt", frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # ESC
#             print("[EXIT] Đã thoát chương trình.")
#             break
#         elif key == ord('q'):
#             if not attendance_mode:
#                 attendance_mode = True
#                 print("[INFO] Bắt đầu điểm danh...")
#             else:
#                 attendance_mode = False
#                 attended_today.clear()
#                 attendance_data.clear()
#                 ACTIVE_TRACKS.clear()
#                 TOTAL_TRACKS.clear()
#                 NEXT_ID = 1
#                 print("[INFO] Đã reset danh sách điểm danh.")
#         elif key == ord('f'):
#             save_attendance()
#     cap.release()
#     cv2.destroyAllWindows()
# if __name__ == "__main__":
#     recognize_faces()

def recognize_faces_from_image(frame):
    global NEXT_ID
    faces = detector.detect_faces(frame)
    results = []
    current_ids = set()

    for face in faces:
        if face['confidence'] < 0.92: 
            continue  
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_img = frame[y:y + h, x:x + w]
        embedding = get_face_embedding(sess, input_tensor, output_tensor, phase_train, face_img, graph)
        if embedding is None:
            continue

        probs = mlp_model.predict(np.expand_dims(embedding, axis=0), verbose=0)[0]
        max_prob = np.max(probs)
        label_idx = np.argmax(probs)
        label = inv_label_mapping[label_idx]
        label_by_dist, dist = get_nearest_label_by_embedding(embedding, known_embeddings, known_labels, threshold=EMBED_DIST_THRESHOLD)
        if max_prob >= MLP_PROB_THRESHOLD and dist <= EMBED_DIST_THRESHOLD:
            label = inv_label_mapping[label_idx]
        elif dist <= EMBED_DIST_THRESHOLD:
            label = label_by_dist
        else:
            label = "Unknown"

        assigned_id = None
        min_dist = 1e9
        if label != "Unknown":
            for pid, info in ACTIVE_TRACKS.items():
                iou_score = iou(face['box'], info['bbox'])
                emb_dist = np.linalg.norm(embedding - info['embedding'])
                if iou_score > 0.3 or emb_dist < 0.6:
                    if emb_dist < min_dist:
                        min_dist = emb_dist
                        assigned_id = pid

        if assigned_id is None:
            assigned_id = f"Person_{NEXT_ID}"
            NEXT_ID += 1
            TOTAL_TRACKS[assigned_id] = label
        ACTIVE_TRACKS[assigned_id] = {'bbox': face['box'], 'embedding': embedding}
        current_ids.add(assigned_id)

        if label != "Unknown":
            if label not in attended_today:
                RECOGNITION_COUNTS[label] = RECOGNITION_COUNTS.get(label, 0) + 1
                if RECOGNITION_COUNTS[label] == RECOGNITION_THRESHOLD:
                    print(f"[INFO] {label} đã được xác nhận sau {RECOGNITION_THRESHOLD} lần nhận diện.")
                    mark_attendance(label)
                    RECOGNITION_COUNTS[label] = 0
                else:
                    continue  
            label_to_return = label
        else:
            label_to_return = "Unknown"

        results.append({
            "id": assigned_id,
            "label": label_to_return,
            "probability": float(max_prob),
            "bbox": [int(x), int(y), int(w), int(h)]
        })

    # Xoá người đã rời khỏi khung hình
    for pid in list(ACTIVE_TRACKS.keys()):
        if pid not in current_ids:
            del ACTIVE_TRACKS[pid]

    return results