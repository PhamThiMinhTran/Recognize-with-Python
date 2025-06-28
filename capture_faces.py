import cv2
import os
import numpy as np
import threading
from mtcnn import MTCNN
import base64
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

# Ẩn cảnh báo TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_PATH = "trained_facenet_model.pb"
DATASET_PATH = "dataset"
VIDEO_PATH = "videos"
BUFFER_SIZE = 3  # Giảm buffer để nhanh hơn

# Tạo thư mục nếu chưa tồn tại
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(VIDEO_PATH, exist_ok=True)
capture_context = {}
# Load mô hình FaceNet
def load_facenet_model():
    with tf.io.gfile.GFile(MODEL_PATH, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

# Lấy các tensor cần thiết từ mô hình
def get_model_tensors(graph):
    return graph.get_tensor_by_name("input:0"), graph.get_tensor_by_name("embeddings:0"), graph.get_tensor_by_name("phase_train:0")


def preprocess_face(face):
    # if face.shape[0] < 20 or face.shape[1] < 20:
    #     return None
    face = cv2.resize(face, (160, 160))
    face = np.expand_dims(face, axis=0)
    return preprocess_input(face)

def get_face_embedding(face, sess, input_tensor, output_tensor, phase_train, graph):
    face = preprocess_face(face)
    if face is None:
        return None
    with graph.as_default():
        return sess.run(output_tensor, feed_dict={input_tensor: face, phase_train: False})[0]

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def augment_image(image):
    augmented_images = [image, cv2.flip(image, 1)]
    h, w = image.shape[:2]
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        augmented_images.append(cv2.warpAffine(image, M, (w, h)))
    for alpha in [0.8, 1.2]:
        augmented_images.append(cv2.convertScaleAbs(image, alpha=alpha, beta=0))
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    augmented_images.append(cv2.add(image, noise))
    return augmented_images

# def capture_faces():
#     graph = load_facenet_model()
#     input_tensor, output_tensor, phase_train = get_model_tensors(graph)
#     sess = tf.compat.v1.Session(graph=graph)
#     print("\033[92m[INFO] Mô hình FaceNet đã tải thành công!\033[0m")
    
#     detector = MTCNN()
#     cap = initialize_camera()
#     print("\033[93m[INFO] Đang bật camera... Nhấn 'Y' để bắt đầu chụp hoặc 'N' để thoát!\033[0m")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("\033[91m[LỖI] Không thể bật camera!\033[0m")
#             return
#         cv2.imshow("Camera Preview", frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('y'):
#             print("\033[94m[INFO] Bắt đầu chụp ảnh!\033[0m")
#             break
#         elif key == ord('n'):
#             print("\033[91m[INFO] Thoát chương trình!\033[0m")
#             cap.release()
#             cv2.destroyAllWindows()
#             sess.close()
#             return
    
#     student_id = input("\033[96m[INFO] Nhập họ và tên sinh viên: \033[0m")
#     video_name = f"{VIDEO_PATH}/{student_id}.avi"
#     frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(video_name, fourcc, 20.0, (frame_width, frame_height))
    
#     print(f"\033[94m[INFO] Video sẽ được lưu tại: {video_name}\033[0m")
    
#     count, reference_embedding, face_embeddings_buffer = 0, None, []
#     while count < 30:
#         ret, frame = cap.read()
#         if not ret:
#             print("\033[91m[LỖI] Không thể đọc dữ liệu từ camera!\033[0m")
#             break
#         faces = detector.detect_faces(frame)
#         if not faces:
#             continue
        
#         for face in faces:
#             x, y, w, h = max(0, face["box"][0]), max(0, face["box"][1]), face["box"][2], face["box"][3]
#             face_img = frame[y:y + h, x:x + w]
            
#             if is_blurry(face_img) or w < 30 or h < 30:
#                 continue
            
#             embedding = get_face_embedding(face_img, sess, input_tensor, output_tensor, phase_train)
#             if embedding is None:
#                 continue
            
#             if reference_embedding is None:
#                 reference_embedding = embedding
#                 print("\033[92m[INFO] Ảnh tham chiếu đã lưu!\033[0m")
            
#             face_embeddings_buffer.append(embedding)
#             if len(face_embeddings_buffer) == BUFFER_SIZE:
#                 avg_embedding = np.mean(face_embeddings_buffer, axis=0)
#                 if np.linalg.norm(reference_embedding - avg_embedding) < 1.2:
#                     for i, img in enumerate(augment_image(face_img)):
#                         img_name = f"{DATASET_PATH}/{student_id}_{count}_{i}.jpg"
#                         cv2.imwrite(img_name, img)
#                         print(f"\033[92m[INFO] Đã lưu ảnh: {img_name}\033[0m")
#                     count += 1
#                     face_embeddings_buffer.clear()
    
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     sess.close()
#     print("\033[92m[INFO] Chụp ảnh hoàn tất! Video đã được lưu tại:", video_name, "\033[0m")

# if __name__ == "__main__":
#     capture_faces()

# Hàm ghi video vào file
def initialize_context(student_id, frame):
    graph = load_facenet_model()
    input_tensor, output_tensor, phase_train = get_model_tensors(graph)
    sess = tf.compat.v1.Session(graph=graph)
    h, w = frame.shape[:2]
    video_name = os.path.join(VIDEO_PATH, f"{student_id}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(video_name, fourcc, 10.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"[ERROR] Không thể mở VideoWriter để ghi video: {video_name}")
    return {
        "sess": sess,
        "graph": graph,
        "input_tensor": input_tensor,
        "output_tensor": output_tensor,
        "phase_train": phase_train,
        "detector": MTCNN(),
        "reference_embedding": None,
        "face_embeddings_buffer": [],
        "count": 0,
        "video_writer": writer,
        "video_name": video_name,
        "done": False
    }
def save_face_images(frame, faces, student_id, context):
    sess = context["sess"]
    input_tensor = context["input_tensor"]
    output_tensor = context["output_tensor"]
    phase_train = context["phase_train"]
    for face in faces:
        x, y, w, h = max(0, face["box"][0]), max(0, face["box"][1]), face["box"][2], face["box"][3]
        face_img = frame[y:y + h, x:x + w]
        if is_blurry(face_img):
            continue
        embedding = get_face_embedding(face_img, sess, input_tensor, output_tensor, phase_train, context["graph"])
        if embedding is None:
            continue
        if context["reference_embedding"] is None:
            context["reference_embedding"] = embedding
            print("[INFO] Ảnh tham chiếu đã lưu")
        context["face_embeddings_buffer"].append(embedding)
        if len(context["face_embeddings_buffer"]) == BUFFER_SIZE:
            avg_embedding = np.mean(context["face_embeddings_buffer"], axis=0)
            if np.linalg.norm(context["reference_embedding"] - avg_embedding) < 1.2:
                for i, img in enumerate(augment_image(face_img)):
                    img_name = os.path.join(DATASET_PATH, f"{student_id}_{context['count']}_{i}.jpg")
                    cv2.imwrite(img_name, img)
                    print(f"\033[92m[INFO] Đã lưu ảnh: {img_name}\033[0m")
                context["count"] += 1
                context["face_embeddings_buffer"].clear()

                if context["count"] >= 50:
                    context["video_writer"].release()
                    context["done"] = True
                    return True
    return False
def process_uploaded_image(image_data_base64, student_id, context_holder):
    try:
        image_data = base64.b64decode(image_data_base64.split(",")[-1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"saved": False, "message": "Ảnh không hợp lệ."}
        if context_holder.get("done"):
            return {"saved": False, "message": f"Đã hoàn tất lưu ảnh cho {student_id}."}
        if "sess" not in context_holder:
            print("[INFO] Đang khởi tạo mô hình và video...")
            context = initialize_context(student_id, frame)
            context_holder.update(context)
            print(f"[INFO] Video sẽ được lưu tại: {context['video_name']}")
        context = context_holder
        if not context.get("done", False):
            context["video_writer"].write(frame)
        faces = context["detector"].detect_faces(frame)
        if not faces:
            return {"saved": False, "message": "Không phát hiện khuôn mặt."}
        done = save_face_images(frame, faces, student_id, context)
        if done:
            return {"saved": True, "done": True, "message": f"Đã hoàn tất lưu ảnh cho {student_id}"}
        return {"saved": True, "done": False, "message": f"Đã xử lý ảnh {context['count']} / 30"}
    except Exception as e:
        print(f"[ERROR] Lỗi xử lý ảnh: {e}")
        return {"saved": False, "message": f"Lỗi xử lý: {str(e)}"}
