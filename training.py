import os
import numpy as np
import cv2
import tensorflow as tf
import pickle
from mtcnn import MTCNN
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Đường dẫn
DATASET_PATH = "dataset"
VIDEO_PATH = "videos"
EMBEDDINGS_PATH = "embeddings.npz"
MODEL_PATH = "trained_facenet_model.pb"
FRAME_INTERVAL = 5
MAX_IMAGES_PER_ID = 200
VALID_VIDEO_EXTS = (".mp4", ".avi", ".mov")

# Load model FaceNet
def load_facenet_model(model_path):
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    sess = tf.compat.v1.Session(graph=graph)
    return sess, graph.get_tensor_by_name("input:0"), \
           graph.get_tensor_by_name("embeddings:0"), \
           graph.get_tensor_by_name("phase_train:0"), graph

# Tiền xử lý ảnh
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

# Lấy embedding
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

# Lấy ảnh từ video
def extract_frames(video_path, frame_interval):
    cap = cv2.VideoCapture(video_path)
    frames, count = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

# Xử lý toàn bộ dữ liệu
def process_data(sess, input_tensor, output_tensor, phase_train, graph):
    print("[INFO] Đang xử lý dữ liệu từ ảnh và video...")
    all_embeddings = []
    all_labels = []

    # Tập hợp các ID
    student_ids = set()
    for filename in os.listdir(DATASET_PATH):
        if "_" in filename:
            student_ids.add(filename.split("_")[0])
    for filename in os.listdir(VIDEO_PATH):
        if filename.endswith(VALID_VIDEO_EXTS):
            student_ids.add(os.path.splitext(filename)[0])

    for student_id in sorted(student_ids):
        embeddings = []

        # 1. Lấy ảnh từ dataset/
        for filename in sorted(os.listdir(DATASET_PATH)):
            if filename.startswith(student_id + "_"):
                img_path = os.path.join(DATASET_PATH, filename)
                img = cv2.imread(img_path)
                emb = get_face_embedding(sess, input_tensor, output_tensor, phase_train, img, graph)
                if emb is not None:
                    embeddings.append(emb)
                    if len(embeddings) >= MAX_IMAGES_PER_ID:
                        break
        # 2. Nếu chưa đủ thì lấy thêm từ video
        if len(embeddings) < MAX_IMAGES_PER_ID:
            video_file = os.path.join(VIDEO_PATH, f"{student_id}.avi")
            if os.path.exists(video_file):
                frames = extract_frames(video_file, FRAME_INTERVAL)
                detector = MTCNN()
                for frame in frames:
                    # Phát hiện khuôn mặt
                    results = detector.detect_faces(frame)
                    if results and results[0]['confidence'] > 0.9:
                        x, y, w, h = results[0]['box']
                        face = frame[y:y+h, x:x+w]
                        emb = get_face_embedding(sess, input_tensor, output_tensor, phase_train, face, graph)
                        if emb is not None:
                            embeddings.append(emb)
                            if len(embeddings) >= MAX_IMAGES_PER_ID:
                                break
                print(f"[VID] Đã bổ sung từ video cho {student_id}: {len(embeddings)} embeddings")
        # Lưu
        for emb in embeddings:
            all_embeddings.append(emb)
            all_labels.append(student_id)

    # Lưu embedding
    np.savez(EMBEDDINGS_PATH, embeddings=np.array(all_embeddings), labels=np.array(all_labels))
    print(f"[INFO] Đã lưu {len(all_embeddings)} embeddings vào {EMBEDDINGS_PATH}")

    train_mlp_classifier(np.array(all_embeddings), all_labels)

# Huấn luyện MLP
def train_mlp_classifier(embeddings, labels, model_path="mlp_classifier.h5", label_map_path="label_mapping.pkl"):
    print("[INFO] Đang huấn luyện mô hình MLP...")
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    y = [label_to_idx[label] for label in labels]
    y_cat = to_categorical(y, num_classes=len(label_to_idx))

    model = Sequential([
        tf.keras.Input(shape=(embeddings.shape[1],)),
        # Dense(521, activation='relu'),
        # Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(len(label_to_idx), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(embeddings, y_cat, epochs=50, batch_size=16, verbose=1)

    model.save(model_path)
    with open(label_map_path, "wb") as f:
        pickle.dump(label_to_idx, f)
    print(f"[INFO] Mô hình đã lưu tại {model_path} và {label_map_path}")

# Chạy chương trình
if __name__ == "__main__":
    sess, input_tensor, output_tensor, phase_train, graph = load_facenet_model(MODEL_PATH)
    process_data(sess, input_tensor, output_tensor, phase_train, graph)
    sess.close()
