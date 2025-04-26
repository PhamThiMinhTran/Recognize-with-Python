import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.metrics.pairwise import cosine_distances

# Load embeddings và labels
data = np.load('embeddings.npz')
X = data['embeddings']
y = data['labels']

# Encode labels (chuỗi -> số)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
label_classes = label_encoder.classes_

# One-hot encoding cho label
y_categorical = to_categorical(y_encoded)

# Train/Test Split
X_train, X_test, y_train_cat, y_test_cat, y_train, y_test = train_test_split(
    X, y_categorical, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Load mô hình MLP đã huấn luyện
model = load_model('mlp_classifier.h5')
print(model.summary())

# Tạo embedding đại diện (mean vector cho mỗi lớp)
ref_embeddings = {}
for i, label in enumerate(label_classes):
    indices = np.where(y_train == i)[0]
    if len(indices) > 0:
        ref_embeddings[label] = np.mean(X_train[indices], axis=0)

# Hàm tìm khoảng cách embedding gần nhất

def get_nearest_label_by_embedding(embedding, ref_embeddings):
    min_dist = float('inf')
    best_label = None

    for label, ref_vec in ref_embeddings.items():
        dist = cosine_distances([embedding], [ref_vec])[0][0]
        if dist < min_dist:
            min_dist = dist
            best_label = label

    return best_label, min_dist

# Tìm tổ hợp ngưỡng tối ưu
prob_thresholds = np.arange(0.1, 1.0, 0.05)
dist_thresholds = np.arange(0.1, 1.0, 0.05)

results = []

y_pred_probs = model.predict(X_test)
y_pred_max_probs = np.max(y_pred_probs, axis=1)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

for prob_thres in prob_thresholds:
    for dist_thres in dist_thresholds:
        correct = 0
        total_accepted = 0
        total_samples = len(X_test)

        for i, prob in enumerate(y_pred_max_probs):
            if prob >= prob_thres:
                predicted_label = label_classes[y_pred_classes[i]]
                true_label = label_classes[y_test[i]]

                embed = X_test[i]
                nearest_label, dist = get_nearest_label_by_embedding(embed, ref_embeddings)

                if dist <= dist_thres:
                    if nearest_label == true_label:
                        correct += 1
                    total_accepted += 1

        acc = correct / total_accepted if total_accepted > 0 else 0
        reject_rate = 1 - (total_accepted / total_samples)

        results.append((prob_thres, dist_thres, acc, reject_rate))

        print(f"[prob >= {prob_thres:.2f}, dist <= {dist_thres:.2f}] -> Accuracy: {acc*100:.2f}%, Reject Rate: {reject_rate*100:.2f}%")

# Vẽ heatmap accuracy
import pandas as pd

results_df = pd.DataFrame(results, columns=["prob_thres", "dist_thres", "accuracy", "reject_rate"])
pivot_acc = results_df.pivot(index="dist_thres", columns="prob_thres", values="accuracy")

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_acc, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Accuracy Heatmap theo (prob_threshold, dist_threshold)")
plt.xlabel("prob_threshold")
plt.ylabel("dist_threshold")
plt.tight_layout()
plt.show()
