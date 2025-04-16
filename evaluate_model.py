import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

# Load embeddings và labels
data = np.load('embeddings.npz')
X = data['embeddings']
y = data['labels']

# Encode labels (chuỗi -> số)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encoding cho label
y_categorical = to_categorical(y_encoded)

# Train/Test Split
X_train, X_test, y_train_cat, y_test_cat, y_train, y_test = train_test_split(
    X, y_categorical, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Load mô hình MLP đã huấn luyện
model = load_model('mlp_classifier.h5')
print(model.summary())
# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

# Dự đoán
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# In báo cáo đánh giá
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()