import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.utils import get_custom_objects


img_size = (224, 224)
input_shape = img_size + (3,)
batch_size = 32

# Test veri setini yükleme
test_ds = tf.keras.utils.image_dataset_from_directory(
  shuffle=False,
  directory='datasets/test',
  label_mode="categorical",
  seed=123,
  image_size=img_size,
  batch_size=batch_size)

class_names = test_ds.class_names

print(class_names)
print(type(class_names))

# preprocess_input fonksiyonunu Keras'a kayıt et
get_custom_objects().update({'preprocess_input': preprocess_input})

# Eğitilmiş modeli yükle
model = tf.keras.models.load_model('saved_model/my_model.h5')
def detect_and_classify_leaves(frame):
    h, w = frame.shape[:2]

    # Görüntüyü 224x224 boyutunda yeniden boyutlandır ve ön işleme yap
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame = preprocess_input(resized_frame)
    resized_frame = np.expand_dims(resized_frame, axis=0)

    # Yaprak sınıflandırma
    preds = model.predict(resized_frame)
    class_idx = np.argmax(preds)
    class_name = class_names[class_idx]
    confidence = preds[0][class_idx] * 100

    # Kare çiz ve sınıflandırma sonucunu göster
    text = f"{class_name}: {confidence:.2f}%"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Kamera akışını başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_and_classify_leaves(frame)
    
    cv2.imshow("Leaf Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()