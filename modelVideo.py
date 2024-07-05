import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input

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

# Custom object olarak preprocess_input fonksiyonunu ekleyerek modeli yükleme
model = load_model('saved_model/my_model.h5', custom_objects={'preprocess_input': preprocess_input})

# Video kamerasını açma
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü işleme
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Kontur bulma
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # En büyük konturu bulma
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Eğer kontur yeterince büyükse tahmin yap
        if w > 50 and h > 50:
            leaf = frame[y:y+h, x:x+w]
            leaf = cv2.resize(leaf, img_size)
            leaf = leaf.astype("float32")
            leaf = np.expand_dims(leaf, axis=0)
            leaf = preprocess_input(leaf)

            # Tahmin yapma
            preds = model.predict(leaf)
            class_idx = np.argmax(preds, axis=1)[0]
            class_name = class_names[class_idx]
            confidence = preds[0][class_idx]

            # Sonuçları görüntü üzerinde gösterme
            cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Görüntüyü gösterme
    cv2.imshow('Leaf Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
