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
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Tahmin yapma
    preds = model.predict(img)
    class_idx = np.argmax(preds, axis=1)[0]
    class_name = class_names[class_idx]
    confidence = preds[0][class_idx]

    # Sonuçları görüntü üzerinde gösterme
    cv2.putText(frame, f'{class_name}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (50, 50), (400, 400), (0, 255, 0), 2)

    # Görüntüyü gösterme
    cv2.imshow('Leaf Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
