---

# Bean Leaf Disease Classification and Detection

This project uses a ResNet50 model to detect diseases in bean leaves. The model is trained using images of bean leaves and can classify them as healthy or diseased. The project also includes scripts to capture video from a camera and detect the diseases in real-time.

## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Real-Time Detection](#real-time-detection)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Bean leaf disease detection is an important task in agriculture. This project aims to automate the process of detecting diseases in bean leaves using deep learning techniques. The model is based on ResNet50 and is trained to classify bean leaves into different categories, including healthy and various diseases.

## Datasets
Two datasets are used in this project:
1. [Kaggle Bean Disease Dataset](https://www.kaggle.com/datasets/therealoise/bean-disease-dataset/code)
2. [AI Lab Makerere iBean Dataset](https://github.com/AI-Lab-Makerere/ibean/)

## Installation
To run this project, you need to install the required libraries. You can install them using the following command:

```bash
pip install -r requirements.txt
```

Make sure you have TensorFlow, Keras, OpenCV, and other necessary libraries installed.

## Usage
Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/bean-leaf-disease-detection.git
cd bean-leaf-disease-detection
```

### Model Training
To train the model, run the `bean_leaf_resnet50.ipynb` notebook. This notebook will guide you through the process of loading the datasets, preprocessing the images, and training the ResNet50 model.

### Real-Time Detection
There are two scripts provided for real-time detection using a webcam:
1. `modelVideo.py`
2. `modelVideoCapture.py`

Both scripts perform the same function but in slightly different ways. To run any of the scripts, use the following command:

```bash
python modelVideo.py
```

or

```bash
python modelVideoCapture.py
```

The script will open the webcam, process the video frames, and display the classification results in real-time.

## Model Training
The `bean_leaf_resnet50.ipynb` notebook contains the code for training the model. It uses the ResNet50 architecture pre-trained on ImageNet. The steps include:
- Loading the datasets
- Preprocessing the images
- Defining the model architecture
- Training the model
- Evaluating the model

### Example:
```python
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

# Define the model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=[EarlyStopping(monitor='val_accuracy', patience=5)])

# Save the model
model.save('saved_model/my_model.h5')
```

## Real-Time Detection
The real-time detection scripts use OpenCV to capture video from the webcam and the trained model to classify the frames. The steps include:
- Capturing video frames
- Preprocessing the frames
- Making predictions using the trained model
- Displaying the results on the video feed

### Example:
```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input

# Load the model
model = load_model('saved_model/my_model.h5', custom_objects={'preprocess_input': preprocess_input})

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    img = cv2.resize(frame, (224, 224))
    img = preprocess_input(img.astype("float32"))
    img = np.expand_dims(img, axis=0)

    # Make predictions
    preds = model.predict(img)
    class_idx = np.argmax(preds, axis=1)[0]
    class_name = class_names[class_idx]
    confidence = preds[0][class_idx]

    # Display the results
    cv2.putText(frame, f'{class_name}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Leaf Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

## Results
The model achieves good accuracy in classifying bean leaf diseases. Here are some example results:

- Confusion Matrix
- Classification Report
- Accuracy and Loss Curves

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License
This project is licensed under the MIT License.

---
