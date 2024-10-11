import os
import cv2
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define the model architecture
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

# Build and compile the model
model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the model weights
weights_path = '/content/model_weights.h5'
model.load_weights(weights_path)

# Load the Haar cascade for face detection
haar_cascade_path = '/content/haarcascade_frontalface_default.xml'

if not os.path.exists(haar_cascade_path):
    raise FileNotFoundError(f"Haar cascade file not found at {haar_cascade_path}")

face_cascade = cv2.CascadeClassifier(haar_cascade_path)

if face_cascade.empty():
    raise IOError("Failed to load Haar cascade file. Please check the path and the file's existence.")

# Function to preprocess an image for model input
def preprocess_image(image):
    face = cv2.resize(image, (224, 224))
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    face = face / 255.0  # Normalize to [0, 1]
    return face

# Function to perform inference on an image file
def process_image(input_image, output_image, y_true=None):
    print(f"Processing image: {input_image}")
    
    # Read the image
    image = cv2.imread(input_image)
    if image is None:
        print(f"Error: Could not open image {input_image}")
        return

    # Detect faces in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    face_detected_count = len(faces)
    y_pred = []

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        preprocessed_face = preprocess_image(face)
        prediction = model.predict(preprocessed_face)
        print(f"Predictions: {prediction}")  # Debugging step to print raw predictions
        
        # Swapped label mapping
        label = 'No Mask' if np.argmax(prediction) == 1 else 'Mask'

        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Track predictions
        y_pred.append(np.argmax(prediction))

    # Save the output image
    cv2.imwrite(output_image, image)
    print(f"Finished processing image: {input_image}")
    print(f"Faces detected: {face_detected_count}")
    print(f"Output image saved to: {output_image}")

    return y_pred

# Perform inference on sample images and track predictions
images = [
    ('/content/0316.jpg', 'output_image1.jpg', 0),  # 0 for Mask, 1 for No Mask
    ('/content/0017.jpg', 'output_image2.jpg', 1),
    ('/content/0193.jpg', 'output_image3.jpg', 0)
]

y_true = []  # Ground truth labels
y_pred = []  # Model predictions

for input_image, output_image, true_label in images:
    if os.path.exists(input_image):
        y_pred_image = process_image(input_image, output_image)
        y_true.extend([true_label] * len(y_pred_image))  # Append true label for all detected faces
        y_pred.extend(y_pred_image)  # Collect predictions
    else:
        print(f"Error: Input image {input_image} does not exist")

# Display Confusion Matrix
if len(y_true) > 0 and len(y_pred) > 0:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Mask', 'No Mask'], yticklabels=['Mask', 'No Mask'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

print("Image processing completed.")
