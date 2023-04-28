import tensorflow as tf
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
from PIL import Image
import numpy as np

def load_data(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(directory, filename))
            img_array = np.array(img)
            images.append(img_array)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels



def preprocess(images):
    processed_images = []
    for image in images:
        print(image.shape[3])
        if image.shape[3] == 1:
            # If the image is already grayscale, normalize and standardize the pixel values
            image = image / 255.0
            image = (image - image.mean()) / image.std()
        else:       
            # Convert the image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          
            # Scale the pixel values between 0 and 1
            image = image / 255.0
          
            # Normalize the pixel values to have zero mean and unit variance
            image = (image - image.mean()) / image.std()
          
        processed_images.append(image)
        
    return np.array(processed_images)

# Define the CNN model
model = tf.keras.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(480, 640, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the data and preprocess it
X_test = load_data("./output")
X_test = preprocess(X_test)
X_train = load_data("./source")
X_train = preprocess(X_train)




# Train the model
model.fit(X_train, epochs=10, validation_data=(X_test))