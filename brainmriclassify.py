# -*- coding: utf-8 -*-
"""brainMRIclassify.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mJ3Lt-7OldHe987muLDH-F9v7Ew7hlko
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")

print("Path to dataset files:", path)

# Import Librarys
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(0)
tf.random.set_seed(42)
# Read & Resize imegs
file_name = "/root/.cache/kagglehub/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/versions/1"

X = []
y = []

yes_path = os.path.join(file_name, "yes")
for img_name in tqdm(os.listdir(yes_path)):
    img_path = os.path.join(yes_path, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        X.append(img)
        y.append(1)

no_path = os.path.join(file_name, "no")
for img_name in tqdm(os.listdir(no_path)):
    img_path = os.path.join(no_path, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        X.append(img)
        y.append(0)

X = np.array(X)
y = np.array(y)

print(f"Total images: {len(X)}, Total labels: {len(y)}")

#normalization for X
X = X / 255.0
# split Data to Train & Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#x :image, y: label

# Visualize the training dataset
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i])
    plt.title("Tumor" if y_train[i] == 1 else "No Tumor")
    plt.axis("off")
plt.show()

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
# Read Img From Training Data
plt.imshow((X_train[0] * 255).astype("uint8"))
plt.axis("off")
plt.show()

from tensorflow.keras.utils import to_categorical

# Convert y_train and y_test to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
# Batches Of Data
# Split Data to Batchs
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
# Data Augmentation
# Layers of Augmentation
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor = 0.15),
        layers.RandomTranslation(height_factor = 0.1, width_factor = 0.1),
        layers.RandomFlip(),
        layers.GaussianNoise(stddev = 0.09),
        layers.RandomContrast(factor = 0.1),
    ],
    name = "img_augmentation"
)
# Get the class names
class_names = np.unique(y_train)
print(class_names)
# [0. 1.]
# Show Augmentation effect
# View the augmentations
plt.figure(figsize=(10, 10))
for image, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = img_augmentation(tf.expand_dims(image[0], axis=0))
        plt.imshow((aug_img[0].numpy() * 255).astype("uint8"))
        plt.title("Tumor" if np.argmax(labels[0].numpy()) == 1 else "No Tumor")
        plt.axis("off")
plt.show()

# Build the Model
from keras.applications import resnet50

# Define input shape
img_rows, img_cols = 224, 224

# Load the ResNet50 model pre-trained on ImageNet
resnet = resnet50.ResNet50(weights='imagenet',
                           include_top=False,
                           input_shape=(img_rows, img_cols, 3))

# Here we freeze the last 4 layers
# Layers are set to trainable as True by default
for layer in resnet.layers:
    layer.trainable = False

# print our layers
for (i,layer) in enumerate(resnet.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
    def lw(bottom_model, num_classes):
        """Creates the top or head of the model that will be
        placed on top of the bottom layers"""

        top_model = bottom_model.output
        top_model = GlobalAveragePooling2D()(top_model)
        top_model = Dense(1024, activation='relu')(top_model)
        top_model = Dense(1024, activation='relu')(top_model)
        top_model = Dense(512, activation='relu')(top_model)
        top_model = Dense(num_classes, activation='softmax')(top_model)

        return top_model

# Import necessary libraries
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
# Edit on RESNET-50 Model
# Define the top layer of the model
def lw(bottom_model, num_classes):
    """Defines the top or head of the model"""
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

num_classes = 2

# Loa base resnet50 model without top layers
resnet = resnet50.ResNet50(weights='imagenet',
                           include_top=False,
                           input_shape=(img_rows, img_cols, 3))

# Freeze the layers in VGG16
for layer in resnet.layers:
    layer.trainable = False

# Add top layer to the VGG16 model
FC_Head = lw(resnet, num_classes)

# final model by connecting VGG16 model and top layer
model = Model(inputs=resnet.input, outputs=FC_Head)
print(model.summary())
# Model: "functional_1"

#  Caching and Prefetching (Optimization)
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)

# Define Callbacks
lr_callback = callbacks.ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.1, patience = 10)
stop_callback = callbacks.EarlyStopping(monitor = 'val_accuracy', patience = 8)



# Compile & Train Model
# EPOCHS=200

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, callbacks = [lr_callback, stop_callback],
                    epochs = 100, verbose = 1, validation_data=(X_test, y_test), initial_epoch=0)

# Show Train & Validation Accuracy
# %matplotlib inline
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_ds}')
print(f'Test Accuracy: {test_accuracy}')

y_pred_prob = model.predict(X_test)  # Get predicted probabilities
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels


import numpy as np

y_test = np.argmax(y_test, axis=1)  # Converts [1.0, 0.0] → 0 (class 0)and [0.0, 1.0] → 1(class 1) as they are one-hot encoded

# Display images with predicted and true labels
def plot_images(images, true_labels, pred_labels, num_images=10):
    plt.figure(figsize=(20,20))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i], cmap='gray')  # Adjust shape if needed for your data
        plt.title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}")
        plt.axis('off')
    plt.show()

# Display the first 10 images from the test set with true and predicted labels
plot_images(X_test, y_test, y_pred, num_images=10)

