**Brain MRI Tumor Classification**

**Overview**

This project is a deep learning-based classification model to detect brain tumors from MRI images. It utilizes the ResNet50 architecture for feature extraction and fine-tuning. The dataset used consists of brain MRI images labeled as "Tumor" and "No Tumor."

**Dataset**

The dataset used for training and evaluation is sourced from Kaggle:
Brain MRI Images for Brain Tumor Detection

**Technologies Used**

Python

TensorFlow / Keras

OpenCV

NumPy / Pandas

Matplotlib

Scikit-learn

Installation

**Clone the Repository**

git clone https://github.com/yourusername/brainMRIclassify.git
cd brainMRIclassify

**Install Dependencies**

Ensure you have Python 3.8+ installed. Then, install required packages:

pip install -r requirements.txt

**Model Training**

Run the Jupyter Notebook to train the model:

jupyter notebook brainMRIclassify.ipynb

**Model Architecture**

The project uses a pretrained ResNet50 model with additional dense layers for classification:

Input: 224x224 RGB MRI images

Convolutional feature extraction using ResNet50 (pretrained on ImageNet)

Fully connected layers with ReLU activation

Output layer with softmax activation for binary classification (Tumor / No Tumor)

**Data Preprocessing**

Images are resized to 224x224

Normalization is applied (pixel values scaled between 0 and 1)

Data augmentation includes random rotation, translation, flipping, contrast adjustments, and noise addition.

**Training & Evaluation**

Optimizer: Adam

Loss Function: Categorical Crossentropy

Batch Size: 32

Epochs: 100 (with early stopping and learning rate reduction callbacks)

Train/Test Split: 80/20

**Model Performance**

The model's performance is evaluated using accuracy and loss curves. Predictions are visualized with true and predicted labels.

Results

Test Accuracy: 82.35%

Test Loss: Reported in the output logs

Model predictions are visualized using sample images.

**Usage**

To use the trained model for predictions on new MRI scans:

import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("brain_tumor_model.h5")

def predict_tumor(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return "Tumor" if np.argmax(prediction) == 1 else "No Tumor"

result = predict_tumor("test_image.jpg")
print("Prediction:", result)

**Contributing**

Feel free to contribute by opening issues or submitting pull requests.
