import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "/Users/saniyaladanavar/Desktop/mini project/asl dataset"  # Replace with the path to your dataset

def load_images(dataset_path):
    images = []
    labels = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64))  # Resize each image to 64x64
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

images, labels = load_images(DATASET_PATH)

# Load images and labels
images, labels = load_images(DATASET_PATH)

# Normalize images
images = images / 255.0  # Normalize pixel values to [0, 1]

# Get unique labels
unique_labels = np.unique(labels)

# Convert labels to numerical representation
labels_numerical = np.array([np.where(unique_labels == label)[0][0] for label in labels])

# Convert labels to categorical
labels_categorical = to_categorical(labels_numerical)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Optionally, save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)