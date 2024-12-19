import numpy as np
from skimage.io import imread
from skimage.transform import resize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from glob import glob
import os
import cv2

# Function to load images and convert them to flattened feature vectors
def load_images(image_paths, image_size):
    images = []
    for img_path in image_paths:
        img = imread(img_path)
        img_resized = resize(img, image_size, anti_aliasing=True)
        img_flattened = img_resized.flatten()
        images.append(img_flattened)
    return np.array(images)

# Parameters
image_dir = 'dataset_exp'
image_dest='dataset_oversamp'
image_size = (64, 64)  # Resize images to 64x64
n_channels = 3  # Assuming RGB images
class_labels = {'Fitness1':0 , 'Eglence':1, 'Gaming2':2, 'Cola Ferahlatici':3, 'Tonic Ferahlatici':4, 'mixed':5, 'Winter':6, 
                'Lemon Ferahlatici':7, 'Iftar':8, 'Driving1':9, 'Driving2':10, 'Gaming1':11, 'Cola Orange':12, 'Driving3':13, 'All Terrain':14,
                'Gaming3':15, 'Orange Ferahlatici':16}

# Load your image paths and labels
image_paths = []
labels = []

for class_name, label in class_labels.items():
    class_files = glob(os.path.join(image_dir, class_name, '*.jpg'))
    image_paths.extend(class_files)
    labels.extend([label] * len(class_files))

# Convert to numpy arrays
labels = np.array(labels)

# Load and preprocess images
X = load_images(image_paths, image_size)

# Split the dataset (optional, to create a test set)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(k_neighbors=2, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Reshape the resampled feature vectors back into images
X_train_resampled_images = X_train_resampled.reshape(-1, *image_size, n_channels)


for i, (resampled, y) in enumerate(zip(X_train_resampled_images, y_train_resampled)):
    label = [k for k, v in class_labels.items() if v == y][0]
    print(label)
    os.makedirs(os.path.join(image_dest, label), exist_ok=True)
    cv2.imwrite(os.path.join(image_dest, label, str(i)+".jpg"),resampled*255)

# Verify shapes
print(f'Original dataset shape: {X_train.shape}, {y_train.shape}')
print(f'Resampled dataset shape: {X_train_resampled.shape}, {y_train_resampled.shape}')
print(f'Reshaped resampled images shape: {X_train_resampled_images.shape}')
