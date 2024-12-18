import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_organized_images(base_dir, target_size=(224, 224)):
    """
    Loads images and labels from a structured directory where folder names are labels.
    Args:
        base_dir (str): Path to the root directory containing subdirectories with images.
        target_size (tuple): Dimensions to resize images (default: 224x224).
    Returns:
        tuple: Arrays of images and labels.
    """
    images = []
    labels = []
    
    # Traverse each subdirectory
    for label_name in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label_name)
        
        # Ensure the label folder is valid
        if os.path.isdir(label_path):
            try:
                # Replace commas with periods to parse as float
                label = float(label_name.replace(",", "."))
            except ValueError:
                print(f"Skipping folder {label_name} - not a valid numeric label.")
                continue
            
            # Process each image in the subdirectory
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                try:
                    # Load image and preprocess
                    img = Image.open(image_path).convert("RGB")  # Ensure RGB
                    img = img.resize(target_size)  # Resize to target size
                    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
                    
                    # Append image and label to the dataset
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    
    # Convert lists to numpy arrays
    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="float32")
    return images, labels

# Path to the organized dataset directory
dataset_directory = "/home/kerem/Desktop/contakbizim/data/organized_images"

# Load and preprocess the dataset
X, y = load_organized_images(dataset_directory)

# Output the dataset shapes
print(f"Loaded {len(X)} images with shape: {X[0].shape}")
print(f"Labels shape: {y.shape}")

np.save('X.npy', X)
np.save('y.npy', y)

