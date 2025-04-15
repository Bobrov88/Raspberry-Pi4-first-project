import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(data_dir, img_size=(100, 100)):
    """
    Load and preprocess images from the dataset directory
    Args:
        data_dir: Directory containing fruit images
        img_size: Target size for images (width, height)
    Returns:
        X: Preprocessed images
        y: Labels
        class_names: List of class names
    """
    images = []
    labels = []
    class_names = []
    
    # Iterate through each class folder
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)
            class_idx = len(class_names) - 1
            
            # Load each image in the class folder
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    # Read and resize image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    
                    # Normalize pixel values
                    img = img.astype('float32') / 255.0
                    
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels), class_names

def create_model(num_classes, input_shape=(100, 100, 3)):
    """
    Create a CNN model for fruit classification
    Args:
        num_classes: Number of fruit classes
        input_shape: Shape of input images (height, width, channels)
    Returns:
        model: Compiled TensorFlow model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def convert_to_tflite(model, class_names):
    """
    Convert the model to TFLite format
    Args:
        model: Trained TensorFlow model
        class_names: List of class names
    """
    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open('fruit_recognition_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # Save class names
    with open('class_names.txt', 'w') as f:
        f.write('\n'.join(class_names))

def main():
    # Path to your dataset
    data_dir = "fruits_train_3_class"
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, class_names = load_and_preprocess_data(data_dir)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    print("Creating and training model...")
    model = create_model(len(class_names))
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Train the model
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Convert and save the model in TFLite format
    print("Converting model to TFLite format...")
    convert_to_tflite(model, class_names)
    print("Model saved as 'fruit_recognition_model.tflite'")

if __name__ == "__main__":
    main() 