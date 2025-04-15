import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import logging
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_class_names(filename):
    """
    Load class names from file
    Args:
        filename: Path to the file containing class names
    Returns:
        List of class names
    """
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image, target_size=(100, 100)):
    """
    Preprocess image for model input
    Args:
        image: Input image
        target_size: Target size for resizing
    Returns:
        Preprocessed image
    """
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Convert to RGB if necessary
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    image = image.astype('float32') / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def save_preprocessing_steps(image_path, output_dir, target_size=(100, 100)):
    """
    Save the preprocessing steps as images
    Args:
        image_path: Path to the image file
        output_dir: Directory to save output images
        target_size: Target size for resizing
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return
    
    # Save original image
    original_path = os.path.join(output_dir, "1_original.jpg")
    cv2.imwrite(original_path, image)
    logging.info(f"Saved original image to {original_path}")
    
    # Resize image
    resized = cv2.resize(image, target_size)
    resized_path = os.path.join(output_dir, "2_resized.jpg")
    cv2.imwrite(resized_path, resized)
    logging.info(f"Saved resized image to {resized_path}")
    
    # Convert to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb_path = os.path.join(output_dir, "3_rgb.jpg")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
    logging.info(f"Saved RGB image to {rgb_path}")
    
    # Normalize (convert to float and scale to 0-1)
    normalized = rgb.astype('float32') / 255.0
    # Convert back to uint8 for saving
    normalized_uint8 = (normalized * 255).astype('uint8')
    normalized_path = os.path.join(output_dir, "4_normalized.jpg")
    cv2.imwrite(normalized_path, cv2.cvtColor(normalized_uint8, cv2.COLOR_RGB2BGR))
    logging.info(f"Saved normalized image to {normalized_path}")
    
    logging.info(f"All preprocessing steps saved to {output_dir}")

def compare_with_training_data(image_path, training_dir, model_path, class_names_path, output_dir):
    """
    Compare an image with training data
    Args:
        image_path: Path to the image file
        training_dir: Directory containing training data
        model_path: Path to the TFLite model
        class_names_path: Path to the class names file
        output_dir: Directory to save output images
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the TFLite model
    logging.info(f"Loading model from {model_path}...")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load class names
    class_names = load_class_names(class_names_path)
    logging.info(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Load and preprocess image
    logging.info(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return
    
    # Save input image
    input_path = os.path.join(output_dir, "input_image.jpg")
    cv2.imwrite(input_path, image)
    logging.info(f"Saved input image to {input_path}")
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    
    # Run inference
    interpreter.invoke()
    
    # Get prediction
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Display predictions
    logging.info("Predictions:")
    for i, prob in enumerate(predictions[0]):
        logging.info(f"{class_names[i]}: {prob:.3f}")
    
    # Find the predicted class
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class]
    confidence = predictions[0][predicted_class]
    
    logging.info(f"Predicted class: {predicted_class_name} (Confidence: {confidence:.3f})")
    
    # Save training examples
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(training_dir, class_name)
        if os.path.isdir(class_dir):
            # Get a sample image from this class
            sample_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if sample_files:
                sample_path = os.path.join(class_dir, sample_files[0])
                sample_image = cv2.imread(sample_path)
                if sample_image is not None:
                    # Add text to the image
                    text = f"Training: {class_name} (Conf: {predictions[0][i]:.3f})"
                    cv2.putText(sample_image, text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Save the image
                    output_path = os.path.join(output_dir, f"training_{class_name}.jpg")
                    cv2.imwrite(output_path, sample_image)
                    logging.info(f"Saved training example for {class_name} to {output_path}")
    
    logging.info(f"All comparison images saved to {output_dir}")

def analyze_model(model_path, class_names_path):
    """
    Analyze the model architecture and parameters
    Args:
        model_path: Path to the TFLite model
        class_names_path: Path to the class names file
    """
    # Load the TFLite model
    logging.info(f"Loading model from {model_path}...")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load class names
    class_names = load_class_names(class_names_path)
    
    # Display model information
    logging.info("Model Information:")
    logging.info(f"Input shape: {input_details[0]['shape']}")
    logging.info(f"Input type: {input_details[0]['dtype']}")
    logging.info(f"Output shape: {output_details[0]['shape']}")
    logging.info(f"Output type: {output_details[0]['dtype']}")
    logging.info(f"Number of classes: {len(class_names)}")
    logging.info(f"Classes: {class_names}")
    
    # Create a dummy input
    dummy_input = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    
    # Run inference
    interpreter.invoke()
    
    # Get prediction
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Display dummy prediction
    logging.info("Dummy prediction (should be uniform distribution):")
    for i, prob in enumerate(predictions[0]):
        logging.info(f"{class_names[i]}: {prob:.3f}")

def main():
    # Paths
    model_path = 'fruit_recognition_model.tflite'
    class_names_path = 'class_names.txt'
    
    # Check if files exist
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return
    
    if not os.path.exists(class_names_path):
        logging.error(f"Class names file not found: {class_names_path}")
        return
    
    # Menu
    print("\nFruit Recognition Diagnostic Tool")
    print("1. Analyze model architecture")
    print("2. Visualize preprocessing steps")
    print("3. Compare with training data")
    print("4. Exit")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == '1':
        # Analyze model
        analyze_model(model_path, class_names_path)
    
    elif choice == '2':
        # Visualize preprocessing
        image_path = input("Enter the path to an image file: ")
        output_dir = input("Enter the path to save output images: ")
        save_preprocessing_steps(image_path, output_dir)
    
    elif choice == '3':
        # Compare with training data
        image_path = input("Enter the path to an image file: ")
        training_dir = input("Enter the path to the training data directory: ")
        output_dir = input("Enter the path to save output images: ")
        compare_with_training_data(image_path, training_dir, model_path, class_names_path, output_dir)
    
    elif choice == '4':
        print("Exiting...")
    
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main() 