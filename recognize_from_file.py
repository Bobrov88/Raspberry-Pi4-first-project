import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import logging

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

def visualize_predictions(image, predictions, class_names, confidence_threshold=0.5):
    """
    Visualize predictions on the image
    Args:
        image: Original image
        predictions: Model predictions
        class_names: List of class names
        confidence_threshold: Threshold for displaying predictions
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_confidences = [predictions[0][i] for i in top_indices]
    
    # Display top 3 predictions
    for i, (class_name, confidence) in enumerate(zip(top_classes, top_confidences)):
        if confidence > confidence_threshold:
            text = f"{class_name}: {confidence:.3f}"
            y_position = 30 + i * 40
            cv2.putText(vis_image, text, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display all class probabilities
    logging.info("All class probabilities:")
    for i, prob in enumerate(predictions[0]):
        logging.info(f"{class_names[i]}: {prob:.3f}")
    
    return vis_image

def recognize_image(image_path, model_path, class_names_path, confidence_threshold=0.5):
    """
    Recognize fruit in a single image
    Args:
        image_path: Path to the image file
        model_path: Path to the TFLite model
        class_names_path: Path to the class names file
        confidence_threshold: Threshold for considering a detection valid
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
    logging.info(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Load and preprocess image
    logging.info(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return None
    
    # Display original image size
    logging.info(f"Original image size: {image.shape}")
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    
    # Run inference
    interpreter.invoke()
    
    # Get prediction
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Print only the probabilities for each class
    print("\nClass probabilities:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {predictions[0][i]:.3f}")
    
    return predictions

def process_directory(directory_path, model_path, class_names_path, output_dir=None, confidence_threshold=0.5):
    """
    Process all images in a directory
    Args:
        directory_path: Path to the directory containing images
        model_path: Path to the TFLite model
        class_names_path: Path to the class names file
        output_dir: Directory to save output images (optional)
        confidence_threshold: Threshold for considering a detection valid
    """
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(directory_path) 
                  if os.path.isfile(os.path.join(directory_path, f)) 
                  and os.path.splitext(f)[1].lower() in image_extensions]
    
    logging.info(f"Found {len(image_files)} images in {directory_path}")
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        logging.info(f"Processing {image_file}...")
        
        # Recognize image
        result = recognize_image(image_path, model_path, class_names_path, confidence_threshold)
        
        if result is not None:
            predictions = result
            
            # Save output image if output directory is specified
            if output_dir:
                output_path = os.path.join(output_dir, f"result_{image_file}")
                cv2.imwrite(output_path, visualize_predictions(cv2.imread(image_path), predictions, load_class_names(class_names_path)))
                logging.info(f"Saved result to {output_path}")

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
    
    # Process a single image
    image_path = input("Enter the path to an image file (or 'dir' to process a directory): ")
    
    if image_path.lower() == 'dir':
        directory_path = input("Enter the path to the directory containing images: ")
        output_dir = input("Enter the path to save output images (or leave empty): ")
        confidence_threshold = float(input("Enter confidence threshold (0.0-1.0, default 0.5): ") or 0.5)
        
        process_directory(directory_path, model_path, class_names_path, output_dir, confidence_threshold)
    else:
        confidence_threshold = float(input("Enter confidence threshold (0.0-1.0, default 0.5): ") or 0.5)
        
        recognize_image(image_path, model_path, class_names_path, confidence_threshold)

if __name__ == "__main__":
    main() 