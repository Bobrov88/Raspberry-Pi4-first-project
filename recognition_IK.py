import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Configuration
img_height, img_width = 32, 32
confidence_threshold = 0.3  # Lowered from 0.7 to 0.3
min_detection_interval = 0.5  # Reduced from 1.0 to 0.5 seconds

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

def preprocess_image(image, target_size=(32, 32)):
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

def main():
    # Check if model and class names files exist
    model_path = 'fruit_recognition_model.tflite'
    class_names_path = 'class_names.txt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    if not os.path.exists(class_names_path):
        print(f"Error: Class names file not found: {class_names_path}")
        return
    
    # Load the TFLite model
    print(f"Loading model from {model_path}...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print model details for debugging
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Load class names
    class_names = load_class_names(class_names_path)
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Initialize webcam
    print("Initializing webcam...")
    
    # Open the camera with DirectShow backend
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Try to read a frame to confirm the camera is working
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read frame from camera. Trying alternative method...")
        cap.release()
        
        # Try with default backend
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read frame with alternative method.")
            return
    
    print(f"Successfully opened camera. Frame shape: {frame.shape}")
    
    # Save a test frame
    cv2.imwrite("camera_test.jpg", frame)
    print("Saved test frame to camera_test.jpg")
    
    # Variables for tracking detections
    last_detection_time = 0
    last_detected_class = None
    detection_count = 0
    frame_count = 0
    debug_interval = 30  # Print debug info every 30 frames
    
    print("Press 'q' to quit")
    print(f"Using confidence threshold: {confidence_threshold}")
    
    try:
        # Create a test window
        cv2.namedWindow('Fruit Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Fruit Recognition', 640, 480)
        
        # Display a test image first
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Camera Test", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Fruit Recognition', test_image)
        cv2.waitKey(1000)  # Wait for 1 second
        
        # Main loop
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to capture image")
                break
            
            frame_count += 1
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Preprocess image
            processed_image = preprocess_image(frame, (img_height, img_width))
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], processed_image)
            
            # Run inference
            interpreter.invoke()
            
            # Get prediction
            predictions = interpreter.get_tensor(output_details[0]['index'])
            
            # Get current time
            current_time = time.time()
            
            # Find the predicted class
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Print debug info periodically
            if frame_count % debug_interval == 0:
                print(f"\nFrame #{frame_count} - Top predictions:")
                # Sort predictions by confidence
                sorted_indices = np.argsort(predictions[0])[::-1]
                for idx in sorted_indices:
                    print(f"{class_names[idx]}: {predictions[0][idx]:.3f}")
            
            # Check if confidence is above threshold and enough time has passed since last detection
            if confidence > confidence_threshold and (current_time - last_detection_time) > min_detection_interval:
                # Check if this is a new detection or the same as last time
                if last_detected_class != predicted_class:
                    detection_count += 1
                    last_detection_time = current_time
                    last_detected_class = predicted_class
                    
                    # Print detection with probabilities for all classes
                    print(f"\nDetection #{detection_count}:")
                    for i, class_name in enumerate(class_names):
                        print(f"{class_name}: {predictions[0][i]:.3f}")
                    
                    # Add detection info to display frame
                    cv2.putText(display_frame, f"{class_names[predicted_class]}: {confidence:.2f}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add text to display frame
            cv2.putText(display_frame, f"Press 'q' to quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Fruit Recognition', display_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Release resources
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Recognition stopped")

if __name__ == "__main__":
    main() 