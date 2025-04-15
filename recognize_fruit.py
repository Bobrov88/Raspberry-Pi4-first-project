import cv2
import numpy as np
import tensorflow as tf
import time
import os
import logging
import sys
import grp
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Configuration
img_height, img_width = 100, 100  # Changed to match model input size
confidence_threshold = 0.8
min_detection_interval = 0.5
min_brightness = 30  # Minimum average brightness (0-255)
min_contrast = 20    # Minimum contrast (0-255)

def check_camera_devices():
    """Check available camera devices"""
    try:
        # Check video devices
        result = subprocess.run(['ls', '-l', '/dev/video*'], capture_output=True, text=True)
        logging.info("Available video devices:")
        logging.info(result.stdout)
        
        # Check USB devices
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        logging.info("USB devices:")
        logging.info(result.stdout)
        
        # Check camera modules
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        if 'uvcvideo' in result.stdout or 'bcm2835-v4l2' in result.stdout:
            logging.info("Camera modules are loaded")
        else:
            logging.warning("Camera modules might not be loaded")
            
    except Exception as e:
        logging.error(f"Error checking camera devices: {str(e)}")

def check_video_group():
    """Check if user is in video group"""
    try:
        video_group = grp.getgrnam('video')
        return os.geteuid() in video_group.gr_mem
    except KeyError:
        logging.warning("Video group not found")
        return False

def check_camera_permissions():
    """Check if we have proper camera permissions"""
    try:
        # Try different camera indices
        for camera_index in range(2):  # Try first two indices
            logging.info(f"Trying camera index {camera_index}")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    logging.info(f"Successfully opened camera {camera_index}")
                    cap.release()
                    return True
                cap.release()
        
        logging.error("Failed to open any camera. Check permissions and connections.")
        return False
    except Exception as e:
        logging.error(f"Camera permission error: {str(e)}")
        return False

def initialize_camera():
    """Initialize camera with proper settings for Raspberry Pi"""
    logging.info("Initializing camera...")
    
    # Try different camera indices and backends
    for camera_index in range(2):  # Try first two indices
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Default"),
            (cv2.CAP_V4L, "V4L")
        ]
        
        for backend, name in backends:
            try:
                logging.info(f"Trying camera {camera_index} with {name} backend...")
                cap = cv2.VideoCapture(camera_index + backend)
                
                if cap.isOpened():
                    # Set camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        logging.info(f"Successfully initialized camera {camera_index} with {name} backend")
                        return cap
                    
                cap.release()
            except Exception as e:
                logging.error(f"Error with camera {camera_index} and {name} backend: {str(e)}")
    
    raise RuntimeError("Failed to initialize any camera")

def load_class_names(filename):
    """Load class names from file"""
    try:
        with open(filename, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        logging.info(f"Loaded {len(class_names)} classes: {class_names}")
        return class_names
    except Exception as e:
        logging.error(f"Error loading class names: {str(e)}")
        raise

def check_image_quality(image):
    """Check if image meets quality standards"""
    # Convert to grayscale for brightness/contrast check
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate average brightness
    brightness = np.mean(gray)
    
    # Calculate contrast (standard deviation)
    contrast = np.std(gray)
    
    # Check if image is too dark or has too little contrast
    if brightness < min_brightness:
        logging.debug(f"Image too dark: brightness={brightness:.1f}")
        return False
    
    if contrast < min_contrast:
        logging.debug(f"Image has too little contrast: contrast={contrast:.1f}")
        return False
    
    return True

def preprocess_image(image, target_size=(100, 100)):
    """Preprocess image for model input"""
    # Check image quality first
    if not check_image_quality(image):
        return None
    
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
    try:
        # Check available camera devices
        check_camera_devices()
        
        # Check video group membership
        if not check_video_group():
            logging.warning("User is not in the video group. This may cause camera access issues.")
            logging.warning("To fix this, run: sudo usermod -a -G video $USER")
            logging.warning("Then log out and log back in.")
        
        # Check camera permissions
        if not check_camera_permissions():
            logging.error("Camera permission check failed.")
            logging.error("Please ensure you are in the video group and have proper permissions.")
            return
        
        # Check if model and class names files exist
        model_path = 'fruit_recognition_model.tflite'
        class_names_path = 'class_names.txt'
        
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            return
        
        if not os.path.exists(class_names_path):
            logging.error(f"Class names file not found: {class_names_path}")
            return
        
        # Load the TFLite model
        logging.info(f"Loading model from {model_path}...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Print model details for debugging
        logging.info(f"Input details: {input_details}")
        logging.info(f"Output details: {output_details}")
        
        # Load class names
        class_names = load_class_names(class_names_path)
        
        # Initialize camera
        cap = initialize_camera()
        
        # Variables for tracking detections
        last_detection_time = 0
        last_detected_class = None
        detection_count = 0
        frame_count = 0
        debug_interval = 30  # Print debug info every 30 frames
        
        logging.info("Press Ctrl+C to quit")
        logging.info(f"Using confidence threshold: {confidence_threshold}")
        logging.info(f"Minimum brightness: {min_brightness}")
        logging.info(f"Minimum contrast: {min_contrast}")
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.error("Failed to capture image")
                break
            
            frame_count += 1
            
            # Preprocess image
            processed_image = preprocess_image(frame, (img_height, img_width))
            
            # Skip frame if it doesn't meet quality standards
            if processed_image is None:
                continue
            
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
                logging.info(f"\nFrame #{frame_count} - Top predictions:")
                # Sort predictions by confidence
                sorted_indices = np.argsort(predictions[0])[::-1]
                for idx in sorted_indices:
                    logging.info(f"{class_names[idx]}: {predictions[0][idx]:.3f}")
            
            # Check if confidence is above threshold and enough time has passed since last detection
            if confidence > confidence_threshold and (current_time - last_detection_time) > min_detection_interval:
                # Check if this is a new detection or the same as last time
                if last_detected_class != predicted_class:
                    detection_count += 1
                    last_detection_time = current_time
                    last_detected_class = predicted_class
                    
                    # Print detection with probabilities for all classes
                    logging.info(f"\nDetection #{detection_count}:")
                    for i, class_name in enumerate(class_names):
                        logging.info(f"{class_name}: {predictions[0][i]:.3f}")
            
            # Add small delay to reduce CPU usage
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        logging.info("\nStopping recognition...")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    
    finally:
        # Release resources
        if 'cap' in locals() and cap is not None:
            cap.release()
        logging.info("Recognition stopped")

if __name__ == "__main__":
    main() 