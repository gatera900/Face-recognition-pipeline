import cv2
import mediapipe as mp
import json
import os

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def load_model(model_dir="models"):
    """
    Load trained LBPH model and label map
    
    Returns:
        recognizer: Trained LBPH recognizer
        label_map: Dictionary mapping labels to names
    """
    model_path = os.path.join(model_dir, "lbph_model.yml")
    label_map_path = os.path.join(model_dir, "label_map.json")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run train_model.py first!")
        return None, None
    
    if not os.path.exists(label_map_path):
        print(f"Error: Label map not found at {label_map_path}")
        return None, None
    
    # Load LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    
    # Load label map
    with open(label_map_path, 'r') as f:
        label_map_str = json.load(f)
        # Convert string keys to integers
        label_map = {int(k): v for k, v in label_map_str.items()}
    
    print("Model loaded successfully!")
    print(f"Recognized persons: {list(label_map.values())}")
    
    return recognizer, label_map

def predict_faces(confidence_threshold=70):
    """
    Real-time face detection and recognition
    
    Args:
        confidence_threshold: Confidence threshold for recognition (lower is better)
    """
    # Load model
    recognizer, label_map = load_model()
    
    if recognizer is None or label_map is None:
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nStarting face recognition...")
    print("Press 'q' to quit")
    print("Press 's' to save a screenshot")
    
    screenshot_count = 0
    
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = face_detection.process(rgb_frame)
            
            # Process detections
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)
                    
                    # Ensure coordinates are within frame
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width > 0 and height > 0:
                        # Extract face ROI
                        face_roi = frame[y:y+height, x:x+width]
                        
                        # Convert to grayscale
                        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        
                        # Resize to match training size
                        gray_face = cv2.resize(gray_face, (200, 200))
                        
                        # Predict
                        label, confidence = recognizer.predict(gray_face)
                        
                        # Get person name
                        if label in label_map and confidence < confidence_threshold:
                            name = label_map[label]
                            color = (0, 255, 0)  # Green for recognized
                            text = f"{name} ({int(confidence)})"
                        else:
                            name = "Unknown"
                            color = (0, 0, 255)  # Red for unknown
                            text = f"Unknown ({int(confidence)})"
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)
                        
                        # Draw label background
                        label_y = y - 10 if y > 30 else y + height + 25
                        cv2.rectangle(frame, (x, label_y-25), (x+width, label_y), color, -1)
                        
                        # Draw name and confidence
                        cv2.putText(frame, text, (x+5, label_y-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit | 's' to screenshot", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition - AI Without ML', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Quit on 'q'
            if key == ord('q'):
                break
            
            # Save screenshot on 's'
            elif key == ord('s'):
                screenshot_path = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
                screenshot_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nFace recognition stopped.")

if __name__ == "__main__":
    print("=" * 50)
    print("Face Recognition System - AI Without ML")
    print("=" * 50)
    print()
    
    # Check if models exist
    if not os.path.exists("models/lbph_model.yml"):
        print("Error: Trained model not found!")
        print("Please run train_model.py first to train the model.")
        exit(1)
    
    try:
        threshold = int(input("Enter confidence threshold (default 70, lower is stricter): ") or "70")
    except ValueError:
        threshold = 70
    
    print(f"\nUsing confidence threshold: {threshold}")
    print("(Lower values mean stricter matching)")
    
    predict_faces(confidence_threshold=threshold)