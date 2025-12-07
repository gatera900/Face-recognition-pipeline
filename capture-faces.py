import cv2
import mediapipe as mp
import os
import time

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def capture_faces(person_name, num_images=100):
    """
    Capture face images for a specific person
    
    Args:
        person_name: Name of the person
        num_images: Number of images to capture (default: 100)
    """
    # Create directory for this person
    dataset_dir = f"dataset/{person_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    count = 0
    print(f"\nCapturing images for {person_name}")
    print(f"Press SPACE to capture image ({num_images} needed)")
    print("Press 'q' to quit early")
    
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = face_detection.process(rgb_frame)
            
            # Draw face detection
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)
                    
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
                    
                    # Display counter
                    cv2.putText(frame, f"Captured: {count}/{num_images}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Person: {person_name}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Press SPACE to capture", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected!", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 0, 255), 2)
            
            cv2.imshow('Capture Faces', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capture image on SPACE press
            if key == ord(' ') and results.detections:
                # Get the first detected face
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Ensure coordinates are valid
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width > 0 and height > 0:
                    # Crop face region
                    face_roi = frame[y:y+height, x:x+width]
                    
                    # Resize to standard size
                    face_roi = cv2.resize(face_roi, (200, 200))
                    
                    # Save image
                    img_path = os.path.join(dataset_dir, f"{person_name}_{count}.jpg")
                    cv2.imwrite(img_path, face_roi)
                    count += 1
                    print(f"Captured image {count}/{num_images}")
                    
                    # Brief pause
                    time.sleep(0.1)
            
            # Quit on 'q' press
            elif key == ord('q'):
                print("\nCapture cancelled by user")
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nCapture complete! Saved {count} images for {person_name}")
    print(f"Images saved in: {dataset_dir}")

if __name__ == "__main__":
    print("=" * 50)
    print("Face Capture System - AI Without ML")
    print("=" * 50)
    
    person_name = input("\nEnter person's name: ").strip()
    
    if not person_name:
        print("Error: Name cannot be empty")
        exit(1)
    
    try:
        num_images = int(input("Enter number of images to capture (default 100): ") or "100")
    except ValueError:
        num_images = 100
    
    capture_faces(person_name, num_images)