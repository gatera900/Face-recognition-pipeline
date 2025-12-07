import cv2
import os
import json
import numpy as np

def load_training_data(dataset_dir="dataset"):
    """
    Load face images and labels from dataset directory
    
    Returns:
        faces: List of face images (grayscale)
        labels: List of corresponding numeric labels
        label_map: Dictionary mapping numeric labels to names
    """
    faces = []
    labels = []
    label_map = {}
    current_label = 0
    
    print("Loading training data...")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        return None, None, None
    
    # Iterate through each person's folder
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        # Assign label to this person
        label_map[current_label] = person_name
        print(f"Loading images for: {person_name} (Label: {current_label})")
        
        # Load all images for this person
        image_count = 0
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            
            # Read image
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Warning: Could not read {image_path}")
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Add to training data
            faces.append(gray)
            labels.append(current_label)
            image_count += 1
        
        print(f"  Loaded {image_count} images")
        current_label += 1
    
    print(f"\nTotal images loaded: {len(faces)}")
    print(f"Total persons: {len(label_map)}")
    
    return faces, labels, label_map

def train_lbph_model(dataset_dir="dataset", model_dir="models"):
    """
    Train LBPH face recognizer and save model
    
    Args:
        dataset_dir: Directory containing face images
        model_dir: Directory to save trained model
    """
    # Load training data
    faces, labels, label_map = load_training_data(dataset_dir)
    
    if faces is None or len(faces) == 0:
        print("Error: No training data found!")
        return
    
    # Convert to numpy arrays
    faces = np.array(faces)
    labels = np.array(labels)
    
    print("\nTraining LBPH Face Recognizer...")
    
    # Create LBPH Face Recognizer
    # Parameters:
    # - radius: 1 (neighborhood size)
    # - neighbors: 8 (number of sample points)
    # - grid_x: 8 (number of cells in horizontal direction)
    # - grid_y: 8 (number of cells in vertical direction)
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )
    
    # Train the recognizer
    recognizer.train(faces, labels)
    
    print("Training complete!")
    
    # Create models directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, "lbph_model.yml")
    recognizer.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save label map
    label_map_path = os.path.join(model_dir, "label_map.json")
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"Label map saved to: {label_map_path}")
    
    print("\n" + "=" * 50)
    print("Training Summary:")
    print("=" * 50)
    print(f"Total training images: {len(faces)}")
    print(f"Number of persons: {len(label_map)}")
    print(f"Label mapping:")
    for label, name in label_map.items():
        print(f"  {label}: {name}")
    print("=" * 50)

if __name__ == "__main__":
    print("=" * 50)
    print("LBPH Model Training - AI Without ML")
    print("=" * 50)
    print()
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("Error: 'dataset' directory not found!")
        print("Please run capture_faces.py first to collect training data.")
        exit(1)
    
    # Check if dataset has any data
    persons = [d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))]
    if len(persons) < 2:
        print("Warning: You need at least 2 different persons for face recognition!")
        print(f"Currently found: {len(persons)} person(s)")
        if len(persons) == 1:
            print(f"  - {persons[0]}")
        
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(1)
    
    # Train model
    train_lbph_model()
    
    print("\nTraining complete! You can now run predict_faces.py")