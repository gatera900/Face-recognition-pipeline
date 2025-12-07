# Face Recognition Pipeline - AI Without ML

A complete face recognition system using **MediaPipe** for face detection and **LBPH (Local Binary Patterns Histograms)** for classical feature-based recognition.

## 🎯 Project Overview

This project demonstrates a two-stage face recognition pipeline:
1. **Detection Stage**: MediaPipe detects faces in real-time video
2. **Recognition Stage**: LBPH classifier identifies detected faces

## 📁 Project Structure

```
face-recognition-project/
├── capture_faces.py      # Script to collect face images
├── train_model.py        # Script to train LBPH model
├── predict_faces.py      # Script for real-time recognition
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── dataset/             # Training images (created after capture)
│   ├── person1/
│   ├── person2/
│   └── ...
└── models/              # Trained model files (created after training)
    ├── lbph_model.yml
    └── label_map.json
```

## 🚀 Installation

### Prerequisites
- Python 3.8, 3.9, 3.10, 3.11, 3.12
- Webcam
- pip (Python package manager)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Face-Recognition-Pipeline.git
   cd Face-Recognition-Pipeline
   ```
   
   Or download as ZIP and extract.

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage Guide

### Step 1: Capture Training Data

Collect face images for each person you want the system to recognize.

```bash
python capture_faces.py
```

**Instructions:**
- Enter the person's name when prompted
- Position your face in front of the webcam
- Press **SPACE** to capture each image
- Capture at least 100 images with varied:
  - Head poses (left, right, up, down)
  - Lighting conditions
  - Expressions
  - Distances from camera
- Press **'q'** to quit early if needed

**Repeat this step for at least 2 different people.**

### Step 2: Train the Model

Train the LBPH recognizer on the collected images.

```bash
python train_model.py
```

This will:
- Load all images from the `dataset/` directory
- Train the LBPH face recognizer
- Save the model to `models/lbph_model.yml`
- Save the label mapping to `models/label_map.json`

### Step 3: Run Real-time Recognition

Test the trained model with live webcam feed.

```bash
python predict_faces.py
```

**Controls:**
- **'q'** - Quit the application
- **'s'** - Save a screenshot

The system will:
- Detect faces using MediaPipe
- Recognize faces using the trained LBPH model
- Display name and confidence score for each face
- Show green boxes for recognized faces
- Show red boxes for unknown faces

## 🔧 How It Works

### 1. Face Detection (MediaPipe)
- **Model**: MediaPipe Face Detection
- **Purpose**: Locates faces in images/video frames
- **Output**: Bounding box coordinates for each detected face

### 2. Face Recognition (LBPH)
- **Algorithm**: Local Binary Patterns Histograms
- **How it works**:
  - Divides face into small regions
  - Computes LBP patterns for each region
  - Creates histogram for each region
  - Concatenates histograms into a feature vector
  - Compares feature vectors using chi-square distance

### 3. Pipeline Flow
```
Camera Frame → MediaPipe Detection → Crop Face ROI → 
Convert to Grayscale → Resize → LBPH Prediction → Display Result
```

## ⚙️ Configuration

### Confidence Threshold
- **Default**: 70
- **Lower values**: Stricter matching (fewer false positives)
- **Higher values**: Looser matching (more false negatives)
- Adjust in `predict_faces.py` or when prompted

### LBPH Parameters
Located in `train_model.py`:
- **radius**: 1 (neighborhood size)
- **neighbors**: 8 (sample points)
- **grid_x**: 8 (horizontal cells)
- **grid_y**: 8 (vertical cells)

## 📊 Performance Tips

1. **Data Quality**
   - Capture 100+ images per person
   - Vary lighting, angles, and expressions
   - Ensure good image quality

2. **Environment**
   - Use consistent lighting during recognition
   - Maintain reasonable distance from camera
   - Avoid extreme angles

3. **Model Tuning**
   - Adjust confidence threshold based on accuracy
   - Experiment with LBPH parameters for better results

## 🐛 Troubleshooting

### Webcam not opening
- Check if another application is using the webcam
- Try changing camera index in code: `cv2.VideoCapture(1)`

### Poor recognition accuracy
- Collect more training images (150-200 per person)
- Ensure varied training data
- Adjust confidence threshold
- Check lighting conditions

### Model file not found
- Ensure you ran `train_model.py` successfully
- Check if `models/` directory contains the .yml file

## 🎬 Demonstration Guide: Detection + Recognition of At Least 2 People

### Quick Start for Demo

To demonstrate detection and recognition of at least 2 people:

1. **Capture faces for Person 1:**
   ```bash
   python capture-faces.py
   # Enter name: Person1 (or any name)
   # Capture 100+ images with varied poses and expressions
   ```

2. **Capture faces for Person 2:**
   ```bash
   python capture-faces.py
   # Enter name: Person2 (or any name)
   # Capture 100+ images with varied poses and expressions
   ```

3. **Train the model:**
   ```bash
   python train-model.py
   # This will train on both Person1 and Person2
   ```

4. **Run recognition demo:**
   ```bash
   python predict-faces.py
   # Enter confidence threshold (default: 70)
   ```

### Creating a Demo Video

For the video demonstration (20-40 seconds):

1. **Setup:**
   - Have both Person1 and Person2 ready
   - Ensure good lighting
   - Position camera to capture both faces

2. **Recording:**
   - Start `predict-faces.py`
   - Have Person1 appear first (should show green box with name)
   - Have Person2 appear (should show green box with their name)
   - Show both people together (both should be recognized simultaneously)
   - Press 's' to save screenshots during the demo

3. **Expected Output:**
   - Green bounding boxes around recognized faces
   - Name labels with confidence scores (e.g., "Person1 (45)", "Person2 (52)")
   - Red boxes for unknown faces (if any)
   - Real-time detection and recognition

### Demo Checklist

- ✅ At least 2 different people captured in dataset
- ✅ Model trained successfully (check `models/` folder)
- ✅ Both people recognized with green boxes
- ✅ Names and confidence scores displayed
- ✅ Video shows smooth detection and recognition

## 📝 Assignment Deliverables

✅ **Video Demonstration** (20-40 seconds)
- Shows detection of at least 2 different people
- Displays recognition with names and confidence scores
- Demonstrates real-time performance

✅ **GitHub Repository**
- All code files included (`capture-faces.py`, `train-model.py`, `predict-faces.py`)
- `models/` folder with trained model (`lbph_model.yml`, `label_map.json`)
- Clear README with setup and usage instructions
- Proper `.gitignore` file

✅ **Code Components**
- `capture-faces.py` - Data collection script
- `train-model.py` - Model training script
- `predict-faces.py` - Real-time prediction script

## 🎓 Learning Objectives

This project demonstrates:
- **Face Detection**: Using MediaPipe's pre-trained models
- **Feature Extraction**: LBPH algorithm for face representation
- **Classical ML**: Non-neural network approach to recognition
- **Pipeline Integration**: Combining detection and recognition stages
- **Real-time Processing**: Efficient webcam-based inference

## 📚 References

- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html)
- [OpenCV LBPH Face Recognizer](https://docs.opencv.org/4.x/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html)
- [Local Binary Patterns Paper](https://ieeexplore.ieee.org/document/1017623)

## 📦 Repository Contents

This repository includes:
- ✅ Complete source code (Python scripts)
- ✅ Requirements file with all dependencies
- ✅ Comprehensive README with instructions
- ✅ Trained model files (in `models/` directory)
- ✅ Proper `.gitignore` configuration

## 🔗 GitHub Setup

To push this project to GitHub:

1. **Initialize git repository** (if not already done):
   ```bash
   git init
   ```

2. **Add all files:**
   ```bash
   git add .
   ```

3. **Commit:**
   ```bash
   git commit -m "Initial commit: Face Recognition Pipeline with MediaPipe and LBPH"
   ```

4. **Create repository on GitHub** and push:
   ```bash
   git remote add origin https://github.com/yourusername/Face-Recognition-Pipeline.git
   git branch -M main
   git push -u origin main
   ```

**Note:** The `.gitignore` file is configured to exclude dataset images but keep the folder structure. Model files are included by default (uncomment in `.gitignore` if you want to exclude them).

## 📸 Screenshots

Screenshots can be saved during recognition by pressing 's' key. They will be saved as `screenshot_0.jpg`, `screenshot_1.jpg`, etc.

## 👨‍💻 Author

**GANZA Chael**

RWANDA CODING ACADEMY

Date: 7-12-2025

## 📄 License

This project is for educational purposes.

---