# Demo Instructions: Detection + Recognition of At Least 2 People

This guide will help you create a demonstration video showing face detection and recognition of at least 2 different people.

## Prerequisites

- Webcam connected and working
- Python environment set up with all dependencies installed
- At least 2 people available for the demonstration

## Step-by-Step Demo Setup

### 1. Capture Training Data for Person 1

```bash
python capture-faces.py
```

**When prompted:**
- Enter name: `Person1` (or any name you prefer)
- Enter number of images: `100` (or more for better accuracy)
- Position Person 1 in front of the camera
- Press **SPACE** to capture each image
- Vary poses: look left, right, up, down, smile, neutral expression
- Capture images at different distances from camera
- Press **'q'** when done

**Expected output:**
- Images saved in `dataset/Person1/` directory
- 100+ images captured

### 2. Capture Training Data for Person 2

```bash
python capture-faces.py
```

**When prompted:**
- Enter name: `Person2` (or any name you prefer)
- Enter number of images: `100` (or more for better accuracy)
- Position Person 2 in front of the camera
- Follow the same process as Person 1
- Press **'q'** when done

**Expected output:**
- Images saved in `dataset/Person2/` directory
- 100+ images captured

### 3. Train the Model

```bash
python train-model.py
```

**Expected output:**
- Model training progress
- Model saved to `models/lbph_model.yml`
- Label map saved to `models/label_map.json`
- Summary showing both Person1 and Person2 loaded

### 4. Run Recognition Demo

```bash
python predict-faces.py
```

**When prompted:**
- Enter confidence threshold: `70` (default, or adjust as needed)

## Creating the Demo Video

### Video Recording Steps

1. **Start the recognition script:**
   ```bash
   python predict-faces.py
   ```

2. **Begin recording your screen/video** (using OBS, Windows Game Bar, or any screen recorder)

3. **Demo sequence (20-40 seconds):**
   - **0-5 seconds:** Show Person 1 appearing
     - Person 1 should be recognized with a **green box**
     - Name and confidence score displayed (e.g., "Person1 (45)")
   
   - **5-15 seconds:** Show Person 2 appearing
     - Person 2 should be recognized with a **green box**
     - Name and confidence score displayed (e.g., "Person2 (52)")
   
   - **15-30 seconds:** Show both people together
     - Both faces should be detected simultaneously
     - Both should be recognized with their respective names
     - Green boxes around both faces
   
   - **30-40 seconds:** Show different angles/poses
     - Move faces around to show robustness
     - Show that recognition persists

4. **Press 's' during demo** to save screenshots (optional)

5. **Press 'q'** to stop the recognition

### What to Show in the Video

✅ **Detection:**
- Faces are detected in real-time
- Bounding boxes appear around faces

✅ **Recognition:**
- At least 2 different people are recognized
- Names are displayed correctly
- Confidence scores are shown
- Green boxes for recognized faces
- Red boxes for unknown faces (if any appear)

✅ **Performance:**
- Real-time processing
- Smooth video feed
- Multiple faces detected simultaneously

## Troubleshooting Demo Issues

### Person not recognized (shows "Unknown")

**Solutions:**
- Ensure you captured enough training images (100+)
- Retrain the model: `python train-model.py`
- Lower the confidence threshold (try 60 or 50)
- Ensure lighting is similar to training conditions
- Person should face the camera directly

### Poor detection

**Solutions:**
- Ensure good lighting
- Person should be clearly visible
- Avoid extreme angles
- Check webcam is working properly

### Model not found error

**Solution:**
- Make sure you ran `train-model.py` successfully
- Check that `models/lbph_model.yml` exists

## Expected Demo Output

Your demo video should clearly show:

1. **Person 1 Recognition:**
   - Green bounding box
   - Label: "Person1 (confidence_score)"
   - Real-time detection

2. **Person 2 Recognition:**
   - Green bounding box
   - Label: "Person2 (confidence_score)"
   - Real-time detection

3. **Both People Together:**
   - Two green boxes
   - Both names displayed
   - Simultaneous recognition

## Tips for Best Demo

1. **Lighting:** Use consistent, good lighting
2. **Distance:** Keep faces at reasonable distance (not too close/far)
3. **Background:** Use a plain background if possible
4. **Stability:** Keep camera stable
5. **Practice:** Run the demo a few times before recording
6. **Screenshots:** Use 's' key to capture good moments

## File Structure After Setup

```
Face-Recognition-Pipeline/
├── capture-faces.py
├── train-model.py
├── predict-faces.py
├── requirements.txt
├── README.md
├── DEMO_INSTRUCTIONS.md (this file)
├── dataset/
│   ├── Person1/
│   │   ├── Person1_0.jpg
│   │   ├── Person1_1.jpg
│   │   └── ... (100+ images)
│   └── Person2/
│       ├── Person2_0.jpg
│       ├── Person2_1.jpg
│       └── ... (100+ images)
└── models/
    ├── lbph_model.yml
    └── label_map.json
```

## Success Criteria

Your demo is successful if:
- ✅ At least 2 different people are shown
- ✅ Both people are correctly recognized (green boxes)
- ✅ Names are displayed correctly
- ✅ Confidence scores are visible
- ✅ Video is 20-40 seconds long
- ✅ Real-time detection and recognition is demonstrated

Good luck with your demonstration! 🎥

