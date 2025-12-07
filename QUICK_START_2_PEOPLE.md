# Quick Guide: Adding a Second Person

You currently have **Raissa** trained. To add a second person:

## Step 1: Capture Second Person

Run this command:
```bash
python capture-faces.py
```

**When prompted:**
- Enter name: `Person2` (or any name, e.g., "John", "Sarah", etc.)
- Enter number of images: `100` (or at least 50)
- Position the second person in front of camera
- Press **SPACE** to capture each image
- Vary: poses, expressions, lighting
- Press **'q'** when done

**Result:** Images saved in `dataset/Person2/` (or whatever name you chose)

## Step 2: Retrain Model (Includes Both People)

After capturing the second person, retrain the model:
```bash
python train-model.py
```

**What happens:**
- Loads images from both `dataset/Raissa/` and `dataset/Person2/`
- Trains model to recognize BOTH people
- Saves updated model to `models/lbph_model.yml`
- Updates `models/label_map.json` with both names

**Expected output:**
```
Loading images for: Raissa (Label: 0)
  Loaded X images
Loading images for: Person2 (Label: 1)
  Loaded Y images
Total persons: 2
```

## Step 3: Run Recognition (Both People)

Now run the prediction:
```bash
python predict-faces.py
```

**When prompted:**
- Enter confidence threshold: `70` (default)

**What you'll see:**
- Both Raissa and Person2 will be recognized
- Green boxes around recognized faces
- Names displayed: "Raissa (confidence)" and "Person2 (confidence)"
- Both can appear together and be recognized simultaneously

## Verification

Check your dataset structure:
```
dataset/
├── Raissa/
│   └── (images)
└── Person2/  (or whatever name you used)
    └── (images)
```

Check your model:
- `models/lbph_model.yml` should exist
- `models/label_map.json` should contain both names

## Tips

- **More images = Better accuracy**: Capture 100+ images per person
- **Variety**: Different angles, expressions, lighting
- **Retrain**: Always retrain after adding new people
- **Test**: Run predict-faces.py to verify both people are recognized

## Troubleshooting

**Second person shows as "Unknown":**
- Make sure you retrained: `python train-model.py`
- Check that images were saved in `dataset/Person2/`
- Lower confidence threshold (try 60 or 50)
- Capture more images for the second person

**Only one person recognized:**
- Verify both folders exist in `dataset/`
- Check `models/label_map.json` contains both names
- Retrain the model again

