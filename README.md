# Hand Gesture Controlled YouTube System

Control YouTube playback with hand gestures via webcam — a CNN classifies the gesture, MediaPipe locates the hand in frame, and PyAutoGUI simulates the matching YouTube keyboard shortcut.

---

## What It Actually Does

- **Hand detection**: MediaPipe Hands locates a single hand per frame and gives landmark coordinates, used to crop a bounding-box ROI around the hand.
- **Gesture classification**: the cropped ROI is resized to 64×64, converted to grayscale, normalized, and passed through a pretrained Keras CNN (`models/best_model.keras`) that predicts a gesture class from the class list in `models/class_names.npy`.
- **Prediction smoothing**: a 5-frame rolling buffer requires the same gesture to appear in at least 3 of the last 5 frames above a 0.75 confidence threshold before it's accepted — reduces flicker/false triggers from single noisy frames.
- **Action cooldown**: a 1.5s cooldown after each triggered action prevents one held gesture from repeatedly firing the same keypress.
- **Currently wired-up gestures** — only 4 of the model's classes are mapped to an action in `youtube_controls.py`:

| Gesture class | YouTube action | Key simulated |
|---|---|---|
| `01_palm` | Play / Pause | `k` |
| `03_fist` | Play / Pause | `k` |
| `06_index` | Forward 10s | `l` |
| `10_down` | Backward 10s | `j` |

The model itself was trained on more gesture classes than this (the LeapGestRecog dataset has 10), but only these 4 are currently mapped to an action — the rest are recognized but do nothing.

---

## Pipeline

```
Webcam frame
     │
MediaPipe Hands → landmark detection → bounding box
     │
Crop → resize (64×64) → grayscale → normalize
     │
CNN (best_model.keras) → gesture class + confidence
     │
5-frame majority-vote smoothing (≥3/5 agreement, conf ≥ 0.75)
     │
Cooldown check (1.5s) → PyAutoGUI keypress (k / l / j)
```

---

## Tech Stack

Python, TensorFlow/Keras, OpenCV, MediaPipe, PyAutoGUI, NumPy

---

## Project Structure (as it actually exists in this repo)

```
CNN-Hand-Gesture-/
├── gesture_controller.py     # webcam loop, MediaPipe, inference, smoothing, HUD overlay
├── youtube_controls.py       # gesture-name → keypress mapping, PyAutoGUI action
├── models/
│   └── class_names.npy       # gesture class label list
└── requirements.txt
```





## How to Run

```bash
pip install -r requirements.txt
# Ensure models/best_model.keras exists (trained separately) alongside models/class_names.npy
# Open YouTube in your browser first, then:
python gesture_controller.py
# Press 'q' to quit
```
