
# Hand Gesture Controlled YouTube System

Real-time hand gesture recognition system that controls YouTube playback using webcam.

## Demo


## Tech Stack
- Python 3.12
- TensorFlow 2.16.2
- MediaPipe 0.10.14
- OpenCV
- PyAutoGUI

## Gestures
| Gesture | Action |
|---|---|
| Open Palm | Play/Pause |
| Fist | Play/Pause |

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/gesture-youtube
cd gesture-youtube
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python gesture_controller.py
```

## Project Structure
```
gesture-youtube/
├── gesture_controller.py
├── youtube_controls.py
├── models/
│   ├── best_model.keras
│   └── class_names.npy
├── requirements.txt
└── README.md
```
