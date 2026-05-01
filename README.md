# 🖐️ Hand Gesture Controlled YouTube Video System

> **Control YouTube playback in real-time using hand gestures — no mouse, no keyboard, just your hands.**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-red)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.9%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

##  Problem Statement

Traditional media controls require physical interaction with input devices. This project builds a **touchless, gesture-based YouTube controller** using a standard webcam — making video playback accessible, intuitive, and hands-free.

A **CNN model** classifies hand gestures in real-time, which are then mapped to YouTube controls like play/pause, volume, skip, and fullscreen — all triggered via **PyAutoGUI** keyboard simulation.

---

##  Features

| Gesture | Action |
|---------|--------|
| ✋ Open Palm | Play / Pause |
| 👉 Point Right | Skip Forward (→) |
| 👈 Point Left | Skip Backward (←) |
| 👍 Thumbs Up | Volume Up |
| 👎 Thumbs Down | Volume Down |
| ✊ Fist | Fullscreen Toggle |

> Gestures are fully customizable by modifying the label-to-action mapping in `controller.py`

---

##  Project Structure

```
hand-gesture-youtube-controller/
│
├── data/
│   └── leapgestrecog/           # LeapGestRecog dataset (Kaggle)
│       ├── 01_palm/
│       ├── 02_l/
│       ├── 03_fist/
│       ├── 04_fist_moved/
│       ├── 05_thumb/
│       ├── 06_index/
│       ├── 07_ok/
│       ├── 08_palm_moved/
│       ├── 09_c/
│       └── 10_down/
│
├── model/
│   └── gesture_cnn_model.h5     # Trained CNN model
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── train.py                 # CNN training script
│   ├── predict.py               # Single image inference
│   └── controller.py            # Real-time gesture → YouTube control
│
├── requirements.txt
└── README.md
```

---

##  Approach

### 1. Dataset
- **LeapGestRecog** dataset from Kaggle
- 10 gesture classes, ~20,000 grayscale images (200×200 px)
- Captured using Leap Motion infrared sensor — clean, high-contrast hand images
- Train/Test split: **80/20**, stratified by class

### 2. CNN Architecture

```
Input (64×64×1 grayscale)
     │
Conv2D(32, 3×3, ReLU) → MaxPooling(2×2)
     │
Conv2D(64, 3×3, ReLU) → MaxPooling(2×2)
     │
Conv2D(128, 3×3, ReLU) → MaxPooling(2×2)
     │
Flatten → Dense(256, ReLU) → Dropout(0.5)
     │
Dense(10, Softmax)          ← 10 gesture classes
```

- **Loss**: Categorical Crossentropy
- **Optimizer**: Adam
- **Regularization**: Dropout (0.5) to prevent overfitting
- **Callbacks**: EarlyStopping + ModelCheckpoint

### 3. Real-Time Pipeline

```
Webcam Frame
     │
MediaPipe Hands → Hand Landmark Detection
     │
ROI Crop → Grayscale → Resize (64×64) → Normalize
     │
CNN Model → Predicted Gesture Class
     │
PyAutoGUI → Keyboard Shortcut → YouTube Action
```

- **MediaPipe** handles hand detection and localization in each frame
- The detected hand region is cropped and preprocessed before CNN inference
- **PyAutoGUI** simulates keypresses (`space`, `→`, `←`, `m`, `f`) to control the browser

---

##  Results

| Metric | Score |
|--------|-------|
| Training Accuracy | ~99.9% |
| Test Accuracy | **~99.9%** |
| Inference Speed | Real-time (webcam @ 30 FPS) |
| Model Size | ~10 MB (`.h5`) |

- Near-perfect accuracy achieved on the LeapGestRecog dataset
- Robust performance across all 10 gesture classes
- Smooth real-time inference with minimal latency on CPU

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.x, Keras |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe |
| Automation | PyAutoGUI |
| Data Handling | NumPy, Matplotlib |
| IDE | VS Code |

---

##  How to Run

### Prerequisites
```bash
pip install tensorflow opencv-python mediapipe pyautogui numpy matplotlib
```

>  **MediaPipe version note**: Use `mediapipe==0.10.x` for Python 3.10 compatibility on Windows. Newer versions may cause import errors.

### Step 1 — Train the Model
```bash
python src/train.py
# Saves model to model/gesture_cnn_model.h5
```

### Step 2 — Run the Controller
```bash
# Open YouTube in your browser first, then run:
python src/controller.py
```

### Step 3 — Use It!
- Show your hand to the webcam
- Hold a gesture steadily for ~0.5 seconds
- Watch YouTube respond in real-time 🎉

---

##  Key Learnings

- **MediaPipe** provides robust hand landmark detection out of the box — no need to train a hand detector from scratch
- CNN achieves near-perfect accuracy on LeapGestRecog because the dataset is clean and well-lit; real-world generalization requires data augmentation
- **PyAutoGUI** is a simple but effective bridge between ML inference and OS-level control
- Learned the importance of **version compatibility** — MediaPipe, OpenCV, and TensorFlow versions must align on Windows to avoid runtime errors

---

##  Future Improvements

- [ ] Add data augmentation for better generalization across lighting conditions
- [ ] Deploy as a **browser extension** (Chrome) for native YouTube integration
- [ ] Support **custom gesture mapping** via a config file
- [ ] Add a **GUI overlay** showing detected gesture and confidence score in real-time
- [ ] Train on a larger, more diverse dataset (e.g., HaGRID) for robustness

---

##  Author

**Jatin** — B.Tech Engineering Physics, Delhi Technological University  
GitHub: [@QuantumWebber](https://github.com/QuantumWebber)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
