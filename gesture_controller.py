import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from collections import deque, Counter
from youtube_controls import perform_action

MODEL_PATH = 'models/best_model.keras'
CLASS_NAMES_PATH = 'models/class_names.npy'
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.75
SMOOTHING_WINDOW = 5
ACTION_COOLDOWN = 1.5

model = tf.keras.models.load_model(MODEL_PATH)
class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
last_action_time = 0
last_action_display = ""
last_action_display_time = 0

GESTURE_LABELS = {
    '01_palm': 'Play/Pause',
    '03_fist': 'Play/Pause',
    '06_index': 'Forward >>',
    '10_down': '<< Backward'
}

def preprocess_hand(hand_roi):
    img = cv2.resize(hand_roi, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = gray / 255.0
    norm = norm[..., np.newaxis]
    norm = np.expand_dims(norm, axis=0)
    return norm

def predict_gesture(hand_roi):
    img = preprocess_hand(hand_roi)
    predictions = model.predict(img, verbose=0)[0]
    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx]
    gesture = class_names[class_idx]
    return gesture, confidence

def get_smooth_prediction(gesture, confidence):
    if confidence >= CONFIDENCE_THRESHOLD:
        prediction_buffer.append(gesture)
    else:
        prediction_buffer.append(None)
    if len(prediction_buffer) == SMOOTHING_WINDOW:
        counts = Counter(prediction_buffer)
        top_gesture, top_count = counts.most_common(1)[0]
        if top_gesture and top_count >= 3:
            return top_gesture
    return None

def can_trigger_action():
    global last_action_time
    now = time.time()
    if now - last_action_time >= ACTION_COOLDOWN:
        last_action_time = now
        return True
    return False

def main():
    global last_action_display, last_action_display_time

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam started! Press q to quit")
    print("YouTube tab focus mein rakho!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam nahi mili!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture_label = "No hand"
        current_gesture = ""
        current_confidence = 0.0

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_list = [lm.x * w for lm in hand_lm.landmark]
            y_list = [lm.y * h for lm in hand_lm.landmark]

            x1 = max(0, int(min(x_list)) - 20)
            y1 = max(0, int(min(y_list)) - 20)
            x2 = min(w, int(max(x_list)) + 20)
            y2 = min(h, int(max(y_list)) + 20)

            hand_roi = frame[y1:y2, x1:x2]

            if hand_roi.size > 0:
                gesture, confidence = predict_gesture(hand_roi)
                current_gesture = gesture
                current_confidence = confidence
                print(f"Detected: {gesture} - {confidence:.0%}")
                gesture_label = f"{gesture} ({confidence:.0%})"

                smooth = get_smooth_prediction(gesture, confidence)
                if smooth and can_trigger_action():
                    perform_action(smooth)
                    last_action_display = GESTURE_LABELS.get(smooth, smooth)
                    last_action_display_time = time.time()

                color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # --- Top HUD bar ---
        cv2.rectangle(frame, (0, 0), (640, 65), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, "Press Q to quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

      
        if current_gesture:
            label_text = GESTURE_LABELS.get(current_gesture, current_gesture)
            conf_text = f"{current_confidence:.0%}"

            # Background box
            cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (10, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, conf_text, (500, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

       
        if time.time() - last_action_display_time < 2.0 and last_action_display:
            cv2.rectangle(frame, (150, 180), (490, 260), (0, 180, 0), -1)
            cv2.putText(frame, "ACTION!", (220, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, last_action_display, (175, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

       
        small = cv2.resize(frame, (320, 240))
        cv2.imshow("Gesture Controller", small)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
