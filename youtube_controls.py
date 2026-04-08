import pyautogui
import time

GESTURE_ACTIONS = {
    '01_palm': 'k',    # Play/Pause
    '03_fist': 'k',    # Play/Pause  
    '06_index': 'l',   # Forward 10 sec
    '10_down': 'j'     # Backward 10 sec
}

def perform_action(gesture_name):
    if gesture_name not in GESTURE_ACTIONS:
        return
    key = GESTURE_ACTIONS[gesture_name]
    time.sleep(0.05)
    pyautogui.press(key)
    print(f"[ACTION] {gesture_name} -> {key}")