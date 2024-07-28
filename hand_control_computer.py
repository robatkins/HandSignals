import cv2
import mediapipe as mp
import pyautogui
import time
import threading

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Initialize previous mouse position and click state
prev_x, prev_y = None, None
clicking = False
frame = None
landmarks = []

def get_hand_landmarks():
    global frame, landmarks
    while True:
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            current_landmarks = []

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        current_landmarks.append((cx, cy))
            landmarks = current_landmarks

def is_click(landmarks):
    if len(landmarks) > 8:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5
        return distance < 20  # Adjusted for stricter click detection
    return False

def smooth_movement(prev, curr, alpha=0.5):
    return prev + alpha * (curr - prev)

# Start the hand landmark detection thread
threading.Thread(target=get_hand_landmarks, daemon=True).start()

while True:
    start_time = time.time()
    
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    if len(landmarks) > 8:
        index_finger_tip = landmarks[8]
        screen_width, screen_height = pyautogui.size()
        frame_height, frame_width, _ = frame.shape

        # Adjust coordinates from frame to screen size
        screen_x = screen_width / frame_width * index_finger_tip[0]
        screen_y = screen_height / frame_height * index_finger_tip[1]

        if prev_x is not None and prev_y is not None:
            screen_x = smooth_movement(prev_x, screen_x)
            screen_y = smooth_movement(prev_y, screen_y)

        pyautogui.moveTo(screen_x, screen_y, duration=0.01)  # Slight duration for smoother movement

        prev_x, prev_y = screen_x, screen_y

        if is_click(landmarks):
            if not clicking:
                pyautogui.mouseDown()
                clicking = True
        else:
            if clicking:
                pyautogui.mouseUp()
                clicking = False

    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Print FPS to monitor performance
    print(f'FPS: {int(fps)}')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()