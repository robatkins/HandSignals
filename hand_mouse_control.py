import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize previous mouse position
prev_x, prev_y = None, None

def get_hand_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    landmarks = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return landmarks

def is_click(landmarks):
    if len(landmarks) > 8:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5
        return distance < 30
    return False

while True:
    start_time = time.time()
    
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    landmarks = get_hand_landmarks(frame)

    if len(landmarks) > 8:
        index_finger_tip = landmarks[8]
        screen_width, screen_height = pyautogui.size()
        frame_height, frame_width, _ = frame.shape

        # Adjust coordinates from frame to screen size
        screen_x = screen_width / frame_width * index_finger_tip[0]
        screen_y = screen_height / frame_height * index_finger_tip[1]

        # Smooth movement by interpolating between positions
        if prev_x is not None and prev_y is not None:
            screen_x = (screen_x + prev_x) / 2
            screen_y = (screen_y + prev_y) / 2

        pyautogui.moveTo(screen_x, screen_y)

        prev_x, prev_y = screen_x, screen_y

        if is_click(landmarks):
            pyautogui.click()

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()