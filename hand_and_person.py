import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Initialize MediaPipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the trained object detection model
model = tf.saved_model.load('saved_model')

# COCO label map
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 
    38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 
    74: 'mouse', 91: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

# Custom label map (example: 91 for gun)
CUSTOM_LABELS = {
    1: 'person',
    75: 'gun'
}

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Function to perform object detection
def detect_objects(frame, model):
    # Convert the image to tensor and ensure it's uint8
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
    
    # Get the model's infer signature
    infer = model.signatures['serving_default']
    
    # Perform the inference
    detections = infer(input_tensor)
    
    return detections

# Set up the hand tracking
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(rgb_frame)

        # Perform object detection
        detections = detect_objects(rgb_frame, model)
        
        # Draw hand annotations on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        # Draw object detection annotations
        detection_boxes = detections['detection_boxes'].numpy()[0]  # Get the first detection batch
        detection_classes = detections['detection_classes'].numpy()[0].astype(int)
        detection_scores = detections['detection_scores'].numpy()[0]

        # Iterate through detections and draw bounding boxes
        for i in range(len(detection_boxes)):
            if detection_scores[i] > 0.5:  # Only consider detections with confidence > 0.5
                ymin, xmin, ymax, xmax = detection_boxes[i]
                start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
                end_point = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
                label_id = detection_classes[i]
                label = CUSTOM_LABELS.get(label_id, f'Class {label_id}')
                score = detection_scores[i]
                cv2.putText(frame, f'{label}: {score:.2f}', (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Hand and Person Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()