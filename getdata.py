import cv2
import mediapipe as mp
import math
import time
import torch
import torch.nn as nn
import json


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands model
hands = mp_hands.Hands()

# Open a webcam
cap = cv2.VideoCapture(0)
#_, frame = cap.read()
#cv2.imshow('Waving Fingertip Tracking', frame)

start_time = time.time()
state = 'break'

dataset = []
data = []

gestures = [ 'left', 'right', 'stop' ]
gesture_num = 0

while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()
    if frame is None:
        continue

    # Check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    # Extract hand landmarks if available
    if results.multi_hand_landmarks:
        # Get the coordinates of the index fingertip (landmark index 8)
        for i in range(len(results.multi_hand_landmarks[0].landmark)):
            tip_x = results.multi_hand_landmarks[0].landmark[i].x * frame.shape[1]
            tip_y = results.multi_hand_landmarks[0].landmark[i].y * frame.shape[0]

            # Draw a circle at the fingertip position
            cv2.circle(frame, (int(tip_x), int(tip_y)), 5, (0, 255, 0), -1)

        lst = list(map(lambda x : [x.x, x.y], results.multi_hand_landmarks[0].landmark))

        if state == 'recording':
            dataset.append({'landmarks':lst.copy(), 'gesture':gesture_num})

    if (time.time() - start_time) > 30 and state == 'recording':
        start_time = time.time()
        state = 'break'
        gesture_num = (gesture_num + 1) % 3
    elif (time.time() - start_time) > 5 and state == 'break':
        start_time = time.time()
        state = 'recording'
    
    cv2.putText(frame, state, (frame.shape[1] // 2, frame.shape[0] // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(frame, gestures[gesture_num], (frame.shape[1] // 2, frame.shape[0] // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Display the output
    cv2.imshow('1', frame)
 
    # Check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

with open("./dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)
