import cv2
import mediapipe as mp
import math
import time
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import time

class Model(nn.Module):
    def __init__(self, input_size, hidden_dim, tagset_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, tagset_size)

    def forward(self, landmarks):
        x = self.fc1(landmarks)
        x = self.fc2(x)
        res = F.log_softmax(x, dim=0)
        return res

INPUT_SIZE = 42
HIDDEN_DIM = 32
TARGET_SIZE = 3

model = Model(INPUT_SIZE, HIDDEN_DIM, TARGET_SIZE)
model.load_state_dict(torch.load('model.pt'))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands model
hands = mp_hands.Hands(
    max_num_hands=4,
)

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


n = 0
s = 0

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
 
    s = time.time_ns() // 1000000
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    e = time.time_ns() // 1000000
    s += e - S
    n += 1
    
    # Extract hand landmarks if available
    if results.multi_hand_landmarks:
        # Get the coordinates of the index fingertip (landmark index 8)
        for i in range(len(results.multi_hand_landmarks[0].landmark)):
            tip_x = results.multi_hand_landmarks[0].landmark[i].x * frame.shape[1]
            tip_y = results.multi_hand_landmarks[0].landmark[i].y * frame.shape[0]

            # Draw a circle at the fingertip position
            cv2.circle(frame, (int(tip_x), int(tip_y)), 5, (0, 255, 0), -1)

        lst = list(map(lambda x : [x.x, x.y], results.multi_hand_landmarks[0].landmark))
        res = list(model(torch.tensor([element for row in lst for element in row], dtype=torch.float)))
        cv2.putText(frame, gestures[res.index(max(res))], (frame.shape[1] // 2, frame.shape[0] // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Display the output
    cv2.imshow('1', frame)
 
    # Check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

print(s/n)

with open("./dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)
