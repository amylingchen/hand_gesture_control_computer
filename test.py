import cv2
import mediapipe as mp
import numpy as np
from model import *

from operations import *

seq_length = SEQ_LENGTH
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load_mode
model = LSTMClassifier(input_size=HAND_LENGTH,hidden_size=HIDDEN_SIZE,num_layers=NUM_LAYERS,num_classes=NUM_CLASSES)
model,history= load_model(model,path='model/lstm_model_15.pth',device=device)

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture('datasets/gesture.mp4')
# cap = cv2.VideoCapture(0)
labels = [0,15,18,23,25]
seq = []
action_seq = []
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while cap.isOpened():
    ret, img = cap.read()

    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feature,hand =extract_hand_features(img,hands)



    if feature is not None:
        seq.append(feature)
        mp_drawing.draw_landmarks(
            img,
            hand,
            mp_hands.HAND_CONNECTIONS,
        )

        if len(seq) < seq_length:
            continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)


        action_id,conf=recognize_gesture(input_data,model,device=device)

        if conf < 0.9:
            continue

        action = ACTIONS[action_id]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if len(action_seq) >= 4 and all(a == action for a in action_seq[-4:]):
            this_action = action
            execute_document_action(action_id, frame=img)
            # print(this_action)
        # cv2.putText(img, f'{this_action.upper()}', org=(int(hand.landmark[0].x * img.shape[1]), int(hand.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # out.write(img0)
    # out2.write(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break