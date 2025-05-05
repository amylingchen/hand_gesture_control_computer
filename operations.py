import cv2
import mediapipe as mp
import numpy as np
import time, os

import pyautogui
from mediapipe.python.solutions import hands
from config import *
from enum import Enum
from model import predict

class Gesture(Enum):
    SWIPE_UP = 1
    SWIPE_DOWN = 2
    ZOOM_IN = 3
    ZOOM_OUT = 4
    NONE = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)


def extract_hand_features(img, hands):
    try:



        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None, None

        hand = results.multi_hand_landmarks[0]


        joint = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                          for lm in hand.landmark])
        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
        v = (v2 - v1) / np.linalg.norm(v2 - v1, axis=1)[:, np.newaxis]

        angle = np.degrees(np.arccos(np.einsum('nt,nt->n',
                                               v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                               v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :])))

        return np.concatenate([joint.flatten(), angle]), hand

    except Exception as e:
        print(f"Error : {str(e)}")
        return None, None

def recognize_gesture(seq,model,device='cpu',actions=ACTIONS):
    prediction,scores = predict(model,seq,device)
    score = scores[0][prediction[0]]
    return prediction[0], score



def execute_document_action(gesture, frame):

    if gesture == 1:
        pyautogui.scroll(300)
        print("↑ Swiped Up ")
    elif gesture == 2:
        pyautogui.scroll(-300)
        print("↓ Swiped Down ")
    elif gesture == 3:
        pyautogui.hotkey('ctrl', '+')
        print(" Zoom In ")
    elif gesture == 4:
        pyautogui.hotkey('ctrl', '-')
        print(" Zoom Out ")


    cv2.putText(frame, f"Action: {ACTIONS[gesture]}",
                (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
