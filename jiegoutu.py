import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import cv2
import mediapipe as mp
from utils import *
from operations import *

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

img = cv2.imread('output/00010.jpg')
img = cv2.flip(img, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
feature,hand =extract_hand_features(img,hands)

landmark_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=0, circle_radius=2)

# 设置连接线的样式
connection_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
mp_drawing.draw_landmarks(
    img,  # 要绘制的图像
    hand,  # 识别到的单只手 landmarks
    mp_hands.HAND_CONNECTIONS,  # 连接关系
    landmark_style,  # 点的样式
    connection_style  # 线的样式
)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow('img', img)
save_path = "output/00010_1.jpg"
cv2.imwrite(save_path, img)

