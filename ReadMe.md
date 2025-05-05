# Real-Time Dynamic Gesture Recognition with Mediapipe & LSTM 👋🖥️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
![Framework-PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)

**Contactless HCI System** for text control using dynamic gesture recognition. Achieves **82% accuracy** with real-time performance (15 FPS).

## ✨ Key Features
- 📹 Real-time dynamic gesture detection (15 FPS)
- 🤖 Hybrid architecture: **MediaPipe** + **LSTM** temporal modeling
- 🖱️ Computer control modes: 
    - Zoom In
    - Zoom Out
    - Swip down
    - Swip up
- 📊 Trained on hybrid dataset: 20BN-Jester + custom gestures

## 🛠️ Installation
```bash
git clone https://github.com/amylingchen/hand_gesture_control_computer.git
cd hand_gesture_control_computer
pip install -r requirements.txt
```

## 🧠 Model Architecture
![MediaPipe_LSTM_Gesture_Recognition.png](output%2FMediaPipe_LSTM_Gesture_Recognition.png)
#### Two-Stage Pipeline:

- **MediaPipe Hands** - Extracts 21 3D hand keypoints (84D features + 15 angles)
- **LSTM Network** - Temporal modeling with 3 LSTM layers + FC classifier

## 🚀 Run demo
```bash
python test.py
```

## Results

![history_15.png](output%2Fhistory_15.png)
![confusion_matrix_norm_15.png](output%2Fconfusion_matrix_norm_15.png)