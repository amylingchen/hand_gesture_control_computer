import cv2
import mediapipe as mp
import numpy as np
import time, os
import  pandas as pd
from dataclasses import dataclass


class Videos:
    def __init__(self, video_id, label, label_id):
        self.video_id = str(video_id)  # 统一为字符串
        self.label = label
        self.label_id = label_id
        self.features = []  # 特征存储
        self.labels = []  # 标签存储


def load_video(data):
    return [Videos(row['video_id'], row['label'], row['label_id'])
            for _, row in data.iterrows()]


def extract_hand_features(image_path, hands):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"图片读取失败: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None, None

        hand = results.multi_hand_landmarks[0]

        # 关节坐标处理
        joint = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                          for lm in hand.landmark])

        # 角度计算（保持原有逻辑）
        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
        v = (v2 - v1) / np.linalg.norm(v2 - v1, axis=1)[:, np.newaxis]

        angle = np.degrees(np.arccos(np.einsum('nt,nt->n',
                                               v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                               v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :])))

        return np.concatenate([joint.flatten(), angle]), results.multi_handedness[0].classification[0].label

    except Exception as e:
        print(f"处理 {image_path} 时出错: {str(e)}")
        return None, None



def extract_keypoints(img, hands):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return np.zeros(65), -1

        hand = results.multi_hand_landmarks[0]

        detection = np.array([0 if hand.classification[0].label == 'Left' else 1,
                              hand.classification[0].score]).flatten()

        landmarks = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[
            0].landmark]).flatten() if results.multi_hand_landmarks else np.zeros(63)

        return np.concatenate([detection, landmarks]), results.multi_handedness[0].classification[0].label

    except Exception as e:

        return None, None



def process_image_dataset(in_dir, output_dir, hands, videos):
    os.makedirs(output_dir, exist_ok=True)

    for video_name in os.listdir(in_dir):
        video_path = os.path.join(in_dir, video_name)
        if not os.path.isdir(video_path):
            continue

        video = next((v for v in videos if v.video_id == video_name), None)
        if not video:
            # print(f"跳过未注册视频: {video_name}")
            continue
        print(f"注册视频: {video_name}")
        frame_files = sorted(
            [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))],
            key=lambda x: int(x.split('.')[0])
        )

        for pic_name in frame_files:
            img_path = os.path.join(video_path, pic_name)
            features, label = extract_hand_features(img_path, hands)

            if features is not None:
                video.features.append(features)
                video.labels.append(label if label else video.label_id)
        if len(video.features)>0:

            save_filename = f"{video.label}_{video.video_id}.npz"
            np.savez(os.path.join(output_dir, save_filename), features=np.array(video.features),
            labels=np.array(video.label_id))
            print(f"Saved {save_filename} with shape {np.array(video.features).shape}")


def load_to_seq(file_dir,output_dir,seq_length,seq_path=None):
    labels =[0,15,18,23,25]
    file_name = file_dir.split('/')[-1]
    seq_features =[]
    seq_labels = []
    for gesture_name in os.listdir(file_dir):
        gesture_path = os.path.join(file_dir, gesture_name)
        gesture = np.load(gesture_path)
        if len(gesture['features'])>=seq_length:
            for i in range(len(gesture['features']) - seq_length + 1):
                seq_features.append(gesture['features'][i:i + seq_length])
                seq_labels.append(labels.index(gesture['labels']))  # 使用视频的整体标签

    if seq_path:
        seq = np.load(seq_path)
        seq_features =seq_features+seq['features'].tolist()
        seq_labels =seq_labels+seq['labels'].tolist()

    if seq_features:
        np.savez(
            os.path.join(output_dir, f'seq_{file_name}'),
            features=np.array(seq_features),
            labels=np.array(seq_labels)
        )
        print(f"保存完成，总序列数: {len(seq_features)}")


if __name__ == '__main__':
    # data_list = pd.read_csv('../datasets/20bn/Train.csv')
    # label_ids = [0,15,18,25,23]
    # train_data = data_list[data_list['label_id'].isin(label_ids)]
    # video_paths ='../datasets/20bn/train'
    # mp_hands = mp.solutions.hands
    # mp_drawing = mp.solutions.drawing_utils
    # hands = mp_hands.Hands(
    #     max_num_hands=1,
    #     min_detection_confidence=0.5,
    #     min_tracking_confidence=0.5)
    #
    # try:
    #     video_segments = load_video(train_data)
    #
    #     process_image_dataset(video_paths,
    #                           '../datasets/gesture/train',
    #                           hands,video_segments)
    #
    #
    # finally:
    #     hands.close()
    gesture_paths='../datasets/gesture/validation'
    output_dir='../datasets/seq/fintune'
    load_to_seq(file_dir=gesture_paths,output_dir=output_dir,seq_length=15)
