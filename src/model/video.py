import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import random

# 데이터셋 클래스 정의
class VideoDataset(Dataset):
    def __init__(self, video_files, labels, transform=None):
        self.video_files = video_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        frames = frames[:135]
        if len(frames) < 135:
            frames += [np.zeros((224, 224, 3), dtype=np.float32)] * (135 - len(frames))

        frames = np.array(frames)
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, D, H, W)
        if self.transform:
            frames = self.transform(torch.tensor(frames, dtype=torch.float32))

        return frames, self.labels[idx]

# 단일 비디오 파일 로딩 함수
def load_single_video(video_path, frame_size=(224, 224), num_frames=135):
    # 비디오 파일을 읽기 위해 cv2.VideoCapture 객체 생성
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # 필요한 프레임 수(num_frames)만큼 프레임을 읽어 리스트에 추가
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:  # 프레임을 더 이상 읽을 수 없으면 반복 종료
            break
        # 프레임 크기를 frame_size로 조정
        frame = cv2.resize(frame, frame_size)
        # 프레임 색상 공간을 BGR에서 RGB로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    # 비디오 캡처 객체 해제
    cap.release()
    
    # 비디오가 필요한 프레임 수보다 짧으면, 부족한 프레임 수만큼 0으로 채워서 리스트에 추가
    while len(frames) < num_frames:
        frames.append(np.zeros((frame_size[1], frame_size[0], 3), dtype=np.float32))
    
    # 프레임 리스트를 numpy 배열로 변환
    frames = np.array(frames)
    # 배열의 차원을 (C, D, H, W) 형태로 변경
    frames = np.transpose(frames, (3, 0, 1, 2))  # (Channels, Depth, Height, Width)
    
    return frames  # 변환된 프레임 배열 반환
