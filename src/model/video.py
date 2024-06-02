import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

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

# 인풋 값이 135프레임의 이미지? 05월 30일 10시 21분 test => 성공
def load_single_video(frames, num_frames=135):
    processed_frames = []

    # 입력된 프레임 리스트가 최대 num_frames 프레임인지 확인하고 자르기
    if len(frames) > num_frames:
        frames = frames[:num_frames]
    
    # 프레임을 RGB로 변환하고 리스트에 추가
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(frame)

    # 프레임 리스트를 numpy 배열로 변환
    processed_frames = np.array(processed_frames)
    # 배열의 차원을 (C, D, H, W) 형태로 변경
    processed_frames = np.transpose(processed_frames, (3, 0, 1, 2))  # (Channels, Depth, Height, Width)
    # print(processed_frames)
    return processed_frames


## 데이터 전처리 224*224, 초당 3프레임 05월 30일 10시 21분 test => 성공
def process_video(input_path, target_fps=3, target_size=(224, 224)):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_path)
    
    # 입력 비디오 정보 가져오기
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # 프레임 스킵 간격 계산
    frame_skip = int(input_fps // target_fps)

    processed_frames = []

    # 프레임 처리
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 스킵하여 프레임 수 줄이기
        if current_frame % frame_skip == 0:
            # 해상도 변경
            resized_frame = cv2.resize(frame, target_size)
            # 프레임 리스트에 추가
            processed_frames.append(resized_frame)
        
        current_frame += 1

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    
    return processed_frames
