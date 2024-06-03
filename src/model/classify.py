import os
import torch
from model.video import load_single_video
from model.resnet import generate_model

# 모델 불러오기
def load_model(model, path='model.pt'):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()  # 평가 모드로 설정

'''
 05월 30일 10시 56분 frames 를 입력 받은 통합 test => 성공
'''
def classify(frames):
    # 상대 경로로 모델 경로 설정
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'resnet200_model.pt')

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 사용 가능한 장치 설정 (GPU가 있으면 GPU 사용, 아니면 CPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 인스턴스 생성 및 파라미터 로드
    loaded_model = generate_model(200, n_input_channels=3, n_classes=2)
    loaded_model.to(device)  # 모델을 장치로 옮기기
    load_model(loaded_model, model_path)

    # 모델 평가 및 예측
    loaded_model.eval()
    video_data = load_single_video(frames)  # 이미 전처리된 프레임 사용
    
    video_tensor = torch.tensor(video_data, dtype=torch.float32).unsqueeze(0).to(device)  # 배치 차원 추가 및 장치로 이동
    
    with torch.no_grad():
        outputs = loaded_model(video_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        class_label = False if class_idx == 0 else True
        print(f'The video is classified as: {class_label}')
    
    return class_label
