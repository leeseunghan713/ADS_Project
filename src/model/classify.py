import sys
import os

# resnet 모듈의 경로를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
import resnet as rn
import video

# 모델 불러오기
def load_model(model, path='model.pt'):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()  # 평가 모드로 설정

# 새로운 영상 데이터를 예측하는 함수
def predict_video_classification(model, video_path, device):
    model.eval()
    video_data = video.load_single_video(video_path)
    video_tensor = torch.tensor(video_data, dtype=torch.float32).unsqueeze(0).to(device)  # 배치 차원 추가 및 CPU로 이동
    
    with torch.no_grad():
        outputs = model(video_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        class_label = 'Normal' if class_idx == 0 else 'Abnormal'
        return class_label

def classify(model_path, video_path):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # CPU 사용 여부 확인
    device = torch.device("cpu")

    # 모델 인스턴스 생성 및 파라미터 로드
    loaded_model = rn.generate_model(200, n_input_channels=3, n_classes=2)
    loaded_model.to(device)  # 모델을 CPU로 옮기기
    load_model(loaded_model, model_path)

    # 새로운 영상 데이터 예측
    new_video_path = video_path
    prediction = predict_video_classification(loaded_model, new_video_path, device)
    print(f'The video is classified as: {prediction}')
    
    return prediction

if __name__=='__main__':
    model_path = r"C:\Users\seung\OneDrive\문서\ADS_Project\src\model\resnet200_model.pt"
    video_path = r"C:\drive-download-20240528T132154Z-001\normal\C_2_2_1_BU_DYA_08-29_10-32-51_CA_RGB_DF1_M1_F1.mp4"
    result = classify(model_path, video_path)
    print(result)
