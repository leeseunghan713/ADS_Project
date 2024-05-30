import sys
import os
from datetime import datetime

# resnet 모듈의 경로를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
import resnet as rn
import video

# 모델 불러오기
def load_model(model, path='model.pt'):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()  # 평가 모드로 설정
    
'''
# 05월 30일 10시 56분 통합 test => 성공
'''

# def classify(model_path, video_path):
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#     # 사용 가능한 장치 설정 (GPU가 있으면 GPU 사용, 아니면 CPU 사용)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 모델 인스턴스 생성 및 파라미터 로드
#     loaded_model = rn.generate_model(200, n_input_channels=3, n_classes=2)
#     loaded_model.to(device)  # 모델을 장치로 옮기기
#     load_model(loaded_model, model_path)

#     # 비디오 프레임 전처리 및 예측
#     frames = video.process_video(video_path)  # 비디오 프레임 전처리
#     loaded_model.eval()
#     video_data = video.load_single_video(frames)
    
#     print(f"Number of frames: {len(video_data)}")
#     video_tensor = torch.tensor(video_data, dtype=torch.float32).unsqueeze(0).to(device)  # 배치 차원 추가 및 장치로 이동
    
#     with torch.no_grad():
#         outputs = loaded_model(video_tensor)
#         _, predicted = torch.max(outputs, 1)
#         class_idx = predicted.item()
#         class_label = 'Normal' if class_idx == 0 else 'Abnormal'
#         print(f'The video is classified as: {class_label}')
    
#     return class_label

'''
 05월 30일 10시 56분 frames 를 입력 받은 통합 test => 성공
'''
def classify(frames):
    model_path = r"C:\Users\user\Documents\ADS_Project\src\model\resnet200_model.pt"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 사용 가능한 장치 설정 (GPU가 있으면 GPU 사용, 아니면 CPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # 모델 인스턴스 생성 및 파라미터 로드
    loaded_model = rn.generate_model(200, n_input_channels=3, n_classes=2)
    loaded_model.to(device)  # 모델을 장치로 옮기기
    load_model(loaded_model, model_path)

    # 모델 평가 및 예측
    loaded_model.eval()
    video_data = video.load_single_video(frames)  # 이미 전처리된 프레임 사용
    
    # print(f"Number of frames: {len(video_data)}")
    video_tensor = torch.tensor(video_data, dtype=torch.float32).unsqueeze(0).to(device)  # 배치 차원 추가 및 장치로 이동
    
    with torch.no_grad():
        outputs = loaded_model(video_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        class_label = False if class_idx == 0 else True
        print(f'The video is classified as: {class_label}')
    
    return class_label

# if __name__ == '__main__':
#     model_path = r"C:\Users\user\Documents\ADS_Project\src\model\resnet200_model.pt"
#     video_path = r"C:\Users\user\Documents\drive-download-20240528T132154Z-001\abnormal\cut_test1.mp4"

#     frames = video.process_video(video_path)
#     '''
#          05월 30일 11시 00분 frames 를 입력 받은 통합 test 
#          과정    1. 영상 경로를 process_video 함수를 통해서 전처리
#                  2. 전처리한 frames를 classift 함수에 넣어서 분류

#         방씨 부분 1. 전처리 후에 영상 자르는 부분 추가
#         내가 해야할거  2. 자른 영상을 인덱스 0번부터 classify 함수에 입력값으로 넣어서 해당 인덱스 값에 출력값을 새로운 배열에 삽입 
      
#     '''
#     my_array = []
#     my_array.append(frames)
#     my_array.append(frames)
#     my_array.append(frames)
#     my_array.append(frames)
#     my_array.append(frames)

#     # 인덱스 초기화
#     index = 0

#     result = []

#     start_time = datetime.now()
#     # while 문을 사용하여 배열의 모든 요소를 순회
#     # 임시 데이터
#     # 자른 영상 대체
#     while index < len(my_array):
        
#         result.append(classify(model_path, my_array[index]))
#         # 인덱스 증가
#         index += 1

#     end_time = datetime.now()
#     print("실행 시간" , end_time - start_time)

#     print(result)