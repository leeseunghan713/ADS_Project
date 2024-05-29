from flask import Flask, request, jsonify
import sys
import os
from datetime import datetime

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

import torch
import resnet as rn
import video
from model.classify import load_model, predict_video_classification

app = Flask(__name__)

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
    return prediction

@app.route('/classify', methods=['GET'])
def classify_video():
    # model_path = request.json.get('model_path')
    # video_path = request.json.get('video_path')
    model_path = r"C:\Users\seung\OneDrive\문서\ADS_Project\src\model\resnet200_model.pt"
    video_path = r"C:\drive-download-20240528T132154Z-001\normal\C_2_2_1_BU_DYA_08-29_10-32-51_CA_RGB_DF1_M1_F1.mp4"

    
    if not model_path or not video_path:
        return jsonify({'error': 'Model path and video path are required'}), 400

    try:
        start_time = datetime.now()
        result = classify(model_path, video_path)
        end_time = datetime.now()
        print(end_time-start_time)
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
