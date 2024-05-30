from flask import Flask, request, jsonify, render_template
import sys
import os
from datetime import datetime
import tempfile

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

import torch
import resnet as rn
import video
from model.classify import classify


app = Flask(__name__)
# 파일 저장하는 방식
# app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/')
def index():
    return render_template('index.html')

'''
    내 로컬 데이터로 처리하는 코드
'''
# @app.route('/classify', methods=['GET'])
# def classify_video():
#     # model_path = request.json.get('model_path')
#     # video_path = request.json.get('video_path')
#     model_path = r"C:\Users\user\Documents\ADS_Project\src\model\resnet200_model.pt"
#     video_path = r"C:\Users\user\Documents\drive-download-20240528T132154Z-001\abnormal\cut_test1.mp4"

    
#     if not model_path or not video_path:
#         return jsonify({'error': 'Model path and video path are required'}), 400

#     try:
#         frames = video.process_video(video_path)

#         my_array = []
#         my_array.append(frames)
#         my_array.append(frames)
#         my_array.append(frames)
#         my_array.append(frames)
#         my_array.append(frames)


#         # 인덱스 초기화
#         index = 0

#         result = []

#         start_time = datetime.now()
#         # while 문을 사용하여 배열의 모든 요소를 순회
#         # 임시 데이터
#         # 자른 영상 대체
#         while index < len(my_array):
            
#             result.append(classify(model_path, my_array[index]))
#             # 인덱스 증가
#             index += 1

#         end_time = datetime.now()
#         print("실행 시간" , end_time - start_time)

#         print(result)

#         return jsonify({'result': result}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

'''
    파일 저장하고 처리하는 방식
'''
# @app.route('/classify', methods=['POST'])
# def classify_video():
#     model_path = request.form['model_path']
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400
    
#     video_file = request.files['video']
#     video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
#     video_file.save(video_path)

#     if not model_path or not video_path:
#         return jsonify({'error': 'Model path and video path are required'}), 400

#     try:
#         frames = video.process_video(video_path)
#         print(len(frames))
#         # my_array = [frames, frames, frames, frames, frames]
#         my_array = [frames]

#         # 인덱스 초기화
#         index = 0
#         result = []

#         start_time = datetime.now()
#         while index < len(my_array):
#             result.append(classify(model_path, my_array[index]))
#             index += 1

#         end_time = datetime.now()
#         print("Execution time:", end_time - start_time)
#         print(result)
        
#         return render_template('result.html', result=result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

'''
    메모리에서 처리하는 방식
    5월 30일 15시 43분 성공
'''
@app.route('/classify', methods=['POST'])
def classify_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name

    print(video_file)

    try:
        frames = video.process_video(temp_file_path)
        print(len(frames))

        if len(frames) == 0:
            return jsonify({'error': 'No frames extracted from video'}), 400

        '''
            여기에 영상 잘라서 배열로 return 해주는 함수 넣어주면 될듯
        '''
        # my_array = [frames, frames, frames, frames, frames]
        my_array = [frames]

        # 인덱스 초기화
        index = 0
        result = []

        start_time = datetime.now()
        while index < len(my_array):
            result.append(classify(my_array[index]))
            index += 1

        end_time = datetime.now()
        print("Execution time:", end_time - start_time)
        print(result)
        
        return render_template('result.html', result=result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
