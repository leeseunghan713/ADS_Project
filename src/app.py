from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from datetime import datetime
import tempfile
import cv2
from capture.capture import capture_frame
from model.classify import classify
from model.video import process_video

app = Flask(__name__)

# 업로드된 파일을 저장할 폴더
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

'''
    메모리에서 처리하는 방식
    5월 30일 15시 43분 성공
'''

@app.route('/classify', methods=['POST'])
def classify_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']

    ## 임시파일로 저장하여 사용
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name

    try:
        frames = process_video(temp_file_path)

        if len(frames) == 0:
            return jsonify({'error': 'No frames extracted from video'}), 400
    
        '''
            영상 분류후에 결과값이 절도면 절도범 캡처해서 절도와 capture 출력, 정상이면 정상만 출력
        '''
        ## 영상 분류
        results = []
        result = classify(frames)
        results.append(result)

        image_path = None
        if result:
            ## 이미지 캡처
            capture = capture_frame(temp_file_path)
            if capture is not None:
                filename = f"captured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                cv2.imwrite(image_path, capture)
                image_path = os.path.join('images', filename) 
        
        return render_template('result.html', result=results, image_path=filename)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
