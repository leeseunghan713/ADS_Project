import os
from datetime import datetime
from capture.test_capture import capture_frame
import cv2
from model.classify import classify
from model.video import process_video

def save_captured_image(image, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = f'captured_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image)
    return filepath

if __name__ == '__main__':
    video_path = r'C:\Users\user\Documents\drive-download-20240528T132154Z-001\abnormal\cut_test1.mp4'
    output_folder = r'C:\Users\user\Documents\ADS_Project\src\static\images'
    
    frames = process_video(video_path)
    classification = classify(frames)
    print(classification)
    
    captured_image = capture_frame(video_path)

    if captured_image is not None:
        image_path = save_captured_image(captured_image, output_folder)
        print(f'Captured image saved to {image_path}')
        cv2.imshow('Captured Image', captured_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image captured.")
