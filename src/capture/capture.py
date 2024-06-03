import torch
import cv2
import dlib
import numpy as np

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model

def load_face_detector():
    detector = dlib.get_frontal_face_detector()
    return detector

def process_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def resize_with_aspect_ratio(image, height=None, inter=cv2.INTER_AREA):
    if height is None:
        return image
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def capture_frame(video_path, movement_threshold=500, output_height=720):
    frames = process_frame(video_path)
    model = load_model()
    detector = load_face_detector()

    fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
    
    prev_position = None
    captured_image = None

    for frame in frames:
        fgmask = fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                (x, y, w, h) = cv2.boundingRect(contour)
                current_position = (x + w // 2, y + h // 2)

                if prev_position is not None:
                    distance = np.linalg.norm(np.array(current_position) - np.array(prev_position))
                    if distance > movement_threshold:
                        prev_position = current_position
                        roi = frame[y:y + h, x:x + w]
                        results = model(roi)

                        labels, cords = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()

                        for i, label in enumerate(labels):
                            if int(label) == 0:
                                x1, y1, x2, y2, conf = cords[i]
                                x1, y1, x2, y2 = int(x1 * w + x), int(y1 * h + y), int(x2 * w + x), int(y2 * h + y)

                                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                faces = detector(gray_roi)

                                if len(faces) > 0:
                                    for face in faces:
                                        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    captured_image = roi
                                else:
                                    captured_image = frame[y1:y2, x1:x2]
                                break
                else:
                    prev_position = current_position

        if captured_image is not None:
            break

    if captured_image is not None:
        captured_image = resize_with_aspect_ratio(captured_image, height=output_height)

    return captured_image
