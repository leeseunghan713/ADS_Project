import torch
import cv2
import dlib
import numpy as np

def load_model():
    print("Loading YOLO model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    print("YOLO model loaded.")
    return model

def load_face_detector():
    print("Loading face detector...")
    detector = dlib.get_frontal_face_detector()
    print("Face detector loaded.")
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

def capture_frame(video_path, movement_threshold=500):
    frames = process_frame(video_path)
    print("Starting capture_frame function")
    
    model = load_model()
    detector = load_face_detector()

    print("Initializing background subtractor...")
    try:
        fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
        print("Background subtractor created successfully.")
    except Exception as e:
        print(f"Error creating background subtractor: {e}")
        return None

    frame_index = 0
    captured = False
    tracker = None
    prev_position = None
    initial_box = None
    initial_frame = None
    captured_image = None

    for frame in frames:
        fgmask = fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not captured:
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
                                    tracker = cv2.legacy.TrackerCSRT_create()
                                    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                                    captured = True
                                    initial_box = (x1, y1, x2, y2)
                                    initial_frame = frame.copy()
                                    print(f"Detected bounding box coordinates:person_frame_{frame_index}, x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                                    break
                            if captured:
                                break
                    else:
                        prev_position = current_position

        else:
            success, box = tracker.update(frame)
            if success:
                p1 = (max(0, int(box[0]) - 20), max(0, int(box[1]) - 20))
                p2 = (min(frame.shape[1], int(box[0] + box[2]) + 20), min(frame.shape[0], int(box[1] + box[3]) + 20))
                print(f"Tracked bounding box coordinates:person_frame_{frame_index}, x1={p1[0]}, y1={p1[1]}, x2={p2[0]}, y2={p2[1]}")

                roi = frame[p1[1]:p2[1], p1[0]:p2[0]]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                faces = detector(gray_roi)

                if len(faces) > 0:
                    for face in faces:
                        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        print(f"Face detected at x={x}, y={y}, w={w}, h={h}")
                    captured_image = roi
                    break
                else:
                    if initial_frame is not None and initial_box is not None:
                        x1, y1, x2, y2 = initial_box
                        captured_image = initial_frame[y1:y2, x1:x2]
                        print("Face not detected, saving bounding box area.")
                        break

        frame_index += 1

    print("capture_frame function finished")
    return captured_image


# 예시 사용
if __name__ == "__main__":
    video_path = r'C:\Users\user\Documents\drive-download-20240528T132154Z-001\abnormal\cut_test1.mp4'
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(len(frames))

    captured_image = capture_frame(frames)

    if captured_image is not None:
        cv2.imshow('Captured Image', captured_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image captured.")