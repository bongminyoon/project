import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from datetime import datetime
import socket

# DNN 기반 얼굴 검출 모델 로드
def load_dnn_model():
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    return net

# 얼굴 검출 함수 (DNN)
def face_extractor_dnn(img, net):
    # 이미지 크기 조정 및 전처리
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    h, w = img.shape[:2]
    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:  # 신뢰도 임계값
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cropped_face = img[y:y1, x:x1]
            faces.append(cropped_face)

    if len(faces) > 0:
        return faces[0]  # 첫 번째 얼굴 반환
    else:
        return None

# 조명 보정을 위한 CLAHE 적용 함수
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

# 얼굴 검출 및 위치 반환 함수
def face_detector_dnn(img, net):
    face = face_extractor_dnn(img, net)
    if face is None:
        return img, None, None

    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_gray = apply_clahe(face_gray)  # CLAHE 조명 보정
    face_resized = cv2.resize(face_gray, (200, 200))

    h, w = face_resized.shape
    cv2.rectangle(img, (0, 0), (w, h), (255, 0, 0), 2)
    
    return img, face_resized, (0, 0, w, h)  # 얼굴이 있는 영역 반환

# 학습된 모델 로드
def load_models():
    model_dir = "model/"
    models = {}

    model_files = [f for f in listdir(model_dir) if isfile(join(model_dir, f))]
    for file in model_files:
        name = file.split('_model.xml')[0]
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(join(model_dir, file))
        models[name] = model
    return models

# 현재 시간 문자열로 가져오기
def get_current_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# 얼굴 인식 실행 함수
def run(models, stranger_dir):
    
    pipeline = "v4l2src device=/dev/video0 ! image/jpeg, width=640, height=480, framerate=30/1 ! jpegdec ! videoconvert ! appsink"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    confidence_suc = 0
    confidence_fai = 0
    confidence_cnt = 0
    HOST = '127.0.0.1'
    PORT = 9000  
    
    window_name = 'Face Recognition'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 600)

    net = load_dnn_model()  # DNN 모델 로드

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b'FR:room201')
        while True:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break

            original_frame = cv2.resize(frame, (512, 600))  
            image, face, face_position = face_detector_dnn(frame, net)  # DNN을 사용하여 얼굴 검출

            try:
                min_score = 999
                min_score_name = ""
                confidence = 0

                if face is not None:
                    for name, model in models.items():
                        result = model.predict(face)
                        if min_score > result[1]:
                            min_score = result[1]
                            min_score_name = name

                    if min_score < 500:
                        confidence = int(100 * (1 - (min_score) / 300))
                        display_string = str(confidence) + '% ' + min_score_name
                    else:
                        display_string = "잠금 상태"

                    cv2.putText(image, display_string, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    confidence_cnt += 1
                    if confidence_cnt < 21:
                        if confidence >= 85:
                            cv2.putText(image, "Unlocked - " + min_score_name, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            confidence_suc += 1
                            if confidence_suc == 15:
                                s.sendall(b'FR:room201:success')
                        else:
                            cv2.putText(image, "Locked", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            confidence_fai += 1
                            if confidence_fai == 5:
                                current_time = get_current_time_str()	
                                failed_img_path = join(stranger_dir, f'{current_time}.jpg')
                                cv2.imwrite(failed_img_path, frame)
                                s.sendall(f'FR:room201:failed:{failed_img_path}'.encode())

                # C 서버로부터 신호 수신
                data = s.recv(1024).decode()
                if data == 'ROOM_201:failure capture':
                    # 실패 신호를 받으면 사진을 찍고 저장
                    current_time = get_current_time_str()
                    capture_path = join(stranger_dir, f'capture_{current_time}.jpg')
                    cv2.imwrite(capture_path, frame)
                    s.sendall(f'FR:room201:failed:{capture_path}'.encode())

                if image is not None:
                    image_resized = cv2.resize(image, (512, 600))  
                else:
                    image_resized = np.zeros((600, 512, 3), dtype=np.uint8)  

                combined_frame = np.hstack((original_frame, image_resized))  
                cv2.imshow(window_name, combined_frame)  

            except Exception as e:
                print(f"Error: {str(e)}")

            if cv2.waitKey(1) == 13:  
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stranger_dir = "stranger/"
    if not exists(stranger_dir):
        makedirs(stranger_dir)
    models = load_models()
    if len(models) == 0:
        print("학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요.")
    else:
        run(models, stranger_dir)
