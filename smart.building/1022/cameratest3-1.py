import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from datetime import datetime  
import socket

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if faces is None or len(faces) == 0:
        return img, None, None

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  
        return img, roi, faces[0] 
    
    return img, None, None


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


def get_current_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run(models, stranger_dir):
    
    pipeline = "v4l2src device=/dev/video0 ! image/jpeg, width=640, height=480, framerate=30/1 ! jpegdec ! videoconvert ! appsink"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    confidence_suc = 0
    confidence_fai = 0
    confidence_cnt = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()  # 원본 영상을 따로 저장
        image, face, face_position = face_detector(frame)

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
                            print("얼굴 인식 성공")
                    else:
                        cv2.putText(image, "Locked", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        confidence_fai += 1
                        if confidence_fai == 5:
                            current_time = get_current_time_str()  
                            failed_img_path = join(stranger_dir, f'{current_time}.jpg')
                            cv2.imwrite(failed_img_path, frame)
                            print(f"Failed capture saved at: {failed_img_path}")
                else:
                    confidence_suc = 0
                    confidence_fai = 0
                    confidence_cnt = 0         

            combined_frame = np.hstack((original_frame, image))  # 원본과 결과 영상을 가로로 붙임
            cv2.imshow('Face Recognition', combined_frame)

        except Exception as e:
            cv2.putText(image, "not found face", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            combined_frame = np.hstack((original_frame, image))  # 원본과 에러 메시지 영상을 가로로 붙임
            cv2.imshow('Face Recognition', combined_frame)
            print(f"Error: {str(e)}")
            pass

        if cv2.waitKey(1) == 13:  
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stranger_dir = "stranger/"  
    models = load_models()
    if len(models) == 0:
        print("학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요.")
    else:
        run(models, stranger_dir)

