import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import socket

# 얼굴 검출기 로드
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 조명 보정을 위한 CLAHE 적용 함수
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

# 얼굴 검출 함수
def face_detector(img):
    # 이미지를 회색조로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 조명 보정을 위해 CLAHE 적용
    gray = apply_clahe(gray)
    
    # 얼굴 검출 - 얼굴을 찾을 때 scaleFactor와 minNeighbors는 파라미터 조정이 가능함
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # 얼굴을 찾지 못한 경우
    if faces is None or len(faces) == 0:
        return img, None

    # 얼굴을 찾은 경우
    for (x, y, w, h) in faces:
        # 얼굴 영역을 회색조로 자르고 크기를 200x200으로 변경
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
        return img, roi
    
    return img, None

# 저장된 모델 로드
def load_models():
    model_dir = "D://bong//bmproject.3//Facial-Recognition-master//models//" 
    models = {}

    model_files = [f for f in listdir(model_dir) if isfile(join(model_dir, f))]
    for file in model_files:
        name = file.split('_model.xml')[0]
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(join(model_dir, file))
        models[name] = model
    return models

# 얼굴 인식 실행 함수
def run(models):
    cap = cv2.VideoCapture(0)

    # 신뢰도 카운터 초기화
    confidence_cnt = 0
    confidence_fai = 0

    HOST = '127.0.0.1'  # C 서버가 실행 중인 IP
    PORT = 9000         # C 서버에서 지정한 포트 번호
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        while True:
            ret, frame = cap.read()
            image, face = face_detector(frame)

            try:
                min_score = 999
                min_score_name = ""
                confidence = 0

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

                if confidence > 85:
                    cv2.putText(image, "Unlocked - " + min_score_name, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    confidence_cnt += 1
                    confidence_fai = 0
                    # 신뢰도가 10번 연속으로 높을 경우 'unlocked' 신호를 전송
                    if confidence_cnt == 10:
                        s.sendall(b'unlocked')
                        confidence_cnt = 0  # 카운터 초기화
                
                #else if confidence > 80 :

            

                else:
                    cv2.putText(image, "Locked", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    confidence_fai += 1
                    confidence_cnt = 0
                    # 실패가 10번 연속으로 발생하면 'failed' 신호 전송 및 사진 촬영
                    if confidence_fai == 10:
                        s.sendall(b'failed')
                        failed_img_path = join(stranger_dir, 'failed_capture_' + str(confidence_fai) + '.jpg')
                        
                        cv2.imwrite(failed_img_path, frame)
                        cv2.imwrite('failed_capture.jpg', frame)
                        confidence_fai = 0  # 카운터 초기화

                cv2.imshow('Face Recognition', image)

            except:
                cv2.putText(image, "얼굴을 찾을 수 없습니다.", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Recognition', image)
                pass

            if cv2.waitKey(1) == 13:  # Enter 키를 누르면 종료
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    models = load_models()
    if len(models) == 0:
        print("학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요.")
    else:
        run(models)
