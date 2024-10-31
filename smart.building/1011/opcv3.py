import cv2
import numpy as np
import Jetson.GPIO as GPIO
from os import listdir, makedirs
from os.path import isfile, join
from datetime import datetime
import time
import socket


# PIR 센서 GPIO 핀 설정
PIR_PIN = 7  # Jetson Nano의 GPIO 핀 7

# GPIO 설정
GPIO.setmode(GPIO.BOARD)  # Jetson Nano의 물리적 핀 번호 사용
GPIO.setup(PIR_PIN, GPIO.IN)  # PIR 센서를 입력으로 설정

# 얼굴 검출기 로드
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 조명 보정을 위한 CLAHE 적용 함수
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

# 얼굴 검출 함수
def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return img, None, []

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
        return img, roi, faces  # 얼굴 좌표를 반환

    return img, None, faces

# 저장된 모델 로드 함수 (추가적인 모델 로드 로직 필요 시 사용)
def load_models():
    model_dir = "path_to_models"  # 모델 파일 경로 설정
    models = {}

    model_files = [f for f in listdir(model_dir) if isfile(join(model_dir, f))]
    for file in model_files:
        name = file.split('_model.xml')[0]
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(join(model_dir, file))
        models[name] = model
    return models

# 현재 날짜와 시간을 기반으로 파일 이름을 생성
def get_current_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# PIR 센서로 모션 감지 함수
def pir_motion_detected():
    return GPIO.input(PIR_PIN) == GPIO.HIGH

# 카메라 실행 및 얼굴 인식 함수
def run_camera(models, stranger_dir):
    cap = cv2.VideoCapture(0)  # 카메라 실행 (CSI 또는 USB 카메라)
    print("카메라가 켜졌습니다.")

    confidence_num = 0
    confidence_cnt = 0
    confidence_fai = 0

    HOST = '127.0.0.1'  # C 서버가 실행 중인 IP
    PORT = 9000         # C 서버에서 지정한 포트 번호

    #with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #    s.connect((HOST, PORT))


    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라를 읽을 수 없습니다.")
            break

        image, face, faces = face_detector(frame)

        try:
            min_score = 999
            min_score_name = ""
            confidence = 0

            # 얼굴 주위에 네모 그리기
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

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

            confidence_num += 1
            if confidence_num < 21:
                if confidence > 83:
                    cv2.putText(image, "Unlocked - " + min_score_name, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    confidence_cnt += 1
                    if confidence_cnt == 15:
                        #s.sendall(b'unlocked')
                        print("얼굴 인식 성공. 카메라를 종료합니다.")
                        cap.release()  # 카메라 종료
                        #return True  # 인식 성공

                else:
                    cv2.putText(image, "Locked", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    confidence_fai += 1
                    if confidence_fai == 5:
                        current_time = get_current_time_str()  # 현재 시간을 가져와 파일 이름에 추가
                        failed_img_path = join(stranger_dir, f'{current_time}.jpg')
                        cv2.imwrite(failed_img_path, frame)
                        print(f"Failed capture saved at: {failed_img_path}")
                        #cap.release()  # 카메라 종료
                        #return False  # 인식 실패

            cv2.imshow('Face Recognition', image)

        except Exception as e:
            cv2.putText(image, "not found face", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', image)
            print(f"Error: {str(e)}")
            pass

        if cv2.waitKey(1) == 13:  # Enter 키를 누르면 종료
            cap.release()  # 카메라 종료
            return False  # 종료

# 메인 루프: 프로그램이 계속 실행되며 PIR 감지 시 카메라 실행
def main():
    stranger_dir = "/home/your_user_name/facial_recognition/stranger/"  # 실패한 사진 저장 경로
    models = load_models()

    if len(models) == 0:
        print("학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요.")
        return

    while True:
        print("모션 감지 대기 중...")
        if pir_motion_detected():
            print("모션 감지됨! 카메라를 실행합니다.")
            success = run_camera(models, stranger_dir)  # 얼굴 인식 시 카메라 실행
            if success:
                print("얼굴 인식 성공!")
            else:
                print("얼굴 인식 실패.")
            time.sleep(2)  # 카메라 종료 후 약간의 대기 시간

        time.sleep(1)  # PIR 신호를 계속 감지하기 위한 주기 설정

if __name__ == "__main__":
    try:
        main()
    finally:
        GPIO.cleanup()  # 프로그램 종료 시 GPIO 설정 해제
