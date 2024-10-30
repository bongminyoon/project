import cv2
import numpy as np
import os
from os import makedirs
from os.path import isdir

# DNN 기반 얼굴 검출 모델 로드
def load_dnn_model():
    model_file = "D://bong//prodnn//res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_file = "D://bong//prodnn//deploy.prototxt.txt"
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

# 얼굴 이미지를 저장하는 함수
def take_pictures_dnn(name):
    face_dirs = "face/"  # 얼굴 이미지를 저장할 디렉토리
    user_dir = face_dirs + name + '/'

    # 사용자 폴더 생성
    if not isdir(user_dir):
        makedirs(user_dir)

    net = load_dnn_model()  # DNN 모델 로드

    cap = cv2.VideoCapture(0)  # 기본 웹캠을 사용하여 비디오 스트림 열기
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    count = len([f for f in os.listdir(user_dir) if f.endswith('.jpg')])

    while count < 100:  # 100장의 사진을 저장
        ret, frame = cap.read()
        face = face_extractor_dnn(frame, net)

        if face is not None:
            face = cv2.resize(face, (200, 200))
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 원본 이미지 저장
            count += 1
            file_name_path = user_dir + 'user' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face_gray)

            cv2.putText(face_gray, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face_gray)
        else:
            print("얼굴을 찾지 못했습니다.")

        if cv2.waitKey(1) == 13 or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'{name} 님의 얼굴 이미지가 저장되었습니다. 총 {count}장 저장되었습니다.')

if __name__ == "__main__":
    name = input("저장할 사용자 이름을 입력하세요: ")
    take_pictures_dnn(name)
