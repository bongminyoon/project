import cv2
import numpy as np
import os  # os 모듈 추가
from os import makedirs
from os.path import isdir

# Haar Cascade 로 얼굴 검출기 로드
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 얼굴 검출 함수
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if faces is None or len(faces) == 0:
        return None
        
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face

    return None

# 얼굴 이미지를 저장하는 함수
def take_pictures(name):
    face_dirs = "D://bong//bmproject.3//Facial-Recognition-master//faces//"  # 얼굴 이미지를 저장할 디렉토리
    user_dir = face_dirs + name + '/'
    
    # 사용자 폴더 생성
    if not isdir(user_dir):
        makedirs(user_dir)

    cap = cv2.VideoCapture(0)
    
    # 사용자 이미지를 몇 장 촬영했는지 확인
    count = len([f for f in os.listdir(user_dir) if f.endswith('.jpg')])
    
    if count >= 500:
        print("이미 500장의 이미지가 저장되었습니다.")
        return
    
    while count < 500:
        ret, frame = cap.read()

        if face_extractor(frame) is not None:
            face = face_extractor(frame)

            face = cv2.resize(face, (200, 200))
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 좌우 반전된 이미지로 데이터 증강
            face_flipped = cv2.flip(face_gray, 1)

            # 원본 이미지 저장
            count += 1
            file_name_path = user_dir + 'user' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face_gray)

            # 좌우 반전 이미지 저장
            file_name_path_flipped = user_dir + 'user_flipped' + str(count) + '.jpg'
            cv2.imwrite(file_name_path_flipped, face_flipped)

            cv2.putText(face_gray, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face_gray)
        else:
            print("Face not Found")
            pass

        # 100장의 이미지를 수집하거나 Enter 키를 누르면 세션 종료
        if cv2.waitKey(1) == 13 or count % 100 == 0:
            print(f"현재 {count}장의 이미지가 저장되었습니다.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'얼굴 이미지 수집이 완료되었습니다! 현재 총 {count}장의 이미지가 저장되었습니다.')

if __name__ == "__main__":
    name = input("저장할 사용자 이름을 입력하세요: ")
    take_pictures(name)
