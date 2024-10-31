import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# 데이터 경로 설정
data_path = "D:/bmproject/Facial-Recognition-master/faces/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
Training_Data, Labels = [], []

# 이미지 데이터와 라벨 생성
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is None:
        continue    
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

if len(Labels) == 0:
    print("There is no data to train.")
    exit()

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!!!!!")

# 얼굴 검출용 XML 파일
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

# 카메라 열기 
cap = cv2.VideoCapture(0)

while True:
    # 카메라로부터 사진 한 장 읽기 
    ret, frame = cap.read()
    # 얼굴 검출 시도 
    image, face = face_detector(frame)
    try:
        # 검출된 사진을 흑백으로 변환 
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # 위에서 학습한 모델로 예측 시도
        result = model.predict(face)
        if result[1] < 500:
            confidence = int(100 * (1 - (result[1] / 300)))
            display_string = str(confidence) + '% Confidence it is user'
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

            # 신뢰도가 70%를 넘으면 변수에 저장
            if confidence > 80:
                recognition_confidence = confidence  # 인식률을 변수에 저장
                print(f"Recognition Confidence: {recognition_confidence}%")

            # 75보다 크면 동일 인물로 간주하여 Unlocked
            if confidence > 75:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                # 75 이하이면 타인으로 간주하여 Locked
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face Cropper', image)
    except Exception as e:
        # 얼굴 검출 안됨 
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)

    if cv2.waitKey(1) == 13:  # Enter 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
