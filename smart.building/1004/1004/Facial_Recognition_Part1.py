import cv2
import numpy as np
import os

# 얼굴 인식용 xml 파일
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 전체 사진에서 얼굴 부위만 잘라 리턴
def face_extractor(img):
    # 흑백처리 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 얼굴 찾기 
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    # 찾은 얼굴이 없으면 None으로 리턴 
    if len(faces) == 0:
        return None
    # 얼굴들이 있으면 
    for (x, y, w, h) in faces:
        # 해당 얼굴 크기만큼 cropped_face에 잘라 넣기 
        cropped_face = img[y:y+h, x:x+w]
    # cropped_face 리턴 
    return cropped_face

# 저장할 경로 설정
save_path = "D:\\bmproject\\Facial-Recognition-master\\faces\\"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 기존에 저장된 사진 개수 확인
existing_files = [f for f in os.listdir(save_path) if f.startswith('users') and f.endswith('.jpg')]
count = len(existing_files)

# 카메라 실행 
cap = cv2.VideoCapture(0)

while count < 1000:  # 최대 1000장까지 저장
    current_session_count = 0  # 현재 세션에서 저장할 이미지 수

    print(f"\n현재 저장된 사진: {count}장, 새로 100장 촬영합니다.")
    
    while current_session_count < 100 and count < 1000:
        # 카메라로부터 사진 1장 얻기 
        ret, frame = cap.read()
        # 얼굴 감지 하여 얼굴만 가져오기 
        face = face_extractor(frame)
        
        if face is not None:
            count += 1
            current_session_count += 1
            
            # 얼굴 이미지 크기를 200x200으로 조정 
            face = cv2.resize(face, (200, 200))
            # 조정된 이미지를 흑백으로 변환 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # faces 폴더에 jpg파일로 저장 
            file_name_path = os.path.join(save_path, f'users{count}.jpg')
            cv2.imwrite(file_name_path, face)
            
            # 화면에 얼굴과 현재 저장 개수 표시          
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face not Found")

        # Enter 키로 종료 또는 1000장 이상 찍으면 종료
        if cv2.waitKey(1) == 13 or count >= 1000:
            break

    print(f"이번 촬영 완료. {current_session_count}장 저장되었습니다.")

    # 100장이 저장된 후 자동으로 종료
    if current_session_count >= 100:
        print("100장이 저장되었습니다. 프로그램을 종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
print('Collecting Samples Complete!!!')
