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

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b'FR:room201')
        while True:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break

    
            original_frame = cv2.resize(frame, (512, 600))  
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
                                s.sendall(b'FR:room201:sucess')
                               
                        else:
                            cv2.putText(image, "Locked", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            confidence_fai += 1
                            if confidence_fai == 5:
                                current_time = get_current_time_str()	
                                failed_img_path = join(stranger_dir, f'{current_time}.jpg')
                                cv2.imwrite(failed_img_path, frame)
                                s.sendall(f'FR:room201:failed:{failed_img_path}'.encode())

                                
                                
                    else:
                        confidence_suc = 0
                        confidence_fai = 0
                        confidence_cnt = 0

            
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
    models = load_models()
    if len(models) == 0:
        print("학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요.")
    else:
        run(models, stranger_dir)

