import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from datetime import datetime  
import socket

# DNN 기반 얼굴 검출 모델 로드 함수
def load_dnn_model():
    model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_file = "deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    return net

# DNN 기반 얼굴 검출 함수
def face_extractor_dnn(img, net):
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    h, w = img.shape[:2]
    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cropped_face = img[y:y1, x:x1]
            faces.append(cropped_face)

    if len(faces) > 0:
        return faces[0]  # 첫 번째 얼굴 반환
    else:
        return None

# 학습된 모델 로드 함수
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

# 얼굴 인식 및 결과 표시 함수
def run(models, stranger_dir):
    net = load_dnn_model()  # DNN 모델 로드
    cap = cv2.VideoCapture(0)  # 기본 웹캠 사용

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    confidence_suc = 0
    confidence_fai = 0
    confidence_cnt = 0

    window_name = 'Face Recognition'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 600)

    # 소켓 연결 설정
    HOST = '192.168.0.15'  # 서버의 IP 주소
    PORT = 9000  # 서버의 포트 번호

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b'FR:room_201')  # 초기 메시지 전송

        while True:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break

            original_frame = cv2.resize(frame, (512, 600))  
            face = face_extractor_dnn(frame, net)  # DNN을 사용한 얼굴 검출

            try:
                min_score = 999
                min_score_name = ""
                confidence = 0

                if face is not None:
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (200, 200))

                    for name, model in models.items():
                        result = model.predict(face_resized)
                        if min_score > result[1]:
                            min_score = result[1]
                            min_score_name = name

                    if min_score < 500:
                        confidence = int(100 * (1 - (min_score) / 300))
                        display_string = str(confidence) + '% ' + min_score_name
                    else:
                        display_string = "잠금 상태"

                    cv2.putText(frame, display_string, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    confidence_cnt += 1

                    if confidence_cnt < 21:
                        if confidence >= 85:
                            cv2.putText(frame, "Unlocked - " + min_score_name, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            confidence_suc += 1
                            if confidence_suc == 15:
                                print("얼굴 인식 성공")
                                s.sendall(b'FR:room201:success')  # 성공 메시지 전송
                        else:
                            cv2.putText(frame, "Locked", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            confidence_fai += 1
                            if confidence_fai == 5:
                                current_time = get_current_time_str()
                                failed_img_path = join(stranger_dir, f'{current_time}.jpg')
                                cv2.imwrite(failed_img_path, frame)
                                print(f"Failed capture saved at: {failed_img_path}")
                                s.sendall(f'FR:room201:failed:{failed_img_path}'.encode())  # 실패 메시지 전송
                    else:
                        confidence_suc = 0
                        confidence_fai = 0
                        confidence_cnt = 0

                # 화면에 결과 출력
                if frame is not None:
                    image_resized = cv2.resize(frame, (512, 600))  
                else:
                    image_resized = np.zeros((600, 512, 3), dtype=np.uint8)  

                combined_frame = np.hstack((original_frame, image_resized))  
                cv2.imshow(window_name, combined_frame)  

            except Exception as e:
                print(f"Error: {str(e)}")

            if cv2.waitKey(1) == 13:  # Enter 키로 종료
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

