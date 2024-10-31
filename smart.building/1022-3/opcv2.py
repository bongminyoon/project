import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir

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

# 사용자별로 모델을 학습하는 함수
def train(name):
    data_path = "face/"  + name + '/'
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []

    net = load_dnn_model()  # DNN 모델 로드

    for i, file in enumerate(face_pics):
        image_path = data_path + face_pics[i]   
        image = cv2.imread(image_path)

        if image is None:
            continue

        # DNN을 사용한 얼굴 검출
        face = face_extractor_dnn(image, net)

        if face is None:
            print(f"얼굴을 검출할 수 없습니다: {file}")
            continue

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_gray = apply_clahe(face_gray)  # 조명 보정
        Training_Data.append(np.asarray(face_gray, dtype=np.uint8))
        Labels.append(i)

    if len(Labels) == 0:
        print("데이터가 충분하지 않습니다.")
        return None

    Labels = np.asarray(Labels, dtype=np.int32)

    # LBPH 얼굴 인식기 생성 및 학습
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print(name + " : 모델 학습이 완료되었습니다!")

    # 학습된 모델 저장
    model_dir = "model/"
    if not isdir(model_dir):
        makedirs(model_dir)

    model.save(model_dir + name + '_model.xml')

    return model

# 모든 사용자를 학습하는 함수
def trains():
    data_path = "face/"
    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path, f))]

    models = {}
    for model_name in model_dirs:
        print('모델 학습 중: ' + model_name)
        result = train(model_name)
        if result is None:
            continue
        models[model_name] = result

    return models

if __name__ == "__main__":
    trains()
