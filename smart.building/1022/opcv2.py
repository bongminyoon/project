import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir

# 조명 보정을 위한 CLAHE 적용 함수
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

# 사용자별로 모델을 학습하는 함수
def train(name):
    data_path = "face/"  + name + '/'
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []

    for i, file in enumerate(face_pics):
        image_path = data_path + face_pics[i]   
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if images is None:
            continue

        images = apply_clahe(images)  # 조명 보정
        Training_Data.append(np.asarray(images, dtype=np.uint8))
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
