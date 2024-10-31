import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import tensorflow as tf

# CNN 모델 생성 함수
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))  # 분류할 사용자 수에 맞게 조정

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 얼굴 인식기 학습을 위한 함수 (CNN)
def train(name):
    data_path = "face/" + name + '/'
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []

    for i, file in enumerate(face_pics):
        image_path = data_path + face_pics[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if images is None:
            continue

        # CNN 모델에 맞게 크기 조정 (200x200)
        images = cv2.resize(images, (200, 200))
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    if len(Labels) == 0:
        print(f"{name}에 대한 데이터가 충분하지 않습니다.")
        return None

    Training_Data = np.asarray(Training_Data)
    Labels = np.asarray(Labels)

    # 데이터를 CNN 모델에 맞게 reshape (samples, height, width, 1)
    Training_Data = Training_Data.reshape(Training_Data.shape[0], 200, 200, 1)
    Labels = np.asarray(Labels)

    # 데이터를 학습용과 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(Training_Data, Labels, test_size=0.2)

    # CNN 모델 생성
    cnn_model = create_cnn_model((200, 200, 1))

    # 모델 학습
    cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    print(f"{name} 님의 모델 학습이 완료되었습니다!")

    # 모델 저장
    model_dir = "model/"
    if not isdir(model_dir):
        makedirs(model_dir)

    cnn_model.save(model_dir + name + '_cnn_model.h5')

    return cnn_model

# 모든 사용자에 대해 모델을 학습하는 함수
def train_all_users():
    face_dir = "face/"
    users = [f for f in listdir(face_dir) if isdir(join(face_dir, f))]

    for user in users:
        print(f'{user} 사용자의 모델을 학습 중입니다...')
        train(user)

if __name__ == "__main__":
    train_all_users()
