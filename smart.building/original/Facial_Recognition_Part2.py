import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# 데이터 경로 설정 (D:\\ 또는 D:/ 모두 사용 가능)
data_path = "D:/bmproject/Facial-Recognition-master/faces/"

# faces 폴더에 있는 파일 리스트 얻기
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# 데이터와 매칭될 라벨 변수
Training_Data, Labels = [], []

# 파일 개수 만큼 루프
for i, file in enumerate(onlyfiles):    
    # 경로와 파일 이름을 결합하여 전체 이미지 경로를 생성
    image_path = join(data_path, file)
    
    # 이미지 불러오기 (흑백 모드로)
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 이미지 파일이 아니거나, 파일을 읽을 수 없다면 무시
    if images is None:
        print(f"Image {image_path} could not be loaded.")
        continue
    
    # Training_Data 리스트에 이미지를 바이트 배열로 추가
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    
    # Labels 리스트에 해당 이미지의 인덱스 번호 추가
    Labels.append(i)

# 훈련할 데이터가 없다면 종료
if len(Labels) == 0:
    print("There is no data to train.")
    exit()

# Labels를 32비트 정수로 변환
Labels = np.asarray(Labels, dtype=np.int32)

# 모델 생성 (LBPH 얼굴 인식 모델)
model = cv2.face.LBPHFaceRecognizer_create()

# 모델 학습 시작
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!!!!!")
