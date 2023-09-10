import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. 저장된 모델 불러오기
model_path = './model/mouseprediction.keras'
loaded_model = load_model(model_path)

# 2. 특정 시작 위치에서 목표 위치까지의 예측 경로 계산
def predict_path_corrected(model, start, target, max_length):
    current_point = start
    path = [current_point]

    # 예측을 반복하여 경로 생성 (예: 50번 반복)
    for _ in range(50):
        input_data = [current_point[0], current_point[1]]
        
        # 입력 데이터를 모델이 기대하는 형태로 변환
        padded_input = np.zeros((1, max_length, 1))
        for i in range(len(input_data)):
            padded_input[0, i, 0] = input_data[i]
        
        predicted_offset = model.predict(padded_input)
        
        # 예측된 변화량을 현재 위치에 추가
        next_point = [current_point[0] + predicted_offset[0][0], current_point[1] + predicted_offset[0][1]]
        path.append(next_point)
        
        # 목표 위치에 도달하면 중단
        if np.linalg.norm(np.array(next_point) - np.array(target)) < 5:  # 5는 임계값, 조정 가능
            break

        current_point = next_point

    return path

start_point = [10, 10]  # 예: [10, 10]
target_point = [200, 200]  # 예: [200, 200]

input_shape = loaded_model.input_shape[1]
predicted_path_corrected = predict_path_corrected(loaded_model, start_point, target_point, input_shape)


# 도식화
x_coords, y_coords = zip(*predicted_path_corrected)

plt.figure(figsize=(10, 10))
plt.plot(x_coords, y_coords, '-o', label='Predicted Path')
plt.scatter(*target_point, color='red', label='Target Point')
plt.legend()
plt.grid(True)
plt.title('Predicted Mouse Movement Path')
plt.show()