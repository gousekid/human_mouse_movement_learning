import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

# 모델 불러오기 (이 부분은 실제 실행 환경에서 수행되어야 합니다)
model = tf.keras.models.load_model('./model/mouseprediction.keras')

def predict_mouse_movement(model, start_point, target_point, max_steps=100):
    """
    주어진 시작점에서 목표점까지의 마우스 이동 예측
    
    Args:
    - model: 학습된 LSTM 모델
    - start_point: 시작점 (x, y)
    - target_point: 목표점 (x, y)
    - max_steps: 최대 예측 스텝
    
    Returns:
    - 예측된 경로
    """
    predicted_path = [start_point]
    current_point = np.array(start_point).reshape(1, 1, 2)  # 초기 입력값 준비
    
    for _ in range(max_steps):
        # 모델 예측
        predicted_point = model.predict(current_point)[0]
        
        # 예측된 경로에 추가
        predicted_path.append(predicted_point)
        
        # 목표점에 도달하면 종료
        if np.linalg.norm(predicted_point - target_point) < 0.05:
            break
            
        # 다음 예측을 위한 입력값 준비
        current_point = np.array(predicted_point).reshape(1, 1, 2)
    
    return np.array(predicted_path)

# 예측 수행 (이 부분은 실제 실행 환경에서 수행되어야 합니다)
start = [0.2, 0.2]
target = [0.8, 0.8]
predicted_path = predict_mouse_movement(model, start, target)

# 결과 도식화
plt.figure(figsize=(8, 8))
plt.plot(predicted_path[:, 0], predicted_path[:, 1], '-o', label='Predicted Path')
plt.scatter(*target, color='red', label='Target Point')
plt.title('Predicted Mouse Movement')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()