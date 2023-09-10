import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# 데이터 준비
def preprocess_mouse_data_with_padding(merged_data):
    # 데이터 로드
    df = merged_data

    # 전처리 결과를 저장할 리스트
    processed_data = []

    # 세션 ID 별로 데이터 분리
    session_ids = df['session_id'].unique()

    for session_id in session_ids:
        session_data = df[df['session_id'] == session_id]
        
        # 시작점과 끝점 추출
        start_point = session_data[session_data['label'] == 'start'][['x', 'y']].values[0]
        end_point = session_data[session_data['label'] == 'end'][['x', 'y']].values[0]
        
        # 이동 경로와 시간 간격 추출
        path_data = session_data[session_data['label'] == 'path']
        path_coords = path_data[['x', 'y']].values.tolist()
        time_intervals = path_data['timestamp'].diff().fillna(0).values.tolist()
        
        processed_data.append({
            'start_point': start_point,
            'end_point': end_point,
            'path': path_coords,
            'time_intervals': time_intervals
        })

    # 입력 데이터와 목표 데이터를 저장할 리스트
    X_list = []
    Y = []

    for session in processed_data:
        start_x, start_y = session['start_point']
        path = session['path']
        
        # 시작점에서의 x, y 좌표와 경로를 따라가는 동안의 x, y 좌표의 변화량들
        input_data = [start_x, start_y]
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            input_data.extend([dx, dy])
        X_list.append(input_data)
        
        # 목표 데이터 (끝점의 x, y 좌표)
        Y.append(session['end_point'])

    # 패딩
    X_padded = pad_sequences(X_list, padding='post', dtype='float32')

    # numpy 배열로 변환
    Y = np.array(Y)

    return X_padded, Y

# 데이터 로드 및 전처리


# 1. 'data' 폴더 내의 모든 CSV 파일 목록 가져오기
data_folder = './data'
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# 2. 각 CSV 파일을 로드하고, 하나의 큰 데이터 프레임으로 합치기
all_data = []
for csv_file in csv_files:
    file_path = os.path.join(data_folder, csv_file)
    df = pd.read_csv(file_path)
    all_data.append(df)

merged_data = pd.concat(all_data, ignore_index=True)



# 데이터 로드 및 전처리
X, Y = preprocess_mouse_data_with_padding(merged_data)

# 데이터 형태 확인
X.shape, Y.shape
# 데이터 형태 조정
X_padded_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

# 모델 구성
model = Sequential()
model.add(LSTM(50, input_shape=(X_padded_reshaped.shape[1], 1)))
model.add(Dense(2, activation='linear'))  # 2개의 출력 (x, y 좌표)

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X_padded_reshaped, Y, epochs=100, batch_size=5)

# 예측
predictions = model.predict(X_padded_reshaped)

model.save('./model/mouseprediction.h5')