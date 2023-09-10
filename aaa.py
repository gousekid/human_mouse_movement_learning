import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import os
  
def preprocess_data_with_normalization(filename):
    # CSV 파일 읽기
    data = pd.read_csv(filename)
    
    # x, y 좌표의 최대값과 최소값을 사용하여 정규화
    x_min, x_max = data['x'].min(), data['x'].max()
    y_min, y_max = data['y'].min(), data['y'].max()
    data['x'] = (data['x'] - x_min) / (x_max - x_min)
    data['y'] = (data['y'] - y_min) / (y_max - y_min)
    
    # 데이터 전처리를 위한 결과 리스트 초기화
    sequences = []
    
    # session_id 별로 데이터 분리
    for session in data['session_id'].unique():
        session_data = data[data['session_id'] == session]
        
        # start부터 end까지의 시퀀스 찾기
        start_idx = session_data.index[session_data['label'] == 'start'][0]
        end_idx = session_data.index[session_data['label'] == 'end'][0]
        
        sequence = session_data.loc[start_idx:end_idx, ['timestamp', 'x', 'y']].values
        sequences.append(sequence)
    
    # 모든 시퀀스의 최대 길이 찾기
    max_len = max(len(seq) for seq in sequences)
    
    # 패딩을 위한 기본값 설정: -1로 설정하면 패딩된 부분을 나중에 구분하기 쉽습니다.
    default_value = [-1, -1, -1]
    
    # 각 시퀀스에 대해 패딩 수행
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        if pad_len != 0:
            padded_seq = np.vstack([seq, [default_value] * pad_len])
            #padded_sequences.append(padded_seq)
        else :
            padded_seq = seq
            
        padded_sequences.append(padded_seq)
        
    return np.array(padded_sequences)

# 예제 CSV 파일을 사용해 전처리 함수를 테스트합니다.
filename = "./data/mouse_data_20230910_220147.csv"
preprocessed_data_normalized = preprocess_data_with_normalization(filename)


# Split the data into training and testing sets
train_size = int(0.8 * len(preprocessed_data_normalized))
train_data = preprocessed_data_normalized[:train_size]
test_data = preprocessed_data_normalized[train_size:]

# Split the data into input and target
X_train = train_data[:, :-1, 1:]
y_train = train_data[:, -1, 1:]
X_test = test_data[:, :-1, 1:]
y_test = test_data[:, -1, 1:]

model = Sequential()
    
# LSTM layer
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64, return_sequences=False))

# Dense layer for regression output
model.add(Dense(2))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

model.save('./model/mouseprediction.keras')