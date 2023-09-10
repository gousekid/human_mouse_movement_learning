import pandas as pd
import time

def preprocess_mouse_data(data_path):
    # 데이터 로드
    df = pd.read_csv(data_path)

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

    return processed_data

# 데이터 전처리 실행
data_path = "./data/mouse_data_20230910_151231.csv"
processed_data = preprocess_mouse_data(data_path)

# 전처리된 데이터의 첫 번째 세션 확인
print(processed_data[0])
