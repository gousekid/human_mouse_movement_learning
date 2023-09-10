import tkinter as tk
from pynput.mouse import Listener
import random
import csv
import uuid
import time

# 데이터 저장 리스트
data = []

# 원을 랜덤한 위치에 그리는 함수
def draw_random_circle():
    global circle_coords
    canvas.delete("all")
    circle_coords = (random.randint(circle_radius, 1890), random.randint(circle_radius, 1050))
    canvas.create_oval(circle_coords[0]-circle_radius, circle_coords[1]-circle_radius,
                       circle_coords[0]+circle_radius, circle_coords[1]+circle_radius, fill='blue')

# 원이 클릭되었는지 확인하는 함수
def check_inside_circle(event):
    global session_id
    if event.num == 3:  # 우클릭 이벤트
        root.quit()
    else:
        distance = ((event.x - circle_coords[0]) ** 2 + (event.y - circle_coords[1]) ** 2) ** 0.5
        if distance <= circle_radius:
            timestamp = time.time()
            data.append((timestamp, event.x, event.y, 'end', session_id))
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in data:
                    writer.writerow(row)
            data.clear()  # 데이터 초기화
            draw_random_circle()  # 새로운 원 그리기
            session_id = str(uuid.uuid4())  # 새로운 세션 ID 생성

# 마우스 움직임을 감지하는 함수
def on_move(x, y):
    timestamp = time.time()
    if not data:
        data.append((timestamp, x, y, 'start', session_id))
    else:
        data.append((timestamp, x, y, 'path', session_id))

# GUI 창 생성
root = tk.Tk()
root.title("Mouse Tracking")
canvas = tk.Canvas(root, bg="white", width=1920, height=1080)
canvas.pack(pady=20)

circle_radius = 30
draw_random_circle()

canvas.bind("<Button-1>", check_inside_circle)
canvas.bind("<Button-3>", check_inside_circle)  # 우클릭 이벤트 바인딩

# 세션 ID 생성
session_id = str(uuid.uuid4())

# 파일 제목에 프로그램 시작 시간 포함
start_time = time.strftime('%Y%m%d_%H%M%S')
filename = f"./data/mouse_data_{start_time}.csv"

# 파일에 헤더 추가 (파일이 없을 경우)
try:
    with open(filename, 'r') as f:
        pass
except FileNotFoundError:
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "x", "y", "label", "session_id"])

# 마우스 리스너 시작
listener = Listener(on_move=on_move)
listener.start()

root.mainloop()
