import socket
import matplotlib.pyplot as plt
from itertools import count
import pylab as pl
from IPython import display
from datashape import null


x_val = [] #x값을 위한 array
y_val = [] #y값을 위한 array
index = count() #x값을 y값에 대응해서 하나씩 올리는 변수와 함수
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('192.168.137.1', 8080))    # 서버노트북의 ip주소, 포트번호 지정
server_socket.listen(0)                          # 클라이언트의 연결요청을 기다리는 상태

client_socket, addr = server_socket.accept()     # 연결 요청을 수락함. 그러면 아이피주소, 포트등 데이터를 return


while True:
    data = client_socket.recv(100) #recv안의 int만큼의 data buffer를 통해 데이터 통신
    msg = data.decode() #byte단위로 온 데이터를 복호화
    if msg != null: # 기본적으로 무한루프이기때문에 의미없는 데이터가 송신되지만 거리정보를 받을때 그래프를 새로 그리기 위한 조건문
        msg2 = float(msg) #decode의 기본값은 str이기때문에 msg를 float화
        print('recv msg:', msg) # 잘 받는지 확인하기 위한 출력문
        x_val.append(next(index)) #x값을 array에 추가
        y_val.append(round(msg2, 2)) #받아온 거리정보를 소수 2째짜리까지 출력
        plt.clf() #pyplotlib의 캔버스를 제외한 나머지 그래프를 초기화 (초기화후 새로 그려야 그래프 중복이 안되서 필수)
        plt.plot(x_val[-20:], y_val[-20:]) #x값과 y값을 최근 20개까지만 그래프화
        display.clear_output(wait=True)
        display.display(pl.gcf()) #그래프를 그림
        plt.pause(0.1)
    client_socket.sendall(msg.encode(encoding='utf-8'))
    if msg == '/end':
        break
server_socket.close()