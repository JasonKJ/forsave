#!/usr/bin/env python
import socket #socket을 import
import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
TRIG = 17
ECHO = 18
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('192.168.137.1', 8080)) # server컴퓨터의 ip와 열어둔 port입력


def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)


def distance():
    GPIO.output(TRIG, 0)
    time.sleep(0.000002)

    GPIO.output(TRIG, 1)
    time.sleep(0.00001)
    GPIO.output(TRIG, 0)

    while GPIO.input(ECHO) == 0:
        a = 0
    time1 = time.time()
    while GPIO.input(ECHO) == 1:
        a = 1
    time2 = time.time()

    during = time2 - time1
    return during * 340 / 2 * 100


def animate():
    dis = distance() #distance를 받아와서
    sock.sendall(str(dis).encode('utf-8')) #distance를 서버로 보내기위해 string화 후 encode
    data = sock.recv(1024) #데이터 수신이 잘되었는지 서버로 부터 수신받는 기능
    print('Received', repr(data.decode())) #송수신이 완료되면 보낸정보를 출력.


def destroy():
    GPIO.cleanup()


if __name__ == "__main__":
    setup()
    try:

        while True:
            animate()

    except KeyboardInterrupt:
        destroy()
