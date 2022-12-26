#!/usr/bin/env python
import socket #socket을 import
import RPi.GPIO as GPIO
import time
import bitarray
#################################for GPIO setting##########################3
GPIO.setwarnings(False)
TRIG = 17
sensor = 18
####################################################################
#################################for 파일 길이를 전송하기 위한 parameter #################
getbinary = lambda x, n: format(x, 'b').zfill(n) ## int to binary
data = [] ## 보낼 data
hamming_length = []
send_real = [] ## hamming코드로 변환후 실제 보낼 데이터
bit_data = bitarray.bitarray()
######################################전송할 데이터 읽어오는 부분######################
print("Input file name with extension")
filename = input()
######################################전송할 파일이름 지정##########################
with open("./"+filename, 'rb') as file:
    bit_data.fromfile(file)
for value in bit_data:
    data.append(str(value))
send_data = ''.join(data)   #파일열고 send_data에 저장 그뒤 4bit 해밍 코드로 변환
print(len(send_data))
for i in range(0, (len(send_data)) - (len(send_data)%4), 4):
    if(send_data[i:i+4] == "0000"):
        send_real.append("0000000")

    if (send_data[i:i + 4] == "0001"):
        send_real.append("0001011")

    if(send_data[i:i+4] == "0010"):
        send_real.append("0010110")

    if (send_data[i:i + 4] == "0011"):
        send_real.append("0011101")

    if(send_data[i:i+4] == "0100"):
        send_real.append("0100111")

    if (send_data[i:i + 4] == "0101"):
        send_real.append("0101100")

    if(send_data[i:i+4] == "0110"):
        send_real.append("0110001")

    if (send_data[i:i + 4] == "0111"):
        send_real.append("0111010")

    if(send_data[i:i+4] == "1000"):
        send_real.append("1000101")

    if (send_data[i:i + 4] == "1001"):
        send_real.append("1001110")

    if(send_data[i:i+4] == "1010"):
        send_real.append("1010011")

    if (send_data[i:i + 4] == "1011"):
        send_real.append("1011000")

    if(send_data[i:i+4] == "1100"):
        send_real.append("1100010")

    if (send_data[i:i + 4] == "1101"):
        send_real.append("1101001")

    if(send_data[i:i+4] == "1110"):
        send_real.append("1110100")

    if (send_data[i:i + 4] == "1111"):
        send_real.append("1111111")
send_binary = []
for value in send_real:
    for value2 in value:
        send_binary.append(value2) ###########실제 보낼 데이터인 send_binary에 저장.
#####################################################################################
file_length = len(send_binary) ###먼저 receiver에게 실제 보낼 데이터의 길이를 보내줘야함
print(file_length)
binary_value = getbinary(file_length, 16) #binary화 하고
for i in range(0, 16, 4): #똑같이 hamming코드로 전송
    if(binary_value[i:i+4] == "0000"):
        hamming_length.append("0000000")

    if (binary_value[i:i + 4] == "0001"):
        hamming_length.append("0001011")

    if(binary_value[i:i+4] == "0010"):
        hamming_length.append("0010110")

    if (binary_value[i:i + 4] == "0011"):
        hamming_length.append("0011101")

    if(binary_value[i:i+4] == "0100"):
        hamming_length.append("0100111")

    if (binary_value[i:i + 4] == "0101"):
        hamming_length.append("0101100")

    if(binary_value[i:i+4] == "0110"):
        hamming_length.append("0110001")

    if (binary_value[i:i + 4] == "0111"):
        hamming_length.append("0111010")

    if(binary_value[i:i+4] == "1000"):
        hamming_length.append("1000101")

    if (binary_value[i:i + 4] == "1001"):
        hamming_length.append("1001110")

    if(binary_value[i:i+4] == "1010"):
        hamming_length.append("1010011")

    if (binary_value[i:i + 4] == "1011"):
        hamming_length.append("1011000")

    if(binary_value[i:i+4] == "1100"):
        hamming_length.append("1100010")

    if (binary_value[i:i + 4] == "1101"):
        hamming_length.append("1101001")

    if(binary_value[i:i+4] == "1110"):
        hamming_length.append("1110100")

    if (binary_value[i:i + 4] == "1111"):
        hamming_length.append("1111111")
hamming_binary = []
for value in hamming_length:
    for value2 in value:
        hamming_binary.append(value2)

################## hamming_binary에 길이 데이터를 저장하고 send_binary에 실제 데이터를 저장
#########################################################################
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(sensor, GPIO.IN)
GPIO.output(TRIG, 0)
GPIO.output(TRIG, 1)
while True: ###Laser맞으면 0 안맞으면 1
    if(GPIO.input(sensor) == 0):
        print("OK")
        break
GPIO.output(TRIG, 0)
time.sleep(0.022)
for value in hamming_binary: #먼저 길이를 보낸다.
    if int(value) == 0:
        GPIO.output(TRIG, 1)
        time.sleep(0.03)
    else:
        GPIO.output(TRIG, 0)
        time.sleep(0.03)

GPIO.output(TRIG, 0)
GPIO.output(TRIG, 1)
while True: ###Laser맞으면 0 안맞으면 1
    if(GPIO.input(sensor) == 0):
        print("ready for send")
        break
GPIO.output(TRIG, 0)
time.sleep(0.022)
for value in send_binary: #그뒤 실제데이터를 보낸다.
    if int(value) == 0:
        GPIO.output(TRIG, 1)
        time.sleep(0.03)
    else:
        GPIO.output(TRIG, 0)
        time.sleep(0.03)
GPIO.output(TRIG, 0)
#######################################################################파일 전송 완료################
