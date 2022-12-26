#!/usr/bin/env python
import socket #socket을 import
import RPi.GPIO as GPIO
import time
import bitarray
error_length = 0
data = []
result_data = bitarray.bitarray() #파일로 저장하게 될 bitarray
############################################## GPIO setting ########################################
GPIO.setwarnings(False)
TRIG = 17
sensor = 18
real_data = []
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(sensor, GPIO.IN)
data = []
trigger = False
##############################################데이터 길이 전송받는 부분 #########################################33
GPIO.output(TRIG, 0)
while True: ###Laser맞으면 0 안맞으면 1
    if GPIO.input(sensor) == 0:
        print("OK")
        break

GPIO.output(TRIG, 1)
time.sleep(0.03)
GPIO.output(TRIG, 0)
for value in range(28):
        data.append(str(GPIO.input(sensor)))
        time.sleep(0.03)
##############################################데이터 길이 전송 완료 ###############################################
##############################################데이터 전송 가공 ### ################################################
hamming2_list = "".join(data) ############# 길이를 join
hamming2_binary = ''.join(hamming2_list)
hamming2_length = []
#############################################해밍변환 ###########################################################
def hamming(code):
    error_length = error_length + 1
    s1 = int(code[0]) ^ int(code[1]) ^ int(code[2]) ^ int(code[4])
    s2 = int(code[1]) ^ int(code[2]) ^ int(code[3]) ^ int(code[5])
    s3 = int(code[0]) ^ int(code[1]) ^ int(code[3]) ^ int(code[6])
    print(code)
    if(s1 == 1 and s3 == 1 and s2 != 1):
        if code[0] == 0:
            code[0] == "1"
        else:
            code[0] == "0"
        return code[0:4]
    elif(s1 == 1 and s2 == 1 and s3 == 1):
        if code[1] == 0:
            code[1] == "1"
        else:
            code[1] == "0"
        return code[0:4]
    elif(s1 == 1 and s2 == 1 and s3 != 1):
        if code[2] == 0:
            code[2] == "1"
        else:
            code[2] == "0"
        return code[0:4]
    elif(s1 != 1 and s2 == 1 and s3 == 1):
        if code[3] == 0:
            code[3] == "1"
        else:
            code[3] == "0"
        return code[0:4]
    else:
        return code[0:4]
    return code[0:4]

for i in range(0, 28, 7):
    if (hamming2_binary[i:i + 7] == "0000000"):
        hamming2_length.append("0000")

    elif (hamming2_binary[i:i + 7] == "0001011"):
        hamming2_length.append("0001")

    elif (hamming2_binary[i:i + 7] == "0010110"):
        hamming2_length.append("0010")

    elif (hamming2_binary[i:i + 7] == "0011101"):
        hamming2_length.append("0011")

    elif (hamming2_binary[i:i + 7] == "0100111"):
        hamming2_length.append("0100")

    elif (hamming2_binary[i:i + 7] == "0101100"):
        hamming2_length.append("0101")

    elif (hamming2_binary[i:i + 7] == "0110001"):
        hamming2_length.append("0110")

    elif (hamming2_binary[i:i + 7] == "0111010"):
        hamming2_length.append("0111")

    elif (hamming2_binary[i:i + 7] == "1000101"):
        hamming2_length.append("1000")

    elif (hamming2_binary[i:i + 7] == "1001110"):
        hamming2_length.append("1001")

    elif (hamming2_binary[i:i + 7] == "1010011"):
        hamming2_length.append("1010")

    elif (hamming2_binary[i:i + 7] == "1011000"):
        hamming2_length.append("1011")

    elif (hamming2_binary[i:i + 7] == "1100010"):
        hamming2_length.append("1100")

    elif (hamming2_binary[i:i + 7] == "1101001"):
        hamming2_length.append("1101")

    elif (hamming2_binary[i:i + 7] == "1110100"):
        hamming2_length.append("1110")

    elif (hamming2_binary[i:i + 7] == "1111111"):
        hamming2_length.append("1111")
    else:
        hamming2_length.append(hamming(hamming2_binary[i:i+7]))
length = []
for value in hamming2_length:
    for value2 in value:
        length.append(value2)
length = "".join(length)
length = int(length, 2)
print(length)
##############################################데이터 전송 가공 완료 ################################################
##############################################실제 데이터 전송 ####################################################
GPIO.output(TRIG, 0)

time.sleep(3)
while True: ###Laser맞으면 0 안맞으면 1
    if GPIO.input(sensor) == 0:
        print("OK")
        break

GPIO.output(TRIG, 1)
time.sleep(0.03)
GPIO.output(TRIG, 0)
for value in range(length):
        real_data.append(str(GPIO.input(sensor)))
        time.sleep(0.03)
data_list = "".join(real_data) ############# 길이를 join
data_binary = ''.join(data_list)
data_length = []

for i in range(0, length, 7):
    if (data_binary[i:i + 7] == "0000000"):
        data_length.append("0000")

    elif (data_binary[i:i + 7] == "0001011"):
        data_length.append("0001")

    elif (data_binary[i:i + 7] == "0010110"):
        data_length.append("0010")

    elif (data_binary[i:i + 7] == "0011101"):
        data_length.append("0011")

    elif (data_binary[i:i + 7] == "0100111"):
        data_length.append("0100")

    elif (data_binary[i:i + 7] == "0101100"):
        data_length.append("0101")

    elif (data_binary[i:i + 7] == "0110001"):
        data_length.append("0110")

    elif (data_binary[i:i + 7] == "0111010"):
        data_length.append("0111")

    elif (data_binary[i:i + 7] == "1000101"):
        data_length.append("1000")

    elif (data_binary[i:i + 7] == "1001110"):
        data_length.append("1001")

    elif (data_binary[i:i + 7] == "1010011"):
        data_length.append("1010")

    elif (data_binary[i:i + 7] == "1011000"):
        data_length.append("1011")

    elif (data_binary[i:i + 7] == "1100010"):
        data_length.append("1100")

    elif (data_binary[i:i + 7] == "1101001"):
        data_length.append("1101")

    elif (data_binary[i:i + 7] == "1110100"):
        data_length.append("1110")

    elif (data_binary[i:i + 7] == "1111111"):
        data_length.append("1111")
    else:
        data_length.append(hamming(data_binary[i:i + 7]))

result = bitarray.bitarray()
for value in data_length:
    for value2 in value:
        result.append(int(value2))
print(len(result))
print(error_length)

print("Input file name with extension")
filename = input()
with open("./"+filename, 'wb') as fh:
    result.tofile(fh)

