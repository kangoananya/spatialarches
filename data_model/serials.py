import serial
ser = serial.Serial('COM3',115200)
print(ser.write('stat'))