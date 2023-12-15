import serial

class Arduino_com():
    def __init__(self, port):
        self.ser = serial.Serial('/dev/'+ port, 9600, timeout=1)

    def send_concentration(self, value):
        try:
            value_str = str(value)
            self.ser.write(value_str.encode())

        except Exception as e:
            print(f"Error: {e}")