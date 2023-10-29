import socket
import cv2
import numpy as np
from PIL import Image


class socket_connection():
    """
    Class for creating socket connection and retrieving images
    """
    def __init__(self, ip, port, camera):
        """
        Init of vars and creating socket connection object.
        Based on user input a different camera can be selected.
        1: Stereo camera 1280*360
        2: Stereo camera 2560*720
        3: Mono camera 320*240
        4: Mono camera 640*480
        """
        # Camera selection
        if camera == 1:
            self.size = 1382400
            self.width = 1280
            self.height = 360
            self.cam_id = 3
            self.res_id = 14
        elif camera == 2:
            self.size = 5529600
            self.width = 2560
            self.height = 720
            self.cam_id = 3
            self.res_id = 13
        elif camera == 3:
            self.size = 230400
            self.width = 320
            self.height = 240
            self.cam_id = 0
            self.res_id = 1
        elif camera == 4:
            self.size = 921600
            self.width = 640
            self.height = 480
            self.cam_id = 0
            self.res_id = 2
        else:
            print("Invalid camera selected... choose between 1 and 4")

        self.COLOR_ID = 13
        self.ip = ip
        self.port = port

        # Initialize socket socket connection
        self.s = socket.socket()
        try:
            self.s.connect((self.ip, self.port))
            print("Successfully connected with {}:{}".format(self.ip, self.port))
        except:
            print("ERR: Failed to connect with {}:{}".format(self.ip, self.port))
            exit(1)


    def get_img(self):
        """
        Send signal to pepper to recieve image data, and convert to image data
        """
        self.s.send(b'getImg')
        pepper_img = b""

        l = self.s.recv(self.size - len(pepper_img))
        while len(pepper_img) < self.size:
            pepper_img += l
            l = self.s.recv(self.size - len(pepper_img))

        im = Image.frombytes("RGB", (self.width, self.height), pepper_img)
        cv_image = cv2.cvtColor(np.asarray(im, dtype=np.uint8), cv2.COLOR_BGRA2RGB)

        return cv_image[:, :, ::-1]

    def close_connection(self):
        """
        Close socket connection after finishing
        """
        return self.s.close()


if __name__ == '__main__':
    connect = socket_connection(ip='192.168.0.196', port=12345, camera=3)
    while True:
        img = connect.get_img()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('pepper stream', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    connect.close_connection()
