import socket
import time
from threading import Thread
from configs.config import *


g_open = b"\xA0\x01\x01\xA2"
g_close = b"\xA0\x01\x00\xA1"

y_open = b"\xA0\x02\x01\xA3"
y_close = b"\xA0\x02\x00\xA2"

r_open = b"\xA0\x03\x01\xA4"
r_close = b"\xA0\x03\x00\xA3"


alarm_signal_dict = signal_dict.copy()


class Alarm:
    def __init__(self, q_thread):
        self.q_thread = q_thread
        self.flag = {}
        self.green = {}
        self.init()

    def init(self):
        for name in alarm_signal_dict.keys():
            self.flag[name] = False
            self.green[name] = True

    def connect(self, server_name, server_port, name):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((server_name, server_port))
            self.flag[name] = True
            while True:
                if alarm_signal_dict[name] == 1:
                    self.green[name] = True
                    s.send(g_close)
                    time.sleep(0.1)
                    s.send(y_close)
                    time.sleep(0.1)
                    s.send(r_open)
                    time.sleep(0.7)
                    s.send(r_close)
                    time.sleep(0.5)
                if alarm_signal_dict[name] == 2:
                    self.green[name] = True
                    s.send(r_close)
                    time.sleep(0.1)
                    s.send(g_close)
                    time.sleep(0.1)
                    s.send(y_open)
                    time.sleep(0.7)
                    s.send(y_close)
                    time.sleep(0.5)
                if alarm_signal_dict[name] == 0:
                    if self.green[name]:
                        self.green[name] = False
                        s.send(r_close)
                        time.sleep(0.1)
                        s.send(y_close)
                        time.sleep(0.1)
                        s.send(g_open)
                    time.sleep(0.5)
        except BaseException:
            self.flag[name] = False
            self.q_thread.text_append.emit(station_name_dict[name] + '：报警灯断开连接')

    def check(self):
        for name in alarm_signal_dict.keys():
            thread = Thread(target=self.connect, args=(server_name_dict[name], server_port, name, ))
            thread.start()
        while True:
            time.sleep(30)
            for name in alarm_signal_dict.keys():
                if not self.flag[name]:
                    thread = Thread(target=self.connect, args=(server_name_dict[name], server_port, name, ))
                    thread.start()


def send_signal(judgment_dict: dict, preds_dict: dict):
    for name in alarm_signal_dict.keys():
        if judgment_dict[name]:
            alarm_signal_dict[name] = 1
        elif preds_dict[name]:
            alarm_signal_dict[name] = 2
        else:
            alarm_signal_dict[name] = 0
