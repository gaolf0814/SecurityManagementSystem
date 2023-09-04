import time
import sys
import logging
import os
import numpy as np
import configparser

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QImage, QPixmap

# from gui.main_window import Ui_MainWindow
from main_window_tr import Ui_MainWindow
from detect import detect_main, change_vis_stream
from configs.config import station_name_dict, station_name_switch_dict, \
    site_1, site_2, site_3, site_4, site_5, site_6, dynamic_config_path, interrupt_switch_section

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, detection_flag):
        super().__init__()
        self.detection_flag = detection_flag
        self.setupUi(self)
        self.showFullScreen()
        self.textBrowser.append(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + '启动检测...')
        self.statusbar.showMessage("系统初始化...")
        self.init_interrupt()

        th = DetectionThread(self)
        th.video_1_change_pixmap.connect(self.set_frame_1)
        th.video_2_change_pixmap.connect(self.set_frame_2)
        th.video_3_change_pixmap.connect(self.set_frame_3)
        th.video_4_change_pixmap.connect(self.set_frame_4)
        th.video_5_change_pixmap.connect(self.set_frame_5)
        th.video_6_change_pixmap.connect(self.set_frame_6)
        th.record_change_pixmap.connect(self.set_record)
        th.text_append.connect(self.append_text)
        th.status_update.connect(self.update_status_message)
        th.start()

        self.pushButton_1.clicked.connect(self.switch_vis_stream_1)
        self.pushButton_2.clicked.connect(self.switch_vis_stream_2)
        self.pushButton_3.clicked.connect(self.switch_vis_stream_3)
        self.pushButton_4.clicked.connect(self.switch_vis_stream_4)
        self.pushButton_5.clicked.connect(self.switch_vis_stream_5)
        self.pushButton_6.clicked.connect(self.switch_vis_stream_6)
        self.action_stop.triggered.connect(self.process_exit)
        self.action_full_screen.triggered.connect(self.showFullScreen)
        self.action_exit_full.triggered.connect(self.showNormal)

        self.action_site_1.triggered.connect(self.site_1_switch)
        self.action_site_2.triggered.connect(self.site_2_switch)
        self.action_site_3.triggered.connect(self.site_3_switch)
        self.action_site_4.triggered.connect(self.site_4_switch)
        self.action_site_5.triggered.connect(self.site_5_switch)
        self.action_site_6.triggered.connect(self.site_6_switch)

    def read_interrupt(self, name):
        config = configparser.ConfigParser()
        config.read(dynamic_config_path)
        return config.getboolean(interrupt_switch_section, name), config

    def write_interrupt(self, key, value, config):
        config.set(interrupt_switch_section, key, str(value))
        o = open(dynamic_config_path, 'w')
        config.write(o)

    def init_interrupt(self):
        judge1, _ = self.read_interrupt(site_1)
        judge2, _ = self.read_interrupt(site_2)
        judge3, _ = self.read_interrupt(site_3)
        judge4, _ = self.read_interrupt(site_4)
        judge5, _ = self.read_interrupt(site_5)
        judge6, _ = self.read_interrupt(site_6)
        if not judge1:
            self.action_site_1.setText(station_name_dict[site_1])
        else:
            self.action_site_1.setText(station_name_switch_dict[site_1])
        if not judge2:
            self.action_site_2.setText(station_name_dict[site_2])
        else:
            self.action_site_2.setText(station_name_switch_dict[site_2])
        if not judge3:
            self.action_site_3.setText(station_name_dict[site_3])
        else:
            self.action_site_3.setText(station_name_switch_dict[site_3])
        if not judge4:
            self.action_site_4.setText(station_name_dict[site_4])
        else:
            self.action_site_4.setText(station_name_switch_dict[site_4])
        if not judge5:
            self.action_site_5.setText(station_name_dict[site_5])
        else:
            self.action_site_5.setText(station_name_switch_dict[site_5])
        if not judge6:
            self.action_site_6.setText(station_name_dict[site_6])
        else:
            self.action_site_6.setText(station_name_switch_dict[site_6])

    @pyqtSlot(QImage, str)
    def set_frame_1(self, image, name):
        self.video_display_1.setPixmap(QPixmap.fromImage(image))
        self.video_title_1.setText(name)

    @pyqtSlot(QImage, str)
    def set_frame_2(self, image, name):
        self.video_display_2.setPixmap(QPixmap.fromImage(image))
        self.video_title_2.setText(name)

    @pyqtSlot(QImage, str)
    def set_frame_3(self, image, name):
        self.video_display_3.setPixmap(QPixmap.fromImage(image))
        self.video_title_3.setText(name)

    @pyqtSlot(QImage, str)
    def set_frame_4(self, image, name):
        self.video_display_4.setPixmap(QPixmap.fromImage(image))
        self.video_title_4.setText(name)

    @pyqtSlot(QImage, str)
    def set_frame_5(self, image, name):
        self.video_display_5.setPixmap(QPixmap.fromImage(image))
        self.video_title_5.setText(name)

    @pyqtSlot(QImage, str)
    def set_frame_6(self, image, name):
        self.video_display_6.setPixmap(QPixmap.fromImage(image))
        self.video_title_6.setText(name)

    @pyqtSlot(QImage)
    def set_record(self, image):
        self.record_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(bool)
    def switch_vis_stream_1(self, trigger):
        change_vis_stream(0)

    @pyqtSlot(bool)
    def switch_vis_stream_2(self, trigger):
        change_vis_stream(1)

    @pyqtSlot(bool)
    def switch_vis_stream_3(self, trigger):
        change_vis_stream(2)

    @pyqtSlot(bool)
    def switch_vis_stream_4(self, trigger):
        change_vis_stream(3)

    @pyqtSlot(bool)
    def switch_vis_stream_5(self, trigger):
        change_vis_stream(4)

    @pyqtSlot(bool)
    def switch_vis_stream_6(self, trigger):
        change_vis_stream(5)

    @pyqtSlot(str)
    def append_text(self, text):
        self.textBrowser.append(text)

    @pyqtSlot(str)
    def update_status_message(self, text):
        self.statusbar.showMessage(text)

    @pyqtSlot(bool)
    def process_exit(self, trigger):
        sys.exit()

    @pyqtSlot(bool)
    def site_1_switch(self, trigger):
        judge, config = self.read_interrupt(site_1)
        if judge:
            self.action_site_1.setText(station_name_dict[site_1])
        else:
            self.action_site_1.setText(station_name_switch_dict[site_1])
        self.write_interrupt(site_1, not judge, config)

    @pyqtSlot(bool)
    def site_2_switch(self, trigger):
        judge, config = self.read_interrupt(site_2)
        if judge:
            self.action_site_2.setText(station_name_dict[site_2])
        else:
            self.action_site_2.setText(station_name_switch_dict[site_2])
        self.write_interrupt(site_2, not judge, config)

    @pyqtSlot(bool)
    def site_3_switch(self, trigger):
        judge, config = self.read_interrupt(site_3)
        if judge:
            self.action_site_3.setText(station_name_dict[site_3])
        else:
            self.action_site_3.setText(station_name_switch_dict[site_3])
        self.write_interrupt(site_3, not judge, config)


    @pyqtSlot(bool)
    def site_4_switch(self, trigger):
        judge, config = self.read_interrupt(site_4)
        if judge:
            self.action_site_4.setText(station_name_dict[site_4])
        else:
            self.action_site_4.setText(station_name_switch_dict[site_4])
        self.write_interrupt(site_4, not judge, config)

    @pyqtSlot(bool)
    def site_5_switch(self, trigger):
        judge, config = self.read_interrupt(site_5)
        if judge:
            self.action_site_5.setText(station_name_dict[site_5])
        else:
            self.action_site_5.setText(station_name_switch_dict[site_5])
        self.write_interrupt(site_5, not judge, config)

    @pyqtSlot(bool)
    def site_6_switch(self, trigger):
        judge, config = self.read_interrupt(site_6)
        if judge:
            self.action_site_6.setText(station_name_dict[site_6])
        else:
            self.action_site_6.setText(station_name_switch_dict[site_6])
        self.write_interrupt(site_6, not judge, config)


class DetectionThread(QThread):
    video_1_change_pixmap = pyqtSignal(QImage, str)
    video_2_change_pixmap = pyqtSignal(QImage, str)
    video_3_change_pixmap = pyqtSignal(QImage, str)
    video_4_change_pixmap = pyqtSignal(QImage, str)
    video_5_change_pixmap = pyqtSignal(QImage, str)
    video_6_change_pixmap = pyqtSignal(QImage, str)

    record_change_pixmap = pyqtSignal(QImage)

    text_append = pyqtSignal(str)
    status_update = pyqtSignal(str)

    popup_message_box = pyqtSignal(str)

    def __init__(self, main_window):
        super().__init__(main_window)
        self.detection_flag = main_window.detection_flag
        self.main_window = main_window

    def run(self):
        logging.info('开始检测')
        detect_main(self)


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


def gui_main(detection_flag):
    sys.excepthook = except_hook  # print the traceback to stdout/stderr

    strftime = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    logging.basicConfig(filename='logs/' + strftime + '.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    logging.info('启动检测程序')

    app = QApplication(sys.argv)
    win = MainWindow(detection_flag)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    import multiprocessing

    flag = multiprocessing.Value('i', 0)
    gui_main(flag)
