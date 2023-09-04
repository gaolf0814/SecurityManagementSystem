switch_dict = {
    'line_3': True,
    'line_4_1': True,
    'line_4_2': True,
    'line_2': True,
    'm5000': True,
    'lvd': True
}

site_interrupt_dict = {
    'line_3': True,
    'line_4_1': True,
    'line_4_2': True,
    'line_2': True,
    'm5000': True,
    'lvd': True
}

pre_dict = {
    'line_3': False,
    'line_4_1': False,
    'line_4_2': False,
    'line_2': False,
    'm5000': False,
    'lvd': False
}

name_array = [
    'line_3',
    'line_4_1',
    'line_4_2',
    'line_2',
    'm5000',
    'lvd'
]

signal_dict = {
    'line_3': 0,
    'line_4_1': 0,
    'line_4_2': 0,
    'line_2': 0,
    'm5000': 0,
    'lvd': 0
}

off_signal_dict = {
    'line_3': 0,
    'line_4_1': 0,
    'line_4_2': 0,
    'line_2': 0,
    'm5000': 0,
    'lvd': 0
}

station_name_dict = {
    'line_3': '专机线-3',
    'line_4_1': '专机线-4-1',
    'line_4_2': '专机线-4-2',
    'line_2': '专机线-2',
    'm5000': 'M5000冲',
    'lvd': 'LVD折'
}

station_name_switch_dict = {
    'line_3': '专机线-3√',
    'line_4_1': '专机线-4-1√',
    'line_4_2': '专机线-4-2√',
    'line_2': '专机线-2√',
    'm5000': 'M5000冲√',
    'lvd': 'LVD折√'
}

site_1 = 'line_3'
site_2 = 'line_4_1'
site_3 = 'line_4_2'
site_4 = 'line_2'
site_5 = 'm5000'
site_6 = 'lvd'

dynamic_config_path = 'configs/dynamic_config.ini'
interrupt_switch_section = 'site_interrupt_dict'

video_stream_paths_dict = {
    # 'line_3': 'rtsp://admin:hdu417417@192.168.2.45/Streaming/Channels/101',
    # 'line_4_1': 'rtsp://admin:hdu417417@192.168.2.47/Streaming/Channels/101',
    # 'line_4_2': 'rtsp://admin:hdu417417@192.168.2.46/Streaming/Channels/101',
    # 'line_2': 'rtsp://admin:hdu417417@192.168.2.48/Streaming/Channels/101',
    # 'm5000': "rtsp://admin:hdu417417@192.168.2.172/Streaming/Channels/101",
    # 'lvd': "rtsp://admin:hdu417417@192.168.2.177/Streaming/Channels/101"

    'line_3': 'rtsp://admin:hdu417417@192.168.1.3/Streaming/Channels/101',
    # 'line_4_1': 'rtsp://admin:hdu417417@10.1.124.72/Streaming/Channels/101',
    # 'line_4_2': 'rtsp://admin:hdu417417@10.1.124.71/Streaming/Channels/101',
    # 'line_2': 'rtsp://admin:hdu417417@10.1.124.72/Streaming/Channels/101',
    # 'm5000': 'rtsp://admin:hdu417417@10.1.124.72/Streaming/Channels/101',
    # 'lvd': 'rtsp://admin:hdu417417@10.1.124.72/Streaming/Channels/101',

}

masks_paths_dict = {
    'line_3': 'images/masks/line_3.jpg',
    'line_4_1': 'images/masks/line_4_1.jpg',
    'line_4_2': 'images/masks/line_4_2.jpg',
    'line_2': 'images/masks/line_2.jpg',
    'm5000': 'images/masks/m5000.jpg',
    'lvd': 'images/masks/lvd.jpg'
}

max_object_bbox_area_dict = {
    'line_3': 15000,
    'line_4_1': 15000,
    'line_4_2': 15000,
    'line_2': 15000,
    'm5000': 15000,
    'lvd': 15000
}


# OPC 服务器 URL
opc_url = 'opc.tcp://127.0.0.1:49320'

# 是否连接OPC服务器，执行紧急停机
open_opc = True
# 开启邮箱OPC报警
open_email_warning = True
# 开启统计闯入次数和邮箱发送报告功能
open_email_report = True
# 开启数据库存储异常记录
open_mysql_save_record = True

mysql_interrupt_table = 'interrupt_cpp'

nodes_dict = {
    'line_3': "ns=2;s=专机3机器人安全监测.机器人安全监测触发.机器人安全检测触发",
    'line_4_1': "ns=2;s=专机4机器人安全监测.机器人安全监测触发.2号喷粉上件机器人安全监测触发",
    'line_4_2': "ns=2;s=专机4机器人安全监测.机器人安全监测触发.1号下线机器人安全监测触发",
    'line_2': "ns=2;s=专机2机器人安全监测.机器人安全监测触发.机器人安全监测触发",
    'm5000': "ns=2;s=直梁对重架冲折.冲折自动线.M5000冲机器人安全监控",
    'lvd': "ns=2;s=直梁对重架冲折.冲折自动线.LVD折弯机器人安全监控"
}

min_object_bbox_area_dict = {
    'line_3': 500,
    'line_4_1': 850,
    'line_4_2': 500,
    'line_2': 500,
    'm5000': 500,
    'lvd': 500
}

excluded_objects_dict = {
    'line_3': [[592, 31, 618, 89]],
    'line_4_1': [[274, 0, 303, 55], [275, 448, 325, 480], [235, 309, 266, 392], [242, 311, 263, 385]],
    'line_4_2': [[573, 176, 587, 230], [402, 176, 421, 251],  [472, 209, 516, 262], [578, 255, 599, 284],
                 [547, 212, 567, 260], ],
    'line_2': [[272, 318, 281, 377], [355, 344, 380, 414], ],
    'm5000': [],
    'lvd': []
}
# 330, 179 356, 221

frame_shape = (480, 640)

vis_name = 'line_3'
prev_vis_name = vis_name

device_name = 'cuda:0'
img_size = 640  # size of each image dimension
config_path = 'yolox/configs/yolox_m.py'  # path to model configs file
weight_path = 'yolox/configs/best_ckpt.pth'
conf_thres = 0.65  # object confidence threshold
nms_thres = 0.75  # iou threshold for non-maximum suppression

mysql_host = 'localhost'
mysql_user = 'root'
mysql_password = '123456'
mysql_db = 'xio'

email_opc_warning_interval = 3600


wechat_send_interval = 30

inter_threshold = 0.15

open_wechat_bot = False

wechat_group = "机器人安全监测"

report_statistics_interval = 3600

server_name_dict = {
    'line_3': '192.168.2.81',
    'line_4_1': '192.168.2.69',
    'line_4_2': '192.168.2.54',
    'line_2': '192.168.2.67',
    'm5000': '192.168.2.136',
    'lvd': '192.168.2.71'
}

server_port = 8080

alarm_link_success = 0
alarm_link_failed = 1

check_detection_process_interval = 65

reboot_time_steps = 24

update_detection_flag_interval = 20

alarm_instruction = True


