import argparse

import cv2
import time
import torch
import logging
import numpy as np

from threading import Thread
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QSize
from PIL import Image, ImageDraw, ImageFont

from handler.database import MySql
from video_stream.video_stream import VideoLoader
from configs.config import *
from model.models import Darknet
from video_stream.visualize import Visualize
from utils.utils import non_max_suppression, load_classes, calc_fps
from model.transform import transform, stack_tensors, preds_postprocess
from handler.intrusion_handling import IntrusionHandling
from handler.alarm import Alarm, send_signal
from handler.opc_client import OpcClient
from handler.send_email import Email
from yolox.interface.yolo_main import set_model








def inference(vis_images: dict):
    preds_output = []
    preds_info = []
    for name in vis_images.keys():
        outputs, img_info = model.inference(vis_images[name])
        preds_output.append(outputs)
        preds_info.append(img_info)
    return preds_output, preds_info


def array_to_q_image(img, size):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    if isinstance(size, QSize):
        q_image = q_image.scaled(size)
    else:
        q_image = q_image.scaled(size[0], size[1])
    return q_image


def img_add_title(img, text, left, top, color, size):

    img = cv2.copyMakeBorder(img, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(41, 24, 20))

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    font_style = ImageFont.truetype("font/simsun.ttc", size, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, color, font=font_style)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def change_vis_stream(index):
    global vis_name
    global prev_vis_name

    prev_vis_name = vis_name
    vis_name = list(video_stream_paths_dict.keys())[index]




def detect_main(q_thread):
    q_thread.status_update.emit('模型加载')
    device = torch.device(device_name)
    # 获取检测模型
    model = get_model(config_path, img_size, weights_path, device)
    model = set_model(0.25, 0.45, 640)



    # # 姿态识别部分
    #
    # par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    # # par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
    # #                  help='Source of camera or video file path.')
    # # par.add_argument('-Ce', '--camera1', default=source,  # required=True,  # default=2,
    # #                  help='Source of camera or video file path.')
    # par.add_argument('--detection_input_size', type=int, default=384,
    #                  help='Size of input in detection model in square must be divisible by 32 (int).')
    # par.add_argument('--pose_input_size', type=str, default='224x160',
    #                  help='Size of input in pose model must be divisible by 32 (h, w)')
    # par.add_argument('--pose_backbone', type=str, default='resnet50',
    #                  help='Backbone model for SPPE FastPose model.')
    # par.add_argument('--show_detected', default=False, action='store_true',
    #                  help='Show all bounding box from detection.')
    # par.add_argument('--show_skeleton', default=True, action='store_true',
    #                  help='Show skeleton pose.')
    # par.add_argument('--save_out', type=str, default='',
    #                  help='Save display to video file.')
    # par.add_argument('--device', type=str, default='cuda',
    #                  help='Device to run model on cpu or cuda.')
    # args = par.parse_args()
    #
    # device = args.device
    #
    # # DETECTION MODEL.
    # inp_dets = args.detection_input_size
    # detect_model = TinyYOLOv3_onecls(inp_dets, device=device)
    #
    # # POSE MODEL.
    # inp_pose = args.pose_input_size.split('x')
    # inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    # pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)
    #
    # # Tracker.
    # max_age = 30
    # tracker = Tracker(max_age=max_age, n_init=3)
    #
    # # Actions Estimate.
    # action_model = TSSTG()
    #
    # # resize_fn = ResizePadding(inp_dets, inp_dets)
    #
    # outvid = False
    # if args.save_out != '':
    #     outvid = True
    #     # codec = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    #     codec = cv2.VideoWriter_fourcc(*'MJPG')
    #     writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))
    #
    # fps_time = 0
    # f = 0
    #







    q_thread.status_update.emit('连接OPC服务')
    if open_opc:
        opc_client = OpcClient(opc_url, nodes_dict)
        strftime = time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime())
        q_thread.text_append.emit(strftime + ' OPC 服务器已连接')
        logging.info('OPC Client created')

    else:
        opc_client = None
        strftime = time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime())
        q_thread.text_append.emit(strftime + ' OPC 服务器未连接')
        logging.warning('OPC Client does not create')
    if open_email_warning:
        warning_email = Email()
    else:
        warning_email = None

    q_thread.status_update.emit('初始化异常处理程序')
    visualize = Visualize(masks_paths_dict)
    handling = IntrusionHandling(masks_paths_dict, opc_client)

    q_thread.status_update.emit('连接报警灯')

    alarm = Alarm(q_thread)
    thread = Thread(target=alarm.check)
    thread.start()

    q_thread.status_update.emit('读取视频流')
    video_loader = VideoLoader(video_stream_paths_dict)
    logging.info('Video streams create: ' + ', '.join(n for n in video_stream_paths_dict.keys()))

    # # 姿态读视频流
    # cam_source = args.camera
    # if type(cam_source) is str and os.path.isfile(cam_source):
    #     # Use loader thread with Q for video file.
    #     cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    # else:
    #     # Use normal thread loader for webcam.
    #     cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
    #                     preprocess=preproc).start()

    # outvid = False
    # if args.save_out != '':
    #     outvid = True
    #     # codec = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    #     codec = cv2.VideoWriter_fourcc(*'MJPG')
    #     writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))
    #
    # fps_time = 0
    # f = 0


    q_thread.status_update.emit('准备就绪')

    since = patrol_opc_nodes_clock_start = update_detection_flag_clock_start = time.time()

    accum_time, curr_fps = 0, 0
    show_fps = 'FPS: ??'

    prevs_frames_dict = None
    logging.info('Enter detection main loop process')
    MySql.add_record2sql(mysql_interrupt_table)
    while True:
        curr_time = time.time()

        if curr_time - update_detection_flag_clock_start > update_detection_flag_interval:
            MySql.add_record2sql(mysql_interrupt_table)
            update_detection_flag_clock_start = curr_time
            q_thread.detection_flag.value = 1

        vis_images_dict = video_loader.getitem()

        img_title_dict = station_name_dict.copy()

        active_streams = []
        for name in vis_images_dict.keys():
            if vis_images_dict[name] is None:
                if prevs_frames_dict is not None:
                    vis_images_dict[name] = prevs_frames_dict[name]
            else:
                active_streams.append(station_name_dict[name])

        if len(vis_images_dict) == 6:
            prevs_frames_dict = vis_images_dict
        elif len(vis_images_dict) == 0:
            print("未读到任何视频帧")
            time.sleep(0.5)
            continue

        # model inference and postprocess
        preds_output, preds_info = inference(vis_images_dict)
        #
        if prevs_frames_dict is None:
            not_none_streams = [x for x in vis_images_dict.keys() if vis_images_dict[x] is not None]
        else:
            not_none_streams = list(vis_images_dict.keys())
        # 返回值只有非None视频流的预测结果
        preds_dict, cls_dict = preds_postprocess(preds_output, preds_info, not_none_streams)

        judgements_dict = handling.judge_intrusion(preds_dict)

        since, accum_time, curr_fps, show_fps = calc_fps(since, accum_time, curr_fps, show_fps)

        vis_images_prev_dict = vis_images_dict.copy()
        vis_images_dict = visualize.draw(vis_images_dict, preds_dict, judgements_dict, show_fps)

        handling.handle_judgement(judgements_dict, vis_images_dict, vis_images_prev_dict)

        send_signal(judgements_dict, preds_dict)

        if vis_name in vis_images_dict:
            img = vis_images_dict[vis_name]
            # img = img_add_title(img, img_title_dict[vis_name], 5, 5, (255, 255, 255), 40)
            q_size = q_thread.main_window.video_display_1.size()
            q_image = array_to_q_image(img, q_size)
            q_thread.video_1_change_pixmap.emit(q_image, " " + img_title_dict[vis_name])

        for name in judgements_dict.keys():
            if judgements_dict[name]:
                time_str = time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime())
                q_thread.text_append.emit(time_str + station_name_dict[name] + ' 启动联锁保护')
                img = vis_images_dict[name]
                q_size = q_thread.main_window.record_label.size()
                q_image = array_to_q_image(img, q_size)
                q_thread.record_change_pixmap.emit(q_image)

        if prev_vis_name in vis_images_dict:
            prev_img = vis_images_dict[prev_vis_name]
            prev_title = img_title_dict[prev_vis_name]
            vis_images_dict[vis_name] = prev_img
            img_title_dict[vis_name] = prev_title
            vis_images_dict.pop(prev_vis_name)
            img_title_dict.pop(prev_vis_name)

        # f += 1
        # frame = cam.getitem()
        # image = frame.copy()
        # # print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        # # print(frame.shape)
        # # Detect humans bbox in the frame with detector model.
        # detected = detect_model.detect(frame, need_resize=False, expand_bb=10)
        #
        # # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        # tracker.predict()
        # # Merge two source of predicted bbox together.
        # for track in tracker.tracks:
        #     det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
        #     detected = torch.cat([detected, det], dim=0) if detected is not None else det
        #
        # detections = []  # List of Detections object for tracking.
        # if detected is not None:
        #     #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
        #     # Predict skeleton pose of each bboxs.
        #     poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
        #
        #     # Create Detections object.
        #     detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
        #                             np.concatenate((ps['keypoints'].numpy(),
        #                                             ps['kp_score'].numpy()), axis=1),
        #                             ps['kp_score'].mean().numpy()) for ps in poses]
        #
        #     # VISUALIZE.
        #     if args.show_detected:
        #         for bb in detected[:, 0:5]:
        #             frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)
        #
        # # Update tracks by matching each track information of current and previous frame or
        # # create a new track if no matched.
        # tracker.update(detections)
        #
        # # Predict Actions of each track.
        # for i, track in enumerate(tracker.tracks):
        #     if not track.is_confirmed():
        #         continue
        #
        #     track_id = track.track_id
        #     bbox = track.to_tlbr().astype(int)
        #     center = track.get_center().astype(int)
        #
        #     action = 'pending..'
        #     clr = (0, 255, 0)
        #     # Use 30 frames time-steps to prediction.
        #     if len(track.keypoints_list) == 30:
        #         pts = np.array(track.keypoints_list, dtype=np.float32)
        #         out = action_model.predict(pts, frame.shape[:2])
        #         action_name = action_model.class_names[out[0].argmax()]
        #         # action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
        #         action = '{}'.format(action_name)
        #         if action_name == 'Fall Down':
        #             clr = (255, 0, 0)
        #         elif action_name == 'Lying Down':
        #             clr = (255, 200, 0)
        #
        #     # VISUALIZE.
        #     if track.time_since_update == 0:
        #         if args.show_skeleton:
        #             frame = draw_single(frame, track.keypoints_list[-1])
        #             # print(track.keypoints_list[-1])
        #         # print(frame.get(cv2.CAP_PROP_FRAME_WIDTH))
        #         frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
        #         frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
        #                             0.4, (255, 0, 0), 2)
        #         frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
        #                             0.4, clr, 1)
        #
        # # Show Frame.
        # frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        # frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
        #                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # frame = frame[:, :, ::-1]
        # fps_time = time.time()
        #
        # if outvid:
        #     writer.write(frame)

        for title, (i, img) in zip(img_title_dict.values(), enumerate(vis_images_dict.values())):
            # img = img_add_title(img, title, 5, 5, (255, 255, 255), 40)

            q_size_v = q_thread.main_window.video_display_2.size()
            q_image_v = array_to_q_image(img, q_size_v)

            q_size_h = q_thread.main_window.video_display_5.size()
            q_image_h = array_to_q_image(img, q_size_h)
            if i == 0:
                q_thread.video_2_change_pixmap.emit(q_image_h, title)
            elif i == 1:
                q_thread.video_3_change_pixmap.emit(q_image_h, title)
            elif i == 2:
                q_thread.video_4_change_pixmap.emit(q_image_h, title)
            elif i == 3:
                q_thread.video_5_change_pixmap.emit(q_image_h, title)
            elif i == 4:
                q_thread.video_6_change_pixmap.emit(q_image_h, title)
            else:
                raise RuntimeError("No so many QLabel!")
