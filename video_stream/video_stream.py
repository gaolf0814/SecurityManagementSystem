import time
from queue import LifoQueue, Empty, Queue
from threading import Thread

import cv2
from configs.config import switch_dict







#
#
# import os
# from threading import Thread
# import threading
#
# import cv2
# import time
# import torch
# import argparse
# import numpy as np
#
# from Detection.Utils import ResizePadding
# from CameraLoader import CamLoader, CamLoader_Q
# from DetectorLoader import TinyYOLOv3_onecls
#
# from PoseEstimateLoader import SPPE_FastPose
# from fn import draw_single
#
# from Track.Tracker import Detection, Tracker
# from ActionsEstLoader import TSSTG
#
# inp_dets=384
# resize_fn = ResizePadding(inp_dets, inp_dets)
# def preproc(image):
#     """preprocess function for CameraLoader.
#     """
#     image = resize_fn(image)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image
#
#
# def kpt2bbox(kpt, ex=20):
#     """Get bbox that hold on all of the keypoints (x,y)
#     kpt: array of shape `(N, 2)`,
#     ex: (int) expand bounding box,
#     """
#     return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
#                      kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))
#
#
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
# resize_fn = ResizePadding(inp_dets, inp_dets)
#
# # cam_source = args.camera
# # if type(cam_source) is str and os.path.isfile(cam_source):
# #     # Use loader thread with Q for video file.
# #     cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
# # else:
# #     # Use normal thread loader for webcam.
# #     cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
# #                     preprocess=preproc).start()
#
# # frame_size = cam.frame_size
# # scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]
#
# outvid = False
# if args.save_out != '':
#     outvid = True
#     # codec = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
#     codec = cv2.VideoWriter_fourcc(*'MJPG')
#     writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))
#
# f=0
#
# # coding=utf-8
# # cv2解决绘制中文乱码
#
# import cv2
# import numpy
# from PIL import Image, ImageDraw, ImageFont
#
#
# def cv2ImgAddText(img, text, left, top, textColor, textSize=20):
#     if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     # 创建一个可以在给定图像上绘图的对象
#     draw = ImageDraw.Draw(img)
#     # 字体的格式
#     fontStyle = ImageFont.truetype(
#         "font/simsun.ttc", textSize, encoding="utf-8")
#     # 绘制文本
#     draw.text((left, top), text, textColor, font=fontStyle)
#     # 转换回OpenCV格式
#     return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
#
#
#















class VideoStream:
    # fps_time = 0
    # # f = 0



    def __init__(self, video_path, stream_name):
        self.video_path = video_path
        self.stream_name = stream_name
        self.capture = self.get_video_capture()









    def reconnect(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = self.get_video_capture()

    def robust_read(self):
        if self.capture is None:
            return None

        ret, frame = self.capture.read()
        reconn_flag = False
        since = time.time()
        while not ret or frame is None:
            self.reconnect()
            if self.capture is None:
                return None
            ret, frame = self.capture.read()
            reconn_flag = True
        if reconn_flag:
            time_consume = time.time() - since
            print('视频流"{}"不稳定,重新连接 {:.2f}'.format(self.stream_name, time_consume))
        # assert frame.shape == (480, 640, 3)
        if frame.shape != (480, 640, 3):
            frame = cv2.resize(frame, (640, 480))
        return frame

    def release(self):
        if self.capture is not None:
            self.capture.release()

    def is_opened(self):
        if self.capture is not None:
            return self.capture.isOpened()

    def get_video_capture(self, timeout=5):
        if not switch_dict[self.stream_name]:
            return None
        res_queue = Queue()
        th = VideoCaptureDaemon(self.video_path, res_queue)
        th.start()
        try:
            return res_queue.get(block=True, timeout=timeout)
        except Empty:
            print('无法连接 {} Timeout occurred after {:.2f}s'.format(self.video_path, timeout))
            return None


class VideoLoader:
    def __init__(self, video_streams_path_dict, queue_maxsize=50):
        self.queue_maxsize = queue_maxsize
        self.video_streams_dict = self.__video_captures(video_streams_path_dict)
        self.queues_dict = self.queues()
        self.start()

    @staticmethod
    def __video_captures(video_streams_path_dict):
        video_streams_dict = {}

        for name in video_streams_path_dict.keys():
            path = video_streams_path_dict[name]
            stream = VideoStream(path, name)
            if stream.capture is not None:
                print(name, "视频流已创建")
            else:
                print(name, "视频流创建失败！")
            video_streams_dict[name] = stream
        return video_streams_dict

    def start(self):
        """
        如果某个视频流连接不上，就没必要创建线程，不然会进入死循环会严重影响性能
        """
        th_capture = {}
        for index, name in enumerate(self.video_streams_dict.keys()):
            if self.video_streams_dict[name].capture is not None:
                th_capture[index] = Thread(target=self.update, args=(name,))
                th_capture[index].daemon = True
                th_capture[index].start()

    def queues(self):
        queues_dict = {}
        for name in self.video_streams_dict.keys():
            q = LifoQueue(maxsize=self.queue_maxsize)
            queues_dict[name] = q
        return queues_dict

    def update(self, name):
        while True:
            if not self.queues_dict[name].full():
                capture = self.video_streams_dict[name]
                frame = capture.robust_read()
                if frame is not None:
                    self.queues_dict[name].put(frame)
            else:
                with self.queues_dict[name].mutex:
                    self.queues_dict[name].queue.clear()

    def getitem(self):
        # return next frame in the queue
        frames_dict = {}
        for name in self.queues_dict.keys():
            if self.video_streams_dict[name].capture is not None:
                try:
                    # 设置timeout，否则一直返回None，主线程陷入死循环导致程序崩溃
                    frames_dict[name] = self.queues_dict[name].get(timeout=2)
                    # print(frames_dict[name].shape)
                    # f += 1





















                    #
                    #
                    # frame=frames_dict[name]
                    # frame = cv2.resize(frame, dsize=(384, 384), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
                    #
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
                    #     # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
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
                    #     clr_rectangle = (0, 255, 0)
                    #     # Use 30 frames time-steps to prediction.
                    #     if len(track.keypoints_list) == 30:
                    #         pts = np.array(track.keypoints_list, dtype=np.float32)
                    #         out = action_model.predict(pts, frame.shape[:2])
                    #         action_name = action_model.class_names[out[0].argmax()]
                    #         # action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                    #         action = '{}'.format(action_name)
                    #         # if action_name == 'Fall Down':
                    #         #     clr = (255, 0, 0)
                    #         # elif action_name == 'Lying Down':
                    #         #     clr = (255, 200, 0)
                    #         if action_name == '站立' or action_name == '行走':
                    #             # clr = (0, 255, 0)
                    #             clr_rectangle = (0, 255, 0)
                    #         else:
                    #             clr = (255, 0, 0)
                    #             clr_rectangle = (0, 0, 255)
                    #
                    #     # VISUALIZE.
                    #     if track.time_since_update == 0:
                    #         if args.show_skeleton:
                    #             frame = draw_single(frame, track.keypoints_list[-1])
                    #             # print(track.keypoints_list[-1])
                    #         # print(frame.get(cv2.CAP_PROP_FRAME_WIDTH))
                    #         frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), clr_rectangle, 1)
                    #         frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                    #                             0.4, (255, 0, 0), 2)
                    #         # frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                    #         #                     0.4, clr, 1)
                    #         # 显示字体，站立或者坐下
                    #         frame = cv2ImgAddText(frame, action, bbox[0] + 5, bbox[1] + 5, clr, 15)
                    #
                    # # Show Frame.
                    # # frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
                    # # frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                    # #                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # # frame = frame[:, :, ::-1]
                    # fps_time = time.time()
                    #
                    # if outvid:
                    #     writer.write(frame)
                    # # frame = frames_dict[name]
                    # frame = cv2.resize(frame, dsize=(480,640), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
                    # frames_dict[name]=frame
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #














                except Empty:
                    frames_dict[name] = None
            else:
                frames_dict[name] = None
        # print("栈长" + str(self.queues_dict[name].qsize()))
        return frames_dict


class VideoCaptureDaemon(Thread):
    """
    由于 cv2.VideoCapture在找不到资源时会一直阻塞，
    而且没有设置timeout的方式，
    所以只能单独创建一个线程来尝试连接
    """
    def __init__(self, video, result_queue):
        super().__init__()
        self.setDaemon(True)
        self.video = video
        self.result_queue = result_queue

    def run(self):
        self.result_queue.put(cv2.VideoCapture(self.video))

