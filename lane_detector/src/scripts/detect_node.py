#!/home/taehoon1018/anaconda3/envs/tensorrt/bin/python3
import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
import sys
from tqdm import tqdm

import rospy
import rospkg

from cv_bridge import CvBridge
from std_msgs.msg import String, Header, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image as Image
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from visualization_msgs.msg import ImageMarker

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from pathlib import Path

H, W = 590, 1640

class LaneDetector:
    def __init__(self, cfg, load_from, image_topic, result_topic, lane_topic):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, load_from)
        
        self.image_topic = image_topic
        self.result_topic = result_topic
        self.lane_topic = lane_topic
        
        self.pub1 = rospy.Publisher(self.lane_topic, Float32MultiArray, queue_size=10)
        self.pub2 = rospy.Publisher(self.result_topic, Image, queue_size=10)
        self.sub = rospy.Subscriber(self.image_topic, Image, self.callback, tcp_nodelay=True)
        
        self.bridge = CvBridge()
    
    def to_numpy(self, msg):
        img = msg.data
        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape((msg.height, msg.width, 3))
        return img
        
    def preprocess(self, x):
        h, w, _ = x.shape # 480 640
        
        if H / h < W / w:
            scale = H / h
            pad_value = int((W - scale * w) // 2) + 1
            pad = (0, 0, pad_value, pad_value)
        else:
            scale = W / w
            pad_value = int((H - scale * h) // 2)
            pad = (pad_value, pad_value, 0, 0)
        
        img = cv2.resize(x, (int(w * scale), int(h * scale)))
        img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, None, value=0)

        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data['ori_img'] = x
        data['scale'] = scale
        data['pad'] = pad
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def postprocess(self, data):
        lanes = [(lane.to_array(self.cfg) - [data['pad'][2], data['pad'][0]]) / data['scale'] for lane in data['lanes']]
        return lanes, imshow_lanes(data['ori_img'], lanes).astype(np.uint8)

    def callback(self, msg):
        data = self.to_numpy(msg)
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        
        lanes, out_img = self.postprocess(data)
        
        lane_msg = Float32MultiArray()
        lanes = np.array(lanes)
        
        if len(lanes.shape) == 3:
            
            lane_msg.data = lanes.flatten()
            lane_msg.layout.dim.append(MultiArrayDimension(label='lane_num', size=lanes.shape[0]))
            lane_msg.layout.dim.append(MultiArrayDimension(label='point_num', size=lanes.shape[1]))
            lane_msg.layout.dim.append(MultiArrayDimension(label='point', size=lanes.shape[2]))
        
        res_msg = self.bridge.cv2_to_imgmsg(out_img)
        res_msg.header.stamp = rospy.Time.now()
        res_msg.header.frame_id = msg.header.frame_id

        self.pub1.publish(lane_msg)
        self.pub2.publish(res_msg)

if __name__ == '__main__':
    rospy.init_node('lane_detector')
    config =       rospy.get_param('~config',             'configs/condlane/resnet101_culane.py')
    load_from =    rospy.get_param('~load_from',          'condlane_r101_culane.pth')
    image_topic =  rospy.get_param('~input_image_topic',  '/webcam1/image_raw')
    result_topic = rospy.get_param('~output_image_topic', '/webcam1/image_lane_detection')
    lane_topic =   rospy.get_param('~result_topic',       '/webcam1/detected_lanes')
    
    config =    os.path.join(rospkg.RosPack().get_path('lane_detector'), 'scripts', config)
    load_from = os.path.join(rospkg.RosPack().get_path('lane_detector'), 'scripts', load_from)

    cfg = Config.fromfile(config)
    detector = LaneDetector(cfg, load_from, image_topic, result_topic, lane_topic)
    rospy.spin()

