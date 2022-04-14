#!/usr/bin/python3
import torch
import os
import argparse
import tqdm
import sys
import cv2

import numpy as np
# from torch2trt import torch2trt
from PIL import Image

print(sys.executable)

import rospy
import rospkg

from cv_bridge import CvBridge
from std_msgs.msg import String, Header, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from visualization_msgs.msg import ImageMarker

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.misc import *
from data.dataloader import *
from data.custom_transforms import *

def get_transform(tfs):
    comp = []
    for key, value in zip(tfs.keys(), tfs.values()):
        if value is not None:
            tf = eval(key)(**value)
        else:
            tf = eval(key)()
        comp.append(tf)
    return transforms.Compose(comp)

class LaneSOD:
    def __init__(self, config, image_topic, result_topic, jit=False):
        self.jit = jit
        self.opt = load_config(config)
        
        self.model = eval(self.opt.Model.name)(depth=self.opt.Model.depth, pretrained=False, base_size=self.opt.Train.Dataset.transforms.resize.size)
        self.model.load_state_dict(torch.load(os.path.join(
            rospkg.RosPack().get_path('lane_detector'), 'scripts',
            self.opt.Test.Checkpoint.checkpoint_dir, 'latest.pth'), map_location=torch.device('cpu')), strict=True)

        self.model.cuda()
        self.model.eval()
        
        if self.jit is True:
            if os.path.isfile(os.path.join(rospkg.RosPack().get_path('lane_detector'), 'scripts', self.opt.Test.Checkpoint.checkpoint_dir, 'jit.pt')) is False:
                self.model = torch.jit.trace(self.model, torch.rand(1, 3, *self.opt.Test.Dataset.transforms.resize.size).cuda())
                torch.jit.save(self.model, os.path.join(rospkg.RosPack().get_path('lane_detector'), 'scripts', self.opt.Test.Checkpoint.checkpoint_dir, 'jit.pt'))
            else:
                del self.model
                self.model = torch.jit.load(os.path.join(rospkg.RosPack().get_path('lane_detector'), 'scripts', self.opt.Test.Checkpoint.checkpoint_dir, 'jit.pt'))
                self.model.cuda()

        self.transform = get_transform(self.opt.Test.Dataset.transforms)
        
        self.image_topic = image_topic
        self.result_topic = result_topic
        
        self.pub = rospy.Publisher(self.result_topic, ImageMsg, queue_size=1)
        self.sub = rospy.Subscriber(self.image_topic, ImageMsg, self.callback, tcp_nodelay=True, queue_size=1, buff_size=2**24)
        
        self.bridge = CvBridge()
        
    def msg_to_numpy(self, msg):
        img = msg.data
        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape((msg.height, msg.width, -1))
        
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
            
        return img

    def callback(self, msg):
        img = Image.fromarray(self.msg_to_numpy(msg))
        sample = {'image': img, 'shape': img.size[::-1], 'original': img}
        sample = self.transform(sample)
        sample = to_cuda(sample)
        sample['image'] = sample['image'].unsqueeze(0)

        with torch.no_grad():
            out = self.model(sample['image'])
        pred = to_numpy(out, sample['shape'])
        # img = np.array(sample['original'])
        # bg = np.stack([np.ones_like(pred)] * 3, axis=-1) * [0, 255, 0]
        # img = bg * pred[..., np.newaxis] + img * (1 - pred[..., np.newaxis])
        # img = img.astype(np.uint8)
        
        img_msg = self.bridge.cv2_to_imgmsg(((pred > .5) * 255).astype(np.uint8))
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = msg.header.frame_id
        
        self.pub.publish(img_msg)
        

if __name__ == '__main__':
    rospy.init_node('lane_detector')
    config =       rospy.get_param('~config',             'configs/Legacy.yaml')
    jit =          rospy.get_param('~jit',                'True')
    image_topic =  rospy.get_param('~input_image_topic',  '/camera1/image_raw')
    result_topic = rospy.get_param('~result_topic',       '/camera1/detected_lanes')
    
    config =    os.path.join(rospkg.RosPack().get_path('lane_detector'), 'scripts', config)
    print(config)

    detector = LaneSOD(config, image_topic, result_topic, jit)
    rospy.spin()

