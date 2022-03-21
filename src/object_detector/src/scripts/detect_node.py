#!/home/taehoon1018/anaconda3/envs/yolov5/bin/python3

import argparse
import os
import sys
from pathlib import Path
from unittest import result

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import rospy
import rospkg

from cv_bridge import CvBridge
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image as Image
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from visualization_msgs.msg import ImageMarker


class ObjectDetector:
    def __init__(self, weights, data, image_topic, result_topic, detection_topic, img_shape, conf_thres, iou_thres, max_det, device, classes, half, dnn):
        self.weights = weights
        self.data = data
        self.image_topic = image_topic
        self.result_topic = result_topic
        self.detection_topic = detection_topic
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = select_device(device)
        self.classes = classes
        self.half  = half 
        self.dnn = dnn
        
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.jit = self.model.jit
        self.onnx = self.model.onnx
        self.engine = self.model.engine
        self.img_shape = check_img_size(img_shape, s=self.stride)
        self.half &= (self.pt or self.jit or self.onnx or self.engine) and self.device.type != 'cpu'
        
        if self.pt or self.jit:
            self.model.model.half() if half else self.model.model.float()
        self.model.warmup((1, 3, *img_shape), half=half)  # warmup
        
        self.pub1 = rospy.Publisher(self.detection_topic, Detection2DArray, queue_size=10)
        self.pub2 = rospy.Publisher(self.result_topic, Image, queue_size=10)
        self.sub = rospy.Subscriber(self.image_topic, Image, self.callback, tcp_nodelay=True, queue_size=1, buff_size=2**24)
        
        self.bridge = CvBridge()
        
    def to_numpy(self, msg):
        img = msg.data
        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape((msg.height, msg.width, -1))
        
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
            
        img = np.ascontiguousarray(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @torch.no_grad()
    def callback(self, msg):
        im0 = self.to_numpy(msg)
        im = cv2.resize(im0, self.img_shape[::-1])
        im = torch.from_numpy(im).to(self.device).permute(2, 0, 1)
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, False, max_det=self.max_det)

        det_msg = Detection2DArray()
        det_msg.header.stamp = rospy.Time.now()
        det_msg.header.frame_id = msg.header.frame_id
        
        res_msg = Image()

        if len(pred):
            pred = pred[0]
            pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(pred):
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                det = Detection2D()
                det.header = det_msg.header

                hypo = ObjectHypothesisWithPose()
                hypo.id = int(cls)
                hypo.score = conf
                
                bbox = BoundingBox2D()
                bbox.center.x = xywh[0]
                bbox.center.y = xywh[1]
                bbox.size_x =   xywh[2]
                bbox.size_y =   xywh[3]
                
                det.results.append(hypo)
                det.bbox = bbox
                
                det_msg.detections.append(det)
                
                annotator = Annotator(im0, line_width=2, example=str(self.names))
                annotator.box_label(xyxy, f'{self.names[int(cls)]} {conf:.2f}', color=colors(int(cls), True))
                im0 = annotator.result()
                res_msg = self.bridge.cv2_to_imgmsg(im0)
                res_msg.header = det_msg.header
        
        self.pub1.publish(det_msg)
        self.pub2.publish(res_msg if len(pred) else msg)
                


if __name__ == "__main__":
    rospy.init_node('object_detector')
    weights =          rospy.get_param('~weights',            'yolov5l.pt')
    data =             rospy.get_param('~data',               'data/coco128.yaml')
    image_topic =      rospy.get_param('~input_image_topic',  '/camera1/image_raw')
    result_topic =     rospy.get_param('~output_image_topic', '/camera1/image_object_detection')
    detection_topic =  rospy.get_param('~result_topic',       '/camera1/detected_objects')
    image_height =     rospy.get_param('~image_height',       480)
    image_width =      rospy.get_param('~image_width',        640)
    conf =             rospy.get_param('~conf_thres',         0.25)
    iou =              rospy.get_param('~iou_thres',          0.45)
    max =              rospy.get_param('~max_det',            1000)
    device =           rospy.get_param('~device',             0)
    classes =          rospy.get_param('~classes',            '[0]')
    half =             rospy.get_param('~half',               False)
    dnn =              rospy.get_param('~dnn',                False)
    
    weights = os.path.join(rospkg.RosPack().get_path('object_detector'), 'scripts', weights)
    data =    os.path.join(rospkg.RosPack().get_path('object_detector'), 'scripts', data)
    
    classes = [int(i) for i in classes[1:-1].split(',')]
    
    check_requirements(exclude=('tensorboard', 'thop'))
    detector = ObjectDetector(weights, data, image_topic, result_topic, detection_topic, (image_height, image_width), conf, iou, max, device, classes, half, dnn)
    rospy.spin()
