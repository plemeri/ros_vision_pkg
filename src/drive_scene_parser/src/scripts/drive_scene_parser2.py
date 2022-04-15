#!/usr/bin/env python
from __future__ import print_function, absolute_import, division

import rospy
import rospkg
import os
import cv2
import numpy as np
import yaml
import image_geometry
import ros_numpy

from cv_bridge import CvBridge
from std_msgs.msg import String, Header
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2

import message_filters as mf

import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


CLS =  {0: {'class': 'person',     'color': [255,   0,   0]},
        1: {'class': 'bicycle',    'color': [  0, 255,   0]},
        2: {'class': 'car',        'color': [  0,   0, 255]},
        3: {'class': 'motorcycle', 'color': [255, 255,   0]},
        5: {'class': 'bus',        'color': [255,   0, 255]},
        7: {'class': 'truck',      'color': [  0, 255, 255]}
        }
class DriveSceneParser:
    def __init__(self, camera_param, size, publish_rate, result_topic, frame_id, object_topic='', lane_topic='', freespace_topic=''):
        self.size = size

        self.frame_id = frame_id
        self.info = CameraInfo()
        self.info.height = self.size[0]
        self.info.width  = self.size[1]
        self.info.distortion_model = 'plumb_bob'
        self.info.D = camera_param['D']
        self.info.K = camera_param['K']
        self.info.R = camera_param['R']
        self.info.P = camera_param['P']
        self.rate = rospy.Rate(publish_rate)
        
        self.object_msg = Detection2DArray
        self.lane_msg = ImageMsg()
        self.freespace_msg = PointCloud2()
        
        self.pub1 = rospy.Publisher(result_topic, ImageMsg, queue_size=10)
        # self.sub_object = rospy.Subscriber(object_topic, Detection2DArray, self.callback_object, tcp_nodelay=True)
        # self.sub_lane = rospy.Subscriber(lane_topic, ImageMsg, self.callback_lane, tcp_nodelay=True)
        # self.sub_freespace = rospy.Subscriber(freespace_topic, PointCloud2, self.callback_freespace, tcp_nodelay=True)
        
        self.sub_object = mf.Subscriber(object_topic, Detection2DArray) if object_topic != '' else None
        self.sub_lane = mf.Subscriber(lane_topic, ImageMsg) if lane_topic != '' else None
        self.sub_freespace = mf.Subscriber(freespace_topic, PointCloud2)  if freespace_topic != '' else None
        
        subs = [self.sub_object, self.sub_lane, self.sub_freespace]
        callbacks = [self.callback_object, self.callback_lane, self.callback_freespace]

        self.subs = []
        self.callbacks = []
        
        for sub, callback in zip(subs, callbacks):
            if sub is not None:
                self.subs.append(sub)
                self.callbacks.append(callback)    
        
        self.ts = mf.ApproximateTimeSynchronizer(self.subs, 10, 0.1)
        self.bridge = CvBridge()
        # self.geometry = image_geometry.PinholeCameraModel()
        # self.geometry.fromCameraInfo(self.info)
        
        # self.listener = tf.TransformListener()
        
        self.tf2_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf2_buffer)
        self.ts.registerCallback(self.callback)
                
    def to_numpy(self, msg, type='gray'):
        img = msg.data
        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape((msg.height, msg.width, -1))
        
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
        img = cv2.resize(img, self.size[::-1])
        return img
    
    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = int((x[:, 0] - x[:, 2] / 2) * self.size[1])  # top left x
        y[:, 1] = int((x[:, 1] - x[:, 3] / 2) * self.size[0])  # top left y
        y[:, 2] = int((x[:, 0] + x[:, 2] / 2) * self.size[1])  # bottom right x
        y[:, 3] = int((x[:, 1] + x[:, 3] / 2) * self.size[0])  # bottom right y
        return y

    def callback_object(self, object_msg):
        self.object_msg = object_msg
        
    def callback_lane(self, lane_msg):
        self.lane_msg = lane_msg
        
    def callback_freespace(self, freespace_msg):
        self.freespace_msg = freespace_msg
        
    def callback(self, *args):
        now = rospy.Time.now()
        
        for i, callback in enumerate(self.callbacks):
            callback(args[i])
            
        img = np.zeros((self.size[0], self.size[1], 3)).astype(np.uint8)
        
        if self.sub_freespace is not None:
            try:
                trans = self.tf2_buffer.lookup_transform(self.freespace_msg.header.frame_id, self.lane_msg.header.frame_id, rospy.Time(0))
            except:
                rospy.logwarn('retrieving tf...')
                return
            
            msg = do_transform_cloud(self.freespace_msg, trans)
            
            xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
            uvw = np.matmul(np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1), np.reshape(self.info.P, (3, 4)).T)
            uv = uvw[:, :2] / uvw[:, -1:]
            uv = uv.astype(int)
            uv = uv[(uv[:, 0] > 0) & (uv[:, 1] > 0) & (uv[:, 0] < self.size[0]) & (uv[:, 1] < self.size[1])]
            mask = np.zeros(self.size).astype(np.uint8)
            np.put(mask, uv, 1)
            
            # rospy.logwarn(xyz)
            rospy.logwarn(uv.shape)
            
            img[mask] = 255
            
            
        if self.sub_lane is not None:
            msg = self.to_numpy(self.lane_msg)
            if msg is None:
                msg = np.zeros(self.size).astype(np.uint8)
            img[msg > 128] = 255
        
        if self.sub_object is not None:
            dets = []
            for det in self.object_msg.detections:
                bbox = det.bbox
                xyxy = self.xywh2xyxy(np.array([[bbox.center.x, bbox.center.y, bbox.size_x, bbox.size_y]])).tolist()[0]
                id = det.results[0].id
                xyxy.append(id)
                dets.append(xyxy)
            
            for det in dets:
                det = [int(i) for i in det]
                img = cv2.rectangle(img, tuple(det[:2]), tuple(det[2:4]), CLS[det[-1]]['color'], -1)
                
        rospy.logwarn(img.shape)
        
        msg = self.bridge.cv2_to_imgmsg(img)
        msg.header.frame_id = self.frame_id
        msg.header.stamp = now
        self.pub1.publish(msg)

    def __len__(self):
        return 0

if __name__ == '__main__':
    rospy.init_node('drive_scene_parser')
    image_height      = rospy.get_param('~image_height',      480)
    image_width       = rospy.get_param('~image_width' ,      640)
    publish_rate      = rospy.get_param('~publish_rate' ,     30)
    object_topic      = rospy.get_param('~object_topic',      '')
    lane_topic        = rospy.get_param('~lane_topic',        '')
    freespace_topic   = rospy.get_param('~freespace_topic',   '')
    result_topic      = rospy.get_param('~result_topic',      '/image_parsed')
    frame_id          = rospy.get_param('~frame_id',          'camera1')
    
    camera_param = yaml.load(open(os.path.join(rospkg.RosPack().get_path('drive_scene_parser'), 'param', 'camera_param.yaml')))
    publisher = DriveSceneParser(camera_param, (image_height, image_width), publish_rate, result_topic, frame_id, object_topic, lane_topic, freespace_topic)
    # publisher.callback()
    
    rospy.spin()