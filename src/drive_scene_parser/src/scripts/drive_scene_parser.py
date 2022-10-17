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
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, TransformStamped, Vector3, Quaternion

import message_filters as mf

import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

import image_geometry

# CLS =  {0: {'class': 'person',     'color': [220, 20, 60]},
#         1: {'class': 'bicycle',    'color': [119, 11, 32]},
#         2: {'class': 'car',        'color': [  0,  0,142]},
#         3: {'class': 'motorcycle', 'color': [  0,  0,230]},
#         5: {'class': 'bus',        'color': [  0, 60,100]},
#         7: {'class': 'truck',      'color': [  0,  0, 70]}
#         }

LARGE_INTEGER = 10000

CLS =  {0: {'class': 'person',     'color': [220, 20, 60]},
        1: {'class': 'bicycle',    'color': [220, 20, 60]},
        2: {'class': 'car',        'color': [  0,  0,142]},
        3: {'class': 'motorcycle', 'color': [220, 20, 60]},
        5: {'class': 'bus',        'color': [  0,  0,142]},
        7: {'class': 'truck',      'color': [  0,  0,142]}
        }
class DriveSceneParser:
    def __init__(self, camera_topic, camera_info_topic, result_topic, frame_id, object_topic='', lane_topic='', freespace_topic='', mask_img=None):
        self.frame_id = frame_id
        self.info = rospy.wait_for_message(camera_info_topic, CameraInfo)
        self.model = image_geometry.PinholeCameraModel()
        self.model.fromCameraInfo(self.info)
        
        self.object_msg = Detection2DArray
        self.lane_msg = ImageMsg()
        self.freespace_msg = MarkerArray()
        
        self.pub1 = rospy.Publisher(result_topic, ImageMsg, queue_size=10)
        self.pub2 = rospy.Publisher(object_topic + '_with_distance', Detection2DArray, queue_size=10)
        
        self.sub_camera = mf.Subscriber(camera_topic, ImageMsg) if camera_topic != '' else None
        self.sub_object = mf.Subscriber(object_topic, Detection2DArray) if object_topic != '' else None
        self.sub_lane = mf.Subscriber(lane_topic, ImageMsg) if lane_topic != '' else None
        self.sub_freespace = mf.Subscriber(freespace_topic, MarkerArray)  if freespace_topic != '' else None
        
        self.mask_img = mask_img
        
        subs = [self.sub_camera, self.sub_object, self.sub_lane, self.sub_freespace]
        callbacks = [self.callback_camera, self.callback_object, self.callback_lane, self.callback_freespace]

        self.subs = []
        self.callbacks = []
        
        for sub, callback in zip(subs, callbacks):
            if sub is not None:
                self.subs.append(sub)
                self.callbacks.append(callback)    
        
        self.ts = mf.ApproximateTimeSynchronizer(self.subs, 10, 0.1, allow_headerless=True)
        self.bridge = CvBridge()
        
        self.tf2_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf2_buffer)
        
        if freespace_topic != '':
            max_trial = 100
            trial = 0
            while not rospy.is_shutdown():
                try:
                    source_frame = rospy.wait_for_message(freespace_topic, MarkerArray).markers[0].header.frame_id
                    target_frame = rospy.wait_for_message(camera_topic, ImageMsg).header.frame_id
                    
                    self.trans = self.tf2_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
                    rospy.logwarn('tf retrieved')
                    break
                except:
                    rospy.logwarn('retrieving tf...')
                    trial += 1
                    
                if trial > max_trial:
                    rospy.logwarn('retrieving tf failed please check if tf is valid')
                    break
            
        self.ts.registerCallback(self.callback)
                
    def to_numpy(self, msg, type='gray'):
        img = msg.data
        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape((msg.height, msg.width, -1))
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
        elif img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    
    def xywh2xyxy(self, x, size):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = int((x[:, 0] - x[:, 2] / 2) * size[1])  # top left x
        y[:, 1] = int((x[:, 1] - x[:, 3] / 2) * size[0])  # top left y
        y[:, 2] = int((x[:, 0] + x[:, 2] / 2) * size[1])  # bottom right x
        y[:, 3] = int((x[:, 1] + x[:, 3] / 2) * size[0])  # bottom right y
        return y

    def callback_camera(self, camera_msg):
        self.camera_msg = camera_msg

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
            
        if self.sub_camera is not None:
            img = self.to_numpy(self.camera_msg).copy()
        else:
            img = np.zeros((self.info.height, self.info.width, 3)).astype(np.uint8)
        
        freespace_dist = []
        freespace_coord = []
        if self.sub_freespace is not None:
            freespace = []
            freespace_id = []
            for marker in self.freespace_msg.markers:
                # if (0 <= marker.id < 90) or (270 < marker.id < 360):
                if marker.id > 180:
                    point = do_transform_point(PointStamped(point=marker.pose.position), self.trans)
                    # point = PointStamped(point=marker.pose.position)
                    point_projected = self.model.project3dToPixel((point.point.x, point.point.y, point.point.z))

                    if (0 < point_projected[0] < self.info.width) and (0 < point_projected[1] < self.info.height):
                        freespace.append([int(point_projected[0]), int(point_projected[1])])
                        freespace_id.append(marker.id)
                        freespace_dist.append(np.sqrt(point.point.x ** 2 + point.point.y ** 2 + point.point.z ** 2))
                        freespace_coord.append([int(point_projected[0]), int(point_projected[1])])
                        
            freespace = np.array(freespace).astype(np.int32)
            if len(freespace) != 0:
                freespace = freespace[freespace.argsort(axis=0)[:, 0]]
                freespace = np.vstack([[0, self.info.height], freespace, [self.info.width, self.info.height]])
            else:
                freespace = np.array([[0, self.info.height]])
                freespace_dist = [LARGE_INTEGER]
                freespace_coord = [[0, 0]]
                
            # img = cv2.fillPoly(img, [freespace], (128, 64, 128))
            
            freespace_coord = np.array(freespace_coord)
        
            
        if self.sub_lane is not None:
            msg = self.to_numpy(self.lane_msg)
            if msg is None:
                msg = np.zeros((self.lane_msg.height, self.lane_msg.width, 3)).astype(np.uint8)
            img[msg > 128] = 255
        
        if self.sub_object is not None:
            dets = []
            forward_det_msg = Detection2DArray()
            forward_det_msg.header = self.object_msg.header
            for det in self.object_msg.detections:
                bbox = det.bbox
                xyxy = self.xywh2xyxy(np.array([[bbox.center.x, bbox.center.y, bbox.size_x, bbox.size_y]]), (self.info.height, self.info.width)).tolist()[0]
                id = det.results[0].id
                
                if self.mask_img is not None and self.mask_img[int(bbox.center.y * self.info.height), int(bbox.center.x * self.info.width)]:
                    xyxy.append(id)
                    dets.append(xyxy)
                    
                    forward_det_msg.detections.append(det)
            
            for idx, det in enumerate(dets):
                det = [int(i) for i in det]
                bottom_center = [int((det[0] + det[2]) / 2), det[3]]
                
                img = cv2.rectangle(img, tuple(det[:2]), tuple(det[2:4]), CLS[det[-1]]['color'], -1)
                
                closest_point = np.argmin(np.abs(freespace_coord[:, 0] - bottom_center[0]))
                if np.linalg.norm(freespace_coord[closest_point] - bottom_center) > 25:
                    dist = -1
                else:
                    dist = freespace_dist[closest_point]
                    
                forward_det_msg.detections[idx].results[0].score = dist
                
                # img = cv2.putText(img, str(int(dist)), tuple(bottom_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # img = cv2.putText(img, str(freespace_coord[closest_point]) + str(bottom_center), tuple(bottom_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
        if self.mask_img is not None and self.sub_camera is None:
            img[~self.mask_img] = [0, 255, 255]
                
        msg = self.bridge.cv2_to_imgmsg(img)
        msg.header.frame_id = self.frame_id
        msg.header.stamp = now
        self.pub1.publish(msg)
        self.pub2.publish(forward_det_msg)

    def __len__(self):
        return 0

if __name__ == '__main__':
    rospy.init_node('drive_scene_parser')
    camera_topic      = rospy.get_param('~camera_topic',      '')
    camera_info_topic = rospy.get_param('~camera_info_topic', '')
    object_topic      = rospy.get_param('~object_topic',      '')
    lane_topic        = rospy.get_param('~lane_topic',        '')
    freespace_topic   = rospy.get_param('~freespace_topic',   '')
    result_topic      = rospy.get_param('~result_topic',      '/image_parsed')
    frame_id          = rospy.get_param('~frame_id',          'camera1')
    mask_img          = rospy.get_param('~mask_img',          '')
    
    if mask_img != '':
        mask_img = cv2.imread(os.path.join(rospkg.RosPack().get_path('drive_scene_parser'), 'mask', mask_img))[:, :, 0] > .5
    else:
        mask_img = None
        
    publisher = DriveSceneParser(camera_topic, camera_info_topic, result_topic, frame_id, object_topic, lane_topic, freespace_topic, mask_img)
    rospy.spin()