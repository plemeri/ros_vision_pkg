#!/usr/bin/python3
import rospy
import rospkg
import os
import cv2
import numpy as np
import yaml

from cv_bridge import CvBridge
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image as Image
from sensor_msgs.msg import CameraInfo

class CameraPublisher:
    def __init__(self, ID, camera_param, size=(480, 640), publish_rate=30, image_topic='camera1'):
        self.ID = int(ID)
        self.size = size
        self.cap = cv2.VideoCapture(self.ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.size[1])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[0])

        self.frame_id = image_topic
        self.info = CameraInfo()
        self.info.height = self.size[0]
        self.info.width  = self.size[1]
        self.info.distortion_model = 'plumb_bob'
        self.info.D = camera_param['D']
        self.info.K = camera_param['K']
        self.info.R = camera_param['R']
        self.info.P = camera_param['P']
        self.rate = rospy.Rate(publish_rate)
        
        self.pub1 = rospy.Publisher('/' + image_topic + '/image_raw', Image, queue_size=10)
        self.pub2 = rospy.Publisher('/' + image_topic + '/camera_info', CameraInfo, queue_size=10)
        # self.sub = rospy.Subscriber(image_topic, ImageMsg, image_callback, tcp_nodelay=True)
        
        self.bridge = CvBridge()
        
    def to_numpy(self, msg):
        img = msg.data
        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape((*self.size, 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def callback(self):
        while not rospy.is_shutdown():
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret is False:
                    frame = np.zeros((*self.size, 3)).astype(np.uint8)
                
                now = rospy.Time.now()
                
                msg1 = self.bridge.cv2_to_imgmsg(frame)
                msg1.header.stamp = now
                msg1.header.frame_id = self.frame_id
                
                msg2 = self.info
                msg2.header.stamp = now
                msg2.header.frame_id = self.frame_id
                
            else:
                msg1 = Image()
                msg2 = CameraInfo()
                
            self.pub1.publish(msg1)
            self.pub2.publish(msg2)
            
            self.rate.sleep()

    def __len__(self):
        return 0

if __name__ == '__main__':
    rospy.init_node('camera_publisher')
    camera_id         = rospy.get_param('~camera_id',         1)
    image_height      = rospy.get_param('~image_height',      480)
    image_width       = rospy.get_param('~image_width' ,      640)
    publish_rate      = rospy.get_param('~publish_rate' ,     30)
    image_topic       = rospy.get_param('~image_topic',       'camera1')

    camera_param = yaml.load(open(os.path.join(rospkg.RosPack().get_path('camera_publisher'), 'param', 'camera_param.yaml')), yaml.FullLoader)
    
    publisher = CameraPublisher(camera_id, camera_param, (image_height, image_width), publish_rate, image_topic)
    publisher.callback()