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

class ImagePublisher:
    def __init__(self, image_dir, camera_param, size=(480, 640), publish_rate=30, image_topic='image_raw', camera_info_topic='camera_info'):
        self.image_dir = image_dir
        self.images = os.listdir(self.image_dir)
        self.images.sort()
        self.size = size

        self.frame_id = 'camera1'
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
        self.idx = 0
        
    def to_numpy(self, msg):
        img = msg.data
        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape((*self.size, 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def callback(self):
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            
            img = cv2.imread(os.path.join(self.image_dir, self.images[self.idx]))
            img = cv2.resize(img, self.size[::-1])
            msg1 = self.bridge.cv2_to_imgmsg(img)
            msg1.header.stamp = now
            msg1.header.frame_id = self.frame_id
            
            msg2 = self.info
            msg2.header.stamp = now
            msg2.header.frame_id = self.frame_id
                
            self.pub1.publish(msg1)
            self.pub2.publish(msg2)
            
            self.rate.sleep()
            
            self.idx += 1
            if self.idx >= len(self.images):
                self.idx = 0

    def __len__(self):
        return 0

if __name__ == '__main__':
    rospy.init_node('image_publisher')
    image_dir         = rospy.get_param('~image_dir',         '/home/taehoon1018/Downloads/save_imgs')
    image_height      = rospy.get_param('~image_height',      480)
    image_width       = rospy.get_param('~image_width' ,      640)
    publish_rate      = rospy.get_param('~publish_rate' ,     30)
    image_topic       = rospy.get_param('~image_topic',       'camera1')

    camera_param = yaml.load(open(os.path.join(rospkg.RosPack().get_path('image_publisher'), 'param', 'camera_param.yaml')), yaml.FullLoader)
    
    publisher = ImagePublisher(image_dir, camera_param, (image_height, image_width), publish_rate, image_topic)
    publisher.callback()
    
