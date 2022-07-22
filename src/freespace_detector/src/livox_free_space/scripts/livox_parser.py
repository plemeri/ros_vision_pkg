#!/usr/bin/env python
from __future__ import print_function, absolute_import, division
from random import sample

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
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Quaternion, Vector3


class LivoxParser:
    def __init__(self, height_offset, sample_rate, result_topic, freespace_topic):
        self.pub = rospy.Publisher(result_topic, MarkerArray, queue_size=10)
        self.sub = rospy.Subscriber(freespace_topic, PointCloud2, self.callback, tcp_nodelay=True)
        self.height_offset = height_offset
        self.sample_rate = sample_rate
        
    def callback(self, msg):
        xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg) # N times 3
        x, y = xyz[:, 0], xyz[:, 1]
        

        theta = np.rad2deg(np.arctan2(x, y)) + 180 # 0 ~ 360 degrees
        theta /= self.sample_rate
        theta = np.round(theta).astype(int)
        theta[theta >= 360] -= 360
        
        dist = np.sqrt(x ** 2 + y ** 2)
        
        max_dists = []
        out_msg = MarkerArray()
        
        for i in range(int(360 / self.sample_rate)):
            dist_i = dist[theta == i]
            if len(dist_i) > 0:
                max_dist = dist_i.max()
                max_dists.append(max_dist)
                min_x, min_y = -np.sin(np.deg2rad(i)) * max_dist, -np.cos(np.deg2rad(i)) * max_dist
                marker = Marker()
                marker.pose.position = Point(x=min_x, y=min_y, z=-self.height_offset)
                marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
                
                marker.type = Marker.SPHERE
                marker.header.frame_id = msg.header.frame_id
                marker.header.stamp = rospy.Time.now()
                marker.id = i
                
                if i == 0:
                    marker.scale = Vector3(x=0.5, y=0.5, z=0.5)
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                else:
                    marker.scale = Vector3(x=0.5, y=0.5, z=0.5)
                    marker.color.r = 0.8
                    marker.color.g = 0.5
                    marker.color.b = 1.0
                    marker.color.a = 1.0
            
                out_msg.markers.append(marker)

        
        # msg.header.frame_id = msg.header.frame_id
        # msg.header.stamp = rospy.Time.now()
        self.pub.publish(out_msg)

    def __len__(self):
        return 0

if __name__ == '__main__':
    rospy.init_node('livox_parser')
    height_offset       = rospy.get_param('~height_offset',       '1.8')
    sample_rate       = rospy.get_param('~sample_rate',       '1.0')
    freespace_topic   = rospy.get_param('~freespace_topic',   '/points_ground')
    result_topic      = rospy.get_param('~result_topic',      '/image_parsed')
    
    publisher = LivoxParser(height_offset, sample_rate, result_topic, freespace_topic)
    
    rospy.spin()