#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import PointCloud2

from std_msgs.msg import Float32MultiArray as Floats
import os
from point_cloud2 import *


def no_ground_to_ring(msg, max_dist, sample_rate):
    # -- make angle based min_distance data
    max_deg = int(360 // sample_rate)
    arr = pointcloud2_to_xyz_array(msg)

    angle = np.arctan2(arr[:, 1], arr[:, 0])
    angle = np.rad2deg(angle) + 180
    angle /= sample_rate
    angle = np.round(angle)
    angle[angle > max_deg - 1] = max_deg - 1

    dist = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
    min_dist = np.ones(max_deg) * max_dist
    for i, ang in enumerate(angle):
        if min_dist[int(ang)] > dist[i]:
            min_dist[int(ang)] = dist[i]
    # -- make angle based min_distance data -- end

    # -- to pointcloud for visualization
    out_angle = np.deg2rad(np.linspace(0, 359, max_deg))
    out_x = -np.cos(out_angle) * min_dist
    out_y = -np.sin(out_angle) * min_dist
    out_z = np.zeros(max_deg)
    out_cloud = np.stack([out_x, out_y, out_z], axis=-1)
    out_cloud = xyz_array_to_pointcloud2(out_cloud, frame_id=msg.header.frame_id)
    out_cloud.header.stamp = rospy.Time.now()
    # -- to pointcloud for visualization -- end

    # -- publish
    pub_points.publish(out_cloud)
    pub_arr.publish(Floats(data=min_dist.astype(np.float32).flatten()))
    # -- publish -- end

if __name__ == '__main__':
    rospy.init_node('points_ring', anonymous=True)

    topic_name = rospy.get_param('~topic')
    max_dist = rospy.get_param('~max_dist')
    sample_rate = rospy.get_param('~sample_rate')

    # publishers
    pub_points = rospy.Publisher('/points_ring', PointCloud2, queue_size=10)
    pub_arr = rospy.Publisher('numpy_ring', numpy_msg(Floats), queue_size=10)

    # subscribers
    sub_points = rospy.Subscriber(topic_name, PointCloud2,
                                  lambda msg: no_ground_to_ring(msg, max_dist, sample_rate))
    rospy.spin()
