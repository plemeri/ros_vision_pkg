#!/usr/bin/env python
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from point_cloud2 import *

prev_pose = np.zeros(3)
aug_points = []

def get_translation(tfstamped):
    t = tfstamped.transform.translation
    return np.array([t.x, t.y, t.z])


def callback(points, frame_num):
    global aug_points
    global prev_pose
    try:
        trans = tf2_buffer.lookup_transform('map', points.header.frame_id, rospy.Time(0))
	#TODO
        # trans_inv = tf2_buffer.lookup_transform('velodyne', 'map', rospy.Time(0))
    except:
        return
    rotated_points = do_transform_cloud(points, trans)

    rotated_points.header.frame_id = 'map'
    aug_points.append(pointcloud2_to_array(rotated_points))

    if len(aug_points) > frame_num:
        del aug_points[0]

    prev_pose = get_translation(trans)
    pub_points.publish(array_to_pointcloud2(np.concatenate(aug_points), frame_id='map'))



if __name__ == '__main__':
    rospy.init_node('aug_points')

    input_topic = rospy.get_param('~input_topic')
    output_topic = rospy.get_param('~output_topic')
    frame_num = rospy.get_param('~frame_num')

    tf2_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf2_buffer)

    # publishers
    pub_points = rospy.Publisher(output_topic, PointCloud2, queue_size=10)

    # synchonized subscribers
    sub_points = rospy.Subscriber(input_topic, PointCloud2, lambda points: callback(points, frame_num))

    rospy.spin()
