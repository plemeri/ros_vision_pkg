import message_filters as mf
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray as Floats, Float32

from point_cloud2 import *
from utils import *


def callback(ring, roll, pose):
    # -- make angle based min_distance data
    dist = ring.data
    num_borders = 9
    num_rollouts = 3

    paths = roll.data.reshape((num_rollouts * num_borders, -1, 2))
    num_waypoints = paths.shape[1]
    pub_nway.publish(Float32(data=float(num_waypoints)))
    # assert paths.shape[0] % 5 == 0
    rollout_num = paths.shape[0]
    print(paths.shape)

    position = pose.pose.position
    position = [position.x, position.y, position.z]

    orientation = pose.pose.orientation
    orientation = [orientation.x, orientation.y, orientation.z, orientation.w]

    _, _, yaw = quaternion_to_euler_angle(orientation)

    paths = (paths - np.array(position[:2])).reshape((-1, 2))
    paths = rotate(paths, angle=-yaw)

    paths_angle = np.arctan2(paths[:, 1], paths[:, 0])
    paths_angle = np.rad2deg(paths_angle) + 180

    paths_angle = np.round(paths_angle).astype(int).clip(0, 359)
    paths_dist = np.sqrt(paths[:, 1] ** 2 + paths[:, 0] ** 2)
    free_path = dist[paths_angle] > paths_dist
    free_path = free_path.reshape((rollout_num, -1))
    paths_dist = paths_dist.reshape((rollout_num, -1))

    free_path_idx = np.logical_not(free_path)
    free_path_idx[np.where(free_path_idx.sum(axis=-1) == 0)[0], -1] = 1
    path_min_point = np.argmax(free_path_idx, axis=-1)
    path_min_dist = paths_dist[np.linspace(0, rollout_num - 1, rollout_num).astype(int), [path_min_point]]

    path_min_point = path_min_point.reshape((-1, num_borders)).min(axis=-1)
    path_min_dist = path_min_dist.reshape((-1, num_borders)).min(axis=-1)

    path_cloud = np.stack([paths[:, 0], paths[:, 1], np.zeros(paths.shape[0])], axis=-1)

    out_cloud = xyz_array_to_pointcloud2(path_cloud, frame_id='velodyne')

    max_path = np.argmax(path_min_dist)
    max_path = paths.reshape((-1, num_waypoints, 2))[max_path]

    max_path_cloud = xyz_array_to_pointcloud2(
        np.stack([max_path[:, 0], max_path[:, 1], np.zeros(max_path.shape[0])], axis=-1), frame_id='velodyne')

    # -- publish
    pub_points.publish(out_cloud)
    pub_max_points.publish(max_path_cloud)
    pub_path.publish(
        Floats(data=np.concatenate([path_min_point, path_min_dist.squeeze()]).astype(np.float32).flatten()))
    # -- publish -- end


if __name__ == '__main__':
    rospy.init_node('path_ring')

    # publishers
    pub_points = rospy.Publisher('/path_ring', PointCloud2, queue_size=10)
    pub_max_points = rospy.Publisher('/path_max_ring', PointCloud2, queue_size=10)
    pub_path = rospy.Publisher('/path_min', numpy_msg(Floats), queue_size=10)
    pub_nway = rospy.Publisher('/num_waypoints', Float32, queue_size=10)

    # synchonized subscribers
    sub_points = mf.Subscriber('/numpy_ring', numpy_msg(Floats))
    sub_rollout = mf.Subscriber('/rollout_arr', numpy_msg(Floats))
    sub_pose = mf.Subscriber('/localizer_pose', PoseStamped)
    ts = mf.ApproximateTimeSynchronizer([sub_points, sub_rollout, sub_pose], queue_size=10, slop=0.1,
                                        allow_headerless=True)
    ts.registerCallback(callback)

    rospy.spin()
