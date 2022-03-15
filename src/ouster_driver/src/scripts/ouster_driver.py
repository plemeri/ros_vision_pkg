#!/home/taehoon1018/anaconda3/envs/yolov5/bin/python3

import rospy
import rospkg
import os
import cv2
import numpy as np
import yaml
import threading
import ouster.client as client
import ros_numpy

from cv_bridge import CvBridge
from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2

class OusterDriver:
    def __init__(self, sensor, pcap, meta, lidar_port, auto_dest, pointcloud_topic, frame_id):
        if sensor:
            hostname = sensor
            if lidar_port or auto_dest:
                config = client.SensorConfig()
                if lidar_port:
                    config.udp_port_lidar = lidar_port
                print("Configuring sensor...")
                client.set_config(hostname, config, udp_dest_auto=auto_dest)
            config = client.get_config(hostname)

            print("Initializing...")
            self.scans = client.Scans.stream(hostname,
                                        config.udp_port_lidar or 7502,
                                        complete=False)

        elif pcap:
            import ouster.pcap as opcap

            if meta:
                metadata_path = meta
            else:
                print("Deducing metadata based on pcap name. "
                    "To provide a different metadata path, use --meta")
                metadata_path = os.path.splitext(pcap)[0] + ".json"

            with open(metadata_path) as json:
                self.info = client.SensorInfo(json.read())

            self.scans = client.Scans(
                opcap.Pcap(pcap, self.info, rate=1.0))

        self.frame_id = frame_id
        # self.rate = rospy.Rate(publish_rate)
        
        self.pub1 = rospy.Publisher(pointcloud_topic, PointCloud2, queue_size=10)
        # self.sub = rospy.Subscriber(image_topic, ImageMsg, image_callback, tcp_nodelay=True)
        
        self.bridge = CvBridge()
        
        
    def callback_scan(self):
        for scan in self.scans:
            # print("frame id: {} ".format(scan.frame_id))
            signal = client.destagger(self.info, scan.field(client.ChanField.SIGNAL))
            signal = client.XYZLut(self.info) #(signal)
            xyz = signal(scan.field(client.ChanField.RANGE))
            # r = scan.field(client.ChanField.REFLECTIVITY)
            # print(r == xyz)
            [x, y, z] = [c.flatten() for c in np.dsplit(xyz, 3)]
            
            signal = np.core.records.fromarrays([x, y, z], names='x, y, z', formats='f4, f4, f4')
            msg = ros_numpy.msgify(PointCloud2, signal, stamp=rospy.Time.now(), frame_id=self.frame_id)
            self.pub1.publish(msg)
            
            if rospy.is_shutdown():
                break

        self.scans.close()


    print("Done")

    def __len__(self):
        return 0

if __name__ == '__main__':
    rospy.init_node('ouster_driver')
    pcap =             rospy.get_param('~pcap',       'OS0-128_Rev-06_Urban-Drive_Dual-Returns.pcap')
    meta =             rospy.get_param('~meta',       'OS0-128_Rev-06_Urban-Drive_Dual-Returns.json')
    
    sensor =           rospy.get_param('~sensor',     'os1-991951000038.local')
    lidar_port =       rospy.get_param('~lidar-port', '')
    auto_dest =        rospy.get_param('~auto-dest',  '')
    pointcloud_topic = rospy.get_param('~pointcloud-topic', '/points_raw')
    frame_id =         rospy.get_param('~frame-id', 'velodyne')

    pcap = os.path.join(rospkg.RosPack().get_path('ouster_driver'), pcap)
    meta = os.path.join(rospkg.RosPack().get_path('ouster_driver'), meta)
    
    publisher = OusterDriver(sensor, pcap, meta, lidar_port, auto_dest, pointcloud_topic, frame_id)
    publisher.callback_scan()
