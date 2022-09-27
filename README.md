# ros_vision_pkg

# Installation

## prerequisite

```
sudo apt install ros-[distro]-vision-msgs
sudo apt install ros-[distro]-cv-bridge
sudo apt install ros-[distro]-pcl-conversions
sudo apt install ros-[distro]-pcl-ros
sudo apt install ros-[distro]-roslint
sudo apt install ros-[distro]-image-geometry
sudo apt install ros-[distro]-tf2-sensor-msgs
sudo apt install ros-[distro]-tf2-geometry-msgs

sudo apt install libyaml-cpp-dev
sudo apt install python3-pip
sudo apt install python-pip

pip3 install --upgrade pip
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r src/requirements.txt
```

## build package

```
cd ros_vision_pkg
catkin_make
source devel/setup.bash
```

## lane_detector

1. change file permission
    ```
    cd src/lane_detector/src/scripts
    chmod +x detect_node.py
    ```

2. install python dependencies
    ``` 
    pip install -r requirements.txt 
    ```

3. checkpoint 
  + Download checkpoint from [Link](https://drive.google.com/file/d/1DONSeQ43PwAnW-Eehpvo5UaRAJP4mhZy/view?usp=sharing)
  + Move file as follows `src/lane_detector/src/scripts/snapshots/Legacy/latest.pth`. Create folder if needed. and locate checkpoint  

4. change shebang
    open file ```detect_node.py``` and change the first line starts with ```#!```.
    ```
    #!/home/taehoon1018/anaconda3/envs/inspyrenet/bin/python3 --> [your python path]
    ```

## object_detector

1. change file permission
    ```
    cd src/object_detector/src/scripts
    chmod +x detect_node.py
    ```

2. install python dependencies
    ``` 
    pip install -r requirements.txt 
    ```

3. change shebang
    open file ```detect_node.py``` and change the first line starts with ```#!```.
    ```
    #!/home/taehoon1018/anaconda3/envs/yolov5/bin/python3 --> [your python path]
    ```

+ checkpoints
[yolov5l](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EYXI2gekiQRFsDtkeR7Z6yUBA1OjXCLD4zbApdv6In-cDw?e=RaMBpa)
[traffic_sign](https://postechackr-my.sharepoint.com/:u:/g/personal/taehoon1018_postech_ac_kr/EfyxOJKmXDVBuLQjuhuhz30BR1j1e9ZWUnCXkwNJgPTnCw?e=4BTbYX)

## drive_scene_parser

1. change file permission
    ```
    cd src/drive_scene_parser/src/scripts
    chmod +x drive_scene_parser.py
    ```

2. install python dependencies
    ``` 
    pip install -r requirements.txt 
    ```

3. change shebang
    open file ```drive_scene_parser.py``` and change the first line starts with ```#!```.
    ```
    #!/usr/bin/python3 --> [your python path]
    ```

# run nodes

## Quick Launch (Carla)

```
source devel/setup.bash
roslaunch src/Launch/vision_carla.launch
```
## Quick Launch (Imcar)

```
source devel/setup.bash
roslaunch src/Launch/vision_imcar.launch
```

## Launch separately

### lane_detector
```
roslaunch lane_detector lane_detector.launch image_topic:=[image topic from CARLA]
```

### object_detector
```
roslaunch object_detector object_detector.launch image_topic:=[image topic from CARLA]
```

### drive_scene_parser
```
roslaunch drive_scene_parser drive_scene_parser.launch
```

The output of the ```drive_scene_parser``` is ```/image_parsed```

