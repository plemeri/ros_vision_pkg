# ros_vision_pkg

# Installation

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

3. change shebang
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

## lane_detector
```
roslaunch lane_detector lane_detector.launch image_topic:=[image topic from CARLA]
```

## object_detector
```
roslaunch object_detector object_detector.launch image_topic:=[image topic from CARLA]
```

## drive_scene_parser
```
roslaunch drive_scene_parser drive_scene_parser.launch
```

The output of the ```drive_scene_parser``` is ```/image_parsed```

