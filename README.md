# Ground Truth Generation

A package to generate and correct the recorded position data of a flying robot and an object (a human) while the robot is being controlled manually to track the object within a flat field covered by some aruco marker, which are used to remove the errors of position data recorded by the embedded visual odometry module of the robot. The goal of such dataset is to be used in developing the real-time appearance-model- or motion-model-based visual object tracking, automatic motion model generation, monocular depth estimation, V-INS sensor fusion, optimal path generation, cinematography, ... . 

The dataset which is to be generated is considered to involve the following data:

1. The 2D position (on the flat field) of the single object in terms of time

2. The 6DOF position and orientation of the flying robot in terms of time

3. The time-sampled images of the object, captured by the mounted camera of the robot

All the above data must be synchronized in terms of time, such that there are a certain set of data in each particular moment.

The system is designed performing real flight tests during which the codes are developed and deployed based on the actual platform configurations. The robot's odometry data is recorded through a wi-fi link, using the ROS framework (kinetic distro). The flying robot was a DJI Ryze Tello quadcopter. The object is considered to be a single person running in a flat rectangular field. To obtain the position of the object, to cameras are placed in the direction of the two axes of the rectangualr field. The corners of the field are marked so that the exact dimensions and position of the field is determined to obtain a percise position dataset. The data of the two camera are fused, rectified. A homographic transform is then used to get the position of the object in the field. Next - knowing the exact dimensions of the field - a mapping is performed to convert the pixel poses into metric poses. The object is detected visually in the two cameras' frames using background subtraction. To correct the flying robot's odometry drifts, the 6DOF visual navigation data obtained from Aruco markers is used. A gradient descent optimization algorithm is executed to fit the drifted odometry data to the pose data relative to the markers, which is recorded in scattered moments. It is obvious the the exact postions and dimensions of the markers must be known. 


## Setup

The system is developed and tested in ubuntu 16.04. The installation of ROS (kinetic distro) is a basic requirement. The whole repository is placed within a ROS workspace. 

It is recommended to create a conda environment to install the necessary to install the softwares within. So start using:

```
conda create --name gtg python=3.6
conda activate gcndepth
```

Required python packages:

```
opencv 			(v: 4.1.1)
rospy			(v: )
sensor_msgs		(v: )
nav_msgs		(v: )
cv_bridge		(v: )
numpy			(v: )
scipy			(v: )
matplotlib		(v: )
pandas			(v: )
yaml			(v: )
rosbag			(v: )
glob			(v: )
```

## Usage

The sequential steps to log, classify and correct the raw data are performed using different features of the system. As follows, the instructions to use the utilities are given so that one can use the package to generate similar datasets. 

First of all, consider that using the ROS's tools requires sourcing the ROS's *setup.bash* file. Whether it is the main bash file in Linux:

```
# In a kinetic-distro ROS:
source /opt/ros/kinetic/devel/setup.bash
```

Or you are working within a workspace:
```
# in the workspace's root:
source devel/setup.bash
```

Also, notice that in ROS, executing different nodes (commands) without a launch file runnig, requires executing:

```
roscore
```


### 1. Log and Visualize the Robot's Odometry

1. Write the marker ID poses inside the *telloGTGeneration.py* script.
[//]: # (TODO: The marker ID poses should be written in a params file)
Also, give the *telloGTGeneration.py* script an address to save the output,
[//]: # (TODO: The address to save the output must be passed with argparse) 
and the address of drone camera calibration file.

2. In a terminal:

```
# Within src/
# Requires ROS
python telloGTGeneration.py 
```

3. In another terminal:

```
# Requires ROS
rosbag play <bag-file-address> 
```

4. When the output image of *telloGTGeneration.py* is shown. Hit **A** key the first time you saw a marker in the robot's camera view. This position will be considered as the inital point of the drone pose.

5. Meanwhile, in a new terminal, run the following so that you can see the poses obtained from markers along with drone odometry plot.

```
# Within src/
# Does not require ROS
python plotOdom.py
```

**OUTPUT:** 
* <saveDir>/odomPoses.csv
* <saveDir>/rawMarkerPoses.csv
* All the drone images in which the markers are visible in <saveDir>/markerDetectionFrames



