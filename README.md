# Ground Truth Generation

A package to generate and correct the recorded position data of a flying robot and an object (a human) while the robot is being controlled manually to track the object within a flat field covered by some aruco marker, which are used to remove the errors of position data recorded by the embedded visual odometry module of the robot. The goal of such dataset is to be used in developing the real-time appearance-model- or motion-model-based visual object tracking, automatic motion model generation, monocular depth estimation, V-INS sensor fusion, optimal path generation, cinematography, ... . 

The dataset which is to be generated is considered to involve the following data:

1. The 2D position (on the flat field) of the single object in terms of time

2. The 6DOF position and orientation of the flying robot in terms of time

3. The time-sampled images of the object, captured by the mounted camera of the robot

All the above data must be synchronized in terms of time, such that there are a certain set of data in each particular moment.

The system is designed performing real flight tests during which the codes are developed and deployed based on the actual platform configurations. The robot's odometry data is recorded through a wi-fi link, using the ROS framework (kinetic distro). The flying robot was a DJI Ryze Tello quadcopter. The object is considered to be a single person running in a flat rectangular field. To obtain the position of the object, to cameras are placed in the direction of the two axes of the rectangualr field. The corners of the field are marked so that the exact dimensions and position of the field is determined to obtain a percise position dataset. The data of the two camera are fused, rectified. A homographic transform is then used to get the position of the object in the field. Next - knowing the exact dimensions of the field - a mapping is performed to convert the pixel poses into metric poses. The object is detected visually in the two cameras' frames using background subtraction. To correct the flying robot's odometry drifts, the 6DOF visual navigation data obtained from Aruco markers is used. A gradient descent optimization algorithm is executed to fit the drifted odometry data to the pose data relative to the markers, which is recorded in scattered moments. It is obvious the the exact postions and dimensions of the markers must be known. 


## Setup

The system is developed and tested in ubuntu 16.04. The installation of ROS (mine was `kinetic` distro) is a basic requirement. The whole repository is placed within a ROS workspace. 

Having anaconda installed, it is recommended to create a conda environment to install the necessary packages to run the software within. So start using:

```
export PATH=~/anaconda3/bin:$PATH
source ~/anaconda3/etc/profile.d/conda.sh
conda create --name gtg python=3.6
conda activate gtg
```

Required python packages:

```
opencv 					(v: 4.6.0 and 4.5.5 tested)
opencv-contrib-python
rospy					(v: )
rosbag					(v: )
cv_bridge				(v: )
sensor_msgs				(v: )
pandas					(v: )
yaml					(v: )
matplotlib
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

All the executable scripts lie inside *src* folder.

[//]:# "TODO: For each data source (dedicated to a particular flight test), all the data, and the following algorithms' inputs and outputs must be read and written in a <save-dir>. The only input given to each algorithm must be a <save-dir> address parsed in the command-line"

[//]:# "The numerical parameters for each data source (a flight test) must be read and written in a 'params' file. Change the sturcture of the scripts so that there is no need to set any number inside modules or read and copy numbers from terminal outputs" 

### 1. Log and Visualize the Robot's Odometry

1. Write the marker ID poses inside the *telloGTGeneration.py* script. (Fill the `idPoses` variable like sample comments)

[//]:# "TODO: The marker ID poses should be written in a <params-file> inside the <save-dir>"

2. Give the *telloGTGeneration.py* script an address to save the output (<save-dir>), and the address of drone camera calibration file. These are two parameters `saveAddress, calibFile` in the constructor of the class `ArucoBasedDroneGroundTruthGeneration`.

3. Set the <save-dir> in `plotOdom.py` (The variable: `saveDir`).

[//]:# "TODO: The address to save the output must be passed with argparse"

2. In a terminal:

```
# Requires ROS
python telloGTGeneration.py 
```

3. In another terminal:

```
# Within <bag-file-dir> 
# Requires ROS
rosbag play <bag-file-address> 
```
[//]:# "TODO: The <bag-file-dir> must be the <save-dir>"

4. When the output image of *telloGTGeneration.py* is shown. Hit **A** key the first time you saw a marker in the robot's camera view. This position will be considered as the initial point of the drone pose. Note that, the **A** key must be hit when a marker is detected.

5. Meanwhile, in a new terminal, run the following so that you can see the poses obtained from markers along with the drone odometry plot.

```
# Does not require ROS
python plotOdom.py
```

**OUTPUT:** 
* <save-dir>/odomPoses.csv
* <save-dir>/rawMarkerPoses.csv
* All the drone images in which the markers are visible in <save-dir>/markerDetectionFrames

### 2. Optimize the drone's path using the aruco markers' data 

Using a gradient descent approach, this feature corrects the drifted drone odometry poses to match the reliable Aruco pose data as much as possible.

*NO ROS REQUIREMENT*

1. In the `dronePoseDataOptimization.py` module, set the directory containing two files: `odomPoses.csv` and `markerPoses.csv`. (In the variable: `path`) 

[//]: # "TODO: The 'odomPoses.csv' and 'markerPoses.csv' must be automatically read from the <save-dir>"

#### Finding the optimal parameters for drone odometry data correction

2. In the `dronePoseDataOptimization.py` module, in `__main__`, uncomment the call to *gradientDescentOptimize()* and comment the *visualize()* function. 

[//]: # "TODO: The two separate functionalities must be passed with argparse, not commenting and uncommenting."

3. Run:

```
python dronePoseDataOptimization.py
```

**OUTPUT:** 
* The log for the optimization in the terminal. Parameters are printed in each optimization step and the convergence procedure can be monitored.

[//]: # "TODO: Optimal parameter values must be saved in the <params-file> inside <save-dir>"

When desired (when the change in parameters is ignorable), kill the program and save the last printed parameters.

#### Correcting the odometry data using the optimal parameters

2. In the *__init__()* function of the class *OptimizeDronePoseData()*, set the optimal parameter values.

[//]: # "TODO: Optimal parameter values must be read from the <params-file> inside <save-dir>"

3. In the end of the *dronePoseDataOptimization.py*, comment the call to *gradientDescentOptimize()* and uncomment the *visualize()* function. 

4. Run:

```
python dronePoseDataOptimization.py
```

**OUTPUT:** 
* 'correctedPose.csv'    (The corrected odometry poses)

[//]: # "TODO: It must be saved in the <save-dir>"

* A plot showing the raw odometry poses, corrected odometry poses, and the Aruco marker poses (as scattered points)

[//]: # "TODO: Save and overwrite the odometry poses plot in <save-dir>"


### 3. Log and Visualize the Object's Path

*NO ROS REQUIREMENT*


1. After setting the address of front and side view camera videos in the *writeVideos.py*, run:

[//]: # "TODO: Pass the address of front and side view camera videos to the 'writeVideos.py' using argparse"

```
python writeVideos.py
```

The numbered frames of the two videos are saved in the two folders *frames_front* and *frames_side* in the root. 

2. Compare and find the number of the initial frame of each video in a particular interval in which you want the object path to be recorded.

3. Set the followings in *objectGTGeneration.py*:

* Initial frame number of each video 
* The field length and width
* The fps of two videos (must be identical! - default: 30)
* The address to the directory containing the both video files as well the name for each file (The variables `front_video_name, side_video_name, videos_path` in `objectGTGeneration.py`).

[//]: # "TODO: All of the above must be set by user in the <params-file> and read by code from it"

4. Run:

[//]: # "FIX: Runnig objectGTGeneration.py, a fake argument file must be passed. If not, there is an error"

```
python objectGTGeneration.py
```

5. Following the instructions, 

```
	a: quit
 	s: crop ROI for front view
 	d: crop ROI for side view
 	f: start processing 
```

For each of front and side views, crop the tightest rectangle over the area in which the field is seen. In the cropped windows, click on the marked points in this order:

	- top-left point in the field
	- top-right point in the field
	- down-right point in the field
	- down-left point in the field

[//]: # "TODO: The above corner pixel addresses are now copied from the terminal. Save them in the <params-file>. Change the code so that if these data are saved in the <params-file>, there is no need to crop the image and select the corners again"

6. If you are to work with a same couple of video files several times, you can change the following part of the `objectGTGeneration.py`:

```
self._ROI_front = None
self._fieldCorners_front = []
self._ROI_side = None
self._fieldCorners_side = []
self._fieldWidth = None
self._fieldLength = None
initFrame_f = None
initFrame_s = None
```

To something like this:

```
self._ROI_front = (6, 451, 1852, 185)
self._fieldCorners_front = [[584, 64], [1377, 75], [1846, 182], [39, 146]]
self._ROI_side = (1, 343, 1272, 123)
self._fieldCorners_side = [[11, 116], [478, 32], [973, 25], [1263, 119]]
self._fieldWidth = 24.2
self._fieldLength = 32
initFrame_f = 31784
initFrame_s = 33070
```

So that you don't need to set these parameters again and again. Note that the above data are printed in the terminal as soon as they are obtained (with the instructions of step 5) in the very first time.


Hit **F**, so the pose estimation starts. 
The data must be saved if the related code block within the function 'savePath' is active.

**OUTPUT:** 
* objectPoses.csv

[//]: # "TODO: It must be saved in the <save-dir>"


### 4. Visualize and save the pose of robot and object simultaneously

1. Fisrt, save the bag images of drone camera. To do so, clone [this repo](https://github.com/hamidrezafahimi/technical_utils). Navigate to the root of the cloned repo. Then:

```
# Requires ROS
# Assuming you're in the root of 'technical_utils' repo

cd ROS1/process_bagfiles

python processImageBags.py "write" "<save-dir>"
```

**OUTPUT:** 
* All numbered bag images within an output folder inside the bag file's directory
* A file named "info.csv" within the output folder including the time instant for each, since beginning

2. Considering the fact that the front view video and side view video have been synchronized (Their frame difference number is recorded and set in *objectGTGeneration.py*), pick one of the folders *frames_front* or *frames_side*, and compare the frame sequence within, with the bag file's frame sequence which is written inside aforementioned output folder. Find the two corresponding frames (which seem to be simultaneous). 

3. Save the bag file's frame time (within *info.csv*) as the initialization time difference between the bag file frame sequence and front (or side) camera frame sequence in the *plotTrackingPath.py* as the value of the variable 'odom_gt_dt'. Note that a positive number means that the bag file is started *before* the video stream, which is often true.   

4. Having the true positions of markers, save the longitudinal and leteral distance of the down-left corner of the field with the Aruco marker number of which is 1 (Normally, the first marker which is placed in front of where the drone takes off the ground) in the *plotTrackingPath.py* as 'object_dx' and 'object_dy'. (Declare: corner pose - marker pose)

[//]: # "TODO: All of the above parameters (odom_gt_dt, object_dx, object_dy) must be set by user in the <params-file> and read by code from it"

5. In a terminal, run:

```
# Requires ROS
python telloGTGeneration.py 
```

6. In another terminal, run:

*You must provade the following data for the next command:*
*- odomPoses.csv*    	(The output of dronePoseDataOptimization.py)
*- markerPoses.csv*	(The output of telloGTGeneration.py)
*- objectPoses.csv*	(The output of objectGTGeneration.py)

[//]: # "TODO: All of the above must be provided in the <save-dir>"

```
python plotTrackingPath.py 
```

[//]: # "TODO: The 'plotTrackingPath.py' must read data from <save-dir>"

**OUTPUT:** 
* Online plot of drone odometry data, markers' data, and object data

[//]: # "TODO: Save and overwrite the above all-data plot in <save-dir>"
