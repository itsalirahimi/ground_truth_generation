# Ground Truth Generation

A package to generate and correct the recorded position data of a flying robot and an object (a human) while the robot is being controlled manually to track the object within a flat field covered by some aruco marker, which are used to remove the errors of position data recorded by the embedded visual odometry module of the robot. The goal of such dataset is to be used in developing the real-time appearance-model- or motion-model-based visual object tracking, automatic motion model generation, monocular depth estimation, V-INS sensor fusion, optimal path generation, cinematography, ... . 

The dataset which is to be generated is considered to involve the following data:

1. The 2D position (on the flat field) of the single object in terms of time

2. The 6DOF position and orientation of the flying robot in terms of time

3. The time-sampled images of the object, captured by the mounted camera of the robot

All the above data must be synchronized in terms of time, such that there are a certain set of data in each particular moment.

The system is designed performing real flight tests during which the codes are developed and deployed based on the actual platform configurations. The robot's odometry data is recorded through a wi-fi link, using the ROS framework (kinetic distro). The flying robot was a DJI Ryze Tello quadcopter. The object is considered to be a single person running in a flat rectangular field. To obtain the position of the object, two cameras are placed in the direction of the two axes of the rectangualr field. The corners of the field are marked so that the exact dimensions and position of the field is determined to obtain a percise position dataset. The data of the two camera are fused, rectified. A homographic transform is then used to get the position of the object in the field. Next - knowing the exact dimensions of the field - a mapping is performed to convert the pixel poses into metric poses. The object is detected visually in the two cameras' frames using background subtraction. To correct the flying robot's odometry drifts, the 6DOF visual navigation data obtained from Aruco markers is used. A gradient descent optimization algorithm is executed to fit the drifted odometry data to the pose data relative to the markers, which is recorded in scattered moments. It is obvious the the exact postions and dimensions of the markers must be known. 


## Setup

The system is developed and tested in ubuntu 16.04. The installation of ROS (mine was `kinetic` distro) is a basic requirement. The whole repository is placed within a ROS workspace. 

Having anaconda installed, it is recommended to create a conda environment to install the necessary packages to run the software within. So start using:

```bash
export PATH=~/anaconda3/bin:$PATH
source ~/anaconda3/etc/profile.d/conda.sh
conda create --name gtg python=3.6
conda activate gtg
```

Installation of required python packages:

```bash
pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag cv_bridge
conda install -c "conda-forge/label/cf202003" ros-sensor-msgs
pip install opencv-python==4.5.5.64
pip install opencv-contrib-python==4.5.5.64
pip install pyyaml
pip install matplotlib
pip install pandas
```


## Step 1: Log and Visualize the Robot's Odometry

### Prerequisites

- Within `<save-dir>`, there must be a script named `testData.py` containing the positions of markers as a Python dictionary.
- Ensure that the Tello camera calibration file named `telloCam.yaml` exists in the `params/` directory.

### Guide

1. Run:
    ```bash
    cd ground_truth_generation
    ./bash/droneLog.sh <save-dir> <bag-file>.bag
    ```
2. When the output image of `telloGTGeneration.py` is shown, hit the `A` key the first time you see a marker in the robot's camera view. This position will be considered the initial point of the drone pose. Note that the `A` key must be hit when a marker is detected. You can see the poses obtained from markers (red) along with the drone odometry plot (blue).

### Output

- `/odomPoses.csv`
- `/rawMarkerPoses.csv`
- All the raw images which are visible in `/rawImage`
- All the drone images in which the markers are visible in `/markerDetectionFrames`

## Step 2: Remove Outliers from Marker Pose Data

1. After performing the instructions to save drone pose data, in the root directory, run:
    ```bash
    ./bash/removeMarkerOutliers.sh <save-dir>
    ```
2. The frames in which pose data is extracted from detected markers will be shown. Based on the appearance of the 3-axes, where they don't make sense, press `d`. Otherwise, press any key until the images are finished.

### Output

- `/cleanMarkerPoses.csv`

## Step 3: Synchronize `odomPoses` and `rawImage`

1. Open `odomPoses.csv` and check the time of the first row of this CSV file.
2. Open the `rawImage` folder. The names of the images indicate the time of that frame. Delete all frames where the time number is less than the time number of the first row of `odomPoses.csv`. The number of images in the `rawImage` folder should now match the number of rows in `odomPoses.csv`.

## Step 4: Clean Outliers of `odomPoses`

1. Run `removeOdomOutliers.py`:
    ```bash
    cd src
    python3 removeOdomOutliers.py -p <The Path to the Bag File>
    ```
2. First, comment these lines from the last part to visualize the 3D plot:
    ```python
    # c = 1
    # for i in z:
    #     if i > 40:
    #         print(c)
    #     c += 1
    # plot x y z
    ```
3. Check that the outlier’s z-axis values are greater than a specific number (call it “c”).
4. Change this line in the code (e.g., `c = 40`). Then uncomment this part:
    ```python
    c = 40
    for i in z:
        if i > 40:
            print(c)
        c += 1
    plot x y z
    ```
    and comment out `plot_3d(x,y,z,t)`.
5. Run again:
    ```bash
    python3 removeOdomOutliers.py -p <The Path to the Bag File>
    ```
6. The script will print the values of row numbers related to outliers. Delete the corresponding rows from `odomPoses.csv` and the related images from `/rawImage` (check them by referencing the time column of the CSV file which shows the frame name in the `/rawImage` folder).

## Step 5: Rename Tello Images

1. Run:
    ```bash
    python3 renameTelloImage.py -p <The Path to the Bag File>
    ```

### Output

- `/cleanImage`

## Step 6: Interpolate Missing Data

1. Before running `getInterpolate.py`, copy `odomPoses.csv` and name it `zodomposes.csv`.
2. Run:
    ```bash
    python3 getInterpolate.py -p <The Path to the Bag File>
    ```
3. Copy the output of the terminal and paste it in the “D” column of `zodomposes.csv`.

## Step 7: Optimize the Drone's Path Using Aruco Markers' Data

Using a gradient descent approach, this feature corrects the drifted drone odometry poses to match the reliable Aruco pose data as much as possible. **No ROS Requirement**

### Finding the Optimal Parameters for Drone Odometry Data Correction

1. `<save-dir>` is the directory containing two files: `odomPoses.csv` and `markerPoses.csv`.
2. In the `dronePoseDataOptimization.py` module, in `__main__`, uncomment the call to `gradientDescentOptimize()` and comment out the `visualize()` function.
3. Run:
    ```bash
    python3 dronePoseDataOptimization.py -p <save-dir>
    ```

### Output

- The log for the optimization in the terminal. Parameters are printed in each optimization step and the convergence procedure can be monitored. When desired (when the change in parameters is ignorable), kill the program and save the last printed parameters.

### Correcting the Odometry Data Using the Optimal Parameters

1. In the `init()` function of the class `OptimizeDronePoseData()`, set the optimal parameter values.
2. At the end of the `dronePoseDataOptimization.py`, comment the call to `gradientDescentOptimize()` and uncomment the `visualize()` function.
3. Run:
    ```bash
    python3 dronePoseDataOptimization.py -p <save-dir>
    ```

### Output

- `correctedPose.csv` (The corrected odometry poses)
- A plot showing the raw odometry poses, corrected odometry poses, and the Aruco marker poses (as scattered points)

## Step 8: YOLO Detection

1. Run:
    ```bash
    python3 yoloDetection.py -p <save-dir>
    ```

### Output

- `YOLOoutput.txt`

## Step 9: Remove YOLO Outliers

1. In the root directory, run:
    ```bash
    ./bash/removeYoloOutliers.sh <save-dir>
    ```
2. For the YOLO output frames, press `o` for frames that have been detected wrongly or have not been detected. Then draw a correct bounding box manually and after that press space/enter. Also, press `n` for frames that detect a person but there is no subject in that frame. Otherwise, press any key until the images are finished, or press `q` to kill the program.

### Output

- `/cleanMarkerPoses.csv`
