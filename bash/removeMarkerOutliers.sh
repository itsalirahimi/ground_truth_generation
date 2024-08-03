#!/bin/bash

# export savDr=/home/hamid/d/NEW/tcs-9-3/data/tello_test/2022-03-10/16-16-18
export savDr=$1
export bagFile=$2

killall roscore
killall python3
killall python
killall rosbag

export repoRoot=`git rev-parse --show-toplevel`
export PATH=~/anaconda3/bin:$PATH
source ~/anaconda3/etc/profile.d/conda.sh
source /opt/ros/noetic/setup.bash
conda activate gtg
set -e
export LD_LIBRARY_PATH=/opt/ros/noetic/lib/:${HOME}/anaconda3/envs/gtg/bin/python
python ${repoRoot}/src/removeMarkerPoseOutliers.py -p `echo ${savDr}` -b `echo ${bagFile}`