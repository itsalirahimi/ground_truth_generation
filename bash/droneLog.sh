#!/bin/bash

killall roscore
killall python3
killall python
killall rosbag

# export savDr=/home/hamid/d/NEW/tcs-9-3/data/tello_test/2022-03-10/16-16-18-clean-raw-odom
# export bagFile=2022-03-10-16-16-18-clean-raw-odom.bag

export savDr=$1
export bagFile=$2

export repoRoot=`git rev-parse --show-toplevel`
export PATH=~/anaconda3/bin:$PATH
source ~/anaconda3/etc/profile.d/conda.sh
source /opt/ros/noetic/setup.bash
roscore &
conda activate gtg
set -e
export LD_LIBRARY_PATH=/opt/ros/noetic/lib/:${HOME}/anaconda3/envs/gtg/bin/python
python ${repoRoot}/src/telloGTGeneration.py -p `echo ${savDr}` -b `echo ${bagFile}` &
conda deactivate
cd `echo ${savDr}`
# source /opt/ros/noetic/setup.bash
# rosbag play `echo ${bagFile}` &
conda activate gtg
python ${repoRoot}/src/plotOdom.py -p `echo ${savDr}`