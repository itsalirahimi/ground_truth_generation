#!/bin/bash

export savDr=/home/hamid/d/NEW/tcs-9-3/data/tello_test/2022-03-10/16-16-18
export bagFile=2022-03-10-16-16-18.bag

export PATH=~/anaconda3/bin:$PATH
source ~/anaconda3/etc/profile.d/conda.sh
source /opt/ros/noetic/setup.bash
conda activate bdenv
cd `git rev-parse --show-toplevel`/src/
set -e
export LD_LIBRARY_PATH=/opt/ros/noetic/lib/:${HOME}/anaconda3/envs/bdenv/bin/python
python telloGTGeneration.py -p `echo ${savDr}` &
conda deactivate
cd `echo ${savDr}`
source /opt/ros/noetic/setup.bash
rosbag play `echo ${bagFile}` 