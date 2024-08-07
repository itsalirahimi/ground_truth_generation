#!/bin/bash

# export savDr=/home/hamid/d/NEW/tcs-9-3/data/tello_test/2022-03-10/16-16-18
export savDr=$1

killall python3
killall python

export repoRoot=`git rev-parse --show-toplevel`
export PATH=~/anaconda3/bin:$PATH
source ~/anaconda3/etc/profile.d/conda.sh
source /opt/ros/noetic/setup.bash
conda activate gtg
set -e
python ${repoRoot}/src/removePoseOutliers.py -p `echo ${savDr}`