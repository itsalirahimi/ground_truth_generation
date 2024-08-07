#!/bin/bash

export savDr=$1

killall python3
killall python

export repoRoot=`git rev-parse --show-toplevel`
export PATH=~/anaconda3/bin:$PATH
source ~/anaconda3/etc/profile.d/conda.sh
source /opt/ros/noetic/setup.bash
conda activate gtg
set -e
python ${repoRoot}/src/removeYoloOutliers.py -p `echo ${savDr}`