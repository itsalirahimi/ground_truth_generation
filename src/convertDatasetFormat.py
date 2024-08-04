#!/usr/bin/env python
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The Path to the save directory.', dest='path')
args, unknown = parser.parse_known_args()

trackerInputDir = os.path.join(args.path, "trackerInput")
if not os.path.isdir(trackerInputDir):
    os.makedirs(trackerInputDir)

# Define file paths
corrected_pose = pd.read_csv(os.path.join(args.path, "correctedPose.csv"), sep=',', header=None)
odom_pose = pd.read_csv(os.path.join(args.path, "zodomPoses.csv"), sep=',', header=None)
clean_object_poses = pd.read_csv(os.path.join(args.path, "clean_object_poses.csv"), sep=',', header=None)
ground_truth = pd.read_csv(os.path.join(args.path, "YOLOoutput.txt"), sep=',', header=None)

# Drop the first row of corrected_pose
corrected_pose = corrected_pose.drop(0).reset_index(drop=True)

# Extract the required columns from correctedPose.csv
# 5th, 2nd, 3rd, and 4th columns -> 1st, 2nd, 3rd, and 4th columns in the new file
camera_states = pd.DataFrame()
camera_states[0] = corrected_pose[4]
camera_states[1] = corrected_pose[1]
camera_states[2] = corrected_pose[2]
camera_states[3] = corrected_pose[3]

# Add the 4th, 5th, and 6th columns of odomPoses.csv to the 4th, 5th, and 6th columns of camera_states
camera_states[4] = odom_pose[4]
camera_states[5] = odom_pose[5]
camera_states[6] = odom_pose[6]

target_poses_df = pd.DataFrame()
num_rows = len(camera_states)

# First column starts from 0 and increments by 1/30
target_poses_df[0] = [i/30 for i in range(num_rows)]

# Add the columns from clean_object_poses
target_poses_df[1] = clean_object_poses[0]
target_poses_df[2] = clean_object_poses[1]
target_poses_df[3] = clean_object_poses[2]

# Add the 5th and 6th columns filled with zeros
target_poses_df[4] = 0
target_poses_df[5] = 0

# Save DataFrames to CSV files
target_poses_df.to_csv(os.path.join(trackerInputDir, "target_poses.txt"), header=False, index=False, sep=',')
camera_states.to_csv(os.path.join(trackerInputDir, "camera_states.txt"), header=False, index=False, sep=',')
ground_truth.to_csv(os.path.join(trackerInputDir, "groundtruth.txt"), header=False, index=False, sep=',')