#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
import pandas as pd
import os
import math
from pathlib import Path
import subprocess
import sys
# import keyboard

import argparse
parser = argparse.ArgumentParser()
############ The Path to the Bag File?? 
parser.add_argument('-p', '--path', help='The Path to the Bag File.', dest='path')
args, unknown = parser.parse_known_args()

rootOfRepo = subprocess.getoutput("git rev-parse --show-toplevel")

sys.path.insert(1, args.path)

class CheckingObjectPoses:
    def __init__(self, saveAddress, checkObjectPoseMode=False):
        self._FramesSideDir = saveAddress + "/frames_side"
        self._FramesFrontDir = saveAddress + "/frames_front"
        self._checkedObjectPoses = saveAddress + "/checkedObjectPoses.csv"
        self._ObjectPosesFileName = saveAddress + "/zodomPoses.csv"
        self.x_front = 0
        self.y_front = 0
        self.x_side = 0
        self.y_side = 0

        if not checkObjectPoseMode:
            self.removeOldLogs()

    def removeOldLogs(self):
        if os.path.exists(self._checkedObjectPoses):
            os.remove(self._checkedObjectPoses)
    
    def Capture_Event_Front(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"(x_front, y_front)=({x}, {y})")
            self.x_front = x
            self.y_front = y
    
    def Capture_Event_Side(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"(x_side, y_side)=({x}, {y})")
            self.x_side = x
            self.y_side = y

    def CheckObjectPose(self):
        print("Checking Object Poses Started.")
        objectPose_df = pd.read_csv(self._ObjectPosesFileName, header=None)
        objectPose_df.info()
        FrontImages = sorted(Path(self._FramesFrontDir).iterdir(), key=os.path.getmtime)
        SideImages = sorted(Path(self._FramesSideDir).iterdir(), key=os.path.getmtime)
        
        for k, (front_img_path, side_img_path) in enumerate(zip(FrontImages, SideImages)):
            print(k)
            front_img = cv.imread(str(front_img_path))
            side_img = cv.imread(str(side_img_path))

            if front_img is None or side_img is None:
                print("Error reading images, skipping to next frame.")
                continue

            cv.imshow("Front image", front_img)
            cv.imshow("Side image", side_img)
            
            key = cv.waitKey()

            if key == ord('q'):
                break
            
            elif key == ord('o'):
                cv.setMouseCallback('Front image', self.Capture_Event_Front)
                cv.setMouseCallback('Side image', self.Capture_Event_Side)
                cv.waitKey(0)
                # print(f"(x_front, y_front)=({self.x_front}, {self.y_front})")
                # print(f"(x_side, y_side)=({self.x_side}, {self.y_side})")

        print("Checking Object Poses Done!")

if __name__ == '__main__' :
	rospy.init_node('checkPoseDetection', anonymous=True)

	cop = CheckingObjectPoses(args.path)

	rospy.spin()