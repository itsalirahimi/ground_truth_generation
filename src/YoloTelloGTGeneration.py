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
parser.add_argument('-p', '--path', help='The Path to the Bag File.', dest='path')
args, unknown = parser.parse_known_args()

rootOfRepo = subprocess.getoutput("git rev-parse --show-toplevel")

sys.path.insert(1, args.path)


ARUCO_DICT = {
	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL
}


class YoloArucoBasedDroneGroundTruthGeneration:
	def __init__(self, saveAddress,  outlierRemovalMode=False):
		#Changed for Yolo Outliers
		self._ImgDir = saveAddress + "/img"
		self._YoloImagesDir = saveAddress + "/Yolo_img"
		self._YoloFileName = saveAddress + "/output.txt"
		self._newYoloFileName = saveAddress + "/cleanYoloPoses.csv"
		
		if not outlierRemovalMode:
			self.removeOldLogs()

	def removeOldLogs(self):
		#Changed for Yolo Outliers
		
		print("----- Removing last saved data from save directory -----")

		if not os.path.exists(self._markerImagesDir):
			os.mkdir(self._markerImagesDir)

		if not os.path.exists(self._TelloImgDir):
			os.mkdir(self._TelloImgDir)

		if not os.path.exists(self._ImgDir):
			os.mkdir(self._ImgDir)
			
		if not os.path.exists(self._YoloImagesDir):
			os.mkdir(self._YoloImagesDir)

		if os.path.exists(self._odomPosesFileName):
			os.remove(self._odomPosesFileName)
		
		if os.path.exists(self._markerPosesFileName):
			os.remove(self._markerPosesFileName)
		
		markerImages = Path(self._markerImagesDir).iterdir()
		for mi in markerImages:
			fileName = str(mi)
			os.remove(fileName)


	def deleteYoloOutliers(self):
		#Changed for Yolo Outliers
		print("Checking Yolo Outliers Started.")
		yolo_df = pd.read_csv(self._YoloFileName, sep=",", header=None)
		yolo_df.info()
		YoloImages = sorted(Path(self._YoloImagesDir).iterdir(), key=os.path.getmtime)
		# num = len(YoloImages)
		# print(num)
		# toBeRemoved = []
		for k, mi in enumerate(YoloImages):
			fileName = str(mi)
			print(fileName)
			print(os.path.isfile(fileName))
			img = cv.imread(fileName)
			cv.imshow("Yolo image", img)
			key = cv.waitKey()

			if key == ord('q'):
				cv.destroyWindow("yolo image")
				break

			elif key == ord('o'):
				x1, y1, w, h = cv.selectROI("Select the ROI", img, fromCenter=False, showCrosshair=False)
				x2 = x1 + w
				x3 = x1 + w
				x4 = x1
				y2 = y1
				y3 = y1 + h
				y4 = y1 + h
				new_row_values = [x1, y1, x2, y2, x3, y3, x4, y4]
				# k or k+1?
				yolo_df.iloc[k] = new_row_values

			elif key == ord('n'):
				new_row_values = [-1, -1, -1, -1, -1, -1, -1, -1]
				# k or k+1?
				yolo_df.iloc[k] = new_row_values

		# print(toBeRemoved)
		# df.drop(df.index[toBeRemoved], inplace=True)
		yolo_df.info()
		yolo_df.to_csv(self._newYoloFileName, index=False, header=False)
		print("Checking Yolo Outliers Done!")

if __name__ == '__main__' :

	rospy.init_node('telloGTGeneration', anonymous=True)

	agtg = YoloArucoBasedDroneGroundTruthGeneration(args.path)

	rospy.spin()