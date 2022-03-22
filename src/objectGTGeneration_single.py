#!/usr/bin/env python

import cv2 as cv
import pandas as pd
import math
import argparse
import yaml
import numpy as np
import time
from enum import Enum
import colorDetection as cldt


parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()



class Modes(Enum):
	SET = 1
	GET = 2
	DONE = 3

# a = 0.0
# size = (image.shape[1], image.shape[0])
# ncm, _ = cv2.getOptimalNewCameraMatrix(self.intrinsics, self.distortion, self.size, a)
# self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.intrinsics, self.distortion, self.R, ncm, self.size, cv2.CV_32FC1)
# return cv2.remap(src, self.mapx, self.mapy, cv2.INTER_LINEAR)
# R = numpy.eye(3, dtype=numpy.float64)

class MovingObjectGroundTruthGeneration:

	# def __init__(self, videoFile_lat, videoFile_lon):
	def __init__(self, videoFile):

		self._cap = cv.VideoCapture(videoFile)
		# self._cap_lat = cv.VideoCapture(videoFile_lat)
		# self._cap_lon = cv.VideoCapture(videoFile_lon)
		self._ct = cldt.ColorThreshold()
		self._autoMode = Modes.DONE

		self._ROI = None
		# self._fieldCorners = []

		# name: 20220210_175208.mp4
		# self._ROI = (25, 333, 1719, 247)
		# self._fieldCorners = [[39, 237], [627, 69], [1216, 54], [1708, 221]]

		# name: record_double.mp4
		# self._ROI = (459, 441, 884, 83)
		# self._fieldCorners = [[164, 45], [700, 40], [836, 61], [25, 73]]


		# self._ROI = (12, 467, 1732, 122)
		# self._fieldCorners = [[16, 80], [591, 55], [1153, 63], [1730, 116]]

		# name: test_1.mp4
		# self._ROI = (519, 529, 1187, 135)
		# self._fieldCorners = [[224, 49], [740, 59], [1158, 125], [16, 105]]
		# self._ROI = (13, 472, 1728, 277)
		# self._fieldCorners = []

		# name: test_1_side.mp4
		# self._ROI = (16, 399, 1728, 215)
		self._fieldCorners = []
		# self._fieldCorners = [[410, 200], [715, 135], [1140, 138], [1572, 198]]

		# name: test_1_side.mp4 / rectified
		# self._ROI = (6, 388, 1738, 230)
		# self._fieldCorners = [[13, 168], [612, 113], [1119, 121], [1736, 208]]
		# self._fieldCorners = [[410, 200], [715, 135], [1140, 138], [1572, 198]]


		# self._fieldCorners = []
		# self._ROI_lat = (19, 340, 1725, 237)
		# self._fieldCorners_lat = [[632, 61], [1221, 46], [1711, 213], [45, 228]]
		# self._fieldCorners = [[529, 47], [992, 39], [1310, 66], [6, 73]]
		# self._ROI = (31, 530, 1331, 79)
		# self._fieldCorners_lon = [[263, 35], [798, 28], [1049, 69], [16, 82]]
		# self._ROI = (476, 444, 857, 79)

		self._delay = 50
		self._presetUI = "Frame"
		self._resultUI = "Result"
		self._homo = None
		# self._dstPts_lon = np.array([[1, 1],[959, 1],[959, 719],[1, 719]])
		# self._dstPts_lat = np.array([[959, 1],[959, 719],[1, 719],[1, 1]])
		self._dstPts = np.array([[1, 1],[959, 1],[959, 719],[1, 719]])
		# self._dstPts = np.array([[959, 1],[959, 719],[1, 719],[1, 1]])
		self._outSize = (960, 2200)
		self._presetUIDestroyed = False
		self._bgs = cv.createBackgroundSubtractorMOG2(varThreshold = 32)
		self._go = True
		cv.namedWindow(self._presetUI)
		cv.setMouseCallback(self._presetUI, self.mCallback)

		# fs = cv.FileStorage('../params/xiaomiParams.yaml', cv.FILE_STORAGE_READ)
		# self.intrinsics_side = fs.getNode("camera_matrix").mat()
		# self.distortion_side = fs.getNode("distortion_coefficients").mat()
		# print(self.intrinsics_side, self.distortion_side)

		self.intrinsics_side = np.float32([ [843.930250, 0.000000, 950.421246],
											[0.000000, 861.669778, 577.905765],
											[0.000000, 0.000000, 1.000000]])
		self.distortion_side = np.float32([ [-0.067901, -0.000153, -0.024577, -0.001551, 0.000000]])

		print("MovingObjectGroundTruthGeneration GUIDE:" + \
			"\n \t a: quit" + \
			"\n \t s: clean corners" + \
			"\n \t d: crop ROI" + \
			"\n \t f: tune color-based auto marker detection" + \
			"\n \t g: color-based auto marker detection" + \
			"\n \t h: manual marker pose determination (default)" + \
			"\n \t j: start processing")


	def process(self, frame):

		if not self._ROI is None:
			image = frame.copy()[int(self._ROI[1]):int(self._ROI[1]+self._ROI[3]),
				int(self._ROI[0]):int(self._ROI[0]+self._ROI[2])]
		else:
			image = frame.copy()

		self.switch(frame)
		# self._ct.process(image, self._autoMode)
		# mImg = self.detectMotion(image)
		transformedFrame = self.transform(image)
		# transformedFrame = self.transform(mImg)

		if not self._presetUIDestroyed:
			for p in self._fieldCorners:
				cv.circle(image, (p[0],p[1]), 5, (0,0,255), -1)
			cv.imshow(self._presetUI, image)

		if not transformedFrame is None:
			# cv.circle(image, movingPoint, 5, (255,255,255), -1)
			cv.imshow(self._resultUI, transformedFrame)


	def switch(self, image):

		self._key = cv.waitKey(self._delay)

		if self._key == ord('a'):
			self._go = False

		elif self._key == ord('s'):
			if len(self._fieldCorners) >= 1:
				self._fieldCorners = self._fieldCorners[:-1]

		elif self._key == ord('d'):
			self._ROI = cv.selectROI(image)
			print('Selected ROI: ', self._ROI)

		elif self._key == ord('f'):
			self._autoMode = Modes.SET

		elif self._key == ord('g'):
			self._fieldCorners = self._ct.process(image, Modes.GET)

		elif self._key == ord('h'):
			self._autoMode = Modes.DONE

		elif self._key == ord('j'):
			self.calcTransform()


	def detectMotion(self, image):
        # print self._kernelSize
        # imageThresholded = cv.dilate(imageThresholded, np.ones((1,self._kernelSize[1]),np.uint8))
		shape = image.shape
		motionImg = np.zeros((shape[0],shape[1],1), dtype=np.uint8)
		fgMask = self._bgs.apply(image)
		imageThresholded = cv.erode(fgMask, np.ones((2,2),np.uint8))
		imageThresholded = cv.dilate(imageThresholded, np.ones((4,4),np.uint8))
		contours, hierarchy = cv.findContours(imageThresholded, cv.RETR_CCOMP,
			cv.CHAIN_APPROX_TC89_KCOS)
		bbox = None
		biggestArea = -1
		for idx, cnt in enumerate(contours):
			if hierarchy[0][idx][3]!=-1:
				continue

			area = cv.contourArea(cnt)
			if area > biggestArea:
				biggestArea = area
				bbox = cv.boundingRect(cnt)


		if not bbox is None:
			y1 = 0
			y2 = shape[0]
			x1 = x2 = int(bbox[0]+bbox[2]/2)
			cv.circle(image, (int(bbox[0]+bbox[2]/2), bbox[1]+bbox[3]), 1,
				(255,255,255), -1)
			# cv.circle(motionImg, (int(bbox[0]+bbox[2]/2), bbox[1]+bbox[3]), 1,
			# 	(255,255,255), -1)
			cv.line(motionImg, (x1, y1), (x2, y2), 255, thickness=2)

		return motionImg

			# cv.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2],
			# 	bbox[1]+bbox[3]), (0,255,0), 1)

		# cv.imshow("Mask", fgMask)
		# cv.imshow("Filtered", imageThresholded)


	def mCallback(self, event, x, y, flags, param):

		if event == cv.EVENT_LBUTTONUP:
			if len(self._fieldCorners) >= 4:
				self._fieldCorners = []
			self._fieldCorners.append([x,y])
			print(self._fieldCorners)


	def runVideo(self):

		ret = True
		while ret and self._go:
			ret, frame = self._cap.read()

			a = 0.0
			size = (frame.shape[1], frame.shape[0])
			R = np.eye(3, dtype=np.float64)
			ncm, _ = cv.getOptimalNewCameraMatrix(self.intrinsics_side, self.distortion_side, size, a)
			self.mapx, self.mapy = cv.initUndistortRectifyMap(self.intrinsics_side, self.distortion_side, R, ncm, size, cv.CV_32FC1)
			rectifiedFrame =  cv.remap(frame, self.mapx, self.mapy, cv.INTER_LINEAR)

			# frame = cv.resize(frame, (1744, 981))
			rectifiedFrame = cv.resize(rectifiedFrame, (800, 450))
			# rectifiedFrame = cv.resize(rectifiedFrame, (1744, 981))
			frame = cv.resize(frame, (800, 450))
			self.process(rectifiedFrame)
			cv.imshow("orig", frame)


	def transform(self, image):

		if self._homo is None:
			return None

		if not self._presetUIDestroyed:
			cv.destroyWindow(self._presetUI)
			self._presetUIDestroyed = True

		return cv.warpPerspective(image, self._homo, self._outSize)


	def calcTransform(self):

		if len(self._fieldCorners) < 4:
			return

		self._homo, status = cv.findHomography(np.array(self._fieldCorners),
			self._dstPts)



	def savePath(self):
	    # global pathPoints, realWidth, realHeight, measuredRect, f

	    if len(pathPoints) == 0:
	        return

	    print(pathPoints)

	    xs, ys, fs = [], [], []
	    for p in pathPoints:
	        sx = (np.float32(realWidth)/measuredRect[0])*p[0]
	        sy = (np.float32(realHeight)/measuredRect[1])*p[1]
	        sf = p[2]*(1.0/np.float32(fps))
	        xs.append(sx)
	        ys.append(sy)
	        fs.append(sf)

	    dict = {'x': xs, 'y':ys, 'frame':fs}
	    df = pd.DataFrame(dict)
	    fileName = 'data_'+str(int(time.time()))+ '.csv'
	    # df.to_csv('csv_files/'+fileName)
	    df.to_csv(fileName)


if __name__ == '__main__' :

    mogtg = MovingObjectGroundTruthGeneration(args.file)
    mogtg.runVideo()


# rospy.init_node('TelloGTGeneration', anonymous=True)
#
# agtg = ArucoGroundTruthGeneration("../params/telloCam.yaml")
#
# rospy.spin()
