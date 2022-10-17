#!/usr/bin/env python

import cv2 as cv
import pandas as pd
# import math
# import yaml
import numpy as np
# import time
from enum import Enum
import os


front_video_name = "VID_20220310_161511.mp4"
side_video_name = "20220310_161504.mp4"
videos_path = "/media/hamid/Data/NEW/tcs-9-3/data/tello_test/2022-03-10/"


class Views(Enum):

	FRONT = 1
	SIDE = 2


class MovingObjectGroundTruthGeneration:

	def __init__(self, path_to_vids, vid_f, vid_s):

		# try:
		# 	os.remove("objectPoses.csv")
		# except:
		# 	pass

		self._vids_address = path_to_vids
		videoFile_front = path_to_vids + vid_f 
		videoFile_side = path_to_vids + vid_s

		self._ROI_front = (6, 451, 1852, 185)
		self._fieldCorners_front = [[584, 64], [1377, 75], [1846, 182], [39, 146]]
		self._ROI_side = (1, 343, 1272, 123)
		self._fieldCorners_side = [[11, 116], [478, 32], [973, 25], [1263, 119]]
		self._fieldWidth = 24.2
		self._fieldLength = 32
		initFrame_f = 2041
		initFrame_s = 2412


		# TEST PARAMS:
		# self._ROI_front = (39, 564, 1850, 156)
		# self._fieldCorners_front = [[539, 41], [1347, 47], [1834, 148], [17, 133]]
		# self._ROI_side = (1, 374, 1279, 117)
		# self._fieldCorners_side = [[11, 108], [491, 21], [984, 15], [1276, 107]]
		# self._fieldWidth = 31.5
		# self._fieldLength = 23.4
		# initFrame_f = 25518
		# initFrame_s = 7063

		# self._ROI_front = (581, 601, 1287, 125)
		# self._fieldCorners_front = [[238, 35], [805, 46], [1266, 120], [9, 98]]
		# self._ROI_side = (2, 503, 1918, 170)
		# self._fieldCorners_side = [[17, 107], [677, 49], [1236, 57], [1914, 157]]
		# self._fieldWidth = 30.0
		# self._fieldLength = 36.0
		# initFrame_f = 0
		# initFrame_s = 0

		# CAMERA PARAMS:
		self._intrinsics_front = None
		self._distortion_front = None
		self._mapx_front = None
		self._mapy_front = None
		self._intrinsics_side = None
		self._distortion_side = None
		self._mapx_side = None
		self._mapy_side = None
		self._isRectificationInitialized = False
		# self._intrinsics_side = np.float32([ [843.930250, 0.000000, 950.421246],
		# 									[0.000000, 861.669778, 577.905765],
		# 									[0.000000, 0.000000, 1.000000]])
		# self._distortion_side = np.float32([ [-0.067901, -0.000153, -0.024577, -0.001551, 0.000000]])

		# VIDEO PLAYBACK PARAMS:
		self._go = True
		self._delay = 50
		self._it = 0
		self._cap_front = cv.VideoCapture(videoFile_front)
		self._cap_front.set(1,initFrame_f)
		self._cap_side = cv.VideoCapture(videoFile_side)
		self._cap_side.set(1,initFrame_s)
		self._fps = 30
		self._time = 0

		# UI PARAMS:
		self._presetUI_front = "Front Frame"
		self._presetUI_side = "Side Frame"
		self._resultUI_front = "Front Result"
		self._resultUI_side = "Side Result"
		self._resultUI = "Final Result"
		self._presetUIDestroyed_front = False
		self._presetUIDestroyed_side = False
		cv.namedWindow(self._presetUI_front)
		cv.namedWindow(self._presetUI_side)
		cv.setMouseCallback(self._presetUI_front, self.mouseCallback_front)
		cv.setMouseCallback(self._presetUI_side, self.mouseCallback_side)
		print("MovingObjectGroundTruthGeneration GUIDE:" + \
			"\n \t a: quit" + \
			"\n \t s: crop ROI for front view" + \
			"\n \t d: crop ROI for side view" + \
			"\n \t f: start processing")

		# HOMOGRAPHIC TRANSFORM MEMBERS:
		self._homo_front = None
		self._homo_side = None
		self._dstPts = np.array([[0, 0],[959, 0],[959, 719],[0, 719]])
		self._pixelMapSize = (960, 720)

		# MOTION DETECTION MEMBERS:
		self._bgs_front = cv.createBackgroundSubtractorMOG2(varThreshold = 32)
		self._bgs_side = cv.createBackgroundSubtractorMOG2(varThreshold = 32)

		# MOTION MODEL PARAMS:
		self._last_front_x = None
		self._last_front_vx = None
		self._last_side_x = None
		self._last_side_vx = None
		self._maxPixelMovement = 50
		self._motionModelStartFrame = 30


	def initRectification(self, image):

		if self._isRectificationInitialized:
			return

		size = (image.shape[1], image.shape[0])
		R = np.eye(3, dtype=np.float64)
		if not self._intrinsics_side is None and not self._distortion_side is None:
			ncm, _ = cv.getOptimalNewCameraMatrix(self._intrinsics_side,
				self._distortion_side, size, 0.0)
			self._mapx_side, self._mapy_side = \
				cv.initUndistortRectifyMap(self._intrinsics_side,
					self._distortion_side, R, ncm, size, cv.CV_32FC1)

		if not self._intrinsics_front is None and not self._distortion_front is None:
			ncm, _ = cv.getOptimalNewCameraMatrix(self._intrinsics_front,
				self._distortion_front, size, 0.0)
			self._mapx_front, self._mapy_front = \
				cv.initUndistortRectifyMap(self._intrinsics_front,
					self._distortion_front, R, ncm, size, cv.CV_32FC1)

		self._isRectificationInitialized = True


	def rectify(self, frame, view):

		if not self._isRectificationInitialized:
			self.initRectification(frame)

		if view == Views.FRONT:

			intrinsics = self._intrinsics_front
			distortion = self._distortion_front
			mx = self._mapx_front
			my = self._mapy_front

		elif view == Views.SIDE:
			
			intrinsics = self._intrinsics_side
			distortion = self._distortion_side
			mx = self._mapx_side
			my = self._mapy_side

		if not mx is None and not my is None:
			return cv.remap(frame, mx, my, cv.INTER_LINEAR)
		else:
			return frame


	def crop(self, front, side):

		if not self._ROI_front is None:
			image_f = front.copy()[int(self._ROI_front[1]):int(self._ROI_front[1]+self._ROI_front[3]),
				int(self._ROI_front[0]):int(self._ROI_front[0]+self._ROI_front[2])]
		else:
			image_f = front.copy()

		if not self._ROI_side is None:
			image_s = side.copy()[int(self._ROI_side[1]):int(self._ROI_side[1]+self._ROI_side[3]),
				int(self._ROI_side[0]):int(self._ROI_side[0]+self._ROI_side[2])]
		else:
			image_s = side.copy()

		return image_f, image_s


	def process(self, frame_f, frame_s):

		frame_f = self.rectify(frame_f, Views.FRONT)
		frame_s = self.rectify(frame_s, Views.SIDE)

		image_f, image_s = self.crop(frame_f, frame_s)
		self.switch(frame_f, frame_s)

		motion_f, m_f = self.detectMotion(image_f, Views.FRONT)
		motion_s, m_s = self.detectMotion(image_s, Views.SIDE)

		transformedView_f, transformedView_s = self.transformViews(image_f, image_s)
		transformedPath_f, transformedPath_s = self.transformViews(motion_f, motion_s)

		result, vmm_front, vmm_side = self.getUnifiedMap(transformedPath_f,
			transformedPath_s, transformedView_f, transformedView_s)

		self.visualize(image_f, image_s, transformedPath_f, transformedPath_s,
			m_f, m_s, result, vmm_front, vmm_side)

		x, y = self.rescaleToMetric(result)
		self.savePath(x, y)


	def getUnifiedMap(self, verLineImg, horLineImg, frontMap, sideMap):

		result = None
		visualizedMotionMap_front = None
		visualizedMotionMap_side = None
		if not verLineImg is None and not horLineImg is None:
			result = cv.bitwise_and(verLineImg, horLineImg, mask = None)
			res = np.zeros((self._pixelMapSize[1], self._pixelMapSize[0], 3),
				dtype=np.uint8)

			res[:,:,0] = result
			res[:,:,1] = result
			res[:,:,2] = result
			visualizedMotionMap_front = cv.bitwise_or(frontMap,	res, mask = None)
			visualizedMotionMap_side = cv.bitwise_or(sideMap, res, mask = None)

		return result, visualizedMotionMap_front, visualizedMotionMap_side


	def rescaleToMetric(self, map):

		box = self.getMainWhiteSegment(map)
		if box is None:
			return None, None

		out_x = (box[0]+box[2]/2) * (self._fieldWidth / self._pixelMapSize[0])
		out_y = self._fieldLength - \
			(box[1]+box[3]/2) * (self._fieldLength / self._pixelMapSize[1])
		# out_y = (box[1]+box[3]/2) * (self._fieldLength / self._pixelMapSize[1]) - self._fieldLength
		return out_x, out_y


	def visualize(self, frontView, sideView, frontResult, sideResult, fMotion,
		sMotion, result, mmap_front, mmap_side):

		if not self._presetUIDestroyed_front:
			for p in self._fieldCorners_front:
				cv.circle(frontView, (p[0],p[1]), 5, (0,0,255), -1)
		cv.imshow(self._presetUI_front, frontView)

		if not self._presetUIDestroyed_side:
			for p in self._fieldCorners_side:
				cv.circle(sideView, (p[0],p[1]), 5, (0,0,255), -1)
			cv.imshow(self._presetUI_side, sideView)

		if not frontResult is None:
			cv.imshow(self._resultUI_front, frontResult)

		if not sideResult is None:
			cv.imshow(self._resultUI_side, sideResult)

		if not result is None:
			cv.imshow("Final Result", result)
			cv.imshow("Final Result in Front View Map", mmap_front)
			cv.imshow("Final Result in Side View Map", mmap_side)


	def switch(self, front, side):

		self._key = cv.waitKey(self._delay)

		if self._key == ord('a'):
			self._go = False

		elif self._key == ord('s'):
			self._ROI_front = cv.selectROI(front)
			print('Selected ROI: ', self._ROI_front)

		elif self._key == ord('d'):
			self._ROI_side = cv.selectROI(side)
			print('Selected ROI: ', self._ROI_side)

		elif self._key == ord('f'):
			self.calcTransforms()


	def detectMotion(self, image, view):

		shape = image.shape
		if view == Views.FRONT:
			fgMask = self._bgs_front.apply(image)
		elif view == Views.SIDE:
			fgMask = self._bgs_side.apply(image)

		imageThresholded = cv.erode(fgMask, np.ones((3,3),np.uint8))
		imageThresholded = cv.dilate(imageThresholded, np.ones((6,6),np.uint8))

		motionImg = self.detectSingleMovingObject(imageThresholded, view)

		return motionImg, imageThresholded


	def detectSingleMovingObject(self, image, view):

		shape = image.shape
		motionImg = np.zeros((shape[0],shape[1],1), dtype=np.uint8)
		y1 = 0
		y2 = shape[0]
		bbox = self.getMainWhiteSegment(image)

		if bbox is None:
			return motionImg

		if view == Views.FRONT:
			lx = self._last_front_x
			lvx = self._last_front_vx
		elif view == Views.SIDE:
			lx = self._last_side_x
			lvx = self._last_side_vx

		x_ = int(bbox[0]+bbox[2]/2)

		# Specially for the first and second iteration, to fill the global cases
		# of lx and lvx:
		flag1 = not lx is None
		flag2 = not lvx is None

		# Displacements bigger than _maxPixelMovement are detected as noise and
		# a linear model for object movement is used instead:
		flag3 = flag1 and abs(lx - x_) > self._maxPixelMovement

		# The aforementioned correction is often useful when a certain number of
		# frames are evaluated after beginning of the code execution
		flag4 = self._it > self._motionModelStartFrame

		if flag1 and flag2 and flag3 and flag4:
			linearEstimatedX = lx + lvx
			if lvx >= 0:
				x = min(shape[1], linearEstimatedX)
			else:
				x = max(0, linearEstimatedX)
		else:
			x = x_
			if flag1:
				lvx = x - lx

		cv.circle(image, (x, bbox[1]+bbox[3]), 1, (255,255,255), -1)
		cv.line(motionImg, (x, y1), (x, y2), 255, thickness=2)
		lx = x

		if view == Views.FRONT:
			self._last_front_x = lx
			self._last_front_vx = lvx
		elif view == Views.SIDE:
			self._last_side_x = lx
			self._last_side_vx = lvx

		return motionImg


	def getMainWhiteSegment(self, image):

		contours, hierarchy = cv.findContours(image, cv.RETR_CCOMP,
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
		return bbox


	def mouseCallback_front(self, event, x, y, flags, param):

		if event == cv.EVENT_LBUTTONUP:
			if len(self._fieldCorners_front) >= 4:
				self._fieldCorners_front = []
			self._fieldCorners_front.append([x,y])
			print(self._fieldCorners_front)


	def mouseCallback_side(self, event, x, y, flags, param):

		if event == cv.EVENT_LBUTTONUP:
			if len(self._fieldCorners_side) >= 4:
				self._fieldCorners_side = []
			self._fieldCorners_side.append([x,y])
			print(self._fieldCorners_side)


	def runVideos(self):

		ret_s = True
		ret_f = True
		while ret_s and ret_f and self._go:
			self._it += 1
			ret_f, frame_f = self._cap_front.read()
			ret_s, frame_s = self._cap_side.read()
			self._time += 1.0/self._fps
			self.process(frame_f, frame_s)


	def transformViews(self, image_f, image_s):

		if (self._homo_front is None) or (self._homo_side is None):
			return None, None

		if not self._presetUIDestroyed_front:
			cv.destroyWindow(self._presetUI_front)
			self._presetUIDestroyed_front = True

		if not self._presetUIDestroyed_side:
			cv.destroyWindow(self._presetUI_side)
			self._presetUIDestroyed_side = True

		out_f = cv.warpPerspective(image_f, self._homo_front, self._pixelMapSize)
		out_s = cv.warpPerspective(image_s, self._homo_side, self._pixelMapSize)
		return out_f, out_s


	def calcTransforms(self):

		if (len(self._fieldCorners_front) < 4) or (len(self._fieldCorners_side) < 4):
			return

		self._homo_front, status = cv.findHomography(np.array(self._fieldCorners_front),
			self._dstPts)

		self._homo_side, status = cv.findHomography(np.array(self._fieldCorners_side),
			self._dstPts)


	def savePath(self, x, y):

		if not x is None and not y is None:
			# print(x,y,self._time)
			df_marker = pd.DataFrame({'Xs':[x], 'Ys':[y], 'Time':[self._time]})
			df_marker.to_csv(self._vids_address+"objectPoses.csv", mode='a', index=False, header=False)
	

	# def extractFileAddress(self, path):
		
	# 	flag = True
	# 	k = len(path)
	# 	end = k-1
		
	# 	for char in path:
	# 		k -= 1
			
	# 		if path[k] == '/':
	# 			end = k+1
	# 			break
		
	# 	return path[0:end]




if __name__ == '__main__' :
	mogtg = MovingObjectGroundTruthGeneration(videos_path, front_video_name, side_video_name)
	# mogtg = MovingObjectGroundTruthGeneration('/media/hamidreza/Local Disk/rosbag/93/mavic_test/22.03.01/test_1.mp4',
	# 	'/media/hamidreza/Local Disk/rosbag/93/mavic_test/22.03.01/test_1_side.mp4')
	mogtg.runVideos()
