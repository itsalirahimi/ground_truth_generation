import cv2
import os
print("file exists?", os.path.exists('output.mp4'))

vid = 'output.mp4'
cap = cv2.VideoCapture(vid)
# ngth = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( cap.isOpened() )