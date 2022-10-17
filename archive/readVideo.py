import cv2
import os

vid = '/media/hamid/Data/NEW/tcs-9-3/data/tello_test/2022-03-10/VID_20220310_161511.mp4'

print("file exists?", os.path.exists(vid))

cap = cv2.VideoCapture(vid)
# ngth = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( cap.isOpened() )

while cap.isOpened():

    _, frame = cap.read()

    cv2.imshow("image", frame)

    cv2.waitKey(30)