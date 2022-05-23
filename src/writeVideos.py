#!/usr/bin/env python

import cv2 as cv

# videoFile_front = '/media/hamidreza/Local Disk/rosbag/93/mavic_test/22.03.01/test_1.mp4'
# videoFile_side = '/media/hamidreza/Local Disk/rosbag/93/mavic_test/22.03.01/test_1_side.mp4'
# initFrame_f = 0
# initFrame_s = 0

# videoFile_front = '/media/hamidreza/Local Disk/rosbag/93/gt_test/22.02.10/record_double.mp4'
# videoFile_side = '/media/hamidreza/Local Disk/rosbag/93/gt_test/22.02.10/record_double_side.mp4'
# initFrame_f = 371
# initFrame_s = 162

videoFile_front = '/media/hamidreza/Local Disk/rosbag/93/tello_test/22.03.03/VID_20220303_173033.mp4'
videoFile_side = '/media/hamidreza/Local Disk/rosbag/93/tello_test/22.03.03/20220303_174053.mp4'
initFrame_f = 25518
initFrame_s = 7063

_cap_front = cv.VideoCapture(videoFile_front)
_cap_side = cv.VideoCapture(videoFile_side)
_cap_front.set(1,initFrame_f)
_cap_side.set(1,initFrame_s)

ret_f = True
ret_s = True
fr = 0
while ret_f and ret_s:
    fr += 1
    ret_f, frame_f = _cap_front.read()
    # frame_f = cv.resize(frame_f, (800, 550))
    ret_s, frame_s = _cap_side.read()
    # frame_s = cv.resize(frame_s, (800, 550))
    # cv.putText(frame_f, "Frame num: "+str(initFrame_f + fr), (100, 25),
    #     cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    # cv.putText(frame_s, "Frame num: "+str(initFrame_s + fr), (100, 25),
    #     cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    # cv.putText(frame_f, "Synchronized frame num: "+str(fr), (100, 50),
    #     cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    # cv.putText(frame_s, "Synchronized frame num: "+str(fr), (100, 50),
    #     cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv.imshow("front image", frame_f)
    cv.imshow("side image", frame_s)
    cv.imwrite("../frames_side/frame_%d.jpg" %fr , frame_s)
    cv.imwrite("../frames_front/frame_%d.jpg" %fr , frame_f)
    if cv.waitKey(1) == ord('q'):
        break
