import cv2 as cv

def writeDoubleVideoFrames(front, side, write = False, initFrame_f = 0, initFrame_s = 0):

    print("This function writes two input videos frame by frame \n press q to stop")

    _cap_front = cv.VideoCapture(front)
    _cap_side = cv.VideoCapture(side)
    _cap_front.set(1,initFrame_f)
    _cap_side.set(1,initFrame_s)

    ret_f = True
    ret_s = True
    fr = 0
    while ret_f and ret_s:
        
        ret_f, frame_f = _cap_front.read()
        ret_s, frame_s = _cap_side.read()
        
        frame_f = cv.resize(frame_f, (800, 550))
        frame_s = cv.resize(frame_s, (800, 550))
        
        fr += 1
        
        cv.putText(frame_f, "Frame num: "+str(initFrame_f + fr), (100, 25),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv.putText(frame_s, "Frame num: "+str(initFrame_s + fr), (100, 25),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv.putText(frame_f, "Synchronized frame num: "+str(fr), (100, 50),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv.putText(frame_s, "Synchronized frame num: "+str(fr), (100, 50),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv.imshow("front image", frame_f)
        cv.imshow("side image", frame_s)

        if write:
            cv.imwrite("../frames_side/frame_%d.jpg" %fr , frame_s)
            cv.imwrite("../frames_front/frame_%d.jpg" %fr , frame_f)

        if cv.waitKey(1) == ord('q'):
            break
