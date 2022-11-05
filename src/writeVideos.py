import writeDoubleVideoFrames as wdvf

videoFile_side = '/home/narin/Videos/Sideview/20220310_161504.mp4'
videoFile_front = '/home/narin/Videos/Frontview/VID_20220310_161511.mp4'

# Step 1:
#wdvf.writeDoubleVideoFrames(videoFile_front, videoFile_side)

# Step 2:
init_f = 31500
init_s = 31500
wdvf.writeDoubleVideoFrames(videoFile_front, videoFile_side, 
    initFrame_f=init_f, initFrame_s=init_s, write=True)

# Step 3:
# init_f = 2041
# init_s = 2412
# wdvf.writeDoubleVideoFrames(videoFile_front, videoFile_side, 
#     initFrame_f=init_f, initFrame_s=init_s, write=True)
