import writeDoubleVideoFrames as wdvf

# videoFile_side = '/home/narin/Videos/Sideview/20220310_161504.mp4'
# videoFile_front = '/home/narin/Videos/Frontview/VID_20220310_161511.mp4'

videoFile_side = '/media/ali/Mypassport/VIOT-2/hvideo/20220310_161504.mp4'
videoFile_front = '/media/ali/Mypassport/VIOT-2/hvideo/VID_20220310_161511.mp4'

# Step 1:
# wdvf.writeDoubleVideoFrames(videoFile_front, videoFile_side)

# Step 2:
# init_f = 31500
# init_s = 31500
# init_f = 9000
# init_s = 9400
# wdvf.writeDoubleVideoFrames(videoFile_front, videoFile_side, 
#     initFrame_f=init_f, initFrame_s=init_s, write=True)

# Step 3:
# init_f = 2041
# init_s = 2412
init_f = 9366
init_s = 9954
wdvf.writeDoubleVideoFrames(videoFile_front, videoFile_side, 
    initFrame_f=init_f, initFrame_s=init_s, write=True)
