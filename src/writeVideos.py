import writeDoubleVideoFrames as wdvf

videoFile_front = '/media/hamid/Data/NEW/tcs-9-3/data/tello_test/2022-03-10/VID_20220310_161511.mp4'
videoFile_side = '/media/hamid/Data/NEW/tcs-9-3/data/tello_test/2022-03-10/20220310_161504.mp4'

# Step 1:
# wdvf.writeDoubleVideoFrames(videoFile_front, videoFile_side)

# Step 2:
# init_f = 2000
# init_s = 2300
# wdvf.writeDoubleVideoFrames(videoFile_front, videoFile_side, 
#     initFrame_f=init_f, initFrame_s=init_s, write=True)

# Step 3:
init_f = 2041
init_s = 2412
wdvf.writeDoubleVideoFrames(videoFile_front, videoFile_side, 
    initFrame_f=init_f, initFrame_s=init_s, write=True)
