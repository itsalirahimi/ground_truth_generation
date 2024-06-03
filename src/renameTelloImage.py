import os
import pandas as pd
import numpy as np

# Get the list of all files and directories
path = "/home/ali/Log_161618/telloimg/"
path1 = "/home/ali/Log_161618/img/"
dir_list = os.listdir(path)



# Read CSV
dataFrame = pd.read_csv("/home/ali/Log_161618/odomPoses.csv", sep=',', header=None)
if not dataFrame is None:
    x = np.array(dataFrame.values[:,0])
    y = np.array(dataFrame.values[:,1])
    z = np.array(dataFrame.values[:,2])
    t = np.array(dataFrame.values[:,4])

list_temp = []
c = 1
for i in t:
    new_name = str('%08d' % c) + ".jpg"
    old_name = str(i)[1:]
    list_temp.append([new_name , old_name])
    # print([new_name , old_name])
    c += 1


# print(len(list_temp))
# print(len(dir_list))


c = 1
for item in list_temp:
    for img in dir_list:
        if img.find(item[1]) == 0:
            print(c, img ,item[1], item[0])

            # dir_list.remove(img)
            os.rename(path+img , path1+item[0])
            c += 1
            break

