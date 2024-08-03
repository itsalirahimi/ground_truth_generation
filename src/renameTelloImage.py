import os
import pandas as pd
import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The Path to the Bag File.', dest='path')
args, unknown = parser.parse_known_args()



path_in = args.path + "/rawImage/"
path_out = args.path + "/clearImage/"

# Get the list of all files and directories
dir_list = os.listdir(path_in)

if not os.path.exists(path_out):
    os.mkdir(path_out)

# Read CSV
dataFrame = pd.read_csv(args.path + "/odomPoses.csv", sep=',', header=None)
if not dataFrame is None:
    x = np.array(dataFrame.values[:,1])
    y = np.array(dataFrame.values[:,2])
    z = np.array(dataFrame.values[:,3])
    t = np.array(dataFrame.values[:,7])

list_temp = []
c = 1
for i in t:
    new_name = str('%08d' % c) + ".jpg"
    old_name = str(i)[1:]
    list_temp.append([new_name , old_name])
    # print([new_name , old_name])
    c += 1


c = 1
for item in list_temp:
    for img in dir_list:
        if img.find(item[1]) == 0:
            print(c, img ,item[1], item[0])

            # dir_list.remove(img)
            cmd = "cp "+ path_in+img + " " + path_out+item[0]
            os.system(cmd)
            # os.rename(path_in+img , path_out+item[0])
            c += 1
            break

