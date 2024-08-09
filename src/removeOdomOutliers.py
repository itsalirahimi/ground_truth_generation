import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The Path to the Bag File.', dest='path')
args, unknown = parser.parse_known_args()


def plot_3d(x,y,z,t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)
    ax.cla()
    ax.plot(x,y,z, color='blue', linewidth=2)
    plt.draw()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

dataFrame = pd.read_csv(args.path+'/odomPoses.csv', sep=',', header=None)

x = np.array(dataFrame.values[:,1])
y = np.array(dataFrame.values[:,2])
z = np.array(dataFrame.values[:,3])
t = np.array(dataFrame.values[:,0])




# to find outliers z aixs row number



# c = 1
# for i in z:
#     if i > 20:
#         print (c)
#     c += 1

plot_3d(x,y,z,t)