import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# dataFrame = pd.read_csv('~/161618log/odomPoses.csv', sep=',', header=None)
dataFrame = pd.read_csv('~/Log_161618/odomPoses.csv', sep=',', header=None)

x = np.array(dataFrame.values[:,0])
y = np.array(dataFrame.values[:,1])
z = np.array(dataFrame.values[:,2])
t = np.array(dataFrame.values[:,3])




# to find outliers z aixs row number



# c = 1
# for i in z:
#     if i > 40:
#         print (c)
#     c += 1


# plot x y z

plot_3d(x,y,z,t)