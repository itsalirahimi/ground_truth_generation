#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import argparse


from scipy.interpolate import UnivariateSpline
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The Path to the Bag File.', dest='path')
args, unknown = parser.parse_known_args()

class PoseData:

    def __init__(self, dataFrame = None, xs = None, ys = None, zs = None, ts = None):

        if not dataFrame is None:
            self.x = np.array(dataFrame.values[:,1])
            self.y = np.array(dataFrame.values[:,2])
            self.z = np.array(dataFrame.values[:,3])
            self.t = np.array(dataFrame.values[:,0])

        else:
            self.x = xs[0]
            self.y = ys[0]
            self.z = zs[0]
            self.t = ts


def cleaner(path):
    actualData = path + "/odomPoses.csv" 
    trueData = path + "/cleanMarkerPoses.csv"
    actualData = PoseData(dataFrame = \
            pd.read_csv(actualData, sep=',', header=None))
    trueData = PoseData(dataFrame = \
            pd.read_csv(trueData, sep=',', header=None))

    dataFrame = pd.read_csv(path + "/odomPoses.csv", sep=',', header=None)
    if not dataFrame is None:
        x = np.array(dataFrame.values[:,1])
        y = np.array(dataFrame.values[:,2])
        z = np.array(dataFrame.values[:,3])
        t = np.array(dataFrame.values[:,0])

    dataFrame = pd.read_csv(path + "/cleanMarkerPoses.csv", sep=',', header=None)
    if not dataFrame is None:
        x2 = np.array(dataFrame.values[:,1])
        y2 = np.array(dataFrame.values[:,2])
        z2 = np.array(dataFrame.values[:,3])
        t2 = np.array(dataFrame.values[:,0])



    list_t = []
    for i in t:
        list_t.append(i)


    list_t2 = []
    for i in t2:
        list_t2.append(i)

    list_x = []
    for i in z2:
        list_x.append(i)

    # interps = np.interp(actualData.t, trueData.t, trueData.z)

    print(actualData.t)
    interp_func = UnivariateSpline(list_t2, list_x)

    interps = interp_func(list_t)

    c= 1
    for i in interps:
        print(i)
        c+=1

if __name__ == '__main__' :

    cleaner(args.path)