#!/usr/bin/env python

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

odom_gt_dt = 5.94
object_dx = -8.9
object_dy = -0.5

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


while True:

    odoms_df = pd.read_csv('odomPoses.csv', sep=',', header=None)
    markers_df = pd.read_csv('markerPoses.csv', sep=',', header=None)
    object_df = pd.read_csv('objectPoses.csv', sep=',', header=None)
    ax.cla()
    xs_o = odoms_df.values[:,0]
    ys_o = odoms_df.values[:,1]
    zs_o = odoms_df.values[:,2]
    t_odom = odoms_df.values[len(zs_o)-1,3] - odoms_df.values[0,3]
    print(t_odom)

    xs_m = markers_df.values[:,0]
    ys_m = markers_df.values[:,1]
    zs_m = markers_df.values[:,2]

    kk = 0
    for k, t_object in enumerate(object_df.values[:,2]):
        if t_object > t_odom-odom_gt_dt:
            kk = k
            break

    xs_ob = object_df.values[0:kk,0] + object_dx
    ys_ob = object_df.values[0:kk,1] + object_dy

    ax.plot(xs_o, ys_o, zs_o, color='blue', linewidth=2)
    ax.plot(xs_ob, ys_ob, color='green', linewidth=2)
    ax.scatter(xs_m, ys_m, zs_m, color='red', linewidth=0)
    set_axes_equal(ax)
    plt.draw()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.pause(1)
