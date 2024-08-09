#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The Path to the Bag File.', dest='path')
args, unknown = parser.parse_known_args()


class PoseData:

    def __init__(self, dataFrame = None, xs = None, ys = None, zs = None, ts = None,
                    phi = None, theta = None, psi = None):

        if not dataFrame is None:
            self.x = np.array(dataFrame.values[:,1])
            self.y = np.array(dataFrame.values[:,2])
            self.z = np.array(dataFrame.values[:,3])
            self.t = np.array(dataFrame.values[:,0]) - dataFrame.values[0,0]
        

        else:
            self.x = xs[0]
            self.y = ys[0]
            self.z = zs[0]
            self.phi = phi[0]
            self.theta = theta[0]
            self.psi = psi[0]
            self.t = ts[0]


    def get(self, time):
        # TODO: There must be easier ways to get "x,y,z" by passing "t". Try
        # building a dict member which simply gets "t" and returns "x,y,z"
        minDiff = 1e14
        idx = None
        for k in range(np.shape(self.t)[0]):
            diff = abs(self.t[k]-time)
            if diff < minDiff:
                minDiff = diff
                idx = k

        return np.array([self.x[idx], self.y[idx], self.z[idx]]).reshape(3,1)



class OptimizeDronePoseData:

    def __init__(self, path):

        actualData = path + "/zodomPoses.csv" 
        trueData = path + "/cleanMarkerPoses.csv"

        self.actualData = PoseData(dataFrame = \
            pd.read_csv(actualData, sep=',', header=None))
        self.trueData = PoseData(dataFrame = \
            pd.read_csv(trueData, sep=',', header=None))
        
        #15-41-08: params0 = [ 0.69177261, -0.02609138, -0.04906787, -0.07002913, -0.03610969,  0.01000377, -0.01623883]
        #15-49-35: params0 = [ 0.49311323,  0.00625921,  0.04271879,  0.01796779, -0.01912674,  0.03912622, -0.0073331]
        #16-16-18: params0 = [0.49032486, -0.02135172,  0.0582496,   0.01863012, -0.02056714,  0.04211169, -0.00614571]
        #16-21-44: params0 = [-0.11346958,  0.015804,    0.1516101,   0.05530578, -0.05039234,  0.13310833, 0.01286981]
        #16-24-54: params0 = [-0.10679356, -0.00352302,  0.22186878, 0.092965,   -0.04934356,  0.13654052, 0.01491284]
        #16-33-13: Diverged!!
        #16-41-11: params0 = [0.20681517,  0.01426136,  0.14140154,  0.02775136, -0.02973258,  0.16680787, 0.02879249]
        #16-49-29: Diverged!!
        #17-00-44: params0 = [0.12653776,  0.00187951,  0.05390366,  0.01718401, -0.03400378,  0.16781687, 0.03194619]
        params0 = [0.12653776,  0.00187951,  0.05390366,  0.01718401, -0.03400378,  0.16781687, 0.03194619]

        self.paramsNum = len(params0)
        self.params = np.array(params0).reshape(self.paramsNum,1)
        self.dp = 1e-4
        self.delta = 1e-7
        self.convergencRadius = 1e-7
        self.allowedSteps = 1000

        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.path = path


    def visualize(self):

        self.calcAllNewData()
        self.ax.cla()
        self.ax.plot(self.actualData.x, self.actualData.y, self.actualData.z,
            color='blue', linewidth=2)
        self.ax.plot(self.correctedData.x, self.correctedData.y, self.correctedData.z,
            color='green', linewidth=2)
        self.ax.scatter(self.trueData.x, self.trueData.y, self.trueData.z,
            color='red', linewidth=0)
        self.set_axes_equal()
        plt.draw()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.show()
        dict = {'x': self.correctedData.x, 'y':self.correctedData.y,
            'z':self.correctedData.z, 't':self.correctedData.t}
        df = pd.DataFrame(dict)
        fileName = self.path + '/correctedPose.csv'
        df.to_csv(fileName)

        df = pd.DataFrame({'Time':self.correctedData.t,
                'Xs':self.correctedData.x,
                'Ys':self.correctedData.y,
				'Zs':self.correctedData.z,
				'phi':self.correctedData.phi,
				'theta':self.correctedData.theta,
				'psi':self.correctedData.psi})
        fileName = self.path + '/cleanCorrectedPose.csv'
        df.to_csv(fileName, mode='w', index=False, header=False)


    def set_axes_equal(self):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


    def calcAllNewData(self):

        xs = []
        ys = []
        zs = []
        phi = []
        theta = []
        psi = []
        time = []
        fullData = pd.read_csv(self.path + "/zodomPoses.csv" , sep=',', header=None)
        self.phi_rad = np.array(fullData.values[:,4])
        self.theta_rad = np.array(fullData.values[:,5])
        self.psi_rad = np.array(fullData.values[:,6])
        self.time = np.array(fullData.values[:,0])

        
        for k in range(np.shape(self.actualData.t)[0]):
            t = self.actualData.t[k]
            newData = self.params[0]*self.actualData.get(t) + \
                self.params[1:4]*t + self.params[4:7]
            xs.append(newData[0])
            ys.append(newData[1])
            zs.append(newData[2])

        for i in self.phi_rad:
            phi.append([i])
        for i in self.theta_rad:
            theta.append([i])
        for i in self.psi_rad:
            psi.append([i])
        for i in self.time:
            time.append([i])



        # print(np.transpose(np.array(ys)))
        # print(np.transpose(np.array(time)))

        # for k in self.actualData.:

        # self.correctedData = PoseData(xs = np.transpose(np.array(xs)),
        #     ys = np.transpose(np.array(ys)), zs = np.transpose(np.array(zs)),
        #     ts = self.actualData.t)

        self.correctedData = PoseData(xs = np.transpose(np.array(xs)),
            ys = np.transpose(np.array(ys)), 
            zs = np.transpose(np.array(zs)),
            phi = np.transpose(np.array(phi)),
            theta = np.transpose(np.array(theta)),
            psi = np.transpose(np.array(psi)),
            ts = np.transpose(np.array(time)))
        print("done")


    def calcPointCorrectedData(self, t, addIdx):

        params = []
        for k in range(self.paramsNum):
            if addIdx == k:
                p = self.params[k] + self.dp
            else:
                p = self.params[k]
            params.append(p)

        params = np.array(params).reshape(self.paramsNum,1)
        return params[0]*self.actualData.get(t) + params[1:4]*t + params[4:7]


    def costFunc(self, addIdx = 10):

        J = 0
        for k in range(np.shape(self.trueData.t)[0]):
            t = self.trueData.t[k]
            currentTrueData = self.trueData.get(t)
            J += np.linalg.norm(currentTrueData - self.calcPointCorrectedData(t,
                addIdx))**2

        return J


    def costFuncGradient(self):

        sigmaJ = []
        back = self.costFunc()
        print("Error value: ")
        print(back)
        for k in range(self.paramsNum):
            forward = self.costFunc(k)
            sigmaJ.append((forward - back)/self.dp)

        return np.array(sigmaJ).reshape(self.paramsNum, 1)


    def gradientDescentOptimize(self):

        ret = False
        k = 0
        while not ret and k < self.allowedSteps:
            print("Optimization is processing - params: ")
            print(np.transpose(self.params))
            k += 1
            newParams = self.params - self.delta * self.costFuncGradient()
            diff = np.linalg.norm(newParams - self.params)
            ret = diff <= self.convergencRadius
            print("params change: ")
            print(diff)
            self.params = newParams

        return ret



if __name__ == '__main__' :

    odpd = OptimizeDronePoseData(args.path)
    # odpd.gradientDescentOptimize()
    odpd.visualize()
