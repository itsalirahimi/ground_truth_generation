import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--in', help='The Path to Input.', dest='inp')
parser.add_argument('-o', '--out', help='The directory for output files.', dest='outp')
args, unknown = parser.parse_known_args()


print("GUIDE:" + \
    "\n \t any key except options: next" + \
    "\n \t d: This point is noise. Remove it" + \
    "\n \t s: Split a separated output file until this point" + \
    "\n \t q: Quite")

indices = []
idx = 0

path_to_data1 = args.inp
data1 = np.array(pd.read_csv(path_to_data1, header=None))

raw_x = data1[:,0]
raw_y = data1[:,1]
raw_t = data1[:,2]

frame = np.random.random((10, 10))

clean_y = []
clean_x = []
clean_t = []

key = -1

milestones = []

for k, num in enumerate(raw_y):
    if key == ord('d'):
        indices.append(idx)
        # print(indices)
    elif key == ord('q'):
        cv.destroyAllWindows()
        break
    elif key == ord('s'):
        milestones.append(idx)
        
    x2 = [raw_y[index] for index in indices]
    t2 = [raw_x[index] for index in indices]
    plt.cla()
    plt.plot(raw_x, raw_y)
    plt.scatter(raw_x[k], raw_y[k], Color='yellow')
    plt.scatter(t2, x2, Color='red')
    plt.grid(True)
    plt.pause(0.001)
    cv.imshow("input window", frame)
    key = cv.waitKey()
    idx = k

it = 0
clean_milestones = []
for k, num in enumerate(raw_y):
    if k in milestones:
        clean_milestones.append(k)
    if not k in indices:
        clean_y.append(num)
        clean_x.append(raw_x[k])
        clean_t.append(raw_t[k])
        it += 1

plt.close()

# print(clean_y)
# print(clean_x)

plt.plot(clean_x, clean_y)
plt.show()

if len(clean_milestones) != 0 or len(indices) != 0:
    m_prior = 0
    for kk, m in enumerate(clean_milestones):
        df_marker = pd.DataFrame({'Xs':clean_x[m_prior:m], 'Ys':clean_y[m_prior:m], 'Time':clean_t[m_prior:m]})
        df_marker.to_csv(args.outp + "/clean_object_poses_part_{}.csv".format(kk+1), mode='w', index=False, header=False)
        m_prior = m
        
    df_marker = pd.DataFrame({'Xs':clean_x[m_prior:len(clean_x)], 'Ys':clean_y[m_prior:len(clean_y)], 
                            'Time':clean_t[m_prior:len(clean_t)]})
    df_marker.to_csv(args.outp + "/clean_object_poses_part_end.csv", mode='w', index=False, header=False)

