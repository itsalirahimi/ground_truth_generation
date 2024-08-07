
from checkPoseDetection import CheckingObjectPoses 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The Path to the Front and Side Frames.', dest='path')
args, unknown = parser.parse_known_args()

cop = CheckingObjectPoses(args.path, checkObjectPoseMode=True)

cop.CheckObjectPose()
