
from telloGTGeneration import ArucoBasedDroneGroundTruthGeneration 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The Path to the Bag File.', dest='path')
parser.add_argument('-b', '--bagdir', help='The Path to the Bag File.', dest='bagdir')
args, unknown = parser.parse_known_args()

abdgtg = ArucoBasedDroneGroundTruthGeneration(args.path, args.bagdir, outlierRemovalMode=True)

abdgtg.deleteOutliers()
