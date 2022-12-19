
from telloGTGeneration import ArucoBasedDroneGroundTruthGeneration 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The Path to the Bag File.', dest='path')
args, unknown = parser.parse_known_args()

abdgtg = ArucoBasedDroneGroundTruthGeneration(args.path, outlierRemovalMode=True)

abdgtg.deleteOutliers()
