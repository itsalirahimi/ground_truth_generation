
from YoloTelloGTGeneration import YoloArucoBasedDroneGroundTruthGeneration 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The Path to the Bag File.', dest='path')
args, unknown = parser.parse_known_args()

abdgtg = YoloArucoBasedDroneGroundTruthGeneration(args.path, outlierRemovalMode=True)

abdgtg.deleteYoloOutliers()
