#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The Path to the CSV File.', dest='path', required=True)
parser.add_argument('-c', '--column', help='Time column number (0-indexed)', dest='column', type=int, required=True)
args, unknown = parser.parse_known_args()


class readCSV:
    def __init__(self, dataFrame=None, time_column=0):
        if dataFrame is not None:
            self.time_column = time_column
            self.dataFrame = dataFrame
            self.t = np.array(dataFrame.values[:, time_column]) - dataFrame.values[0, time_column]
    
    def replace_time_column(self, output_path=None):
        self.dataFrame.iloc[:, self.time_column] = self.t
        
        if output_path:
            self.dataFrame.to_csv(output_path, sep=',', header=False, index=False)
        else:
            self.dataFrame.to_csv(args.path, sep=',', header=False, index=False)
        
        print(f"The CSV file has been updated with the new time values at {output_path or args.path}.")

# Read the CSV file
csv_obj = readCSV(dataFrame=pd.read_csv(args.path, sep=',', header=None), time_column=args.column)

# Replace the time column
csv_obj.replace_time_column()

