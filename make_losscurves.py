import numpy as np
import matplotlib.pyplot as plt
import argparse 

from analysis_tools import plot_training_curves

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default = "results/training_curves.npz", help = "Path to loss curves file to plot (.npz)")
parser.add_argument("--output_path", type=str, default = "plots/training_curves.png", help = "Path to save output file (.png)")
args = parser.parse_args()

#loading data file & extracting single curves
plot_training_curves(file_path = args.input_path, outpath = args.output_path)