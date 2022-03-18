# Specific imports
from models import *
from dataloader import *
from utils import *
# Generic libraries
import numpy as np
import os
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse
import glob
# Set matplotlib option to plot while in screen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(1234)


# Parser to specify the normalization method to perform analysis #####################################
parser = argparse.ArgumentParser(description='Code for TrAC-GCN implementation.')
parser.add_argument('--norm', type=str, default="TPM",
                    help='The normalization method to be loaded via files. Can be raw, TPM or TMM.')
parser.add_argument('--log2', type=bool, default=True,
                    help='Parameter indicating if a log2 transformation is done under input data.')
args = parser.parse_args()
######################################################################################################

data_name = args.norm + "_log2" if args.log2 else args.norm

# Load data
dataset_info = load_dataset('TPM', log2=True, force_compute=False)

