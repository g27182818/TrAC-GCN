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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Set matplotlib option to plot while in screen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(1234)

# # Add optional timmer to delay the execution of the code
# import time
# time.sleep(6000)


# Parser to specify the normalization method to perform analysis #####################################
parser = argparse.ArgumentParser(description='Code for TrAC-GCN implementation.')
# Dataset parameters ##################################################################################
parser.add_argument('--norm',           type=str,   default="tpm",       help='The normalization method to be loaded via files. Can be raw, tpm or tmm.')
parser.add_argument('--log2',           type=str,   default='True',      help='Parameter indicating if a log2 transformation is done under input data.')
parser.add_argument('--ComBat',         type=str,   default='False',     help='Parameter indicating if a dataset with ComBat batch correction is loaded. Can be True just if log2 = True.')
parser.add_argument('--ComBat_seq',     type=str,   default='False',     help= 'Parameter indicating if a dataset with ComBat_seq batch correction is loaded.')
parser.add_argument('--filter_type',    type=str,   default='none',      help = 'filtering to be applied to genes, can be none, 1000var, 1000diff, 100var or 100diff')
# Graph parameters ###################################################################################
parser.add_argument('--string',         type=str,   default='False',     help='Parameter indicating if the graph made using STRING database.')
parser.add_argument('--all_string',     type=str,   default='False',     help='Parameter indicating if all STRING channels should be used otherwise combined_score will be used.')
parser.add_argument('--conf_thr',       type=float, default=0.0,         help='The confidence threshold to staablish connections in STRING graphs.')
parser.add_argument('--corr_thr',       type=float, default=0.8,         help='The correlation threshold to be used for definning graph connectivity.')
# Model parameters ###################################################################################
parser.add_argument('--model',          type=str,   default='baseline',  help='The model to be used.', choices= ['baseline', 'deepergcn', 'MLR', 'MLP', 'holzscheck_MLP', 'wang_MLP', 'baseline_pool', 'graph_head', 'trac_gcn'] )
parser.add_argument('--hidden_chann',   type=int,   default=8,           help='The number of hidden channels to use in the graph based models.')
parser.add_argument('--dropout',        type=float, default=0.0,         help='Dropout rate to be used in models.')
parser.add_argument('--final_pool',     type=str,   default=None,        help='Final pooling type over nodes to be used in graph based models.', choices= ['mean', 'max', 'add', 'none'])
# Training parameters ################################################################################
parser.add_argument('--exp_name',       type=str,   default='misc_test', help='Experiment name to be used for saving files. Default is misc_test. If set to -1 the name will be generated automatically.')
parser.add_argument('--loss',           type=str,   default='mse',       help='Loss function to be used for training. Can be mse or l1.')
parser.add_argument('--lr',             type=float, default=0.00005,     help='Learning rate for training.')
parser.add_argument('--epochs',         type=int,   default=100,         help='Number of epochs for training.')
parser.add_argument('--batch_size',     type=int,   default=20,          help='Batch size for training.')
parser.add_argument('--adv_e_test',     type=float, default=0.00,        help='Adversarial upper bound of perturbations during test.')
parser.add_argument('--adv_e_train',    type=float, default=0.00,        help='Adversarial upper bound of perturbations during train.')
parser.add_argument('--n_iters_apgd',   type=int,   default=50,          help='Number of iterations for APGD during train.')
args = parser.parse_args()
args_dict = vars(args)
######################################################################################################


# ---------------------------------------- Important variable parameters ---------------------------------------------------------------------------------#
# Miscellaneous parameters -------------------------------------------------------------------------------------------------------------------------------#
torch.manual_seed(12345)                                             # Set torch manual seed                                                              #
device = torch.device("cuda")                                        # Set cuda device                                                                    #
# Dataset parameters -------------------------------------------------------------------------------------------------------------------------------------#
val_fraction = 0.2                                                   # Fraction of the data used for validation                                           #
test_fraction = 0.2                                                  # Fraction of the data used for test                                                 #
batch_size = args.batch_size                                         # Batch size parameter                                                               #
norm = args.norm                                                     # Normalization method used in the input data. Can be 'raw', 'TPM' or 'TMM'          #
log2_bool = args.log2 == 'True'                                      # Whether to make a Log2 transformation of the input data                            #
filter_type = args.filter_type                                       # Filter applied to genes can be 'none', '1000var', '1000diff', '100var' or '100diff'#
ComBat = args.ComBat == 'True'                                       # Whether to load ComBat batch corrected dataset. # TODO: Make single parameter      #
ComBat_seq = args.ComBat_seq == 'True'                               # Whether to load ComBat_seq batch corrected dataset                                 #
# Graph parameters ---------------------------------------------------------------------------------------------------------------------------------------#
string = args.string == 'True'                                       # Whether to use STRING data to define graph                                         #
conf_thr = args.conf_thr                                             # Confidence threshold to be used for defining graph connectivity with STRING        #
all_string = args.all_string == 'True'                               # Whether to use all STRING channels or just combined_score                          #
coor_thr = args.corr_thr                                             # Spearman correlation threshold for declaring graph topology                        #
p_value_thr = 0.05                                                   # P-value Spearman correlation threshold for declaring graph topology                #
# Model parameters ---------------------------------------------------------------------------------------------------------------------------------------#
hidd = args.hidden_chann                                             # Hidden channels parameter for graph models                                         #
model_type = args.model                                              # Model type                                                                         #
dropout = args.dropout                                               # Dropout parameter for models                                                       #
final_pool = args.final_pool                                         # Final pooling string for graph based models                                        #
# Training parameters ------------------------------------------------------------------------------------------------------------------------------------#
experiment_name = args.exp_name                                      # Experiment name to define path were results are stored                             #
loss_fn = args.loss                                                  # Loss function to be used for training. Can be mse or l1.                           #
lr = args.lr                                                         # Learning rate of the Adam optimizer (was changed from 0.001 to 0.00001)            #
total_epochs = args.epochs                                           # Total number of epochs to train                                                    #
train_eps = args.adv_e_train                                         # Adversarial epsilon for train                                                      #
n_iters_apgd = args.n_iters_apgd                                     # Number of performed APGD iterations in train                                       #
# Test parameters ----------------------------------------------------------------------------------------------------------------------------------------#
test_eps = args.adv_e_test                                           # Adversarial epsilon for test                                                       #
# --------------------------------------------------------------------------------------------------------------------------------------------------------#

# All posible channels for STRING graphs
str_all_channels = ['combined_score', 'textmining', 'database', 'experimental', 'coexpression', 'cooccurence', 'fusion', 'neighborhood']
channels_string = str_all_channels if all_string else ['combined_score']

# Handle automatic generation of experiment name
if experiment_name == '-1':
    # Handle different batch correction methods
    if ComBat:
        batch_str = '_batch_corr_ComBat_'
    elif ComBat_seq:
        batch_str = '_batch_corr_ComBat_seq_'
    else:
        batch_str = '_batch_corr_none_'
    # Define experiment name based on parameters
    experiment_name = norm + batch_str + "_" + filter_type + "_filtering_coor_thr=" + str(coor_thr)

# Load data
dataset_info = load_dataset(norm=norm, log2=log2_bool, corr_thr=coor_thr, p_thr=p_value_thr, force_compute=False,
                            val_frac=val_fraction, test_frac=test_fraction, filter_type=filter_type,
                            ComBat=ComBat, ComBat_seq=ComBat_seq, string = string, conf_thr = conf_thr,
                            channels_string = channels_string)

splits = dataset_info['split']
x_train = splits['x_train']
y_train = splits['y_train']
x_val = splits['x_val']
y_val = splits['y_val']




regr = RandomForestRegressor(n_jobs=-1, verbose=2, random_state=0)
regr.fit(x_train, y_train)

# Predict on validation set
y_pred = regr.predict(x_val)

# Compute regression metrics
# Compute MAE with sklearn
mae = mean_absolute_error(y_val, y_pred)
# Compute MSE with sklearn
mse = mean_squared_error(y_val, y_pred)
# Compute RMSE with sklearn
rmse = np.sqrt(mse)
# Compute R2 with sklearn
r2 = r2_score(y_val, y_pred)
# Print regression metrics
print('\n\nRegression metrics:\nMAE: {0}\nRMSE: {1}\nR2: {2}'.format(mae, rmse, r2))



