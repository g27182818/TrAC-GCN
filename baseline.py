import pandas as pd
import os
import h2o
from h2o.automl import H2OAutoML
import matplotlib.pyplot as plt
import numpy as np
import sys
from datasets import *
from utils import *


# Get Parser
parser = get_main_parser()
# Add additional specific arguments
parser.add_argument('--max_time',   type=int,   default=60,     help='The maximum time allowed to autoML to train models.')
args = parser.parse_args()
args_dict = vars(args)

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
# --------------------------------------------------------------------------------------------------------------------------------------------------------#

# Experiment folder to save results and other paths
exp_folder = os.path.join('Results', 'baseline', f'{args.exp_name}_{args.max_time}_s')
os.makedirs(exp_folder, exist_ok=True)
log_path = os.path.join(exp_folder, 'log.txt')
model_path = os.path.join(exp_folder, 'models')
pred_path = os.path.join(exp_folder, 'preds_val.npy')
figure_path = os.path.join(exp_folder, 'val_prediction.png')

# Print experiment parameters
with open(log_path, 'a') as f:
    print_both('Argument list to program',f)
    print_both('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                    for arg in args_dict]),f)
    print_both('\n\n',f)

# Load data
dataset = ShokhirevDataset(os.path.join("data","Shokhirev_2020"), norm=norm, log2=log2_bool, val_frac = val_fraction,  test_frac = test_fraction, 
                            corr_thr=0.8, p_thr=0.05, filter_type=filter_type, ComBat=ComBat, ComBat_seq=ComBat_seq, batch_sample_thr = 100,
                            exp_frac_thr = 0.5, batch_norm=True, string=False, conf_thr=0.7, channels_string = [], shuffle_seed=0,
                            force_compute=False)




# TODO: Make this a real dataframe with sample names and gene names as columns
# Merge both dataframes to enter H2O API
expression_age_train = np.hstack((dataset.split_dict['x_train'], dataset.split_dict['y_train'].reshape((-1, 1))))
expression_age_val = np.hstack((dataset.split_dict['x_val'], dataset.split_dict['y_val'].reshape((-1, 1))))
print(expression_age_train)

# Start H2O API
h2o.init()
# Dataframe declaration
expression_age_h2o_train = h2o.H2OFrame(expression_age_train)
expression_age_h2o_val = h2o.H2OFrame(expression_age_val)

# Declare name of dependent variable
y = expression_age_train.shape[1]-1

# Perform dataframe split
# splits = expression_age_h2o.split_frame(ratios = [0.8], seed = 1)
# train = splits[0]
# test = splits[1]

# Declare and perform algorithm training
aml = H2OAutoML(max_runtime_secs = args.max_time, seed = 1, project_name = args.exp_name,
                include_algos=['DRF', 'GLM', 'XGBoost', 'GBM', 'DeepLearning', 'StackedEnsemble'],
                verbosity='info')

# FIXME: When including cross-validation the training, leaderboard and validation frames must be modified 
aml.train(y = y, training_frame = expression_age_h2o_train, leaderboard_frame = expression_age_h2o_val, validation_frame = expression_age_h2o_val)

# Get detailed performance of leader model
best_perf = aml.leader.model_performance(expression_age_h2o_val)


###################################################################################################
__console__=sys.stdout
f = open(log_path, 'w')
sys.stdout = f

# Print leader-board
print("--------------------------------------------")
print("COMPLETE AUTOML LEADERBOARD:")
print("--------------------------------------------")
print(h2o.automl.get_leaderboard(aml, extra_columns = "ALL"))
print("--------------------------------------------")
print("BEST MODEL DETAILED PERFORMANCE:")
print("--------------------------------------------")
print(best_perf)

sys.stdout=__console__

f.close()
#####################################################################################################

# Print leader-board
print(aml.leaderboard)
# Get detailed performance of leader model
print(best_perf)

# Download leader model
downloaded_model = h2o.download_model(aml.leader, path=model_path)

# Show predictions
pred = aml.leader.predict(expression_age_h2o_val)

pred_list = h2o.as_list(pred, use_pandas=False) 
gt_list = h2o.as_list(expression_age_h2o_val[:, y], use_pandas=False)

h2o.cluster().shutdown()

gt_array = np.array(gt_list[1:]).astype(float)
pred_array = np.array(pred_list[1:]).astype(float)

# Save ground truths and predictions
np.save(os.path.join(exp_folder,"gt_val.npy"), gt_array)
np.save(pred_path, pred_array)

# Makes regression plot
plt.figure(figsize=(7,7))
plt.plot(gt_array, pred_array, '.k')
plt.plot(np.arange(111), np.arange(111), 'k')
plt.xlim([0, 110])
plt.ylim([0,110])
plt.grid()
plt.title('Test visualization\nH2O AutoML best', fontsize = 16)
plt.xlabel('Real Age (Yr)', fontsize = 16)
plt.ylabel('Predicted Age (Yr)', fontsize = 16)
plt.tight_layout()
plt.savefig(figure_path, dpi=200)