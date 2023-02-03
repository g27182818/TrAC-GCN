import pandas as pd
import os
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import sys
from datasets import *
from utils import *

# TODO: Crear bash para este
# python baseline.py --exp_name baseline_exp_combat_seq --max_time 21600 --batch_corr combat_seq

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
dataset = ShokhirevDataset( path = os.path.join("data","Shokhirev_2020"),   norm = args.norm,                           log2 = args.log2 == 'True',
                            val_frac = args.val_frac,                       test_frac = args.test_frac,                 corr_thr = args.corr_thr,
                            p_thr = args.p_thr,                             filter_type = args.filter_type,             batch_corr = args.batch_corr,
                            batch_sample_thr = args.batch_sample_thr,       exp_frac_thr = args.exp_frac_thr,           batch_norm = args.batch_norm == 'True',
                            string = args.string == 'True',                 conf_thr = args.conf_thr,                   channels_string = ['combined_score'],
                            shuffle_seed = args.shuffle_seed,               force_compute = args.force_compute == 'True')



# Define the number of bins
bin_num = 10
# Define a unified y to compute quantiles of global data
glob_y = np.hstack((dataset.split_dict['y_train'], dataset.split_dict['y_test']))
# Compute the bin frontiers with quantiles
bin_frontiers = np.quantile(glob_y, np.linspace(0,1, bin_num+1))
# Get class of each data
class_vec = np.digitize(glob_y, bin_frontiers, right=True)-1
# Slightly modify class_vec to ensure consistency
class_vec[class_vec==-1] = 0
# Pass class vec to string to make the model go to classification
class_vec = class_vec.astype(str)
class_vec = np.array([f'Class {class_vec[i]}' for i in range(len(class_vec))])

class_y_train = class_vec[:len(dataset.split_dict['y_train'])]
class_y_test = class_vec[len(dataset.split_dict['y_train']):]

# TODO: Make this a real dataframe with sample names and gene names as columns
# Merge both dataframes to enter H2O API
# expression_age_train = np.hstack((dataset.split_dict['x_train'], dataset.split_dict['y_train'].reshape((-1, 1))))
# expression_age_val = np.hstack((dataset.split_dict['x_val'], dataset.split_dict['y_val'].reshape((-1, 1))))
expression_age_train = pd.DataFrame(dataset.split_dict['x_train'])
expression_age_train['class'] = class_y_train.reshape((-1, 1))
expression_age_val = pd.DataFrame(dataset.split_dict['x_test'])
expression_age_val['class'] = class_y_test.reshape((-1, 1))


# Start H2O API
h2o.init()
# Dataframe declaration
expression_age_h2o_train = h2o.H2OFrame(expression_age_train)
expression_age_h2o_val = h2o.H2OFrame(expression_age_val)

# Declare name of dependent variable
y = 'class'

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
f = open(log_path, 'a')
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

# gt_array = np.array(gt_list[1:]).astype(float)
# pred_array = np.array(pred_list[1:]).astype(float)
gt_array = np.array(gt_list[1:])
pred_array = np.array(pred_list[1:])


print(classification_report(gt_array, pred_array[:, 0]))
__console__=sys.stdout
f = open(log_path, 'a')
sys.stdout = f
print(classification_report(gt_array, pred_array[:, 0]))
sys.stdout=__console__
f.close()

# Save ground truths and predictions
np.save(os.path.join(exp_folder,"gt_val.npy"), gt_array)
np.save(pred_path, pred_array)

# # Makes regression plot
# plt.figure(figsize=(7,7))
# plt.plot(gt_array, pred_array, '.k')
# plt.plot(np.arange(111), np.arange(111), 'k')
# plt.xlim([0, 110])
# plt.ylim([0,110])
# plt.grid()
# plt.title('Test visualization\nH2O AutoML best', fontsize = 16)
# plt.xlabel('Real Age (Yr)', fontsize = 16)
# plt.ylabel('Predicted Age (Yr)', fontsize = 16)
# plt.tight_layout()
# plt.savefig(figure_path, dpi=200)