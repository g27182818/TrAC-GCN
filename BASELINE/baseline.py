import pandas as pd
import os
import h2o
from h2o.automl import H2OAutoML
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


# Parser to specify the normalization method to perform analysis #####################################
parser = argparse.ArgumentParser(description='Baseline code for age estimation using AutoML.')
parser.add_argument('--norm', type=str, default="TPM",
                    help='The normalization method to be loaded via files. Can be raw, TPM or TMM.')
parser.add_argument('--max_time', type=int, default=60,
                    help='Maximum time to let the AutoML algorithm to run.')
parser.add_argument('--log2', type=bool, default=True,
                    help='Parameter indicating if a log2 transformation is done.')
args = parser.parse_args()
######################################################################################################

# Handle paths depending on log2 argument
if args.log2 == True:
    log_path = os.path.join("logs", args.norm +"_log2_"+ str(args.max_time)+ "s_final_performance.txt")
    pred_path = os.path.join("preds",args.norm + "_log2_"+ str(args.max_time) + "s_preds_test.npy")
    figure_path = os.path.join("images", args.norm + "_log2_"+ str(args.max_time)+ "s")
    model_path = os.path.join("models", args.norm + "_log2_"+ str(args.max_time)+ "s")
    project_name = args.norm + "_log2_"+ str(args.max_time)+ "s"
else:    
    log_path = os.path.join("logs", args.norm +"_"+ str(args.max_time)+ "s_final_performance.txt")
    pred_path = os.path.join("preds",args.norm + "_"+ str(args.max_time) + "s_preds_test.npy")
    figure_path = os.path.join("images", args.norm + "_"+ str(args.max_time)+ "s")
    model_path = os.path.join("models", args.norm + "_"+ str(args.max_time)+ "s")
    project_name = args.norm + "_"+ str(args.max_time)+ "s"

# Declare expression data and metadata path 
expression_path = os.path.join("data","Shokhirev_2020","normalized", args.norm + "_dataset.csv")
meta_path = os.path.join("data","Shokhirev_2020","meta_filtered.txt")


# Read expression file and make minnor modifications
expression = pd.read_csv(expression_path, sep=",", header=0)
expression = expression.rename(columns = {'Unnamed: 0':'SRR.ID'})
expression = expression.set_index('SRR.ID')
expression = expression.T

if args.log2 == True:
    # Perform the log2 transform of data
    expression = np.log2(expression + 1)


# Read metadata file and make minnor modifications
meta = pd.read_csv(meta_path, sep="\t", header=0)
del meta['Unnamed: 0']
meta = meta.set_index('SRR.ID')

# Merge both dataframes to enter H2O API
expression_age = expression.merge(meta['Age'], left_index=True, right_index=True)
print(expression_age)

# Satart H2O API
h2o.init()
# Dataframe declaration
expression_age_h2o = h2o.H2OFrame(expression_age)

# Declare name of dependet variable
y = 'Age'

# Perform dataframe split
splits = expression_age_h2o.split_frame(ratios = [0.8], seed = 1)
train = splits[0]
test = splits[1]

# Declare and perform algorithm training
aml = H2OAutoML(max_runtime_secs = args.max_time, seed = 1, project_name = project_name,
                include_algos=['DRF', 'GLM', 'XGBoost', 'GBM', 'DeepLearning', 'StackedEnsemble'],
                verbosity='info')
aml.train(y = y, training_frame = train, leaderboard_frame = test)

# Get detailed performance of leader model
best_perf = aml.leader.model_performance(test)


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

f.close()
sys.stdout=__console__
#####################################################################################################

# Print leader-board
print(aml.leaderboard)
# Get detailed performance of leader model
print(best_perf)

# Download leader model
downloaded_model = h2o.download_model(aml.leader, path=model_path)

# Show predictions
pred = aml.leader.predict(test)

pred_list = h2o.as_list(pred, use_pandas=False) 
gt_list = h2o.as_list(test['Age'], use_pandas=False)

h2o.cluster().shutdown()

gt_array = np.array(gt_list[1:]).astype(np.float)
pred_array = np.array(pred_list[1:]).astype(np.float)

# Save ground truths and predictions
np.save(os.path.join("preds","gt_test.npy"), gt_array)
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