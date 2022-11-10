# Specific imports
from models import *
from datasets import *
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

# # Add optional timmer to delay the execution of the code
# import time
# time.sleep(6000)

# Get terminal line arguments
parser = get_main_parser()
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
# Graph parameters ---------------------------------------------------------------------------------------------------------------------------------------#
string = args.string == 'True'                                       # Whether to use STRING data to define graph                                         #
conf_thr = args.conf_thr                                             # Confidence threshold to be used for defining graph connectivity with STRING        #
all_string = args.all_string == 'True'                               # Whether to use all STRING channels or just combined_score                          #
corr_thr = args.corr_thr                                             # Spearman correlation threshold for declaring graph topology                        #
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
    experiment_name = norm + batch_str + "_" + filter_type + "_filtering_coor_thr=" + str(corr_thr)

# TODO: Change the result saving pipeline to a file tree
# Declare results path
results_path = os.path.join("Results", experiment_name)
# Declare log path
train_log_path = os.path.join(results_path, "TRAINING_LOG.txt")
# Declare metric dicts path
metrics_log_path = os.path.join(results_path, "metric_dicts.pickle")
# Declare path to save performance training plot
train_performance_fig_path = os.path.join(results_path, "training_performance.png")
# Declare path to save val set predictions plot
val_prediction_fig_path = os.path.join(results_path, "val_prediction.png")
# Declare path to save best model
best_model_path = os.path.join(results_path, "best_model.pt")

# Create results directory
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# Print experiment parameters
with open(train_log_path, 'a') as f:
    print_both('Argument list to program',f)
    print_both('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                    for arg in args_dict]),f)
    print_both('\n\n',f)

# Load data
dataset = ShokhirevDataset(os.path.join("data","Shokhirev_2020"), norm=norm, log2=log2_bool, val_frac = val_fraction,  test_frac = test_fraction, 
                            corr_thr=corr_thr, p_thr=p_value_thr, filter_type=filter_type, ComBat=ComBat, ComBat_seq=ComBat_seq, batch_sample_thr = 100,
                            exp_frac_thr = 0.5, batch_norm=False, string=string, conf_thr=conf_thr, channels_string = channels_string, shuffle_seed=0,
                            force_compute=False)

train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=batch_size)
n_genes = dataset.num_valid_genes

# This handles the model type and the weight decay
if model_type == "baseline":
    model = BaselineModel(hidden_channels=hidd, input_size=n_genes,
                          out_size=1,
                          dropout=dropout,
                          final_pool=final_pool).to(device)
    weight_decay = 0.0

elif model_type == "graph_head":
    model = GraphHead(hidden_channels=hidd, input_size=n_genes,
                      out_size=1,
                      dropout=dropout,
                      final_pool=final_pool).to(device)
    weight_decay = 0.01

elif model_type == "trac_gcn":
    model = TracGCN(hidden_channels=hidd, input_size=n_genes,
                      out_size=1,
                      dropout=dropout,
                      final_pool=final_pool).to(device)
    weight_decay = 0.01

elif model_type == "baseline_pool":
    model = BaselineModelPool(hidden_channels=hidd, input_size=n_genes,
                              out_size=1,
                              dropout=dropout,
                              final_pool=final_pool,
                              cluster_num=1014).to(device)
    weight_decay = 0.0

elif model_type == "deepergcn":
    model = DeeperGCN(hidden_channels=hidd, input_size=n_genes,
                      input_node_channels=1,
                      num_layers=5,
                      dropout=dropout,
                      final_pool=final_pool).to(device)
    weight_decay = 0.0

elif model_type == "MLR":
    model = MLP(h_sizes=[n_genes], out_size=1, init_weights=None, dropout=dropout).to(device)
    weight_decay = 0.0

elif model_type == "MLP":
    model = MLP(h_sizes=[n_genes, 1000], out_size=1, init_weights=None, dropout=dropout).to(device)
    weight_decay = 0.0

elif model_type == "holzscheck_MLP":
    model = MLP(h_sizes=[n_genes, 350, 350, 350, 50],
                out_size=1, act="elu",
                init_weights=torch.nn.init.kaiming_uniform_,
                dropout=dropout).to(device)
    weight_decay = 0.01

elif model_type == "wang_MLP":
    model = MLP(h_sizes=[n_genes, 256, 256, 32],
                out_size=1, act="relu",
                init_weights=torch.nn.init.kaiming_uniform_,
                dropout=dropout).to(device)
    weight_decay = 5e-4

elif model_type == "DFS":
    model = DFS(h_sizes=[n_genes, 512, 256, 128],
                out_size=1, act="elu",
                dropout=dropout).to(device)
    weight_decay = 0.01
else:
    raise NotImplementedError

# Print to console model definition
# Print experiment parameters
with open(train_log_path, 'a') as f:
    print_both("The model definition is:", f)
    print_both(str(model), f)

# Define optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Handle multiple losses
if loss_fn == 'mse':
    criterion = torch.nn.MSELoss()
elif loss_fn == 'l1':
    criterion = torch.nn.L1Loss()
else:
    raise NotImplementedError


# Decide whether to train and test adversarially or not
train_adversarial = train_eps > 0.0
test_adversarial = train_eps > 0.0

# Lists declarations
train_metric_lst = []
val_metric_lst = []
adv_val_metric_lst = []
loss_list = []

# Best metric variables declaration
best_train_metric = {'MAE': None, 'RMSE': None, 'R^2': None}
best_val_metric = {'MAE': 1e10, 'RMSE': None, 'R^2': None}
best_adv_val_metric = {'MAE': None, 'RMSE': None, 'R^2': None}


# Train/test cycle
for epoch in range(total_epochs):
    print('-----------------------------------------')
    print("Epoch " + str(epoch+1) + ":")
    print('                                         ')
    print("Start training:")
    # Train one epoch adversarially
    if train_adversarial:
        loss = train(train_loader, model, device, criterion, optimizer,
                        adversarial=True, attack=apgd_graph, epsilon=train_eps,
                        n_iter=n_iters_apgd)
    # Train one epoch normally
    else:
        loss = train(train_loader, model, device, criterion, optimizer)

    # Obtain test metrics for each epoch in all groups
    print('                                         ')
    print("Obtaining train metrics:")
    train_metrics = test(train_loader, model, device)

    print('                                         ')
    print("Obtaining val metrics:")
    val_metrics = test(val_loader, model, device)

    # Handle if adversarial testing is required
    if test_adversarial:
        print('                                         ')
        print("Obtaining adversarial val metrics:")
        # This test is set to use 50 iterations of APGD
        adv_val_metrics = test(val_loader, model, device,
                                optimizer=optimizer, adversarial=True,
                                attack=apgd_graph, criterion=criterion,
                                epsilon=test_eps, n_iter=50)

    # If adversarial testing is not required adversarial test metrics are the same normal metrics
    else:
        adv_val_metrics = val_metrics

    # Add epoch information to the dictionaries
    train_metrics["epoch"] = epoch
    val_metrics["epoch"] = epoch
    adv_val_metrics["epoch"] = epoch

    # Append data to list
    train_metric_lst.append(train_metrics)
    val_metric_lst.append(val_metrics)
    adv_val_metric_lst.append(adv_val_metrics)
    loss_list.append(loss.cpu().detach().numpy())

    # Print performance
    print_epoch(train_metrics, val_metrics, adv_val_metrics, loss, epoch, train_log_path)

    # Save model if it is the best so far
    if val_metrics['MAE'] < best_val_metric['MAE']:
        # Update best metric variables
        best_val_metric = val_metrics.copy()
        best_train_metric = train_metrics.copy()
        best_adv_val_metric = adv_val_metrics.copy()
        # Save best model until now
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss},
            best_model_path)

# Save metrics dicts
complete_metric_dict = {"train": train_metric_lst,
                        "test": val_metric_lst,
                        "adv_test": adv_val_metric_lst,
                        "loss": loss_list}

with open(metrics_log_path, 'wb') as f:
    pickle.dump(complete_metric_dict, f)

# Generate training performance plot and save it to train_performance_fig_path
plot_training(train_metric_lst, val_metric_lst, adv_val_metric_lst, loss_list, train_performance_fig_path)

# Get best model from results_path/best_model.pt
best_model_dict = torch.load(os.path.join(results_path, "best_model.pt"))
model.load_state_dict(best_model_dict['model_state_dict'])

# Print best performance on val set to log
with open(train_log_path, 'a') as f:
    print_both('-----------------------------------------',f)
    print_both("Best MAE performance:",f)
print_epoch(best_train_metric,
            best_val_metric, 
            best_adv_val_metric, 
            best_model_dict['loss'], 
            best_model_dict['epoch'],
            train_log_path)

# Generate val predictions plot and save it to val_predictions_fig_path
plot_predictions(model, device, val_loader, val_prediction_fig_path)


# TODO: Do a 'just_plot' option to just plot the training performance and not train the model






