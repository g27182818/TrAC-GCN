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

# # Add optional timer to delay the execution of the code
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
# ----------------------------------------------------------------------------------------------------------------------------------------------------------#
# ComBat = args.ComBat == 'True'                                       # Whether to load ComBat batch corrected dataset.                                    # # TODO: Make single parameter
# ComBat_seq = args.ComBat_seq == 'True'                               # Whether to load ComBat_seq batch corrected dataset                                 #
# all_string = args.all_string == 'True'                               # Whether to use all STRING channels or just combined_score                          # # TODO: Add the possibility to add specific string channels
# ----------------------------------------------------------------------------------------------------------------------------------------------------------#


# All posible channels for STRING graphs
str_all_channels = ['combined_score', 'textmining', 'database', 'experimental', 'coexpression', 'cooccurence', 'fusion', 'neighborhood']
channels_string = str_all_channels if args.all_string == 'True' else ['combined_score']

# Handle automatic generation of experiment name
if args.exp_name == '-1':
    # Handle different batch correction methods
    if args.ComBat == 'True':
        batch_str = '_batch_corr_ComBat_'
    elif args.ComBat_seq == 'True':
        batch_str = '_batch_corr_ComBat_seq_'
    else:
        batch_str = '_batch_corr_None_'
    # Define experiment name based on parameters
    args.exp_name = args.norm + batch_str + "_" + args.filter_type + "_filtering_coor_thr=" + str(args.corr_thr)

# TODO: Change the result saving pipeline to a file tree
# Declare results path
results_path = os.path.join("Results", args.exp_name)
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
dataset = ShokhirevDataset( path = os.path.join("data","Shokhirev_2020"),   norm = args.norm,                           log2 = args.log2 == 'True',
                            val_frac = args.val_frac,                       test_frac = args.test_frac,                 corr_thr = args.corr_thr,
                            p_thr = args.p_thr,                             filter_type = args.filter_type,             ComBat = args.ComBat == 'True',
                            ComBat_seq = args.ComBat_seq == 'True',         batch_sample_thr = args.batch_sample_thr,   exp_frac_thr = args.exp_frac_thr, 
                            batch_norm = args.batch_norm == 'True',         string = args.string == 'True',             conf_thr = args.conf_thr,
                            channels_string = channels_string,              shuffle_seed = args.shuffle_seed,           force_compute = args.force_compute == 'True')

train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size = args.batch_size)
n_genes = dataset.num_valid_genes

# This handles the model type and the weight decay
if args.model == "baseline":
    model = BaselineModel(hidden_channels = args.hidden_chann, input_size=n_genes,
                          out_size=1,
                          dropout = args.dropout,
                          final_pool = args.final_pool).to(device)
    weight_decay = 0.0

elif args.model == "graph_head":
    model = GraphHead(hidden_channels = args.hidden_chann, input_size=n_genes,
                      out_size=1,
                      dropout = args.dropout,
                      final_pool = args.final_pool).to(device)
    weight_decay = 0.01

elif args.model == "trac_gcn":
    model = TracGCN(hidden_channels = args.hidden_chann, input_size=n_genes,
                      out_size=1,
                      dropout = args.dropout,
                      final_pool = args.final_pool).to(device)
    weight_decay = 0.01

elif args.model == "baseline_pool":
    model = BaselineModelPool(hidden_channels = args.hidden_chann, input_size=n_genes,
                              out_size=1,
                              dropout = args.dropout,
                              final_pool = args.final_pool,
                              cluster_num=1014).to(device)
    weight_decay = 0.0

elif args.model == "deepergcn":
    model = DeeperGCN(hidden_channels = args.hidden_chann, input_size=n_genes,
                      input_node_channels=1,
                      num_layers=5,
                      dropout = args.dropout,
                      final_pool = args.final_pool).to(device)
    weight_decay = 0.0

elif args.model == "MLR":
    model = MLP(h_sizes=[n_genes], out_size=1, init_weights=None, dropout = args.dropout).to(device)
    weight_decay = 0.0

elif args.model == "MLP":
    model = MLP(h_sizes=[n_genes, 1000], out_size=1, init_weights=None, dropout = args.dropout).to(device)
    weight_decay = 0.0

elif args.model == "holzscheck_MLP":
    model = MLP(h_sizes=[n_genes, 350, 350, 350, 50],
                out_size=1, act="elu",
                init_weights=torch.nn.init.kaiming_uniform_,
                dropout = args.dropout).to(device)
    weight_decay = 0.01

elif args.model == "wang_MLP":
    model = MLP(h_sizes=[n_genes, 256, 256, 32],
                out_size=1, act="relu",
                init_weights=torch.nn.init.kaiming_uniform_,
                dropout = args.dropout).to(device)
    weight_decay = 5e-4

elif args.model == "DFS":
    model = DFS(h_sizes=[n_genes, 512, 256, 128],
                out_size=1, act="elu",
                dropout = args.dropout).to(device)
    weight_decay = 0.01
else:
    raise NotImplementedError

# Print to console model definition
# Print experiment parameters
with open(train_log_path, 'a') as f:
    print_both("The model definition is:", f)
    print_both(str(model), f)

# Define optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=weight_decay)

# Handle multiple losses
if args.loss == 'mse':
    criterion = torch.nn.MSELoss()
elif args.loss == 'l1':
    criterion = torch.nn.L1Loss()
else:
    raise NotImplementedError


# Decide whether to train and test adversarially or not
train_adversarial = args.adv_e_train > 0.0
test_adversarial = args.adv_e_train > 0.0 # TODO: CHeck this bool it seems strange

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
for epoch in range(args.epochs):
    print('-----------------------------------------')
    print("Epoch " + str(epoch+1) + ":")
    print('                                         ')
    print("Start training:")
    # Train one epoch adversarially
    if train_adversarial:
        loss = train(   train_loader,           model,                      device,
                        criterion,              optimizer,                  adversarial = True,
                        attack = apgd_graph,    epsilon = args.adv_e_train, n_iter = args.n_iters_apgd)
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
        adv_val_metrics = test( val_loader,                 model,                      device,
                                optimizer = optimizer,      adversarial = True,         attack=apgd_graph,
                                criterion = criterion,      epsilon = args.adv_e_test,  n_iter=50)

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






