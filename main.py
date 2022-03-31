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
parser.add_argument('--log2', type=str, default='True',
                    help='Parameter indicating if a log2 transformation is done under input data.')
parser.add_argument('--exp_name', type=str, default='misc_test',
                    help='Experiment name to be used for saving files. Default is misc_test. If set to -1 the name will be generated automatically.')
parser.add_argument('--loss', type=str, default='mse',
                    help='Loss function to be used for training. Can be mse or l1.')
parser.add_argument('--adv_e_test', type=float, default=0.00)
parser.add_argument('--adv_e_train', type=float, default=0.00)
parser.add_argument('--n_iters_apgd', type=int, default=50)
args = parser.parse_args()
######################################################################################################


# ---------------------------------------- Important variable parameters ----------------------------------------------#
# Miscellaneous parameters --------------------------------------------------------------------------------------------#
torch.manual_seed(12345)            # Set torch manual seed                                                            #
device = torch.device("cuda")       # Set cuda device                                                                  #
# Dataset parameters --------------------------------------------------------------------------------------------------#
val_fraction = 0.2                  # Fraction of the data used for validation                                         #
test_fraction = 0.2                 # Fraction of the data used for test                                               #
batch_size = 100                    # Batch size parameter                                                             #
coor_thr = 0.8                      # Spearman correlation threshold for declaring graph topology                      #
p_value_thr = 0.05                  # P-value Spearman correlation threshold for declaring graph topology              #
norm = args.norm                    # Normalization method used in the input data. Can be 'raw', 'TPM' or 'TMM'        #
log2_bool = args.log2 == 'True'     # Whether to make a Log2 transformation of the input data                          #
# Model parameters ----------------------------------------------------------------------------------------------------#
hidd = 2                            # Hidden channels parameter for baseline model                                     #
model_type = "baseline_simple"      # Model type, can be # TODO: Complete model types                                  #
# Training parameters -------------------------------------------------------------------------------------------------#
experiment_name = args.exp_name     # Experiment name to define path were results are stored                           #
loss_fn = args.loss                 # Loss function to be used for training. Can be mse or l1.                         #
lr = 0.0001                         # Learning rate of the Adam optimizer (was changed from 0.001 to 0.00001)          #
total_epochs = 20                    # Total number of epochs to train                                                  #
train_eps = args.adv_e_train        # Adversarial epsilon for train                                                    #
n_iters_apgd = args.n_iters_apgd    # Number of performed APGD iterations in train                                     #
# Test parameters -----------------------------------------------------------------------------------------------------#
test_eps = args.adv_e_test          # Adversarial epsilon for test                                                     #
# ---------------------------------------------------------------------------------------------------------------------#

# Handle automatic generation of experiment name
if experiment_name == '-1':
    experiment_name = args.norm + "_log2_True_" + loss_fn if log2_bool else args.norm + "_log2_False_" + loss_fn

# Load data
dataset_info = load_dataset(norm=norm, log2=log2_bool, corr_thr=coor_thr, p_thr=p_value_thr, force_compute=False,
                            val_frac=val_fraction, test_frac=test_fraction)
# Extract graph information
edge_indices, edge_attributes = dataset_info['graph']

# Pass splits to torch
split_dictionary = dataset_info['split']
torch_split = {}
for k in split_dictionary.keys():
    torch_split[k] = torch.tensor(split_dictionary[k], dtype=torch.float)

# Define datalists of graphs
train_graph_list = [Data(x=torch.unsqueeze(torch_split['x_train'][i, :], 1),
                         y=torch_split['y_train'][i],
                         edge_index=edge_indices,
                         edge_attributes=edge_attributes,
                         num_nodes=len(torch_split['x_train'][i, :])) for i in range(torch_split['x_train'].shape[0])]
val_graph_list = [Data(x=torch.unsqueeze(torch_split['x_val'][i, :], 1),
                         y=torch_split['y_val'][i],
                         edge_index=edge_indices,
                         edge_attributes=edge_attributes,
                         num_nodes=len(torch_split['x_val'][i, :])) for i in range(torch_split['x_val'].shape[0])]
test_graph_list = [Data(x=torch.unsqueeze(torch_split['x_test'][i, :], 1),
                         y=torch_split['y_test'][i],
                         edge_index=edge_indices,
                         edge_attributes=edge_attributes,
                         num_nodes=len(torch_split['x_test'][i, :])) for i in range(torch_split['x_test'].shape[0])]

# Dataloader declaration
train_loader = DataLoader(train_graph_list, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graph_list, batch_size=batch_size)
test_loader = DataLoader(test_graph_list, batch_size=batch_size)

if model_type == "baseline":
    model = BaselineModel(hidden_channels=hidd, input_size=torch_split['x_train'].shape[1], out_size=1).to(device)
elif model_type == "baseline_cheb":
    model = BaselineModelCheb(hidden_channels=hidd, input_size=torch_split['x_train'].shape[1], out_size=1).to(device)
elif model_type == "baseline_simple":
    model = BaselineModelSimple(hidden_channels=hidd, input_size=torch_split['x_train'].shape[1], out_size=1).to(device)
else:
    raise NotImplementedError

# Print to console model definition
print("The model definition is:")
print(model)

# Define optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

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


# Declare results path
results_path = os.path.join("Results", experiment_name)
# Declare log path
train_log_path = os.path.join(results_path, "TRAINING_LOG.txt")
# Declare metric dicts path
metrics_log_path = os.path.join(results_path, "metric_dicts.pickle")
# Declare path to save performance training plot
train_performance_fig_path = os.path.join(results_path, "training_performance.png")
# Declare path to save val set predictions plot
val_performance_fig_path = os.path.join(results_path, "val_performance.png")

# Create results directory
if not os.path.isdir(results_path):
    os.makedirs(results_path)

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

    # Save checkpoints every 2 epochs
    if (epoch+1) % 2 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss},
            os.path.join(results_path, "checkpoint_epoch_"+str(epoch+1)+".pt"))

# Save metrics dicts
complete_metric_dict = {"train": train_metric_lst,
                        "test": val_metric_lst,
                        "adv_test": adv_val_metric_lst,
                        "loss": loss_list}

with open(metrics_log_path, 'wb') as f:
    pickle.dump(complete_metric_dict, f)

# Generate training performance plot and save it to train_performance_fig_path
plot_training(train_metric_lst, val_metric_lst, adv_val_metric_lst, loss_list, train_performance_fig_path)

# Generate val performance plot and save it to val_performance_fig_path
plot_predictions(model, device, val_loader, val_performance_fig_path)
