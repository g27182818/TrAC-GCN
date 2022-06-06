# Generic libraries
import numpy as np
import pandas as pd
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse
# Specific imports
from models import *
from dataloader import *
from utils import *
# Set matplotlib option to plot while in screen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(1234)

# # Add optional timmer to delay the execution of the code
# import time
# time.sleep(6000)


# Parser to specify the normalization method to perform analysis #####################################
parser = argparse.ArgumentParser(description='Code for TrAC-GCN test implementation.')
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
parser.add_argument('--model',          type=str,   default='baseline',  help='The model to be used.', choices= ['baseline', 'deepergcn', 'MLR', 'MLP', 'holzscheck_MLP', 'wang_MLP', 'baseline_pool', 'graph_head', 'trac_gcn', 'DFS'] )
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


# Declare results path
results_path = os.path.join("Results", experiment_name)
# Declare path to save best model
best_model_path = os.path.join(results_path, "best_model.pt")
# Declare path to save a_plot 
a_plot_path = os.path.join(results_path, "a_plot.png")
# Declare path to save gene ranking csv
gene_ranking_path = os.path.join(results_path, "gene_ranking.csv")


# Load data
dataset_info = load_dataset(norm=norm, log2=log2_bool, corr_thr=coor_thr, p_thr=p_value_thr, force_compute=False,
                            val_frac=val_fraction, test_frac=test_fraction, filter_type=filter_type,
                            ComBat=ComBat, ComBat_seq=ComBat_seq, string = string, conf_thr = conf_thr,
                            channels_string = channels_string)
# Extract graph information
edge_indices, edge_attributes = dataset_info['graph']
edge_attributes = edge_attributes.type(torch.float)


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

# This handles the model type and the weight decay
if model_type == "baseline":
    model = BaselineModel(hidden_channels=hidd, input_size=torch_split['x_train'].shape[1],
                          out_size=1,
                          dropout=dropout,
                          final_pool=final_pool).to(device)
    weight_decay = 0.0

elif model_type == "graph_head":
    model = GraphHead(hidden_channels=hidd, input_size=torch_split['x_train'].shape[1],
                      out_size=1,
                      dropout=dropout,
                      final_pool=final_pool).to(device)
    weight_decay = 0.01

elif model_type == "trac_gcn":
    model = TracGCN(hidden_channels=hidd, input_size=torch_split['x_train'].shape[1],
                      out_size=1,
                      dropout=dropout,
                      final_pool=final_pool).to(device)
    weight_decay = 0.01

elif model_type == "baseline_pool":
    model = BaselineModelPool(hidden_channels=hidd, input_size=torch_split['x_train'].shape[1],
                              out_size=1,
                              dropout=dropout,
                              final_pool=final_pool,
                              cluster_num=1014).to(device)
    weight_decay = 0.0

elif model_type == "deepergcn":
    model = DeeperGCN(hidden_channels=hidd, input_size=torch_split['x_train'].shape[1],
                      input_node_channels=1,
                      num_layers=5,
                      dropout=dropout,
                      final_pool=final_pool).to(device)
    weight_decay = 0.0

elif model_type == "MLR":
    model = MLP(h_sizes=[torch_split['x_train'].shape[1]], out_size=1, init_weights=None, dropout=dropout).to(device)
    weight_decay = 0.0

elif model_type == "MLP":
    model = MLP(h_sizes=[torch_split['x_train'].shape[1], 1000], out_size=1, init_weights=None, dropout=dropout).to(device)
    weight_decay = 0.0

elif model_type == "holzscheck_MLP":
    model = MLP(h_sizes=[torch_split['x_train'].shape[1], 350, 350, 350, 50],
                out_size=1, act="elu",
                init_weights=torch.nn.init.kaiming_uniform_,
                dropout=dropout).to(device)
    weight_decay = 0.01

elif model_type == "wang_MLP":
    model = MLP(h_sizes=[torch_split['x_train'].shape[1], 256, 256, 32],
                out_size=1, act="relu",
                init_weights=torch.nn.init.kaiming_uniform_,
                dropout=dropout).to(device)
    weight_decay = 5e-4

elif model_type == "DFS":
    model = DFS(h_sizes=[torch_split['x_train'].shape[1], 512, 256, 128],
                out_size=1, act="elu",
                dropout=dropout).to(device)
    weight_decay = 0.01
else:
    raise NotImplementedError

# Print to console model definition
print("The model definition is:")
print(str(model))

# Define optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Handle multiple losses
if loss_fn == 'mse':
    criterion = torch.nn.MSELoss()
elif loss_fn == 'l1':
    criterion = torch.nn.L1Loss()
else:
    raise NotImplementedError

# Load best model dicts
total_saved_dict = torch.load(best_model_path)
model_dict = total_saved_dict['model_state_dict']
optimizer_dict = total_saved_dict['optimizer_state_dict']

# Load state dicts to model and optimizer
model.load_state_dict(model_dict)
optimizer.load_state_dict(optimizer_dict)

# Put model in eval mode
model.eval()

metric_result, glob_delta, glob_true, glob_pred = test_and_get_attack(val_loader, model, device,
                                                                        optimizer=optimizer,
                                                                        attack=pgd_linf, criterion=criterion,
                                                                        epsilon=test_eps, n_iter=20, alpha=0.001)

# abs_sum_pert = np.abs(glob_delta).sum(axis=0)
sum_pert = glob_delta.sum(axis=0)

abs_array = np.abs(sum_pert)
# get sorted indexes in descending order
sorted_indexes = abs_array.argsort()[::-1]
print(sorted_indexes)

gene_names = dataset_info['gene_names']
gene_rank = np.array(gene_names)[sorted_indexes]
sum_pert_ranked = sum_pert[sorted_indexes]

# Print top 10 genes and their perturbations as a prety table
print("Top 10 Predictor Genes:")
[print(gene_rank[i], round(sum_pert_ranked[i], 3)) for i in range(10)]

# Add gene_rank and sum_pert_ranked to pandas dataframe
df_rank = pd.DataFrame(data=np.array([gene_rank, sum_pert_ranked]).T, columns=['gene', 'a'])
# add column at the beginning of the dataframe with the index called rank
df_rank.insert(0, 'rank', df_rank.index + 1)

# Save dataframe to csv
df_rank.to_csv(gene_ranking_path, index=False)


# Make bar plot of the data
plt.figure()
plt.plot(range(len(sum_pert)), sum_pert, 'ok', markersize=2)
plt.grid('on')
plt.xlim(0, len(sum_pert))
plt.xlabel('Gene', fontsize=16)
plt.ylabel('$a($Gene$)$', fontsize=16)
plt.title('$a($Gene$)$', fontsize=20)
plt.show()
plt.savefig(a_plot_path, dpi=300)

