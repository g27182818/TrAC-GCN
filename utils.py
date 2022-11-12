import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import seaborn as sn
import pandas as pd
from AutoPGD.auto_pgd import apgd
from tqdm import tqdm
import matplotlib.colors as colors
import argparse

def get_main_parser():

    parser = argparse.ArgumentParser(description='Code for TrAC-GCN implementation.')
    # Dataset parameters #####################################################################################################################################################################
    parser.add_argument('--norm',           type=str,       default="tpm",          help='The normalization method to be loaded via files.',                                                            choices=['raw', 'tpm', 'tmm'])
    parser.add_argument('--log2',           type=str,       default='True',         help='Parameter indicating if a log2 transformation is done under input data.',                                     choices=['True', 'False'])
    parser.add_argument('--ComBat',         type=str,       default='False',        help='Parameter indicating if a dataset with ComBat batch correction is loaded. Can be True just if log2 = True.' , choices=['True', 'False'])
    parser.add_argument('--ComBat_seq',     type=str,       default='True',         help='Parameter indicating if a dataset with ComBat_seq batch correction is loaded.',                               choices=['True', 'False'])
    parser.add_argument('--filter_type',    type=str,       default='none',         help='Filtering to be applied to genes specified by string',                                                        choices= ['1000var', '1000diff', '100var', '100diff'])
    parser.add_argument('--val_frac',       type=float,     default=0.2,            help='The fraction of all samples in the validation group. Must be in the range (0, 1).')
    parser.add_argument('--test_frac',      type=float,     default=0.2,            help='The fraction of all samples in the test group. Must be in the range (0, 1).')
    # Graph parameters ######################################################################################################################################################################
    parser.add_argument('--string',         type=str,       default='False',        help='Parameter indicating if the graph made using STRING database.',                                               choices=['True', 'False'])
    parser.add_argument('--all_string',     type=str,       default='False',        help='Parameter indicating if all STRING channels should be used otherwise combined_score will be used.',           choices=['True', 'False'])
    parser.add_argument('--conf_thr',       type=float,     default=0.7,            help='The confidence threshold to stablish connections in STRING graphs.')
    parser.add_argument('--corr_thr',       type=float,     default=0.8,            help='The correlation threshold to be used for defining graph connectivity in coexpression graphs.')
    parser.add_argument('--p_thr',          type=float,     default=0.05,           help='The p value threshold to be used for defining graph connectivity in coexpression graphs.')
    # Model parameters ######################################################################################################################################################################
    parser.add_argument('--model',          type=str,       default='baseline',     help='The model to be used.',                                                                                       choices= ['baseline', 'deepergcn', 'MLR', 'MLP', 'holzscheck_MLP', 'wang_MLP', 'baseline_pool', 'graph_head', 'trac_gcn', 'DFS'] )
    parser.add_argument('--hidden_chann',   type=int,       default=8,              help='The number of hidden channels to use in the graph based models.')
    parser.add_argument('--dropout',        type=float,     default=0.0,            help='Dropout rate to be used in models. Must be in the range (0, 1).')
    parser.add_argument('--final_pool',     type=str,       default='none',         help='Final pooling type over nodes to be used in graph based models.',                                             choices= ['mean', 'max', 'add', 'none'])
    # Training parameters ###################################################################################################################################################################
    parser.add_argument('--exp_name',       type=str,       default='misc_test',    help='Experiment name to be used for saving files. If set to -1 the name will be generated automatically.')
    parser.add_argument('--loss',           type=str,       default='mse',          help='Loss function to be used for training.',                                                                      choices=['l1', 'mse'])
    parser.add_argument('--lr',             type=float,     default=0.0005,         help='Learning rate for training.')
    parser.add_argument('--epochs',         type=int,       default=100,            help='Number of epochs for training.')
    parser.add_argument('--batch_size',     type=int,       default=20,             help='Batch size for training.')
    parser.add_argument('--adv_e_test',     type=float,     default=0.00,           help='Adversarial upper bound of perturbations during test.')
    parser.add_argument('--adv_e_train',    type=float,     default=0.00,           help='Adversarial upper bound of perturbations during train.')
    parser.add_argument('--n_iters_apgd',   type=int,       default=50,             help='Number of iterations for APGD during train.')
    
    return parser


def train(train_loader, model, device, criterion, optimizer, adversarial=False, attack=None, **kwargs):
    """
    This function performs 1 training epoch in a graph classification model with the possibility of adversarial
    training using the attach function.
    :param train_loader: (torch.utils.data.DataLoader) Pytorch dataloader containing training data.
    :param model: (torch.nn.Module) The prediction model.
    :param device: (torch.device) The CUDA or CPU device to parallelize.
    :param criterion: (torch.nn loss function) Loss function to optimize (For this task CrossEntropy is used).
    :param optimizer: (torch.optim.Optimizer) The model optimizer to minimize the loss function.
    :param adversarial: (bool) Parameter indicating to perform an adversarial attack (Default = False).
    :param attack: The adversarial attack function (Default = None).
    :param kwargs: Keyword arguments of the attack function
    :return: mean_loss: (torch.Tensor) The mean value of the loss function over the epoch.
    """
    # Put model in train mode
    model.train()
    # Start the mean loss value
    mean_loss = 0
    # Start a counter
    count = 0
    with tqdm(train_loader, unit="batch") as t_train_loader:
        # Training cycle over the complete training batch
        for data in t_train_loader:  # Iterate in batches over the training dataset.
            t_train_loader.set_description(f"Batch {count+1}")
            # Get the inputs of the model (x) and the groundtruth (y)
            input_x, input_y = data.x.to(device), data.y.to(device)
            # Handle the adversarial attack
            if adversarial:
                delta = attack(model,
                               input_x,
                               input_y,
                               data.edge_index.to(device),
                               data.batch.to(device),
                               criterion,
                               **kwargs)
                optimizer.zero_grad()
                # Obtain adversarial input
                input_x = input_x + delta

            out = model(input_x, data.edge_index.to(device), data.edge_attributes.to(device), data.batch.to(device))  # Perform a single forward pass.
            loss = criterion(out, input_y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            mean_loss += loss
            count += 1

            # Update terminal descriptor
            t_train_loader.set_postfix(loss=loss.item())

    mean_loss = mean_loss/count
    return mean_loss


def test(loader, model, device, optimizer=None, adversarial=False, attack=None, criterion=None, **kwargs):
    """
    This function calculates MAE (Mean Absolute Error), RMSE (Root Mean Square Error) and R^2 (Determination coefficient) 
    for any regression problem that consists of graph input data. This function can also test an adversarial attack on
    the inputs.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Pytorch dataloader containing data to test.
    model : torch.nn.Module
        The prediction model.
    device : torch.device
        The CUDA or CPU device to parallelize.
    optimizer : torch.optim.Optimizer, optional
        The model optimizer to delete gradients after the adversarial attack, by default None.
    adversarial : bool, optional
        Whether to perform an adversarial attack during test, by default False.
    attack : _type_, optional
        The adversarial attack function, by default None.
    criterion : _type_, optional
        Loss function to optimize the adversarial attack in case adversarial==True, by default None.
    **kwargs: _type_, optional
        Keyword arguments of the attack function. 
    Returns
    -------
    metric_result: Dict
        Dictionary containing the metric results:
                            metric_result['MAE']: Mean Absolute error of the data in loader evaluationg with model.
                            metric_result['RMSE']: Root mean square error of the data in loader evaluationg with model.
                            metric_result['R^2']: Coefficient of determination R^2 of the data in loader evaluationg with model.
    """
    # Put model in evaluation mode
    model.eval()

    # Global true tensor
    glob_true = np.array([])
    # Global probability tensor
    glob_pred = np.array([])

    count = 1
    # Computing loop
    with tqdm(loader, unit="batch") as t_loader:
        for data in t_loader:  # Iterate in batches over the training/test dataset.
            t_loader.set_description(f"Batch {count}")
            # Get the inputs of the model (x) and the groundtruth (y)
            input_x, input_y = data.x.to(device), data.y.to(device)
            # Handle the adversarial attack
            if adversarial:
                delta = attack(model,
                               input_x,
                               input_y,
                               data.edge_index.to(device),
                               data.batch.to(device),
                               criterion,
                               **kwargs)
                optimizer.zero_grad()
                input_x = input_x+delta

            # Get the model predictions
            pred = model(input_x, data.edge_index.to(device), data.edge_attributes.to(device), data.batch.to(device)).cpu().detach().numpy()
            true = input_y.cpu().numpy()
            # Stack cases with previous ones
            glob_pred = np.hstack([glob_pred, pred]) if glob_pred.size else pred
            glob_true = np.hstack((glob_true, true)) if glob_true.size else true
            # Update counter
            count += 1

    # Results dictionary declaration and metrics computation
    metric_result = {'MAE' : sklearn.metrics.mean_absolute_error(glob_true, glob_pred),
                     'RMSE': sklearn.metrics.mean_squared_error(glob_true, glob_pred, squared=False),
                     'R^2' : sklearn.metrics.r2_score(glob_true, glob_pred)}
    return metric_result

def test_and_get_attack(loader, model, device, optimizer=None, attack=None, criterion=None, **kwargs):
    
    # Put model in evaluation mode
    model.eval()

    # Global true tensor
    glob_true = np.array([])
    # Global probability tensor
    glob_pred = np.array([])
    # Global delta array
    glob_delta = np.array([])

    count = 1
    # Computing loop
    with tqdm(loader, unit="batch") as t_loader:
        for data in t_loader:  # Iterate in batches over the training/test dataset.
            t_loader.set_description(f"Batch {count}")
            # Get the inputs of the model (x) and the groundtruth (y)
            input_x, input_y = data.x.to(device), data.y.to(device)
            # Handle the adversarial attack
            delta = attack(model, input_x, input_y, data.edge_index.to(device), data.edge_attributes.to(device), data.batch.to(device), criterion, **kwargs)
            optimizer.zero_grad()
            input_x = input_x+delta

            # Get the model predictions
            pred = model(input_x, data.edge_index.to(device), data.edge_attributes.to(device), data.batch.to(device)).cpu().detach().numpy()
            true = input_y.cpu().numpy()
            # Stack cases with previous ones
            glob_pred = np.hstack([glob_pred, pred]) if glob_pred.size else pred
            glob_true = np.hstack((glob_true, true)) if glob_true.size else true
            # Stack deltas with previous ones
            delta = torch.reshape(delta, (data.batch.max()+1, -1))
            glob_delta = np.vstack([glob_delta, delta.cpu().detach().numpy()]) if glob_delta.size else delta.cpu().detach().numpy()
            # Update counter
            count += 1

    # Results dictionary declaration and metrics computation
    metric_result = {'MAE' : sklearn.metrics.mean_absolute_error(glob_true, glob_pred),
                     'RMSE': sklearn.metrics.mean_squared_error(glob_true, glob_pred, squared=False),
                     'R^2' : sklearn.metrics.r2_score(glob_true, glob_pred)}

    return metric_result, glob_delta, glob_true, glob_pred            

def pgd_linf(model, X, y, edge_index, edge_attributes, batch, criterion, epsilon=0.01, alpha=0.001, n_iter=20, randomize=True):
    # Handle starting point randomization
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    # Optimization cycle of delta
    for t in range(n_iter):
        loss = criterion(model(X + delta, edge_index, edge_attributes, batch), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()



# TODO: Adapt this kind of attack to work with a regression problem
def apgd_graph(model, x, y, edge_index, edge_attributes, batch, criterion, epsilon=0.01, **kwargs):
    """
    Construct AutoPGD adversarial examples in L_inf ball over the examples X (IMPORTANT: It returns the perturbation
    (i.e. delta)). This is a slight modification of the AutoPGD implementation presented in
    https://github.com/jeromerony/adversarial-library adjusted to recieve graph input.
    :param model: (torch.nn.Module) Classification model to construct the adversarial attack.
    :param x: (torch.Tensor) The model inputs. Node features.
    :param y: (torch.Tensor) Groundtruth classification of X.
    :param edge_index: (torch.Tensor) Node conections of the input graph expected by the model. They come in the form of
                        an adjacency list.
    :param batch: (torch.Tensor) The batch vector specifying node correspondence to each complete graph on the batch.
    :param criterion: (torch.optim.Optimizer) Loss function to optimize the adversarial attack (For this task
                       CrossEntropy is used).
    :param epsilon: (float) Radius of the L_inf ball to compute the perturbation (Default = 0.01).
    :return: delta: (torch.Tensor) The optimized perturbation to the input X. This perturbarion is returned for the
                    complete batch.
    """
    # Perform needed reshape
    x = torch.reshape(x, (torch.max(batch).item() + 1, -1))
    # Compute the imported apgd function
    optim_x = apgd(model=model, inputs=x, labels=y, edge_index=edge_index, edge_attributes=edge_attributes, batch_vec=batch,
                   give_crit=True, crit=criterion, eps=epsilon, norm=float('inf'), **kwargs)
    # Obtain perturbation
    delta = optim_x-x
    # Final reshape
    delta = torch.reshape(delta, (-1, 1))
    return delta.detach()

def plot_training(train_list, val_list, adversarial_val_list, loss, save_path):
    # TODO: Update docstring of plot_training() function
    """
    This function plots a 2X1 figure. The left figure has the training performance in train, test, and adversarial test
    measured by mACC, mAP or both. The rigth figure has the evolution of the mean training loss over the epochs.
    :param metric: (str) The metric to avaluate the performance can be 'acc', 'mAP' or 'both'.
    :param train_list: (dict list) List containing the train metric dictionaries acording to the test() function. One
                        value per epoch.
    :param val_list: (dict list) List containing the validation metric dictionaries acording to the test() function. One
                        value per epoch.
    :param adversarial_val_list: (dict list) List containing the adversarial test metric dictionaries acording to the
                                  test() function. One value per epoch.
    :param loss: (list) Training loss value list. One value per epoch.
    :param save_path: (str) The path to save the figure.
    """
    # Number of training epochs
    total_epochs = len(loss)

    # The order of plot_data is:
    # Index 0: 3 metrics (MAE, RMSE, R^2)
    # Index 1: len(train_list) epochs
    # Index 2: 2 evaluation groups (Train, Val, adversarial val) 
    plot_data = np.zeros((3, len(train_list), 3))
    metric_names = ['MAE', 'RMSE', 'R^2']
    # plot_data assignation from dictionaries
    for i in range(total_epochs):
        for j in range(len(metric_names)):
            plot_data[j, i, 0] = train_list[i][metric_names[j]]
            plot_data[j, i, 1] = val_list[i][metric_names[j]]
            plot_data[j, i, 2] = adversarial_val_list[i][metric_names[j]]
    # Legends of plots
    legends = ['Train', 'Adv. Val', 'Val']
    
    # Impose mathematical notation for metrics title
    metric_names = ["MAE", "RMSE", "$R^2$"]

    # TODO: Optimize vectorized version of plots and order of gloups to ger non overlaping lines
    # Generate performance plot
    plt.figure(figsize=(20, 6))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(np.arange(total_epochs), plot_data[i,:,0], '-ok')
        plt.plot(np.arange(total_epochs), plot_data[i,:,2], '-ob')
        plt.plot(np.arange(total_epochs), plot_data[i,:,1], '-or')
        plt.grid()
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel(metric_names[i], fontsize=16)
        plt.title("Model performance\nwith "+metric_names[i], fontsize=20)
        plt.legend(legends, fontsize=12)
        if metric_names[i] == "$R^2$":
            plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)


def plot_predictions(model, device, val_loader, save_path):
    """
    This funtion plots the predictions of the model over the val set.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    device : torch.device
        The device to use.
    val_loader : torch.utils.data.DataLoader
        Dataset loader for the validation set.
    save_path : str
        Path to save the plot.
    """
    # Put model in evaluation mode
    model.eval()
    # Global definition of true and predicted ages.
    y_true = np.array([])
    y_pred = np.array([])
    # Cycle to compute the predictions
    for data in val_loader:  # Iterate in batches over the val dataset.
        # Get the inputs of the model (x) and the groundtruth (y)
        input_x, input_y = data.x.to(device), data.y
        # Get the model predictions
        pred = model(input_x, data.edge_index.to(device), data.edge_attributes.to(device), data.batch.to(device)).cpu().detach().numpy()
        true = input_y.numpy()
        # Stack cases with previous ones
        y_pred = np.hstack([y_pred, pred]) if y_pred.size else pred
        y_true = np.hstack((y_true, true)) if y_true.size else true
    plt.figure(figsize=(6, 6))
    plt.plot(y_true, y_pred, 'ok')
    plt.plot(np.arange(0, max(max(y_true), max(y_pred))+1), np.arange(0, max(max(y_true), max(y_pred))+1), '-k')
    plt.ylim((0, max(max(y_true), max(y_pred))))
    plt.xlim((0, max(max(y_true), max(y_pred))))
    plt.grid()
    plt.xlabel("True Age (Yr)", fontsize=16)
    plt.ylabel("Predicted Age (Yr)", fontsize=16)
    plt.title("Model predictions", fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)




def print_epoch(train_dict, val_dict, adv_val_dict, loss, epoch, path):
    # TODO: Update docstring of print_epoch() function
    """
    This function prints in terminal a table with all available metrics in all test groups (train, test, adversarial
    test) for an specific epoch. It also writes this table to the training log specified in path.
    :param train_dict: (Dict) Dictionary containing the train set metrics acording to the test() function.
    :param val_dict: (Dict) Dictionary containing the test set metrics acording to the test() function.
    :param adv_val_dict: (Dict) Dictionary containing the adversarial test set metrics acording to the test() function.
    :param loss: (float) Mean epoch loss value.
    :param epoch: (int) Epoch number.
    :param path: (str) Training log path.
    """
    rows = ["Train", "Val", "Adv. Val"]
    data = np.zeros((3, 1))
    headers = []
    counter = 0

    # Construccion of the metrics table
    for k in train_dict.keys():
        # Handle metrics that cannot be printed
        if (k == "epoch"):
            continue
        headers.append(k)

        if counter > 0:
            data = np.hstack((data, np.zeros((3, 1))))

        data[0, counter] = train_dict[k]
        data[1, counter] = val_dict[k]
        data[2, counter] = adv_val_dict[k]
        counter += 1

    # Declare dataframe to print
    data_frame = pd.DataFrame(data, index=rows, columns=headers)

    # Print metrics to a log and terminal
    with open(path, 'a') as f:
        print_both('-----------------------------------------',f)
        print_both('                                         ',f)
        print_both("Epoch " + str(epoch + 1) + ":",f)
        print_both("Loss = " + str(loss.cpu().detach().numpy()),f)
        print_both('                                         ',f)
        print(data_frame)
        print(data_frame, file=f)
        print_both('                                         ',f)



def read_csv_pgbar(csv_path, chunksize, usecols, dtype=object):
    # print('Getting row count of csv file')
    rows = sum(1 for _ in open(csv_path, 'r')) - 1 # minus the header
    # chunks = rows//chunksize + 1
    # print('Reading csv file')
    chunk_list = []
 
    with tqdm(total=rows, desc='Rows read: ') as bar:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=usecols, dtype=dtype):
            chunk_list.append(chunk)
            bar.update(len(chunk))
 
    df = pd.concat((f for f in chunk_list), axis=0)
    print('Finish reading csv file')

def print_both(p_string, f):
    """
    This function prints p_string in terminal and to a .txt file with handle f 

    Parameters
    ----------
    p_string : str
        String to be printed.
    f : file
        Txt file handle indicating where to print. 
    """
    print(p_string)
    f.write(p_string)
    f.write('\n')