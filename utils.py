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

            out = model(input_x, data.edge_index.to(device), data.batch.to(device))  # Perform a single forward pass.
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
            pred = model(input_x, data.edge_index.to(device), data.batch.to(device)).cpu().detach().numpy()
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


def apgd_graph(model, x, y, edge_index, batch, criterion, epsilon=0.01, **kwargs):
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
    optim_x = apgd(model=model, inputs=x, labels=y, edge_index=edge_index, batch_vec=batch,
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
    
    # TODO: Optimize vectorized version of plots and order of gloups to ger non overlaping lines
    # Generate performance plot
    plt.figure(figsize=(30, 10))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(np.arange(total_epochs), plot_data[i,:,0], '-o')
        plt.plot(np.arange(total_epochs), plot_data[i,:,2], '-o')
        plt.plot(np.arange(total_epochs), plot_data[i,:,1], '-o')
        plt.grid()
        plt.xlabel("Epochs", fontsize=20)
        plt.ylabel(metric_names[i], fontsize=20)
        plt.title("Model performance with\n"+metric_names[i], fontsize=25)
        plt.legend(legends)

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
    f.write(p_string + '\n')