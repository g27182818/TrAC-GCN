import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import glob
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
import networkx as nx
import pickle as pkl
from utils import *
# Set random seed
np.random.seed(1234)

def read_shokhirev(norm, log2):
    """
    This function loads one of 3 csv files of the Shokirev dataset proposed in DOI: 10.1111/acel.13280
    it performs a log2 transform if it is indicated and shuffles the data.

    Parameters
    ----------
    norm : str
        Normalization method used for raw counts. Can be raw, TPM or TMM.
    log2 : bool
        Whether to perform a log2(x+1) transform on input data.

    Returns
    -------
    x_np : np.array
        The shuffled matrix of gene expression in the specified normalization. Rows are samples, columns are genes.
    y_np : np.array
        The shuffled vector of patient age.
    """
    # Declare expression data path 
    expression_path = os.path.join("data","Shokhirev_2020","normalized", norm+'_dataset.csv')
    # Read expression file and make minnor modifications
    expression = pd.read_csv(expression_path, sep=",", header=0)
    expression = expression.rename(columns = {'Unnamed: 0':'SRR.ID'})
    expression = expression.set_index('SRR.ID')
    expression = expression.T

    # Handle posible log2 transformation
    if log2:
        # Perform the log2 transform of data
        expression = np.log2(expression + 1)

    # Declare mateadata path
    meta_path = os.path.join("data","Shokhirev_2020","meta_filtered.txt")
    # Read metadata file and make minnor modifications
    meta = pd.read_csv(meta_path, sep="\t", header=0)
    del meta['Unnamed: 0']
    meta = meta.set_index('SRR.ID')

    # Declare numpy variables
    x_np = expression.values
    y_np = meta['Age'].values

    # Shuffle variables 
    np.random.seed(1234)
    shuffler = np.random.permutation(len(x_np))
    x_np = x_np[shuffler]
    y_np = y_np[shuffler]

    return x_np, y_np

def split_data(x, y, val_frac = 0.2, test_frac = 0.2):
    """
    This function splits the data in train, validation, and test. Test and validation fractions can be 
    changed by parameters. It returns a dictionary with the dataset partitions. 

    Parameters
    ----------
    x : Str
        Matrix of gene expression. Rows are samples, columns are genes.
    y : bool
        Vector of patient age.
    val_frac : float, optional
        Fraction of validation data in the complete dataset, by default 0.2.
    test_frac : float, optional
        Fraction of test data in the complete dataset, by default 0.2.

    Returns
    -------
    data_dict: dict
        Python dictionary containing the following entries:
            data_dict['x_train']: Train matrix.
            data_dict['x_val']: Validation matrix.
            data_dict['x_test']: Test matrix.
            data_dict['y_train']: Train annotations.
            data_dict['y_val']: Validation annotations.
            data_dict['y_test']: Test annotations.
    """
    # Define the complete dataset size
    n = x.shape[0]
    # Define number of samples asigned to each group
    num_val = int(n*val_frac)
    num_test = int(n*test_frac)
    # Declare dictionary
    data_dict = {'x_train': x[0:n-num_val-num_test, :],
                 'x_val':   x[num_val:n-num_test, :],
                 'x_test':  x[n-num_test:, :],
                 'y_train': y[0:n-num_val-num_test],
                 'y_val':   y[num_val:n-num_test],
                 'y_test':  y[n-num_test:]}
    return data_dict

def compute_graph_shokirev(x, corr_thr, norm, log2, p_thr=0.05, force_compute=False):
    """
    This function computes and saves the edge indices and edge atributes of the graph asociated with 
    the gene expression matrix x. In this graph if the Spearman correlation (in absolute value) between
    any two genes is greater than coor_thr with a p_value of less than p_thr, a link with weight 1 is 
    stablished.

    If the graph has already been computed, the function loads the information from file.

    Parameters
    ----------
    x : np.array
        Matrix of m samples (rows) and n genes (columns) containing the gene expression information
    corr_thr : float
        Correlation threshold to define graph edges.
    norm : str
        Normalization type previously applied in input data. It is not used for anything but to save/load
        the graph information.
    log2 : bool
        Indicates if Log2(x+1) transform was applied over input data. It is not used for anything but to save/load
        the graph information.
    p_thr : float, optional
        P_value threshold to define the graph edges, by default 0.05.
    force_compute : bool, optional
        Forces the computation of a new graph even if there is an already existing file with the corresponding name, 
        by default False.

    Returns
    -------
    edge_indices: torch.Tensor
        2D pytorch array that defines the edges between nodes of the graph.
    edge_attributes: torch.Tensor
        Weights associated to the previously defined edges.
    """
    # Define dir, graph and info names
    dir = os.path.join('graphs', 'shokhirev', norm, 'log2='+str(log2))
    name_graph = os.path.join(dir, 'graph_corr_thr_'+str(corr_thr)+'_p_thr_'+str(p_thr)+'.pkl')
    name_info = os.path.join(dir, 'graph_info_corr_thr_'+str(corr_thr)+'_p_thr_'+str(p_thr)+'.txt')
    
    # Handle load of previous info
    if glob.glob(name_graph) and (not force_compute):
        print('Loading graph and info from: \n'+name_graph+'\n'+name_info)
        edge_indices, edge_attributes = pd.read_pickle(name_graph)
        # Read info from text file and print it in terminal
        with open(name_info, 'r') as f:
            info = f.readlines()
            [print(line[:-1]) for line in info]

    # If there is no file or force_compute == True compute a new graph an info
    else:
        print('Computing graph and info to save in ' + dir)
        # Create save directories
        try:
            os.makedirs(os.path.join(dir))
        except:
            print('Directory: ' + dir + ' already exists...')
        # Compute correlation graph
        correlation, p_value = spearmanr(x)                                                 # Compute Spearman correlation
        init_adjacency_matrix = np.abs(correlation) > corr_thr                              # Filter based on the correlation threshold
        valid_p_value = p_value < p_thr                                                     # Filter based in p_value threshold
        adjacency_matrix = np.logical_and(init_adjacency_matrix, valid_p_value).astype(int) # Compose both filters
        adjacency_matrix = adjacency_matrix - np.eye(adjacency_matrix.shape[0], dtype=int)  # Discard self loops
        adjacency_sparse = coo_matrix(adjacency_matrix)                                     # Pass to sparse matrix
        adjacency_sparse.eliminate_zeros()                                                  # Delete zeros from representation
        edge_indices, edge_attributes = from_scipy_sparse_matrix(adjacency_sparse)          # Get edges and weights in tensors
        # Saving graph files 
        with open(os.path.join(name_graph), 'wb') as f:
            pkl.dump((edge_indices, edge_attributes), f)

        # Computing graph characteristics
        print('Computing graph info...')
        nx_graph = nx.from_scipy_sparse_matrix(adjacency_sparse)
        connected_bool = nx.is_connected(nx_graph)
        length_connected = [len(c) for c in sorted(nx.connected_components(nx_graph), key=len, reverse=False)]
        with open(name_info, 'w') as f:
            print_both('Total amount of nodes: ' + str(x.shape[1]), f)
            print_both('Total amount of edges: ' + str(edge_attributes.shape[0]), f)
            print_both('Average graph degree: ' + str(round(edge_attributes.shape[0]/x.shape[1], 3)), f)
            print_both('Is the graph connected: ' + str(connected_bool), f)
            print_both('List of connected componnents size: ' + str(sorted(length_connected, reverse=True)), f)

    return edge_indices, edge_attributes

def load_dataset(norm, log2, val_frac = 0.2, test_frac = 0.2, corr_thr=0.6, p_thr=0.05, force_compute=False):
    """
    This function loads a the complete Shokhirev dataset (DOI: 10.1111/acel.13280) for transcriptomic age regression
    It performs data shuffle, splits and defines a graph based on the Spearman correlation between genes.   

    Parameters
    ----------
    norm : str
        Normalization method used for raw counts in input data. Can be raw, TPM or TMM.
    log2 : bool
        Whether to perform a log2(x+1) transform on input data.
    val_frac : float, optional
        Fraction of validation data in the complete dataset, by default 0.2.
    test_frac : float, optional
        Fraction of test data in the complete dataset, by default 0.2.
    corr_thr : float, optional
        Correlation threshold to define graph edges, by default 0.6.
    p_thr : float, optional
        P_value threshold to define the graph edges, by default 0.05.
    force_compute : bool, optional
        Forces the computation of a new graph even if there is an already existing file with the corresponding name, 
        by default False.

    Returns
    -------
    dataset_info : dict
        Dictionary containing all the important data from the training setup with the following keys:
            dataset_info['split']: Dictionary containning dataset partitions for both samples and annotations
                                   It is in the format specified by the split_data() function.
            dataset_info['graph']: Tuple containing (edge_indices, edge_attributes) that define the correlation
                                   graph defined by the parameters in the training set. 
    """
    # Get x_np and y_np from read_shokhirev()
    print('Reading, transforming and splitting data...')
    x_np, y_np = read_shokhirev(norm, log2)
    # Split dataset using split_data()
    split_dict = split_data(x_np, y_np, val_frac = val_frac, test_frac = test_frac)
    # Use Train x_p to compute or load the graph with compute_graph_shokirev()
    edge_indices, edge_attributes = compute_graph_shokirev(split_dict['x_train'], corr_thr= corr_thr,
                                                           norm=norm, log2=log2, p_thr=p_thr,
                                                           force_compute=force_compute)
    # Append everything in a single dictionary
    dataset_info = {'split': split_dict,
                    'graph': (edge_indices, edge_attributes)}
    return dataset_info

