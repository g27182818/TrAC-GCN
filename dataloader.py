import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Set random seed
np.random.seed(1234)

def load_shokhirev(norm, log2):
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

def split_data(norm, log2, val_frac = 0.2, test_frac = 0.2):
    """
    This function splits the data from the dataset of Shokhirev (DOI: 10.1111/acel.13280) in train, validation,
    and test. Test and validation fractions can be changed by parameters. It can specify the normalization and
    whther to perform a log2(x+1) transformation over the data (It uses the load_shokhirev() function). It
    returns a dictionary with the dataset partitions. 

    Parameters
    ----------
    norm : Str
        Normalization method used for raw counts. Can be raw, TPM or TMM.
    log2 : bool
        Whether to perform a log2(x+1) transform on input data.
    val_frac : float, optional
        Fraction of validation data in the complete dataset, by default 0.2
    test_frac : float, optional
        Fraction of test data in the complete dataset, by default 0.2

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
    # TODO: Compute the split dictionary
    data_dict = {}
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
    edge_indices: np.array
        2D numpy array that defines the edges between nodes of the graph.
    edge_attributes: np.array
        Weights associated to the previously defined edges.
    """
    # TODO: Define graph name
    # TODO: Handle compute/load structure
    # TODO: Compute correlation graph
    edge_indices = 0
    edge_attributes = 0
    return edge_indices, edge_attributes

