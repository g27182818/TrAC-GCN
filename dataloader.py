import os
import time
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import glob
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
import networkx as nx
import pickle as pkl
from biomart import BiomartServer
import json
from utils import *
# Set random seed
np.random.seed(1234)

def read_shokhirev(norm, log2, filter_type = 'none', ComBat = False, ComBat_seq = False):
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
    gene_list : list
        Ordered list of genes from the input matrix.
    """
    # Handle multiple possible errors
    # Handle the case where user defines both ComBat and ComBat_seq as True
    if ComBat and ComBat_seq:
        raise ValueError('ComBat and ComBat_seq cannot be both True')
    if ComBat and (not log2):
        raise ValueError('ComBat requires log2 transform because it was computed over log2(x+1) data')

    # Handle all the cases to assign the correct file name to load
    if ComBat:
        dataset_filename = norm+'_dataset_log2_combat.csv'
    elif ComBat_seq:
        dataset_filename = norm+'_dataset_combat_seq.csv'
    else:
        dataset_filename = norm+'_dataset.csv'

    # Declare expression data path and path to tables containing the filtered genes 
    expression_path = os.path.join("data","Shokhirev_2020","normalized", dataset_filename)
    gene_filtering_path = os.path.join("data","Shokhirev_2020","tables.xlsx")
    # Read expression file and make minor modifications
    expression = pd.read_csv(expression_path, sep=",", header=0)
    expression = expression.rename(columns = {'Unnamed: 0':'SRR.ID'})
    expression = expression.set_index('SRR.ID')
    expression = expression.T

    # filtering type dictionary mapper
    filtering_type_dict = {'1000var':'1000 Variable Gene',
                           '1000diff':'1000 Differential Gene',
                           '100var':'100 Variable Gene',
                           '100diff':'100 Differential Gene'}

    if filter_type == 'none':
        # Obtain complete gene list
        gene_names = expression.columns.tolist()
    else:
        # Read gene filtering tables
        gene_filtering_xlsx = pd.read_excel(gene_filtering_path, sheet_name = "Table S5 (Related to Fig. 3)", header=3)
        # Obtain filtered gene series
        gene_names_series = gene_filtering_xlsx[filtering_type_dict[filter_type]]
        # Obtain filtered gene list
        gene_names = gene_names_series.tolist()
        # Filter expression data to keep only the filtered genes
        expression = expression.loc[:, expression.columns.isin(gene_names)]

    # Handle posible log2 transformation. In case ComBat is True, the expression matrix is already log2
    # transformed from the loaded file.
    if log2 and (not ComBat):
        # Perform the log2 transform of data
        expression = np.log2(expression + 1)

    # Declare mateadata path
    meta_path = os.path.join("data","Shokhirev_2020","meta_filtered.txt")
    # Read metadata file and make minnor modifications
    meta = pd.read_csv(meta_path, sep="\t", header=0)
    del meta['Unnamed: 0']
    meta = meta.set_index('SRR.ID')

    # Get ordered list of genes
    gene_names = expression.columns.tolist()
    # Declare numpy variables
    x_np = expression.values
    y_np = meta['Age'].values

    # Shuffle variables 
    np.random.seed(1234)
    shuffler = np.random.permutation(len(x_np))
    x_np = x_np[shuffler]
    y_np = y_np[shuffler]

    return x_np, y_np, gene_names

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
                 'x_val':   x[n-num_val-num_test:n-num_test, :],
                 'x_test':  x[n-num_test:, :],
                 'y_train': y[0:n-num_val-num_test],
                 'y_val':   y[n-num_val-num_test:n-num_test],
                 'y_test':  y[n-num_test:]}
    return data_dict

def compute_graph_shokirev(x, corr_thr, norm, log2, p_thr=0.05, filter_type = 'none',
                           ComBat = False, ComBat_seq = False, force_compute=False):
    """
    This function computes and saves the edge indices and edge atributes of the graph asociated with 
    the gene expression matrix x. In this graph if the Spearman correlation (in absolute value) between
    any two genes is greater than coor_thr with a p_value of less than p_thr, a link of weight 1 is 
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
    filter_type : str, optional
        Type of gene filtering applied to the genes from Shokhirev analysis. It is not used for anything but to save/load. 
        Default is 'none'.
    ComBat : bool, optional
        Indicates if a dataset with ComBat batch correction was loaded, by default False.
    ComBat_seq : bool, optional
        Indicates if a dataset with ComBat_seq batch correction was loaded, by default False.
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
    dir = os.path.join('graphs', 'shokhirev', norm, 'log2='+str(log2), 'ComBat='+str(ComBat),
                       'ComBat_seq='+str(ComBat_seq), 'filter_type='+filter_type)
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

# Function to get a graph using srting data
def get_string_graph(path, gene_list, conf_thr = 0.0, channels:list = ['combined_score']):
    """
    Get a graph from string dtabase matrix thresholded by the conf_thr and with the channels specified.
    Parameters
    ----------
    path : str
        Path to the folder containing string data in txt format.
    gene_list : list
        List of genes to be included in the graph. This list comes from the read_shokhirev() function. It is ordered.
    """
    edge_indices = []
    edge_attributes = []

    # Check if string hugo mapped string data exists
    if not os.path.exists(os.path.join(path, 'string_hugo_mapped.txt')):
        # Check if json file exists
        if not os.path.exists(os.path.join(path,'ensp_2_hugo_mapper.json')):
            # Get Biomart server
            server = BiomartServer( "http://uswest.ensembl.org/biomart" )
            # Get human dataset
            dataset = server.datasets["hsapiens_gene_ensembl"]
            # Search for ENSPs and get the corresponding Hugo names
            response = dataset.search({'attributes': ['ensembl_peptide_id', 'hgnc_symbol']})
            # save response to ensp_2_hugo_mapper dict
            ensp_2_hugo_mapper = {}
            # Iter over response
            for line in response.iter_lines():
                line = line.decode('utf-8')
                content = line.split("\t")
                if content[0] != '' and content[1] != '':
                    ensp_2_hugo_mapper[content[0]] = content[1]
            # Save dict as json file
            with open(os.path.join('data', 'string','ensp_2_hugo_mapper.json'), 'w') as f:
                json.dump(ensp_2_hugo_mapper, f, indent=4)
        else:
            # Load json file
            with open(os.path.join('data', 'string','ensp_2_hugo_mapper.json'), 'r') as f:
                ensp_2_hugo_mapper = json.load(f)
        
        # Open STRING txt file with pandas
        # TODO: Make an automatic download of the original string file
        string_df = pd.read_csv(os.path.join('data', 'string','9606.protein.links.detailed.v11.5.txt'), sep=' ')

        # delete '9606.' from values in columns `protein1` and `protein2`
        string_df['protein1'] = string_df['protein1'].str.replace('9606.', '')
        string_df['protein2'] = string_df['protein2'].str.replace('9606.', '')

        # Delete rows with protein1 values that are not in ensp_2_hugo_mapper keys
        string_df = string_df[string_df['protein1'].isin(ensp_2_hugo_mapper.keys())]
        # Delete rows with protein2 values that are not in ensp_2_hugo_mapper keys
        string_df = string_df[string_df['protein2'].isin(ensp_2_hugo_mapper.keys())]

        # Map protein1 and protein2 values to hugo names
        string_df['protein1'] = string_df['protein1'].map(ensp_2_hugo_mapper)
        string_df['protein2'] = string_df['protein2'].map(ensp_2_hugo_mapper)

        # Get probability scores from STRING
        string_df.loc[:, ~ string_df.columns.isin(['protein1', 'protein2'])] = string_df.loc[:, ~ string_df.columns.isin(['protein1', 'protein2'])]/1000 

        # Create merged column with protein1 and protein2 values
        string_df['p12'] = string_df['protein1']+','+string_df['protein2']

        # Compute mean of duplicates in merged column
        string_df = string_df.groupby('p12').mean()
        
        # Put merged column back in dataframe
        string_df = string_df.reset_index()
        # Split merged column to get protein1 and protein2 values
        string_df['protein1'] = string_df['p12'].str.split(',').str[0]
        string_df['protein2'] = string_df['p12'].str.split(',').str[1]
        
        # Delete merged column
        string_df = string_df.drop('p12', axis=1)

        # Save mapped dataframe to txt file
        string_df.to_csv(os.path.join(path,'string_hugo_mapped.txt'), sep=' ', index=False)

    # If hugo mapped string data exists, load it
    else:
        string_df = pd.read_csv(os.path.join(path,'string_hugo_mapped.txt'), sep=' ')

    print('Loading STRING graph for the dataset...')
    # Filter string dataframe to keep only the genes in the gene_list
    string_df = string_df[string_df['protein1'].isin(gene_list)]
    string_df = string_df[string_df['protein2'].isin(gene_list)]

    # Filter string_df rows by the conf_thr
    string_df = string_df[string_df['combined_score'] > conf_thr]

    # make mapper for gene indexes
    gene_index_mapper = {gene: i for i, gene in enumerate(gene_list)}

    # Get index of genes in the gene_list for string_df['protein1'] and string_df['protein2']
    string_df['protein1_index'] = string_df['protein1'].map(gene_index_mapper)
    string_df['protein2_index'] = string_df['protein2'].map(gene_index_mapper)

    # string_df['protein1_index'] and string_df['protein2_index'] contain now graph edges in both directions
    # Now we declare the edges as tensor and edge_attributes as a tensor
    edges_np = np.stack([string_df['protein1_index'], string_df['protein2_index']], axis=0)
    attributes_np = string_df[channels].values
    edge_indices = torch.tensor(edges_np, dtype=torch.long)
    edge_attributes = torch.tensor(attributes_np)

    # Compute graph features with networkx
    print('Computing graph features with networkx...')
    edges_nx = sorted([(int(edges_np[0, i]), int(edges_np[1, i])) for i in range(edges_np.shape[1])])
    nx_graph = nx.from_edgelist(edges_nx)
    length_connected = [len(c) for c in sorted(nx.connected_components(nx_graph), key=len, reverse=False)]
    print('Total amount of nodes: ' + str(len(gene_list)))
    print('Total amount of edges: ' + str(edge_attributes.shape[0]))
    print('Average graph degree: ' + str(round(edge_attributes.shape[0]/len(gene_list), 3)))
    print('Biggest connected component size: ' + str(length_connected[-1]))
    return edge_indices, edge_attributes


def load_dataset(norm, log2, val_frac = 0.2, test_frac = 0.2, corr_thr=0.6, p_thr=0.05, force_compute=False,
                filter_type = 'none', ComBat = False, ComBat_seq = False, string = False, conf_thr = 0.0,
                channels_string = ['combined_score']):
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
    ComBat : bool, optional
        Whether to load ComBat batch corrected dataset, by default False.
    ComBat_seq : bool, optional
        Whether to load ComBat_seq batch corrected dataset, by default False.
    string : bool, optional
        Whether to load STRING data into the graph. In case it is true corr_thr and p_thr are ignored,
        by default False.
    conf_thr : float, optional
        Confidence threshold to filter STRING edges. It is just used when string==True, by default 0.0.
    channels_string : list, optional
        List of string channels to be used for STRING data. Can be a compination of: combined_score, textmining, database,
        experimental, coexpression, cooccurence, fusion, neighborhood. By default ['combined_score'].

    Returns
    -------
    dataset_info : dict
        Dictionary containing all the important data from the training setup with the following keys:
            dataset_info['split']: Dictionary containning dataset partitions for both samples and annotations.
                                   It is in the format specified by the split_data() function.
            dataset_info['graph']: Tuple containing (edge_indices, edge_attributes) that define the correlation
                                   graph defined by the parameters in the training set. 
    """
    # Get x_np and y_np from read_shokhirev()
    print('Reading, transforming and splitting data...')
    x_np, y_np, gene_names = read_shokhirev(norm, log2, filter_type = filter_type, ComBat = ComBat, ComBat_seq = ComBat_seq)

    # Split dataset using split_data()
    split_dict = split_data(x_np, y_np, val_frac = val_frac, test_frac = test_frac)

    if string:
        edge_indices, edge_attributes = get_string_graph(os.path.join('data', 'string'), gene_names,
                                                     conf_thr = conf_thr, channels = channels_string)
    else:
        # Use Train x_p to compute or load the graph with compute_graph_shokirev()
        edge_indices, edge_attributes = compute_graph_shokirev(split_dict['x_train'], corr_thr= corr_thr,
                                                            norm=norm, log2=log2, p_thr=p_thr,
                                                            filter_type = filter_type,
                                                            ComBat=ComBat, ComBat_seq=ComBat_seq,
                                                            force_compute=force_compute)
    # Append everything in a single dictionary
    dataset_info = {'split': split_dict,
                    'graph': (edge_indices, edge_attributes),
                    'gene_names': gene_names}
    return dataset_info

# # Test of the complete pipeline
# test_dataset_info = load_dataset(norm='raw', log2=True, string=True, conf_thr=0.9)
