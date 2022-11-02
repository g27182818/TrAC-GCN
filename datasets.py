from importlib.metadata import metadata
import os
import statistics
import time
from unicodedata import numeric
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import glob
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
import pickle as pkl
from biomart import BiomartServer
import json
import copy
from utils import *
# Set random seed
np.random.seed(1234)


class ShokhirevDataset:
    def __init__(self, path,  norm, log2, val_frac = 0.2, test_frac = 0.2, corr_thr=0.6, p_thr=0.05,
                filter_type = 'none', ComBat = False, ComBat_seq = False, batch_sample_thr = 10, exp_frac_thr = 0.1, 
                batch_norm = True, string = False, conf_thr = 0.0, channels_string = ['combined_score'], force_compute=False,
                shuffle_seed = 0):

        # Set all the important attributes of the class 
        self.path = path
        self.norm = norm
        self.log2 = log2
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.corr_thr = corr_thr
        self.p_thr = p_thr
        self.filter_type = filter_type
        self.ComBat = ComBat
        self.ComBat_seq = ComBat_seq
        self.batch_sample_thr = batch_sample_thr
        self.exp_frac_thr = exp_frac_thr
        self.batch_norm = batch_norm
        self.string = string
        self.conf_thr = conf_thr
        self.channels_string = channels_string
        self.force_compute = force_compute
        self.shuffle_seed = shuffle_seed 

        # Check that all the parameters are valid
        self.check_parameters_validity()

        # TODO: Add additional filter for just using healthy data
        # TODO: Add cross validation function after split_dict function

        # Do the main Bioinformatic pipeline
        self.dataset_filename = self.get_dataset_filename()                         # Get the raw dataset filename
        self.x_np, self.y_np, self.gene_names, self.metadata_df =  self.read_data() # Read the raw data. This also performs a log2 transformation if specified.
        self.statistics_dict = self.compute_statistics()                            # Compute o load statistics of complete dataset
        self.batch_filtered_dict  =  self.filter_batches()                          # Filter out batches based in the sample number
        self.exp_frac_filtered_dict = self.filter_exp_frac()                        # Filter out genes by expression fraction. This filtering is made over the batch filtered data. 
        self.batch_normalized_dict = self.batch_normalize()                         # Normalize by batches if specified. If self.batch_norm == False then this function returns self.exp_frac_filtered_dict.
        self.split_dict = self.split_data()                                         # Split and shuffle data
        self.edge_indices, self.edge_attributes = self.get_graph()                  # Compute graph. It will be a co-expression graph or a STRING graph depending on self.string
        # Make dataloader function



    def check_parameters_validity(self):
        """
        Function to check that all the parameters are in a valid combination

        Raises:
            ValueError: If both ComBat and ComBat_seq are selected at the same time
            ValueError: If Combat is selected without log2 transform
        """
        print('Checking parameters...')

        if self.ComBat and self.ComBat_seq:
            raise ValueError('ComBat and ComBat_seq cannot be both True')

        if self.ComBat and (not self.log2):
            raise ValueError('ComBat requires log2 transform because it was computed over log2(x+1) data')

        print('All dataset parameters are valid :)')
    
    def get_dataset_filename(self):
        """
        Simple function to get the dataset filename

        Returns:
            str: Dataset file name
        """
        if self.ComBat:
            return self.norm+'_dataset_log2_combat.csv'
        elif self.ComBat_seq:
            return self.norm+'_dataset_combat_seq.csv'
        else:
            return self.norm+'_dataset.csv'

    def read_data(self):
        """
        This function loads one of 3 csv files of the Shokhirev dataset proposed in DOI: 10.1111/acel.13280 and
        performs a log2 transform if it is indicated. It also returns the metadata for all the dataset.

        Returns:
            x_np (numpy.array): The gene expression matrix.
            y_np (numpy.array): Vector with all the ages of the samples.
            gene_names (list): A list with the names of all the genes in the expression matrix. They can be filtered if self.filter_type is specified.
            metadata (pandas.DataFrame): A dataframe with all the metadata of the samples.
        """
        print('Started reading expression data...')
        
        start = time.time()
        
        # Declare expression data path and path to tables containing the filtered genes 
        expression_path = os.path.join(self.path ,"normalized", self.dataset_filename)
        gene_filtering_path = os.path.join(self.path ,"tables.xlsx")
        # Read expression file and make minor modifications
        expression = pd.read_csv(expression_path, sep=",", header=0)
        expression = expression.rename(columns = {'Unnamed: 0':'SRR.ID'})
        expression = expression.set_index('SRR.ID')
        expression = expression.T

        end = time.time()
        print(f'It took {round(end-start, 2)} s to read the expression data.')

        # filtering type dictionary mapper
        filtering_type_dict = {'1000var':'1000 Variable Gene',
                                '1000diff':'1000 Differential Gene',
                                '100var':'100 Variable Gene',
                                '100diff':'100 Differential Gene'}

        if self.filter_type == 'none':
            print('Gene filtering based in differential or most variant genes NOT performed...')
            # Obtain complete gene list
            gene_names = expression.columns.tolist()
        else:
            print(f'Performed gene filtering based in the {filtering_type_dict[self.filter_type]}s...')
            # Read gene filtering tables
            gene_filtering_xlsx = pd.read_excel(gene_filtering_path, sheet_name = "Table S5 (Related to Fig. 3)", header=3)
            # Obtain filtered gene series
            gene_names_series = gene_filtering_xlsx[filtering_type_dict[self.filter_type]]
            # Obtain filtered gene list
            gene_names = gene_names_series.tolist()
            # Filter expression data to keep only the filtered genes
            expression = expression.loc[:, expression.columns.isin(gene_names)]

        print('Performing log2 transformation...')
        # Handle possible log2 transformation. In case ComBat is True, the expression matrix is already log2
        # transformed from the loaded file.
        if self.log2 and (not self.ComBat):
            # Perform the log2 transform of data
            expression = np.log2(expression + 1)

        print('Reading metadata...')
        # Declare meta data path
        meta_path = os.path.join(self.path, "meta_filtered.txt")
        # Read metadata file and make minor modifications
        meta = pd.read_csv(meta_path, sep="\t", header=0)
        del meta['Unnamed: 0']
        meta = meta.set_index('SRR.ID')

        # Get ordered list of genes
        gene_names = expression.columns.tolist()
        # Declare numpy variables
        x_np = expression.values
        y_np = meta['Age'].values

        print('Finished reading all the data...')

        return x_np, y_np, gene_names, meta

    def compute_statistics(self):
        """
        This function computes or loads dataframes with various statistics of the complete dataset: Mean, standard deviation,
        fraction of samples expressiong a gene and the number of samples in a given batch. It is important to note that all the
        statistics are computed in every batch of the data

        Returns:
            dict: Statistics dictionary with the following keys:
                  'mean': Pandas DataFrame with mean expression (rows are genes, columns are batches)
                  'std': Pandas DataFrame with standard deviation of expression (rows are genes, columns are batches)
                  'exp_frac': Pandas DataFrame with the fraction of samples expression a given gene (rows are genes, columns are batches)
                  'batch_samples': Pandas DataFrame with the number of samples in each batch (Index is batches the single column is the sample number)
        """
        # Create directory of statistics if it does not exists
        os.makedirs(os.path.join(self.path, 'normalized', 'statistics'), exist_ok=True)

        # Get the name of the directory containing the csv files
        statistics_path = os.path.join(self.path, 'normalized', 'statistics', 
                                        self.dataset_filename.split('.')[0])

        # If statistic csvs exist load them
        if os.path.exists(statistics_path) and (self.force_compute == False):

            print(f'Loading statistics from {statistics_path}')

            mean_df = pd.read_csv(os.path.join(statistics_path, 'mean.csv'), index_col = 0)
            std_df = pd.read_csv(os.path.join(statistics_path, 'std.csv'), index_col = 0)
            exp_frac_df = pd.read_csv(os.path.join(statistics_path, 'exp_frac.csv'), index_col = 0)
            batch_samples_df = pd.read_csv(os.path.join(statistics_path, 'batch_samples.csv'), index_col = 0)
        
        # If it does not exist then compute it and save it 
        else:
            print(f'Computing statistics and saving them in {statistics_path}')
            # Make directory
            os.makedirs(statistics_path, exist_ok=True)
            # Compute the minimum value in x_np for the expression fraction computation 
            min_exp = np.min(self.x_np)
            # Get unique batch list in metadata
            unique_batches = self.metadata_df['Batch'].unique()
            # Declare statistic dataframes (mean, std, expression_frac) full of nans
            mean_df = pd.DataFrame(index = self.gene_names, columns = unique_batches)
            std_df = pd.DataFrame(index = self.gene_names, columns = unique_batches)
            exp_frac_df = pd.DataFrame(index = self.gene_names, columns = unique_batches)

            # Cycle over batches
            for batch in unique_batches:
                # Get batch expression matrix
                batch_sample_indexes = self.metadata_df['Batch'] == batch
                batch_expression = self.x_np[batch_sample_indexes, :]
                # get the mean and add it to dataframe
                mean_df[batch] = batch_expression.mean(axis=0)
                # Get std and add it to dataframe
                std_df[batch] = batch_expression.std(axis=0)
                # Get expression frac and add  to dataframes
                exp_frac_df[batch] = (batch_expression > min_exp).sum(axis=0) / batch_expression.shape[0]

            # Make value count over metadata and get dataframe of the number of samples in every batch
            batch_samples_df = pd.DataFrame(self.metadata_df['Batch'].value_counts())
            batch_samples_df.rename({'Batch': 'samples'}, inplace=True, axis=1)
            batch_samples_df.index.set_names('Batch', inplace=True)

            # Save the csvs
            mean_df.to_csv(os.path.join(statistics_path, 'mean.csv'), index = True)
            std_df.to_csv(os.path.join(statistics_path, 'std.csv'), index = True)
            exp_frac_df.to_csv(os.path.join(statistics_path, 'exp_frac.csv'), index = True)
            batch_samples_df.to_csv(os.path.join(statistics_path, 'batch_samples.csv'), index = True)
        
        # Join all dataframes in a single dictionary
        statistics_dict = {'mean': mean_df, 'std': std_df, 'exp_frac': exp_frac_df, 'batch_samples': batch_samples_df} 
        
        # Return statistics dict
        return statistics_dict

    def filter_batches(self):
        """
        This function filters out batches with a number of samples lower than the defined threshold self.batch_sample_thr and returns a dictionary
        with filtered datastructures.

        Returns:
            dict: Dictionary containing all the filtered data. The keys are the following:
                  'x': Numpy expression matrix with samples in rows and genes in columns.
                  'y': Numpy Age annotation vector of the filtered samples.
                  'metadata': Pandas DataFrame with the same format as self.metadata_df but filtered by valid batches.
                  'valid_batches': Numpy array of ints having the valid batches used.
        """

        print(f'Filtering out batches with fewer than {self.batch_sample_thr} samples...')

        batch_df = self.statistics_dict['batch_samples']
        
        # Get the valid batches to filter samples
        valid_batches = batch_df.index.to_numpy()[batch_df['samples'] > self.batch_sample_thr]
        
        # Get boolean valid indexes
        valid_bool_index = self.metadata_df['Batch'].isin(valid_batches).values
        
        # Filter metadata, x_np and y_np
        batch_fil_metadata = self.metadata_df.loc[valid_bool_index, :]
        batch_fil_x_np = self.x_np[valid_bool_index, :]
        batch_fil_y_np = self.y_np[valid_bool_index]

        # Add filtering results to dictionary
        batch_filtered_dict = {'x': batch_fil_x_np, 'y': batch_fil_y_np, 'metadata': batch_fil_metadata, 'valid_batches': valid_batches}

        print(f'There are {len(valid_batches)}/{batch_df.shape[0]} batches and {batch_fil_x_np.shape[0]}/{self.x_np.shape[0]} samples remaining after batch based filtering.')

        return batch_filtered_dict

    def filter_exp_frac(self):
        """
        This function filters out genes that are expressed in a fraction of samples lower than self.exp_fraction_thr for any batch.
        In other words, the genes remaining are expressed by at least self.exp_frac_thr X 100% of the samples in all the batches. This 
        filtering is done over the batches remaining after the filter_batches() function. The function returns a dictionary with all the
        datastructures after applying all the previous filtering. 

        Returns:
            dict: Dictionary containing all the filtered data. The keys are the following:
                  'x': Numpy expression matrix with samples in rows and genes in columns. Here, the samples and genes are filtered.
                  'y': Numpy Age annotation vector of the filtered samples. This is the same as filter_batches() dict entry.
                  'metadata': Pandas DataFrame with the same format as self.metadata_df but filtered by valid batches. This is the same as filter_batches() dict entry.
                  'valid_batches': Numpy array of strings having the valid batches used. This is the same as filter_batches() dict entry but the elements are strings.
                  'valid_genes': Numpy array of strings containing the valid genes after the filtering.
        """

        print(f'Filtering out genes with an expression fraction bellow {self.exp_frac_thr}')
        # Get expression fraction statistic dataframe
        exp_frac_df = self.statistics_dict['exp_frac']
        # Get valid batches to work with
        valid_batches = self.batch_filtered_dict['valid_batches'].astype('str')
        
        # Filter exp_frac dataframe by valid batches
        valid_exp_frac_df = exp_frac_df.loc[:, (exp_frac_df.columns).astype(str).isin(valid_batches)]
        
        # Compute minimum expression fraction of each gene over all valid batches
        min_exp_frac = valid_exp_frac_df.min(axis=1)
        
        # Obtain valid genes boolean index
        valid_genes_bool_index = (min_exp_frac > self.exp_frac_thr).values
        
        # Get a valid gene list
        valid_genes = exp_frac_df.index.to_numpy()[valid_genes_bool_index]
        
        # Filter out invalid genes from the expression matrix
        exp_frac_filtered_x_np = self.batch_filtered_dict['x'][:, valid_genes_bool_index]

        # Put X, Y, Metadata, valid batches and valid genes in a single dictionary
        exp_frac_filtered_dict = {  'x': exp_frac_filtered_x_np,
                                    'y': self.batch_filtered_dict['y'],
                                    'metadata': self.batch_filtered_dict['metadata'],
                                    'valid_batches': valid_batches,
                                    'valid_genes': valid_genes}
        
        print(f'After gene filtering by expression fraction there are {exp_frac_filtered_x_np.shape[1]}/{self.batch_filtered_dict["x"].shape[1]} genes remaining')

        return exp_frac_filtered_dict

    def batch_normalize(self):
        """
        This function uses all the previous results to perform a batch z score normalization over the valid gene expression matrix.
        The results are returned in a dictionary with the same format of the one returned by the filter_exp_frac() function. All the keys
        of this dictionary are identical to the self.exp_frac_filtered_dict dictionary excepting 'x' which is now batch normalized. Also if
        the batch normalization is not specified (self.batch_norm == False) then this function returns a deep copy of the  
        self.exp_frac_filtered_dict dictionary.

        Returns:
            dict: Dictionary containing all the filtered data and batch normalized data. The keys are identical to the ones in self.exp_frac_filtered_dict
                  with the following exception:
                  'x': Batch normalized numpy expression matrix (samples in rows, genes in columns). If self.batch_norm == False the there is no
                       batch normalization and a deep copy of self.exp_frac_filtered_dict['x'] is returned.
        """
        # If batch normalization is not specified, don't do anything and return the previous filtering results
        if self.batch_norm == False:
            print('Did not perform batch z-score normalization...')
            return copy.deepcopy(self.exp_frac_filtered_dict)

        # Else perform batch normalization
        else:
            print('Performing batch z-score normalization...')

            # Get mean and std dataframe statistics
            mean_df, std_df = self.statistics_dict['mean'], self.statistics_dict['std']

            # Get valid batches, valid genes and valid metadata
            valid_batches = self.exp_frac_filtered_dict['valid_batches']
            valid_genes = self.exp_frac_filtered_dict['valid_genes']
            valid_metadata = self.exp_frac_filtered_dict['metadata']

            # Get valid expression data to normalize
            no_norm_x = self.exp_frac_filtered_dict['x']
            
            # Filter mean and std dataframes to leave just valid batches
            mean_df = mean_df.loc[:, (mean_df.columns).astype(str).isin(valid_batches)]
            std_df = std_df.loc[:, (std_df.columns).astype(str).isin(valid_batches)]

            # Filter mean and std dataframes to leave just valid genes
            valid_mean_df = mean_df.loc[mean_df.index.isin(valid_genes), :]
            valid_std_df = std_df.loc[std_df.index.isin(valid_genes), :]

            # Declare a zeros matrix of the needed size to strode the normalized data
            batch_norm_x_np = np.zeros_like(no_norm_x)

            # Cycle over batches to normalize
            for batch in valid_batches:
                # Get mean and std vector for current batch
                curr_mean = valid_mean_df.loc[:, (valid_mean_df.columns).astype(str).isin([batch])].values.reshape((1, -1))
                curr_std = valid_std_df.loc[:, (valid_std_df.columns).astype(str).isin([batch])].values.reshape((1, -1))

                # Obtain boolean indexes of current batch in metadata
                curr_batch_bool_indexes = (valid_metadata['Batch'] == int(batch)).values

                # Normalize the data and assign it to the zeros matrix
                batch_norm_x_np[curr_batch_bool_indexes, :] = (no_norm_x[curr_batch_bool_indexes, :] - curr_mean) / curr_std

            # Join X, Y, Metadata, Valid batches, and Valid genes in the same dictionary
            # First we deep copy de dictionary from the previous exp_frac_filtering step
            batch_normalized_dict = copy.deepcopy(self.exp_frac_filtered_dict)
            # And then we replace the expression matrix by the one we have just computed
            batch_normalized_dict['x'] = batch_norm_x_np
            
            return batch_normalized_dict

    def split_data(self):
        """
        This function uses the self.batch_normalized_dict to shuffle and split the data. The shuffle can be controlled using self.shuffle_seed.
        The output is a dictionary with all the information shuffled and split into train, val, and test groups. It also contains valid genes and
        valid batches. This dictionary is the end of the bioinformatic pipeline over the data and will be used to compute the correlation graph and
        the PyTorch geometric dataloaders. The partition is controlled by self.val_frac and self.test_frac

        Returns:
            dict: Dictionary containing all the filtered and batch normalized data and metadata. The keys are the following:

                  'x_train':  Training numpy expression matrix. Only valid batches and valid genes remain. Rows are samples, columns are genes. This matrix is already shuffled.
                  'x_val':  Validation numpy expression matrix. Only valid batches and valid genes remain. Rows are samples, columns are genes. This matrix is already shuffled.
                  'x_test':  Test numpy expression matrix. Only valid batches and valid genes remain. Rows are samples, columns are genes. This matrix is already shuffled.

                  'y_train': Training one dimensional groundtruth vector. Only valid batches remain. Each component is a sample. This vector is already shuffled.
                  'y_val': Validation one dimensional groundtruth vector. Only valid batches remain. Each component is a sample. This vector is already shuffled.
                  'y_test': Test one dimensional groundtruth vector. Only valid batches remain. Each component is a sample. This vector is already shuffled.

                  'metadata_train': Pandas DataFrame (rows are samples) with all the metadata of the training samples. This dataframe is already shuffled.
                  'metadata_vsl': Pandas DataFrame (rows are samples) with all the metadata of the validation samples. This dataframe is already shuffled.
                  'metadata_test': Pandas DataFrame (rows are samples) with all the metadata of the test samples. This dataframe is already shuffled.

                  'x_shuffled': Expression matrix with joint train/val/test data. Only valid batches and valid genes remain. Rows are samples, columns are genes. The shuffle is the same of the x keys.
                  'y_shuffled': Joint train/val/test one dimensional groundtruth vector. Only valid batches remain. Each component is a sample. The shuffle is the same of the y keys.
                  'metadata_shuffled': Pandas DataFrame (rows are samples) with all the metadata of the joint train/val/test samples. The shuffle is the same of the metadata keys.

                  'valid_batches': Numpy array of strings having the valid batches used.
                  'valid_genes': Numpy array of strings containing the valid genes after the filtering.
        """

        print(f'Shuffling and splitting the data using seed {self.shuffle_seed} ...')

        # Get expression, annotation and metadata
        x = self.batch_normalized_dict['x']
        y = self.batch_normalized_dict['y']
        meta = self.batch_normalized_dict['metadata']

        # Get shuffler according to seed
        np.random.seed(self.shuffle_seed)
        shuffler = np.random.permutation(len(x))
        
        # Get shuffled versions of x, y, and metadata
        x_shuffled = x[shuffler, :]
        y_shuffled = y[shuffler]
        metadata_shuffled = meta.iloc[shuffler, :]

        # Get the total number of samples
        n = len(x_shuffled)

        # Compute the number of train, val and test samples
        num_val = int(n*self.val_frac)
        num_test = int(n*self.test_frac)
        num_train = n - num_val - num_test

        print(f'The current partition has {num_train}/{num_val}/{num_test} train/val/test samples with fractions {round(1-self.val_frac-self.test_frac, 3)}/{self.val_frac}/{self.test_frac}')

        # Assign everything to a dict
        split_dict = {  'x_train': x_shuffled[:num_train, :],
                        'x_val': x_shuffled[num_train:num_train+num_val, :],
                        'x_test': x_shuffled[num_train+num_val:, :],
                        
                        'y_train': y_shuffled[:num_train],
                        'y_val': y_shuffled[num_train:num_train+num_val],
                        'y_test': y_shuffled[num_train+num_val:],
                        
                        'metadata_train': metadata_shuffled.iloc[:num_train, :],
                        'metadata_val': metadata_shuffled.iloc[num_train:num_train+num_val, :],
                        'metadata_test': metadata_shuffled.iloc[num_train+num_val:, :],

                        'x_shuffled': x_shuffled,
                        'y_shuffled': y_shuffled,
                        'metadata_shuffled': metadata_shuffled,

                        'valid_batches': self.batch_normalized_dict['valid_batches'],
                        'valid_genes': self.batch_normalized_dict['valid_genes']}        

        return split_dict

    def get_correlation_graph(self):
        # TODO: Write documentation

        # Define save dir, graph and info names
        dir = os.path.join('graphs', 'shokhirev', self.norm, f'log2={self.log2}', f'ComBat={self.ComBat}_ComBat_seq={self.ComBat_seq}', 'filter_type='+self.filter_type,
                            f'batch_norm={self.batch_norm}_batch_sample_thr={self.batch_sample_thr}', f'exp_frac_thr={self.exp_frac_thr}')
        name_graph = os.path.join(dir, f'graph_corr_thr_{self.corr_thr}_p_thr_{self.p_thr}.pkl')
        name_info = os.path.join(dir, f'graph_info_corr_thr_{self.corr_thr}_p_thr_{self.p_thr}.txt')
        
        # Handle load of previous info
        if glob.glob(name_graph) and (not self.force_compute):
            print('Loading correlation graph and info from: \n'+name_graph+'\n'+name_info)
            edge_indices, edge_attributes = pd.read_pickle(name_graph)
            # Read info from text file and print it in terminal
            with open(name_info, 'r') as f:
                info = f.readlines()
                [print(line[:-1]) for line in info]

        # If there is no file or force_compute == True compute a new graph an info
        else:
            print('Computing correlation graph and info to save in ' + dir)
            # Create save directories
            os.makedirs(os.path.join(dir), exist_ok=True)

            # Compute correlation graph
            correlation, p_value = spearmanr(self.split_dict['x_train'])                        # Compute Spearman correlation
            init_adjacency_matrix = np.abs(correlation) > self.corr_thr                         # Filter based on the correlation threshold
            valid_p_value = p_value < self.p_thr                                                # Filter based in p_value threshold
            adjacency_matrix = np.logical_and(init_adjacency_matrix, valid_p_value).astype(int) # Compose both filters
            adjacency_matrix = adjacency_matrix - np.eye(adjacency_matrix.shape[0], dtype=int)  # Discard self loops
            adjacency_sparse = coo_matrix(adjacency_matrix)                                     # Pass to sparse matrix
            adjacency_sparse.eliminate_zeros()                                                  # Delete zeros from representation
            edge_indices, edge_attributes = from_scipy_sparse_matrix(adjacency_sparse)          # Get edges and weights in tensors
            
            # Ensure all attributes are float
            edge_attributes = edge_attributes.type(torch.float)                                  
            
            # Saving graph files 
            with open(os.path.join(name_graph), 'wb') as f:
                pkl.dump((edge_indices, edge_attributes), f)

            # Computing graph characteristics
            print('Computing correlation graph info...')
            nx_graph = nx.from_scipy_sparse_array(adjacency_sparse)
            connected_bool = nx.is_connected(nx_graph)
            length_connected = [len(c) for c in sorted(nx.connected_components(nx_graph), key=len, reverse=False)]
            with open(name_info, 'w') as f:
                print_both('Total amount of nodes: ' + str(self.split_dict['x_train'].shape[1]), f)
                print_both('Total amount of edges: ' + str(edge_attributes.shape[0]), f)
                print_both('Average graph degree: ' + str(round(edge_attributes.shape[0]/self.split_dict['x_train'].shape[1], 3)), f)
                print_both('Is the graph connected: ' + str(connected_bool), f)
                print_both('Top 5 connected components in size: ' + str(sorted(length_connected, reverse=True)[:5]), f)

        return edge_indices, edge_attributes

    def get_string_graph(self):
        # TODO: Write documentation
        """
        Get a graph from string database matrix thresholded by the conf_thr and with the channels specified.
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
        if (not os.path.exists(os.path.join('data', 'string', 'string_hugo_mapped.txt'))) or self.force_compute:
            print('Computing STRING connections and saving them to ' + os.path.join('data', 'string' 'string_hugo_mapped.txt'))

            # Check if json file exists
            if not os.path.exists(os.path.join('data', 'string','ensp_2_hugo_mapper.json')):
                print(f'Getting mapping from ensemble proteins to hugo symbols using Biomart and saving to {os.path.join("data", "string","ensp_2_hugo_mapper.json")}...')
                # Get Biomart server
                server = BiomartServer("http://uswest.ensembl.org/biomart" )
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
                print(f'Loading ensemble proteins to hugo symbols mapping from {os.path.join("data", "string","ensp_2_hugo_mapper.json")}...')
                # Load json file
                with open(os.path.join('data', 'string','ensp_2_hugo_mapper.json'), 'r') as f:
                    ensp_2_hugo_mapper = json.load(f)
            
            # Open STRING txt file with pandas
            print('Opening STRING raw data...')
            # TODO: Make an automatic download of the original string file
            string_df = pd.read_csv(os.path.join('data', 'string','9606.protein.links.detailed.v11.5.txt'), sep=' ')

            # delete '9606.' from values in columns `protein1` and `protein2`
            string_df['protein1'] = string_df['protein1'].str.replace('9606.', '', regex = True)
            string_df['protein2'] = string_df['protein2'].str.replace('9606.', '', regex = True)

            # Delete rows with protein1 values that are not in ensp_2_hugo_mapper keys
            string_df = string_df[string_df['protein1'].isin(ensp_2_hugo_mapper.keys())]
            # Delete rows with protein2 values that are not in ensp_2_hugo_mapper keys
            string_df = string_df[string_df['protein2'].isin(ensp_2_hugo_mapper.keys())]

            print('Mapping string data to Hugo symbols')
            # Map protein1 and protein2 values to hugo names
            string_df['protein1'] = string_df['protein1'].map(ensp_2_hugo_mapper)
            string_df['protein2'] = string_df['protein2'].map(ensp_2_hugo_mapper)

            # Get probability scores from STRING
            string_df.loc[:, ~ string_df.columns.isin(['protein1', 'protein2'])] = string_df.loc[:, ~ string_df.columns.isin(['protein1', 'protein2'])]/1000 

            # Create merged column with protein1 and protein2 values
            string_df['p12'] = string_df['protein1']+','+string_df['protein2']

            print('Merging redundant connections with average...')
            # Compute mean of duplicates in merged column
            string_df = string_df.groupby('p12').mean( numeric_only=True )
            
            # Put merged column back in dataframe
            string_df = string_df.reset_index()
            # Split merged column to get protein1 and protein2 values
            string_df['protein1'] = string_df['p12'].str.split(',').str[0]
            string_df['protein2'] = string_df['p12'].str.split(',').str[1]
            
            # Delete merged column
            string_df = string_df.drop('p12', axis=1)

            print('Saving mapped STRING ...')
            # Save mapped dataframe to txt file
            string_df.to_csv(os.path.join('data', 'string','string_hugo_mapped.txt'), sep=' ', index=False)

        # If hugo mapped string data exists, load it
        else:
            print(f'Loading Hugo mapped STRING connections from {os.path.join("data", "string", "string_hugo_mapped.txt")} ...')
            string_df = pd.read_csv(os.path.join('data', 'string','string_hugo_mapped.txt'), sep=' ')

        print('Filtering out STRING connections that are not in the valid genes ...')
        # Filter string dataframe to keep only the genes in the gene_list
        string_df = string_df[string_df['protein1'].isin(self.split_dict['valid_genes'])]
        string_df = string_df[string_df['protein2'].isin(self.split_dict['valid_genes'])]

        print('Filtering string by confidence ...')
        # Filter string_df rows by the conf_thr
        string_df = string_df[string_df['combined_score'] > self.conf_thr]

        # make mapper for gene indexes
        gene_index_mapper = {gene: i for i, gene in enumerate(self.split_dict['valid_genes'])}

        # Get index of genes in the gene_list for string_df['protein1'] and string_df['protein2']
        string_df['protein1_index'] = string_df['protein1'].map(gene_index_mapper)
        string_df['protein2_index'] = string_df['protein2'].map(gene_index_mapper)

        print('Obtaining edges and edges attributes in tensors ...')
        # string_df['protein1_index'] and string_df['protein2_index'] contain now graph edges in both directions
        # Now we declare the edges as tensor and edge_attributes as a tensors
        edges_np = np.stack([string_df['protein1_index'], string_df['protein2_index']], axis=0)
        attributes_np = string_df[self.channels_string].values
        edge_indices = torch.tensor(edges_np, dtype=torch.long)
        edge_attributes = torch.tensor(attributes_np)

        # Compute graph features with networkx
        print('Computing STRING graph features with networkx...')
        edges_nx = sorted([(int(edges_np[0, i]), int(edges_np[1, i])) for i in range(edges_np.shape[1])])
        nx_graph = nx.from_edgelist(edges_nx)
        length_connected = [len(c) for c in sorted(nx.connected_components(nx_graph), key=len, reverse=False)]
        print(f'Total amount of nodes:  {len(self.split_dict["valid_genes"])}')
        print(f'Total amount of edges: {edge_attributes.shape[0]}')
        print(f'Average graph degree: {round(edge_attributes.shape[0]/len(self.split_dict["valid_genes"]), 3)}') # TODO: Check if this degree is computed correctly
        print(f'Biggest connected component size: {length_connected[-1]}')

        return edge_indices, edge_attributes

    def get_graph(self):
        #TODO: Write documentation
        
        # Get string or co-expression graph depending in self.string 
        if self.string:
            edge_indexes, edge_attributes = self.get_string_graph()
        else:
            edge_indexes, edge_attributes = self.get_correlation_graph()
        
        return edge_indexes, edge_attributes

    def get_dataloaders(self, batch_size=100):

        # Define partitions and tensor keys
        partitions = ['train', 'val', 'test']
        tensor_keys = ['x_train', 'x_val', 'x_test', 'y_train', 'y_val', 'y_test']

        # Pass data and groundtruth to tensor
        tensor_dict = {k: torch.tensor(self.split_dict[k], dtype=torch.float) for k in tensor_keys}

        # Define dataloader list
        dataloader_list = []

        # Cycle to obtain dataloader for each partition group
        for i in range(len(partitions)):
            x_name, y_name = f'x_{partitions[i]}', f'y_{partitions[i]}'
            # Define graph list
            graph_list = [Data( x=torch.unsqueeze(tensor_dict[x_name][i, :], 1),
                                y=tensor_dict[y_name][i],
                                edge_index = self.edge_indices,
                                edge_attributes = self.edge_attributes,
                                num_nodes = tensor_dict[x_name].shape[1]) for i in range(tensor_dict[x_name].shape[0])]
            # Append dataloader to list
            dataloader_list.append(DataLoader(graph_list, batch_size=batch_size, shuffle=True))
        
        # Return train/val/test dataloaders

        return dataloader_list[0], dataloader_list[1], dataloader_list[2]


# Path to use
# os.path.join("data","Shokhirev_2020")

# test code
# test_dataset = ShokhirevDataset(os.path.join("data","Shokhirev_2020"), norm='tpm', log2=True, force_compute=False,
#                                batch_sample_thr = 100, exp_frac_thr = 0.5, batch_norm=True, string=True, conf_thr=0.5)

# test_dataset.edge_attributes
# test_dataset.get_dataloaders(batch_size=100)

# breakpoint()