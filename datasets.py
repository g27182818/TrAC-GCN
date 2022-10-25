import os
import statistics
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
import copy
from utils import *
# Set random seed
np.random.seed(1234)


class ShokhirevDataset:
    def __init__(self, path,  norm, log2, val_frac = 0.2, test_frac = 0.2, corr_thr=0.6, p_thr=0.05,
                filter_type = 'none', ComBat = False, ComBat_seq = False, batch_sample_thr = 10, exp_frac_thr = 0.1, 
                batch_norm = True, string = False, conf_thr = 0.0, channels_string = ['combined_score'], force_compute=False):

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

        # Check that all the parameters are valid
        self.check_parameters_validity()

        # TODO: Add additional filter for just using healthy data

        # Do the main Bioinformatic pipeline
        self.dataset_filename = self.get_dataset_filename()                         # Get the raw dataset filename
        self.x_np, self.y_np, self.gene_names, self.metadata_df =  self.read_data() # Read the raw data. This also performs a log2 transformation if specified.
        self.statistics_dict = self.compute_statistics()                            # Compute o load statistics of complete dataset
        self.batch_filtered_dict  =  self.filter_batches()                          # Filter out batches based in the sample number
        self.exp_frac_filtered_dict = self.filter_exp_frac()                        # Filter out genes by expression fraction. This filtering is made over the batch filtered data. 
        self.batch_normalized_dict = self.batch_normalize()                         # Normalize by batches if specified. If self.batch_norm == False then this function returns self.exp_frac_filtered_dict.
        # Split and shuffle data
        # Compute graph
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
        his function loads one of 3 csv files of the Shokhirev dataset proposed in DOI: 10.1111/acel.13280 and
        performs a log2 transform if it is indicated. It also returns the metadata for all the dataset.

        Returns:
            x_np (numpy.array): The gene expression matrix.
            y_np (numpy.array): Vector with all the ages of the samples.
            gene_names (list): A list with the names of all the genes in the expression matrix.
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
        valid_exp_frac_df = exp_frac_df.loc[:, exp_frac_df.columns.isin(valid_batches)]
        
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
            mean_df = mean_df.loc[:, mean_df.columns.isin(valid_batches)]
            std_df = std_df.loc[:, std_df.columns.isin(valid_batches)]

            # Filter mean and std dataframes to leave just valid genes
            valid_mean_df = mean_df.loc[mean_df.index.isin(valid_genes), :]
            valid_std_df = std_df.loc[std_df.index.isin(valid_genes), :]

            # Declare a zeros matrix of the needed size to strode the normalized data
            batch_norm_x_np = np.zeros_like(no_norm_x)

            # Cycle over batches to normalize
            for batch in valid_batches:
                # Get mean and std vector for current batch
                curr_mean = valid_mean_df[batch].values.reshape((1, -1))
                curr_std = valid_std_df[batch].values.reshape((1, -1))

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

# TODO: Add shuffle in the end with the data split
# # Shuffle variables 
# np.random.seed(1234)
# shuffler = np.random.permutation(len(x_np))
# x_np = x_np[shuffler]
# y_np = y_np[shuffler]

# metadata = meta.iloc[shuffler, :]


# Path to use
# os.path.join("data","Shokhirev_2020")

# test code
test_dataset = ShokhirevDataset(os.path.join("data","Shokhirev_2020"), norm='tpm', log2=True, force_compute=False,
                                batch_sample_thr = 100, exp_frac_thr = 0.5, batch_norm=False)

breakpoint()