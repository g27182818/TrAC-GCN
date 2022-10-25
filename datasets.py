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
from utils import *
# Set random seed
np.random.seed(1234)


class ShokhirevDataset:
    def __init__(self, path,  norm, log2, val_frac = 0.2, test_frac = 0.2, corr_thr=0.6, p_thr=0.05,
                filter_type = 'none', ComBat = False, ComBat_seq = False, string = False, conf_thr = 0.0,
                channels_string = ['combined_score'], force_compute=False):

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
        self.string = string
        self.conf_thr = conf_thr
        self.channels_string = channels_string
        self.force_compute = force_compute

        # Get dataset filename
        if self.ComBat:
            self.dataset_filename = self.norm+'_dataset_log2_combat.csv'
        elif self.ComBat_seq:
            self.dataset_filename = self.norm+'_dataset_combat_seq.csv'
        else:
            self.dataset_filename = self.norm+'_dataset.csv'


        # Check that all the parameters are valid
        self.check_parameters_validity()

        # Do the main bioinformatic pipeline

        # Read the data
        self.x_np, self.y_np, self.gene_names, self.metadata_df =  self.read_data()
        self.compute_statistics()
         

    # Function to check that all the parameters are in a valid combination 
    def check_parameters_validity(self):

        if self.ComBat and self.ComBat_seq:
            raise ValueError('ComBat and ComBat_seq cannot be both True')

        if self.ComBat and (not self.log2):
            raise ValueError('ComBat requires log2 transform because it was computed over log2(x+1) data')
        

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

        # Declare expression data path and path to tables containing the filtered genes 
        expression_path = os.path.join(self.path ,"normalized", self.dataset_filename)
        gene_filtering_path = os.path.join(self.path ,"tables.xlsx")
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

        if self.filter_type == 'none':
            # Obtain complete gene list
            gene_names = expression.columns.tolist()
        else:
            # Read gene filtering tables
            gene_filtering_xlsx = pd.read_excel(gene_filtering_path, sheet_name = "Table S5 (Related to Fig. 3)", header=3)
            # Obtain filtered gene series
            gene_names_series = gene_filtering_xlsx[filtering_type_dict[self.filter_type]]
            # Obtain filtered gene list
            gene_names = gene_names_series.tolist()
            # Filter expression data to keep only the filtered genes
            expression = expression.loc[:, expression.columns.isin(gene_names)]

        # Handle possible log2 transformation. In case ComBat is True, the expression matrix is already log2
        # transformed from the loaded file.
        if self.log2 and (not self.ComBat):
            # Perform the log2 transform of data
            expression = np.log2(expression + 1)

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

        return x_np, y_np, gene_names, meta

    def compute_statistics(self):
        
        # Create directory of statistics if it does not exists
        os.makedirs(os.path.join(self.path, 'normalized', 'statistics'), exist_ok=True)

        # Get the name of the directory containing the csv files
        statistics_path = os.path.join(self.path, 'normalized', 'statistics', 
                                        self.dataset_filename.split('.')[0])

        # If statistic csvs exist load them
        if os.path.exists(statistics_path) and (self.force_compute == False):
            mean_df = pd.read_csv(os.path.join(statistics_path, 'mean.csv'), index_col = 0)
            std_df = pd.read_csv(os.path.join(statistics_path, 'std.csv'), index_col = 0)
            exp_frac_df = pd.read_csv(os.path.join(statistics_path, 'exp_frac.csv'), index_col = 0)
            batch_samples_df = pd.read_csv(os.path.join(statistics_path, 'batch_samples.csv'), index_col = 0)
        
        # If it does not exist then compute it and save it 
        else:
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

            # TODO: Fix names
            # FIXME: End function
            # Make value count over metadata and get dataframe of the number of samples in every batch
            batch_samples_df = pd.DataFrame(self.metadata_df['Batch'].value_counts(), columns = ['Batch', 'samples'])

            breakpoint()

            # Save the csvs
            mean_df.to_csv(os.path.join(statistics_path, 'mean.csv'), index = True)
            std_df.to_csv(os.path.join(statistics_path, 'std.csv'), index = True)
            exp_frac_df.to_csv(os.path.join(statistics_path, 'exp_frac.csv'), index = True)
            batch_samples_df.to_csv(os.path.join(statistics_path, 'batch_samples.csv'), index = True)
        
        # Return 4 statistic dataframes with genes in rows and batches in columns 
        return mean_df, std_df, exp_frac_df, batch_samples_df

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
test_dataset = ShokhirevDataset(os.path.join("data","Shokhirev_2020"), norm='tpm', log2=True, force_compute=True)

breakpoint()