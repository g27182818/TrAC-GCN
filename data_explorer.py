import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap, ScalarMappable
import matplotlib
import numpy as np
import torch
from datasets import ShokhirevDataset
from utils import get_main_parser, print_both
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pd.options.mode.chained_assignment = None
# Set figure fontsizes
params = {'legend.fontsize': 'large', # 'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)


"""
This code is in charge of plotting all relevant statistics and raw data analysis
"""

# Get terminal line arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

torch.manual_seed(12345)        # Set torch manual seed for dataset (not really needed)


# All posible channels for STRING graphs
str_all_channels = ['combined_score', 'textmining', 'database', 'experimental', 'coexpression', 'cooccurence', 'fusion', 'neighborhood']
channels_string = str_all_channels if args.all_string == 'True' else ['combined_score']

# Handle automatic generation of experiment name
if args.exp_name == '-1':
    # Define experiment name based on some parameters
    args.exp_name = f'log2_{args.log2}_batch_corr_{args.batch_corr}_{args.norm}'

# Make directory to store all the figures
results_path = os.path.join('figures', 'data_exploration', args.exp_name)
log_path = os.path.join(results_path, 'dataset_log.txt')
os.makedirs(results_path, exist_ok=True)


# Print experiment parameters
with open(log_path, 'a') as f:
    print_both('Argument list to program',f)
    print_both('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                    for arg in args_dict]),f)
    print_both('\n\n',f)


# Load data
dataset = ShokhirevDataset( path = os.path.join("data","Shokhirev_2020"),   norm = args.norm,                           log2 = args.log2 == 'True',
                            val_frac = args.val_frac,                       test_frac = args.test_frac,                 corr_thr = args.corr_thr,
                            p_thr = args.p_thr,                             filter_type = args.filter_type,             batch_corr = args.batch_corr,
                            batch_sample_thr = args.batch_sample_thr,       exp_frac_thr = args.exp_frac_thr,           batch_norm = args.batch_norm == 'True',
                            string = args.string == 'True',                 conf_thr = args.conf_thr,                   channels_string = channels_string,
                            shuffle_seed = args.shuffle_seed,               force_compute = args.force_compute == 'True')


# TODO: Make mean vs std scatter plot of pre and after filtering



# Get all the raw data
valid_batches = dataset.split_dict['valid_batches']                         # Get the valid batches in the dataset
raw_meta_df = dataset.metadata_df                                           # Get the raw metadata
raw_valid_index = raw_meta_df['Batch'].astype(str).isin(valid_batches)      # Get the index of raw samples that belong to valid batches
raw_data = dataset.x_np[raw_valid_index, :]                                 # Get the raw data. These data includes all the genes
raw_meta_df = raw_meta_df[raw_valid_index]                                  # Filter raw_metadata to just the samples in valid batches

# Get all processed data
processed_data = dataset.split_dict['x_shuffled']
processed_meta_df = dataset.split_dict['metadata_shuffled']


# Transforming with pca
print('Reducing dimensions with PCA...')
raw_pca = PCA(n_components=2, random_state=0)
processed_pca = PCA(n_components=2, random_state=0)
r_raw_pca = raw_pca.fit_transform(raw_data)
r_processed_pca = processed_pca.fit_transform(processed_data)

# Transforming data with T-SNE
# If this shows a warning with OpenBlast you can solve it using this GitHub issue https://github.com/ultralytics/yolov5/issues/2863
# You just need to write in terminal "export OMP_NUM_THREADS=1"
print('Reducing dimensions with TSNE...')
raw_tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='pca')       # learning_rate and init were set due to a future warning
processed_tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='pca')
r_raw_tsne = raw_tsne.fit_transform(raw_data)
r_processed_tsne = processed_tsne.fit_transform(processed_data)

# Transforming data with UMAP
print('Reducing dimensions with UMAP...')
raw_umap = UMAP(n_components=2, random_state=0, n_neighbors=64, local_connectivity=32)          # n_neighbors and local_connectivity are set to ensure that the graph is connected
processed_umap = UMAP(n_components=2, random_state=0, n_neighbors=64, local_connectivity=32)
r_raw_umap = raw_umap.fit_transform(raw_data)
r_processed_umap = processed_umap.fit_transform(processed_data)

reduced_dict = {'raw_pca': r_raw_pca,               'raw_tsne': r_raw_tsne,                 'raw_umap': r_raw_umap,
                'processed_pca': r_processed_pca,   'processed_tsne': r_processed_tsne,     'processed_umap':  r_processed_umap}


def plot_dim_reduction(reduced_dict, raw_meta_df, processed_meta_df, color_type='batch', cmap='viridis'):

    # Get dictionaries to have different options to colorize the scatter points
    batch_dict = {batch: i/len(raw_meta_df.Batch.unique()) for i, batch in enumerate(sorted(raw_meta_df.Batch.unique()))}
    tissue_dict = {tissue: i/len(raw_meta_df.Tissue.unique()) for i, tissue in enumerate(sorted(raw_meta_df.Tissue.unique()))}
    age_dict = {age: age/raw_meta_df.Age.max() for i, age in enumerate(sorted(raw_meta_df.Age.unique()))}

    # Define color map
    cmap = get_cmap(cmap)
    
    # Compute color values
    if color_type == 'batch':
        raw_meta_df['color'] = raw_meta_df['Batch'].map(batch_dict)
        processed_meta_df['color'] = processed_meta_df['Batch'].map(batch_dict)
    elif color_type == 'tissue':
        raw_meta_df['color'] = raw_meta_df['Tissue'].map(tissue_dict)
        processed_meta_df['color'] = processed_meta_df['Tissue'].map(tissue_dict)
    elif color_type == 'age':
        raw_meta_df['color'] = raw_meta_df['Age'].map(age_dict)
        processed_meta_df['color'] = processed_meta_df['Age'].map(age_dict)

    # Plot figure
    fig, ax = plt.subplots(2, 3)

    ax[0, 0].scatter(reduced_dict['raw_pca'][:,0],          reduced_dict['raw_pca'][:,1],           c = raw_meta_df['color'],       s=1, cmap=cmap, vmin=0, vmax=1)
    ax[1, 0].scatter(reduced_dict['processed_pca'][:,0],    reduced_dict['processed_pca'][:,1],     c = processed_meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1)
    ax[0, 1].scatter(reduced_dict['raw_tsne'][:,0],         reduced_dict['raw_tsne'][:,1],          c = raw_meta_df['color'],       s=1, cmap=cmap, vmin=0, vmax=1)
    ax[1, 1].scatter(reduced_dict['processed_tsne'][:,0],   reduced_dict['processed_tsne'][:,1],    c = processed_meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1)
    ax[0, 2].scatter(reduced_dict['raw_umap'][:,0],         reduced_dict['raw_umap'][:,1],          c = raw_meta_df['color'],       s=1, cmap=cmap, vmin=0, vmax=1)
    ax[1, 2].scatter(reduced_dict['processed_umap'][:,0],   reduced_dict['processed_umap'][:,1],    c = processed_meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1)

    label_dict = {0: 'PC', 1: 'T-SNE', 2: 'UMAP'}
    title_dict = {0: 'PCA', 1: 'T-SNE', 2: 'UMAP'}

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i,j].spines.right.set_visible(False) 
            ax[i,j].spines.top.set_visible(False)
            ax[i,j].set_xlabel(f'{label_dict[j]}1')
            ax[i,j].set_ylabel(f'{label_dict[j]}2')
            ax[i,j].set_title(f'{title_dict[j]} of Raw Data' if i==0 else f'{title_dict[j]} of Processed Data')

    # Set the legend or colorbar
    if (color_type == 'batch'):
        handles = [mpatches.Patch(facecolor=cmap(val), edgecolor='black', label=f'Batch {key}') for key, val in batch_dict.items()]
        fig.set_size_inches(13, 7)
        ax[0, 2].legend(handles=handles, loc='center left',bbox_to_anchor=(1.15, 0.49))
        plt.tight_layout()

    elif (color_type == 'tissue'):
        handles = [mpatches.Patch(facecolor=cmap(val), edgecolor='black', label=f'Tissue {key.replace(";", " ")}') for key, val in tissue_dict.items()]
        fig.set_size_inches(15, 7)
        ax[0, 2].legend(handles=handles, loc='center left',bbox_to_anchor=(1.15, 0.49))
        plt.tight_layout()

    elif (color_type == 'age'):
        fig.set_size_inches(14, 7)
        plt.tight_layout()
        # Colorbar Code
        norm = matplotlib.colors.Normalize(vmin=raw_meta_df.Age.min(), vmax=raw_meta_df.Age.max())
        m = ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(m, ax=[ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2]], label='Age (yrs)', aspect= 35, pad=0.0, anchor = (0.4, 0.5))
        cbar.ax.tick_params(labelsize=15)
    
    else:
        raise ValueError('Invalid color type...')

    # Save
    fig.savefig(os.path.join(results_path, f'dim_reduction_{color_type}.png'), dpi=300)
    plt.close()

# Plot dimensionality reduction with the three different color styles
plot_dim_reduction(reduced_dict, raw_meta_df, processed_meta_df, color_type='batch', cmap='viridis')
plot_dim_reduction(reduced_dict, raw_meta_df, processed_meta_df, color_type='tissue', cmap='viridis')
plot_dim_reduction(reduced_dict, raw_meta_df, processed_meta_df, color_type='age', cmap='viridis')

# TODO: Make multiple experiments with these plots
