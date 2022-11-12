import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import LayerNorm, Linear, ReLU
import torch_geometric
from torch_geometric.nn import GCNConv, ChebConv, DeepGCNLayer, GENConv, DenseGCNConv
import torch.nn as nn
import math
from torch.nn import Parameter

# TODO: Propose diferent models


class BaselineModel(torch.nn.Module):
    def __init__(self, hidden_channels, input_size, out_size, dropout=0.5, final_pool='None'):
        """
        Class constructor for baseline using ChebConv, Inherits from torch.nn.Module
        :param hidden_channels: (Int) Hidden channels in every layer.
        :param out_size: (Int) Number of output classes.
        :param dropout: (Float) Dropout probability.
        :param final_pool: (str) Final pooling type over nodes.
        """
        super(BaselineModel, self).__init__()
        # Class atributes
        self.hidd = hidden_channels
        self.input_size = input_size
        self.out_size = out_size
        self.dropout = dropout
        self.final_pool = final_pool
        self.lin_input_size = self.input_size * self.hidd if final_pool == 'None' else self.input_size
        # Convolution definitions
        self.conv1 = ChebConv(1, hidden_channels, K=5)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=5)
        # Code for GCN test
        # self.conv1 = GCNConv(1, hidden_channels, improved=True, add_self_loops=True)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, improved=True, add_self_loops=True)
        self.lin1 = Linear(self.lin_input_size, 1000)
        self.lin2 = Linear(1000, self.out_size)
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Performs a forward pass.
        :param x: (torch.Tensor) Input features of each node.
        :param edge_index: (torch.Tensor) Edges indicating graph connectivity.
        :param edge_attr: (torch.Tensor) Edge attributes.
        :param batch: (torch.Tensor) Batch vector indicating the correspondence of each node in the batch.
        :return: (torch.Tensor) Matrix of logits, each row corresponds with a patient in the batch and each column represent a
                 cancer or normal type logit.
        """
        # TODO: Include edge attributes in convolutions
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        if self.final_pool == 'None':
            pass
        elif self.final_pool == 'mean':
            # Get mean from each node
            x = x.mean(dim=1)
        elif self.final_pool == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.final_pool == 'add':
            x = x.sum(dim=1)
        else:
            raise ValueError('Invalid final pooling type')

        x = self.lin1(torch.reshape(x, (torch.max(batch).item() + 1, self.lin_input_size)))
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training) # Added dropout
        x = torch.squeeze(self.lin2(x))
        x = torch.clamp(x, 0, 110) # Assures that the predictions are between 0 and 110
        return x

class TracGCN(torch.nn.Module):
    def __init__(self, hidden_channels, input_size, out_size, dropout=0.5, final_pool='None'):
        """
        Class constructor for baseline using ChebConv, Inherits from torch.nn.Module
        :param hidden_channels: (Int) Hidden channels in every layer.
        :param out_size: (Int) Number of output classes.
        :param dropout: (Float) Dropout probability.
        :param final_pool: (str) Final pooling type over nodes.
        """
        super(TracGCN, self).__init__()
        # Class atributes
        self.hidd = hidden_channels
        self.input_size = input_size
        self.out_size = out_size
        self.dropout = dropout
        self.final_pool = final_pool
        self.lin_input_size = self.input_size * self.hidd if final_pool == 'None' else self.input_size
        # Convolution definitions
        self.conv1 = ChebConv(1, hidden_channels, K=5)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=5)
        
        self.gcn_lin1 = Linear(self.lin_input_size, 1000)
        self.gcn_lin2 = Linear(1000, 50)

        self.holzscheck_MLP = MLP(h_sizes=[self.input_size, 350, 350, 350],
                                  out_size=50, act="elu",
                                  init_weights=torch.nn.init.kaiming_uniform_,
                                  dropout=self.dropout)

        self.out = Linear(100, self.out_size)
    
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Performs a forward pass.
        :param x: (torch.Tensor) Input features of each node.
        :param edge_index: (torch.Tensor) Edges indicating graph connectivity.
        :param edge_attr: (torch.Tensor) Edge attributes.
        :param batch: (torch.Tensor) Batch vector indicating the correspondence of each node in the batch.
        :return: (torch.Tensor) Matrix of logits, each row corresponds with a patient in the batch and each column represent a
                 cancer or normal type logit.
        """
        x_mlp = x 

        # TODO: Include edge attributes in convolutions
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        if self.final_pool == 'None':
            pass
        elif self.final_pool == 'mean':
            # Get mean from each node
            x = x.mean(dim=1)
        elif self.final_pool == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.final_pool == 'add':
            x = x.sum(dim=1)
        else:
            raise ValueError('Invalid final pooling type')

        x = self.gcn_lin1(torch.reshape(x, (torch.max(batch).item() + 1, self.lin_input_size)))
        x = x.relu()
        x = torch.squeeze(self.gcn_lin2(x))
        
        # Final graph representation
        x = x.relu()

        # Holzscheck MLP
        x_mlp = self.holzscheck_MLP(x_mlp, edge_index, edge_attr, batch)

        # Concatenate
        x = torch.cat((x, x_mlp), dim=1)

        x = F.dropout(x, p=self.dropout, training=self.training) # Added dropout
        
        # Output layer
        x = torch.squeeze(self.out(x))

        x = torch.clamp(x, 0, 110) # Assures that the predictions are between 0 and 110
        return x

class GraphHead(torch.nn.Module):
    def __init__(self, hidden_channels, input_size, out_size, dropout=0.5, final_pool='None'):
        """
        Class constructor for baseline using ChebConv, Inherits from torch.nn.Module
        :param hidden_channels: (Int) Hidden channels in every layer.
        :param out_size: (Int) Number of output classes.
        :param dropout: (Float) Dropout probability.
        :param final_pool: (str) Final pooling type over nodes.
        """
        super(GraphHead, self).__init__()
        # Class atributes
        self.hidd = hidden_channels
        self.input_size = input_size
        self.out_size = out_size
        self.dropout = dropout
        self.final_pool = final_pool
        self.lin_input_size = self.input_size * self.hidd if final_pool == 'None' else self.input_size
        # Convolution definitions
        self.conv1 = ChebConv(1, hidden_channels, K=5)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=5)

        self.lin1 = Linear(self.lin_input_size, 1000)
        self.lin2 = Linear(1000, 50)
        self.lin3 = Linear(50, self.out_size)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Performs a forward pass.
        :param x: (torch.Tensor) Input features of each node.
        :param edge_index: (torch.Tensor) Edges indicating graph connectivity.
        :param edge_attr: (torch.Tensor) Edge attributes.
        :param batch: (torch.Tensor) Batch vector indicating the correspondence of each node in the batch.
        :return: (torch.Tensor) Matrix of logits, each row corresponds with a patient in the batch and each column represent a
                 cancer or normal type logit.
        """
        # TODO: Include edge attributes in convolutions
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        if self.final_pool == 'None':
            pass
        elif self.final_pool == 'mean':
            # Get mean from each node
            x = x.mean(dim=1)
        elif self.final_pool == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.final_pool == 'add':
            x = x.sum(dim=1)
        else:
            raise ValueError('Invalid final pooling type')

        x = self.lin1(torch.reshape(x, (torch.max(batch).item() + 1, self.lin_input_size)))
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()

        x = F.dropout(x, p=self.dropout, training=self.training) # Added dropout
        x = torch.squeeze(self.lin3(x))
        x = torch.clamp(x, 0, 110) # Assures that the predictions are between 0 and 110
        return x



class BaselineModelPool(torch.nn.Module):
    def __init__(self, hidden_channels, input_size, out_size, dropout=0.5, final_pool='None', cluster_num=1014):

        super(BaselineModelPool, self).__init__()
        # Class atributes
        self.hidd = hidden_channels
        self.input_size = input_size
        self.out_size = out_size
        self.dropout = dropout
        self.final_pool = final_pool
        self.cluster_num = cluster_num
        self.lin_input_size = self.cluster_num * self.hidd if final_pool == 'None' else self.cluster_num
        # Convolution definitions
        self.conv1_embed = DenseGCNConv(1, hidden_channels, improved=True)
        self.conv1_pool = DenseGCNConv(1, self.cluster_num, improved=True)
        self.conv2_embed = DenseGCNConv(hidden_channels, hidden_channels, improved=True)
        self.lin1 = Linear(self.lin_input_size, 1000)
        self.lin2 = Linear(1000, self.out_size)

    def forward(self, x, edge_index, edge_attr, batch):
        adj = torch_geometric.utils.to_dense_adj(edge_index = edge_index[:, :edge_index.shape[1]//int(batch.max()+1)],
                                                 edge_attr= torch.squeeze(edge_attr[:edge_attr.shape[0]//int(batch.max()+1)]),
                                                 max_num_nodes=self.input_size)

        x = torch_geometric.utils.to_dense_batch(x, batch)[0]
        s = self.conv1_pool(x, adj, add_loop=True)
        x = self.conv1_embed(x, adj, add_loop=True)
        x = x.relu()

        x, adj, l1, e1 = torch_geometric.nn.dense_diff_pool(x, adj, s)
        x = self.conv2_embed(x, adj, add_loop=True)
        x = x.relu()

        # Flatten second and third dimension
        x = torch.reshape(x, (x.shape[0], -1))
        
        if self.final_pool == 'None':
            pass
        elif self.final_pool == 'mean':
            # Get mean from each node
            x = x.mean(dim=1)
        elif self.final_pool == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.final_pool == 'add':
            x = x.sum(dim=1)
        else:
            raise ValueError('Invalid final pooling type')

        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training) # Added dropout
        x = torch.squeeze(self.lin2(x))
        x = torch.clamp(x, 0, 110) # Assures that the predictions are between 0 and 110
        return x



class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, input_size, input_node_channels=1, input_edge_channels=1, out_size=1, dropout=0.3, final_pool='None'):
        super().__init__()

        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.node_encoder = Linear(input_node_channels, hidden_channels)
        self.edge_encoder = Linear(input_edge_channels, hidden_channels)
        self.dropout = dropout
        self.final_pool = final_pool
        self.lin_input_size = self.input_size * self.hidd if final_pool == 'None' else self.input_size
        self.out_size = out_size

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin1 = Linear(self.lin_input_size, 1000)
        self.lin2 = Linear(1000, self.out_size)

    def forward(self, x, edge_index, edge_attr, batch):
        
        x = torch.reshape(x, (torch.max(batch).item() + 1, self.input_size))
        edge_attr = torch.reshape(edge_attr, (torch.max(batch).item() + 1, -1))
        encoded_edge_attr = self.edge_encoder(torch.unsqueeze(edge_attr[0], dim=1))
        processed_edge_attr = encoded_edge_attr.repeat(torch.max(batch).item()+1, 1)
        processed_x = None

        for i in range(torch.max(batch).item()+1):
            encoded_sample = self.node_encoder(torch.unsqueeze(x[i], dim=1))
            processed_x = torch.cat((processed_x, encoded_sample), dim=0) if processed_x is not None else encoded_sample

        x = processed_x
        edge_attr = processed_edge_attr

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))

        if self.final_pool == 'None':
            pass
        elif self.final_pool == 'mean':
            x = x.mean(dim=1)
        elif self.final_pool == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.final_pool == 'add':
            x = x.sum(dim=1)
        else:
            raise ValueError('Invalid final pooling type')

        lin_x = None
        for i in range(torch.max(batch).item()+1):
            sample_x = x[batch==i, :]
            sample_x = torch.flatten(sample_x)
            sample_x = F.relu(self.lin1(sample_x))
            lin_x = torch.cat((lin_x, torch.unsqueeze(sample_x, dim=0)), dim=0) if lin_x is not None else torch.unsqueeze(sample_x, dim=0)    
        x = lin_x
        x = F.dropout(x, p=self.dropout, training=self.training)
        y = torch.squeeze(self.lin2(x))
        return y

# MLP module for simple comparison
class MLP(torch.nn.Module):
    def __init__(self, h_sizes, out_size, act="relu", init_weights=None, dropout=0.0):
        """
        Class constructor for simple comparison, Inherits from torch.nn.Module. This model DOES NOT include graph
        connectivity information or any other. It uses raw input.
        :param h_sizes: (list) List of sizes of the hidden layers. Does not include the output size.
        :param out_size: (int) Number of output classes.
        :param act: (str) Paramter to specify the activation function. Can be "relu", "sigmoid", "elu" or "gelu". Default
                    "relu" (Default = "relu").
        :param init_weights: (function) Funtion to initialize the weights. Default is None.
        :param dropout: (float) Parameter to specify the dropout rate. Default is 0.0.
        """
        super(MLP, self).__init__()
        # Activation function definition
        self.activation_str = act
        # Init weights function
        self.init_weights = init_weights
        # Dropout rate
        self.dropout = dropout
        # Sizes definition
        self.hidd_sizes = h_sizes
        self.out_size = out_size
        
        # Activation function
        if self.activation_str == "relu":
            self.act = F.relu
        elif self.activation_str == "sigmoid":
            self.act = F.sigmoid
        elif self.activation_str == "elu":
            self.act = F.elu
        elif self.activation_str == "gelu":
            self.act = F.gelu
        else:
            raise ValueError("Activation function not supported")

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
            if self.init_weights is not None:
                self.hidden[-1].weight.data = self.init_weights(self.hidden[-1].weight.data)
        # Output layer
        self.out = nn.Linear(h_sizes[-1], self.out_size)
        if self.init_weights is not None:
            self.out.weight.data = self.init_weights(self.out.weight.data)
        

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Performs a forward pass of the MLP model. To provide coherence in the training and testing, this function asks
        for edge indices. However, this parameter is ignored.
        :param x: (torch.Tensor) Input features of each node.
        :param edge_index: (torch.Tensor) Edges indicating graph connectivity. This parameter is ignored.
        :param batch: (torch.Tensor) Batch vector indicating the correspondence of each node in the batch. Just used for a
                      reshape.
        :return: (torch.Tensor) Matrix of logits, each row corresponds with a patient in the batch and each column represent a
                 cancer or normal type logit.
        """
        # Resahpe x
        x = torch.reshape(x, (torch.max(batch).item() + 1, self.hidd_sizes[0]))
        # Feedforward
        for layer in self.hidden:
            x = self.act(layer(x))
        
        # Apply dropout
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer. This is the only one used in multinomial logistic regression
        output = torch.squeeze(self.out(x))
        return output

# Hadamard product operator to use in the deep feature selection model. This implementation is in the paper oficial github
# link: https://github.com/cyustcer/Deep-Feature-Selection/tree/ce7ce301b5a62783c79511c8d296463f0f46a0d2
# DOI: 10.13140/2.1.3673.6327
class DotProduct(torch.nn.Module):
    def __init__(self, in_features):
        super(DotProduct, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        #self.weight.data.normal_(0, stdv)
    def forward(self, input):
        output_np = input * self.weight.expand_as(input)
        return output_np
    def __ref__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ')'

### Nonlinear classification model
class DFS(torch.nn.Module):
    def __init__(self, h_sizes, out_size, act="relu", dropout=0.0):
        super(DFS, self).__init__()

        # Activation function definition
        self.activation_str = act
        # Dropout rate
        self.dropout = dropout
        # Sizes definition
        self.hidd_sizes = h_sizes
        self.out_size = out_size

        # Activation function
        if self.activation_str == "relu":
            self.act = F.relu
        elif self.activation_str == "sigmoid":
            self.act = F.sigmoid
        elif self.activation_str == "elu":
            self.act = F.elu
        elif self.activation_str == "gelu":
            self.act = F.gelu
        else:
            raise ValueError("Activation function not supported")
        
        self.select_layer = DotProduct(self.hidd_sizes[0]) # Selection Layer

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], self.out_size)
        

	
    def forward(self, x, edge_index, edge_attr, batch):
        # Resahpe x
        x = torch_geometric.utils.to_dense_batch(x, batch)[0]
        x = torch.squeeze(x, dim=2)
        # Pass through selection layer
        x = self.select_layer(x)

        # Feedforward
        for layer in self.hidden:
            x = self.act(layer(x))
        
        # Apply dropout
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer. This is the only one used in multinomial logistic regression
        output = torch.squeeze(self.out(x))

        return output