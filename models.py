import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, ChebConv
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv
import torch.nn as nn

# TODO: Propose diferent models


class BaselineModelSimple(torch.nn.Module):
    def __init__(self, hidden_channels, input_size, out_size):
        """
        Class constructor for baseline using ChebConv, Inherits from torch.nn.Module
        :param hidden_channels: (Int) Hidden channels in every layer.
        :param out_size: (Int) Number of output classes.
        """
        super(BaselineModelSimple, self).__init__()
        # Class atributes
        self.hidd = hidden_channels
        self.input_size = input_size
        # Convolution definitions
        self.conv1 = ChebConv(1, hidden_channels, K=5)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=5)
        self.lin1 = Linear(hidden_channels * self.input_size, 1000)
        self.lin2 = Linear(1000, out_size)
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
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.lin1(torch.reshape(x, (torch.max(batch).item() + 1, self.input_size * self.hidd)))
        x = x.relu()
        x = torch.squeeze(self.lin2(x))
        # x = torch.sigmoid(x)*110 # Assures that the predictions are between 0 and 110
        return x

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, input_size, input_node_channels=1, input_edge_channels=1, out_size=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.node_encoder = Linear(input_node_channels, hidden_channels)
        self.edge_encoder = Linear(input_edge_channels, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin1 = Linear(self.hidden_channels * self.input_size, 1000)
        self.lin2 = Linear(1000, out_size)

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

        lin_x = None
        for i in range(torch.max(batch).item()+1):
            sample_x = x[batch==i, :]
            sample_x = torch.flatten(sample_x)
            sample_x = F.relu(self.lin1(sample_x))
            lin_x = torch.cat((lin_x, torch.unsqueeze(sample_x, dim=0)), dim=0) if lin_x is not None else torch.unsqueeze(sample_x, dim=0)    
        x = lin_x
        x = F.dropout(x, p=0.3, training=self.training)
        y = torch.squeeze(self.lin2(x))
        # y = torch.sigmoid(y)*110 # Assures that the predictions are between 0 and 110
        return y

# MLP module for simple comparison
class MLP(torch.nn.Module):
    def __init__(self, h_sizes, out_size, act="relu"):
        """
        Class constructor for simple comparison, Inherits from torch.nn.Module. This model DOES NOT include graph
        connectivity information or any other. It uses raw input.
        :param h_sizes: (list) List of sizes of the hidden layers. Does not include the output size.
        :param out_size: (int) Number of output classes.
        :param act: (str) Paramter to specify the activation function. Can be "relu", "sigmoid" or "gelu". Default
                    "relu" (Default = "relu").
        """
        super(MLP, self).__init__()
        # Activation function definition
        self.activation = act
        # Sizes definition
        self.hidd_sizes = h_sizes
        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Performs a forward pass of the MLP model. To provide coherence in the training and etsting, this function asks
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
            if self.activation == "relu":
                x = F.relu(layer(x))
            elif self.activation == "gelu":
                x = F.gelu(layer(x))
            elif self.activation == "sigmoid":
                x = F.sigmoid(layer(x))
            else:
                raise NotImplementedError("Activation function not impemented")

        # Output layer. This is the only one used in multinomial logistic regression
        output = torch.squeeze(self.out(x))
        return output