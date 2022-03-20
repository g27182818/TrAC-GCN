import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, ChebConv

# TODO: Propose diferent models

# GCNConv class model, initial baseline
class BaselineModel(torch.nn.Module):
    def __init__(self, hidden_channels, input_size, out_size):
        """
        Class constructor for baseline using ChebConv, Inherits from torch.nn.Module
        :param hidden_channels: (Int) Hidden channels in every layer.
        :param out_size: (Int) Number of output classes.
        """
        super(BaselineModel, self).__init__()
        # Class atributes
        self.hidd = hidden_channels
        self.input_size = input_size
        # Convolution definitions
        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels * self.input_size, 4096)
        self.lin2 = Linear(4096, 1024)
        self.lin3 = Linear(1024, 256)
        self.lin4 = Linear(256, out_size)
    def forward(self, x, edge_index, batch):
        """
        Performs a forward pass.
        :param x: (torch.Tensor) Input features of each node.
        :param edge_index: (torch.Tensor) Edges indicating graph connectivity.
        :param batch: (torch.Tensor) Batch vector indicating the correspondence of each node in the batch.
        :return: (torch.Tensor) Matrix of logits, each row corresponds with a patient in the batch and each column represent a
                 cancer or normal type logit.
        """
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.lin1(torch.reshape(x, (torch.max(batch).item() + 1, self.input_size * self.hidd)))
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        x = x.relu()
        x = torch.squeeze(self.lin4(x))
        return x

        # GCNConv class model, initial baseline
class BaselineModelCheb(torch.nn.Module):
    def __init__(self, hidden_channels, input_size, out_size):
        """
        Class constructor for baseline using ChebConv, Inherits from torch.nn.Module
        :param hidden_channels: (Int) Hidden channels in every layer.
        :param out_size: (Int) Number of output classes.
        """
        super(BaselineModelCheb, self).__init__()
        # Class atributes
        self.hidd = hidden_channels
        self.input_size = input_size
        # Convolution definitions
        self.conv1 = ChebConv(1, hidden_channels, K=5)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=5)
        self.conv3 = ChebConv(hidden_channels, hidden_channels, K=5)
        self.lin1 = Linear(hidden_channels * self.input_size, 4096)
        self.lin2 = Linear(4096, 1024)
        self.lin3 = Linear(1024, 256)
        self.lin4 = Linear(256, out_size)
    def forward(self, x, edge_index, batch):
        """
        Performs a forward pass.
        :param x: (torch.Tensor) Input features of each node.
        :param edge_index: (torch.Tensor) Edges indicating graph connectivity.
        :param batch: (torch.Tensor) Batch vector indicating the correspondence of each node in the batch.
        :return: (torch.Tensor) Matrix of logits, each row corresponds with a patient in the batch and each column represent a
                 cancer or normal type logit.
        """
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.lin1(torch.reshape(x, (torch.max(batch).item() + 1, self.input_size * self.hidd)))
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        x = x.relu()
        x = torch.squeeze(self.lin4(x))
        return x


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
        self.lin1 = Linear(hidden_channels * self.input_size, 256)
        self.lin2 = Linear(256, out_size)
    def forward(self, x, edge_index, batch):
        """
        Performs a forward pass.
        :param x: (torch.Tensor) Input features of each node.
        :param edge_index: (torch.Tensor) Edges indicating graph connectivity.
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
        return x