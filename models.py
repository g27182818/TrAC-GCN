import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, ChebConv
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv

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

        self.lin = Linear(self.hidden_channels*self.input_size, out_size)

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
        x = F.dropout(x, p=0.1, training=self.training)

        y = None
        for i in range(torch.max(batch).item()+1):
            sample_x = x[batch==i, :]
            sample_x = torch.flatten(sample_x)
            y = torch.cat((y, self.lin(sample_x)), dim=0) if y is not None else self.lin(sample_x)

        return y