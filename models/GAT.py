import torch.nn as nn
import torch
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv
from collections import Counter
import torch_geometric.transforms as T
from layer import *

class GATModel(nn.Module):
    def __init__(self, input_dim, number_of_classes, hidden_layers=1, number_of_heads=1, hidden_layer=16, dp_rate=0, activation='ReLU'):
        super(GATModel, self).__init__()

        """GATModel

        Args:
            input_dim: Dimension of input features
            hidden_layer: Dimension of hidden features
            number_of_classes: Dimension of the output features. Usually number of classes in classification
            hidden_layers: Number of "hidden" graph layers
            number_of_heads: Number of heads for GAT
            dp_rate: Dropout rate to apply throughout the network
        """

        self.model_layers = [GATv2Conv(input_dim, hidden_layer, heads=number_of_heads, dropout=dp_rate)]

        if activation == 'ReLU':
            self.model_layers.append(nn.ReLU())
        elif activation == 'LeakyReLU':
            self.model_layers.append(nn.LeakyReLU())
        elif activation=='ELU':
            self.model_layers.append(nn.ELU())

        self.model_layers.append(nn.Dropout(dp_rate))

        for i in range(hidden_layers):
            self.model_layers.append(GATv2Conv(hidden_layer, hidden_layer, heads=1, dropout=dp_rate))

            if activation == 'ReLU':
                self.model_layers.append(nn.ReLU())
            elif activation == 'LeakyReLU':
                self.model_layers.append(nn.LeakyReLU())
            elif activation=='ELU':
                self.model_layers.append(nn.ELU())
            
            self.model_layers.append(nn.Dropout(dp_rate))

        self.model_layers.append(GATv2Conv(hidden_layer, number_of_classes, heads=1, dropout=dp_rate))
        self.model_layers = nn.Sequential(*self.model_layers)

    def forward(self, x, A):
        for conv in self.model_layers:
            if isinstance(conv, GATv2Conv):
                x = conv(x, A)
            else:
                x = conv(x)
            
            return x