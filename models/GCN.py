import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from layer import *

class GCNModel(torch.nn.Module): #one which takes in the number of layers
  def __init__(self, input_dim, number_of_classes, layers = 3, hidden_layer=16, layer_name='GCN', dp_rate=0):
    """GNNModel.

    Args:
        input_dim: Dimension of input features
        hidden_layer: Dimension of hidden features
        number_of_classes: Dimension of the output features. Usually number of classes in classification
        layers: Number of "hidden" graph layers
        layer_name: String of the graph layer to use
        dp_rate: Dropout rate to apply throughout the network
        kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)

    """
    super(GCNModel, self).__init__()

    self.layer_name = layer_name
    self.gcn_layers = {'GCN_Conv': GCN_Conv, 'GCN': GCNConv}
    gcn_layer = self.gcn_layers[layer_name]

    model_layers = [gcn_layer(input_dim, hidden_layer)]
    model_layers.append(nn.ReLU())
    model_layers.append(nn.Dropout(dp_rate))

    for i in range(layers):
      model_layers.append(gcn_layer(hidden_layer, hidden_layer))
      model_layers.append(nn.ReLU())
      model_layers.append(nn.Dropout(dp_rate))

    model_layers.append(gcn_layer(hidden_layer, number_of_classes))
    self.model_layers = nn.Sequential(*model_layers)

  def forward(self, x, A):
    for conv in self.model_layers:
      if isinstance(conv, self.gcn_layers[self.layer_name]):
        x = conv(x, A)
      else:
        x = conv(x)

    output = torch.softmax(x, dim=1)

    return output

