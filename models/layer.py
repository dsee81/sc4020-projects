import torch
import torch.nn as nn
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from torch_geometric.utils import degree
from collections import Counter
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj

class GCN_Conv(torch.nn.Module):
  def __init__(self, input_dim, output_dim, bias=False):
    super(GCN_Conv, self).__init__()
    self.W = nn.Linear(input_dim, output_dim, bias=bias)
    self.B = nn.Linear(input_dim, output_dim, bias=False)

  def forward(self, X, A):
    neigh = A @ X
    return self.W(neigh) + self.B(X)

class GATLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GATLayer, self).__init__()

    
    ## add the Graph Addition Layer Later 

    