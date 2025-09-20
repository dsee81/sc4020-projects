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

class GCNLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, self_loops=True, bias=False, include_bias=True): #the normalization can be done by the matrix or the row average
        super(GCNLayer, self).__init__()

        self.include_bias = include_bias
        self.self_loops = self_loops 

        self.weight = nn.Linear(input_dim, output_dim, bias=bias)

        if self.include_bias==True:
            self.bias = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, node_features, adj_matrix):
      N = adj_matrix.size(0)
      I = torch.eye(N, device=adj_matrix.device, dtype=adj_matrix.dtype)
      A = adj_matrix + I      

      row_sums = A.sum(dim=1, keepdim=True)
      row_sums = torch.clamp(row_sums, min=1.0)          # avoid /0 (isolates stay all-zeros)
      A = A / row_sums 

      node_features_norm = self.weight(A @ node_features)   

      return node_features_norm

class GATLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GATLayer, self).__init__()

    
    ## add the Graph Addition Layer Later 

    