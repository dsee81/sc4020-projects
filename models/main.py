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
import networkx as nx
from torch_geometric.utils import degree
from collections import Counter
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from layer import *
import argparse

def train(model, data, epochs, adjacency):

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, adjacency)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss}')

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def test(model, data, adjacency):
    model.eval()
    out = model(data.x, adjacency)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc