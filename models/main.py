import torch
import torch.nn as nn
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import torch.nn as nn
import networkx as nx
from torch_geometric.utils import degree
from collections import Counter
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from layer import *
import argparse

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def train(model, data, epochs, adjacency): #adjacency can be adjacency matrix or the edge_index

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) #adam optimizer

    model.train()
    losses =[]
    accur_train = []
    accur_val = []
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, adjacency)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
          pred_train = out.argmax(dim=1)
          train_acc = accuracy(pred_train[data.train_mask], data.y[data.train_mask])
        
        model.eval()
        with torch.no_grad():
          out_val = model(data.x, adjacency)
          pred_val = out_val.argmax(dim=1)
          val_loss = criterion(out_val[data.val_mask], data.y[data.val_mask]).item()
          val_acc = accuracy(pred_val[data.val_mask], data.y[data.val_mask])

        losses.append(loss.detach().numpy())
        accur_train.append(train_acc)
        accur_val.append(val_acc)
        print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | val_loss {val_loss:.4f} | "
      f"train_acc {train_acc:.3f} | val_acc {val_acc:.3f}")

        if epoch == 350:
          visualize(out, color=data.y)
    return (losses, accur_train, accur_val)

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def test(model, data, adjacency):
    model.eval()
    out = model(data.x, adjacency)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

train()