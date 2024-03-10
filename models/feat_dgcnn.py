import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.linear1 = nn.Linear(512, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(256, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = F.relu(self.bn6(self.linear1(x)))
        return x