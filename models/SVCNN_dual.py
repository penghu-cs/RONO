# from models.MVCNN import MVCNN
from __future__ import division, absolute_import
from models.dgcnn import DGCNN
from models.resnet import resnet18
from tools.utils import calculate_accuracy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse
import torch.optim as optim
import time

# from .Model import Model

class SingleViewNet(nn.Module):

    def __init__(self, pre_trained = None):
        super(SingleViewNet, self).__init__()

        if pre_trained:
            self.img_net = torch.load(pre_trained)
        else:
            print('---------Loading ImageNet pretrained weights --------- ')
            resnet18 = models.resnet18(pretrained=True)
            resnet18 = list(resnet18.children())[:-1]
            self.img_net = nn.Sequential(*resnet18)
            self.linear1 = nn.Linear(512, 256, bias=True)
            self.bn6 = nn.BatchNorm1d(256)

    def forward(self, img, img_v):

        img_feat = self.img_net(img)
        img_feat_v = self.img_net(img_v)
        img_feat = img_feat.squeeze(3)
        img_feat = img_feat.squeeze(2)
        img_feat_v = img_feat_v.squeeze(3)
        img_feat_v = img_feat_v.squeeze(2)

        img_feat = F.relu(self.bn6(self.linear1(img_feat)))
        img_feat_v = F.relu(self.bn6(self.linear1(img_feat_v)))

        # final_feat = img_feat
        final_feat = 0.5*(img_feat + img_feat_v)

        return final_feat

class CorrNet(nn.Module):

    def __init__(self, img_net, pt_net, mesh_net, num_classes):
        super(CorrNet, self).__init__()
        self.img_net = img_net
        self.pt_net = pt_net
        self.mesh_net = mesh_net
        self.num_classes=num_classes
	    #shared head for all feature encoders
        self.head = nn.Sequential(*[nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, self.num_classes)])

    def forward(self, pt, img, img_v, centers, corners, normals, neighbor_index):

	#extract image features
        img_feat = self.img_net(img, img_v)

        pt_feat = self.pt_net(pt)
	#extract mesh features
        mesh_feat = self.mesh_net(centers, corners, normals, neighbor_index)

        cmb_feat = (img_feat  + mesh_feat +pt_feat)/3.0

	#the classification predictions based on image features
        img_pred = self.head(img_feat)

        pt_pred = self.head(pt_feat)
	#the classification prediction based on mesh featrues
        mesh_pred = self.head(mesh_feat)

        cmb_pred = self.head(cmb_feat)

        return img_pred, pt_pred, mesh_pred, img_feat,pt_feat, mesh_feat

class HeadNet(nn.Module):

    def __init__(self, num_classes):
        super(HeadNet, self).__init__()
        self.num_classes=num_classes
	    #shared head for all feature encoders
        self.head = nn.Sequential(*[nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.num_classes)])

    def forward(self, img_feat, pt_feat):

	
	#the classification predictions based on image features
        img_pred = self.head(img_feat)

        pt_pred = self.head(pt_feat)



        return img_pred, pt_pred


class HeadNet_3(nn.Module):

    def __init__(self, num_classes):
        super(HeadNet_3, self).__init__()
        self.num_classes=num_classes
	    #shared head for all feature encoders
        self.head = nn.Sequential(*[nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, self.num_classes)])

    def forward(self, img_feat, pt_feat,mesh_feat):

	
	#the classification predictions based on image features
        img_pred = self.head(img_feat)
        mesh_pred = self.head(mesh_feat)
        pt_pred = self.head(pt_feat)



        return img_pred, pt_pred,mesh_pred