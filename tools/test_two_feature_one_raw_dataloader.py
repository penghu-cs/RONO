import os
import sys
import glob
import h5py
import json
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch


def load_data(partition,dataset_dir):
    all_data = []
    all_label = []
    img_lst = []
    for h5_name in glob.glob(os.path.join(dataset_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        split = h5_name[-4]
        jason_name = dataset_dir+'modelnet40_ply_hdf5_2048/ply_data_' + partition +'_' + split + '_id2file.json'
        with open(jason_name) as json_file:
            images = json.load(json_file)
        img_lst = img_lst + images
        f = h5py.File(h5_name, mode='r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) 
    print(len(all_data), len(all_label), len(img_lst))
    return all_data, all_label, img_lst


def load_modelnet10_data(partition, dataset_dir):
    all_data = []
    all_label = []
    img_lst = []
    for h5_name in glob.glob(os.path.join(dataset_dir, 'modelnet10_hdf5_2048', '%s*.h5'%partition)):
        split = h5_name[-4]
        jason_name = dataset_dir+'modelnet10_hdf5_2048/'+partition + split + '_id2file.json'
        with open(jason_name) as json_file:
            images = json.load(json_file)
        img_lst = img_lst + images
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
    print(len(all_data), len(all_label), len(img_lst))
    return all_data, all_label, img_lst

class TestDataloader(Dataset):
    def __init__(self, dataset, dataset_dir, partition='test'):
        self.dataset_dir = dataset_dir
        self.dataset = dataset
        if self.dataset == 'ModelNet40':
            self.data, self.label, self.img_lst = load_data(partition, self.dataset_dir)
            self.data = np.load('./data/ModelNet40/test_pt_feat_small.npy')
            self.mesh_feat = np.load('./data/ModelNet40/test_mesh_feat.npy')
            self.label = np.load('./data/ModelNet40/test_ori_label.npy')
        else:
            self.data, self.label, self.img_lst = load_modelnet10_data(partition, self.dataset_dir)
            self.data = np.load('./data/ModelNet10/test_pt_feat_small.npy')
            self.mesh_feat = np.load('./data/ModelNet10/test_mesh_feat.npy')
            self.label = np.load('./data/ModelNet10/test_ori_label.npy')

        self.partition = partition

        self.img_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_data(self, item):
        # Get Image Data first
        names = self.img_lst[item]
        names = names.split('/')        
        ##############################
        #random select one image from the images for each object
        img_list = []
        img_index_list = []

        for view in (0,2):
            img_idx = random.randint(0, 179)
            while img_idx in img_index_list:
                img_idx = random.randint(0, 179)
            img_index_list.append(img_idx)
            
            img_names = self.dataset_dir+'ModelNet40-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], view)
            img = Image.open(img_names).convert('RGB')
            img = self.img_transform(img)
            img_list.append(img)
        ##############################
        label = self.label[item]
        ##############################
        pointcloud = self.data[item]
        # choice = np.random.choice(len(pointcloud), self.num_points, replace=True)
        # pointcloud = pointcloud[choice, :]
        return pointcloud, label, img_list[0],img_list[1]




    def get_mesh(self, item):
        mesh_feat = self.mesh_feat[item]
        label = self.label[item]
        

        return mesh_feat,label

    def __getitem__(self, item):

        pt, target, img_1, img_2 = self.get_data(item)
        mesh_feat,target = self.get_mesh(item)
        pt_feat = torch.from_numpy(pt)
        mesh_feat = torch.from_numpy(mesh_feat)
        

        return  img_1, img_2, pt_feat, mesh_feat, target


    def __len__(self):
        return self.data.shape[0]

