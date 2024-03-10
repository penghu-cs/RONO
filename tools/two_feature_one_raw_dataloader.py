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
import numpy as np

# from tools.visualize import showpoints

def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)


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
        f = h5py.File(h5_name,mode='r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
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


class TripletDataloader(Dataset):
    def __init__(self, dataset, num_classes, dataset_dir, partition='train'):
        self.dataset = dataset
        self.dataset_dir = dataset_dir

        if self.dataset == 'ModelNet40':
            self.data, self.label, self.img_lst = load_data(partition,self.dataset_dir)
            self.label = self.label.squeeze()
            self.data = np.load('./data/ModelNet40/train_pt_feat_small.npy')
            # self.mesh_feat = np.load('../3d22d_noise/extracted_features/feature/ModelNet40/train_mesh_feat.npy')
            self.label = np.load('./data/ModelNet40/train_label_40.npy')
            self.ori_label = np.load('./data/ModelNet40/train_ori_label.npy')
            # self.tag = np.load('./data/ModelNet40/train_tag_40.npy')
        else:
            self.data, self.label, self.img_lst = load_modelnet10_data(partition, self.dataset_dir)
            self.data = np.load('./data/ModelNet10/train_pt_feat_small.npy')
            # self.mesh_feat = np.load('../3d22d_noise/extracted_features/feature/ModelNet10/train_mesh_feat.npy')
            self.label = np.load('./data/ModelNet10/train_label_40.npy')
            self.ori_label = np.load('./data/ModelNet10/train_ori_label.npy')
            # self.tag = np.load('../3d22d_noise/extracted_features/feature/ModelNet10/train_tag_40.npy')
        data, label, img_lst = self.data, self.label, self.img_lst
        # import pdb
        # pdb.set_trace()
        self.partition = partition
        self.num_classes=num_classes

        self.img_train_transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.Resize(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.img_test_transform = transforms.Compose([
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
        
        #randomly select one image from the 12 images for each object
        img_idx = random.randint(0, 179)
        #img_idx = 1
        img_names =self.dataset_dir+'ModelNet40-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], img_idx)
        img = Image.open(img_names).convert('RGB')

        img_idx2 = random.randint(0, 179)
        #img_idx2 = 90
        while img_idx == img_idx2:
            img_idx2 = random.randint(0, 179)

        img_name2 =self.dataset_dir+'ModelNet40-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], img_idx2)
        img2 = Image.open(img_name2).convert('RGB')


        label = self.label[item]
        # ori_label = self.ori_label[item]
        # tag = self.tag[item]

        pointcloud = self.data[item]

        if self.partition == 'train':
            img = self.img_train_transform(img)
            img2 = self.img_train_transform(img2)
        else:
            img = self.img_test_transform(img)
            img2 = self.img_test_transform(img2)

        return pointcloud, label, img, img2


    def get_mesh(self, item):
        mesh_feat = self.mesh_feat[item]
        label = self.label[item]
        ori_label = self.ori_label[item]
        tag = self.tag[item]
        
        return mesh_feat,label,ori_label,tag


    def __getitem__(self, item):

        pt, target, img, img_v = self.get_data(item)
        mesh_feat,target,ori_label,tag = self.get_mesh(item)
        pt_feat = torch.from_numpy(pt)
        mesh_feat = torch.from_numpy(mesh_feat)
        return  img, img_v,pt_feat, mesh_feat, target,ori_label,tag

    def __len__(self):
        return self.data.shape[0]

