import numpy as np
import os
from torch.autograd import Variable
import argparse
import torch
from sklearn.preprocessing import normalize
import scipy
from tools.test_two_feature_one_raw_dataloader import TestDataloader


def extract(args):
    ''' The model we trained '''
    # img_net = torch.load('./checkpoints/ModelNet40/img_net.pkl' % (args.model_folder, args.iterations),
    #                      map_location=lambda storage, loc: storage)
    # img_net = img_net.eval()
    # dgcnn = torch.load('./checkpoints/ModelNet40/pt_net.pkl' % (args.model_folder, args.iterations),
    #                    map_location=lambda storage, loc: storage)
    
    ''' Reader-customized trained model '''
    img_net = torch.load('./checkpoints/%s/test_result/%d-img_net.pkl' % (args.model_folder, args.iterations),
                         map_location=lambda storage, loc: storage)
    img_net = img_net.eval()
    dgcnn = torch.load('./checkpoints/%s/test_result/%d-pt_net.pkl' % (args.model_folder, args.iterations),
                       map_location=lambda storage, loc: storage)
    dgcnn = dgcnn.eval()
    

    torch.cuda.empty_cache()
    #################################
    test_set = TestDataloader(dataset=args.dataset, dataset_dir=args.dataset_dir, partition='test')
    data_loader_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    print('length of the dataset: ', len(test_set))
    #################################
    img_feat_list = np.zeros((len(test_set), 256))
    pt_feat_list = np.zeros((len(test_set), 256))
    mesh_feat_list = np.zeros((len(test_set), 512))
    label = np.zeros((len(test_set)))
    #################################
    iteration = 0
    for data in data_loader_loader:
        img_feat_1, img_feat_2, pt_feat, mesh_feat, target = data

        img_feat_1 = Variable(img_feat_1).to(torch.float32).to('cuda')
        img_feat_2 = Variable(img_feat_2).to(torch.float32).to('cuda')
        pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
        target = Variable(target).to(torch.long).to('cuda')
        ##########################################
        img_net = img_net.to('cuda')
        feat_view = img_net(img_feat_1, img_feat_2)
        dgcnn = dgcnn.to('cuda')
        cloud_feat = dgcnn(pt_feat)


        ########################################
        img_feat_list[iteration, :] = feat_view.data.cpu().numpy()
        pt_feat_list[iteration, :] = cloud_feat.data.cpu().numpy()

        label[iteration] = target.data.cpu().numpy()
        iteration = iteration + 1
    np.save(args.save + '/img_feat', img_feat_list)
    np.save(args.save + '/pt_feat', pt_feat_list)
    np.save(args.save + '/label', label)


def fx_calc_map_label(view_1, view_2, label_test):
    dist = scipy.spatial.distance.cdist(view_1, view_2, 'cosine')  # rows view_1 , columns view 2
    ord = dist.argsort()
    numcases = dist.shape[0]
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(numcases):
            if label_test[i] == label_test[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)


def eval_func(img_pairs):
    print('number of img views: ', img_pairs)
    img_feat = np.load(args.save + '/img_feat.npy')
    pt_feat = np.load(args.save + '/pt_feat.npy')
    label = np.load(args.save + '/label.npy')
    ########################################
    img_test = normalize(img_feat, norm='l1', axis=1)
    cloud_test = normalize(pt_feat, norm='l1', axis=1)

    ########################################
    # par_list = [
    #     (img_feat, img_feat, 'Image2Image'),
    #     (img_feat, mesh_feat, 'Image2Mesh'),
    #     (img_feat, pt_feat, 'Image2Point'),
    #     (mesh_feat, mesh_feat, 'Mesh2Mesh'),
    #     (mesh_feat, img_feat, 'Mesh2Image'),
    #     (mesh_feat, pt_feat, 'Mesh2Point'),
    #     (pt_feat, pt_feat, 'Point2Point'),
    #     (pt_feat, img_feat, 'Point2Image'),
    #     (pt_feat, mesh_feat, 'Point2Mesh')]

    par_list = [
        (img_test, img_test, 'Image2Image'),
        (img_test, cloud_test, 'Image2Point'),
        (cloud_test, cloud_test, 'Point2Point'),
        (cloud_test, img_test, 'Point2Image')]
    ########################################
    mean_acc = 0
    for index in range(4):
        view_1, view_2, name = par_list[index]
        print(name + '---------------------------')
        acc = fx_calc_map_label(view_1, view_2, label)
        acc_round = round(acc * 100, 2)
        print(str(acc_round))
        mean_acc = mean_acc + acc
    print("mean acc: "+ str(mean_acc/4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')

    parser.add_argument('--dataset', type=str, default='ModelNet40', metavar='dataset', help='ModelNet10 or ModelNet40')

    parser.add_argument('--dataset_dir', type=str, default='./datasets/',
                        metavar='dataset_dir', help='dataset_dir')

    #parser.add_argument('--model_folder', type=str, default='ModelNet40', help='path to load model')

    parser.add_argument('--model_folder', type=str, default='ModelNet40/test_result', help='path to load model')

    parser.add_argument('--iterations', type=int, default=15400, help='iteration to load the model')

    parser.add_argument('--gpu_id', type=str, default='0', help='GPU used to train the network')

    parser.add_argument('--save', type=str, default='./extracted_features/ModelNet40/test_retult', help='save features')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    extract(args)
    eval_func(1)

