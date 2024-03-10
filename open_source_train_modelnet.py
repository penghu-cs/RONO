from __future__ import division, absolute_import
from pickle import FALSE
import os
import time
import torch
import argparse
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from MAE import MeanAbsoluteError
from models.feat_dgcnn import DGCNN
from models.SVCNN_dual import SingleViewNet, HeadNet
from tools.two_feature_one_raw_dataloader import TripletDataloader
from tools.utils import calculate_accuracy
from rdc_loss import RDC_loss
from cross_modal_loss import CrossModalLoss
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
#from torch.utils.tensorboard import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def training(args):
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    img_net = SingleViewNet(pre_trained = None)
    pt_net = DGCNN(args)
    head_net = HeadNet(num_classes=args.num_classes)

    img_net.train(True)
    pt_net.train(True)
    head_net.train(True)
    img_net = img_net.to('cuda')
    pt_net = pt_net.to('cuda')
    head_net = head_net.to('cuda')
    

    ce_criterion = MeanAbsoluteError(num_classes=args.num_classes)

    cmc_criterion = RDC_loss(num_classes=args.num_classes,alpha=args.alpha, feat_dim=args.emb_dims, warmup=args.warm_up)
   
    mse_criterion = CrossModalLoss()
 
    optimizer_img = optim.SGD(img_net.parameters(), lr=args.lr_img, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_pt = optim.SGD(pt_net.parameters(), lr=args.lr_pt, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer_head = optim.SGD(head_net.parameters(), lr=args.lr_head, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_centloss = optim.SGD(cmc_criterion.parameters(), lr=args.lr_center)

    train_set = TripletDataloader(dataset=args.dataset, dataset_dir=args.dataset_dir,num_classes=args.num_classes, partition='train')  
    data_loader_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
   
    iteration = 0
    start_time = time.time()



    for epoch in range(args.epochs):
        for data in data_loader_loader:
            img_feat_1, img_feat_2, pt_feat, mesh_feat, target, ori_label, _ = data
            
            
            img_feat_1 = Variable(img_feat_1).to(torch.float32).to('cuda')
            img_feat_2 = Variable(img_feat_2).to(torch.float32).to('cuda')
            pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')

            target = Variable(target).to(torch.long).to('cuda')
            ori_label = Variable(ori_label).to(torch.long).to('cuda')

            optimizer_img.zero_grad()
            optimizer_pt.zero_grad()
            optimizer_head.zero_grad()
            optimizer_centloss.zero_grad()


            _img_feat = img_net(img_feat_1,img_feat_2)
            _pt_feat = pt_net(pt_feat)

            _img_pred, _pt_pred = head_net(_img_feat, _pt_feat)
            
            #cross-entropy loss for all the three modalities
            
            pt_ce_loss = ce_criterion(_pt_pred, target)
            img_ce_loss = ce_criterion(_img_pred, target)          
            ce_loss = pt_ce_loss + img_ce_loss 
           
            cmc_loss, centers = cmc_criterion(torch.cat((_img_feat, _pt_feat), dim = 0), torch.cat((target, target), dim = 0),torch.cat((ori_label, ori_label), dim = 0),  epoch)

            mse_loss = mse_criterion(torch.cat((_img_feat, _pt_feat), dim = 0))


      
	
	    
            loss = args.weight_ce * ce_loss +  args.weight_center * cmc_loss +  args.weight_mse * mse_loss
        
            #loss = 0.5 * ce_loss + args.weight_center * cmc_loss + mse_loss

            loss.backward()

            optimizer_img.step()
            optimizer_pt.step()
            optimizer_head.step()
            optimizer_centloss.step()

            img_acc = calculate_accuracy(_img_pred, ori_label)
            pt_acc = calculate_accuracy(_pt_pred, ori_label)



            if (iteration%args.lr_step) == 0:
                lr_img = args.lr_img * (0.1 ** (iteration // args.lr_step))
                lr_pt = args.lr_pt * (0.1 ** (iteration // args.lr_step))
                lr_head = args.lr_head * (0.1 ** (iteration // args.lr_step))
                
                for param_group in optimizer_img.param_groups:
                    param_group['lr_img'] = lr_img
                for param_group in optimizer_pt.param_groups:
                    param_group['lr_pt'] = lr_pt
                for param_group in optimizer_head.param_groups:
                    param_group['lr_head'] = lr_head
            


            # update the learning rate of the center loss
            if (iteration%args.center_lr_step) == 0:
                lr_center = args.lr_center * (0.1 ** (iteration // args.lr_step))
                print('New Center LR:     ' + str(lr_center))
                for param_group in optimizer_centloss.param_groups:
                    param_group['lr'] = lr_center


            if iteration % args.per_print == 0:
                print("loss: %f  center_loss: %f  ce_loss: %f  mse_loss: %f" % (loss.item(), cmc_loss.item(), ce_loss, mse_loss))
                print('[%d][%d]  img_acc: %f pt_acc %f time: %f  vid: %d' % (epoch, iteration, img_acc, pt_acc,time.time() - start_time, target.size(0))) 
                #print("loss: %f  center_loss: %f  ce_loss: %f  mse_loss: %f" % (loss.item(), args.weight_center * cmc_loss.item(), ce_loss,  0.1 * mse_loss))
                start_time = time.time()

          
            iteration = iteration + 1
         
            if((iteration+1) % args.per_save) ==0:
                print('----------------- Save The Network ------------------------')
                with open(args.save + str(iteration+1)+'-head_net.pkl', 'wb') as f:
                    torch.save(head_net, f)
                with open(args.save + str(iteration+1)+'-img_net.pkl', 'wb') as f:
                    torch.save(img_net, f)
                with open(args.save + str(iteration+1)+'-pt_net.pkl', 'wb') as f:
                    torch.save(pt_net, f)
                np.save(args.save + str(iteration+1)+'-centers', centers.cpu().detach().numpy())
                        
                    

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')
    ### 自定义
    parser.add_argument('--k', type=int, default=20, metavar='k',
                        help='k')

    parser.add_argument('--emb_dims', type=str, default=256, metavar='emb_dims',
                        help='emb_dims')

    parser.add_argument('--dropout', type=str, default=0.5, metavar='dropout',
                        help='dropout')
    
    ''' Please modify this to change the dataset '''
    parser.add_argument('--dataset', type=str, default='ModelNet40', metavar='dataset',
                        help='ModelNet10 or ModelNet40')
    
    parser.add_argument('--num_classes', type=int, default=40, metavar='num_classes',
                        help='10 or 40')

    parser.add_argument('--dataset_dir', type=str, default='./datasets/', metavar='dataset_dir',
                        help='dataset_dir')
    
    '''  Custom settings '''
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch)')

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train 200')
    
    parser.add_argument('--lr_img', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--lr_pt', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--lr_head', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')

    parser.add_argument('--lr_step', type=int, default=3500,  
                        help='how many iterations to decrease the learning rate')   # 10: 1500, 40: 3500
    
    parser.add_argument('--center_lr_step', type=int,  default=3500,
                        help='how many iterations to decrease the learning rate')

    parser.add_argument('--lr_center', type=float, default=0.001, metavar='LR',
                        help='learning rate for center loss (default: 0.5)  0.001')
                                         
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    #DGCNN
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    
    parser.add_argument('--warm_up', type=float, default=20, metavar='M',
                        help='SGD momentum (default: 0.9)')   #15 - 30
    #loss
    parser.add_argument('--alpha', type=float, default=0.01, metavar='alpha',
                        help='alpha' )

    parser.add_argument('--weight_center', type=float, default=1, metavar='weight_center',
                        help='weight center ' )   # 0.1

    parser.add_argument('--weight_ce', type=float, default=0.1, metavar='weight_ce',
                        help='weight ce' )
                        
    parser.add_argument('--weight_mse', type=float, default=0.3, metavar='weight_mse',
                        help='weight mse' )

    parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='weight_decay',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--per_save', type=int,  default=200,
                        help='how many iterations to save the model')

    parser.add_argument('--per_print', type=int,  default=100,
                        help='how many iterations to print the loss and accuracy')
                        
    parser.add_argument('--save', type=str,  default='./checkpoints/ModelNet40/test_result/',
                        help='path to save the final model')

    parser.add_argument('--gpu_id', type=str,  default='0,1',
                        help='GPU used to train the network')

    parser.add_argument('--log', type=str,  default='log/',
                        help='path to the log information') #9000

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = True
    training(args)