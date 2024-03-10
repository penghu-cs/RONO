import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class RDC_loss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self,alpha = 1, num_classes = 10, feat_dim=512, warmup=15,a=0.2):
        super(RDC_loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.warmup = warmup
        self.alpha1 = alpha
        self.alpha2 = alpha*50
        self.a = a
        
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, x, labels, ori_labels, epoch):
        """
        Args:
            centers: (num_class, feat_dim)
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # normalization
        x = F.normalize(x, p=2, dim=1)
        centers = self.centers
        centers = F.normalize(centers, p=2, dim=1)

        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().cuda()
        all_centers_sum = torch.div(centers.sum(dim=0).unsqueeze(0).repeat(batch_size,1),self.num_classes)  # [288,512]
        compute_center = centers.unsqueeze(0).repeat(batch_size,1,1)  # [288,40,512]
        compute_one_hot = label_one_hot.unsqueeze(2).repeat(1,1,self.feat_dim) # [288,40,512]
        one_centers_sum = torch.div(torch.mul(compute_center, compute_one_hot).sum(1),(self.num_classes/(self.num_classes+1)))
        #loss = torch.div(torch.mul((all_centers_sum - one_centers_sum),x).sum(1).sum(0),(self.alpha*self.num_classes))
        loss_1 = torch.div((torch.exp(torch.mul(all_centers_sum,x).sum(1)) - torch.exp(torch.mul(one_centers_sum,x).sum(1))),self.alpha1).sum(0) / batch_size


        one_centers_sum = torch.mul(compute_center, compute_one_hot).sum(1)
        loss_2 = (-torch.abs(torch.div(torch.add(torch.exp(torch.mul(all_centers_sum,x).sum(1))- torch.exp(torch.mul(one_centers_sum,x).sum(1)),self.a),self.alpha2))).sum(0)/batch_size 
        
        v = min(epoch/(self.warmup-1), 1)

        loss = (1-v) * loss_1 + v * loss_2

        # x_ = x.unsqueeze(1).repeat(1,self.num_classes,1)
        # dist = torch.nn.functional.pairwise_distance(x_, centers, p=2)
        # print("________________")
        # print(torch.eq(torch.min(dist,dim=1).indices, labels).sum(dim=0)/batch_size)
        # print(torch.eq(torch.min(dist,dim=1).indices, ori_labels).sum(dim=0)/batch_size)

        return loss, self.centers