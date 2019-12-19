import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self,margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        

    def forward(self, output1, output2, target, size_average=True):
        distances  = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss function
    Based on: James Philbin. Facenet: A unified embedding for face recognition and clustering. CVPR 2015.
    https://arxiv.org/abs/1503.03832
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        print('triplet loss')

    def forward(self, ancohor, postive, negative, size_average=True):
        distance_positive = (anchor - postive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)

        losses = F.relu(distance_positive - distance_negative + self.margin)
        if size_average:
            return losses.mean()  
        else:
            return losses.sum()




