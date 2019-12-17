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

    def forward(self, output1, output2, y):
        euclidean_distance = F.pairwise_distance(output1,output2)
        contrastive_loss = torch.mean((1-y) * torch.pow(euclidean_distance, 2 ) + y * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return contrastive_loss


class TripletLoss(nn.Module):
    """
    Triplet loss function
    Based on: James Philbin. Facenet: A unified embedding for face recognition and clustering. CVPR 2015.
    https://arxiv.org/abs/1503.03832
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, ancohor, postive, negative, size_average=True):
        distance_positive = (anchor - postive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)

        losses = F.relu(distance_positive - distance_negative + self.margin)
        if size_average:
            return losses.mean()  
        else:
            return losses.sum()




