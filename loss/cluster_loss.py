import torch
import torch.nn as nn

class Cluster_Loss(nn.Module):
    def __init__(self, margin=0.5, num_classes=21):
        super(Cluster_Loss, self).__init__()
        self.margin = margin
        self.class_label = torch.tensor(range(num_classes)).cuda()

    def forward(self, features, centers, labels):

        R = (labels.unsqueeze(1) == self.class_label.unsqueeze(0)).float()

        # Hamming Distance
        features = features.sign()
        centers = centers.sign()
        q = features.shape[1]
        distance = 0.5 * (q - torch.matmul(features, centers.T))

        loss = R * distance + (1.0 - R) * (self.margin - distance).clamp(min=0.0)
        loss_mean = loss.sum() / (features.size(0) * centers.size(0))

        return loss_mean


