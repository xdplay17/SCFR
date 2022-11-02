from __future__ import print_function

import torch
import torch.nn as nn

class Contrastive_Loss(nn.Module):
    def __init__(self, temp=0.7):
        super(Contrastive_Loss, self).__init__()
        self.temp = temp

    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        contrast_feature = features

        cosine = torch.div(torch.matmul(contrast_feature, features.T), self.temp)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 1).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        logits = torch.exp(cosine) * logits_mask
        e = torch.log(torch.exp(cosine))
        log_prob = e - torch.log(logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-5)

        loss = - mean_log_prob_pos.mean()

        return loss