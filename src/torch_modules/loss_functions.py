import torch
import torch.nn as nn


class PearsonCorrelationCoefficientLoss(nn.Module):

    def __init__(self):

        super(PearsonCorrelationCoefficientLoss, self).__init__()

    def forward(self, inputs, targets):

        vx = inputs - torch.mean(inputs)
        vy = targets - torch.mean(targets)

        loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 1 - loss
