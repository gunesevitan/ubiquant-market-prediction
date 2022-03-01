import torch
import torch.nn as nn
import torch.nn.functional as F


class PearsonCorrelationCoefficientLoss(nn.Module):

    def __init__(self):

        super(PearsonCorrelationCoefficientLoss, self).__init__()

    def forward(self, inputs, targets):

        x = inputs - torch.mean(inputs)
        y = targets - torch.mean(targets)

        loss = torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))
        return -loss


class WeightedRegressionCorrelationLoss(nn.Module):

    def __init__(self, w):

        super(WeightedRegressionCorrelationLoss, self).__init__()

        self.w = w

    def forward(self, inputs, targets):

        x = inputs - torch.mean(inputs)
        y = targets - torch.mean(targets)

        pearson_correlation_coefficient_loss = torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))
        mse_loss = F.mse_loss(inputs, targets)
        loss = (self.w * -1 * pearson_correlation_coefficient_loss) + (1 - self.w * mse_loss)

        return loss
