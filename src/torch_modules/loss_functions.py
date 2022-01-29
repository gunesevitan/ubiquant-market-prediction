import torch
import torch.nn as nn


class PearsonCorrelationCoefficientLoss(nn.Module):

    def __init__(self):

        super(PearsonCorrelationCoefficientLoss, self).__init__()

    def forward(self, inputs, targets):

        x = inputs - torch.mean(inputs)
        y = targets - torch.mean(targets)

        loss = torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))
        return -loss


class CosineSimilarityLoss(nn.Module):

    def __init__(self):

        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, inputs, targets):

        x = inputs - torch.mean(inputs)
        y = targets - torch.mean(targets)

        loss = self.cosine_similarity(x.view(-1, 1), y.view(-1, 1))
        return -loss
