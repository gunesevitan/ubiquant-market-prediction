import torch.nn as nn


class DenseBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super(DenseBlock, self).__init__()
        self.dense_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x):

        x = self.dense_block(x)
        return x


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim):

        super(MultiLayerPerceptron, self).__init__()

        self.dense_block1 = DenseBlock(input_dim=input_dim, hidden_dim=128, output_dim=256)
        self.dense_block2 = DenseBlock(input_dim=256, hidden_dim=384, output_dim=512)
        self.dense_block3 = DenseBlock(input_dim=512, hidden_dim=768, output_dim=1024)
        self.dense_block4 = DenseBlock(input_dim=1024, hidden_dim=768, output_dim=512)
        self.dense_block5 = DenseBlock(input_dim=512, hidden_dim=384, output_dim=256)
        self.dense_block6 = DenseBlock(input_dim=256, hidden_dim=128, output_dim=64)
        self.linear = nn.Linear(64, 1)

    def forward(self, x):

        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = self.dense_block3(x)
        x = self.dense_block4(x)
        x = self.dense_block5(x)
        x = self.dense_block6(x)
        output = self.linear(x)

        return output.view(-1)
