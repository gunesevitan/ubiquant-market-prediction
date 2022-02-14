import torch
import torch.nn as nn


class DenseEncoder(nn.Module):

    def __init__(self, input_dim, latent_space_dim):

        super(DenseEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=256, bias=True),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=latent_space_dim, bias=True)
        )

    def forward(self, x):

        output = self.encoder(x)
        return output


class DenseDecoder(nn.Module):

    def __init__(self, output_dim, latent_space_dim):

        super(DenseDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_space_dim, out_features=128, bias=True),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=256, bias=True),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=output_dim, bias=True)
        )

    def forward(self, x):

        output = self.decoder(x)
        return output


class DenseAutoEncoder(nn.Module):

    def __init__(self, input_dim, latent_space_dim):

        super(DenseAutoEncoder, self).__init__()

        self.encoder = DenseEncoder(input_dim=input_dim, latent_space_dim=latent_space_dim)
        self.decoder = DenseDecoder(output_dim=input_dim, latent_space_dim=latent_space_dim)
        self.mask_head = torch.nn.Linear(in_features=input_dim, out_features=input_dim)
        self.reconstruction_head = torch.nn.Linear(in_features=input_dim * 2, out_features=input_dim)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        mask_output = torch.sigmoid(self.mask_head(x))
        reconstruction_output = self.reconstruction_head(torch.cat([x, mask_output], dim=1))

        return mask_output, reconstruction_output


class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, feedforward_dim):

        super(TransformerEncoder).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=feedforward_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=feedforward_dim, out_features=embed_dim, bias=True)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        attention_output, attention_output_weights = self.attention(x, x, x)
        x = self.layernorm1(x + attention_output)
        feedforward_output = self.feed_forward(x)
        x = self.layer_norm2(x + feedforward_output)

        return x


class DAELoss(nn.Module):

    def __init__(self, lambda_=0.5):

        super(DAELoss, self).__init__()

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.lambda_ = lambda_

    def forward(self, x_ground_truth, x_predictions, mask_ground_truth, mask_predictions):

        """
        Calculate reconstruction and mask losses

        Parameters
        ----------
        x_ground_truth [torch.FloatTensor of shape (batch_size, n_dimensions)]: Ground-truth input tensor
        x_predictions [torch.FloatTensor of shape (batch_size, n_dimensions)]: Reconstructed input tensor
        mask_ground_truth [torch.FloatTensor of shape (batch_size, n_dimensions)]: Ground-truth binary swap mask
        mask_predictions [torch.FloatTensor of shape (batch_size, n_dimensions)]:  Predicted swap mask probabilities

        Returns
        -------
        total_loss [torch.FloatTensor of shape (batch_size)]: Reconstruction and mask losses
        """

        reconstruction_loss = self.mse_loss(x_predictions, x_ground_truth)
        mask_loss = self.bce_loss(mask_predictions, mask_ground_truth)
        total_loss = (self.lambda_ * reconstruction_loss) + ((1 - self.lambda_) * mask_loss)

        return total_loss


class SwapNoise:

    def __init__(self, swap_probabilities):

        self.swap_probabilities = torch.from_numpy(swap_probabilities)

    def apply(self, x):

        """
        Swap input values randomly for every dimension based on given probabilities

        Parameters
        ----------
        x [torch.FloatTensor of shape (batch_size, n_dimensions)]: Input tensor

        Returns
        -------
        x_swapped [torch.FloatTensor of shape (batch_size, n_dims)]: Input tensor with randomly swapped values
        swap_mask [torch.FloatTensor of shape (batch_size, n_dims)]: Binary mask tensor of swapped input values
        """

        # Create binary swap matrix sampled from a Bernoulli distribution with specified probabilities
        swap_matrix = torch.bernoulli(self.swap_probabilities.to(x.device) * torch.ones(x.shape).to(x.device))
        x_swapped = torch.where(swap_matrix == 1, x[torch.randperm(x.shape[0])], x)
        swap_mask = (x_swapped != x).float()

        return x_swapped, swap_mask
