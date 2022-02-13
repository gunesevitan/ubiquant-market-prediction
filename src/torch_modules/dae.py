import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TransformerAutoEncoder(nn.Module):

    def __init__(self, num_inputs, hidden_size=1024, num_subspaces=8, embed_dim=128, num_heads=8, dropout=0, feedforward_dim=512, emphasis=.75, task_weights=[10, 14], mask_loss_weight=2):

        super(TransformerAutoEncoder).__init__()
        assert hidden_size == embed_dim * num_subspaces
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.num_subspaces = num_subspaces
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.emphasis = emphasis
        self.task_weights = np.array(task_weights) / sum(task_weights)
        self.mask_loss_weight = mask_loss_weight

        self.excite = torch.nn.Linear(in_features=num_inputs, out_features=hidden_size)
        self.encoder_1 = TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)
        self.encoder_2 = TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)
        self.encoder_3 = TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim)

        self.mask_predictor = torch.nn.Linear(in_features=hidden_size, out_features=num_inputs)
        self.reconstructor = torch.nn.Linear(in_features=hidden_size + num_inputs, out_features=num_inputs)

    def divide(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.num_subspaces, self.embed_dim)).permute((1, 0, 2))
        return x

    def combine(self, x):
        batch_size = x.shape[1]
        x = x.permute((1, 0, 2)).reshape((batch_size, -1))
        return x

    def forward(self, x):
        x = torch.nn.functional.relu(self.excite(x))

        x = self.divide(x)
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x = self.combine(x3)

        predicted_mask = self.mask_predictor(x)
        reconstruction = self.reconstructor(torch.cat([x, predicted_mask], dim=1))
        return (x1, x2, x3), (reconstruction, predicted_mask)

    def split(self, t):
        return torch.split(t, [self.n_cats, self.n_nums], dim=1)

    def feature(self, x):
        attn_outs, _ = self.forward(x)
        return torch.cat([self.combine(x) for x in attn_outs], dim=1)

    def loss(self, x, y, mask, reduction='mean'):
        _, (reconstruction, predicted_mask) = self.forward(x)
        x_cats, x_nums = self.split(reconstruction)
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(mask * self.emphasis + (1 - mask) * (1 - self.emphasis))

        cat_loss = self.task_weights[0] * torch.mul(w_cats, bce_logits(x_cats, y_cats, reduction='none'))
        num_loss = self.task_weights[1] * torch.mul(w_nums, mse(x_nums, y_nums, reduction='none'))

        reconstruction_loss = torch.cat([cat_loss, num_loss],
                                        dim=1) if reduction == 'none' else cat_loss.mean() + num_loss.mean()
        mask_loss = self.mask_loss_weight * bce_logits(predicted_mask, mask, reduction=reduction)

        return reconstruction_loss + mask_loss if reduction == 'mean' else [reconstruction_loss, mask_loss]


class DAELoss(nn.Module):

    def __init__(self, loss_weights=0.5):

        super(DAELoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.loss_weights = loss_weights

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

        reconstruction_loss = self.mse_loss(x_ground_truth, x_predictions)
        mask_loss = self.bce_loss(mask_ground_truth, mask_predictions)
        total_loss = (self.loss_weights * reconstruction_loss) + ((1 - self.loss_weights) * mask_loss)

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
