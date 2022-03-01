import torch.nn as nn

from .activation_functions import Swish


def init_weights(module, linear_weight_init_type, linear_weight_init_args, linear_bias_fill_value,
                 batch_normalization_weight_fill_value, batch_normalization_bias_fill_value):

    """
    Initialize weights and biases of given layers with specified configurations

    Parameters
    ----------
    module (torch.nn.Module): Layer
    linear_weight_init_type (str): Weight initialization method of the linear layer
    linear_weight_init_args (dict): Weight initialization arguments of the linear layer
    linear_bias_fill_value (float): Bias initialization fill value of the linear layer
    batch_normalization_weight_fill_value (float): Weight initialization fill value of the batch normalization layer
    batch_normalization_bias_fill_value (float): Bias initialization fill value of the batch normalization layer
    """

    if isinstance(module, nn.Linear):
        # Initialize weights of linear layer
        if linear_weight_init_type == 'uniform':
            nn.init.uniform_(
                module.weight,
                a=linear_weight_init_args['a'],
                b=linear_weight_init_args['b']
            )
        elif linear_weight_init_type == 'normal':
            nn.init.normal_(
                module.weight,
                mean=linear_weight_init_args['mean'],
                std=linear_weight_init_args['std']
            )
        elif linear_weight_init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(
                module.weight,
                gain=nn.init.calculate_gain(
                    nonlinearity=linear_weight_init_args['nonlinearity'],
                    param=linear_weight_init_args['nonlinearity_param']
                )
            )
        elif linear_weight_init_type == 'xavier_normal':
            nn.init.xavier_normal_(
                module.weight,
                gain=nn.init.calculate_gain(
                    nonlinearity=linear_weight_init_args['nonlinearity'],
                    param=linear_weight_init_args['nonlinearity_param']
                )
            )
        elif linear_weight_init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(
                module.weight,
                a=linear_weight_init_args['nonlinearity_param'],
                mode=linear_weight_init_args['mode'],
                nonlinearity=linear_weight_init_args['nonlinearity']
            )
        elif linear_weight_init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(
                module.weight,
                a=linear_weight_init_args['nonlinearity_param'],
                mode=linear_weight_init_args['mode'],
                nonlinearity=linear_weight_init_args['nonlinearity']
            )
        elif linear_weight_init_type == 'orthogonal':
            nn.init.orthogonal_(
                module.weight,
                gain=nn.init.calculate_gain(
                    nonlinearity=linear_weight_init_args['nonlinearity'],
                    param=linear_weight_init_args['nonlinearity_param']
                )
            )
        elif linear_weight_init_type == 'sparse':
            nn.init.sparse_(
                module.weight,
                sparsity=linear_weight_init_args['sparsity'],
                std=linear_weight_init_args['std']
            )
        # Initialize biases of Linear layer
        if module.bias is not None:
            nn.init.constant_(module.bias, val=linear_bias_fill_value)

    elif isinstance(module, nn.BatchNorm1d):
        # Initialize weights and biases of batch normalization layer
        nn.init.constant_(module.weight, val=batch_normalization_weight_fill_value)
        nn.init.constant_(module.bias, val=batch_normalization_bias_fill_value)


class DenseBlock(nn.Module):

    def __init__(self, input_dim, output_dim, batch_normalization, weight_normalization, dropout_probability, activation, activation_args, init_args):

        """
        Vanilla dense block (Linear -> Batch Normalization -> Dropout -> Activation)

        Parameters
        ----------
        input_dim (int): Number of input dimensions of the dense layer
        output_dim (int): Number of output dimensions of the dense layer
        batch_normalization (bool): Whether to add batch normalization or not
        weight_normalization (bool): Whether to add weight normalization or not
        dropout_probability (int): Probability of the dropout
        activation (str): Class name of the activation function
        activation_args (dict): Class arguments of the activation function
        init_args (dict): Weight and bias initialization arguments of the layers
        """

        super(DenseBlock, self).__init__()

        if weight_normalization:
            self.linear = nn.utils.weight_norm(nn.Linear(in_features=input_dim, out_features=output_dim, bias=True))
        else:
            self.linear = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)

        if batch_normalization:
            self.batch_normalization = nn.BatchNorm1d(num_features=output_dim)
        else:
            self.batch_normalization = nn.Identity()

        if dropout_probability > 0.0:
            self.dropout = nn.Dropout(p=dropout_probability)
        else:
            self.dropout = nn.Identity()

        if activation is not None:
            self.activation = getattr(nn, activation)(**activation_args)
        elif activation == 'Swish':
            self.activation = Swish()
        else:
            self.activation = nn.Identity()

        if init_args is not None:
            init_weights(self.linear, **init_args)
            if batch_normalization:
                init_weights(self.batch_normalization, **init_args)

    def forward(self, x):

        x = self.linear(x)
        x = self.batch_normalization(x)
        x = self.dropout(x)
        x = self.activation(x)

        return x


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, batch_normalization, weight_normalization, dropout_probability, activation, activation_args, init_args):

        """
        MLP (Multi Layer Perceptron) with multiple dense blocks

        Parameters
        ----------
        input_dim (int): Number of input dimensions of the first dense block
        batch_normalization (bool): Whether to add batch normalization to dense blocks or not
        weight_normalization (bool): Whether to add weight normalization to dense blocks or not
        dropout_probability (int): Probability of the dropout in dense blocks
        activation (str): Class name of the activation function in dense blocks
        activation_args (dict): Class arguments of the activation function in dense blocks
        init_args (dict): Weight and bias initialization arguments of the layers in dense blocks
        """

        super(MultiLayerPerceptron, self).__init__()

        self.dense_block1 = DenseBlock(
            input_dim=input_dim,
            output_dim=512,
            batch_normalization=batch_normalization,
            weight_normalization=weight_normalization,
            dropout_probability=dropout_probability,
            activation=activation,
            activation_args=activation_args,
            init_args=init_args)
        self.dense_block2 = DenseBlock(
            input_dim=512,
            output_dim=768,
            batch_normalization=batch_normalization,
            weight_normalization=weight_normalization,
            dropout_probability=dropout_probability,
            activation=activation,
            activation_args=activation_args,
            init_args=init_args)
        self.dense_block3 = DenseBlock(
            input_dim=768,
            output_dim=1024,
            batch_normalization=batch_normalization,
            weight_normalization=weight_normalization,
            dropout_probability=dropout_probability,
            activation=activation,
            activation_args=activation_args,
            init_args=init_args)
        self.dense_block4 = DenseBlock(
            input_dim=1024,
            output_dim=768,
            batch_normalization=batch_normalization,
            weight_normalization=weight_normalization,
            dropout_probability=dropout_probability,
            activation=activation,
            activation_args=activation_args,
            init_args=init_args)
        self.dense_block5 = DenseBlock(
            input_dim=768,
            output_dim=512,
            batch_normalization=batch_normalization,
            weight_normalization=weight_normalization,
            dropout_probability=dropout_probability,
            activation=activation,
            activation_args=activation_args,
            init_args=init_args)
        self.dense_block6 = DenseBlock(
            input_dim=512,
            output_dim=256,
            batch_normalization=batch_normalization,
            weight_normalization=weight_normalization,
            dropout_probability=dropout_probability,
            activation=activation,
            activation_args=activation_args,
            init_args=init_args)

        self.linear = nn.Linear(256, 1)

    def forward(self, x):

        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = self.dense_block3(x)
        x = self.dense_block4(x)
        x = self.dense_block5(x)
        x = self.dense_block6(x)
        output = self.linear(x)

        return output.view(-1)
