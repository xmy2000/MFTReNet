import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, EdgeConv, summary


def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    """
    Helper function to create a 1D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv1d, BatchNorm1d and LeakyReLU layers
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm1d(out_channels),
        # nn.LeakyReLU(),
        nn.Mish(),
    )


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    """
    Helper function to create a 2D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv2d, BatchNorm2d and LeakyReLU layers
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        # nn.LeakyReLU(),
        nn.Mish(),
    )


def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        # nn.LeakyReLU(),
        nn.Mish(),
    )


class _MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        MLP with linear output
        Args:
            num_layers (int): The number of linear layers in the MLP
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden feature dimensions for all hidden layers
            output_dim (int): Output feature dimension

        Raises:
            ValueError: If the given number of layers is <1
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.mish(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class UVNetCurveEncoder(nn.Module):
    def __init__(self, in_channels=6, output_dims=64):
        """
        This is the 1D convolutional network that extracts features from the B-rep edge
        geometry described as 1D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         curve tangents. Defaults to 6.
            output_dims (int, optional): Output curve embedding dimension. Defaults to 64.
        """
        super(UVNetCurveEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(16, 32, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(32, 64, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(64, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class UVNetSurfaceEncoder(nn.Module):
    def __init__(self, in_channels=7, output_dims=64):
        """
        This is the 2D convolutional network that extracts features from the B-rep face
        geometry described as 2D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         surface normals and 1 for the trimming mask. Defaults
                                         to 7.
            output_dims (int, optional): Output surface embedding dimension. Defaults to 64.
        """
        super(UVNetSurfaceEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 16, 3, padding=1, bias=False)
        self.conv2 = _conv2d(16, 32, 3, padding=1, bias=False)
        self.conv3 = _conv2d(32, 64, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(64, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class _NodeConv(nn.Module):
    def __init__(
            self,
            node_feats,
            out_feats,
            edge_feats,
    ):
        """
        This module implements Eq. 1 from the paper where the node features are
        updated using the neighboring node and edge features.

        Args:
            node_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_NodeConv, self).__init__()
        self.graph_conv = NNConv(
            in_channels=node_feats,
            out_channels=out_feats,
            nn=nn.Linear(edge_feats, node_feats * out_feats),
            aggr="add",
            bias=True,
        )
        self.batch_norm = nn.BatchNorm1d(out_feats)

    def forward(self, node_feature, edge_index, edge_feature):
        h = self.graph_conv(node_feature, edge_index, edge_feature)  # num_nodes * out_channels
        h = F.leaky_relu(self.batch_norm(h))
        return h


class _EdgeConv(nn.Module):
    def __init__(
            self,
            node_feats,
            out_feats,
    ):
        """
        This module implements Eq. 2 from the paper where the edge features are
        updated using the node features at the endpoints.

        Args:
            edge_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_EdgeConv, self).__init__()
        self.conv = EdgeConv(nn=nn.Linear(2 * node_feats, out_feats), aggr="add")
        self.batch_norm = nn.BatchNorm1d(out_feats)

    def forward(self, node_features, edge_index):
        h = self.conv(node_features, edge_index)  # num_nodes * out_channels
        h = F.leaky_relu(self.batch_norm(h))
        return h


class UVNetGraphEncoder(nn.Module):
    def __init__(
            self,
            input_node_dim,
            input_edge_dim,
            output_dim,
            hidden_dim=64,
            num_layers=3
    ):
        """
        This is the graph neural network used for message-passing features in the
        face-adjacency graph.  (see Section 3.2, Message passing in paper)

        Args:
            input_dim ([type]): [description]
            input_edge_dim ([type]): [description]
            output_dim ([type]): [description]
            hidden_dim (int, optional): [description]. Defaults to 64.
            learn_eps (bool, optional): [description]. Defaults to True.
            num_layers (int, optional): [description]. Defaults to 3.
            num_mlp_layers (int, optional): [description]. Defaults to 2.
        """
        super(UVNetGraphEncoder, self).__init__()
        self.num_layers = num_layers

        # List of layers for node and edge feature message passing
        self.node_conv_layers = torch.nn.ModuleList()
        # self.edge_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_feats = input_node_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                _NodeConv(
                    node_feats=node_feats,
                    out_feats=hidden_dim,
                    edge_feats=edge_feats
                )
            )
            # self.edge_conv_layers.append(
            #     _EdgeConv(
            #         node_feats=node_feats,
            #         out_feats=hidden_dim
            #     )
            # )

        self.drop = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features, edge_index, edge_features, batch):
        for i in range(self.num_layers - 1):
            # use node_conv
            node_features = self.node_conv_layers[i](node_features, edge_index,
                                                     edge_features)  # num_nodes * hidden_channels
            # use edge_conv
            # he = self.edge_conv_layers[i](node_features, edge_index)  # num_nodes * out_channels

        out = self.drop(node_features)
        out = self.linear(out)

        return out
