import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, global_mean_pool, summary, GATv2Conv, ResGatedGraphConv, TransformerConv, \
    GMMConv, GINEConv, MLP as tgmlp, GraphNorm
from torch_geometric.utils import dropout_path


class GraphEncoder(nn.Module):
    def __init__(
            self,
            node_dim,
            num_layers
    ):
        super(GraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.node_convs = nn.ModuleList()
        # self.edge_convs = nn.ModuleList()

        for _ in range(num_layers):
            self.node_convs.append(
                GENConv(
                    in_channels=node_dim,
                    out_channels=node_dim,
                    aggr="softmax",
                    learn_t=True,
                    learn_p=True,
                    msg_norm=True,
                    learn_msg_scale=True,
                    norm='layer',
                    num_layers=2,
                    expansion=2,
                    eps=1e-07,
                    bias=True
                )
            )

        self.linear = nn.Linear(node_dim, node_dim)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

    def forward(self, node_feature, edge_index, edge_feature, batch):
        for i in range(self.num_layers):
            node_feature = self.node_convs[i](node_feature, edge_index, edge_feature)
        local_feature = self.norm1(node_feature)

        global_feature = global_mean_pool(local_feature, batch)
        global_feature = self.linear(global_feature)
        global_feature = self.norm2(global_feature)

        return local_feature, global_feature


class GraphEncoder_GAT(nn.Module):
    def __init__(self, node_dim, edge_dim, head, num_layers):
        super(GraphEncoder_GAT, self).__init__()
        self.num_layers = num_layers
        self.node_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.node_convs.append(
                GATv2Conv(
                    in_channels=-1,
                    out_channels=node_dim,
                    heads=head,
                    concat=True,
                    dropout=0.5,
                    edge_dim=edge_dim
                )
            )
        self.node_linear = nn.Linear(node_dim * head, node_dim)
        self.graph_linear = nn.Linear(node_dim * head, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.graph_norm = nn.LayerNorm(node_dim)

    def forward(self, node_feature, edge_index, edge_feature, batch):
        for i in range(self.num_layers - 1):
            node_feature = self.node_convs[i](node_feature, edge_index, edge_feature)
            node_feature = node_feature.relu()
        node_feature = self.node_convs[-1](node_feature, edge_index, edge_feature)

        local_feature = self.node_norm(self.node_linear(node_feature))
        global_feature = global_mean_pool(node_feature, batch)
        global_feature = self.graph_norm(self.graph_linear(global_feature))

        return local_feature, global_feature


class GraphEncoder_ResGate(nn.Module):
    def __init__(self, node_dim, edge_dim, num_layers):
        super(GraphEncoder_ResGate, self).__init__()
        self.num_layers = num_layers
        self.node_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.node_convs.append(
                ResGatedGraphConv(
                    in_channels=node_dim,
                    out_channels=node_dim,
                    edge_dim=edge_dim
                )
            )
        self.graph_linear = nn.Linear(node_dim, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.graph_norm = nn.LayerNorm(node_dim)

    def forward(self, node_feature, edge_index, edge_feature, batch):
        for i in range(self.num_layers - 1):
            node_feature = self.node_convs[i](node_feature, edge_index, edge_feature)
            node_feature = node_feature.relu()
            node_feature = F.dropout(node_feature, p=0.6, training=self.training)
        node_feature = self.node_convs[-1](node_feature, edge_index, edge_feature)

        local_feature = self.node_norm(node_feature)
        global_feature = global_mean_pool(node_feature, batch)
        global_feature = self.graph_norm(self.graph_linear(global_feature))

        return local_feature, global_feature


class GraphEncoder_Transformer(nn.Module):
    def __init__(self, node_dim, edge_dim, head, num_layers):
        super(GraphEncoder_Transformer, self).__init__()
        self.num_layers = num_layers
        self.node_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.node_convs.append(
                TransformerConv(
                    in_channels=-1,
                    out_channels=node_dim,
                    heads=head,
                    concat=True,
                    dropout=0.5,
                    edge_dim=edge_dim
                )
            )
        self.node_linear = nn.Linear(node_dim * head, node_dim)
        self.graph_linear = nn.Linear(node_dim * head, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.graph_norm = nn.LayerNorm(node_dim)

    def forward(self, node_feature, edge_index, edge_feature, batch):
        for i in range(self.num_layers - 1):
            node_feature = self.node_convs[i](node_feature, edge_index, edge_feature)
            node_feature = node_feature.relu()
        node_feature = self.node_convs[-1](node_feature, edge_index, edge_feature)

        local_feature = self.node_norm(self.node_linear(node_feature))
        global_feature = global_mean_pool(node_feature, batch)
        global_feature = self.graph_norm(self.graph_linear(global_feature))

        return local_feature, global_feature


class GraphEncoder_GMM(nn.Module):
    def __init__(self, node_dim, edge_dim, num_layers):
        super(GraphEncoder_GMM, self).__init__()
        self.num_layers = num_layers
        self.node_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.node_convs.append(
                GMMConv(
                    in_channels=node_dim,
                    out_channels=node_dim,
                    dim=128,
                    kernel_size=8
                )
            )
        self.graph_linear = nn.Linear(node_dim, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.graph_norm = nn.LayerNorm(node_dim)

    def forward(self, node_feature, edge_index, edge_feature, batch):
        for i in range(self.num_layers - 1):
            node_feature = self.node_convs[i](node_feature, edge_index, edge_feature)
            node_feature = node_feature.relu()
            node_feature = F.dropout(node_feature, p=0.5, training=self.training)
        node_feature = self.node_convs[-1](node_feature, edge_index, edge_feature)

        local_feature = self.node_norm(node_feature)
        global_feature = global_mean_pool(node_feature, batch)
        global_feature = self.graph_norm(self.graph_linear(global_feature))

        return local_feature, global_feature


class GraphEncoder_GINE(nn.Module):
    def __init__(self, node_dim, edge_dim, num_layers):
        super(GraphEncoder_GINE, self).__init__()
        self.num_layers = num_layers
        self.node_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.node_convs.append(
                GINEConv(
                    nn=tgmlp(
                        in_channels=node_dim,
                        hidden_channels=node_dim * 2,
                        out_channels=node_dim,
                        num_layers=2,
                        dropout=0.5,
                        act=nn.Mish()
                    ),
                    train_eps=True,
                    edge_dim=edge_dim
                )
            )
        self.graph_linear = nn.Linear(node_dim, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.graph_norm = nn.LayerNorm(node_dim)

    def forward(self, node_feature, edge_index, edge_feature, batch):
        for i in range(self.num_layers - 1):
            node_feature = self.node_convs[i](node_feature, edge_index, edge_feature)
            node_feature = node_feature.relu()
            node_feature = F.dropout(node_feature, p=0.5, training=self.training)
        node_feature = self.node_convs[-1](node_feature, edge_index, edge_feature)

        local_feature = self.node_norm(node_feature)
        global_feature = global_mean_pool(node_feature, batch)
        global_feature = self.graph_norm(self.graph_linear(global_feature))

        return local_feature, global_feature


from models.encoders.layers import MyResGatedGraphConv
from torch_geometric.nn import MLP


class GraphEncoder_MultiAggr(nn.Module):
    def __init__(self, node_dim, edge_dim, num_layers, aggr='mean', aggr_kwargs=None):
        super(GraphEncoder_MultiAggr, self).__init__()
        self.num_layers = num_layers
        self.node_convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_convs.append(
            MyResGatedGraphConv(
                in_channels=node_dim,
                out_channels=node_dim,
                edge_dim=edge_dim,
                aggr=copy.deepcopy(aggr),
                aggr_kwargs=aggr_kwargs
            )
        )
        self.norms.append(GraphNorm(node_dim * len(aggr)))
        for _ in range(num_layers - 1):
            self.node_convs.append(
                MyResGatedGraphConv(
                    in_channels=node_dim * len(aggr),
                    out_channels=node_dim,
                    edge_dim=edge_dim,
                    aggr=copy.deepcopy(aggr),
                    aggr_kwargs=aggr_kwargs
                )
            )
            self.norms.append(GraphNorm(node_dim * len(aggr)))
        self.node_linear = nn.Linear(node_dim * len(aggr), node_dim)
        self.graph_readout = GATv2Conv(
            in_channels=node_dim,
            out_channels=node_dim,
            heads=8,
            dropout=0.25,
            edge_dim=edge_dim
        )
        self.graph_linear = nn.Linear(node_dim * 8, node_dim)

    def forward(self, node_feature, edge_index, edge_feature, batch):
        for i in range(self.num_layers - 1):
            node_feature = self.node_convs[i](node_feature, edge_index, edge_feature)
            node_feature = F.mish(node_feature)
            node_feature = self.norms[i](node_feature)
            node_feature = F.dropout(node_feature, p=0.25, training=self.training)
        node_feature = self.node_convs[-1](node_feature, edge_index, edge_feature)
        local_feature = self.node_linear(node_feature)
        global_feature = self.graph_readout(local_feature, edge_index, edge_feature)
        global_feature = global_mean_pool(global_feature, batch)
        global_feature = self.graph_linear(global_feature)

        return local_feature, global_feature
