from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter, Sigmoid

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor


class MyResGatedGraphConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            act: Optional[Callable] = Sigmoid(),
            edge_dim: Optional[int] = None,
            root_weight: bool = True,
            bias: bool = True,
            **kwargs,
    ):

        # kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.edge_dim = edge_dim
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        edge_dim = edge_dim if edge_dim is not None else 0
        self.lin_key = Linear(in_channels[1] + edge_dim, out_channels)
        self.lin_query = Linear(in_channels[0] + edge_dim, out_channels)
        self.lin_value = Linear(in_channels[0] + edge_dim, out_channels)

        if root_weight:
            self.lin_skip = Linear(in_channels[1], out_channels * len(self.aggr), bias=False)
        else:
            self.register_parameter('lin_skip', None)

        if bias:
            self.bias = Parameter(Tensor(out_channels * len(self.aggr)))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # In case edge features are not given, we can compute key, query and
        # value tensors in node-level space, which is a bit more efficient:
        if self.edge_dim is None:
            k = self.lin_key(x[1])
            q = self.lin_query(x[0])
            v = self.lin_value(x[0])
        else:
            k, q, v = x[1], x[0], x[0]

        # propagate_type: (k: Tensor, q: Tensor, v: Tensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr,
                             size=None)

        if self.root_weight:
            out = out + self.lin_skip(x[1])

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, k_i: Tensor, q_j: Tensor, v_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        assert (edge_attr is not None) == (self.edge_dim is not None)

        if edge_attr is not None:
            k_i = self.lin_key(torch.cat([k_i, edge_attr], dim=-1))
            q_j = self.lin_query(torch.cat([q_j, edge_attr], dim=-1))
            v_j = self.lin_value(torch.cat([v_j, edge_attr], dim=-1))

        return self.act(k_i + q_j) * v_j
