from typing import Optional, Tuple

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import MLP, LayerNorm
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
import torchmetrics.functional.classification as tfc

from models.loss import *

EPS = 1e-15
MAX_LOGSTD = 10


class InnerProductDecoder(torch.nn.Module):
    def forward(self, z: Tensor, edge_index: Tensor,
                sigmoid: bool = True) -> Tensor:
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GAE(torch.nn.Module):
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        from torchmetrics.functional.classification import binary_average_precision, binary_f1_score

        pos_y = z.new_ones(pos_edge_index.size(1), dtype=torch.long)
        neg_y = z.new_zeros(neg_edge_index.size(1), dtype=torch.long)
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=False)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=False)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        ap = binary_average_precision(pred, y).item()
        f1score = binary_f1_score(pred, y).item()

        return ap, f1score


class VGAE(GAE):
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        """"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))


class MultiClassInnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, softmax: bool = True):
        value = (z[edge_index[0]] * z[edge_index[1]])
        return torch.softmax(value, dim=1) if softmax else value

    def forward_all(self, z, softmax: bool = True):
        num_nodes = z.shape[0]
        num_relations = z.shape[1]
        adj = torch.zeros(num_nodes, num_nodes, num_relations).to(z.device)
        for i in range(num_nodes):
            for j in range(num_nodes):
                z_decode = z[i] * z[j]
                adj[i, j] = z_decode
        return torch.softmax(adj, dim=1) if softmax else adj


class MultiClassVGAE(nn.Module):
    def __init__(self, num_relations, encoder, decoder=None):
        super().__init__()
        self.num_relations = num_relations
        self.encoder = encoder
        self.decoder = MultiClassInnerProductDecoder() if decoder is None else decoder
        # self.loss = DiceLoss()
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs):  # pragma: no cover
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def recon_loss(self, z, pos_edge_index, pos_edge_type, neg_edge_index=None):
        loss = DiceLoss()

        pos_z_decode = self.decoder(z, pos_edge_index, softmax=False)
        pos_loss = F.cross_entropy(pos_z_decode, pos_edge_type)
        # pos_loss = self.loss(pos_z_decode, pos_edge_type)

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_edge_type = torch.zeros(neg_edge_index.shape[1], dtype=torch.long).to(z.device)
        neg_z_decode = self.decoder(z, neg_edge_index, softmax=False)
        neg_loss = F.cross_entropy(neg_z_decode, neg_edge_type)
        # neg_loss = self.loss(neg_z_decode, neg_edge_type)

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, pos_y, neg_edge_index):
        neg_y = torch.zeros(neg_edge_index.size(1), dtype=torch.long).to(z.device)
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, softmax=False)
        neg_pred = self.decoder(z, neg_edge_index, softmax=False)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        ap = tfc.multiclass_average_precision(pred, y, num_classes=self.num_relations).item()
        f1score = tfc.multiclass_f1_score(pred, y, num_classes=self.num_relations).item()
        # precision = tfc.multiclass_precision(pred, y, num_classes=self.num_relations).item()
        # recall = tfc.multiclass_recall(pred, y, num_classes=self.num_relations).item()

        return ap, f1score


class MultiClassVGAEV2(nn.Module):
    def __init__(self, num_relations, emb_dim, encoder, decoder=None):
        super().__init__()
        self.num_relations = num_relations
        self.encoder = encoder
        self.decoder = MultiClassInnerProductDecoder() if decoder is None else decoder
        self.decode_classify = MLP(
            in_channels=emb_dim,
            hidden_channels=emb_dim * 2,
            out_channels=num_relations,
            num_layers=2,
            dropout=0.25,
            act=nn.Mish(),
            norm=LayerNorm(emb_dim * 2)
        )
        # self.loss = MultiFocalLoss(class_num=num_relations)
        self.loss = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs):  # pragma: no cover
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def recon_loss(self, z, pos_edge_index, pos_edge_type, neg_edge_index=None):
        pos_z_decode = self.decoder(z, pos_edge_index, softmax=False)
        pos_out = self.decode_classify(pos_z_decode)
        # pos_loss = F.cross_entropy(pos_out, pos_edge_type)
        pos_loss = self.loss(pos_out, pos_edge_type)

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_edge_type = torch.zeros(neg_edge_index.shape[1], dtype=torch.long).to(z.device)
        neg_z_decode = self.decoder(z, neg_edge_index, softmax=False)
        neg_out = self.decode_classify(neg_z_decode)
        # neg_loss = F.cross_entropy(neg_out, neg_edge_type)
        neg_loss = self.loss(neg_out, neg_edge_type)

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, pos_y, neg_edge_index):
        neg_y = torch.zeros(neg_edge_index.size(1), dtype=torch.long).to(z.device)
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_decode = self.decoder(z, pos_edge_index, softmax=False)
        pos_out = self.decode_classify(pos_decode)
        neg_decode = self.decoder(z, neg_edge_index, softmax=False)
        neg_out = self.decode_classify(neg_decode)
        pred = torch.cat([pos_out, neg_out], dim=0)

        ap = tfc.multiclass_average_precision(pred, y, num_classes=self.num_relations).item()
        f1score = tfc.multiclass_f1_score(pred, y, num_classes=self.num_relations).item()

        return ap, f1score
