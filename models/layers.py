import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, LayerNorm
from models.decoders import *
from models.gae import *


class ClassifyLoss(nn.Module):
    def __init__(self, num_classes, gamma=2):
        super().__init__()
        self.loss = MultiFocalLoss(class_num=num_classes, gamma=gamma)

    def forward(self, y_cls_pred, y_cls):
        cls_loss = self.loss(y_cls_pred, y_cls)
        # cls_loss = F.cross_entropy(y_cls_pred, y_cls)
        return cls_loss


class SegmentLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.loss = BinaryFocalLoss(gamma=gamma)

    def forward(self, y_seg_pred, y_seg, batch):
        batch_size = torch.max(batch) + 1
        y_seg = y_seg.float()
        seg_loss = 0
        for i in range(batch_size):
            batch_index = torch.nonzero(batch == i).squeeze()
            matrix_index = torch.meshgrid(batch_index, batch_index)
            # seg_loss += F.binary_cross_entropy_with_logits(
            #     y_seg_pred[matrix_index], y_seg[matrix_index],
            #     pos_weight=self.weight
            # )
            seg_loss += self.loss(y_seg_pred[matrix_index].view(-1, 1), y_seg[matrix_index].view(-1))

        return seg_loss / batch_size


class RelationPredLoss(nn.Module):
    def __init__(self, num_relations, gamma=2):
        super().__init__()
        self.num_relations = num_relations
        self.loss = MultiFocalLoss(class_num=num_relations, gamma=gamma)

    def forward(self, y_rel_pred, y_rel, batch):
        batch_size = torch.max(batch) + 1
        rel_loss = 0
        for i in range(batch_size):
            batch_index = torch.nonzero(batch == i).squeeze()
            matrix_index = torch.meshgrid(batch_index, batch_index)
            # rel_loss += F.cross_entropy(
            #     y_rel_pred[matrix_index].view(-1, self.num_relations),
            #     y_rel[matrix_index].view(-1),
            #     weight=self.weight
            # )
            rel_loss += self.loss(y_rel_pred[matrix_index].view(-1, self.num_relations), y_rel[matrix_index].view(-1))

        return rel_loss / batch_size


class ClassifyHead(nn.Module):
    def __init__(self, final_out_emb, classify_hidden_dim, num_classes):
        super().__init__()
        self.classify = MLP(
            in_channels=final_out_emb,
            hidden_channels=classify_hidden_dim,
            out_channels=num_classes,
            num_layers=3,
            dropout=0.25,
            act=nn.Mish(),
            norm=LayerNorm(classify_hidden_dim)
        )

    def forward(self, local_global_feat):
        classify_out, classify_emb = self.classify(local_global_feat, return_emb=True)
        return classify_out, classify_emb


class SegmentHead(nn.Module):
    def __init__(self, classify_hidden_dim, final_out_emb, segment_hidden_dim, segment_output_dim):
        super().__init__()
        self.classify_seg_emb = MLP(
            in_channels=classify_hidden_dim,
            hidden_channels=classify_hidden_dim * 2,
            out_channels=classify_hidden_dim,
            num_layers=3,
            dropout=0.5,
            act=nn.Mish(),
            norm=LayerNorm(classify_hidden_dim * 2)
        )
        self.segment_emb = MLP(
            in_channels=final_out_emb,
            hidden_channels=final_out_emb * 2,
            out_channels=final_out_emb,
            num_layers=3,
            dropout=0.5,
            act=nn.Mish(),
            norm=LayerNorm(final_out_emb * 2)
        )
        self.segment_encoder = MLP(
            in_channels=final_out_emb + classify_hidden_dim,
            hidden_channels=segment_hidden_dim,
            out_channels=segment_output_dim,
            num_layers=3,
            dropout=0.5,
            norm=LayerNorm(segment_hidden_dim),
            act=nn.Mish()
        )
        self.segment_decoder = SegmentDecoder()

    def forward(self, local_global_feat, classify_emb, cls_pred_label, batch):
        seg_emb = self.segment_emb(local_global_feat)
        cls_seg_emb = self.classify_seg_emb(classify_emb)

        seg_feat = torch.concat([seg_emb, cls_seg_emb], dim=1)
        seg_out, seg_emb = self.segment_encoder(seg_feat, return_emb=True)  # num_nodes * seg_out_dim
        seg_out = self.segment_decoder(seg_out, cls_pred_label, batch)
        return seg_out, seg_emb


class Segment_VGAE(nn.Module):
    def __init__(self, classify_hidden_dim, final_out_emb, segment_emb_dim):
        super().__init__()
        self.classify_seg_emb = MLP(
            in_channels=classify_hidden_dim,
            hidden_channels=classify_hidden_dim * 2,
            out_channels=classify_hidden_dim,
            num_layers=2,
            dropout=0.25,
            act=nn.Mish(),
            norm=LayerNorm(classify_hidden_dim * 2)
        )
        input_dim = classify_hidden_dim + final_out_emb
        self.mu_encoder = MLP(
            in_channels=input_dim,
            hidden_channels=input_dim * 2,
            out_channels=segment_emb_dim,
            num_layers=2,
            dropout=0.25,
            act=nn.Mish(),
            norm=LayerNorm(input_dim * 2)
        )
        self.logstd_encoder = MLP(
            in_channels=input_dim,
            hidden_channels=input_dim * 2,
            out_channels=segment_emb_dim,
            num_layers=2,
            dropout=0.25,
            act=nn.Mish(),
            norm=LayerNorm(input_dim * 2)
        )

    def forward(self, local_global_feat, classify_emb):
        cls_seg_emb = self.classify_seg_emb(classify_emb)
        seg_feat = torch.concat([local_global_feat, cls_seg_emb], dim=1)
        mu = self.mu_encoder(seg_feat)
        logstd = self.logstd_encoder(seg_feat)

        return mu, logstd


class SegmentHeadV2(nn.Module):
    def __init__(self, classify_hidden_dim, final_out_emb, segment_hidden_dim, segment_output_dim):
        super().__init__()
        self.classify_seg_emb = MLP(
            in_channels=classify_hidden_dim,
            hidden_channels=classify_hidden_dim * 2,
            out_channels=final_out_emb,
            num_layers=3,
            dropout=0.5,
            act=nn.Mish(),
            norm=LayerNorm(classify_hidden_dim * 2)
        )
        self.segment_emb = MLP(
            in_channels=final_out_emb,
            hidden_channels=final_out_emb * 2,
            out_channels=final_out_emb,
            num_layers=3,
            dropout=0.5,
            act=nn.Mish(),
            norm=LayerNorm(final_out_emb * 2)
        )
        self.segment_encoder1 = MLP(
            in_channels=final_out_emb * 2,
            hidden_channels=segment_hidden_dim,
            out_channels=segment_output_dim,
            num_layers=2,
            dropout=0.5,
            norm=LayerNorm(segment_hidden_dim),
            act=nn.Mish()
        )
        self.segment_encoder2 = MLP(
            in_channels=final_out_emb * 2,
            hidden_channels=segment_hidden_dim,
            out_channels=segment_output_dim,
            num_layers=2,
            dropout=0.5,
            norm=LayerNorm(segment_hidden_dim),
            act=nn.Mish()
        )
        self.segment_decoder = SegmentDecoderV2()

    def forward(self, local_global_feat, classify_emb, cls_pred_label, batch):
        seg_emb = self.segment_emb(local_global_feat)
        cls_seg_emb = self.classify_seg_emb(classify_emb)

        seg_feat = torch.concat([seg_emb, cls_seg_emb], dim=1)
        seg_out1 = self.segment_encoder1(seg_feat)  # num_nodes * seg_out_dim
        seg_out2 = self.segment_encoder2(seg_feat)  # num_nodes * seg_out_dim
        seg_out = self.segment_decoder(seg_out1, seg_out2, cls_pred_label, batch)
        return seg_out, seg_emb


class RelationPredHeadV2(nn.Module):
    def __init__(self, classify_hidden_dim, segment_hidden_dim, final_out_emb, predict_hidden_dim, num_relations):
        super().__init__()
        self.classify_pred_emb = MLP(
            in_channels=classify_hidden_dim,
            hidden_channels=classify_hidden_dim * 2,
            out_channels=final_out_emb,
            num_layers=3,
            dropout=0.5,
            act=nn.Mish(),
            norm=LayerNorm(classify_hidden_dim * 2)
        )
        self.seg_pred_emb = MLP(
            in_channels=final_out_emb,
            hidden_channels=final_out_emb * 2,
            out_channels=final_out_emb,
            num_layers=3,
            dropout=0.5,
            act=nn.Mish(),
            norm=LayerNorm(final_out_emb * 2)
        )
        self.prediction_emb = MLP(
            in_channels=final_out_emb,
            hidden_channels=final_out_emb * 2,
            out_channels=final_out_emb,
            num_layers=3,
            dropout=0.5,
            act=nn.Mish(),
            norm=LayerNorm(final_out_emb * 2)
        )
        self.rel_pred_encoder1 = MLP(
            in_channels=final_out_emb * 3,
            hidden_channels=predict_hidden_dim,
            out_channels=num_relations,
            num_layers=2,
            act=nn.Mish(),
            norm=LayerNorm(predict_hidden_dim)
        )
        self.rel_pred_encoder2 = MLP(
            in_channels=final_out_emb * 3,
            hidden_channels=predict_hidden_dim,
            out_channels=num_relations,
            num_layers=2,
            act=nn.Mish(),
            norm=LayerNorm(predict_hidden_dim)
        )
        self.rel_pred_decoder = FeatureRelationDecoderV2(num_relations=num_relations)

    def forward(self, local_global_feat, classify_emb, seg_emb, seg_matrix, cls_pred_label, batch):
        pred_emb = self.prediction_emb(local_global_feat)
        cls_pred_emb = self.classify_pred_emb(classify_emb)
        seg_pred_emb = self.seg_pred_emb(seg_emb)

        predict_feat = torch.concat([pred_emb, cls_pred_emb, seg_pred_emb], dim=1)
        predict_emb1 = self.rel_pred_encoder1(predict_feat)  # num_nodes * num_relations
        predict_emb2 = self.rel_pred_encoder2(predict_feat)  # num_nodes * num_relations
        predict_out = self.rel_pred_decoder(predict_emb1, predict_emb2, seg_matrix, cls_pred_label, batch)
        return predict_out


class RelationPredHead(nn.Module):
    def __init__(self, classify_hidden_dim, segment_hidden_dim, final_out_emb, predict_hidden_dim, num_relations):
        super().__init__()
        self.classify_pred_emb = MLP(
            in_channels=classify_hidden_dim,
            hidden_channels=classify_hidden_dim * 2,
            out_channels=classify_hidden_dim,
            num_layers=3,
            dropout=0.5,
            act=nn.Mish(),
            norm=LayerNorm(classify_hidden_dim * 2)
        )
        self.seg_pred_emb = MLP(
            in_channels=segment_hidden_dim,
            hidden_channels=segment_hidden_dim * 2,
            out_channels=segment_hidden_dim,
            num_layers=3,
            dropout=0.5,
            act=nn.Mish(),
            norm=LayerNorm(segment_hidden_dim * 2)
        )
        self.prediction_emb = MLP(
            in_channels=final_out_emb,
            hidden_channels=final_out_emb * 2,
            out_channels=final_out_emb,
            num_layers=3,
            dropout=0.5,
            act=nn.Mish(),
            norm=LayerNorm(final_out_emb * 2)
        )
        self.rel_pred_encoder = MLP(
            in_channels=final_out_emb + classify_hidden_dim + segment_hidden_dim,
            hidden_channels=predict_hidden_dim,
            out_channels=num_relations,
            num_layers=3,
            act=nn.Mish(),
            norm=LayerNorm(predict_hidden_dim)
        )
        self.rel_pred_decoder = FeatureRelationDecoder(num_relations=num_relations)

    def forward(self, local_global_feat, classify_emb, seg_emb, seg_matrix, cls_pred_label, batch):
        pred_emb = self.prediction_emb(local_global_feat)
        cls_pred_emb = self.classify_pred_emb(classify_emb)
        seg_pred_emb = self.seg_pred_emb(seg_emb)

        predict_feat = torch.concat([pred_emb, cls_pred_emb, seg_pred_emb], dim=1)
        predict_emb = self.rel_pred_encoder(predict_feat)  # num_nodes * num_relations
        predict_out = self.rel_pred_decoder(predict_emb, seg_matrix, cls_pred_label, batch)
        return predict_out


class RelPred_VGAE(nn.Module):
    def __init__(self, classify_hidden_dim, final_out_emb, segment_emb_dim, rel_emb_dim):
        super().__init__()
        self.classify_pred_emb = MLP(
            in_channels=classify_hidden_dim,
            hidden_channels=classify_hidden_dim * 2,
            out_channels=classify_hidden_dim,
            num_layers=2,
            dropout=0.25,
            act=nn.Mish(),
            norm=LayerNorm(classify_hidden_dim * 2)
        )
        self.seg_pred_emb = MLP(
            in_channels=segment_emb_dim,
            hidden_channels=segment_emb_dim * 2,
            out_channels=segment_emb_dim,
            num_layers=2,
            dropout=0.25,
            act=nn.Mish(),
            norm=LayerNorm(segment_emb_dim * 2)
        )
        input_dim = classify_hidden_dim + segment_emb_dim + final_out_emb
        self.mu_encoder = MLP(
            in_channels=input_dim,
            hidden_channels=input_dim * 2,
            out_channels=rel_emb_dim,
            num_layers=2,
            dropout=0.25,
            act=nn.Mish(),
            norm=LayerNorm(input_dim * 2)
        )
        self.logstd_encoder = MLP(
            in_channels=input_dim,
            hidden_channels=input_dim * 2,
            out_channels=rel_emb_dim,
            num_layers=2,
            dropout=0.25,
            act=nn.Mish(),
            norm=LayerNorm(input_dim * 2)
        )

    def forward(self, local_global_feat, classify_emb, seg_emb):
        cls_pred_emb = self.classify_pred_emb(classify_emb)
        seg_pred_emb = self.seg_pred_emb(seg_emb)
        pred_feat = torch.concat([local_global_feat, cls_pred_emb, seg_pred_emb], dim=1)
        mu = self.mu_encoder(pred_feat)
        logstd = self.logstd_encoder(pred_feat)

        return mu, logstd
