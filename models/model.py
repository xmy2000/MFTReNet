import torch
import lightning as L
import torchmetrics
from torch import nn
from torch_geometric import nn as tgnn
from torch_geometric.nn import MLP, LayerNorm
from torch_geometric.utils import negative_sampling
from lightning.pytorch.utilities.model_summary import ModelSummary

from models.encoders.uvgrid_encoder import UVNetSurfaceEncoder, UVNetCurveEncoder
from models.encoders.graph_encoder import GraphEncoder_MultiAggr
from models.layers import ClassifyHead, ClassifyLoss, Segment_VGAE, VGAE, RelPred_VGAE, MultiClassVGAEV2, geometric_loss


class GraphEmb(nn.Module):
    def __init__(
            self,
            node_attr_dim,
            node_attr_emb,
            node_grid_dim,
            node_grid_emb,
            edge_attr_dim,
            edge_attr_emb,
            edge_grid_dim,
            edge_grid_emb,
            graph_encoder_layers
    ):
        super(GraphEmb, self).__init__()

        # 面属性特征编码器
        self.node_attr_encoder = MLP(
            in_channels=node_attr_dim,
            hidden_channels=node_attr_emb * 2,
            out_channels=node_attr_emb,
            num_layers=2,
            dropout=0.25,
            act=nn.Mish(),
            norm=LayerNorm(node_attr_emb * 2)
        )
        # 面网格编码器
        self.node_grid_encoder = UVNetSurfaceEncoder(node_grid_dim, node_grid_emb)
        # 边属性编码器
        self.edge_attr_encoder = MLP(
            in_channels=edge_attr_dim,
            hidden_channels=edge_attr_emb * 2,
            out_channels=edge_attr_emb,
            num_layers=2,
            dropout=0.25,
            act=nn.Mish(),
            norm=LayerNorm(edge_attr_emb * 2)
        )
        # 边网格编码器
        self.edge_grid_encoder = UVNetCurveEncoder(edge_grid_dim, edge_grid_emb)

        self.node_emb = node_attr_emb + node_grid_emb  # 节点特征维度
        self.edge_emb = edge_attr_emb + edge_grid_emb  # 边特征维度
        self.final_out_emb = 2 * self.node_emb

        # 图编码器
        aggr = [
            tgnn.MeanAggregation(),
            tgnn.SumAggregation(),
            tgnn.MaxAggregation(),
            tgnn.SoftmaxAggregation(learn=True)
        ]
        self.graph_encoder = GraphEncoder_MultiAggr(self.node_emb, self.edge_emb, num_layers=graph_encoder_layers,
                                                    aggr=aggr)

    def forward(self, batch_graph):
        face_grid = batch_graph['face_grid']  # num_nodes * 7 * num_u * num_v
        face_attr = batch_graph['face_attr']  # num_nodes * 14
        edge_grid = batch_graph['edge_grid']  # num_edges * 12 * num_u
        edge_attr = batch_graph['edge_attr']  # num_edges * 15
        edge_index = batch_graph['edge_index']  # 2 * num_edges
        batch = batch_graph['batch']  # num_nodes

        face_attr_feat = self.node_attr_encoder(face_attr)  # num_nodes * 64
        face_grid_feat = self.node_grid_encoder(face_grid)  # num_nodes * 64
        node_feat = torch.concat([face_attr_feat, face_grid_feat], dim=1)  # num_nodes * 128

        edge_attr_feat = self.edge_attr_encoder(edge_attr)  # num_edges * 64
        edge_grid_feat = self.edge_grid_encoder(edge_grid)  # num_edges * 64
        edge_feat = torch.concat([edge_attr_feat, edge_grid_feat], dim=1)  # num_edges * 128

        node_emb, graph_emb = self.graph_encoder(node_feat, edge_index, edge_feat,
                                                 batch)  # num_edges * 128, batch_size * 128

        # concatenated to the per-node embeddings
        local_global_feat = torch.cat([node_emb, graph_emb[batch]], dim=1)

        return local_global_feat


class Expert(nn.Module):
    def __init__(self, input_feature_size, expert_hidden_size, expert_output_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_feature_size, expert_hidden_size)
        self.fc2 = nn.Linear(expert_hidden_size, expert_output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class CGC(nn.Module):
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden):
        super(CGC, self).__init__()

        self.input_size = input_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden

        self.experts_shared = nn.ModuleList(
            [Expert(self.input_size, self.experts_hidden, self.experts_out) for i in range(self.num_shared_experts)])
        self.experts_task1 = nn.ModuleList(
            [Expert(self.input_size, self.experts_hidden, self.experts_out) for i in range(self.num_specific_experts)])
        self.experts_task2 = nn.ModuleList(
            [Expert(self.input_size, self.experts_hidden, self.experts_out) for i in range(self.num_specific_experts)])
        self.experts_task3 = nn.ModuleList(
            [Expert(self.input_size, self.experts_hidden, self.experts_out) for i in range(self.num_specific_experts)])

        self.soft = nn.Softmax(dim=1)

        self.gate_task1 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                        nn.Softmax(dim=1))
        self.gate_task2 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                        nn.Softmax(dim=1))
        self.gate_task3 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                        nn.Softmax(dim=1))

    def forward(self, x):
        experts_shared_o = [e(x) for e in self.experts_shared]
        experts_shared_o = torch.stack(experts_shared_o)

        experts_task1_o = [e(x) for e in self.experts_task1]
        experts_task1_o = torch.stack(experts_task1_o)

        experts_task2_o = [e(x) for e in self.experts_task2]
        experts_task2_o = torch.stack(experts_task2_o)

        experts_task3_o = [e(x) for e in self.experts_task3]
        experts_task3_o = torch.stack(experts_task3_o)

        # gate1
        selected_task1 = self.gate_task1(x)
        gate_expert_output1 = torch.cat((experts_task1_o, experts_shared_o), dim=0)
        gate_task1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected_task1)

        # gate2
        selected_task2 = self.gate_task2(x)
        gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
        gate_task2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected_task2)

        # gate3
        selected_task3 = self.gate_task3(x)
        gate_expert_output3 = torch.cat((experts_task3_o, experts_shared_o), dim=0)
        gate_task3_out = torch.einsum('abc, ba -> bc', gate_expert_output3, selected_task3)

        return [gate_task1_out, gate_task2_out, gate_task3_out]


class FRModel(L.LightningModule):
    def __init__(
            self,
            graph_emb,
            mtl_cgc,
            classify_hidden_dim,
            segment_hidden_dim,
            rel_hidden_dim,
            num_classes,
            num_relations
    ):
        super(FRModel, self).__init__()
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.graph_emb = graph_emb
        self.mtl_cgc = mtl_cgc
        num_experts = self.mtl_cgc.num_specific_experts + self.mtl_cgc.num_shared_experts
        final_out_emb = self.mtl_cgc.experts_out

        # 特征分类器
        self.classify_head = ClassifyHead(
            final_out_emb=final_out_emb,
            classify_hidden_dim=classify_hidden_dim,
            num_classes=num_classes
        )
        self.classify_loss = ClassifyLoss(num_classes=num_classes)

        # 分割器
        self.segment_encoder = Segment_VGAE(
            classify_hidden_dim=classify_hidden_dim,
            final_out_emb=final_out_emb,
            segment_emb_dim=segment_hidden_dim
        )
        self.segment_head = VGAE(encoder=self.segment_encoder)

        # 链接预测
        self.rel_pred_encoder = RelPred_VGAE(
            classify_hidden_dim=classify_hidden_dim,
            final_out_emb=final_out_emb,
            segment_emb_dim=segment_hidden_dim,
            rel_emb_dim=rel_hidden_dim
        )
        self.rel_pred_head = MultiClassVGAEV2(num_relations=num_relations, emb_dim=rel_hidden_dim,
                                              encoder=self.rel_pred_encoder)

        # 分类指标
        self.classify_train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        self.classify_train_miou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=num_classes)
        self.classify_val_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        self.classify_val_miou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=num_classes)
        self.classify_test_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        self.classify_test_miou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=num_classes)

    def forward(self, batch_graph):
        batch = batch_graph['batch']  # num_nodes

        # graph emb
        local_global_feat = self.graph_emb(batch_graph).to(self.device)  # num_nodes * 256
        gate_cls_feat, gate_seg_feat, gate_rel_feat = self.mtl_cgc(local_global_feat)

        # classify: num_nodes * 26, num_nodes * classify_hidden_dim
        classify_out, classify_emb = self.classify_head(gate_cls_feat)
        # cls_pred_label = torch.argmax(classify_out, dim=1).long()

        # segment
        seg_emb = self.segment_head.encode(gate_seg_feat, classify_emb)

        # relation prediction  output: num_nodes * num_nodes * num_relations
        predict_out = self.rel_pred_head.encode(gate_rel_feat, classify_emb, seg_emb)
        # predict_out = None

        return classify_out, seg_emb, predict_out

    def step(self, batch_graph):
        y_cls = batch_graph['cls_label'].long()
        y_seg = batch_graph['seg_label'].long()
        y_seg_num = batch_graph['seg_label_num'].long()
        batch = batch_graph['batch'].long()
        y_rel = batch_graph['rel_edge'].long()
        y_rel_type = batch_graph['rel_type'].long()
        y_rel_num = batch_graph['rel_num'].long()
        y_cls_pred, y_seg_emb, y_rel_emb = self.forward(batch_graph)

        # classify process
        classify_return = (y_cls_pred, y_cls)

        # segment process
        batch_size = torch.max(batch) + 1
        label_index = 0
        seg_loss = 0
        f1score_lst, ap_lst = [], []
        for b in range(batch_size):
            node_index = torch.nonzero(batch == b).squeeze()
            num_nodes = node_index.shape[0]
            label_num = y_seg_num[b]
            if label_num == 0:
                continue

            z_batch = y_seg_emb[node_index]
            label_batch = y_seg[label_index: label_index + label_num]
            label_batch = label_batch.t().contiguous()
            neg_edge = negative_sampling(label_batch, z_batch.size(0))

            seg_loss += self.segment_head.recon_loss(z_batch, pos_edge_index=label_batch, neg_edge_index=neg_edge) + (
                    1 / num_nodes) * self.segment_head.kl_loss()

            ap, f1score = self.segment_head.test(
                z_batch,
                pos_edge_index=label_batch,
                neg_edge_index=neg_edge
            )
            ap_lst.append(ap)
            f1score_lst.append(f1score)

            label_index += label_num

        seg_loss = seg_loss / batch_size
        seg_f1 = torch.mean(torch.tensor(f1score_lst))
        seg_ap = torch.mean(torch.tensor(ap_lst))
        seg_return = (seg_loss, seg_ap, seg_f1)

        # rel-prediction process
        label_index = 0
        rel_loss = 0
        ap_lst, f1score_lst = [], []
        for b in range(batch_size):
            node_index = torch.nonzero(batch == b).squeeze()
            num_nodes = node_index.shape[0]
            label_num = y_rel_num[b]
            if label_num == 0:
                continue

            z_batch = y_rel_emb[node_index]
            edge_batch = y_rel[label_index: label_index + label_num]
            edge_batch = edge_batch.t().contiguous()
            edge_type_batch = y_rel_type[label_index: label_index + label_num]
            neg_edge = negative_sampling(edge_batch, z_batch.size(0))

            rel_loss += self.rel_pred_head.recon_loss(
                z_batch, pos_edge_index=edge_batch, pos_edge_type=edge_type_batch, neg_edge_index=neg_edge
            ) + (1 / num_nodes) * self.rel_pred_head.kl_loss()

            ap, f1score = self.rel_pred_head.test(
                z_batch,
                pos_edge_index=edge_batch,
                pos_y=edge_type_batch,
                neg_edge_index=neg_edge
            )
            ap_lst.append(ap)
            f1score_lst.append(f1score)

            label_index += label_num

        rel_loss = rel_loss / batch_size
        rel_ap = torch.mean(torch.tensor(ap_lst))
        rel_f1score = torch.mean(torch.tensor(f1score_lst))

        rel_return = (rel_loss, rel_ap, rel_f1score)
        # rel_return = None

        return classify_return, seg_return, rel_return

    def training_step(self, batch_graph, batch_idx):
        batch = batch_graph['batch'].long()
        (y_cls_pred, y_cls), (seg_loss, seg_ap, seg_f1), (rel_loss, rel_ap, rel_f1score) = self.step(batch_graph)

        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True)

        # 分类
        cls_loss = self.classify_loss(y_cls_pred, y_cls)
        self.classify_train_accuracy(y_cls_pred, y_cls)
        self.classify_train_miou(y_cls_pred, y_cls)
        self.log("cls_train_loss", cls_loss, prog_bar=True)
        self.log("cls_train_accuracy", self.classify_train_accuracy, on_step=True, on_epoch=False, prog_bar=True)
        self.log("cls_train_mIOU", self.classify_train_miou, on_step=True, on_epoch=False, prog_bar=True)

        # 分割
        self.log("seg_train_loss", seg_loss, prog_bar=True)
        self.log("seg_train_f1", seg_f1, on_step=True, on_epoch=False, prog_bar=True)
        self.log("seg_train_ap", seg_ap, on_step=True, on_epoch=False, prog_bar=True)

        # 链接预测
        self.log("rel_train_loss", rel_loss, prog_bar=True)
        self.log("rel_train_ap", rel_ap, on_step=True, on_epoch=False, prog_bar=True)
        self.log("rel_train_f1", rel_f1score, on_step=True, on_epoch=False, prog_bar=True)

        # loss = cls_loss + seg_loss + rel_loss
        # loss = cls_loss + seg_loss
        loss = geometric_loss([cls_loss, seg_loss, rel_loss])
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch_graph, batch_idx):
        batch = batch_graph['batch'].long()
        (y_cls_pred, y_cls), (seg_loss, seg_ap, seg_f1), (rel_loss, rel_ap, rel_f1score) = self.step(batch_graph)

        # 分类
        cls_loss = self.classify_loss(y_cls_pred, y_cls)
        self.classify_val_accuracy(y_cls_pred, y_cls)
        self.classify_val_miou(y_cls_pred, y_cls)
        self.log("cls_val_loss", cls_loss, prog_bar=True)
        self.log("cls_val_accuracy", self.classify_val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cls_val_mIOU", self.classify_val_miou, on_step=False, on_epoch=True, prog_bar=True)

        # 分割
        self.log("seg_val_loss", seg_loss, prog_bar=True)
        self.log("seg_val_f1", seg_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("seg_val_ap", seg_ap, on_step=False, on_epoch=True, prog_bar=True)

        # 链接预测
        self.log("rel_val_loss", rel_loss, prog_bar=True)
        self.log("rel_val_ap", rel_ap, on_step=False, on_epoch=True, prog_bar=True)
        self.log("rel_val_f1", rel_f1score, on_step=False, on_epoch=True, prog_bar=True)

        # val_loss = cls_loss + seg_loss + rel_loss
        # val_loss = cls_loss + seg_loss
        val_loss = geometric_loss([cls_loss, seg_loss, rel_loss])
        self.log("val_loss", val_loss, prog_bar=True)

    def test_step(self, batch_graph, batch_idx):
        batch = batch_graph['batch'].long()
        (y_cls_pred, y_cls), (seg_loss, seg_ap, seg_f1), (rel_loss, rel_ap, rel_f1score) = self.step(batch_graph)

        # 分类
        cls_loss = self.classify_loss(y_cls_pred, y_cls)
        self.classify_test_accuracy(y_cls_pred, y_cls)
        self.classify_test_miou(y_cls_pred, y_cls)
        self.log("cls_test_loss", cls_loss, prog_bar=True)
        self.log("cls_test_accuracy", self.classify_test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cls_test_mIOU", self.classify_test_miou, on_step=False, on_epoch=True, prog_bar=True)

        # 分割
        self.log("seg_test_loss", seg_loss, prog_bar=True)
        self.log("seg_test_f1", seg_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("seg_test_ap", seg_ap, on_step=False, on_epoch=True, prog_bar=True)

        # 链接预测
        self.log("rel_test_loss", rel_loss, prog_bar=True)
        self.log("rel_test_ap", rel_ap, on_step=False, on_epoch=True, prog_bar=True)
        self.log("rel_test_f1", rel_f1score, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-2)
        return optimizer
