import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics.functional as tmf
from tqdm import tqdm
from models.model import GraphEmb, FRModel

import warnings

warnings.filterwarnings("ignore")

device = 'cuda'


def decode_seg_emb(seg_emb, graph, model):
    num_nodes = graph['num_nodes']
    label_num = graph['seg_label_num']
    if label_num == 0:
        return -1, -1
    label = graph['seg_label']
    label_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.long) + torch.eye(num_nodes, dtype=torch.long)
    label_matrix = label_matrix.to(seg_emb.device)
    for i, j in label:
        label_matrix[i, j] = label_matrix[j, i] = 1
    seg_pred = model.segment_head.decoder.forward_all(seg_emb)  # num_nodes * num_nodes
    # seg_acc = tmf.classification.binary_accuracy(seg_pred, label_matrix).item()
    seg_f1 = tmf.classification.binary_f1_score(seg_pred, label_matrix).item()
    seg_ap = tmf.classification.binary_average_precision(seg_pred, label_matrix).item()
    return seg_f1, seg_ap


def decode_rel_emb(rel_emb, graph, model):
    num_nodes = graph['num_nodes']
    label_num = graph['rel_num']
    if label_num == 0:
        return -1, -1
    rel_edge = graph['rel_edge']
    rel_type = graph['rel_type']
    label_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.long).to(rel_emb.device)
    for i in range(rel_edge.shape[0]):
        row, col = rel_edge[i]
        label_matrix[row, col] = rel_type[i]
    rel_emb_decode = model.rel_pred_head.decoder.forward_all(rel_emb,
                                                             softmax=False)  # num_nodes * num_nodes * num_relations
    rel_pred = model.rel_pred_head.decode_classify(rel_emb_decode)
    num_relations = rel_pred.shape[2]
    y = label_matrix.view(-1)
    pred = rel_pred.view(-1, num_relations)
    # rel_acc = tmf.classification.multiclass_accuracy(pred, y, num_classes=num_relations).item()
    rel_f1 = tmf.classification.multiclass_f1_score(pred, y, num_classes=num_relations).item()
    rel_ap = tmf.classification.multiclass_average_precision(pred, y, num_classes=num_relations).item()
    return rel_f1, rel_ap


def decode_seg_emb_v2(seg_emb, graph, model):
    label_num = graph['seg_label_num']
    if label_num == 0:
        return -1, -1
    pos_edge_index = graph['seg_label']
    pos_edge_index = pos_edge_index.t().contiguous()
    seg_pred = model.segment_head.decoder.forward(seg_emb, pos_edge_index, sigmoid=False)
    label = torch.ones(pos_edge_index.shape[1], dtype=torch.long).to(seg_pred.device)

    seg_f1 = tmf.classification.binary_f1_score(seg_pred, label).item()
    seg_ap = tmf.classification.binary_average_precision(seg_pred, label).item()
    return seg_f1, seg_ap


def decode_rel_emb_v2(rel_emb, graph, model):
    label_num = graph['rel_num']
    if label_num == 0:
        return -1, -1
    rel_edge_index = graph['rel_edge'].t().contiguous()
    rel_type = graph['rel_type']

    rel_emb_decode = model.rel_pred_head.decoder.forward(rel_emb, rel_edge_index, softmax=False)
    rel_pred = model.rel_pred_head.decode_classify(rel_emb_decode)
    num_relations = rel_pred.shape[1]

    rel_f1 = tmf.classification.multiclass_f1_score(rel_pred, rel_type, num_classes=num_relations).item()
    rel_ap = tmf.classification.multiclass_average_precision(rel_pred, rel_type, num_classes=num_relations).item()
    return rel_f1, rel_ap


if __name__ == '__main__':
    graph_emb = GraphEmb(
        node_attr_dim=14,
        node_attr_emb=64,
        node_grid_dim=7,
        node_grid_emb=64,
        edge_attr_dim=15,
        edge_attr_emb=64,
        edge_grid_dim=12,
        edge_grid_emb=64,
        graph_encoder_layers=3
    )
    model = FRModel(
        graph_emb=graph_emb,
        classify_hidden_dim=64,
        segment_hidden_dim=16,
        rel_hidden_dim=16,
        num_classes=27,
        num_relations=8
    )
    checkpoint = torch.load(
        "../checkpoints3/FRModel-full/lightning_logs/version_11/checkpoints/epoch=27-cls_val_accuracy=0.8258-seg_val_ap=0.9388-rel_val_ap=0.9755.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    train_path = "../data/dataset/train.pt"
    val_path = "../data/dataset/val.pt"
    test_path = "../data/dataset/test.pt"
    train_dataset = torch.load(train_path)
    val_dataset = torch.load(val_path)
    test_dataset = torch.load(test_path)

    dataset = []
    dataset.extend(train_dataset)
    dataset.extend(val_dataset)
    dataset.extend(test_dataset)
    del train_dataset, val_dataset, test_dataset

    # data_path = "../data/dataset/clean_dataset.pt"
    # dataset = torch.load(data_path)

    num_data = len(dataset)
    result = pd.DataFrame(data=None,
                          columns=['name', 'cls_acc', 'seg_f1', 'seg_ap', 'rel_f1', 'rel_ap'])
    for i in tqdm(range(num_data)):
        graph = dataset[i].to(device)
        graph['batch'] = torch.zeros(graph['num_nodes'], dtype=torch.long)
        graph_name = graph['name']
        cls_label = graph['cls_label']

        with torch.no_grad():
            classify_out, seg_emb, predict_emb = model(graph)

        cls_acc = tmf.classification.multiclass_accuracy(
            F.softmax(classify_out, dim=1), cls_label, num_classes=27).item()
        seg_f1, seg_ap = decode_seg_emb_v2(seg_emb, graph, model)
        rel_f1, rel_ap = decode_rel_emb_v2(predict_emb, graph, model)

        result.loc[i] = [graph_name, cls_acc, seg_f1, seg_ap, rel_f1, rel_ap]

    result.to_csv("./test_result.csv")
