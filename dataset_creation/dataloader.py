import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

relation_type = {
    "no_relation": 0,
    "superpose_on": 1,
    "transition": 2,
    "general_paratactic": 3,
    "line_array": 4,
    "circle_array": 5,
    "mirror": 6,
    "intersecting": 7
}


def attribute_standardization(data, mean, std):
    index = (std != 0.0)
    data = np.array(data)  # num_nodes * 14
    data[:, index] = (data[:, index] - mean[index]) / std[index]
    return data.tolist()


def load_one_graph(file_path):
    with open(file_path, "r") as fp:
        data = json.load(fp)

    graph_name = data[0]
    attribute_map = data[1]
    graph = attribute_map['graph']
    face_attr = attribute_map['graph_face_attr']
    face_grid = attribute_map['graph_face_grid']
    edge_attr = attribute_map['graph_edge_attr']
    edge_grid = attribute_map['graph_edge_grid']

    return graph_name, graph, face_attr, face_grid, edge_attr, edge_grid


def process_label(label_file_path):
    with open(label_file_path, "r") as fp:
        label = json.load(fp)
    cls_map, seg_lst = label['cls'], label['seg']

    cls_label = list(cls_map.values())
    cls_label_tensor = torch.tensor(cls_label, dtype=torch.float)

    edges = []
    for seg_faces in seg_lst:
        num_faces = len(seg_faces)
        if num_faces == 0:
            continue
        for i in range(num_faces - 1):
            src_face = seg_faces[i]
            for j in range(i + 1, num_faces):
                tar_face = seg_faces[j]
                edges.append([src_face, tar_face])
                # edges.append([tar_face, src_face])
    edge_tensor = torch.tensor(edges, dtype=torch.long)
    edge_tensor_unique = torch.unique(edge_tensor, dim=0)
    if edge_tensor.shape != edge_tensor_unique.shape:
        print(f"WARNING! {label_file_path} has same edge in seg label.")

    return cls_label_tensor, edge_tensor_unique


def generate_relation_label(label_file_path, relation_file_path):
    with open(label_file_path, "r") as lf:
        label = json.load(lf)
    lf.close()
    cls_map, seg_lst = label['cls'], label['seg']

    with open(relation_file_path, 'r') as rf:
        relation = json.load(rf)
    rf.close()
    rel_lst = relation['relation']

    relation_edge = []
    relation_edge_type = []
    for rel_name, feature_lst in rel_lst:
        rel_cls = relation_type[rel_name]

        for feature_id1 in feature_lst:
            for feature_id2 in feature_lst:
                if feature_id1 != feature_id2:
                    face_lst1 = seg_lst[feature_id1]
                    face_lst2 = seg_lst[feature_id2]

                    for node1 in face_lst1:
                        for node2 in face_lst2:
                            relation_edge.append([node1, node2])
                            relation_edge_type.append(rel_cls)
    relation_edge_tensor = torch.tensor(relation_edge, dtype=torch.long)
    relation_edge_type_tensor = torch.tensor(relation_edge_type, dtype=torch.long)
    return relation_edge_tensor, relation_edge_type_tensor


def read_txt(file_path):
    f = open(file_path, 'r')
    result = f.read().splitlines()
    f.close()
    return result


def generate_dataset(
        graph_path,
        label_path,
        partition_path,
        standardization=True,
):
    attr_stat_path = "../data2/attr_stat.json"
    with open(attr_stat_path, "r") as asp:
        attr_stat = json.load(asp)
    asp.close()
    mean_face_attr = np.array(attr_stat['mean_face_attr'])
    std_face_attr = np.array(attr_stat['std_face_attr'])
    mean_edge_attr = np.array(attr_stat['mean_edge_attr'])
    std_edge_attr = np.array(attr_stat['std_edge_attr'])

    graph_names = read_txt(partition_path)
    graph_dataset = []
    for gn in tqdm(graph_names, total=len(graph_names)):
        gnp = os.path.join(graph_path, gn + ".json")
        graph_name, graph, face_attr, face_grid, edge_attr, edge_grid = load_one_graph(gnp)
        try:
            if standardization:
                face_attr = attribute_standardization(face_attr, mean_face_attr, std_face_attr)
                edge_attr = attribute_standardization(edge_attr, mean_edge_attr, std_edge_attr)

            face_attr = torch.tensor(face_attr, dtype=torch.float)
            face_grid = torch.tensor(face_grid, dtype=torch.float)  # num_face*7*5*5
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            edge_grid = torch.tensor(edge_grid, dtype=torch.float)  # num_edge*12*5
            edge_index = torch.tensor(graph['edges'], dtype=torch.long)

            label_file = os.path.join(label_path, graph_name + ".json")
            relation_file = os.path.join(label_path, graph_name + "_rel.json")

            cls_label, seg_label = process_label(label_file)
            seg_label_num = seg_label.shape[0]

            rel_edge, rel_type = generate_relation_label(label_file, relation_file)

            data = Data(
                name=graph_name,
                num_nodes=graph['num_nodes'],
                edge_index=edge_index,
                face_attr=face_attr,
                face_grid=face_grid,
                edge_attr=edge_attr,
                edge_grid=edge_grid,
                cls_label=cls_label,
                seg_label=seg_label,
                seg_label_num=seg_label_num,
                rel_edge=rel_edge,
                rel_type=rel_type,
                rel_num=rel_edge.shape[0]
            )
            graph_dataset.append(data)
        except Exception as e:
            print(e)
            continue

    print("Data Count: ", len(graph_dataset))
    return graph_dataset


def get_dataloader(dataset_path, batch_size=32, shuffle=True, num_workers=1):
    # graph_dataset = generate_dataset(graph_path, label_path, partition_path, standardization)
    dataset = torch.load(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_dataloader_and_split(dataset_path, batch_size=32, num_workers=1):
    dataset = torch.load(dataset_path)

    data_size = len(dataset)
    train_size = int(data_size * 0.7)
    val_size = int(data_size * 0.15)
    test_size = data_size - train_size - val_size

    train_data = dataset[0:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    dataset_type = ["train", "val", "test"]
    for dt in dataset_type:
        graph_path = "../data/graphs"
        label_path = "../data/labels"
        partition_path = f"../data/partition/{dt}.txt"

        dataset = generate_dataset(graph_path, label_path, partition_path, standardization=True)
        torch.save(dataset, f"../data/dataset/{dt}.pt")
        print(f"{dt} dataset finish!")
