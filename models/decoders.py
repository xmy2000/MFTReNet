import torch
from torch import nn


class SegmentDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, cls_label, batch):
        """
        同一个graph内的同类标签节点才进行解码运算
        :param z: num_nodes * seg_emb
        :param cls_label: num_nodes
        :return:
        """
        batch_size = torch.max(batch) + 1
        num_nodes = cls_label.shape[0]
        seg_out = torch.zeros((num_nodes, num_nodes), dtype=torch.float).to(z.device)

        current_size = 0
        for i in range(batch_size):
            batch_index = torch.nonzero(batch == i).squeeze()
            one_z = z[batch_index]
            one_label = cls_label[batch_index]
            for cls in torch.unique(one_label):
                if cls == 24 or cls == 25 or cls == 26:
                    continue
                index = torch.nonzero(one_label == cls).squeeze()
                if index.dim() == 0 or len(index) == 0:
                    continue

                # 使用向量化操作计算内积
                z_cls = one_z[index]
                inner_product = torch.matmul(z_cls, z_cls.t())
                seg_out[torch.meshgrid(current_size + index, current_size + index)] = inner_product
            current_size += batch_index.shape[0]

        seg_out[range(num_nodes), range(num_nodes)] = 0.0
        return seg_out


class SegmentDecoderV2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2, cls_label, batch):
        """
        同一个graph内的同类标签节点才进行解码运算
        :param z: num_nodes * seg_emb
        :param cls_label: num_nodes
        :return:
        """
        batch_size = torch.max(batch) + 1
        num_nodes = cls_label.shape[0]
        seg_out = torch.zeros((num_nodes, num_nodes), dtype=torch.float).to(z1.device)

        current_size = 0
        for i in range(batch_size):
            batch_index = torch.nonzero(batch == i).squeeze()
            one_z1 = z1[batch_index]
            one_z2 = z2[batch_index]
            one_label = cls_label[batch_index]
            for cls in torch.unique(one_label):
                if cls == 24 or cls == 25 or cls == 26:
                    continue
                index = torch.nonzero(one_label == cls).squeeze()
                if index.dim() == 0 or len(index) == 0:
                    continue

                # 使用向量化操作计算内积
                z_cls1 = one_z1[index]
                z_cls2 = one_z2[index]
                inner_product = torch.matmul(z_cls1, z_cls2.t())
                seg_out[torch.meshgrid(current_size + index, current_size + index)] = inner_product
            current_size += batch_index.shape[0]

        seg_out[range(num_nodes), range(num_nodes)] = 0.0
        return seg_out


class FeatureRelationDecoder(nn.Module):
    def __init__(self, num_relations):
        super().__init__()
        self.num_relations = num_relations

    def forward(self, z, seg_matrix, cls_label, batch):
        """

        :param z: num_nodes * num_relations
        :param seg_matrix: num_nodes * num_nodes
        :param cls_label: num_nodes
        :return: num_nodes * num_nodes * num_relations
        """
        batch_size = torch.max(batch) + 1
        num_nodes = cls_label.shape[0]
        seg_matrix += torch.eye(num_nodes).to(z.device)
        result = torch.concat([
            torch.ones((num_nodes, num_nodes, 1), dtype=torch.float),
            torch.zeros((num_nodes, num_nodes, self.num_relations - 1), dtype=torch.float)
        ], dim=2).to(z.device)  # num_nodes * num_nodes * num_relations

        node_mask = ((cls_label != 24) & (cls_label != 25) & (cls_label != 26))  # num_nodes * 1
        seg_mask = seg_matrix == 0  # num_nodes * num_nodes
        for i in range(batch_size):
            batch_mask = batch == i
            mask = node_mask * batch_mask
            indices = torch.nonzero(mask).squeeze()  # indices of nodes to consider
            row_indices, col_indices = torch.meshgrid(indices, indices)  # create meshgrid of indices

            valid_indices = seg_mask[row_indices, col_indices]  # filter valid indices

            row_indices = row_indices[valid_indices]
            col_indices = col_indices[valid_indices]

            result[row_indices, col_indices] = z[row_indices] * z[col_indices]

        return result


class FeatureRelationDecoderV2(nn.Module):
    def __init__(self, num_relations):
        super().__init__()
        self.num_relations = num_relations

    def forward(self, z1, z2, seg_matrix, cls_label, batch):
        """

        :param z: num_nodes * num_relations
        :param seg_matrix: num_nodes * num_nodes
        :param cls_label: num_nodes
        :return: num_nodes * num_nodes * num_relations
        """
        batch_size = torch.max(batch) + 1
        num_nodes = cls_label.shape[0]
        seg_matrix += torch.eye(num_nodes).to(z1.device)
        result = torch.concat([
            torch.ones((num_nodes, num_nodes, 1), dtype=torch.float),
            torch.zeros((num_nodes, num_nodes, self.num_relations - 1), dtype=torch.float)
        ], dim=2).to(z1.device)  # num_nodes * num_nodes * num_relations

        node_mask = ((cls_label != 24) & (cls_label != 25) & (cls_label != 26))  # num_nodes * 1
        seg_mask = seg_matrix == 0  # num_nodes * num_nodes
        for i in range(batch_size):
            batch_mask = batch == i
            mask = node_mask * batch_mask
            indices = torch.nonzero(mask).squeeze()  # indices of nodes to consider
            row_indices, col_indices = torch.meshgrid(indices, indices)  # create meshgrid of indices

            valid_indices = seg_mask[row_indices, col_indices]  # filter valid indices

            row_indices = row_indices[valid_indices]
            col_indices = col_indices[valid_indices]

            result[row_indices, col_indices] = z1[row_indices] * z2[col_indices]

        return result


class FeatureRelationDecoderV3(nn.Module):
    def __init__(self, num_relations):
        super().__init__()
        self.num_relations = num_relations

    def forward(self, z, seg_matrix, cls_label, batch):
        batch_size = torch.max(batch) + 1
        num_nodes = cls_label.shape[0]
        seg_matrix += torch.eye(num_nodes).to(z.device)
        result = torch.concat([
            torch.ones((num_nodes, num_nodes, 1), dtype=torch.float),
            torch.zeros((num_nodes, num_nodes, self.num_relations - 1), dtype=torch.float)
        ], dim=2).to(z.device)  # num_nodes * num_nodes * num_relations

        mask = seg_matrix.bool()
        for b in range(batch_size):
            batch_node_index = torch.nonzero(batch == b).squeeze()

            node_set = set()
            seg_lst = []
            for node in batch_node_index:
                if node.item() in node_set:
                    continue
                seg = torch.nonzero(mask[node]).squeeze()
                if seg.numel() == 1:
                    node_set.add(seg.item())
                else:
                    seg_lst.append(seg)
                    node_set.update(seg.tolist())

            seg_num = len(seg_lst)
            for i in range(seg_num - 1):
                for j in range(i + 1, seg_num):
                    feature_face_index1, feature_face_index2 = seg_lst[i], seg_lst[j]
                    # z_lst = []
                    for f1 in feature_face_index1:
                        z1 = z[f1]
                        for f2 in feature_face_index2:
                            z2 = z[f2]
                            z_decode = (z1 * z2).unsqueeze(dim=0)
                            # z_lst.append(z_decode)
                            result[f1, f2] = z_decode
                            result[f2, f1] = z_decode
                    # z_emb = torch.cat(z_lst, dim=0)
                    # res_z = torch.mean(z_emb, dim=0)
                    # result[torch.meshgrid(feature_face_index1, feature_face_index2)] = res_z
                    # result[torch.meshgrid(feature_face_index2, feature_face_index1)] = res_z

        return result
