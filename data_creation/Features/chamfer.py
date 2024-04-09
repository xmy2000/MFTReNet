import random

from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer

import Utils.shape_factory as shape_factory
import Utils.parameters as param
from Features.machining_features import MachiningFeature
from utils.step_utils import edge_on_face


class Chamfer(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names, edges):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = None
        self.bound_type = None
        self.depth_type = None
        self.feat_type = "chamfer"
        self.edges = edges
        self.feature_relationship = []

    def add_feature(self, base_feature_index=-1):
        while True:
            chamfer_maker = BRepFilletAPI_MakeChamfer(self.shape)

            # random choose a edge to make chamfer
            try:
                edge = random.choice(self.edges)
            except IndexError:
                print("No more edges")
                edge = None
                break

            try:
                depth = random.uniform(param.chamfer_depth_min, param.chamfer_depth_max)

                chamfer_maker.Add(depth, edge)
                shape = chamfer_maker.Shape()
                self.edges.remove(edge)
                break
            except RuntimeError:
                # if chamfer make fails, make chamfer with chamfer_depth_min
                try:
                    chamfer_maker = BRepFilletAPI_MakeChamfer(self.shape)
                    depth = param.chamfer_depth_min
                    chamfer_maker.Add(depth, edge)
                    shape = chamfer_maker.Shape()
                    self.edges.remove(edge)
                    break
                except:
                    self.edges.remove(edge)
                    continue

        try:
            fmap = shape_factory.map_face_before_and_after_feat(self.shape, chamfer_maker)
            labels = shape_factory.map_from_shape_and_name(fmap, self.labels,
                                                           shape, self.feat_names.index('chamfer'),
                                                           None)
            if base_feature_index > -1 and edge is not None:  # 添加特征关联关系标签
                self.feature_relationship.append(("transition", [len(labels[1]) - 1, base_feature_index]))
                # print("Add chamfer depend-on")

            return shape, labels, self.edges
        except:
            return self.shape, self.labels, self.edges
