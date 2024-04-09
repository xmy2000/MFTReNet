import random

from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.GProp import GProp_GProps
from OCC.Core._BRepGProp import brepgprop_SurfaceProperties

import Utils.shape_factory as shape_factory
import Utils.parameters as param
from Features.machining_features import MachiningFeature

import OCCUtils.edge
from utils.step_utils import edge_on_face


def ask_surface_area(f):
    props = GProp_GProps()

    brepgprop_SurfaceProperties(f, props)
    area = props.Mass()
    return area


class Round(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names, edges):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = None
        self.bound_type = None
        self.depth_type = None
        self.feat_type = "round"
        self.edges = edges
        self.feature_relationship = []

    def add_feature(self, base_feature_index=-1):
        fillet_maker = BRepFilletAPI_MakeFillet(self.shape)
        new_edges = []

        for edge in self.edges:
            e_util = OCCUtils.edge.Edge(edge)
            if e_util.length() >= self.min_len:
                new_edges.append(edge)

        self.edges = new_edges

        edge = None
        while len(self.edges) > 0:
            edge = random.choice(self.edges)
            e_util = OCCUtils.edge.Edge(edge)
            max_radius = e_util.length() / 10

            if max_radius > param.round_radius_max:
                max_radius = param.round_radius_max

            radius = random.uniform(param.round_radius_min, max_radius)

            try:
                fillet_maker.Add(radius, edge)
                shape = fillet_maker.Shape()
                self.edges.remove(edge)
                break

            except:
                self.edges.remove(edge)
                edge = None
                continue

        try:
            fmap = shape_factory.map_face_before_and_after_feat(self.shape, fillet_maker)
            labels = shape_factory.map_from_shape_and_name(fmap, self.labels,
                                                           shape, self.feat_names.index(self.feat_type),
                                                           None)
            if base_feature_index > -1 and edge is not None:  # 添加特征关联标签
                self.feature_relationship.append(("transition", [len(labels[1]) - 1, base_feature_index]))
                # print("Add round depend-on")

            return shape, labels, self.edges

        except:
            return self.shape, self.labels, self.edges
