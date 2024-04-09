import copy
import json
import os.path
import random

import numpy as np

import Utils.parameters as param
import Utils.shape_factory as shape_factory
from Utils import occ_utils

from occwl.solid import Solid
from occwl.io import save_step
from occwl.compound import Compound

from utils.step_utils import *

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.TopoDS import (
    TopoDS_Solid,
    TopoDS_Compound,
    TopoDS_CompSolid,
)

from Features.o_ring import ORing
from Features.through_hole import ThroughHole
from Features.round import Round
from Features.chamfer import Chamfer
from Features.triangular_passage import TriangularPassage
from Features.rectangular_passage import RectangularPassage
from Features.six_sides_passage import SixSidesPassage
from Features.triangular_through_slot import TriangularThroughSlot
from Features.rectangular_through_slot import RectangularThroughSlot
from Features.circular_through_slot import CircularThroughSlot
from Features.rectangular_through_step import RectangularThroughStep
from Features.two_sides_through_step import TwoSidesThroughStep
from Features.slanted_through_step import SlantedThroughStep
from Features.blind_hole import BlindHole
from Features.triangular_pocket import TriangularPocket
from Features.rectangular_pocket import RectangularPocket
from Features.six_sides_pocket import SixSidesPocket
from Features.circular_end_pocket import CircularEndPocket
from Features.rectangular_blind_slot import RectangularBlindSlot
from Features.v_circular_end_blind_slot import VCircularEndBlindSlot
from Features.h_circular_end_blind_slot import HCircularEndBlindSlot
from Features.triangular_blind_step import TriangularBlindStep
from Features.circular_blind_step import CircularBlindStep
from Features.rectangular_blind_step import RectangularBlindStep

feat_names = ['chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', '6sides_passage',
              'triangular_through_slot', 'rectangular_through_slot', 'circular_through_slot',
              'rectangular_through_step', '2sides_through_step', 'slanted_through_step', 'Oring', 'blind_hole',
              'triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket',
              'rectangular_blind_slot', 'v_circular_end_blind_slot', 'h_circular_end_blind_slot',
              'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step', 'round',
              'plane', 'cylinder', 'cone']

feat_classes = {"chamfer": Chamfer, "through_hole": ThroughHole, "triangular_passage": TriangularPassage,
                "rectangular_passage": RectangularPassage, "6sides_passage": SixSidesPassage,
                "triangular_through_slot": TriangularThroughSlot, "rectangular_through_slot": RectangularThroughSlot,
                "circular_through_slot": CircularThroughSlot, "rectangular_through_step": RectangularThroughStep,
                "2sides_through_step": TwoSidesThroughStep, "slanted_through_step": SlantedThroughStep, "Oring": ORing,
                "blind_hole": BlindHole, "triangular_pocket": TriangularPocket, "rectangular_pocket": RectangularPocket,
                "6sides_pocket": SixSidesPocket, "circular_end_pocket": CircularEndPocket,
                "rectangular_blind_slot": RectangularBlindSlot, "v_circular_end_blind_slot": VCircularEndBlindSlot,
                "h_circular_end_blind_slot": HCircularEndBlindSlot, "triangular_blind_step": TriangularBlindStep,
                "circular_blind_step": CircularBlindStep, "rectangular_blind_step": RectangularBlindStep,
                "round": Round}

through_blind_features = ["triangular_passage", "rectangular_passage", "6sides_passage", "triangular_pocket",
                          "rectangular_pocket", "6sides_pocket", "through_hole", "blind_hole", "circular_end_pocket",
                          "Oring"]

transform_features = ['through_hole', 'triangular_passage', 'rectangular_passage', '6sides_passage', 'Oring',
                      'blind_hole', 'triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket']

transition_features = ["chamfer", "round"]

step_features = ["rectangular_through_step", "2sides_through_step", "slanted_through_step", "triangular_blind_step",
                 "circular_blind_step", "rectangular_blind_step"]

transform_types = ["general_paratactic", "line_array", "circle_array", "mirror", "intersecting"]


def rearrange_combo(combination):
    # 调整特征列表顺序
    transition_feats = []
    step_feats = []
    slot_feats = []
    through_feats = []
    blind_feats = []
    o_ring_feats = []

    for cnt, val in enumerate(combination):
        if val == param.feat_names.index("chamfer") or val == param.feat_names.index("round"):
            transition_feats.append(val)
        elif val == param.feat_names.index("rectangular_through_step") \
                or val == param.feat_names.index("2sides_through_step") \
                or val == param.feat_names.index("slanted_through_step") \
                or val == param.feat_names.index("triangular_blind_step") \
                or val == param.feat_names.index("circular_blind_step") \
                or val == param.feat_names.index("rectangular_blind_step"):
            step_feats.append(val)

        elif val == param.feat_names.index("triangular_through_slot") \
                or val == param.feat_names.index("rectangular_through_slot") \
                or val == param.feat_names.index("circular_through_slot") \
                or val == param.feat_names.index("rectangular_blind_slot") \
                or val == param.feat_names.index("v_circular_end_blind_slot") \
                or val == param.feat_names.index("h_circular_end_blind_slot"):
            slot_feats.append(val)

        elif val == param.feat_names.index("through_hole") \
                or val == param.feat_names.index("triangular_passage") \
                or val == param.feat_names.index("rectangular_passage") \
                or val == param.feat_names.index("6sides_passage"):
            through_feats.append(val)

        elif val == param.feat_names.index("blind_hole") \
                or val == param.feat_names.index("triangular_pocket") \
                or val == param.feat_names.index("rectangular_pocket") \
                or val == param.feat_names.index("6sides_pocket") \
                or val == param.feat_names.index("circular_end_pocket"):
            blind_feats.append(val)

        elif val == param.feat_names.index("Oring"):
            o_ring_feats.append(val)

    new_combination = step_feats + slot_feats + through_feats + blind_feats + o_ring_feats + transition_feats

    return new_combination


def generate_box_dims(larger_stock):
    # 生成基体三个维度的尺寸
    if larger_stock:  # too much features need larger stock for avoiding wrong topology
        stock_min_x = param.stock_min_x * 2
        stock_min_y = param.stock_min_y * 2
        stock_min_z = param.stock_min_z * 2
    else:
        stock_min_x = param.stock_min_x
        stock_min_y = param.stock_min_y
        stock_min_z = param.stock_min_z
    param.stock_dim_x = random.uniform(stock_min_x, param.stock_max_x)
    param.stock_dim_y = random.uniform(stock_min_y, param.stock_max_y)
    param.stock_dim_z = random.uniform(stock_min_z, param.stock_max_z)


def common_shape(shape1, shape2):
    # 实体交集操作
    result = BRepAlgoAPI_Common(shape1, shape2).Shape()
    unif = ShapeUpgrade_UnifySameDomain(result, False, True, False)
    unif.Build()
    result = unif.Shape()
    if result is None:
        raise Exception("common布尔操作失败")
    return result


def fuse_shape(shape1, shape2):
    # 实体并集操作
    result = BRepAlgoAPI_Fuse(shape1, shape2).Shape()
    unif = ShapeUpgrade_UnifySameDomain(result, False, True, False)
    unif.Build()
    result = unif.Shape()
    if result is None:
        raise Exception("fuse布尔操作失败")
    return result


def generate_shape_back(num_shape=1):  # 多个实体需要启动large
    # 产生组合体
    base_types = ["stock", "cylinder"]  # 基体类型
    base_type = random.choice(base_types)
    # shape_types = ["stock", "cylinder", "cone"]
    shape_types = ["stock", "cylinder"]  # 组合体类型
    shape_list = []
    shape_heights = allocation_amount(num_shape, param.stock_dim_z)  # 根据z方向总长度，随机生成n的总长一定的高度值

    if base_type == "stock":  # 建立立方体基体
        shape_list.append(Solid.make_box(param.stock_dim_x, param.stock_dim_y, shape_heights[0]).topods_shape())
    elif base_type == "cylinder":  # 建立圆柱体基体
        param.stock_dim_x = param.stock_dim_y = max(param.stock_dim_x, param.stock_dim_y)
        radius = param.stock_dim_x / 2
        shape_list.append(Solid.make_cylinder(radius=radius, height=shape_heights[0],
                                              base_point=(radius, radius, 0)).topods_shape())

    shape_x = [random.uniform(param.stock_min_x, param.stock_dim_x) for _ in range(num_shape - 1)]
    shape_x.sort()  # 生成各个组合体的x方向尺寸
    shape_x.append(param.stock_dim_x)
    shape_y = [random.uniform(param.stock_min_y, param.stock_dim_y) for _ in range(num_shape - 1)]
    shape_y.sort()  # # 生成各个组合体的y方向尺寸
    shape_y.append(param.stock_dim_y)

    # 各个组合体基面建立的xyz坐标系位置
    z_loc = 0
    x_loc = 0
    y_loc = 0
    for i in range(1, num_shape):
        shape_type = random.choice(shape_types)
        z_loc += shape_heights[i - 1]

        x_size = shape_x[num_shape - 1 - i]
        y_size = shape_y[num_shape - 1 - i]
        x_loc = x_loc + random.uniform(0, shape_x[num_shape - i] - x_size)
        y_loc = y_loc + random.uniform(0, shape_y[num_shape - i] - y_size)

        if shape_type == "stock":
            shape_list.append(
                Solid.make_box(x_size, y_size, shape_heights[i],
                               base_point=(x_loc, y_loc, z_loc)).topods_shape()
            )
        elif shape_type == "cylinder":
            radius = min(x_size, y_size) / 2
            shape_list.append(
                Solid.make_cylinder(radius=radius, height=shape_heights[i],
                                    base_point=(x_loc + radius, y_loc + radius, z_loc)).topods_shape()
            )
        elif shape_type == "cone":
            radius_bottom = min(x_size, y_size) / 2
            radius_top = random.uniform(0, radius_bottom)
            shape_list.append(
                Solid.make_cone(radius_bottom=radius_bottom, radius_top=radius_top, height=shape_heights[i],
                                base_point=(x_loc + radius_bottom, y_loc + radius_bottom, z_loc)).topods_shape()
            )
    # shape合并
    result_shape = shape_list[0]
    for i in range(1, num_shape):
        shape = shape_list[i]
        result_shape = fuse_shape(result_shape, shape)
    return result_shape


def generate_shape(num_shape=1):
    shape_types = ["stock", "cylinder", "cone"]
    shape_list = []
    shape_heights = allocation_amount(num_shape, param.stock_dim_z)  # 根据z方向总长度，随机生成n的总长一定的高度值

    x_size, y_size = param.stock_dim_x, param.stock_dim_y
    x_center, y_center, z_center = x_size / 2, y_size / 2, 0

    for i in range(num_shape):
        s_type = random.choice(shape_types)
        z_size = shape_heights[i]
        if s_type == "stock":
            shape_list.append(
                Solid.make_box(x_size, y_size, z_size,
                               base_point=(x_center - x_size / 2, y_center - y_size / 2, z_center)).topods_shape()
            )
        elif s_type == "cylinder":
            radius = min(x_size, y_size) / 2
            x_size = y_size = radius * 2 - 1
            shape_list.append(
                Solid.make_cylinder(radius=radius, height=z_size,
                                    base_point=(x_center, y_center, z_center)).topods_shape()
            )
        elif s_type == "cone":
            radius_bottom = min(x_size, y_size) / 2
            radius_top = random.uniform(radius_bottom / 2, radius_bottom)
            x_size = y_size = radius_top * 2 - 1
            shape_list.append(
                Solid.make_cone(radius_bottom=radius_bottom, radius_top=radius_top, height=z_size,
                                base_point=(x_center, y_center, z_center)).topods_shape()
            )

        x_size = random.uniform(x_size / 2, x_size)
        y_size = random.uniform(y_size / 2, y_size)
        z_center += z_size

    # shape合并
    result_shape = shape_list[0]
    for i in range(1, num_shape):
        s = shape_list[i]
        result_shape = fuse_shape(result_shape, s)
    return result_shape


def covert_face_to_id(shape, label_map):
    # 将OCC的面列表转为顺序id列表
    # 可能会报not found错误
    cls_dict, seg_list, bottom_dict = label_map
    faces = occ_utils.list_face(shape)

    cls_id = {}
    for f, c in cls_dict.items():
        cls_id[faces.index(f)] = c
    seg_id = []
    for seg in seg_list:
        ids = []
        for f in seg:
            ids.append(faces.index(f))
        seg_id.append(ids)
    bottom_id = {}
    for f, c in bottom_dict.items():
        bottom_id[faces.index(f)] = c

    return [cls_id, seg_id, bottom_id]


def label_map_to_json(label_map, file_path):
    # 将标签字典转为json储存，json无法储存OCC对象
    cls_dict, seg_list, bottom_dict = label_map
    result_json = {"cls": cls_dict, "seg": seg_list, "bottom": bottom_dict}

    json_file = open(file_path, mode='w')
    json.dump(result_json, json_file)


def rel_lst_to_json(rel_lst, file_path):
    # 将关系标签储存为json文件
    json_dict = {"relation": rel_lst}
    json_file = open(file_path, mode='w')
    json.dump(json_dict, json_file)


def create_feature_on_box(combo, box):
    # 在包围盒上创建特征
    try_cnt = 0
    find_edges = True
    combo = rearrange_combo(combo)  # rearrange machining feature combinations
    count = 0
    bounds = []  # 可行的草图包围边界列表
    feature_rel = []  # 关系列表

    while True:
        label_map = shape_factory.map_from_name(box, param.feat_names.index('plane'))
        for fid in combo:
            feat_name = param.feat_names[fid]
            apply_transform = None
            apply_depend = False
            if feat_name in transform_features:  # 是否加入特征变换
                apply_transform = random.choice(transform_types)
                # apply_transform = "intersecting"  # for debug
                # apply_transform = "circle_array"  # for debug
                # apply_transform = "mirror"  # for debug
            if feat_name not in step_features:  # 是否进行重叠特征构建
                apply_depend = random.choice([True, True, False])
                # apply_depend = random.choice([True])  # for debug

            if feat_name == "chamfer":  # 倒角特征处理
                base_feature_index = -1  # 倒角特征依赖的特征编号，任何特征均可以成为倒角和圆角特征的依赖特征
                if apply_depend and not isinstance(label_map, dict):
                    cls_dict = label_map[0]
                    seg_lst = label_map[1]
                    num_features = len(seg_lst)
                    step_feat_num = -1
                    for i in range(num_features):  # 寻找特征列表中非倒角和圆角特征的编号范围
                        face_id = 0
                        feature_cls = None
                        while face_id < len(seg_lst[i]):
                            try:
                                face = seg_lst[i][face_id]
                                feature_cls = cls_dict[face]
                                break
                            except KeyError as ke:
                                face_id += 1
                                continue
                        if feature_cls:
                            if param.feat_names[feature_cls] not in transition_features:
                                step_feat_num = i
                        else:
                            break
                    if step_feat_num > -1:
                        base_feature_index = random.randint(0, step_feat_num)

                if base_feature_index > -1:  # 如果有依赖特征，则施加倒角的边集为该特征的边集
                    seg_face_lst = label_map[1][base_feature_index]
                    edges = []
                    for sf in seg_face_lst:
                        edges.extend(occ_utils.list_edge(sf))
                else:
                    edges = occ_utils.list_edge(box)

                # create new feature object
                new_feat = feat_classes[feat_name](box, label_map, param.min_len,
                                                   param.clearance, param.feat_names, edges)
                box, label_map, edges = new_feat.add_feature(base_feature_index)

                if len(edges) == 0:
                    break

            elif feat_name == "round":  # 处理圆角特征
                base_feature_index = -1
                if apply_depend and not isinstance(label_map, dict):
                    cls_dict = label_map[0]
                    seg_lst = label_map[1]
                    num_features = len(seg_lst)
                    step_feat_num = -1
                    for i in range(num_features):
                        face_id = 0
                        feature_cls = None
                        while face_id < len(seg_lst[i]):
                            try:
                                face = seg_lst[i][face_id]
                                feature_cls = cls_dict[face]
                                break
                            except KeyError as ke:
                                face_id += 1
                                continue
                        if feature_cls:
                            if param.feat_names[feature_cls] not in transition_features:
                                step_feat_num = i
                        else:
                            break
                    if step_feat_num > -1:
                        base_feature_index = random.randint(0, step_feat_num)

                if base_feature_index > -1:
                    seg_face_lst = label_map[1][base_feature_index]
                    edges = []
                    for sf in seg_face_lst:
                        edges.extend(occ_utils.list_edge(sf))
                else:
                    edges = occ_utils.list_edge(box)

                new_feat = feat_classes[feat_name](box, label_map, param.min_len,
                                                   param.clearance, param.feat_names, edges)
                box, label_map, edges = new_feat.add_feature(base_feature_index)

                if len(edges) == 0:
                    break

            else:  # 处理倒角和圆角外的其他特征
                triangulate_shape(box)
                new_feat = feat_classes[feat_name](box, label_map, param.min_len, param.clearance, param.feat_names)

                base_step_index = -1  # 其他特征的依赖特征只能为step，寻找step特征的编号范围
                if apply_depend and not isinstance(label_map, dict):
                    cls_dict = label_map[0]
                    seg_lst = label_map[1]
                    num_features = len(seg_lst)
                    step_feat_num = -1
                    for i in range(num_features):
                        face_id = 0
                        feature_cls = None
                        while face_id < len(seg_lst[i]):
                            try:
                                face = seg_lst[i][face_id]
                                feature_cls = cls_dict[face]
                                break
                            except KeyError as ke:
                                face_id += 1
                                continue
                        if feature_cls:
                            if param.feat_names[feature_cls] in step_features:
                                step_feat_num = i
                        else:
                            break
                    if step_feat_num > -1:
                        base_step_index = random.randint(0, step_feat_num)

                if count == 0:
                    box, label_map, bounds = new_feat.add_feature(bounds, find_bounds=True,
                                                                  transformer=apply_transform,
                                                                  base_feature=base_step_index)

                    if feat_name in through_blind_features:
                        count += 1
                else:
                    box, label_map, bounds = new_feat.add_feature(bounds, find_bounds=True,
                                                                  transformer=apply_transform,
                                                                  base_feature=base_step_index)  # orignial: False
                    count += 1

                if apply_transform in ["general_paratactic", "intersecting"]:  # 寻找与当前特征并列的特征，非当前特征的类型
                    transfer_try_cnt = 0
                    transfer_success = False
                    while transfer_try_cnt < 10:  # 尝试10次
                        transfer_try_cnt += 1

                        transform_features_copy = transform_features.copy()
                        transform_features_copy.remove(feat_name)
                        paratactic_feature = random.choice(transform_features_copy)  # 进行并列操作的特征

                        sketch_attribute = new_feat.sketch_attribute  # 读取当前特征的草图属性
                        origin_bound = sketch_attribute["face_bound"]
                        direction = random.choice(sketch_attribute["face_directions"])
                        box_length = sketch_attribute["face_box_length"]
                        if apply_transform == "general_paratactic":
                            offset = random.uniform(np.sqrt(box_length * box_length),
                                                    np.sqrt(box_length * box_length) + 5)
                        elif apply_transform == "intersecting":
                            offset = random.uniform(box_length / 2, np.sqrt(box_length * box_length))

                        new_bound = transfer_bound(origin_bound, direction, offset)  # 通过对原草图边界偏移得到新特征的草图边界
                        if True in np.isnan(new_bound):  # 新草图边界中有nan
                            continue
                        # 判断移动后的边界仍然在面上
                        box_face_lst = occ_utils.list_face(box)
                        for box_face in box_face_lst:
                            on_face = True
                            occwl_face = Face(box_face)
                            if occwl_face.surface_type() == "plane":
                                for i in range(4):
                                    point = new_bound[i]
                                    face_point = occwl_face.find_closest_point_data(point)
                                    distance = face_point.distance
                                    if distance > 1e-7:
                                        on_face = False
                                        break
                            if on_face is True:
                                break
                        if on_face is False:
                            continue

                        # 添加并列特征
                        triangulate_shape(box)  # 网格化实体，不然add_feature会报错！！！
                        new_feat_2 = feat_classes[paratactic_feature](box, label_map, param.min_len,
                                                                      param.clearance,
                                                                      param.feat_names)
                        try_box, try_label_map, _ = new_feat_2.add_feature([new_bound], find_bounds=False)

                        origin_face = new_feat.sketch_face
                        new_face = new_feat_2.sketch_face
                        if new_face is None:  # 并列特征添加失败
                            raise Exception("new face is None")

                        overlap = face_is_overlap(origin_face, new_face)  # 判断原特征草图和新特征草图是否有重叠
                        if apply_transform == "general_paratactic" and overlap is False:  # 并列要求无重叠
                            transfer_success = True
                            break
                        elif apply_transform == "intersecting" and overlap is True:  # 干涉要求有重叠
                            transfer_success = True
                            break
                    if transfer_success:
                        box = try_box
                        label_map = try_label_map
                        seg_lst = label_map[1]
                        feature_rel.append((apply_transform, [len(seg_lst) - 1, len(seg_lst) - 2]))  # 添加并列标签和依赖标签
                        if base_step_index > -1:
                            feature_rel.append(("superpose_on", [len(seg_lst) - 1, base_step_index]))

            if new_feat.feature_relationship is not None:
                feature_rel.extend(new_feat.feature_relationship)

        if box is not None:
            break

        try_cnt += 1
        if try_cnt > len(combo):
            box = None
            label_map = None
            break

    return box, label_map, feature_rel


def label_map_from_box_to_shape(result_path, box_path, label_map_id_path):
    # 将包围盒的特征标签转换到组合体上
    box = Compound.load_from_step(box_path).topods_shape()
    result = Compound.load_from_step(result_path).topods_shape()
    box_faces = occ_utils.list_face(box)
    res_faces = occ_utils.list_face(result)

    f = open(label_map_id_path, 'r')
    content = f.read()
    label_map = json.loads(content)
    bf_cls, bf_seg, bf_bottom = label_map["cls"], label_map["seg"], label_map["bottom"]
    f.close()

    # 建立result和box面映射
    rf_to_bf = {}
    for i in range(len(res_faces)):
        find = False
        rf = res_faces[i]
        for j in range(len(box_faces)):
            bf = box_faces[j]
            if face_is_same(rf, bf):
                if not find:
                    # print(i, "->", j)
                    rf_to_bf[i] = j
                    find = True
                else:
                    raise Exception("多个面相同")
        if not find:
            rf_to_bf[i] = None
    # print(rf_to_bf)

    res_face_cls = {}
    for rf_id in range(len(res_faces)):
        rf = res_faces[rf_id]
        face = Face(rf)
        rf_type = face.surface_type()
        if rf_type == "plane":
            res_face_cls[rf_id] = 24
        elif rf_type == "cylinder":
            res_face_cls[rf_id] = 25
        elif rf_type == "cone":
            res_face_cls[rf_id] = 26
        else:
            res_face_cls[rf_id] = 24

        bf_id = rf_to_bf[rf_id]
        if bf_id is not None:
            cls = bf_cls[str(bf_id)]
            if cls != 24:
                res_face_cls[rf_id] = cls

    res_face_seg = []  # seg中存在空列表
    bfs = list(rf_to_bf.values())
    rfs = list(rf_to_bf.keys())
    for seg in bf_seg:
        res_seg = []
        for bf_id in seg:
            if bf_id in bfs:
                rf = rfs[bfs.index(bf_id)]
                res_seg.append(rf)
        if len(res_seg) > 0:
            res_face_seg.append(res_seg)
        else:
            res_face_seg.append([])

    res_face_bottom = {}
    for rf_id in range(len(res_faces)):
        bf_id = rf_to_bf[rf_id]
        if bf_id:
            res_face_bottom[rf_id] = bf_bottom[str(bf_id)]
        else:
            res_face_bottom[rf_id] = 0

    res_label_map = [res_face_cls, res_face_seg, res_face_bottom]
    return res_label_map


def rel_lst_from_box_to_shape(box_rel_lst, res_label_map):
    # 将包围盒的关系列表转换到组合体上
    res_rel_lst = []
    if len(box_rel_lst) == 0:
        return res_rel_lst

    res_seg = res_label_map[1]
    for rel_name, rel_lst in box_rel_lst:
        tmp_lst = []
        min_feature_num = 2  # 特征关系最少需要两个特征
        if rel_name == "circle_array" or rel_name == "line_array":
            min_feature_num = 3

        for seg_id in rel_lst:
            if len(res_seg[seg_id]) > 0:
                tmp_lst.append(seg_id)
        if len(tmp_lst) >= min_feature_num:
            res_rel_lst.append((rel_name, tmp_lst))

    return res_rel_lst


def info(result_step_path, res_label_map, res_rel_lst):
    res_cls, res_seg = res_label_map[0], res_label_map[1]

    cls_count = len(set(res_cls.values())) - 2

    seg_count = 0
    for seg_lst in res_seg:
        if len(seg_lst) > 0:
            seg_count += 1

    rel_count = len(res_rel_lst)
    rel_type = set()
    for rel_lst in res_rel_lst:
        rel_type.add(rel_lst[0])

    print(f"""
###############################################################
step file: {result_step_path}, 
feature type count: {cls_count}, feature count: {seg_count}, 
relationship count: {rel_count}, relationship type: {rel_type} 
###############################################################
    """)


def process(combo, f_name, step_path, label_path):
    # 文件保存目录
    shape_step_path = os.path.join(step_path, f_name + "_shape.step")
    box_step_path = os.path.join(step_path, f_name + "_box.step")
    result_step_path = os.path.join(step_path, f_name + "_result.step")

    box_label_path = os.path.join(label_path, f_name + "_box.json")
    box_rel_path = os.path.join(label_path, f_name + "_box_rel.json")
    result_label_path = os.path.join(label_path, f_name + "_result.json")
    result_rel_path = os.path.join(label_path, f_name + "_result_rel.json")

    # 生成组合体
    generate_box_dims(larger_stock=True)
    num_shape = random.randint(1, 3)
    shape = generate_shape(num_shape)
    # 检查生成的组合体shape
    if shape is None:
        raise Exception("shape is None")
    if Compound(shape).num_solids() > 1:
        raise Exception("shape solid > 1")
    save_step([Solid(shape, True)], shape_step_path)

    # 包围盒特征生成与保存
    box = Solid.make_box(param.stock_dim_x, param.stock_dim_y, param.stock_dim_z).topods_shape()
    triangulate_shape(box)
    box, label_map, box_rel_lst = create_feature_on_box(combo, box)
    save_step([Solid(box, True)], box_step_path)
    label_map_id = covert_face_to_id(box, label_map)
    label_map_to_json(label_map_id, box_label_path)
    rel_lst_to_json(box_rel_lst, box_rel_path)

    # 组合体特征生成与保存
    result = common_shape(box, shape)
    # 检查生成的结果result
    if result is None:
        raise Exception("result is None")
    if not isinstance(result, (TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid)):
        raise Exception('generated shape is {}, not supported'.format(type(shape)))
    if Compound(result).num_solids() > 1:
        raise Exception("result solid > 1")
    faces_list = occ_utils.list_face(result)
    if len(faces_list) == 0:
        raise Exception('empty result shape')
    save_step([Solid(result, True)], result_step_path)

    res_label_map = label_map_from_box_to_shape(result_step_path, box_step_path, box_label_path)
    check = False
    for seg in res_label_map[1]:
        if len(seg) > 0:
            check = True
            break
    if check is False:
        raise Exception('no seg')

    label_map_to_json(res_label_map, result_label_path)
    res_rel_lst = rel_lst_from_box_to_shape(box_rel_lst, res_label_map)
    rel_lst_to_json(res_rel_lst, result_rel_path)

    info(result_step_path, res_label_map, res_rel_lst)
    print(f"SUCCESS!")
    return True
