import random
import json

from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.AIS import AIS_ColoredShape
from OCC.Display.SimpleGui import init_display
from OCC.Display.OCCViewer import rgb_color

from occwl.compound import Compound

LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

FEAT_NAMES = ["Chamfer", "Through hole", "Triangular passage", "Rectangular passage", "6-sides passage",
              "Triangular through slot", "Rectangular through slot", "Circular through slot",
              "Rectangular through step", "2-sides through step", "Slanted through step", "O-ring", "Blind hole",
              "Triangular pocket", "Rectangular pocket", "6-sides pocket", "Circular end pocket",
              "Rectangular blind slot", "Vertical circular end blind slot", "Horizontal circular end blind slot",
              "Triangular blind step", "Circular blind step", "Rectangular blind step", "Round", "Stock"]

COLORS = {"Chamfer": 0, "Through hole": 490, "Triangular passage": 500, "Rectangular passage": 470,
          "6-sides passage": 100,
          "Triangular through slot": 120, "Rectangular through slot": 140, "Circular through slot": 160,
          "Rectangular through step": 180, "2-sides through step": 200, "Slanted through step": 220, "O-ring": 240,
          "Blind hole": 260,
          "Triangular pocket": 280, "Rectangular pocket": 300, "6-sides pocket": 320, "Circular end pocket": 340,
          "Rectangular blind slot": 360, "Vertical circular end blind slot": 380,
          "Horizontal circular end blind slot": 400,
          "Triangular blind step": 420, "Circular blind step": 440, "Rectangular blind step": 460, "Round": 480,
          "Stock": 60}


def list_face(shape):
    '''
    input
        shape: TopoDS_Shape
    output
        fset: {TopoDS_Face}
    '''
    """
    fset = set()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        s = exp.Current()
        exp.Next()
        face = topods.Face(s)
        fset.add(face)
    return list(fset)
    """
    topo = TopologyExplorer(shape)

    return list(topo.faces())


def display(shape, face_colors):
    occ_display.EraseAll()
    AIS = AIS_ColoredShape(shape)

    for f, c in face_colors.items():
        # set a custom color per-face
        AIS.SetCustomColor(f, c)

    occ_display.Context.Display(AIS, True)
    occ_display.View_Iso()
    occ_display.FitAll()


def display_with_color(shape, cls_map):
    cls = list(set(cls_map.values()))
    faces = list_face(shape)
    colors = []
    face_colors = {}
    for i in range(len(cls)):
        colors.append(rgb_color(random.random(), random.random(), random.random()))
    for i in range(len(faces)):
        face_colors[faces[i]] = colors[cls.index(cls_map[str(i)])]
    display(shape, face_colors)


def display_with_relationship(shape, seg_lst, rel_lst):
    nums = len(rel_lst)
    print("num of relationship: ", nums)
    faces = list_face(shape)
    face_color = {}
    base_color = rgb_color(1.0, 1.0, 1.0)
    relationship_colors = {"general_paratactic": rgb_color(235 / 255, 51 / 255, 36 / 255),  # red
                           "superpose_on": rgb_color(117 / 255, 250 / 255, 97 / 255),  # green
                           "intersecting": rgb_color(255 / 255, 253 / 255, 85 / 255),  # yellow
                           "transition": rgb_color(0 / 255, 35 / 255, 245 / 255),  # blue
                           "line_array": rgb_color(234 / 255, 54 / 255, 128 / 255),  # pink
                           "circle_array": rgb_color(142 / 255, 64 / 255, 58 / 255),  # brown
                           "mirror": rgb_color(240 / 255, 134 / 255, 80 / 255),  # orange
                           }

    for i in range(len(faces)):
        face_color[faces[i]] = base_color

    for i in range(nums):
        rel_name, feature_lst = rel_lst[i]
        # if rel_name != "intersecting":
        #     continue
        for fid in feature_lst:
            feature_faces = seg_lst[fid]
            for feature_face_id in feature_faces:
                face_color[faces[feature_face_id]] = relationship_colors[rel_name]

    display(shape, face_color)


if __name__ == '__main__':
    fid = 7
    time_info = "20231223_204137"
    # shape_name = "box"
    shape_name = "result"
    occ_display, start_occ_display, add_menu, add_function_to_menu = init_display()

    # shape = Compound.load_from_step(f"../data/steps/{time_info}_{fid}_{shape_name}.step").topods_shape()
    # f = open(f"../data/labels/{time_info}_{fid}_{shape_name}.json", 'r')
    # content = f.read()
    # label_map = json.loads(content)
    # cls_map = label_map['cls']
    # f.close()
    # display_with_color(shape, cls_map)

    shape = Compound.load_from_step(f"../data/steps/{time_info}_{fid}_result.step").topods_shape()
    f = open(f"../data/labels/{time_info}_{fid}_result.json", 'r')
    content = f.read()
    label_map = json.loads(content)
    seg_lst = label_map["seg"]
    f.close()
    f = open(f"../data/labels/{time_info}_{fid}_result_rel.json", 'r')
    content = f.read()
    rel_map = json.loads(content)
    rel_lst = rel_map["relation"]
    f.close()
    display_with_relationship(shape, seg_lst, rel_lst)

    start_occ_display()
