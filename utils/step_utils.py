import numpy as np

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.STEPConstruct import stepconstruct_FindEntity
from OCC.Core.TCollection import TCollection_HAsciiString
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepPrim import BRepPrim_Builder
from OCC.Core.TopoDS import TopoDS_Vertex
from OCC.Core.gp import gp_Ax1
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Extend.ShapeFactory import rotate_shape, translate_shp
from OCC.Core.gp import gp_Trsf

from occwl.geometry import geom_utils
from occwl.edge import Edge
from occwl.face import Face
from occwl.vertex import Vertex


def triangulate_shape(shape):
    linear_deflection = 0.1
    angular_deflection = 0.5
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()
    assert mesh.IsDone()


def face_is_same(face1, face2):
    is_same = True
    explorer = TopExp_Explorer(face1, TopAbs_VERTEX)
    while explorer.More():
        v = topods.Vertex(explorer.Current())
        dss = BRepExtrema_DistShapeShape()
        dss.LoadS1(v)
        dss.LoadS2(face2)
        dss.Perform()
        assert dss.IsDone()
        if dss.Value() > 1e-7:
            is_same = False
            break
        explorer.Next()
    if is_same:
        adaptor = BRepAdaptor_Surface(face1)
        umin = adaptor.FirstUParameter()
        umax = adaptor.LastUParameter()
        vmin = adaptor.FirstVParameter()
        vmax = adaptor.LastVParameter()
        midu = (umax + umin) / 2
        midv = (vmax + vmin) / 2
        center = gp_Pnt()
        v1, v2 = gp_Vec(), gp_Vec()
        adaptor.D1(midu, midv, center, v1, v2)
        dss = BRepExtrema_DistShapeShape()
        builder = BRepPrim_Builder()
        vertex = TopoDS_Vertex()
        builder.MakeVertex(vertex, center)
        dss.LoadS1(vertex)
        dss.LoadS2(face2)
        dss.Perform()
        assert dss.IsDone()
        if dss.Value() > 1e-7:
            is_same = False
    return is_same


def face_is_overlap(face1, face2):
    is_overlap = False
    explorer = TopExp_Explorer(face1, TopAbs_VERTEX)
    while explorer.More():
        v = topods.Vertex(explorer.Current())
        dss = BRepExtrema_DistShapeShape()
        dss.LoadS1(v)
        dss.LoadS2(face2)
        dss.Perform()
        assert dss.IsDone()
        if dss.Value() < 1e-7:
            is_overlap = True
            break
        explorer.Next()
    return is_overlap


def edge_on_face(edge, face):
    edge = Edge(edge)
    start_point = edge.start_vertex().point()
    end_point = edge.end_vertex().point()
    mid_point = (start_point + end_point) / 2
    test_points = [start_point, end_point, mid_point]

    is_same = True
    for tp in test_points:
        tpv = Vertex.make_vertex(tp).topods_shape()
        dss = BRepExtrema_DistShapeShape()
        dss.LoadS1(tpv)
        dss.LoadS2(face)
        dss.Perform()
        assert dss.IsDone()
        if dss.Value() > 1e-5:
            is_same = False
            break
    return is_same


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


def shape_with_fid_to_step(filename, shape, id_map):
    """Save shape to a STEP file format.

    :param filename: Name to save shape as.
    :param shape: Shape to be saved.
    :param id_map: Variable mapping labels to faces in shape.
    :return: None
    """
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)

    finderp = writer.WS().TransferWriter().FinderProcess()
    faces = list_face(shape)
    loc = TopLoc_Location()
    for face in faces:
        item = stepconstruct_FindEntity(finderp, face, loc)
        if item is None:
            print(face)
            continue
        item.SetName(TCollection_HAsciiString(str(id_map[face])))

    writer.Write(filename)


def allocation_amount(num, amount):
    """
    生成总和为amount的num个随机数序列
    :param num: 随机数个数
    :param amount: 总和
    :return: 随机数序列
    """
    # 生成小数随机数
    a = [np.random.uniform(0, amount) for i in range(num - 1)]
    # 生成整数随机数
    # a = [np.random.randint(0, amount) for i in range(num_people-1)]
    a.append(0)
    a.append(amount)
    a.sort()
    # print(a)
    b = [a[i + 1] - a[i] for i in range(num)]  # 列表推导式，计算列表中每两个数之间的间隔
    # print(b)
    return b


def generate_orthogonal_vectors(vector, num_vectors):
    """
    随机生成与给定向量正交的n个单位向量
    :param vector:
    :param num_vectors:
    :return:
    """
    vectors = []
    vectors.append(vector / np.linalg.norm(vector))  # 将给定向量标准化为单位向量

    for _ in range(num_vectors):
        random_vector = np.random.rand(len(vector))  # 生成随机向量
        orthogonal_vector = random_vector - sum(np.dot(random_vector, v) * v for v in vectors)  # Gram-Schmidt正交化
        orthogonal_vector /= np.linalg.norm(orthogonal_vector)  # 将正交向量标准化为单位向量
        vectors.append(orthogonal_vector)

    return vectors[1:]


def face_translate(face, offset):
    """
    线性移动
    :param face: 面
    :param offset: numpy，移动向量
    :return:
    """
    return translate_shp(face, geom_utils.numpy_to_gp_vec(offset), True)


def face_rotate(face, axis, degree, origin=np.zeros(3, dtype=np.float32)):
    """
    旋转
    :param face: 面
    :param axis: 旋转轴
    :param degree: 角度（度）
    :param origin: 旋转轴原点
    :return:
    """
    return rotate_shape(face,
                        gp_Ax1(geom_utils.numpy_to_gp(origin), geom_utils.numpy_to_gp_dir(axis)),
                        degree, unite="deg")


def face_mirror(face, axis, origin):
    """
    镜像面
    :param face: 面
    :param axis: 镜像轴
    :param origin: 镜像轴原点
    :return:
    """
    mirror_axis = gp_Ax1(geom_utils.numpy_to_gp(origin), geom_utils.numpy_to_gp_dir(axis))

    trsf = gp_Trsf()
    trsf.SetMirror(mirror_axis)
    brep_trsf = BRepBuilderAPI_Transform(face, trsf, True)

    return brep_trsf.Shape()


def line_array(face, array_num, direction, space):
    """
    线性阵列
    :param face: 面
    :param array_num: 阵列数量
    :param direction: numpy， 阵列方向向量
    :param space: 阵列间距
    :return:
    """
    current_size = np.linalg.norm(direction)
    scale_factor = space / current_size
    offset = direction * scale_factor
    result = [face]
    for i in range(1, array_num):
        result.append(face_translate(result[i - 1], offset))
    return result


def circle_array(face, array_num, axis, origin=np.zeros(3, dtype=np.float32)):
    """
    圆周阵列
    :param face: 面
    :param array_num: 阵列数量
    :param axis: 圆心轴
    :param origin: 圆心坐标
    :return:
    """
    degree = 360 / array_num
    result = [face]
    for i in range(1, array_num):
        result.append(face_rotate(result[i - 1], axis, degree, origin))
    return result


def transfer_bound(origin_bound, direction, offset):
    current_size = np.linalg.norm(direction)
    scale_factor = offset / current_size
    transfer_vector = direction * scale_factor
    result_bound = np.zeros((5, 3))
    for i in range(4):
        point = origin_bound[i]
        result_bound[i] = point + transfer_vector
    result_bound[4] = origin_bound[4]
    return result_bound


if __name__ == "__main__":
    # start = np.array([0, 0, 0])
    # end = np.array([0, 5, 0])
    # edge = Edge.make_line_from_points(start, end)
    # face = Face.make_prism(edge, np.array([5, 0, 0]))
    # print(edge_on_face(edge.topods_shape(), face.topods_shape()))
    origin_bound = np.array([
        [0, 0, 0],
        [5, 0, 0],
        [5, 5, 0],
        [0, 5, 0],
        [0, 0, 1]
    ])
    direction = np.array([1, 1, 0])
    offset = 5
    result_bound = transfer_bound(origin_bound, direction, offset)
    print(result_bound)
    for i in range(4):
        print(np.linalg.norm(result_bound[i] - origin_bound[i]))
