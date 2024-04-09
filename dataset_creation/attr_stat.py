import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

graph_path = Path("../data/graphs")

face_attr = []
edge_attr = []

graph_name_list = list(graph_path.glob("*_result.json"))
for graph_name in tqdm(graph_name_list, total=len(graph_name_list)):
    try:
        with open(graph_name, "r") as fp:
            data = json.load(fp)
        fp.close()
        attribute_map = data[1]
        face_attr.extend(attribute_map['graph_face_attr'])
        edge_attr.extend(attribute_map['graph_edge_attr'])
    except Exception as e:
        print(e)
        print(graph_name)
        continue

face_attr = np.array(face_attr)
edge_attr = np.array(edge_attr)

mean_face_attr = np.mean(face_attr, axis=0).tolist()
std_face_attr = np.std(face_attr, axis=0).tolist()
mean_edge_attr = np.mean(edge_attr, axis=0).tolist()
std_edge_attr = np.std(edge_attr, axis=0).tolist()

result_json = {
    "mean_face_attr": mean_face_attr,
    "std_face_attr": std_face_attr,
    "mean_edge_attr": mean_edge_attr,
    "std_edge_attr": std_edge_attr
}

json_file = open("../data/attr_stat.json", mode='w')
json.dump(result_json, json_file)
