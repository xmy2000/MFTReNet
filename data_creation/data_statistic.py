import os
import glob
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from Utils import parameters as param

# plt.style.use("ggplot")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc('font', family='Times New Roman')
# plt.rcParams.update({"font.size": 18})


def process_rel_json(json_path):
    f = open(json_path, 'r')
    content = f.read()
    rel_map = json.loads(content)
    rel_lst = rel_map['relation']
    f.close()

    rel_count = len(rel_lst)
    rel_name_list = []
    if rel_count > 0:
        for r in rel_lst:
            rel_name_list.append(r[0])

    return rel_count, rel_name_list


def process_res_json(json_path):
    f = open(json_path, 'r')
    content = f.read()
    res_map = json.loads(content)
    cls_map = res_map['cls']
    seg_lst = res_map['seg']
    bottom_map = res_map['bottom']
    f.close()

    feature_count = 0
    feature_type_count = {}
    for i in range(24):
        feature_type_count[param.feat_names[i]] = 0

    for seg in seg_lst:
        if len(seg) > 0:
            feature_count += 1
            feature_type = None
            try:
                for face in seg:
                    f_type = cls_map[str(face)]
                    if feature_type is None or feature_type == f_type:
                        feature_type = f_type
                    else:
                        raise Exception(f"{json_path} has seg error!")
            except Exception as e:
                print(e)
                continue

            feature_type_count[param.feat_names[feature_type]] += 1

    return feature_count, feature_type_count


if __name__ == "__main__":
    data_path = "../data/labels"
    rel_lst_path = glob.glob(os.path.join(data_path, "*_result_rel.json"))
    res_map_path = glob.glob(os.path.join(data_path, "*_result.json"))
    assert len(rel_lst_path) == len(res_map_path)

    # 生成总的样本数量
    sample_count = len(rel_lst_path)
    print("generate samples: ", sample_count)

    # 关系标签
    print("*" * 30)
    rel_count_lst = []  # 每个模型的关系数量
    rel_type_count = {"general_paratactic": 0, "line_array": 0, "circle_array": 0, "mirror": 0,
                      "intersecting": 0, "transition": 0, "superpose_on": 0}  # 每个关系类型的数量
    for rlp in rel_lst_path:
        rel_count, rel_name_lst = process_rel_json(rlp)
        rel_count_lst.append(rel_count)
        for rel_name in rel_name_lst:
            rel_type_count[rel_name] += 1
    print("每个模型平均包含的关系数量: ", sum(rel_count_lst) / sample_count)
    print("各个类型关系的数量: ", rel_type_count)

    rel_type_count['coplanar'] = rel_type_count.pop("general_paratactic")
    rel_type_count['depend-on'] = rel_type_count.pop("superpose_on")

    plt.figure()
    sns.histplot(rel_count_lst, shrink=2)
    plt.title("relationship count")
    plt.xlabel("rel_count")
    plt.ylabel("sample_count")
    plt.xticks(list(range(20)))

    plt.figure()
    sns.barplot(x=list(rel_type_count.values()), y=list(rel_type_count.keys()))
    plt.title("Distribution of topological relationship types")
    plt.xlabel("Number of instances")
    plt.tight_layout()
    plt.savefig('./rel_type.png', dpi=300, bbox_inches='tight')

    # 特征标签
    print("*" * 30)
    feature_count_lst = []
    feature_type_count = {}
    for i in range(24):
        feature_type_count[param.feat_names[i]] = 0

    for rmp in res_map_path:
        feature_count, type_count = process_res_json(rmp)
        feature_count_lst.append(feature_count)
        for type, count in type_count.items():
            if count > 0:
                feature_type_count[type] += count
    print("每个模型平均包含的特征数量: ", sum(feature_count_lst) / sample_count)
    print("各个类型特征的数量: ", feature_type_count)

    plt.figure()
    sns.histplot(feature_count_lst, shrink=2)
    plt.title("feature count")
    plt.xlabel("feature_count")
    plt.ylabel("sample_count")
    plt.xticks(list(range(25)))

    plt.figure()
    sns.barplot(x=list(feature_type_count.values()), y=list(feature_type_count.keys()))
    plt.title("Distribution of feature types")
    plt.xlabel("Number of instances")
    plt.tight_layout()
    plt.savefig('./feature_type.png', dpi=300, bbox_inches='tight')

    plt.show()
