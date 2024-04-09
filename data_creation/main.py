import os
import gc
import sys
import time
import random
import traceback
import feature_creation
from itertools import repeat
from multiprocessing.pool import Pool


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


now = time.localtime()
now_time = time.strftime("%Y%m%d_%H%M%S", now)
sys.stdout = Logger("../data/logs/" + now_time + '.txt', sys.stdout)


def generate_shape(args):
    combo, step_path, label_path = args
    f_name, combination = combo
    print("*" * 50)
    print(f"Creating: {f_name}")

    num_try = 0  # first try
    while True:
        num_try += 1
        print('try count', num_try)
        if num_try > 10:
            # fails too much, pass
            print('number of fails > 10, pass')
            break
        try:
            success = feature_creation.process(combination, f_name, step_path, label_path)
            if success:
                break
        except Exception as e:
            print('Fail to generate:')
            # print(e)
            traceback.print_exc()
            continue


def initializer():
    import signal
    """
    Ignore CTRL+C in the worker process.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == "__main__":
    # 参数初始化
    dataset_dir = '../data'
    num_samples = 10000  # 生成样本数量
    combo_range = [3, 10]  # 样本特征数量采样范围
    num_features = 24  # 特征种类数量
    # num_workers = 1
    num_workers = 16

    # 数据保存目录创建
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    step_path = os.path.join(dataset_dir, 'steps')
    label_path = os.path.join(dataset_dir, 'labels')
    if not os.path.exists(step_path):
        os.mkdir(step_path)
    if not os.path.exists(label_path):
        os.mkdir(label_path)

    # 特征列表采样
    combos = []
    for idx in range(num_samples):
        num_inter_feat = random.randint(combo_range[0], combo_range[1])
        combo = [random.randint(0, num_features - 1) for _ in range(num_inter_feat)]  # no stock face

        file_name = now_time + '_' + str(idx)
        combos.append((file_name, combo))

    # combos = []
    # file_name = now_time + '_' + str(0)
    # combos.append((file_name, [22, 1, 4, 23]))

    if num_workers == 1:
        for combo in combos:
            generate_shape((combo, step_path, label_path))
    elif num_workers > 1:  # multiprocessing
        pool = Pool(processes=num_workers, initializer=initializer)
        try:
            result = list(pool.imap(generate_shape, zip(combos, repeat(step_path), repeat(label_path))))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
    else:
        AssertionError('error number of workers')

    gc.collect()
