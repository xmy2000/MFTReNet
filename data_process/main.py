import gc
import json
import tqdm
import os.path as osp
import numpy as np
from pathlib import Path
from itertools import repeat
from multiprocessing.pool import Pool

from graph_extractor import GraphExtractor


def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


def save_json_data(path_name, data):
    """Export a data to a json file"""
    with open(path_name, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


def process_one_file(args):
    file_path, config = args
    try:
        extractor = GraphExtractor(file_path, config, scale_body=True)
        out = extractor.process()
        graph_index = str(file_path.stem)
        graph = [graph_index, out]
        save_json_data(osp.join(output_path, graph_index + '.json'), graph)
        return [str(file_path.stem)]
    except Exception as e:
        print(e)
        return []


def initializer():
    import signal
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == '__main__':
    step_path = "../data/steps"
    output = "../data/graphs"
    attribute_config_path = "./attribute_config.json"
    num_workers = 16

    step_path = Path(step_path)
    output_path = Path(output)
    if not output_path.exists():
        output_path.mkdir()
    attribute_config_path = Path(attribute_config_path)

    attribute_config = load_json(attribute_config_path)
    step_files = list(step_path.glob("20240125*_result.step"))

    pool = Pool(processes=num_workers, initializer=initializer)
    try:
        results = list(tqdm.tqdm(
            pool.imap(process_one_file, zip(step_files, repeat(attribute_config))),
            total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

    pool.terminate()
    pool.join()

    graph_count = 0
    fail_count = 0
    graphs = []
    for res in results:
        if len(res) > 0:
            graph_count += 1
            graphs.append(res)
        else:
            fail_count += 1

    gc.collect()
    print(f"Process {len(results)} files. Generate {graph_count} graphs. Has {fail_count} failed files.")
