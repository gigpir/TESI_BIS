import multiprocessing
import os
import sys
# insert at 1, 0 is the script path (or '' in REPL)
import time
from functools import partial
from tqdm import tqdm

sys.path.insert(1, '/home/crottondi/PIRISI_TESI/TESI_BIS/')
import numpy as np
import argparse
from primary.data_io import save_data, load_data
from primary.heatmap import compute_heatmap_distance, compute_cross_correlation_distance, \
    compute_cross_correlation_distance_normalized
from operator import itemgetter
import os
import pandas as pd


artists = None
metric = None

def build_matrix_slave(metric,artists_ids):
    global artists
    d = dict()
    for i,a_outer_id in enumerate(artists_ids):
        d[a_outer_id] = dict()
        for a_inner_id, a_inner in artists.items():
            try:
                if artists[a_outer_id].tsne_heatmap is not None and a_inner.tsne_heatmap is not None:
                    if metric == 'cc_peak_1':
                        d[a_outer_id][a_inner_id] = compute_cross_correlation_distance(h1=artists[a_outer_id].tsne_heatmap,
                                                                                       h2=a_inner.tsne_heatmap)
                    elif metric == 'cc_peak_2':
                        d[a_outer_id][a_inner_id] = compute_cross_correlation_distance_normalized(
                            h1=artists[a_outer_id].tsne_heatmap,
                            h2=a_inner.tsne_heatmap)
            except Exception as e:
                print(e)
        print(i, ' / ', len(artists_ids))
    return d

def merge_dictionaries(result):
    final_dict = dict()
    for r in result:
        for id_, d in r.items():
            final_dict[id_] = d
    return final_dict

def build_matrix_master():
    global artists
    global metric
    start = time.time()
    func = partial(build_matrix_slave, metric)

    nproc = multiprocessing.cpu_count()
    artists_ids = list(artists.keys())

    split = np.array_split(artists_ids, nproc)

    with multiprocessing.Pool(nproc) as p:
        result = p.map(func, split)

    d = merge_dictionaries(result=result)

    print(time.time() - start)
    return d


def main(args):
    input_path = args.input_pkl
    output_path = args.output_path
    global metric
    metric = args.metric
    global artists
    artists = load_data(input_path)

    d = build_matrix_master()

    # save
    #df = pd.DataFrame(data=d)
    #df.to_excel(output_path)


    save_data(filename=output_path, dict=d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_pkl', '-i', required=True, type=str,
                        help='path to pkl artists dictionary it has to include heatmaps attached')
    parser.add_argument('--output_path', '-o', required=False, type=str, default='',
                        help='path where output data will be saved')
    parser.add_argument('--metric', '-m', required=False, type=str, default='cc_peak_1',
                        choices=['cc_peak_1', 'cc_peak_2'], help='metric type:\n'
                                                                 'cc_peak_1 con peak_thresh=1\n'
                                                                 'cc_peak_2 calcolare la distanza da shift_0 e normalizzare (dividere) la distanza per il valore del picco')

    args = parser.parse_args()
    main(args)
