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




def build_max_length_ranking(distances):
    d = dict()
    for id_, v_ in tqdm(distances.items()):
        a = sorted(v_.items(), key=lambda x: x[1])
        ranking = [(t[0]) for t in a]
        d[id_] = ranking
    return d

def main(args):
    distances_filename = args.distances
    note = args.note
    distances = load_data(filename=distances_filename)

    max_length_ranking = build_max_length_ranking(distances=distances)

    output_path = os.path.dirname(distances_filename)
    basename = 'max_length_ranking_'+note+'.pkl'
    final_pathname = os.path.join(output_path,basename)
    save_data(filename=final_pathname, dict=max_length_ranking)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distances', '-d', required=False, type=str, default='./distances_cc_peak_1.pkl',
                        help='path to pkl distances file')
    parser.add_argument('--note', '-n', required=False, type=str, default='',
                        help='annotation')
    args = parser.parse_args()
    main(args)