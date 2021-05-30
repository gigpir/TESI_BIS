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



def main(args):
    input_path = args.input_pkl
    output_path = args.output_path
    if output_path[-1] != '/':
        output_path += '/'

    output_names = output_path + 'names.pkl'
    output_heatmaps = output_path + 'heatmaps.pkl'
    output_gt = output_path + 'ground_truth.pkl'

    artists = load_data(filename=input_path)

    names = dict()
    heatmaps = dict()
    ground_truth = dict()

    for id_, artist in artists.items():
        names[id_] = artist.id
        heatmaps[id_] = artist.tsne_heatmap
        ground_truth[id_] = artist.similar_artists

    save_data(filename=output_heatmaps, dict=heatmaps)
    save_data(filename=output_names, dict=names)
    save_data(filename=output_gt, dict=ground_truth)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_pkl', '-i', required=True, type=str,
                        help='path to pkl artists dictionary it has to include heatmaps attached')
    parser.add_argument('--output_path', '-o', required=False, type=str, default='.',
                        help='path where output data will be saved')

    args = parser.parse_args()
    main(args)
