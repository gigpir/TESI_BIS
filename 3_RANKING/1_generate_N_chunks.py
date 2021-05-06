import multiprocessing
import sys
# insert at 1, 0 is the script path (or '' in REPL)
import time
from functools import partial

from tqdm import tqdm

sys.path.insert(1, '/home/crottondi/PIRISI_TESI/TESI_BIS/')
import numpy as np
import argparse
from primary.data_io import retrieve_artist_dict, save_data, load_data
from primary.prep import gen_dataset, remove_outlier, normalize
from primary.tsne import prepare_dataset, tsne,attach_tsne_to_art_dict, remove_outliers_lof, get_features_dict
from primary.utility import optimize_artists_dictionary, clean_similar_artists, divide_dict_in_N_parts
from primary.heatmap import heatmap

import matplotlib.pyplot as plt

def main(args):
    input_pkl = args.i_path
    n_chunks = args.n_chunks
    output_folder = args.o_path
    if output_folder[-1] != '/':
        output_folder += '/'

    print('LOADING PKL ', input_pkl)

    artists = load_data(filename=input_pkl)
    clean_list = dict()
    #remove from lists those artists that don't have an heatmap
    for k, v in artists.items():
        if v.tsne_heatmap is not None:
            clean_list[k] = k

    divide_dict_in_N_parts(artists=clean_list, n=n_chunks, save_to_pkl=True, output_path=output_folder)

    print(str(len(clean_list)), ' over ', str(len(artists)), ' artists, those with no heatmap were not added to lists')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--i_path', '-i', required=True, type=str,
                        help='path to pkl artists dictionary it has to include heatmaps attached')
    parser.add_argument('--n_chunks', '-n', required=False, type=int, default=1,
                        help='number of chunk lists to create')
    parser.add_argument('--o_path', '-o', required=False, type=str, default=1,
                        help='folder where to save data')
    args = parser.parse_args()
    main(args)