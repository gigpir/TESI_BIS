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
from primary.utility import optimize_artists_dictionary, clean_similar_artists
from primary.heatmap import heatmap

import matplotlib.pyplot as plt

output_path=None

d = {0:'mean',
     1:'mean + var',
     2:'mean + var + 1st',
     3:'mean + var + 1st +2st'
     }

artists = None
dimension = None
min = None
max = None

def main(args):
    input_folder = args.i_path
    global output_path
    output_path = args.o_path
    if output_path[-1] != '/':
        output_path += '/'
    global artists
    print('LOADING PKL...')
    artists = load_data(filename=input_folder)



    print('PREPROCESSING')

    X, y = gen_dataset(artists=artists, mode=3)

    for t in np.arange(1,3,0.2):
        A ,b = remove_outlier(X=X, y=y, thresh=t)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i_path', '-i', required=True, type=str, help='path to pkl artists dictionary')
    parser.add_argument('--o_path', '-o', required=True, type=str, help='path where output data will be saved')
    #parser.add_argument('--threshold', '-t', required=True, type=float, help='threshold value for outlier remotion')

    args = parser.parse_args()

    main(args)

