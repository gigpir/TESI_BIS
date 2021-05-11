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
def gen_heatmaps_slave(dimension, min, max, list_ids):
    result = dict()
    global artists
    # min = np.array([-0.00012, -0.00012])
    # max = np.array([ 0.0000785, 0.00013])

    # TMP
    # min = np.array([-70, -70])
    # max = np.array([ 70, 70])

    for id_ in list_ids:

        if len(artists[id_].song_list.values()) != 0:
            #the artist has actually songs associated
            result[id_] = np.zeros((dimension, dimension))
            n_outliers = 0
            for s in artists[id_].song_list.values():
                try:
                    row_idx = int(((s.tsne[0] + abs(min[0])) / (max[0] + abs(min[0]))) * dimension)
                    col_idx = int(((s.tsne[1] + abs(min[1])) / (max[1] + abs(min[1]))) * dimension)
                    result[id_][row_idx, col_idx] += 1
                except:
                    n_outliers += 1
            # normalize by number of artists song
            try:
                if len(artists[id_].song_list) - n_outliers != 0:
                    result[id_] /= len(artists[id_].song_list) - n_outliers
                else:
                    #empty heatmap
                    result[id_] = None
            except:
                print(id_, ' has ', len(artists[id_].song_list), ' song with ', n_outliers,
                      ' outliers with respect to range selected')
        else:
            result[id_] = None
    return result

def merge_heatmaps(artists, result):
    for r in result:
        for id_, hm in r.items():
            artists[id_].tsne_heatmap = hm
    return artists

def gen_heatmaps_master(dimension, min, max):
    global artists
    start = time.time()
    func = partial(gen_heatmaps_slave, dimension, min, max)
    pbar = tqdm(len(artists))

    nproc = multiprocessing.cpu_count()
    artists_ids = list(artists.keys())

    split = np.array_split(artists_ids, nproc)

    with multiprocessing.Pool(nproc) as p:
        result = p.map(func, split)

    artists = merge_heatmaps(artists=artists, result=result)

    print(time.time() - start)
    return artists

def plot_heatmaps_slave(dimension,min, max, list_ids):
    global artists
    global output_path
    range_r = np.zeros((dimension))
    range_c = np.zeros((dimension))
    step_r = (max[0] - min[0]) / dimension
    step_c = (max[1] - min[1]) / dimension
    for i,n in enumerate(range_c):
        left = min[0]+i*step_r
        right = min[0]+(i+1)*step_r
        range_r[i] = (right+left)/2

        left = min[1] + i * step_c
        right = min[1] + (i + 1) * step_c
        range_c[i] = (right + left) / 2

    range_c = [np.format_float_scientific(s, exp_digits=2, precision=1) for s in range_c]
    range_r = [np.format_float_scientific(s, exp_digits=2, precision=1) for s in range_r]

    for id_ in list_ids:
        a = artists[id_]
        if a.tsne_heatmap is not None:
            fig, ax = plt.subplots()

            im, cbar = heatmap(a.tsne_heatmap, range_r, range_c, ax=ax,
                               cmap="viridis", cbarlabel="songs concentration")

            try:
                title = "TSNE Heatmap for " + a.name
                filename = output_path + a.id
                ax.set_title(title)
                fig.tight_layout()
            except:
                title = "TSNE Heatmap for " + a.id
                filename = output_path + a.id
                ax.set_title(title)
                fig.tight_layout()

            plt.savefig(filename, dpi=300)
            plt.close('all')

def plot_heatmaps_master(dimension ,min, max):
    global artists
    start = time.time()
    func = partial(plot_heatmaps_slave, dimension ,min, max)
    pbar = tqdm(len(artists))

    nproc = multiprocessing.cpu_count()
    artists_ids = list(artists.keys())

    split = np.array_split(artists_ids, nproc)

    with multiprocessing.Pool(nproc) as p:
        result = p.map(func, split)

    print(time.time() - start)




def main(args):
    input_folder = args.i_path
    threshold = args.threshold
    output_pkl = args.output_pkl
    global output_path
    output_path = args.o_path
    if output_path[-1] != '/':
        output_path += '/'
    mode = args.mode
    global artists
    print('LOADING PKL...')
    artists = load_data(filename=input_folder)


    print('PREPROCESSING ', d[mode])
    X, y = gen_dataset(artists=artists, mode=mode)
    X, y = remove_outlier(X=X, y=y, thresh=threshold)
    X = normalize(X=X)
    print('TSNE')
    X = tsne(X=X, lr=1000)
    artists = optimize_artists_dictionary(artists)
    artists = attach_tsne_to_art_dict(artists=artists, X=X, y=y)
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    dimension = 20
    print('[TSNE-1 - TSNE-2]')
    print('min values')
    print(np.amin(X, axis=0))
    print('max values')
    print(np.amax(X, axis=0))
    print('mean values')
    print(np.mean(X, axis=0))
    print('variance values')
    print(np.var(X, axis=0))
    artists = clean_similar_artists(artists=artists)
    print('GENERATE HEATMAPS')
    gen_heatmaps_master(dimension=dimension, min=min, max=max)
    print('SAVING DATA')
    save_data(artists, filename=output_pkl)


    print('PLOT HEATMAPS in ', output_path)
    plot_heatmaps_master(dimension=dimension, min=min, max=max)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i_path', '-i', required=True, type=str, help='path to pkl artists dictionary')
    parser.add_argument('--o_path', '-o', required=True, type=str, help='path where output data will be saved')
    parser.add_argument('--threshold', '-t', required=True, type=float, help='threshold value for outlier remotion')
    parser.add_argument('--mode', '-m', required=True, type=int, help='select between '
                                                                      '\n0: mean values\n'
                                                                      '1: 0 + variance\n'
                                                                      '2: 1 + first derivative \n'
                                                                    '3: second derivative')
    parser.add_argument('--output_pkl', '-O', required=False, type=str,default='', help='path where output data will be saved')
    args = parser.parse_args()

    main(args)

