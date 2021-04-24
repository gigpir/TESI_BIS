import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/crottondi/PIRISI_TESI/TESI_BIS/')
import numpy as np
import argparse
from primary.data_io import retrieve_artist_dict, save_data, load_data
from primary.prep import gen_dataset, remove_outlier, normalize
from primary.tsne import prepare_dataset, tsne,attach_tsne_to_art_dict, remove_outliers_lof, get_features_dict
from primary.utility import optimize_artists_dictionary, clean_similar_artists


import matplotlib.pyplot as plt

PRINT_DISTRIBUTION = False

d = {0:'mean',
     1:'mean + var',
     2:'mean + var + 1st',
     3:'mean + var + 2st'
     }

def main(args):
    input_folder = args.i_path
    threshold = args.threshold
    mode = args.mode

    print('LOADING PKL...')
    artists = load_data(filename=input_folder)



    print('PREPROCESSING ', d[mode])

    X, y = gen_dataset(artists=artists, mode=mode)
    X, y = remove_outlier(X=X, y=y, thresh=threshold)
    X = normalize(X=X)
    print('TSNE')
    X = tsne(X)

    artists = optimize_artists_dictionary(artists)
    artists = attach_tsne_to_art_dict(artists=artists, X=X, y=y)

    tsne_min = np.amin(X, axis=0)
    tsne_max = np.amax(X, axis=0)


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

    #save_data(artists, filename=output_filename)



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
    args = parser.parse_args()

    main(args)

    # TODO
    # istogramma distribuzione numero di feature sballate per brano
    # escludere pitch min e max
    # -----
    # Approccio incrementale (configurazione percentile based thr 3.5)
    # calcorare tsne + Heatmap
    # -> solo medie
    # -> + var
    # -> + derivata 1°
    # -> + derivata 2°
    # -----
    # provare differenti valori di threshold (senza applicare tsne)
