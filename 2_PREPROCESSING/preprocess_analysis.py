import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/crottondi/PIRISI_TESI/TESI_BIS/')
import numpy as np
import argparse
from primary.data_io import retrieve_artist_dict, save_data, load_data
from primary.prep import gen_dataset, remove_outlier, normalize
from primary.tsne import prepare_dataset, tsne,attach_tsne_to_art_dict, remove_outliers_lof, get_features_dict
from primary.utility import optimize_artists_dictionary

import matplotlib.pyplot as plt

PRINT_DISTRIBUTION = False


def main(args):
    input_folder = args.i_path
    threshold = args.threshold
    artists = load_data(filename=input_folder)

    X, y = gen_dataset(artists=artists)

    if PRINT_DISTRIBUTION:
        #PRINT VALUES BEFORE OUTLIER REMOTION
        feat_names = get_features_dict()
        x = np.array(X)
        for i in range(x.shape[1]):
            ax = plt.hist(x[:,i], bins=200)
            filename = args.o_path + '/BEFORE/' +feat_names[i]+'.png'
            title = feat_names[i] +'BEFORE outlier remotion'
            plt.title(title)
            plt.savefig(filename)
            plt.close('all')

    X, y = remove_outlier(X=X, y=y, thresh=threshold,verbose=False, save_histogram=True)
    #X = normalize(X=X)
    #X, y = remove_outliers_lof(data=X, y=y)

    if PRINT_DISTRIBUTION:
        # PRINT VALUES AFTER OUTLIER REMOTION
        x = np.array(X)
        for i in range(x.shape[1]):
            ax = plt.hist(x[:, i], bins=200)
            filename = args.o_path + '/AFTER/' + feat_names[i] + '.png'
            title = feat_names[i] + 'AFTER outlier remotion'
            plt.title(title)
            plt.savefig(filename)
            plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i_path', '-i', required=True, type=str, help='path to pkl artists dictionary')
    parser.add_argument('--o_path', '-o', required=True, type=str, help='path where output pkl will be saved')
    parser.add_argument('--threshold', '-t', required=True, type=float, help='threshold value for outlier remotion')

    args = parser.parse_args()

    main(args)


    #TODO
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