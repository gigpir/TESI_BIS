from primary.data_io import retrieve_artist_dict, save_data, load_data
from primary.tsne import prepare_dataset, tsne, attach_tsne_to_art_dict, get_features_dict
from primary.utility import optimize_artists_dictionary, clean_similar_artists
import argparse
import numpy as np
import pandas as pd

def main(args):
    input_folder = args.i_path

    if args.o_path[-1] == '/':
        output_filename = args.o_path + args.o_name
    else:
        output_filename = args.o_path + '/' + args.o_name

    artists = load_data(filename=input_folder)
    print('PREPROCESSING')
    X , y = prepare_dataset(artists=artists, remove_outliers=True, mode=3, local_outlier=False,print_stats=args.stats)


    X = tsne(X)

    artists = optimize_artists_dictionary(artists)
    artists = attach_tsne_to_art_dict(artists=artists, X=X, y=y)

    tsne_min = np.amin(X, axis=0)
    tsne_max = np.amax(X, axis=0)

    if args.verbosity > 0:
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

    save_data(artists, filename=output_filename)



    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i_path', '-i', required=True, type=str, help='path to pkl artists dictionary')
    parser.add_argument('--o_path', '-O', required=True, type=str, help='path where output pkl will be saved')
    parser.add_argument('--o_name', '-o', required=True, type=str, help='output filename')
    parser.add_argument('--heatmaps', '-H', default=None, type=str, help='save heatmaps of each artist in the desired folder')
    parser.add_argument("-v", "--verbosity", default=0 ,action="count", help="increase output verbosity")
    parser.add_argument('--stats', '-s', dest='stats', action='store_true')
    args = parser.parse_args()

    main(args)
