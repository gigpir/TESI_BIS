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
    intervals = args.resolution
    print('LOADING PKL...')
    artists = load_data(filename=input_folder)

    print('PREPROCESSING ', d[mode])

    X, y = gen_dataset(artists=artists, mode=mode)
    X, y = remove_outlier(X=X, y=y, thresh=threshold)
    X = normalize(X=X)

    for lr in [10,100,500,1000]:
        print('TSNE with learning rate =', lr)
        X_emb = tsne(X,lr=lr)

        print('[TSNE-1 - TSNE-2]')
        print('min values')
        print(np.amin(X_emb, axis=0))
        print('max values')
        print(np.amax(X_emb, axis=0))
        print('mean values')
        print(np.mean(X_emb, axis=0))
        print('variance values')
        print(np.var(X_emb, axis=0))


    #artists = optimize_artists_dictionary(artists)
    #artists = attach_tsne_to_art_dict(artists=artists, X=X, y=y)

    tsne_min = np.amin(X, axis=0)
    tsne_max = np.amax(X, axis=0)





    #artists = clean_similar_artists(artists=artists)

    #save_data(artists, filename=output_filename)



if __name__ == '__main__':
    '''
        search for an optimal learning rate 
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--i_path', '-i', required=True, type=str, help='path to pkl artists dictionary')
    parser.add_argument('--o_path', '-o', required=True, type=str, help='path where output data will be saved')
    parser.add_argument('--threshold', '-t', required=True, type=float, help='threshold value for outlier remotion')
    parser.add_argument('--mode', '-m', required=True, type=int, help='select between '
                                                                      '\n0: mean values\n'
                                                                      '1: 0 + variance\n'
                                                                      '2: 1 + first derivative \n'
                                                                      '3: second derivative')
    parser.add_argument('--resolution', '-r', required=True, type=int, help='the learning rate interval will be divided into n steps')
    args = parser.parse_args()

    main(args)


# TODO
# provare lr 10,100,1000 sul subset 1%
# plot andamento del gradiente e dell'errore
#