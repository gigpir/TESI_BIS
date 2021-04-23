import numpy as np
import pandas as pd
from tqdm import tqdm

from primary.tsne import print_feature_stats, remove_outliers_global, remove_outliers_lof, get_features_dict
from primary.utility import resize_matrix, robust_scaler


def gen_dataset(artists,mode=3):
    """
                Extract information from data and build a dataset of the type X, y
                The features are standartized

                Parameters
                ----------
                artists : dict(Artist_id, Artist_obj)

                remove_outliers : Bool
                    Perform outlier remotion or not {default = False}
                mode : int
                    select what information to include in X {default = 0}

                    0 : mfcc mean values (12), picht mean values (12),

                    1 : mfcc mean values (12), pitch mean values (12)
                        mfcc min values (12), pitch min values (12),
                        mfcc max values (12), pitch max values (12),
                        mfcc var values (12), pitch var values (12),
                        s.tempo, s.loudness

                    2 : mfcc mean values (12), pitch mean values (12)
                        mfcc min values (12), pitch min values (12),
                        mfcc max values (12), pitch max values (12),
                        mfcc var values (12), pitch var values (12),
                        mfcc 1st_grad values (12), pitch 1st_grad values (12),
                        s.tempo, s.loudness

                    3 : mfcc mean values (12), pitch mean values (12)
                        mfcc min values (12), pitch min values (12),
                        mfcc max values (12), pitch max values (12),
                        mfcc var values (12), pitch var values (12),
                        mfcc 1st_grad values (12), pitch 1st_grad values (12),
                        mfcc 1nd_grad values (12), pitch 2nd_grad values (12),
                        s.tempo, s.loudness

                Output
                ---------
                X , y

        """

    rows = 1  # mean_n_rows(artists)
    pbar = tqdm(total=len(artists))

    # data will be in the format [feature_vector, ARTIST_ID, SONG_ID]
    X = []
    y = []
    for a in artists.values():
        for s in a.song_list:
            mfcc_mat = s.segments_timbre
            pitch_mat = s.segments_pitches

            if mode >= 0:
                feat_row = np.append(resize_matrix(mfcc_mat, rows), resize_matrix(pitch_mat, rows))
                feat_row = np.append(feat_row, [s.tempo, s.loudness])
            if mode >= 1:
                # append min, max, variance of each coloumn
                feat_row = np.append(feat_row, resize_matrix(mfcc_mat, rows, min_max_var=True))
                feat_row = np.append(feat_row, resize_matrix(pitch_mat, rows, min_max_var=True))
            if mode >= 2:
                # append first  derivative
                feat_row = np.append(feat_row, resize_matrix(mfcc_mat, rows, gradient=1))
                feat_row = np.append(feat_row, resize_matrix(pitch_mat, rows, gradient=1))
            if mode >= 3:
                # append second derivative
                feat_row = np.append(feat_row, resize_matrix(mfcc_mat, rows, gradient=2))
                feat_row = np.append(feat_row, resize_matrix(pitch_mat, rows, gradient=2))

            X.append(feat_row)
            lab_row = [a.id, s.id]

            # include also the first genre term if present
            if len(a.terms) > 0:
                lab_row.append(a.terms[0])
            else:
                lab_row.append('NULL')
            y.append(lab_row)
        pbar.update()
    pbar.close()

    return X,y

def remove_outlier(X,y,thresh,verbose=False,save_histogram=False):
    if verbose:
        print_feature_stats(np.array(X).astype(np.float), note='before_global_outlier_remotion')

    X, y = remove_outliers_global(np.array(X).astype(np.float), y,
                                  print_outlier_percentage_p_feature=verbose,
                                  outlier_trheshold=thresh,
                                  save_histogram=save_histogram)
    if verbose:
        print_feature_stats(np.array(X).astype(np.float), note='after_global_outlier_remotion')
    #X, y = remove_outliers_lof(X, y)
    return X, y

def normalize(X):
    X = robust_scaler(X)
    return X

def save_dataset_csv(X, y, output_path):

    X_file = output_path + '_X.csv'
    y_file = output_path + '_y.csv'
    np.savetxt(X_file, np.array(X), delimiter=",")
    # convert array into dataframe
    DF = pd.DataFrame(y)
    #TODO
    # fix this function

    # save the dataframe as a csv file
    #DF.to_csv(y,index=False)
    return



