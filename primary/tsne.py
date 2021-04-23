import csv

from sklearn.ensemble import IsolationForest

from primary.data_io import load_data
from sklearn.svm import OneClassSVM
from sklearn.manifold import TSNE
import numpy as np
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from primary.utility import resize_matrix, z_normalize, power_transform, normalize, quantile_transform, robust_scaler, \
    generate_color_text_list, gen_colors, str_vect_to_dict
from primary.ellipses_plot_wrapper import confidence_ellipse
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.neighbors import LocalOutlierFactor

def tsne(X, n_comp = 2,lr=400,perp=30):
    X_embedded = TSNE(n_components=n_comp,learning_rate=lr,n_jobs=-1,random_state=49,perplexity=perp).fit_transform(X)# init = 'pca',
    #manifold.TSNE(n_components=n_components, init='pca',random_state=0)
    return X_embedded

def tsne_plot_centroids(centroids,filename='tsne_centroids'):
    fig = plt.figure()
    y = np.array(centroids)[:, 0]
    X = np.array(centroids)[:, 1:6]
    X = X.astype(np.float)
    new_y = []
    ax = fig.add_subplot(1, 1, 1)

    vect = y
    y_dic = str_vect_to_dict(vect)
    for el in vect:
        new_y.append(y_dic[el])
    new_y = np.array(new_y)
    sizes = np.absolute(X[:, 4]) ** (1/2)
    ax.scatter(X[:, 0], X[:, 1], c=new_y, sizes=(sizes*200))
    ax.set_title('TSNE-centroids')
    pad = 20
    #ax.set_xlim([min(X[:, 0])-pad, max(X[:, 0])+pad])
    ax.set_xlim([-50, 60])
    ax.set_xlabel('tsne-1')
    #ax.set_ylim([min(X[:, 1])-pad, max(X[:, 1])+pad])
    ax.set_ylim([-50, 60])
    ax.set_ylabel('tsne-2')
    for i, txt in enumerate(vect):
        ax.annotate(np.array(centroids)[i, 6], (X[i, 0], X[i, 1]))
    fname = './plots/'+filename+'.png'
    plt.savefig(fname, dpi=600)

def tsne_plot_elliptical(X,y,artists,colors,filename='tsne_centroids',note='with outliers'):
    fig, ax_nstd = plt.subplots(figsize=(6, 6))

    out = []  # [<A.ID><A.C1><A.C2><v11><v22><v12>]
    # get unique artist ids
    artist_ids = set(np.array(y)[:, 0])

    # aggregate horizontally X and y to filter
    X_y = np.hstack((X, y))

    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    #colors = generate_color_text_list(len(artist_ids))
    #colors = gen_colors(len(artist_ids))
    for i, id in enumerate(artist_ids):
        # filter only coordinates related to that id
        filtered_x = X_y[np.where(X_y[:, 2] == id)][:, :2].astype(np.float)

        x = filtered_x[:, 0]
        y = filtered_x[:, 1]
        ax_nstd.scatter(x, y, s=0.5, c=colors[i])
        confidence_ellipse(x, y, ax_nstd, n_std=1, label=r'$1\sigma$', edgecolor=colors[i])
        ax_nstd.scatter(np.mean(x), np.mean(y), c=colors[i], s=3)
        ax_nstd.annotate(artists[id].name, (np.mean(x), np.mean(y)),color=colors[i])

    fname = './plots/'+ filename+note+'.png'
    ax_nstd.set_title('TSNE-space centroids '+note)
    # ax.set_xlim([min(X[:, 0]), max(X[:, 0])])
    ax_nstd.set_xlim([-60, 60])
    ax_nstd.set_xlabel('tsne-1')
    # ax.set_ylim([min(X[:, 1]), max(X[:, 1])])
    ax_nstd.set_ylim([-50, 60])
    ax_nstd.set_ylabel('tsne-2')
    #ax_nstd.legend()
    plt.savefig(fname, dpi=600)

def tsne_plot(X,y,n_comp = 2, genre_annot=False,note=''):
    fig = plt.figure()
    y = np.array(y)
    new_y = []

    if genre_annot:
        #each first song genre becomes an integer
        vect = y[:, 2]  # artist.terms[0] of each song
    else:
        # each distinct artist becomes an integer
        vect = y[:, 0]  # artist.id of each song

    y_dic = str_vect_to_dict(vect)
    for el in vect:
        new_y.append(y_dic[el])
    new_y = np.array(new_y)
    #new_y is an ordered array of integers, each integer correspond to an artist/genre

    if n_comp == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X[:, 0], X[:, 1], c=new_y, s=10)
    else:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=new_y, s=10)

    ax.set_title('TSNE-space plot')
    #ax.set_xlim([min(X[:, 0]), max(X[:, 0])])
    ax.set_xlim([-50, 60])
    ax.set_xlabel('tsne-1')
    #ax.set_ylim([min(X[:, 1]), max(X[:, 1])])
    ax.set_ylim([-50, 60])
    ax.set_ylabel('tsne-2')
    if n_comp == 3:
        ax.set_zlim([min(X[:, 2]), max(X[:, 2])])
        ax.set_zlabel('tsne-3')
    if genre_annot:
        for i, txt in enumerate(vect):
            if i % 100 == 0:
                ax.annotate(txt, (X[i, 0], X[i, 1]))
        fname = './plots/tsne_'+str(n_comp)+note+'_genre.png'
    else:
        fname = './plots/tsne_' + str(n_comp) +note+ '.png'

    plt.savefig(fname, dpi=600)

def mean_n_rows(artists):
    # retrieve the mean value of rows of each mfcc matrix
    mean = 0
    n = 0
    min = 9999999
    for a in artists.values():
        for s in a.song_list:
            print(s.segments_timbre.shape)
            mean += s.segments_timbre.shape[0]
            n += 1
            if s.segments_timbre.shape[0] < min:
                min = s.segments_timbre.shape[0]
    return mean

def get_features_dict():
    output = {0: 'mfcc_mean_0',
              1: 'mfcc_mean_1',
              2: 'mfcc_mean_2',
              3: 'mfcc_mean_3',
              4: 'mfcc_mean_4',
              5: 'mfcc_mean_5',
              6: 'mfcc_mean_6',
              7: 'mfcc_mean_7',
              8: 'mfcc_mean_8',
              9: 'mfcc_mean_9',
              10: 'mfcc_mean_10',
              11: 'mfcc_mean_11',
              12: 'pitch_mean_0',
              13: 'pitch_mean_1',
              14: 'pitch_mean_2',
              15: 'pitch_mean_3',
              16: 'pitch_mean_4',
              17: 'pitch_mean_5',
              18: 'pitch_mean_6',
              19: 'pitch_mean_7',
              20: 'pitch_mean_8',
              21: 'pitch_mean_9',
              22: 'pitch_mean_10',
              23: 'pitch_mean_11',

              24: 'tempo',
              25: 'loudness',

              26: 'mfcc_min_0',
              27: 'mfcc_max_0',
              28: 'mfcc_var_0',
              29: 'mfcc_min_1',
              30: 'mfcc_max_1',
              31: 'mfcc_var_1',
              32: 'mfcc_min_2',
              33: 'mfcc_max_2',
              34: 'mfcc_var_2',
              35: 'mfcc_min_3',
              36: 'mfcc_max_3',
              37: 'mfcc_var_3',
              38: 'mfcc_min_4',
              39: 'mfcc_max_4',
              40: 'mfcc_var_4',
              41: 'mfcc_min_5',
              42: 'mfcc_max_5',
              43: 'mfcc_var_5',
              44: 'mfcc_min_6',
              45: 'mfcc_max_6',
              46: 'mfcc_var_6',
              47: 'mfcc_min_7',
              48: 'mfcc_max_7',
              49: 'mfcc_var_7',
              50: 'mfcc_min_8',
              51: 'mfcc_max_8',
              52: 'mfcc_var_8',
              53: 'mfcc_min_9',
              54: 'mfcc_max_9',
              55: 'mfcc_var_9',
              56: 'mfcc_min_10',
              57: 'mfcc_max_10',
              58: 'mfcc_var_10',
              59: 'mfcc_min_11',
              60: 'mfcc_max_11',
              61: 'mfcc_var_11',

              62: 'pitch_min_0',
              63: 'pitch_max_0',
              64: 'pitch_var_0',
              65: 'pitch_min_1',
              66: 'pitch_max_1',
              67: 'pitch_var_1',
              68: 'pitch_min_2',
              69: 'pitch_max_2',
              70: 'pitch_var_2',
              71: 'pitch_min_3',
              72: 'pitch_max_3',
              73: 'pitch_var_3',
              74: 'pitch_min_4',
              75: 'pitch_max_4',
              76: 'pitch_var_4',
              77: 'pitch_min_5',
              78: 'pitch_max_5',
              79: 'pitch_var_5',
              80: 'pitch_min_6',
              81: 'pitch_max_6',
              82: 'pitch_var_6',
              83: 'pitch_min_7',
              84: 'pitch_max_7',
              85: 'pitch_var_7',
              86: 'pitch_min_8',
              87: 'pitch_max_8',
              88: 'pitch_var_8',
              89: 'pitch_min_9',
              90: 'pitch_max_9',
              91: 'pitch_var_9',
              92: 'pitch_min_10',
              93: 'pitch_max_10',
              94: 'pitch_var_10',
              95: 'pitch_min_11',
              96: 'pitch_max_11',
              97: 'pitch_var_11',

              98: 'mfcc_1st_0',
              99: 'mfcc_1st_1',
              100: 'mfcc_1st_2',
              101: 'mfcc_1st_3',
              102: 'mfcc_1st_4',
              103: 'mfcc_1st_5',
              104: 'mfcc_1st_6',
              105: 'mfcc_1st_7',
              106: 'mfcc_1st_8',
              107: 'mfcc_1st_9',
              108: 'mfcc_1st_10',
              109: 'mfcc_1st_11',
              110: 'pitch_1st_0',
              111: 'pitch_1st_1',
              112: 'pitch_1st_2',
              113: 'pitch_1st_3',
              114: 'pitch_1st_4',
              115: 'pitch_1st_5',
              116: 'pitch_1st_6',
              117: 'pitch_1st_7',
              118: 'pitch_1st_8',
              119: 'pitch_1st_9',
              120: 'pitch_1st_10',
              121: 'pitch_1st_11',

              122: 'mfcc_2nd_0',
              123: 'mfcc_2nd_1',
              124: 'mfcc_2nd_2',
              125: 'mfcc_2nd_3',
              126: 'mfcc_2nd_4',
              127: 'mfcc_2nd_5',
              128: 'mfcc_2nd_6',
              129: 'mfcc_2nd_7',
              130: 'mfcc_2nd_8',
              131: 'mfcc_2nd_9',
              132: 'mfcc_2nd_10',
              133: 'mfcc_2nd_11',
              134: 'pitch_2nd_0',
              135: 'pitch_2nd_1',
              136: 'pitch_2nd_2',
              137: 'pitch_2nd_3',
              138: 'pitch_2nd_4',
              139: 'pitch_2nd_5',
              140: 'pitch_2nd_6',
              141: 'pitch_2nd_7',
              142: 'pitch_2nd_8',
              143: 'pitch_2nd_9',
              144: 'pitch_2nd_10',
              145: 'pitch_2nd_11',
              }
    return output

def prepare_dataset(artists,remove_outliers=False, mode=0, local_outlier=True, print_stats=None, print_outlier_percentage_p_feature=False, outlier_trheshold=3.5, drop_feature=None):

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


    rows = 1 #mean_n_rows(artists)
    pbar = tqdm(total=len(artists))

    #data will be in the format [feature_vector, ARTIST_ID, SONG_ID]
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
                #append second derivative
                feat_row = np.append(feat_row, resize_matrix(mfcc_mat, rows, gradient=2))
                feat_row = np.append(feat_row, resize_matrix(pitch_mat, rows, gradient=2))


            X.append(feat_row)
            lab_row = [a.id, s.id]

            #include also the first genre term if present
            if len(a.terms) > 0:
                lab_row.append(a.terms[0])
            else:
                lab_row.append('NULL')
            y.append(lab_row)
        pbar.update()
    pbar.close()
    #X = feature_selection(np.array(X).astype(np.float),np.array(y)[:,0])

    if drop_feature is not None:
        X=np.array(X)
        feat_vector = X[:, drop_feature]
        tmp_lab=(np.array(y))[:,:2]
        table = np.column_stack((tmp_lab, feat_vector))
        df = pd.DataFrame(data=table, columns=['Art_id', 'Song_id', str(drop_feature)])
        filename = 'feature_'+str(drop_feature)+'_values.csv'
        df.to_csv(filename, index=True)
        #X = np.delete(X, drop_feature, 1)
    if remove_outliers:
        #X, y = remove_outliers_lof_general(X, y)
        if local_outlier:
            X, y = remove_outliers_lof(X, y)
            X = robust_scaler(X)
        else:
            if print_stats != None:
                print_feature_stats(np.array(X).astype(np.float), note='before_global_outlier_remotion')
            X, y = remove_outliers_global(np.array(X).astype(np.float), y, print_outlier_percentage_p_feature=print_outlier_percentage_p_feature, outlier_trheshold=outlier_trheshold)
            if print_stats != None:
                print_feature_stats(np.array(X).astype(np.float), note='after_global_outlier_remotion')
            X = robust_scaler(X)
            X, y = remove_outliers_lof(X, y)
        # X = z_normalize(X)
        # X = power_transform(X)
        # X = quantile_transform(X)
    else:
        X = robust_scaler(X)
    return X, y

def print_feature_stats(X, note=''):

    '''
    save a csv file with with a table (146, 4)
        for every feature print mean, min, max, variance
    :param X:
    :param note:
    :return:
    '''

    feat_dict = get_features_dict()

    column_names = ['feature','mean','max','min','var']

    n_features = X.shape[1]

    row_list = []
    for i in range(X.shape[1]):
        dictionary = dict()
        array = X[:,i]
        feat_name = feat_dict[i]
        dictionary['feature'] = feat_name
        dictionary['mean'] = np.mean(array)
        dictionary['max'] = np.max(array)
        dictionary['min'] = np.min(array)
        dictionary['var'] = np.var(array)

        row_list.append(dictionary)

    df = pd.DataFrame(data=row_list, columns=column_names)

    filename = 'feature_stats'
    if note != '':
        filename += '_'+note+'.csv'
    else:
        filename += '.csv'

    df.to_csv(filename, index=True)


def attach_tsne_to_art_dict(artists, X,y):
    """
        each song of each Artist will have its tsne coordinates associated
    """
    print("Attaching tsne coordinates to artist dictionary")
    pbar = tqdm(total=len(y))
    for i, row_label in enumerate(y):
        artists[row_label[0]].song_list[row_label[1]].tsne = [X[i][0],X[i][1]]
        pbar.update()
    pbar.close()
    return artists

def remove_outliers_lof(data, y):
    #Unsupervised Outlier Detection using Local Outlier Factor (LOF)
    out = []  # same format as X_tsne

    # get unique artist ids
    artist_ids = set(np.array(y)[:, 0])

    # aggregate horizontally X and y to filter
    X_y = np.hstack((data, y))

    black_list = []

    for id in artist_ids:
        # filter only coordinates related to that id
        filtered = X_y[np.where(X_y[:, -3] == id)]
        y = filtered[:, -3:]
        X = np.delete(filtered,np.s_[-3:],axis=1).astype(np.float)
        if len(X) > 5:
        #remove outliers only if there are more samples than neighbors
            clf = LocalOutlierFactor(algorithm='auto',metric='euclidean',n_neighbors=5)
            pr = clf.fit_predict(X)
            for i, p in enumerate(pr):
                #if p == -1 we have an outlier
                if p == -1:
                    black_list.append(y[i][1])
    X = []
    y = []
    for r in X_y:
        if r[-2] not in black_list:
            X.append(list(r[:-3]))
            y.append(list(r[-3:]))

    X = np.array(X).astype(np.float)
    y = np.array(y)
    #print("Outlier remotion: (%d - %d)= %d " % (X_y.shape[0], X_y.shape[0]-X.shape[0], X.shape[0]))
    print("Before Outlier remotion: ", X_y.shape)
    print("After Outlier remotion: ", X.shape)

    return X,y


def mad_based_outlier(points, thresh=8):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    if med_abs_deviation != 0:
        modified_z_score = 0.6745 * diff / med_abs_deviation
    else:
        modified_z_score = 0.6745 * diff
        return modified_z_score < 0
    return modified_z_score > thresh

def mean_std_based_outlier(array, thresh=3):
    mi, std = np.mean(array), np.std(array)
    pr = []
    for val in array:
        if val < (mi-thresh*std) or val > (mi+thresh*std):
            pr.append(True)
        else:
            pr.append(False)
    return np.array(pr)

def remove_outliers_global(data, y, print_outlier_percentage_p_feature, outlier_trheshold=3.5, save_histogram=False):
    # Unsupervised Outlier Detection using
    print("Outlier remotion started with threshold ", outlier_trheshold)
    # get unique artist ids
    artist_ids = set(np.array(y)[:, 0])

    # aggregate horizontally X and y to filter
    X_y = np.hstack((data, y))

    black_list = dict()

    if print_outlier_percentage_p_feature:
        table = []
        feat_names = get_features_dict()

    for i in range(data.shape[1]):
        array = data[:, i]

        #pr = mad_based_outlier(array,thresh=outlier_trheshold)
        pr = mean_std_based_outlier(array=array,thresh=outlier_trheshold)

        if print_outlier_percentage_p_feature:
            trues = np.sum(pr)
            table.append([feat_names[i], (trues/len(pr)), trues])

        for i, p in enumerate(pr):

            if p and y[i][1] not in black_list:
                black_list[y[i][1]] = 1
            elif p and y[i][1] in black_list:
                black_list[y[i][1]] += 1

    if save_histogram:
        #istogramma distribuzione numero di feature sballate per brano
        ax = plt.hist(list(black_list.values()), bins=data.shape[1])
        filename = 'outliers_vs_outlier_features.png'
        title = 'Number of outliers vs Number of outlier features'
        plt.title(title)
        plt.savefig(filename)
        plt.close('all')
    X = []
    y = []
    for r in X_y:
        if r[-2] not in black_list:
            X.append(list(r[:-3]))
            y.append(list(r[-3:]))

    X = np.array(X).astype(np.float)
    y = np.array(y)
    # print("Outlier remotion: (%d - %d)= %d " % (X_y.shape[0], X_y.shape[0]-X.shape[0], X.shape[0]))
    print("Before Outlier remotion: ", X_y.shape)
    print("After Outlier remotion: ", X.shape)
    if print_outlier_percentage_p_feature:
        df = pd.DataFrame(table, columns=['feature', 'outlier_ratio','n_outlier'])
        filename = 'outlier_t'+str(outlier_trheshold)+'_percentage_p_feature.csv'
        df.to_csv(filename,index=True)
    return X, y

def remove_outliers_lof_general(data, y):
    # Unsupervised Outlier Detection using Local Outlier Factor (LOF)
    out = []  # same format as X_tsne

    # get unique artist ids
    artist_ids = set(np.array(y)[:, 0])

    # aggregate horizontally X and y to filter
    X_y = np.hstack((data, y))

    black_list = []

    clf = LocalOutlierFactor(algorithm='auto', metric='euclidean', n_neighbors=10)
    pr = clf.fit_predict(data)
    for i, p in enumerate(pr):
        # if p == -1 we have an outlier
        if p == -1:
            black_list.append(y[i][1])
    X = []
    y = []
    for r in X_y:
        if r[-2] not in black_list:
            X.append(list(r[:-3]))
            y.append(list(r[-3:]))

    X = np.array(X).astype(np.float)
    y = np.array(y)
    # print("Outlier remotion: (%d - %d)= %d " % (X_y.shape[0], X_y.shape[0]-X.shape[0], X.shape[0]))
    print("Before Outlier remotion: ", X_y.shape)
    print("After Outlier remotion: ", X.shape)

    return X, y


