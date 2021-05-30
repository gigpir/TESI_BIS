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
import matplotlib.pyplot as plt

def distance_vs_gt_position(ground_truth, distances):
    gt_distances = np.zeros((len(ground_truth), 100))
    gt_distances.fill(np.nan)
    n = 0
    for i, (id_outer, l) in enumerate(ground_truth.items()):
        for j, id_inner in enumerate(l):
            try:
                gt_distances[i, j] = cc_peak_1_dist[id_outer][id_inner]
            except:
                n += 1

    print(n)
    return gt_distances


def print_histograms(gt_distances, folder):
    os.path.dirname(folder)
    path = os.path.join(folder,'HISTOGRAMS')
    if not os.path.exists(path):
        os.makedirs(path)

    for position in range(100):
        try:
            _ = plt.hist(gt_distances[:, position], bins=30)
            plt.xlabel('distance value')
            plt.ylabel('occurrences', fontsize=16)
            title = 'ground truth distances distribution @ position ' + str(position)
            plt.title(title)
            filename = str(position) + '.png'
            path_name = os.path.join(path, filename)
            plt.savefig(path_name, dpi=300)
            plt.close('all')
        except:
            print('Could not print histogram for position '+ str(position))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distances', '-d', required=False, type=str, default='./distances_cc_peak_1.pkl',
                        help='path to pkl distances file')
    parser.add_argument('--ground_truth', '-g', required=False, type=str, default='./ground_truth.pkl',
                        help='path to pkl ground truth file')
    parser.add_argument('--heatmaps', '-hm', required=False, type=str, default='./heatmaps.pkl',
                        help='path to pkl heatmap file')
    parser.add_argument('--names', '-n', required=False, type=str, default='./names.pkl',
                        help='path to pkl heatmap file')
    parser.add_argument('--ranking', '-r', required=False, type=str, default='./.pkl',
                        help='path to pkl heatmap file')
    args = parser.parse_args()

    #names = load_data(filename='/home/gigi/PycharmProjects/TESI_BIS/OUTPUT/names.pkl')
    #heatmaps = load_data(filename='/home/gigi/PycharmProjects/TESI_BIS/OUTPUT/heatmaps.pkl')
    ground_truth = load_data(filename='/home/gigi/PycharmProjects/TESI_BIS/OUTPUT/ground_truth.pkl')
    cc_peak_1_dist = load_data(filename='/home/gigi/PycharmProjects/TESI_BIS/OUTPUT/distances_cc_peak_1.pkl')
    #cc_peak_2_dist = load_data(filename='/home/gigi/PycharmProjects/TESI_BIS/OUTPUT/distances_cc_peak_2.pkl')
    #ranking = load_data(filename='/home/gigi/PycharmProjects/TESI_BIS/OUTPUT/max_length_ranking_cc_peak_1.pkl')

    print_histograms(gt_distances=distance_vs_gt_position(ground_truth=ground_truth,distances=cc_peak_1_dist), folder='/home/gigi/PycharmProjects/TESI_BIS/OUTPUT/')