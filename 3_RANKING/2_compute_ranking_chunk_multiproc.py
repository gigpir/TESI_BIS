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
from primary.heatmap import compute_heatmap_distance, compute_cross_correlation_distance
from operator import itemgetter

CENTROID = False
output_path=None

chunk = None
artists = None

metric = None
peak_thresh = None

def average_linkage_distance(a1,a2):
    out = 0
    a1_outlier=0
    a2_outlier=0
    for s1 in a1.song_list.values():
        try:
            tmp = s1.tsne[1]
        except:
            a1_outlier += 1
            continue

        for s2 in a2.song_list.values():
            try:
                d = abs(s1.tsne[0]-s2.tsne[0])+abs(s1.tsne[1]-s2.tsne[1])
            except:
                a2_outlier += 1
                continue
            out += d
    try:
        out /= (len(a1.song_list)*len(a2.song_list))
    except:
        print('ARTIST WITH ZERO SONGS ALERT')

    return out


def compute_ranking_slave(dimension, top_k, list_ids):
    global artists
    global peak_thresh
    rankings = dict()
    for outer_id in list_ids:
        if len(artists[outer_id].similar_artists) > 0:
            distances = []
            for inner_a in artists.values():
                if inner_a.id != outer_id and inner_a.tsne_heatmap is not None:
                    if metric == 'minkowski':
                        d = compute_heatmap_distance(h1=artists[outer_id].tsne_heatmap, h2=inner_a.tsne_heatmap,
                                         dimension=dimension)
                    elif metric == 'cc_peak':
                        d = compute_cross_correlation_distance(h1=artists[outer_id].tsne_heatmap, h2=inner_a.tsne_heatmap,peak_thresh=peak_thresh ,dimension=dimension)
                    distances.append([inner_a.id, d])
            a=0
            try:
                #distances = heapq.nsmallest(4, distances, key=itemgetter(1))
                distances.sort(key=itemgetter(1), reverse=False)
                top_k = len(artists[outer_id].similar_artists)
                distances = distances[:top_k]
                ranking = [row[0] for row in distances]
                rankings[outer_id] = ranking
            except:
                print(str(distances))
    return rankings



def compute_ranking_master(dimension=20, top_k=30):
    global chunk

    start = time.time()
    func = partial(compute_ranking_slave, dimension, top_k)

    nproc = multiprocessing.cpu_count()
    artists_ids = list(chunk)

    split = np.array_split(artists_ids, nproc)

    with multiprocessing.Pool(nproc) as p:
        result = p.map(func, split)


    chunk_level_ranking = merge_proc_rankings(result=result)

    print(time.time() - start)
    return chunk_level_ranking

def merge_proc_rankings(result):
    chunk_level_ranking=dict()
    for r in result:
        for id_, rank_ in r.items():
            chunk_level_ranking[id_] = rank_
    return chunk_level_ranking

def main(args):
    artists_filename = args.i_path
    chunk_filename = args.i_chunk
    global output_path
    output_path = args.output_path
    if output_path[-1] != '/':
        output_path += '/'

    global metric
    metric = args.metric
    global peak_thresh
    peak_thresh = args.peak_thresh



    global artists
    print('LOADING PKL...', end='')
    artists = load_data(filename=artists_filename)
    print('DONE')

    global chunk
    print('LOADING CHUNK...')
    chunk = load_data(filename=chunk_filename)
    print('DONE')



    print('COMPUTE RANKING of selection ', chunk_filename)
    chunk_level_ranking = compute_ranking_master()
    output_filename= os.path.basename(chunk_filename)
    output_filename += '_OUT.pkl'
    output_path += output_filename

    save_data(chunk_level_ranking, filename=output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i_path', '-i', required=True, type=str, help='path to pkl artists dictionary it has to include heatmaps attached')
    parser.add_argument('--i_chunk', '-ic', required=True, type=str, help='path to pkl chunk where a list of ids is saved')
    parser.add_argument('--output_path', '-o', required=False, type=str,default='', help='path where output data will be saved')
    parser.add_argument('--metric', '-m', required=False, type=str, default='minkowski', choices=['minkowski', 'cc_peak', 'intersection_1', 'instersection_2'], help='metric type')
    parser.add_argument('--peak_thresh', '-t', required=False, type=float, default=1.1, help='peak threshold')

    args = parser.parse_args()

    main(args)

