import sys
# insert at 1, 0 is the script path (or '' in REPL)
from primary.heatmap import compute_heatmap_distance, compute_cross_correlation_distance
import pandas as pd
sys.path.insert(1, '/home/crottondi/PIRISI_TESI/TESI_BIS/')
import argparse
from primary.data_io import save_data, load_data
import primary.rbo as rbo
import numpy as np

heatmap_metric = None
ranking_metric = None
peak_thresh = None

def compute_ranking_score(artists,ranking_metric, heatmap_metric):
    """
        Give a score to a ranking algorithm
                    Parameters
                    ----------
                    artists : dict of Artist object

                    Output
                    ---------
                    score : float
    """
    scores = []
    tmp = []
    print('Computing ranking score my rank and ground truth...')
    for a in artists.values():
        if len(a.similar_artists) > 0 and a.my_similar_artists is not None:
            if ranking_metric=='rbo':
                score = rbo.RankingSimilarity(a.similar_artists, a.my_similar_artists).rbo()
            elif ranking_metric =='intersection':
                score = ranking_intersection_similarity(ranking_gt=a.similar_artists, my_ranking = a.my_similar_artists)
            elif ranking_metric =='minimum_cardinality':
                score = minimum_cardinality_similarity(ranking_gt=a.similar_artists, my_ranking = a.my_similar_artists)

            scores.append(score)

            tmp.append([a.id,rbo.RankingSimilarity(a.similar_artists, a.my_similar_artists).rbo()])
    scores = np.array(scores)

    print('The Average score (', ranking_metric,') for the metric ',heatmap_metric,' is ', np.mean(scores))

def print_rankings(artists, filename,ranking_metric):

    f = open(filename, "w+")

    for a in artists.values():

        f.write("%s - %s\n" % (a.id, a.name))
        f.write('Ground truth lenght: %d - ' % (len(a.similar_artists)))
        names = [artists[id_].name for id_ in a.similar_artists]
        f.write(str(names))
        f.write('\n')
        try:
            if ranking_metric=='rbo':
                score = rbo.RankingSimilarity(a.similar_artists, a.my_similar_artists).rbo()
            elif ranking_metric =='intersection':
                score = ranking_intersection_similarity(ranking_gt=a.similar_artists, my_ranking = a.my_similar_artists)
            elif ranking_metric =='minimum_cardinality':
                score = minimum_cardinality_similarity(ranking_gt=a.similar_artists, my_ranking = a.my_similar_artists)
            names = [artists[id_].name for id_ in a.my_similar_artists]
        except:
            score = 0
            names = []
        f.write('%s. RBO: %.2f ' % ('minkowsky', score))
        f.write(str(names))
        f.write('\n')
        f.write('\n')
    f.close()


def ranking_intersection_similarity(ranking_gt, my_ranking):
    # prendere in considerazione come metrica (alternativamente ad rbo) il
    # rapporto tra la semplice intersezione e la cardinalità del ranking di gt

    intersection = [value for value in ranking_gt if value in my_ranking]
    sim = len(intersection)/len(ranking_gt)
    return sim


def minimum_cardinality_similarity(ranking_gt, my_ranking):
    # prendere in considerazione di calcolare la cardinalità minima
    # della lista predetta necessaria per raggiungere un intersezione pari alla cardinalità di gt
    return 0

def print_rankings_verbose(artists, filename, output_path, heatmap_metric, ranking_metric, peak_thresh):

    f = open(filename, "w+")

    for a in artists.values():

        f.write("%s - %s\n" % (a.id, a.name))
        f.write('Ground truth lenght: %d - ' % (len(a.similar_artists)))
        names = [artists[id_].name for id_ in a.similar_artists]
        f.write(str(names))
        f.write('\n')
        try:
            if ranking_metric == 'rbo':
                score = rbo.RankingSimilarity(a.similar_artists, a.my_similar_artists).rbo()
            elif ranking_metric == 'intersection':
                score = ranking_intersection_similarity(ranking_gt=a.similar_artists, my_ranking=a.my_similar_artists)
            elif ranking_metric == 'minimum_cardinality':
                score = minimum_cardinality_similarity(ranking_gt=a.similar_artists, my_ranking=a.my_similar_artists)

            names = [artists[id_].name for id_ in a.my_similar_artists]
        except:
            score = 0
            names = []
        f.write('%s. %s: %.2f ' % (heatmap_metric, ranking_metric, score))
        f.write(str(names))
        f.write('\n')
        f.write('\n')
    f.close()



    header = ['GT_ID', 'GT_NAME', 'GT_DIST', 'MY_ID', 'MY_NAME', 'MY_DIST']

    for a in artists.values():
        data = []
        n_out = 0
        for i, sim_a in enumerate(a.similar_artists):
            try:
                if heatmap_metric == 'minkowski':
                    dist = compute_heatmap_distance(h1=a.tsne_heatmap, h2=artists[a.similar_artists[i]].tsne_heatmap)
                elif heatmap_metric == 'cc_peak':
                    dist = compute_cross_correlation_distance(h1=a.tsne_heatmap, h2=artists[a.similar_artists[i]].tsne_heatmap, peak_thresh=peak_thresh)

                row = [artists[a.similar_artists[i]].id, artists[a.similar_artists[i]].name, dist, artists[a.my_similar_artists[i]].id, artists[a.my_similar_artists[i]].name, a.my_similar_artists_distances[i]]
                data.append(row)
            except Exception as e:
                #print(a.tsne_heatmap, artists[a.similar_artists[i]].tsne_heatmap)
                n_out += 1


        out_pathname = output_path + a.id + '.xlsx'
        df = pd.DataFrame(data=data, columns=header)
        writer = pd.ExcelWriter(out_pathname, engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Sheet1')

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()





def main(args):
    input_pkl= args.i_path
    input_ranking = args.i_ranking
    output_path = args.output_path

    if output_path[-1] != '/':
        output_path += '/'

    global heatmap_metric
    global ranking_metric
    global peak_thresh
    heatmap_metric = args.heatmap_metric
    ranking_metric = args.ranking_metric
    peak_thresh=args.peak_thresh

    print('LOADING PKL ARTISTS...', end='')
    artists = load_data(filename=input_pkl)
    print('DONE')

    if input_pkl != 'NO':
        # update artists file
        print('LOADING PKL RANKING...', end='')
        ranking = load_data(filename=input_ranking)
        print('DONE')

        for k, v in ranking.items():
            arr = np.array(v)
            artists[k].my_similar_artists = arr[:, 0]
            artists[k].my_similar_artists_distances = arr[:, 1]

    ranking_pathname = output_path + 'ranking.txt'
    print_rankings_verbose(artists=artists, filename=ranking_pathname,output_path=output_path,heatmap_metric=heatmap_metric,
                           ranking_metric=ranking_metric, peak_thresh=peak_thresh)
    compute_ranking_score(artists=artists, ranking_metric=ranking_metric, heatmap_metric=heatmap_metric)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--i_path', '-i', required=True, type=str,
                        help='path to pkl artists dictionary it has to include heatmaps attached')
    parser.add_argument('--i_ranking', '-r', required=False, type=str, default='NO',
                        help='path to pkl merged ranking')
    parser.add_argument('--output_path', '-o', required=True, type=str, default='',
                        help='path where output data will be saved')
    parser.add_argument('--heatmap_metric', '-hm', required=False, type=str, default='minkowski',
                        choices=['minkowski', 'cc_peak',], help='similarity metric when comparing heatmaps')
    parser.add_argument('--ranking_metric', '-rm', required=False, type=str, default='rbo',
                        choices=['rbo','intersection', 'minimum_cardinality', ], help='similarity metric when comparing ranking')
    parser.add_argument('--peak_thresh', '-t', required=False, type=float, default=1.1, help='peak threshold')
    args = parser.parse_args()
    main(args)