import sys
# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/home/crottondi/PIRISI_TESI/TESI_BIS/')
import argparse
from primary.data_io import save_data, load_data
import primary.rbo as rbo
import numpy as np
def compute_ranking_score(artists):
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
            scores.append(rbo.RankingSimilarity(a.similar_artists, a.my_similar_artists).rbo())
            tmp.append([a.id,rbo.RankingSimilarity(a.similar_artists, a.my_similar_artists).rbo()])
    scores = np.array(scores)

    print('The Average score (RBO) for the metric minkowsy is ', np.mean(scores))

def print_rankings(artists, filename):

    f = open(filename, "w+")

    for a in artists.values():

        f.write("%s - %s\n" % (a.id, a.name))
        f.write('Ground truth lenght: %d - ' % (len(a.similar_artists)))
        names = [artists[id_].name for id_ in a.similar_artists]
        f.write(str(names))
        f.write('\n')
        try:
            rbo_score = rbo.RankingSimilarity(a.similar_artists, a.my_similar_artists).rbo()
            names = [artists[id_].name for id_ in a.my_similar_artists]
        except:
            rbo_score = 0
            names = []
        f.write('%s. RBO: %.2f ' % ('minkowsky', rbo_score))
        f.write(str(names))
        f.write('\n')
        f.write('\n')
    f.close()

def main(args):
    input_pkl= args.i_path
    input_ranking = args.i_ranking
    output_path = args.output_path

    if output_path[-1] != '/':
        output_path += '/'

    print('LOADING PKL ARTISTS...', end='')
    artists = load_data(filename=input_pkl)
    print('DONE')

    if input_pkl != 'NO':
        # update artists file
        print('LOADING PKL RANKING...', end='')
        ranking = load_data(filename=input_ranking)
        print('DONE')

        for k, v in ranking.items():
            artists[k].my_similar_artists = v

    ranking_pathname = output_path + 'ranking.txt'
    print_rankings(artists=artists, filename=ranking_pathname)
    compute_ranking_score(artists=artists)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--i_path', '-i', required=True, type=str,
                        help='path to pkl artists dictionary it has to include heatmaps attached')
    parser.add_argument('--i_ranking', '-r', required=False, type=str, default='NO',
                        help='path to pkl merged ranking')
    parser.add_argument('--output_path', '-o', required=True, type=str, default='',
                        help='path where output data will be saved')
    args = parser.parse_args()
    main(args)