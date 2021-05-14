import sys
# insert at 1, 0 is the script path (or '' in REPL)
from primary.heatmap import compute_heatmap_distance
import pandas as pd
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

def print_rankings_verbose(artists, filename, output_path):



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



    header = ['GT_ID', 'GT_NAME', 'GT_DIST', 'MY_ID', 'MY_NAME', 'MY_DIST']

    for a in artists.values():
        data = []
        n_out = 0
        for i, sim_a in enumerate(a.similar_artists):
            try:
                dist = compute_heatmap_distance(h1=a.tsne_heatmap, h2=artists[a.similar_artists[i]].tsne_heatmap)
                row = [artists[a.similar_artists[i]].id, artists[a.similar_artists[i]].name, dist, artists[a.my_similar_artists[i]].id, artists[a.my_similar_artists[i]].name, a.my_similar_artists_distances[i]]
                data.append(row)
            except:
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
    print_rankings_verbose(artists=artists, filename=ranking_pathname,output_path=output_path)
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