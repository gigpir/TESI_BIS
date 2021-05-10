import sys
# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/home/crottondi/PIRISI_TESI/TESI_BIS/')
import pandas as pd
import argparse
from primary.data_io import load_data

def main(args):
    input_filename = args.i_path
    output_folder = args.o_folder
    type = args.type
    if output_folder[-1] != '/':
        output_folder += '/'

    artists = load_data(filename=input_filename)

    if type == 'only_tsne':
        out_patname = output_folder + 'dataset_tsne.xlsx'
        columns = ['SONG_ID', 'SONG_NAME', 'ARTIST_ID', 'ARTIST_NAME', 'TSNE_1', 'TSNE_2', 'SIMILAR_ARTISTS']
        n_out = 0
        data = []
        for a in artists.values():
            for s in a.song_list.values():
                try:
                    row = [s.id, s.name, a.id, a.name, s.tsne[0], s.tsne[1], a.similar_artists]
                    data.append(row)
                except:
                    n_out += 1
        df = pd.DataFrame(data=data, columns=columns)
        writer = pd.ExcelWriter(out_patname, engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name='Sheet1')

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i_path', '-i', required=True, type=str, help='path to pkl artists dictionary')
    parser.add_argument('--o_folder', '-o', required=True, type=str, help='output folder')
    parser.add_argument('--type', '-t', choices=['only_tsne'], help='export mode')
    args = parser.parse_args()

    main(args)
