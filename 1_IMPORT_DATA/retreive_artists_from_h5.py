from primary.data_io import retrieve_artist_dict, save_data

import argparse


def main(args):
    input_folder = args.i_path

    if args.o_path[-1] == '/':
        output_filename = args.o_path + args.o_name
    else:
        output_filename = args.o_path + '/' + args.o_name

    artists = retrieve_artist_dict(basedir=input_folder)

    save_data(dict=artists, filename=output_filename)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i_path', required=True, type=str, help='path to data directory of MSD')
    parser.add_argument('--o_path', required=True, type=str, help='path where MSD pkl will be saved')
    parser.add_argument('--o_name', required=True, type=str, help='output filename')
    args = parser.parse_args()

    main(args)
