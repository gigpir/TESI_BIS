import sys
# insert at 1, 0 is the script path (or '' in REPL)
from primary.heatmap import compute_heatmap_distance
import pandas as pd
sys.path.insert(1, '/home/crottondi/PIRISI_TESI/TESI_BIS/')
import argparse
from primary.data_io import save_data, load_data
import primary.rbo as rbo
import numpy as np







if __name__ == '__main__':

    artists = load_data(filename='/home/gigi/PycharmProjects/TESI_BIS/PKL/artists_subset_hm.pkl')
    max=0
    max_id=''
    for id_, a_ in artists.items():
        if len(artists[id_].song_list) > max:
            max_id = id_
            max = len(artists[id_].song_list)
    print(max_id, max)