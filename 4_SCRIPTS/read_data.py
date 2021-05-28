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





def main(args):
    input_path = args.input_pkl

    df = pd.read_excel(io=input_path)



'''
    cc_peak_1 -> peak_thresh=1
    cc_peak_2 calcolare la distanza da shift_0 e normalizzare (dividere) la distanza per il valore del picco'

    -> i file distances_<metrica>.pkl contengono un dizionario la cui chiave è l'id dell'artista
        ogni valore del dizionario è a sua volta un dizionario la cui chiave è l'id dell'artista
        
        esempio: una volta caricato il file è possibile accedere alla distanza tra artista id_1 e id_2 con la seguente sintassi:
            dist = dizionario['id_1']['id_2']    
    
    -> il file gt.pkl contiene un dizionario la cui chiave è l'id dell'artista e il cui valore è una lista di id
    
    -> il file heatmaps.pkl contiene un dizionario la cui chiave è l'id dell'artista e il cui valore è una matrice numpy 20x20
        gli artisti che non possiedono una heatmap hanno esplicitamente il valore None (null)
        
    -> il file names.pkl contiene un dizionario la cui chiave è l'id dell'artista e il cui valore è una stringa col nome dell'artista
    
    
'''






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pkl', '-i', required=True, type=str,
                        help='path to pkl artists dictionary it has to include heatmaps attached')

    args = parser.parse_args()
    main(args)