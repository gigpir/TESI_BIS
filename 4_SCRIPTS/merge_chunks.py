import sys
# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/home/crottondi/PIRISI_TESI/TESI_BIS/')
import argparse
from primary.data_io import save_data, load_data

def getCurrentMemoryUsage():
    ''' Memory usage in kB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

    return int(memusage.strip())


def main(args):
    n_chunks = args.n_chunks
    chunk_folder = args.chunk_folder
    if chunk_folder[-1] != '/':
        chunk_folder += '/'

    #group all chunk level ranking in a single ranking file
    dictionary = dict()
    for i in range(n_chunks):
        chunk_filename = 'chunk_' + str(i) + '_OUT.pkl'
        chunk_pathname = chunk_folder+chunk_filename
        chunk_out = load_data(filename=chunk_pathname)

        for k,v in chunk_out.items():
            dictionary[k]=v
        del chunk_out

        print('chunk ', str(i), 'Memory (GB) : ', getCurrentMemoryUsage()/(2**20))
    final_pathname = chunk_folder+'merged_OUT.pkl'

    save_data(dict=dictionary, filename=final_pathname)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_chunks', '-n', required=False, type=int, default=1,
                        help='number of chunk lists to create')
    parser.add_argument('--chunk_folder', '-c', required=False, type=str, default=1,
                        help='folder where _OUT chunks are')

    args = parser.parse_args()
    main(args)