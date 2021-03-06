import glob
import multiprocessing
import os
import pickle
import time
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from functools import partial
import primary.h5py_getters_msd as getters
from primary.data_class import Artist, Song
N_min = 300


def file_to_obj(h5, reduced=True):
    if not reduced:
        ar = Artist(getters.get_artist_name(h5), getters.get_artist_id(h5))
        ar.terms = getters.get_artist_terms(h5)
        ar.terms_freq = getters.get_artist_terms_freq(h5)
        ar.terms_weight = getters.get_artist_terms_weight(h5)
        ar.similar_artists = getters.get_similar_artists(h5)
        song = Song(getters.get_song_id(h5), getters.get_song_title(h5), getters.get_song_hotttnesss(h5))
        song.bars_start = getters.get_bars_start(h5)
        song.bars_confidence = getters.get_bars_confidence(h5)
        song.beats_start = getters.get_beats_start(h5)
        song.beats_confidence = getters.get_beats_confidence(h5)
        song.danceability = getters.get_danceability(h5)
        song.duration = getters.get_duration(h5)
        song.end_of_fade_in = getters.get_end_of_fade_in(h5)
        song.energy = getters.get_energy(h5)
        song.key = getters.get_key(h5)
        song.loudness = getters.get_loudness(h5)
        song.mode = getters.get_mode(h5)
        song.mode_confidence = getters.get_mode_confidence(h5)
        song.start_of_fade_out = getters.get_start_of_fade_out(h5)
        song.tempo = getters.get_tempo(h5)
        song.time_signature = getters.get_time_signature(h5)
        song.time_signature_confidence = getters.get_time_signature_confidence(h5)
        song.track_id = getters.get_track_id(h5)
        song.segments_start = getters.get_segments_start(h5)
        song.segments_confidence = getters.get_segments_confidence(h5)
        song.segments_pitches = getters.get_segments_pitches(h5)
        song.segments_timbre = getters.get_segments_timbre(h5)
        song.segments_loudness_max = getters.get_segments_loudness_max(h5)
        song.segments_loudness_max_time = getters.get_segments_loudness_max_time(h5)
        song.segments_loudness_start = getters.get_segments_loudness_start(h5)
        song.sections_start = getters.get_sections_start(h5)
        song.sections_confidence = getters.get_sections_confidence(h5)
        song.tatums_start = getters.get_tatums_start(h5)
        song.tatums_confidence = getters.get_tatums_confidence(h5)
        ar.song_list.append(song)
    else:
        ar = Artist(getters.get_artist_name(h5), getters.get_artist_id(h5))
        ar.terms = getters.get_artist_terms(h5)
        ar.terms_freq = getters.get_artist_terms_freq(h5)
        ar.terms_weight = getters.get_artist_terms_weight(h5)
        ar.similar_artists = getters.get_similar_artists(h5)
        song = Song(getters.get_song_id(h5), getters.get_song_title(h5), getters.get_song_hotttnesss(h5))
        song.loudness = getters.get_loudness(h5)
        song.tempo = getters.get_tempo(h5)
        song.segments_pitches = getters.get_segments_pitches(h5)
        song.segments_timbre = getters.get_segments_timbre(h5)
        ar.song_list.append(song)
    return ar


def save_data(dict, filename="data_subset.pkl"):
    with open(filename, "wb") as a_file:
        pickle.dump(dict, a_file, pickle.HIGHEST_PROTOCOL)
    return True


def load_data(filename='data_subset.pkl'):
    with open(filename, "rb") as a_file:
        output = pickle.load(a_file)
    return output


def retrieve_artist_dict(basedir='./millionsongsubset_full/MillionSongSubset/data/', ext='.h5', filter=None):
    artist_dict = dict()
    start = time.time()
    func = partial(worker, filter)
    pbar = tqdm(total=10000)

    nproc = multiprocessing.cpu_count()
    files_tot = []
    #collect all filenames
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + ext))

        if len(files) > 1:
            files_tot.extend(files)
            if len(files_tot) > N_min:
                #launch n_proc
                split = np.array_split(files_tot, nproc)
                result = []
                with multiprocessing.Pool(nproc) as p:
                    result = p.map(func, split)
                #merge all dicts into one
                result.insert(0, artist_dict)
                artist_dict = merge_dictionaries(result)
                pbar.update(len(files_tot))
                files_tot = []
        if len(files) == 1:
            files_tot.append(files[0])
    result = []
    result.append(worker(filter, files_tot))
    # merge all dicts into one
    result.insert(0, artist_dict)
    artist_dict = merge_dictionaries(result)
    pbar.update(len(files_tot))

    pbar.close()
    print(time.time()-start)
    return artist_dict





def merge_dictionaries(dicts):
    final = dicts[0]
    for i, d in enumerate(dicts):
        if i > 0:
            for a in d.values():
                if a.id not in final.keys():
                    final[a.id] = a
                else:
                    final[a.id].song_list.extend(a.song_list)

    return final


def worker(filter, files):
    d = dict()

    if not isinstance(files, str):
        #files is a list of filenames
        for f in files:
            #read info from file
            art = None
            try:
                h5 = getters.open_read(f)
                if filter is None:
                    art = file_to_obj(h5)
                else:
                    if getters.get_artist_id(h5) in filter:
                        art = file_to_obj(h5)
                    else:
                        art = None
                h5.close()
            except:
                print("Cannot open file %s" % (f))


            if art is not None:
                if art.id not in d.keys():
                    d[art.id] = art
                else:
                    d[art.id].song_list.append(art.song_list[0])
    else:
        #files is a single filename
        art = None
        try:
            h5 = getters.open_read(files)
            if filter is None:
                art = file_to_obj(h5)
            else:
                if getters.get_artist_id(h5) in filter:
                    art = file_to_obj(h5)
                else:
                    art = None
            h5.close()
        except:
            print("Cannot open file %s" % (files))

        if art is not None:
            if art.id not in d.keys():
                d[art.id] = art
            else:
                d[art.id].song_list.append(art.song_list[0])
    return d

def merge_dictionaries_exp(dicts):
    final = dicts[0]
    for i, d in enumerate(dicts):
        if i > 0:
            for key, value in d.items():
                if key not in final.keys():
                    final[key] = value
                else:
                    final[key] += value

    return final

def worker_exp(files):
    d = dict()

    if not isinstance(files, str):
        #files is a list of filenames
        for f in files:
            #read info from file
            artist_id = None
            try:
                h5 = getters.open_read(f)
                artist_id = getters.get_artist_id(h5)
                h5.close()
            except:
                print("Cannot open file %s" % (f))


            if artist_id is not None:
                if artist_id not in d.keys():
                    d[artist_id] = 1
                else:
                    d[artist_id] += 1
    else:
        #files is a single filename
        artist_id = None
        try:
            h5 = getters.open_read(files)
            artist_id = getters.get_artist_id(h5)
            h5.close()
        except:
            print("Cannot open file %s" % (files))

        if artist_id is not None:
            if artist_id not in d.keys():
                d[artist_id] = 1
            else:
                d[artist_id] += 1
    return d

def traverse_exp(basedir='./millionsongsubset_full/MillionSongSubset/data/', ext='.h5'):
    # traverse for all file .h5
    # return a dictionary <Artist.ID><Number of song>
    # low memory function
    artist_dict = dict()
    start = time.time()

    pbar = tqdm(total=1000000)

    nproc = multiprocessing.cpu_count()
    files_tot = []
    # collect all filenames
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + ext))

        if len(files) > 1:
            files_tot.extend(files)
            if len(files_tot) > N_min:
                # launch n_proc
                split = np.array_split(files_tot, nproc)
                result = []
                with multiprocessing.Pool(nproc) as p:
                    result = p.map(worker_exp, split)
                # merge all dicts into one
                result.insert(0, artist_dict)
                artist_dict = merge_dictionaries_exp(result)
                pbar.update(len(files_tot))
                files_tot = []
        if len(files) == 1:
            files_tot.append(files[0])
    result = []
    result.append(worker_exp(files_tot))
    # merge all dicts into one
    result.insert(0, artist_dict)
    artist_dict = merge_dictionaries_exp(result)
    pbar.update(len(files_tot))

    pbar.close()
    print(time.time() - start)
    return artist_dict

def create_filter(pkl_dict_filename= 'dict_art_nsongs.pkl', max_songs=10000):
    exp_data = load_data(filename=pkl_dict_filename)

    list_id_sngs = []
    for key, value in exp_data.items():
        list_id_sngs.append([key, value])

    #sort by songlist lenght
    list_id_sngs = sorted(list_id_sngs,key=lambda x:x[1],reverse=True)

    new_dict = dict()
    n=0
    #create a dictionary with max_songs keys
    for id_sng in list_id_sngs:
        if n+id_sng[1] >= max_songs:
            break
        else:
            new_dict[id_sng[0]] = id_sng[1]
            n += id_sng[1]


    return new_dict