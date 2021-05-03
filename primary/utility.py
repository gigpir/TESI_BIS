import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from random import choice
import colorsys
import primary.data_io


def str_vect_to_dict(vect):
    #given a list of string
    #return a dict where the key is a string anc value is an int
    y = dict()
    n = 0
    for str in vect:
        if str not in y:
            y[str] = n
            n += 1
    return y


def decrease_vector_size(vect,final_size):
    # Decrease array size by averaging adjacent values with numpy
    final_vect = []

    in_dim = vect.shape[0]
    window_size = int(in_dim / final_size)

    for i in range(final_size):
        idx_in = i * window_size
        idx_out = idx_in + window_size
        #calculate average of the group
        if i == final_size - 1:
            # last element, pick til the end
            final_vect.append(np.mean(vect[idx_in:]))
        else:
            final_vect.append(np.mean(vect[idx_in:idx_out]))
    return np.array(final_vect)


def increase_vector_size(vect,final_size):
    # Increase array size by duplicating each value in order to reach the target size
    in_dim = vect.shape[0]
    dup_factor = int(final_size/in_dim)

    for i in range(in_dim-1):
        v = np.full((dup_factor), vect[i])
        if i == 0:
            final_vect = v
        else:
            final_vect = np.append(final_vect, v)

    #create a last vector in order to reach the target size
    if final_size-final_vect.shape[0] != 0:
        v = np.full((final_size-final_vect.shape[0]), vect[in_dim-1])
        final_vect = np.append(final_vect, v)
    return final_vect


def resize_matrix(matrix, row, gradient=0, min_max_var=False, only_var = False):
    # reduce or increase the number of rows of matrix in order
    # to achieve dimension specified by row

    if matrix.shape[0]>row:
        resize_vect = decrease_vector_size
    else:
        resize_vect = increase_vector_size

    for i in range(matrix.shape[1]):
        if min_max_var:
            # extract min max and variance of coloumn vector
            if i == 0:
                final = []
            current_v = matrix[:,i]

            min = np.min(current_v)
            max = np.max(current_v)
            var = np.var(current_v)
            final = np.append(final, min)
            final = np.append(final, max)
            final = np.append(final, var)
        elif only_var:
            # extract just variance of coloumn vector
            if i == 0:
                final = []
            current_v = matrix[:,i]
            var = np.var(current_v)
            final = np.append(final, var)
        else:
            #extract and transpose the corresponding coloumn vector
            if gradient == 0:
                grad = matrix[:,i]
            elif gradient == 1:
                grad = np.gradient(matrix[:, i])
            elif gradient == 2:
                grad = np.gradient(matrix[:, i])
                grad = np.gradient(grad)


            new_vect = resize_vect(grad,row)
            if i == 0:
                final = new_vect
            else:
                final = np.append(final,new_vect)
    return final

def z_normalize(data):
    return StandardScaler().fit(data).transform(data)

def power_transform(data):
    pt = PowerTransformer( standardize=False)
    return pt.fit_transform(data)

def normalize(data):
    #Normalization is the process of scaling individual samples to have unit norm.
    # This process can be useful if you plan to use a quadratic form such as the
    # dot-product or any other kernel to quantify the similarity of any pair of samples.
    return preprocessing.normalize(data, norm='l2')

def quantile_transform(data):
    #QuantileTransformer provides a non-parametric
    # transformation to map the data to a uniform distribution with values between 0 and 1
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',random_state=0)
    return quantile_transformer.fit_transform(data)

def robust_scaler(data):
    #Scale features using statistics that are robust to outliers

    return preprocessing.RobustScaler(quantile_range=[15,85]).fit_transform(data)

def random_color_code():
    hex_chars = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    color_code = '#'
    for i in range(0,6):
        color_code += choice(hex_chars)
    return color_code

def generate_color_text_list(n_elements):
    out = []
    for i in range(n_elements):
        out.append(random_color_code())
    return out

def gen_colors(N):

    HSV_tuples = [(x * 1.0 / N, 0.67, 0.84) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

def divide_dict_in_N_parts(artists, n=10, save_to_pkl=True, output_path='./'):

    chunks = np.array_split(list(artists.keys()), n)

    if save_to_pkl:
        for i, chunk in enumerate(chunks):
            filename = output_path + 'chunk_'+str(i)+'.pkl'
            primary.data_io.save_data(chunk, filename=filename)

    return chunks

def optimize_artists_dictionary(artists):

    """
        convert each song list to a dictionary of songs
        this is useful for speeding the search later
    """

    for a in artists.values():
        a.song_list = {a.song_list[i].id: a.song_list[i] for i in range(0, len(a.song_list))}
    return artists

def clean_similar_artists(artists):
    """
        Remove similar artists that are not present in the current
        selection
            Parameters
            ----------
            Output
            --------
            artists : dict of Artist object
    """
    zero_lenght_ranks = 0
    lenghts_array = []
    for a in artists.values():
        new_list = []
        for sim_a in a.similar_artists:
            if sim_a in artists:
                new_list.append(sim_a)
        a.similar_artists = new_list

        #just for statistical purposes
        if len(new_list) == 0:
            zero_lenght_ranks += 1
        else:
            lenghts_array.append(len(new_list))

    lenghts_array = np.array(lenghts_array)
    mean_lenght = np.mean(lenghts_array)

    print('Cleaning phase completed:')
    print(zero_lenght_ranks ,' artists have zero similar artists in their ground truth')
    print('Mean lenght of ranking array is (zero rank were not considered) ', mean_lenght)

    return artists