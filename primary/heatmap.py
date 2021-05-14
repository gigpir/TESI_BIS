import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def gen_heatmaps(artists, dimension, max, min):
    """
        given the dictionary of artists
        retrive a tsne heatmap for each one of them
        return the modified dictionary
    """

    print('Generating artist heatmaps')
    pbar = tqdm(total=len(artists))
    for a in artists.values():
        a.tsne_heatmap = np.zeros((dimension,dimension))
        n_outliers = 0
        for s in a.song_list.values():
            if hasattr(s, 'tsne'):
                row_idx = int(((s.tsne[0]+abs(min[0]))/(max[0]+abs(min[0]))) * dimension)
                col_idx = int(((s.tsne[1]+abs(min[1]))/(max[1]+abs(min[1]))) * dimension)
                a.tsne_heatmap[row_idx-1, col_idx-1] += 1
            else:
                n_outliers += 1
        #normalize by number of artists song
        a.tsne_heatmap /= len(a.song_list)-n_outliers
        pbar.update()
    pbar.close()
    return artists

def plot_heatmaps(artists,dimension, min, max):

    range_r = np.zeros((dimension))
    range_c = np.zeros((dimension))
    step_r = (max[0] - min[0]) / dimension
    step_c = (max[1] - min[1]) / dimension
    for i,n in enumerate(range_c):
        left = min[0]+i*step_r
        right = min[0]+(i+1)*step_r
        range_r[i] = (right+left)/2

        left = min[1] + i * step_c
        right = min[1] + (i + 1) * step_c
        range_c[i] = (right + left) / 2

    range_c = [np.format_float_scientific(s, exp_digits=2, precision=1) for s in range_c]
    range_r = [np.format_float_scientific(s, exp_digits=2, precision=1) for s in range_r]

    for a in artists.values():

        fig, ax = plt.subplots()

        im, cbar = heatmap(a.tsne_heatmap, range_r, range_c, ax=ax,
                           cmap="viridis", cbarlabel="songs concentration")

        title = "TSNE Heatmap for "+ a.name
        filename ='./Heatmaps/'+a.id
        ax.set_title(title)
        fig.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close('all')


def compute_heatmap_distance(h1,h2,dimension=20,metric='minkowski_2'):
    """
        Parameters
        ----------
        h1 : 2d array
            The name of the animal
        h2 : 2d array
            The sound the animal makes
        dimension : int
            array dimension
        metric : str
            [minkowski_2, soergel_7, not_intersection_11, kullback-leibler_37]
            see http://www.fisica.edu.uy/~cris/teaching/Cha_pdf_distances_2007.pdf for info

        Output
        ---------
        total_d : float
            the greater total_d is the farther h1 and h2 are
        """
    total_d = 0
    total_div = 0
    for i in range(dimension):
        for j in range(dimension):
            if metric == 'minkowski_2':
                d = abs(h1[i][j]-h2[i][j])
                total_d += d
            if metric == 'soergel_7':
                d = abs(h1[i][j]-h2[i][j])
                div = max(h1[i][j],h2[i][j])
                total_d += d
                total_div += div
            if metric == 'not_intersection_11':
                d = min(h1[i][j],h2[i][j])
                total_d += d
            if metric == 'kullback-leibler_37':
                d = h1[i][j] * np.log(h1[i][j]/h2[i][j])
                total_d += d

    if metric == 'soergel_7':
        total_d /= total_div
    if metric == 'not_intersection_11':
        total_d = 1 - total_d
    return total_d