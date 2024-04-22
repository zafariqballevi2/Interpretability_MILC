import numpy as np
import pylab as plt


DIR = './FNC_matrices/'
FBIRN05 = np.load(DIR+'FBIRN_fnc.npy')

np.fill_diagonal(FBIRN05, 0)


def fnc_plot(M, title, clim=(-1,1)):
    plt.imshow(M, interpolation=None, clim=clim,
               cmap=plt.cm.seismic)
    ax = plt.gca()
    plt.title(title)
    plt.colorbar()
    # Minor ticks
    groups = [-0.5, 4.5, 6.5, 15.5, 24.5, 41.5, 48.5, 52.5]
    ax.set_xticks(groups, minor=True)
    ax.set_yticks(groups, minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
    # Major ticks
    group_centers = [2, 5.5, 11, 20, 33, 45, 50.5]
    ax.set_xticks(group_centers)
    ax.set_yticks(group_centers)

    # Labels for major ticks
    labels = ['SC', 'AU', 'SM', 'VI', 'CC', 'DM', 'CB']
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=90, fontdict={'verticalalignment': 'center'})
    ax.tick_params(axis='both', which='major', length=0)
    #ax.set_frame_on(False)
    plt.savefig(title+'.svg', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


plt.figure(figsize=(10,5))
fnc_plot(FBIRN05, 'FBIRN', clim=(-0.6, 0.6))

