import numpy as np
from gadget import *
from gadget_subfind import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = '/output/fof_subhalo_tab_'
band_array = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}


def rotate_value(value, matrix):
    new_value = pylab.zeros(pylab.shape(value))
    for i in range(3):
        new_value[:, i] = (value * matrix[i, :][None, :]).sum(axis=1)
    return new_value


def plot_3D_positions(runs, dirs, outpath, snap, suffix):
    fac = 2.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1 * fac]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True)
        s.calc_sf_indizes(sf)
        rotmatrix = s.select_halo(sf, use_principal_axis=True, do_rotation=False)
        
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, main_halo=True, photometry=True)
        subhalos.subhalospos = rotate_value(subhalos.subhalospos, rotmatrix)
        magnitudes = subhalos.subhalosphotometry[:, band_array['V']]
        
        ax = subplot(111, projection='3d')
        
        sunposition = 8.5e-3  # in Mpc
        x = subhalos.subhalospos[:, 1] - sunposition
        y = subhalos.subhalospos[:, 2]
        z = subhalos.subhalospos[:, 0]
        
        print
        'central galaxy position', x[0], y[0], z[0]
        
        # only plot subhalos with stars
        j, = np.where(subhalos.subhaloslentype[:, 4] > 0)
        print
        'Total satellites', len(j)
        
        if len(j) == 0:
            continue
        x = x[j] * 1.0e3
        y = y[j] * 1.0e3
        z = z[j] * 1.0e3
        size = 50 * magnitudes[j] / magnitudes[j].min()
        
        # plot the central galaxy
        ax.scatter(x[0], y[0], z[0], c='g', marker='^', s=size[0])
        # plot the satellites
        ax.scatter(x[1:], y[1:], z[1:], c='b', marker='o', s=size[1:])
        ax.set_xlabel("$\\rm{x_{helio}\\,[kpc]}$")
        ax.set_ylabel("$\\rm{y_{helio}\\,[kpc]}$")
        ax.set_zlabel("$\\rm{z_{helio}\\,[kpc]}$")
        
        ax.set_xlim(-400., 400.)
        ax.set_ylim(-400., 400.)
        ax.set_zlim(-400., 400.)
        
        ax.view_init(15.0, -45.0)
        
        savefig("%s/3Dpositions_%s.%s" % (outpath, runs[d], suffix), dpi=300)
        fig.clf()