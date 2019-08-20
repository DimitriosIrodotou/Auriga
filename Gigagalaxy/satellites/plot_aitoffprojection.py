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


# from Tollerud et al. (2008)
def SDSS_detection_limit(Vbandmag, f=0.194, a=0.6, b=5.23):
    radius = (3.0 / 4.0 * np.pi * f) ** (1.0 / 3.0) * 10 ** ((-a * Vbandmag - b) / 3.0)
    return radius


def plot_aitoffprojection(runs, dirs, outpath, snap, suffix):
    fac = 2.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True)
        s.calc_sf_indizes(sf)
        rotmatrix = s.select_halo(sf, use_principal_axis=True, do_rotation=False)
        
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, photometry=True)
        subhalos.subhalospos = rotate_value(subhalos.subhalospos, rotmatrix)
        magnitudes = subhalos.subhalosphotometry[:, band_array['V']]
        maxrad = SDSS_detection_limit(magnitudes)
        
        ax = subplot(111, projection='aitoff')
        ax.grid(True, color='gray')
        deg2rad = np.pi / 180.
        
        ax.set_longitude_grid_ends(95.)
        ylab = ylabel('Galactic latitude (degrees)')
        ax.text(0, -115 * deg2rad, 'Galactic longitude (degrees)', ha='center', va='center', size=ylab.get_size())
        
        ax.set_xticks(
                [-np.pi, -5.0 * np.pi / 6.0, -2.0 * np.pi / 3.0, -np.pi / 2.0, -np.pi / 3.0, -np.pi / 6.0, 0.0, np.pi / 6.0, np.pi / 3.0, np.pi / 2.0,
                 2.0 * np.pi / 3.0, 5.0 * np.pi / 6.0, np.pi])
        
        ax.set_yticks([-np.pi / 2.0, -np.pi / 3.0, -np.pi / 6.0, 0.0, np.pi / 6.0, np.pi / 3.0, np.pi / 2.0])
        
        fig.canvas.draw()
        
        labels = [item.get_text() for item in ax.get_xticklabels()]
        
        for i in range(0, len(labels), 2):
            labels[i] = ""
        
        ax.set_xticklabels(labels)
        
        galactic_latitude = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        galactic_longitude = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        
        sunposition = 8.5e-3  # in Mpc
        x = subhalos.subhalospos[:, 1] - sunposition
        y = subhalos.subhalospos[:, 2]
        z = subhalos.subhalospos[:, 0]
        r = sqrt(x * x + y * y + z * z)
        R = sqrt(x * x + y * y)
        
        # selecting the star-forming gas for each of the subhalos
        galactic_latitude[:] = np.arcsin(z / r)
        galactic_longitude[:] = np.arccos(-x / R)
        j, = np.where(y > 0)
        galactic_longitude[j] = -galactic_longitude[j]
        
        # only plot subhalos with stars
        j, = np.where(subhalos.subhaloslentype[:, 4] > 0)
        print
        'Total satellites', len(j)
        
        if len(j) == 0:
            continue
        
        galactic_longitude = galactic_longitude[j]
        galactic_latitude = galactic_latitude[j]
        r = r[j]
        maxrad = maxrad[j]
        magnitudes = magnitudes[j]
        size = 10 * magnitudes / magnitudes.min()
        
        j, = np.where(r < maxrad)
        print
        'Detectable satellites', len(j)
        if len(j) > 0:
            for index in j:
                ax.plot(galactic_longitude[index], galactic_latitude[index], 'o', ms=size[index], color='blue', label="$\\rm{%s}$" % runs[d])
        
        k, = np.where(r >= maxrad)
        print
        'Non-detectable satellites', len(k)
        if len(k) > 0:
            for index in k:
                ax.plot(galactic_longitude[index], galactic_latitude[index], '^', ms=size[index], color='red')
        
        savefig("%s/aitoffprojection_%s.%s" % (outpath, runs[d], suffix), dpi=300)
        fig.clf()