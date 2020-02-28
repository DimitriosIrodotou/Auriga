from __future__ import division

import os
import re
import glob
import pickle
import projections

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from loadmodules import *
from matplotlib import gridspec
from parallel_decorators import vectorize_parallel
from scripts.gigagalaxy.util import satellite_utilities

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}


def get_names_sorted(names):
    if list(names)[0].find("_"):
        names_sorted = np.array(list(names))
        names_sorted.sort()
        
        return names_sorted
    else:
        values = np.zeros(len(names))
        for i in range(len(names)):
            name = names[i]
            value = 0
            while not name[0].isdigit():
                value = value * 256 + ord(name[0])
                name = name[1:]
            values[i] = value * 256 + np.int32(name)
        isort = values.argsort()
        return np.array(names)[isort]


def create_axis(f, idx, ncol=5):
    ix = idx % ncol
    iy = idx // ncol
    
    s = 1.
    
    ax = f.iaxes(0.5 + ix * (s + 0.5), 0.3 + s + iy * (s + 0.6), s, s, top=False)
    ax2 = ax.twiny()
    return ax, ax2


def set_axis(s, ax, ax2, ylabel, ylim=None, ncol=5):
    z = np.array([5., 3., 2., 1., 0.5, 0.2, 0.0])
    a = 1. / (1 + z)
    
    times = np.zeros(len(a))
    for i in range(len(a)):
        times[i] = s.cosmology_get_lookback_time_from_a(a[i])
    
    lb = []
    for v in z:
        if v >= 1.0:
            lb += ["%.0f" % v]
        else:
            if v != 0:
                lb += ["%.1f" % v]
            else:
                lb += ["%.0f" % v]
    
    ax.set_xlim(0., 13.)
    ax.invert_xaxis()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(times)
    ax2.set_xticklabels(lb)
    
    ax.set_ylabel(ylabel, size=6)
    ax.set_xlabel("$t_\mathrm{look}\,\mathrm{[Gyr]}$", size=6)
    ax2.set_xlabel("$z$", size=6)
    
    for a in [ax, ax2]:
        for label in a.xaxis.get_ticklabels():
            label.set_size(6)
        for label in a.yaxis.get_ticklabels():
            label.set_size(6)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    return None


def set_axis_evo(ax, ax2):
    z = np.array([5., 3., 2., 1., 0.5, 0.2, 0.0])
    a = 1. / (1 + z)
    
    times = np.zeros(len(a))
    for i in range(len(a)):
        times = satellite_utilities.return_lookbacktime_from_a((z + 1.0) ** (-1.0))  # in Gyr
    
    lb = []
    for v in z:
        if v >= 1.0:
            lb += ["%.0f" % v]
        else:
            if v != 0:
                lb += ["%.1f" % v]
            else:
                lb += ["%.0f" % v]
    
    ax.set_xlim(0, 13)
    ax.invert_xaxis()
    ax.set_ylabel(r'AGN feedback energy [ergs]', size=16)
    ax.set_xlabel(r'$t_\mathrm{look}\,\mathrm{[Gyr]}$', size=16)
    ax.tick_params(direction='out', which='both', right='on')
    
    ax2.set_xticks(times)
    ax2.set_xticklabels(lb)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel(r'$z$', size=16)
    ax2.tick_params(direction='out', which='both', top='on', right='on')
    
    return None


def sfr(pdf, data, levels):
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], 0.)
        nhalos += data.selected_current_nsnaps
    
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))
    
    nbins = 100
    tmin = 0
    tmax = 13.
    timebin = (tmax - tmin) / nbins
    
    for il in range(nlevels):
        level = levels[il]
        data.select_haloes(level, 0., loadonlytype=[4], loadonlyhalo=0)
        
        isnap = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])
            
            mask, = np.where((s.data['age'] > 0.) & (s.r() < 0.005) & (s.pos[:, 2] < 0.003))
            age = s.cosmology_get_lookback_time_from_a(s.data['age'][mask], is_flat=True)
            
            ax, ax2 = create_axis(f, isnap)
            ax.hist(age, weights=s.data['gima'][mask] * 1e10 / 1e9 / timebin, histtype='step', bins=nbins, range=[tmin, tmax])
            set_axis(s, ax, ax2, "$\\mathrm{Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$")
            
            ax.text(0.05, 0.92, "Au%s-%d r < 5kpc" % (s.haloname, level), color='k', fontsize=6, transform=ax.transAxes)
            
            isnap += 1
    
    pdf.savefig(f, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def compute_bfld_halo(snapid, halo, offset):
    s = halo.snaps[snapid].loadsnap(loadonlytype=[0], loadonlyhalo=0)
    s.centerat(s.subfind.data['fpos'][0, :])
    
    time = s.cosmology_get_lookback_time_from_a(s.time, is_flat=True)
    i, = np.where(s.r() < 0.001)
    bfld = np.sqrt(((s.data['bfld'][i, :] ** 2.).sum(axis=1) * s.data['vol'][i]).sum() / s.data['vol'][i].sum()) * bfac
    return time, bfld


def bfld(pdf, data, levels):
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], 0.)
        nhalos += data.selected_current_nsnaps
    
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))
    
    for il in range(nlevels):
        level = levels[il]
        
        res = {}
        rpath = "../plots/data/bfld_%d.npy" % level
        if os.path.exists(rpath):
            with open(rpath, 'rb') as ff:
                res = pickle.load(ff)
        
        halos = data.get_haloes(level)
        for name, halo in halos.items():
            if name not in res:
                redshifts = halo.get_redshifts()
                
                i, = np.where(redshifts < 10.)
                snapids = np.array(list(halo.snaps.keys()))[i]
                
                dd = np.array(compute_bfld_halo(snapids, halo, snapids.min()))
                res[name] = {}
                res[name]["time"] = dd[:, 0]
                res[name]["bfld"] = dd[:, 1]
        
        with open(rpath, 'wb') as ff:
            pickle.dump(res, ff, pickle.HIGHEST_PROTOCOL)
        
        names = get_names_sorted(res.keys())
        for idx, name in enumerate(names):
            ax, ax2 = create_axis(f, idx)
            
            ax.semilogy(res[name]["time"], res[name]["bfld"] * 1e6)
            set_axis(list(halos[name].snaps.values())[0].loadsnap(), ax, ax2, "$B_\mathrm{r<1\,kpc}\,\mathrm{[\mu G]}$", [1e-2, 1e2])
            ax.set_xlim([13., 11.])
    
    pdf.savefig(f, bbox_inches='tight')  # Save the figure.
    plt.close()
    
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))
    
    for il in range(nlevels):
        level = levels[il]
        
        res = {}
        rpath = "../plots/data/bfld_%d.npy" % level
        if os.path.exists(rpath):
            with open(rpath, 'rb') as ff:
                res = pickle.load(ff)
        
        names = get_names_sorted(res.keys())
        for idx, name in enumerate(names):
            ax, ax2 = create_axis(f, idx)
            
            ax.plot(res[name]["time"], res[name]["bfld"] * 1e6)
            set_axis(list(halos[name].snaps.values())[0].loadsnap(), ax, ax2, "$B_\mathrm{r<1\,kpc}\,\mathrm{[\mu G]}$", [0., 100.])
    
    pdf.savefig(f, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_bh_mass(snapid, halo, bhid):
    s = halo.snaps[snapid].loadsnap(loadonlytype=[5], loadonly=['mass', 'id'])
    if 'id' not in s.data:
        return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), 0.
    
    i, = np.where(s.data['id'] == bhid)
    if len(i) > 0:
        mass = s.mass[i[0]]
    else:
        mass = 0.
    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), mass


def bh_mass(pdf, data, levels):
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], 0.)
        nhalos += data.selected_current_nsnaps
    
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))
    
    for il in range(nlevels):
        level = levels[il]
        
        res = {}
        rpath = "../plots/data/bhmass_%d.npy" % level
        if os.path.exists(rpath):
            with open(rpath, 'rb') as ff:
                res = pickle.load(ff)
        
        halos = data.get_haloes(level)
        for name, halo in halos.items():
            if name not in res:
                redshifts = halo.get_redshifts()
                
                i, = np.where(redshifts < 10.)
                snapids = np.array(list(halo.snaps.keys()))[i]
                
                s = halo.snaps[snapids.argmax()].loadsnap(loadonlytype=[5], loadonlyhalo=0)
                bhid = s.data['id'][s.mass.argmax()]
                
                dd = np.array(get_bh_mass(snapids, halo, bhid))
                res[name] = {}
                res[name]["time"] = dd[:, 0]
                res[name]["mass"] = dd[:, 1]
        
        with open(rpath, 'wb') as ff:
            pickle.dump(res, ff, pickle.HIGHEST_PROTOCOL)
        
        names = get_names_sorted(res.keys())
        for idx, name in enumerate(names):
            ax, ax2 = create_axis(f, idx)
            
            ax.semilogy(res[name]["time"], res[name]["mass"] * 1e10)
            set_axis(list(halos[name].snaps.values())[0].loadsnap(), ax, ax2, "$M_\mathrm{BH}\,\mathrm{[M_\odot]}$")
            ax.text(0.05, 0.92, "Au%s" % name, color='k', fontsize=6, transform=ax.transAxes)
    
    pdf.savefig(f, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


def bar_strength_evolution(pdf, data, read):
    """
        Calculate the evolution of bar strength from Fourier modes of surface density.
        :param pdf:
        :param data:
        :param read: boolean.
        :return:
        """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'bse/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        redshift_cut = 5.0
        A2, names = [], []  # Declare lists to store the data.
        # Get all available redshifts #
        haloes = data.get_haloes(4)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()
        
        for redshift in redshifts[np.where(redshifts <= redshift_cut)]:
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [4]
            attributes = ['age', 'mass', 'pos']
            data.select_haloes(4, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
            
            # Loop over all haloes #
            for s in data:
                # Check if any of the haloes' data already exists, if not then read and save it #
                names = glob.glob(path + '/name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                if str(s.haloname) in names:
                    continue
                
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned to the z-axis #
                s.calc_sf_indizes(s.subfind)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
                
                mask, = np.where(s.data['age'] > 0.0)  # Mask the data: select stellar particles.
                
                # Rotate the particle positions so the bar is along the x-axis #
                z_rotated, y_rotated, x_rotated = projections.rotate_bar(s.pos[mask, 0] * 1e3, s.pos[mask, 1] * 1e3,
                                                                         s.pos[mask, 2] * 1e3)  # Distances are in Mpc.
                s.pos = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.pos attribute in kpc.
                x, y = s.pos[:, 2] * 1e3, s.pos[:, 1] * 1e3  # Load positions and convert from Mpc to Kpc.
                
                # Split up galaxy in radius bins and calculate the Fourier components #
                nbins = 40  # Number of radial bins.
                r = np.sqrt(x[:] ** 2 + y[:] ** 2)  # Radius of each particle.
                
                # Initialise Fourier components #
                r_m = np.zeros(nbins)
                beta_2 = np.zeros(nbins)
                alpha_0 = np.zeros(nbins)
                alpha_2 = np.zeros(nbins)
                
                # Calculate the Fourier components for each bin as in sec 2.3.2 from Athanassoula et al. 2013 #
                for i in range(0, nbins):
                    r_s = float(i) * 0.25
                    r_b = float(i) * 0.25 + 0.25
                    r_m[i] = float(i) * 0.25 + 0.125
                    xfit = x[(r < r_b) & (r > r_s)]
                    yfit = y[(r < r_b) & (r > r_s)]
                    for k in range(0, len(xfit)):
                        th_i = np.arctan2(yfit[k], xfit[k])
                        alpha_0[i] = alpha_0[i] + 1
                        alpha_2[i] = alpha_2[i] + np.cos(2 * th_i)
                        beta_2[i] = beta_2[i] + np.sin(2 * th_i)
                
                # Calculate bar strength A_2 #
                A2.append(max(np.divide(np.sqrt(alpha_2[:] ** 2 + beta_2[:] ** 2), alpha_0[:])))
                
                # Save data for each halo in numpy arrays #
                np.save(path + 'A2_' + str(s.haloname), A2)
                np.save(path + 'name_' + str(s.haloname), s.haloname)
                np.save(path + 'redshifts_' + str(s.haloname), redshifts[np.where(redshifts <= redshift_cut)])
    
    # Load and plot the bar strength as a function of redshift #
    names = glob.glob(path + '/name_*')
    names.sort()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(names))))
    
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        f, ax = plt.subplots(1, figsize=(10, 7.5))
        plt.grid(True)
        plt.xlim(0, 2)
        plt.ylim(0, 1)
        plt.xlabel(r'Redshift', size=16)
        plt.ylabel(r'$\mathrm{A_{2}}$', size=16)
        
        A2 = np.load(path + 'A2_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts = np.load(path + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        plt.plot(redshifts, A2, color=next(colors), label='Au-' + str(re.split('_|.npy', names[i])[1]))
        ax.legend(loc='upper left', fontsize=12, frameon=False, numpoints=1)
        
        pdf.savefig(f, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def gas_temperature_fraction_evolution(pdf, data, read):
    """
        Calculate the evolution of gas fraction in different temperature regimes.
        :param pdf:
        :param data:
        :param read: boolean.
        :return:
        """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'gtfe/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        redshift_cut = 5.0
        sfg_ratios, wg_ratios, hg_ratios, names, masses = [], [], [], [], []  # Declare lists to store the data.
        
        # Get all available redshifts #
        haloes = data.get_haloes(4)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()
        
        for redshift in redshifts[np.where(redshifts <= redshift_cut)]:
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [0, 4]
            attributes = ['age', 'mass', 'ne', 'pos', 'rho', 'u']
            data.select_haloes(4, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
            
            # Loop over all haloes #
            for s in data:
                # Check if any of the haloes' data already exists, if not then read and save it #
                names = glob.glob(path + '/name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                if str(s.haloname) in names:
                    continue
                
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned to the z-axis #
                s.calc_sf_indizes(s.subfind, verbose=False)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
                
                mask, = np.where(
                    (s.r() < s.subfind.data['frc2'][0]) & (s.type == 0))  # Mask the data: select gas cells within the virial radius R200 #
                
                # Calculate the temperature of the gas cells #
                ne = s.data['ne'][mask]
                metallicity = s.data['gz'][mask]
                XH = s.data['gmet'][mask, element['H']]
                yhelium = (1 - XH - metallicity) / (4. * XH)
                mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
                u = GAMMA_MINUS1 * s.data['u'][mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN
                
                # Calculate the mass of the gas cells within three temperatures regimes #
                mass = s.data['mass'][mask]
                masses.append(np.sum(mass))
                sfg_ratios.append(np.sum(mass[np.where((u < 2e4))]) / np.sum(mass))
                wg_ratios.append(np.sum(mass[np.where((u >= 2e4) & (u < 5e5))]) / np.sum(mass))
                hg_ratios.append(np.sum(mass[np.where((u >= 5e5))]) / np.sum(mass))
        
        # Save data for each halo in numpy arrays #
        np.save(path + 'masses_' + str(s.haloname), masses)
        np.save(path + 'name_' + str(s.haloname), s.haloname)
        np.save(path + 'sfg_ratios_' + str(s.haloname), sfg_ratios)
        np.save(path + 'wg_ratios_' + str(s.haloname), wg_ratios)
        np.save(path + 'hg_ratios_' + str(s.haloname), hg_ratios)
        np.save(path + 'redshifts_' + str(s.haloname), redshifts[np.where(redshifts <= redshift_cut)])
    
    # Generate the figure and define its parameters #
    f, ax = plt.subplots(1, figsize=(10, 7.5))
    plt.ylim(-0.2, 1.2)
    plt.xlim(0, 5)
    plt.grid(True)
    plt.ylabel(r'Gas fraction', size=16)
    plt.xlabel(r'Redshift', size=16)
    
    # Load and plot the data #
    names = glob.glob(path + '/name_17.*')
    names.sort()
    
    for i in range(len(names)):
        sfg_ratios = np.load(path + 'sfg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        wg_ratios = np.load(path + 'wg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        hg_ratios = np.load(path + 'hg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts = np.load(path + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        # Uncomment the following lines if you want to make a stacked bar plot #
        # redshifts = np.insert(redshifts, 0, 5.1)
        # sfg_ratios = np.flip(sfg_ratios)
        # wg_ratios = np.flip(wg_ratios)
        # hg_ratios = np.flip(hg_ratios)
        # redshifts = np.flip(redshifts)
        
        # for j in range(len(redshifts) - 1):  # b1, = plt.bar(redshifts[j], sfg_ratios[j], width=redshifts[j + 1] - redshifts[j], alpha=0.6,
        # color='blue', align='edge', edgecolor='none')  # b2, = plt.bar(redshifts[j], wg_ratios[j], bottom=sfg_ratios[j], width=redshifts[j + 1] -
        # redshifts[j], alpha=0.6, color='green',  #               align='edge', edgecolor='none')  # b3, = plt.bar(redshifts[j], hg_ratios[j],
        # bottom=np.sum(np.vstack([sfg_ratios[j], wg_ratios[j]]).T),  #               width=redshifts[j + 1] - redshifts[j], alpha=0.6,
        # align='edge', color='red', edgecolor='none')
        
        b1, = plt.plot(redshifts, sfg_ratios, color='blue')
        b2, = plt.plot(redshifts, wg_ratios, color='green')
        b3, = plt.plot(redshifts, hg_ratios, color='red')
    
    plt.legend([b3, b2, b1], [r'Hot gas', r'Warm gas', r'Cold star-forming gas'], loc='upper left', fontsize=12, frameon=False, numpoints=1)
    ax.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=ax.transAxes)
    pdf.savefig(f, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


def AGN_modes_cumulative(date, data, read):
    """
        Get information about different black hole modes from log files and plot the evolution of the cumulative feedback.
        :param date: .
        :param data: .
        :param read: boolean.
        :return: None
        """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmc/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    data.select_haloes(4, 0, loadonlytype=None, loadonlyhalo=0, loadonly=None)
    
    # Loop over all haloes #
    for s in data:
        # Check if any of the haloes' data already exists, if not then read and save it #
        names = glob.glob(path + '/name_*')
        names = [re.split('_|.npy', name)[1] for name in names]
        if str(s.haloname) in names:
            continue
        
        # Declare arrays to store the desired words and lines that contain these words #
        redshift_lines, redshifts, feedback_lines, thermals, mechanicals = [], [], [], [], []
        # Read the data #
        if read is True:
            with open('/u/di43/Auriga/output/halo_' + str(s.haloname) + '/Au-' + str(s.haloname) + '.txt') as file:
                # Iterate over each line #
                for line in file:
                    # Convert the characters in the line to lowercase and split the line into words #
                    line = line.lower()
                    line = line.strip()  # Remove '\n' at end of line.
                    words = line.split(" ")
                    
                    # Search for the word 'redshift:' and get the next word which is the redshift value #
                    if 'redshift:' in words:
                        redshift_lines.append(line)
                        redshifts.append(words[words.index('redshift:') + 1])
                    
                    # Search for the words 'thermal' and 'mechanical' and get the words after next which are the energy values in ergs #
                    if 'black_holes:' in words and '(cumulative)' in words and 'is' in words and 'thermal' in words and 'mechanical' in words:
                        feedback_lines.append(line)
                        thermals.append(words[words.index('thermal') + 2])
                        mechanicals.append(words[words.index('mechanical') + 2])
            
            file.close()  # Close the opened file.
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'thermals_' + str(s.haloname), thermals)
            np.save(path + 'redshifts_' + str(s.haloname), redshifts)
            np.save(path + 'mechanicals_' + str(s.haloname), mechanicals)
    
    # Load and plot the data #
    names = glob.glob(path + '/name_18*')
    names.sort()
    
    # Load and plot the data #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(1, 2, wspace=0.05, width_ratios=[1, 0.05])
        ax00 = plt.subplot(gs[0, 0])
        axcbar = plt.subplot(gs[:, 1])
        
        ax00.grid(True)
        ax00.set_xscale('log')
        ax00.set_yscale('log')
        ax00.set_xlim(1e54, 1e62)
        ax00.set_ylim(1e54, 1e62)
        ax00.set_xlabel(r'Cumulative thermal feedback energy [ergs]', size=16)
        ax00.set_ylabel(r'Cumulative mechanical feedback energy [ergs]', size=16)
        f.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=ax00.transAxes)
        
        thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts = np.load(path + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Transform the arrays to comma separated strings and convert each element to float #
        redshifts = [re.sub(',', '', i) for i in redshifts]  # Remove the commas at the end of each redshift string.
        thermals = ','.join(thermals)
        redshifts = ','.join(redshifts)
        mechanicals = ','.join(mechanicals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')
        redshifts = np.fromstring(redshifts, dtype=np.float, sep=',')
        mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
        
        # Mask the data and plot the scatter #
        ax00.plot([1e54, 1e62], [1e54 / 10, 1e62 / 10], label='1:10')
        ax00.plot([1e54, 1e62], [1e54 / 50, 1e62 / 50], label='1:50')
        
        mask, = np.where((mechanicals != 0) | (thermals != 0))
        sc = ax00.scatter(thermals[mask], mechanicals[mask], edgecolor='None', s=50, c=redshifts[mask], vmin=0, vmax=7, cmap='jet')
        cb = plt.colorbar(sc, cax=axcbar)
        cb.set_label(r'Redshift', size=16)
        ax00.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
        
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmc-' + date + '.png', bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def AGN_modes_histogram(date, data, read):
    """
        Get information about different black hole modes from log files and plot a histogram of the evolution of the step feedback.
        :param date: .
        :param data: .
        :param read: boolean.
        :return: None
        """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmh/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    data.select_haloes(4, 0, loadonlytype=None, loadonlyhalo=0, loadonly=None)
    
    # Loop over all haloes #
    for s in data:
        # Check if any of the haloes' data already exists, if not then read and save it #
        names = glob.glob(path + '/name_*')
        names = [re.split('_|.npy', name)[1] for name in names]
        if str(s.haloname) in names:
            continue
        
        # Declare arrays to store the desired words and lines that contain these words #
        redshift_lines, redshifts, feedback_lines, thermals, mechanicals = [], [], [], [], []
        # Read the data #
        if read is True:
            with open('/u/di43/Auriga/output/halo_' + str(s.haloname) + '/Au-' + str(s.haloname) + '.txt') as file:
                # Iterate over each line #
                for line in file:
                    # Convert the characters in the line to lowercase and split the line into words #
                    line = line.lower()
                    line = line.strip()  # Remove '\n' at end of line.
                    words = line.split(" ")
                    
                    # Search for the word 'redshift:' and get the next word which is the redshift value #
                    if 'redshift:' in words:
                        redshift_lines.append(line)
                        redshifts.append(words[words.index('redshift:') + 1])
                    
                    # Search for the words 'thermal' and 'mechanical' and get the words after next which are the energy values in ergs #
                    if 'black_holes:' in words and '(step)' in words and 'is' in words and 'thermal' in words and 'mechanical' in words:
                        feedback_lines.append(line)
                        thermals.append(words[words.index('thermal') + 2])
                        mechanicals.append(words[words.index('mechanical') + 2])
            
            file.close()  # Close the opened file.
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'thermals_' + str(s.haloname), thermals)
            np.save(path + 'redshifts_' + str(s.haloname), redshifts)
            np.save(path + 'mechanicals_' + str(s.haloname), mechanicals)
    
    # Load and plot the data #
    names = glob.glob(path + '/name_18*')
    names.sort()
    
    # Load and plot the data #
    for i in range(len(names)):
        
        # Generate the figure and define its parameters #
        f, ax = plt.subplots(1, figsize=(10, 7.5))
        ax.grid(True)
        ax2 = ax.twiny()
        ax.set_yscale('log')
        set_axis_evo(ax, ax2)
        ax.text(0.01, 0.95, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=ax.transAxes)
        
        # Load and plot the data #
        thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts = np.load(path + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Transform the arrays to comma separated strings and convert each element to float #
        redshifts = [re.sub(',', '', i) for i in redshifts]  # Remove the commas at the end of each redshift string.
        thermals = ','.join(thermals)
        redshifts = ','.join(redshifts)
        mechanicals = ','.join(mechanicals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')
        redshifts = np.fromstring(redshifts, dtype=np.float, sep=',')
        mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
        
        # Convert redshifts to lookback times and plot #
        lookback_times = satellite_utilities.return_lookbacktime_from_a((redshifts + 1.0) ** (-1.0))  # in Gyr
        ax.hist(lookback_times, weights=thermals, histtype='step', bins=100, label=r'Thermal')
        ax.hist(lookback_times, weights=mechanicals, histtype='step', bins=100, label=r'Mechanical')
        
        ax.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmh-' + date + '.png', bbox_inches='tight')  # Save the figure.
        plt.close()
    
    return None


def AGN_modes_distribution(date, data, read):
    """
        Get information about different black hole modes from log files and plot the evolution of the step feedback.
        :param date: .
        :param data: .
        :param read: boolean.
        :return: None
        """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    data.select_haloes(4, 0, loadonlytype=None, loadonlyhalo=0, loadonly=None)
    
    # Loop over all haloes #
    for s in data:
        # Check if any of the haloes' data already exists, if not then read and save it #
        names = glob.glob(path + '/name_*')
        names = [re.split('_|.npy', name)[1] for name in names]
        if str(s.haloname) in names:
            continue
        
        # Declare arrays to store the desired words and lines that contain these words #
        redshift_lines, redshifts, feedback_lines, thermals, mechanicals = [], [], [], [], []
        # Read the data #
        if read is True:
            with open('/u/di43/Auriga/output/halo_' + str(s.haloname) + '/Au-' + str(s.haloname) + '.txt') as file:
                # Iterate over each line #
                for line in file:
                    # Convert the characters in the line to lowercase and split the line into words #
                    line = line.lower()
                    line = line.strip()  # Remove '\n' at end of line.
                    words = line.split(" ")
                    
                    # Search for the word 'redshift:' and get the next word which is the redshift value #
                    if 'redshift:' in words:
                        redshift_lines.append(line)
                        redshifts.append(words[words.index('redshift:') + 1])
                    
                    # Search for the words 'thermal' and 'mechanical' and get the words after next which are the energy values in ergs #
                    if 'black_holes:' in words and '(step)' in words and 'is' in words and 'thermal' in words and 'mechanical' in words:
                        feedback_lines.append(line)
                        thermals.append(words[words.index('thermal') + 2])
                        mechanicals.append(words[words.index('mechanical') + 2])
            
            file.close()  # Close the opened file.
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'thermals_' + str(s.haloname), thermals)
            np.save(path + 'redshifts_' + str(s.haloname), redshifts)
            np.save(path + 'mechanicals_' + str(s.haloname), mechanicals)
    
    # Load and plot the data #
    names = glob.glob(path + '/name_06*')
    names.sort()
    
    # Load and plot the data #
    for i in range(len(names)):
        
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.1, height_ratios=[0.05, 1])
        ax00 = plt.subplot(gs[1, 0])
        axcbar = plt.subplot(gs[0, 0])
        ax02 = plt.subplot(gs[1, 1])
        axcbar2 = plt.subplot(gs[0, 1])
        
        ax02.yaxis.set_label_position("right")
        ax02.yaxis.tick_right()
        for a in [ax00, ax02]:
            a.grid(True)
            a.set_xlim(12, 0)
            a.set_yscale('log')
            a.set_ylim(1e51, 1e61)
            a.set_xlabel(r'Redshift', size=16)
            a.tick_params(direction='out', which='both', right='on', left='on')
        ax00.set_ylabel(r'Mechanical feedback energy [ergs]', size=16)
        ax02.set_ylabel(r'Thermal feedback energy [ergs]', size=16)
        f.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=ax00.transAxes)
        
        # Load and plot the data #
        thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts = np.load(path + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Transform the arrays to comma separated strings and convert each element to float #
        redshifts = [re.sub(',', '', i) for i in redshifts]  # Remove the commas at the end of each redshift string.
        thermals = ','.join(thermals)
        redshifts = ','.join(redshifts)
        mechanicals = ','.join(mechanicals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')
        redshifts = np.fromstring(redshifts, dtype=np.float, sep=',')
        mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
        lookback_times = satellite_utilities.return_lookbacktime_from_a((redshifts + 1.0) ** (-1.0))  # Convert redshifts to lookback times in Gyr.
        
        # Plot hexbins #
        hb = ax00.hexbin(lookback_times[np.where(mechanicals > 0)], mechanicals[np.where(mechanicals > 0)], yscale='log', cmap='gist_heat_r',
                         gridsize=100)
        cb = plt.colorbar(hb, cax=axcbar, orientation='horizontal')
        cb.set_label(r'Counts per hexbin', size=16)
        
        hb = ax02.hexbin(lookback_times[np.where(thermals > 0)], thermals[np.where(thermals > 0)], yscale='log', cmap='gist_heat_r', gridsize=100)
        cb2 = plt.colorbar(hb, cax=axcbar2, orientation='horizontal')
        cb2.set_label(r'Counts per hexbin', size=16)
        
        for a in [axcbar, axcbar2]:
            a.xaxis.tick_top()
            a.xaxis.set_label_position("top")
            a.tick_params(direction='out', which='both', right='on')
        
        # Calculate median and 1-sigma #
        nbin = int((max(lookback_times[np.where(mechanicals > 0)]) - min(lookback_times[np.where(mechanicals > 0)])) / 0.02)
        x_value = np.empty(nbin)
        median = np.empty(nbin)
        slow = np.empty(nbin)
        shigh = np.empty(nbin)
        x_low = min(lookback_times[np.where(mechanicals > 0)])
        for j in range(nbin):
            index = np.where((lookback_times[np.where(mechanicals > 0)] >= x_low) & (lookback_times[np.where(mechanicals > 0)] < x_low + 0.05))[0]
            x_value[j] = np.mean(np.absolute(lookback_times[np.where(mechanicals > 0)])[index])
            if len(index) > 0:
                median[j] = np.sum(mechanicals[np.where(mechanicals > 0)][index])
                slow[j] = np.nanpercentile(mechanicals[np.where(mechanicals > 0)][index], 15.87)
                shigh[j] = np.nanpercentile(mechanicals[np.where(mechanicals > 0)][index], 84.13)
            x_low += 0.05
        
        # Plot median and 1-sigma lines #
        median, = ax00.plot(x_value, median, color='black', zorder=5)
        # ax00.fill_between(x_value, shigh, slow, color='black', alpha='0.3', zorder=5)
        
        # Calculate median and 1-sigma #
        nbin = int((max(lookback_times[np.where(thermals > 0)]) - min(lookback_times[np.where(thermals > 0)])) / 0.02)
        x_value = np.empty(nbin)
        median = np.empty(nbin)
        slow = np.empty(nbin)
        shigh = np.empty(nbin)
        x_low = min(lookback_times[np.where(thermals > 0)])
        for j in range(nbin):
            index = np.where((lookback_times[np.where(thermals > 0)] >= x_low) & (lookback_times[np.where(thermals > 0)] < x_low + 0.05))[0]
            x_value[j] = np.mean(np.absolute(lookback_times[np.where(thermals > 0)])[index])
            if len(index) > 0:
                median[j] = np.sum(thermals[np.where(thermals > 0)][index])
                slow[j] = np.nanpercentile(thermals[np.where(thermals > 0)][index], 15.87)
                shigh[j] = np.nanpercentile(thermals[np.where(thermals > 0)][index], 84.13)
            x_low += 0.05
        
        # Plot median and 1-sigma lines #
        median, = ax02.plot(x_value, median, color='black', zorder=5)
        # ax02.fill_between(x_value, shigh, slow, color='black', alpha='0.3', zorder=5)
        
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmd-' + date + '.png', bbox_inches='tight')  # Save the figure.
        plt.close()
    
    return None


def AGN_modes_step(date, data, read):
    """
        Get information about different black hole modes from log files and plot the step feedbacks.
        :param date: .
        :param data: .
        :param read: boolean.
        :return: None
        """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNms/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    data.select_haloes(4, 0, loadonlytype=None, loadonlyhalo=0, loadonly=None)
    
    # Loop over all haloes #
    for s in data:
        # Check if any of the haloes' data already exists, if not then read and save it #
        names = glob.glob(path + '/name_*')
        names = [re.split('_|.npy', name)[1] for name in names]
        if str(s.haloname) in names:
            continue
        
        # Declare arrays to store the desired words and lines that contain these words #
        feedback_lines, thermals, mechanicals = [], [], []
        # Read the data #
        if read is True:
            with open('/u/di43/Auriga/output/halo_' + str(s.haloname) + '/Au-' + str(s.haloname) + '.txt') as file:
                # Iterate over each line #
                for line in file:
                    # Convert the characters in the line to lowercase and split the line into words #
                    line = line.lower()
                    line = line.strip()  # Remove '\n' at end of line.
                    words = line.split(" ")
                    
                    # Search for the words 'thermal' and 'mechanical' and get the words after next which are the energy values in ergs #
                    if 'black_holes:' in words and '(step)' in words and 'is' in words and 'thermal' in words and 'mechanical' in words:
                        feedback_lines.append(line)
                        thermals.append(words[words.index('thermal') + 2])
                        mechanicals.append(words[words.index('mechanical') + 2])
            
            file.close()  # Close the opened file.
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'thermals_' + str(s.haloname), thermals)
            np.save(path + 'mechanicals_' + str(s.haloname), mechanicals)
    
    # Load and plot the data #
    names = glob.glob(path + '/name_17*')
    names.sort()
    
    # Load and plot the data #
    for i in range(len(names)):
        
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 7.5))
        gs = plt.GridSpec(2, 2, hspace=0.07, wspace=0.07, height_ratios=[0.5, 1], width_ratios=[1, 0.5])
        ax00 = f.add_subplot(gs[0, 0])
        ax10 = f.add_subplot(gs[1, 0])
        ax11 = f.add_subplot(gs[1, 1])
        
        for a in [ax00, ax10, ax11]:
            a.grid(True)
            a.tick_params(direction='out', which='both', right='on', left='on')
        
        for a in [ax00, ax11]:
            a.yaxis.set_ticks_position('left')
            a.xaxis.set_ticks_position('bottom')
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
        
        ax00.set_xticklabels([])
        ax11.set_yticklabels([])
        
        ax10.set_xlim(0, 2e56)
        ax10.set_ylim(0, 5e57)
        ax00.set_yscale('log')
        ax11.set_xscale('log')
        ax00.set_ylabel(r'PDF', size=16)
        ax11.set_xlabel(r'PDF', size=16)
        ax10.set_ylabel(r'Thermal feedback energy [ergs]', size=16)
        ax10.set_xlabel(r'Mechanical feedback energy [ergs]', size=16)
        f.text(1.01, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=ax10.transAxes)
        
        # Load and plot the data #
        thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Transform the arrays to comma separated strings and convert each element to float #
        thermals = ','.join(thermals)
        mechanicals = ','.join(mechanicals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')
        mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
        
        # Plot the scatter and the axes histograms #
        ax10.scatter(mechanicals, thermals, s=50, edgecolor='none', c='k', marker="1")
        weights = np.ones_like(mechanicals) / float(len(mechanicals))

        ax00.hist(mechanicals, bins=np.linspace(0, 2e56, 100), histtype='step', weights=weights, orientation='vertical', color='k')
        weights = np.ones_like(thermals) / float(len(thermals))
        ax11.hist(thermals, bins=np.linspace(0, 5e57, 100), histtype='step', weights=weights, orientation='horizontal', color='k')
        
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNms-' + date + '.png', bbox_inches='tight')  # Save the figure.
        plt.close()
    
    return None


def AGN_modes_gas(date):
    """
        Get information about different black hole modes from log files and plot the step feedbacks.
        :param date: .
        :return: None
        """
    path_gas = '/u/di43/Auriga/plots/data/' + 'gtfe/'
    path_modes = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    
    # Load and plot the data #
    names = glob.glob(path_modes + '/name_*')
    names.sort()
    
    # Load and plot the data #
    for i in range(len(names)):
        
        # Generate the figure and define its parameters #
        f, ax = plt.subplots(1, figsize=(10, 7.5))
        plt.grid(True)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlim(0, 2)
        ax2 = ax.twinx()
        ax2.set_ylim(1e52, 1e56)
        ax2.set_yscale('log')
        ax.set_xlabel(r'Redshift', size=16)
        ax.set_ylabel(r'Gas fraction', size=16)
        ax2.set_ylabel(r'AGN feedback energy [ergs]', size=16)
        ax.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=ax.transAxes)
        
        # Load and plot the data #
        wg_ratios = np.load(path_gas + 'wg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        hg_ratios = np.load(path_gas + 'hg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts_gas = np.load(path_gas + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sfg_ratios = np.load(path_gas + 'sfg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        thermals = np.load(path_modes + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts_modes = np.load(path_modes + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        mechanicals = np.load(path_modes + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Transform the arrays to comma separated strings and convert each element to float #
        redshifts = [re.sub(',', '', i) for i in redshifts_modes]  # Remove the commas at the end of each redshift string.
        thermals = ','.join(thermals)
        redshifts = ','.join(redshifts)
        mechanicals = ','.join(mechanicals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')
        redshifts = np.fromstring(redshifts, dtype=np.float, sep=',')
        mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
        
        # Plot the gas fractions #
        b1, = ax.plot(redshifts_gas, sfg_ratios, color='blue')
        b2, = ax.plot(redshifts_gas, wg_ratios, color='green')
        b3, = ax.plot(redshifts_gas, hg_ratios, color='red')
        ax.legend([b3, b2, b1], [r'Hot gas', r'Warm gas', r'Cold star-forming gas'], loc='upper left', fontsize=12, frameon=False, numpoints=1)
        
        # Calculate median and 1-sigma #
        nbin = int((max(redshifts[np.where(mechanicals > 0)]) - min(redshifts[np.where(mechanicals > 0)])) / 0.02)
        x_value = np.empty(nbin)
        median = np.empty(nbin)
        slow = np.empty(nbin)
        shigh = np.empty(nbin)
        x_low = min(redshifts[np.where(mechanicals > 0)])
        for j in range(nbin):
            index = np.where((redshifts[np.where(mechanicals > 0)] >= x_low) & (redshifts[np.where(mechanicals > 0)] < x_low + 0.05))[0]
            x_value[j] = np.mean(np.absolute(redshifts[np.where(mechanicals > 0)])[index])
            if len(index) > 0:
                median[j] = np.nanmedian(mechanicals[np.where(mechanicals > 0)][index])
                slow[j] = np.nanpercentile(mechanicals[np.where(mechanicals > 0)][index], 15.87)
                shigh[j] = np.nanpercentile(mechanicals[np.where(mechanicals > 0)][index], 84.13)
            x_low += 0.05
        
        # Plot median and 1-sigma lines #
        median, = ax2.plot(x_value, median, color='black', zorder=5, label=r'Mechanical')
        # plt.fill_between(x_value, shigh, slow, color='black', alpha='0.3', zorder=5)
        
        # Calculate median and 1-sigma #
        nbin = int((max(redshifts[np.where(thermals > 0)]) - min(redshifts[np.where(thermals > 0)])) / 0.02)
        x_value = np.empty(nbin)
        median = np.empty(nbin)
        slow = np.empty(nbin)
        shigh = np.empty(nbin)
        x_low = min(redshifts[np.where(thermals > 0)])
        for j in range(nbin):
            index = np.where((redshifts[np.where(thermals > 0)] >= x_low) & (redshifts[np.where(thermals > 0)] < x_low + 0.05))[0]
            x_value[j] = np.mean(np.absolute(redshifts[np.where(thermals > 0)])[index])
            if len(index) > 0:
                median[j] = np.nanmedian(thermals[np.where(thermals > 0)][index])
                slow[j] = np.nanpercentile(thermals[np.where(thermals > 0)][index], 15.87)
                shigh[j] = np.nanpercentile(thermals[np.where(thermals > 0)][index], 84.13)
            x_low += 0.05
        
        # Plot median and 1-sigma lines #
        median, = ax2.plot(x_value, median, color='black', zorder=5, linestyle="dashed", label=r'Thermal')
        # plt.fill_between(x_value, shigh, slow, color='black', alpha='0.3', zorder=5)
        
        ax2.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmg-' + date + '.png', bbox_inches='tight')  # Save the figure.
        plt.close()
    
    return None