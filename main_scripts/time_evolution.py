from __future__ import division

import os
import re
import glob
import pickle
import plot_tools
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

res = 512
level = 4
boxsize = 0.06
colors = ['black', 'tab:red', 'tab:green', 'tab:blue', 'tab:orange']
element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}


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


def create_axis(figure, idx, ncol=5):
    ix = idx % ncol
    iy = idx // ncol
    
    s = 1.
    
    axis = figure.iaxes(0.5 + ix * (s + 0.5), 0.3 + s + iy * (s + 0.6), s, s, top=False)
    axis2 = axis.twiny()
    return axis, axis2


def set_axis(s, axis, axis2, ylabel, ylim=None):
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
    
    axis.set_xlim(0., 13.)
    axis.invert_xaxis()
    axis2.set_xlim(axis.get_xlim())
    axis2.set_xticks(times)
    axis2.set_xticklabels(lb)
    
    axis.set_ylabel(ylabel, size=6)
    axis.set_xlabel(r'$\mathrm{t_{look}\;[Gyr]}$', size=6)
    axis2.set_xlabel(r'$\mathrm{z}$', size=6)
    
    for axis in [axis, axis2]:
        for label in axis.xaxis.get_ticklabels():
            label.set_size(6)
        for label in axis.yaxis.get_ticklabels():
            label.set_size(6)
    
    if ylim is not None:
        axis.set_ylim(ylim)
    
    return None


def sfr(pdf, data, level):
    nhalos = 0
    data.select_haloes(level, 0.)
    nhalos += data.selected_current_nsnaps
    
    figure = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))
    
    time_bin = (13 - 0) / 100
    
    data.select_haloes(level, 0., loadonlytype=[4], loadonlyhalo=0)
    
    isnap = 0
    for s in data:
        s.centerat(s.subfind.data['fpos'][0, :])
        
        mask, = np.where((s.data['age'] > 0.) & (s.r() < 0.005) & (s.data['pos'][:, 2] < 0.003))
        age = s.cosmology_get_lookback_time_from_a(s.data['age'][mask], is_flat=True)
        
        axis, axis2 = create_axis(figure, isnap)
        axis.hist(age, weights=s.data['gima'][mask] * 1e10 / 1e9 / time_bin, histtype='step', bins=100, range=[0, 13])
        set_axis(s, axis, axis2, r'$\mathrm{Sfr\;[M_\odot\;yr^{-1}]}$')
        
        axis.text(0.05, 0.92, 'Au%s-%d r < 5kpc' % (s.haloname, level), color='k', fontsize=6, transform=axis.transAxes)
        
        isnap += 1
    
    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
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


def bfld(pdf, data, level):
    nhalos = 0
    data.select_haloes(level, 0.)
    nhalos += data.selected_current_nsnaps
    
    figure = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))
    
    res = {}
    rpath = '../plots/data/bfld_%d.npy' % level
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
        axis, axis2 = create_axis(figure, idx)
        
        axis.semilogy(res[name]["time"], res[name]["bfld"] * 1e6)
        set_axis(list(halos[name].snaps.values())[0].loadsnap(), axis, axis2, "$B_\mathrm{r<1\;kpc}\;\mathrm{[\mu G]}$", [1e-2, 1e2])
        axis.set_xlim([13., 11.])
    
    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
    plt.close()
    
    figure = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))
    
    res = {}
    rpath = '../plots/data/bfld_%d.npy' % level
    if os.path.exists(rpath):
        with open(rpath, 'rb') as ff:
            res = pickle.load(ff)
    
    names = get_names_sorted(res.keys())
    for idx, name in enumerate(names):
        axis, axis2 = create_axis(figure, idx)
        
        axis.plot(res[name]["time"], res[name]["bfld"] * 1e6)
        set_axis(list(halos[name].snaps.values())[0].loadsnap(), axis, axis2, '$B_\mathrm{r<1\;kpc}\;\mathrm{[\mu G]}$', [0., 100.])
    
    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_blackhole_data(snapshot_ids, halo, blackhole_id):
    """
    Parallelised method to get black hole properties.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :param blackhole_id: id of the black hole particle.
    :return: black hole properties.
    """
    
    # Check if there is a black hole particle, if not return only lookback times #
    s = halo.snaps[snapshot_ids].loadsnap(loadonlytype=[5], loadonly=['bhma', 'bcmr', 'bcmq', 'bhmd', 'bhmr', 'bhmq', 'id'])
    if 'id' not in s.data:
        return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), 0, 0, 0, 0, 0, 0
    
    # Return lookback times, black hole masses and growth rates #
    blackhole_mask, = np.where(s.data['id'] == blackhole_id)
    if len(blackhole_mask) > 0:
        black_hole_mass = s.data['bhma'][blackhole_mask[0]]
        black_hole_dmass = s.data['bhmd'][blackhole_mask[0]]
        black_hole_cmass_radio = s.data['bcmr'][blackhole_mask[0]]
        black_hole_dmass_radio = s.data['bhmr'][blackhole_mask[0]]
        black_hole_cmass_quasar = s.data['bcmq'][blackhole_mask[0]]
        black_hole_dmass_quasar = s.data['bhmq'][blackhole_mask[0]]
    else:
        black_hole_mass, black_hole_cmass_radio, black_hole_cmass_quasar, black_hole_dmass, black_hole_dmass_radio, black_hole_dmass_quasar, = 0, \
                                                                                                                                               0, \
                                                                                                                                               0, \
                                                                                                                                               0, 0, 0
    return s.cosmology_get_lookback_time_from_a(s.time,
                                                is_flat=True), black_hole_mass, black_hole_cmass_radio, black_hole_cmass_quasar, black_hole_dmass, \
           black_hole_dmass_radio, black_hole_dmass_quasar


def blackhole_masses(pdf, data, read):
    """
    Plot the evolution of black hole mass, accretion rate, cumulative mass accreted onto the BH in the low and high accretion-state.
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'bm/'
    path_modes = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        halos = data.get_haloes(level)
        
        for name, halo in halos.items():
            # Get all snapshots with redshift less than 7 #
            redshifts = halo.get_redshifts()
            redshift_mask, = np.where(redshifts < 7)
            snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]
            
            # Find the black hole's id and use it to get black hole data #
            s = halo.snaps[snapshot_ids.argmax()].loadsnap(loadonlytype=[5], loadonlyhalo=0)
            blackhole_id = s.data['id'][s.data['mass'].argmax()]
            blackhole_data = np.array(get_blackhole_data(snapshot_ids, halo, blackhole_id))
            lookback_times = blackhole_data[:, 0]
            black_hole_masses = blackhole_data[:, 1]
            black_hole_cmasses_radio, black_hole_cmasses_quasar = blackhole_data[:, 2], blackhole_data[:, 3]
            black_hole_dmasses, black_hole_dmasses_radio, black_hole_dmasses_quasar = blackhole_data[:, 4], blackhole_data[:, 5], blackhole_data[:, 6]
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(name), name)
            np.save(path + 'lookback_times_' + str(name), lookback_times)
            np.save(path + 'black_hole_masses_' + str(name), black_hole_masses)
            np.save(path + 'black_hole_dmasses_' + str(name), black_hole_dmasses)
            np.save(path + 'black_hole_cmasses_radio_' + str(name), black_hole_cmasses_radio)
            np.save(path + 'black_hole_dmasses_radio_' + str(name), black_hole_dmasses_radio)
            np.save(path + 'black_hole_cmasses_quasar_' + str(name), black_hole_cmasses_quasar)
    
    # Load and plot the data #
    names = glob.glob(path + '/name_18.*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.2)
        axis00 = plt.subplot(gs[0, 0])
        axis01 = plt.subplot(gs[0, 1])
        axis10 = plt.subplot(gs[1, 0])
        axis11 = plt.subplot(gs[1, 1])
        axis002 = axis00.twiny()
        axis012 = axis01.twiny()
        axis102 = axis10.twiny()
        axis112 = axis11.twiny()
        
        for axis in [axis00, axis01, axis10, axis11, axis002, axis012, axis102, axis112]:
            axis.set_yscale('log')
        for axis in [axis01, axis11]:
            axis.yaxis.set_label_position("right")
            axis.yaxis.tick_right()
        for axis in [axis00, axis01, axis10, axis11]:
            axis.grid(True, color='gray', linestyle='-')
            axis.tick_params(direction='out', which='both', top='on', right='on', left='on')
        
        plot_tools.set_axis_evo(axis00, axis002, ylabel=r'$\mathrm{M_\mathrm{BH}/\mathrm{M_\odot}}$')
        plot_tools.set_axis_evo(axis01, axis012, ylabel=r'$\mathrm{M_\mathrm{BH,mode}/\mathrm{M_\odot}}$')
        plot_tools.set_axis_evo(axis10, axis102, ylabel=r'$\mathrm{\dot{M}_\mathrm{BH}/(\mathrm{M_\odot\; Gyr^{-1}})}$')
        plot_tools.set_axis_evo(axis11, axis112, ylabel=r'$\mathrm{(AGN\;feedback\;energy)/ergs}$')
        
        # Load and plot the data #
        thermals = np.load(path_modes + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        mechanicals = np.load(path_modes + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_masses = np.load(path + 'black_hole_masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_dmasses = np.load(path + 'black_hole_dmasses_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times_modes = np.load(path_modes + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_cmasses_radio = np.load(path + 'black_hole_cmasses_radio_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_dmasses_radio = np.load(path + 'black_hole_dmasses_radio_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_cmasses_quasar = np.load(path + 'black_hole_cmasses_quasar_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_dmasses_quasar = np.load(path + 'black_hole_cmasses_quasar_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Convert cumulative masses to instantaneous measurements #
        black_hole_imasses_radio = np.insert(np.diff(black_hole_cmasses_radio), 0, 0)
        black_hole_imasses_quasar = np.insert(np.diff(black_hole_cmasses_quasar), 0, 0)
        
        axis00.plot(lookback_times, black_hole_masses * 1e10, c=colors[0])
        axis01.plot(lookback_times, black_hole_imasses_radio * 1e10, c=colors[1])
        axis01.plot(lookback_times, black_hole_imasses_quasar * 1e10, c=colors[2])
        plt10, = axis10.plot(lookback_times, black_hole_dmasses, c=colors[0])
        plt101, = axis10.plot(lookback_times, black_hole_dmasses_radio, c=colors[1])
        plt102, = axis10.plot(lookback_times, black_hole_dmasses_quasar, c=colors[2])
        
        # Transform the arrays to comma separated strings and convert each element to float #
        thermals = ','.join(thermals)
        mechanicals = ','.join(mechanicals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')
        mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
        
        # Calculate and plot the thermals energy sum #
        modes = [mechanicals, thermals]
        for i, mode in enumerate(modes):
            n_bins = int((max(lookback_times_modes[np.where(mode > 0)]) - min(lookback_times_modes[np.where(mode > 0)])) / 0.02)
            sum = np.empty(n_bins)
            x_value = np.empty(n_bins)
            x_low = min(lookback_times_modes[np.where(mode > 0)])
            for j in range(n_bins):
                index = np.where((lookback_times_modes[np.where(mode > 0)] >= x_low) & (lookback_times_modes[np.where(mode > 0)] < x_low + 0.02))[0]
                x_value[j] = np.mean(np.absolute(lookback_times_modes[np.where(mode > 0)])[index])
                if len(index) > 0:
                    sum[j] = np.sum(mode[np.where(mode > 0)][index])
                x_low += 0.02
            axis11.plot(x_value, sum, color=colors[1 + i], zorder=5)
        
        axis00.text(0.0, 0.9, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=axis00.transAxes)
        axis10.legend([plt10, plt102, plt101], [r'$\mathrm{BH}$', r'$\mathrm{Thermal}$', r'$\mathrm{Mechanical}$'], loc='lower right', fontsize=16,
                      frameon=False, numpoints=1)
        pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def bar_strength(pdf, data, read):
    """
    Calculate the evolution of bar strength from Fourier modes of surface density.
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'bse/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        redshift_cut = 5.0
        max_A2s = []  # Declare lists to store the data.
        # Get all available redshifts #
        haloes = data.get_haloes(level)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()
        
        for redshift in redshifts[np.where(redshifts <= redshift_cut)]:
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [4]
            attributes = ['age', 'mass', 'pos']
            data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
            
            # Loop over all haloes #
            for s in data:
                # Check if any of the haloes' data already exists, if not then read and save it #
                names = glob.glob(path + '/name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                if str(s.haloname) in names:
                    continue
                
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
                s.calc_sf_indizes(s.subfind)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
                
                mask, = np.where(s.data['age'] > 0.0)  # Mask the data: select stellar particles.
                
                # Rotate the particle positions so the bar is along the x-axis #
                z_rotated, y_rotated, x_rotated = projections.rotate_bar(s.data['pos'][mask, 0] * 1e3, s.data['pos'][mask, 1] * 1e3,
                                                                         s.data['pos'][mask, 2] * 1e3)  # Distances are in Mpc.
                s.data['pos'] = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.data['pos'] attribute in kpc.
                x, y = s.data['pos'][:, 2] * 1e3, s.data['pos'][:, 1] * 1e3  # Load positions and convert from Mpc to Kpc.
                
                # Split up galaxy in radius bins and calculate the Fourier components #
                n_bins = 40  # Number of radial bins.
                r = np.sqrt(x[:] ** 2 + y[:] ** 2)  # Radius of each particle.
                
                # Initialise Fourier components #
                r_m = np.zeros(n_bins)
                beta_2 = np.zeros(n_bins)
                alpha_0 = np.zeros(n_bins)
                alpha_2 = np.zeros(n_bins)
                
                # Calculate the Fourier components for each bin as in sec 2.3.2 from Athanassoula et al. 2013 #
                for i in range(0, n_bins):
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
                max_A2s.append(max(np.divide(np.sqrt(alpha_2[:] ** 2 + beta_2[:] ** 2), alpha_0[:])))
                
                # Save data for each halo in numpy arrays #
                np.save(path + 'name_' + str(s.haloname), s.haloname)
                np.save(path + 'max_A2s_' + str(s.haloname), max_A2s)
                np.save(path + 'redshifts_' + str(s.haloname), redshifts[np.where(redshifts <= redshift_cut)])
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_18.*')
    names.sort()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(names))))
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plt.grid(True, color='gray', linestyle='-')
        plt.ylim(0, 1)
        plt.xlabel(r'$\mathrm{Redshift}$', size=16)
        axis2 = axis.twiny()
        plot_tools.set_axis_evo(axis, axis2, ylabel=r'$\mathrm{A_{2}}$')
        
        # Load and plot the data #
        max_A2s = np.load(path + 'max_A2s_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts = np.load(path + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = satellite_utilities.return_lookbacktime_from_a((redshifts + 1.0) ** (-1.0))  # In Gyr.
        
        plt.plot(lookback_times, max_A2s, color=next(colors), label='Au-' + str(re.split('_|.npy', names[i])[1]))
        
        # Create the legends and save the figure #
        axis2.legend(loc='upper left', fontsize=16, frameon=False, numpoints=1)
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def gas_temperature_fraction(pdf, data, read):
    """
    Calculate the evolution of gas fraction in different temperature regimes.
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'gtfe/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        redshift_cut = 5.0
        sfg_ratios, wg_ratios, hg_ratios, masses = [], [], [], []  # Declare lists to store the data.
        
        # Get all available redshifts #
        haloes = data.get_haloes(level)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()
        
        for redshift in redshifts[np.where(redshifts <= redshift_cut)]:
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [0, 4]
            attributes = ['age', 'mass', 'ne', 'pos', 'rho', 'u']
            data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
            
            # Loop over all haloes #
            for s in data:
                # Check if any of the haloes' data already exists, if not then read and save it #
                names = glob.glob(path + '/name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                if str(s.haloname) in names:
                    continue
                
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
                s.calc_sf_indizes(s.subfind)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
                
                mask, = np.where(
                    (s.r() < s.subfind.data['frc2'][0]) & (s.data['type'] == 0))  # Mask the data: select gas cells within the virial radius R200 #
                
                # Calculate the temperature of the gas cells #
                ne = s.data['ne'][mask]
                metallicity = s.data['gz'][mask]
                XH = s.data['gmet'][mask, element['H']]
                yhelium = (1 - XH - metallicity) / (4. * XH)
                mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
                temperature = GAMMA_MINUS1 * s.data['u'][mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN
                
                # Calculate the mass of the gas cells within three temperatures regimes #
                mass = s.data['mass'][mask]
                masses.append(np.sum(mass))
                sfg_ratios.append(np.sum(mass[np.where((temperature < 2e4))]) / np.sum(mass))
                wg_ratios.append(np.sum(mass[np.where((temperature >= 2e4) & (temperature < 5e5))]) / np.sum(mass))
                hg_ratios.append(np.sum(mass[np.where((temperature >= 5e5))]) / np.sum(mass))
        
        # Save data for each halo in numpy arrays #
        np.save(path + 'masses_' + str(s.haloname), masses)
        np.save(path + 'name_' + str(s.haloname), s.haloname)
        np.save(path + 'sfg_ratios_' + str(s.haloname), sfg_ratios)
        np.save(path + 'wg_ratios_' + str(s.haloname), wg_ratios)
        np.save(path + 'hg_ratios_' + str(s.haloname), hg_ratios)
        np.save(path + 'redshifts_' + str(s.haloname), redshifts[np.where(redshifts <= redshift_cut)])
    
    # Load and plot the data #
    names = glob.glob(path + '/name_18.*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plt.grid(True, color='gray', linestyle='-')
        plt.ylim(-0.2, 1.2)
        plt.ylabel(r'$\mathrm{Gas\;fraction}$', size=16)
        plt.xlabel(r'$\mathrm{Redshift}$', size=16)
        axis.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=axis.transAxes)
        axis2 = axis.twiny()
        plot_tools.set_axis_evo(axis, axis2, ylabel=r'$\mathrm{A_{2}}$')
        
        sfg_ratios = np.load(path + 'sfg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        wg_ratios = np.load(path + 'wg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        hg_ratios = np.load(path + 'hg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts = np.load(path + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = satellite_utilities.return_lookbacktime_from_a((redshifts + 1.0) ** (-1.0))  # In Gyr.
        
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
        
        b1, = plt.plot(lookback_times, sfg_ratios, color='blue')
        b2, = plt.plot(lookback_times, wg_ratios, color='green')
        b3, = plt.plot(lookback_times, hg_ratios, color='red')
    
    # Create the legends and save the figure #
    plt.legend([b3, b2, b1], [r'$\mathrm{Hot\;gas}$', r'$\mathrm{Warm\;gas}$', r'$\mathrm{Cold\;gas}$'], loc='upper left', fontsize=16, frameon=False,
               numpoints=1)
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def AGN_modes_cumulative(date, data, read):
    """
    Get information about different black hole modes from log files and plot the evolution of the cumulative feedback.
    :param date: .
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmc/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        data.select_haloes(level, 0, loadonlytype=None, loadonlyhalo=0, loadonly=None)
        
        # Loop over all haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            
            # Declare arrays to store the desired words and lines that contain these words #
            redshift_lines, redshifts, feedback_lines, thermals, mechanicals = [], [], [], [], []
            
            with open('/u/di43/Auriga/output/halo_' + str(s.haloname) + '/halo_' + str(s.haloname) + '.txt') as file:
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
    names = glob.glob(path + '/name_06NOR*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(1, 2, wspace=0, width_ratios=[1, 0.05])
        axis00 = plt.subplot(gs[0, 0])
        axiscbar = plt.subplot(gs[:, 1])
        
        axis00.grid(True, color='gray', linestyle='-')
        axis00.set_xscale('log')
        # axis00.set_yscale('log')
        axis00.set_aspect('equal')
        axis00.set_xlim(1e54, 1e62)
        axis00.set_ylim(-1, 1)
        axis00.tick_params(direction='out', which='both', right='on', left='on')
        axis00.set_xlabel(r'$\mathrm{Cumulative\;thermal\;feedback\;energy\;[ergs]}$', size=16)
        axis00.set_ylabel(r'$\mathrm{Cumulative\;mechanical\;feedback\;energy\;[ergs]}$', size=16)
        figure.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=axis00.transAxes)
        
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
        
        lookback_times = satellite_utilities.return_lookbacktime_from_a((redshifts + 1.0) ** (-1.0))  # In Gyr.
        
        # Mask the data and plot the scatter #
        plot000 = axis00.plot([1e54, 1e62], [1e54 / 10, 1e62 / 10])
        plot001 = axis00.plot([1e54, 1e62], [1e54 / 50, 1e62 / 50])
        
        mask, = np.where((mechanicals != 0) | (thermals != 0))
        sc = axis00.scatter(thermals[mask], mechanicals[mask], edgecolor='None', s=50, c=lookback_times[mask], vmin=0, vmax=max(lookback_times),
                            cmap='jet')
        cb = plt.colorbar(sc, cax=axiscbar)
        cb.set_label(r'$\mathrm{t_{look}\;[Gyr]}$', size=16)
        axiscbar.tick_params(direction='out', which='both', right='on', left='on')
        axiscbar.yaxis.tick_left()
        
        # Create the legends and save the figure #
        axis00.legend([plot000, plot001], [r'$\mathrm{1:10}$', r'$\mathrm{1:50}$'], loc='upper center', fontsize=16, frameon=False, numpoints=1)
        axis00.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmc-' + date + '.png', bbox_inches='tight')
        plt.close()
    return None


def AGN_modes_histogram(date, data, read):
    """
    Get information about different black hole modes from log files and plot a histogram of the evolution of the step feedback.
    :param date: .
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmh/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        
        data.select_haloes(level, 0, loadonlytype=None, loadonlyhalo=0, loadonly=None)
        
        # Loop over all haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            
            # Declare arrays to store the desired words and lines that contain these words #
            redshift_lines, redshifts, feedback_lines, thermals, mechanicals = [], [], [], [], []
            
            with open('/u/di43/Auriga/output/halo_' + str(s.haloname) + '/halo_' + str(s.haloname) + '.txt') as file:
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
    names = glob.glob(path + '/name_06NOR*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis.grid(True, color='gray', linestyle='-')
        axis2 = axis.twiny()
        axis.set_yscale('log')
        plot_tools.set_axis_evo(axis, axis2, ylabel=r'$\mathrm{AGN\;feedback\;energy\;[ergs]}$')
        axis.text(0.01, 0.95, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=axis.transAxes)
        
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
        lookback_times = satellite_utilities.return_lookbacktime_from_a((redshifts + 1.0) ** (-1.0))  # In Gyr.
        hist0 = axis.hist(lookback_times, weights=thermals, histtype='step', bins=100)
        hist1 = axis.hist(lookback_times, weights=mechanicals, histtype='step', bins=100)
        
        # Create the legends and save the figure #
        axis.legend([hist0, hist1], [r'$\mathrm{Thermal}$', r'$\mathrm{Mechanical}$'], loc='upper right', fontsize=16, frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmh-' + date + '.png', bbox_inches='tight')
        plt.close()
    
    return None


def AGN_modes_distribution(date, data, read):
    """
    Get information about different black hole modes from log files and plot the evolution of the step feedback.
    :param date: .
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        data.select_haloes(level, 0, loadonlytype=None, loadonlyhalo=0, loadonly=None)
        
        # Loop over all haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            
            # Declare arrays to store the desired words and lines that contain these words #
            redshift_lines, redshifts, feedback_lines, thermals, mechanicals = [], [], [], [], []
            
            with open('/u/di43/Auriga/output/halo_' + str(s.haloname) + '/halo_' + str(s.haloname) + '.txt') as file:
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
            
            #  Convert redshifts to lookback times in Gyr #
            redshifts = [re.sub(',', '', i) for i in redshifts]  # Remove the commas at the end of each redshift string.
            redshifts = ','.join(redshifts)  # Transform the arrays to comma separated strings.
            redshifts = np.fromstring(redshifts, dtype=np.float, sep=',')  # Convert each element to float.
            lookback_times = satellite_utilities.return_lookbacktime_from_a((redshifts + 1.0) ** (-1.0))  # In Gyr.
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'thermals_' + str(s.haloname), thermals)
            np.save(path + 'mechanicals_' + str(s.haloname), mechanicals)
            np.save(path + 'lookback_times_' + str(s.haloname), lookback_times)
    
    # Load and plot the data #
    names = glob.glob(path + '/name_06NOR*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.1, height_ratios=[0.05, 1])
        axiscbar = plt.subplot(gs[0, 0])
        axis00 = plt.subplot(gs[1, 0])
        axiscbar2 = plt.subplot(gs[0, 1])
        axis02 = plt.subplot(gs[1, 1])
        
        axis02.yaxis.set_label_position("right")
        axis02.yaxis.tick_right()
        for axis in [axis00, axis02]:
            axis.grid(True, color='gray', linestyle='-')
            axis.set_xlim(12, 0)
            axis.set_yscale('log')
            axis.set_ylim(1e51, 1e60)
            axis.set_xlabel(r'$\mathrm{t_{look}\;[Gyr]}$', size=16)
            axis.tick_params(direction='out', which='both', right='on', left='on', labelsize=16)
        axis00.set_ylabel(r'$\mathrm{Mechanical\;feedback\;energy\;[ergs]}$', size=16)
        axis02.set_ylabel(r'$\mathrm{Thermal\;feedback\;energy\;[ergs]}$', size=16)
        figure.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=axis00.transAxes)
        
        # Load and plot the data #
        thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Transform the arrays to comma separated strings and convert each element to float #
        thermals = ','.join(thermals)
        mechanicals = ','.join(mechanicals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')
        mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
        
        # Plot hexbins #
        hb = axis00.hexbin(lookback_times[np.where(mechanicals > 0)], mechanicals[np.where(mechanicals > 0)], yscale='log', cmap='gist_heat_r',
                           gridsize=(100, 50))
        cb = plt.colorbar(hb, cax=axiscbar, orientation='horizontal')
        cb.set_label(r'$\mathrm{Counts\;per\;hexbin}$', size=16)
        
        hb = axis02.hexbin(lookback_times[np.where(thermals > 0)], thermals[np.where(thermals > 0)], yscale='log', cmap='gist_heat_r')
        # gridsize=(100, 50 * np.int(len(np.where(thermals > 0)[0]) / len(np.where(mechanicals > 0)[0]))))
        
        cb2 = plt.colorbar(hb, cax=axiscbar2, orientation='horizontal')
        cb2.set_label(r'$\mathrm{Counts\;per\;hexbin}$', size=16)
        
        for axis in [axiscbar, axiscbar2]:
            axis.xaxis.tick_top()
            axis.xaxis.set_label_position("top")
            axis.tick_params(direction='out', which='both', top='on', right='on')
        
        # # Calculate and plot the mechanical energy sum #
        # n_bins = int((max(lookback_times[np.where(mechanicals > 0)]) - min(lookback_times[np.where(mechanicals > 0)])) / 0.02)
        # sum = np.empty(n_bins)
        # x_value = np.empty(n_bins)
        # x_low = min(lookback_times[np.where(mechanicals > 0)])
        # for j in range(n_bins):
        #     index = np.where((lookback_times[np.where(mechanicals > 0)] >= x_low) & (lookback_times[np.where(mechanicals > 0)] < x_low + 0.02))[0]
        #     x_value[j] = np.mean(np.absolute(lookback_times[np.where(mechanicals > 0)])[index])
        #     if len(index) > 0:
        #         sum[j] = np.sum(mechanicals[np.where(mechanicals > 0)][index])
        #     x_low += 0.02
        #
        # sum00, = axis00.plot(x_value, sum, color='black', zorder=5)
        
        # Calculate and plot the mechanical energy sum #
        n_bins = int((max(lookback_times[np.where(thermals > 0)]) - min(lookback_times[np.where(thermals > 0)])) / 0.02)
        sum = np.empty(n_bins)
        x_value = np.empty(n_bins)
        x_low = min(lookback_times[np.where(thermals > 0)])
        for j in range(n_bins):
            index = np.where((lookback_times[np.where(thermals > 0)] >= x_low) & (lookback_times[np.where(thermals > 0)] < x_low + 0.02))[0]
            x_value[j] = np.mean(np.absolute(lookback_times[np.where(thermals > 0)])[index])
            if len(index) > 0:
                sum[j] = np.sum(thermals[np.where(thermals > 0)][index])
            x_low += 0.02
        
        # Plot sum #
        sum02, = axis02.plot(x_value, sum, color='black', zorder=5)
        
        # Create the legends and save the figure #
        # axis00.legend([sum00], [r'$\mathrm{Sum}$'], loc='upper left', fontsize=16, frameon=False, numpoints=1)
        axis02.legend([sum02], [r'$\mathrm{Sum}$'], loc='upper left', fontsize=16, frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmd-' + date + '.png', bbox_inches='tight')
        plt.close()
    
    return None


def AGN_modes_step(date, data, read):
    """
    Get information about different black hole modes from log files and plot the step feedbacks.
    :param date: .
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNms/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        data.select_haloes(level, 0, loadonlytype=None, loadonlyhalo=0, loadonly=None)
        
        # Loop over all haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            
            # Declare arrays to store the desired words and lines that contain these words #
            feedback_lines, thermals, mechanicals = [], [], []
            
            with open('/u/di43/Auriga/output/halo_' + str(s.haloname) + '/halo_' + str(s.haloname) + '.txt') as file:
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
    names = glob.glob(path + '/name_17NOR*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(10, 7.5))
        gs = plt.GridSpec(2, 2, hspace=0.07, wspace=0.07, height_ratios=[0.5, 1], width_ratios=[1, 0.5])
        axis00 = figure.add_subplot(gs[0, 0])
        axis10 = figure.add_subplot(gs[1, 0])
        axis11 = figure.add_subplot(gs[1, 1])
        
        for axis in [axis00, axis10, axis11]:
            axis.grid(True, color='gray', linestyle='-')
            axis.tick_params(direction='out', which='both', right='on', left='on')
        
        for axis in [axis00, axis11]:
            axis.yaxis.set_ticks_position('left')
            axis.xaxis.set_ticks_position('bottom')
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
        
        axis00.set_xticklabels([])
        axis11.set_yticklabels([])
        
        axis10.set_xlim(0, 4e55)  # 06:4e55 17:2e56 18:2e56
        axis10.set_ylim(0, 2e57)  # 06:6e56 17:5e57 18:5e57
        axis11.set_xscale('log')
        axis00.set_yscale('symlog', subsy=[2, 3, 4, 5, 6, 7, 8, 9], linthreshy=1)
        axis00.set_ylabel(r'$\mathrm{PDF}$', size=16)
        axis11.set_xlabel(r'$\mathrm{PDF}$', size=16)
        axis10.set_ylabel(r'$\mathrm{(Thermal\;feedback\;energy)/ergs}$', size=16)
        axis10.set_xlabel(r'$\mathrm{(Mechanical\;feedback\;energy)/ergs}$', size=16)
        figure.text(1.01, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=axis10.transAxes)
        
        # Load and plot the data #
        thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Transform the arrays to comma separated strings and convert each element to float #
        thermals = ','.join(thermals)
        mechanicals = ','.join(mechanicals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')
        mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
        
        # Plot the scatter and the axes histograms #
        axis10.scatter(mechanicals, thermals, s=50, edgecolor='none', c='k', marker='1')
        weights = np.ones_like(mechanicals) / float(len(mechanicals))
        axis00.hist(mechanicals, bins=np.linspace(0, 2e56, 100), histtype='step', weights=weights, orientation='vertical', color='k')
        weights = np.ones_like(thermals) / float(len(thermals))
        axis11.hist(thermals, bins=np.linspace(0, 5e57, 100), histtype='step', weights=weights, orientation='horizontal', color='k')
        
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
    names = glob.glob(path_modes + '/name_06NOR*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plt.grid(True, color='gray', linestyle='-')
        axis.set_xlim(12, 0)
        axis.set_ylim(-0.2, 1.2)
        axis2 = axis.twinx()
        axis2.set_yscale('log')
        axis2.set_ylim(1e56, 1e60)
        axis.set_ylabel(r'$\mathrm{Gas\;fraction}$', size=16)
        axis.set_xlabel(r'$\mathrm{t_{look}\;[Gyr]}$', size=16)
        axis2.set_ylabel(r'$\mathrm{AGN\;feedback\;energy\;[ergs]}$', size=16)
        axis.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=axis.transAxes)
        
        # Load and plot the data #
        wg_ratios = np.load(path_gas + 'wg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        hg_ratios = np.load(path_gas + 'hg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sfg_ratios = np.load(path_gas + 'sfg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts_gas = np.load(path_gas + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        thermals = np.load(path_modes + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        mechanicals = np.load(path_modes + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path_modes + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Transform the arrays to comma separated strings and convert each element to float #
        thermals = ','.join(thermals)
        mechanicals = ','.join(mechanicals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')
        mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
        lookback_times_gas = satellite_utilities.return_lookbacktime_from_a((redshifts_gas + 1.0) ** (-1.0))  # In Gyr.
        
        # Plot the gas fractions #
        plot1, = axis.plot(lookback_times_gas, sfg_ratios, color='blue')
        plot1, = axis.plot(lookback_times_gas, wg_ratios, color='green')
        plot3, = axis.plot(lookback_times_gas, hg_ratios, color='red')
        
        # Calculate and plot the thermals energy sum #
        n_bins = int((max(lookback_times[np.where(thermals > 0)]) - min(lookback_times[np.where(thermals > 0)])) / 0.02)
        sum = np.empty(n_bins)
        x_value = np.empty(n_bins)
        x_low = min(lookback_times[np.where(thermals > 0)])
        for j in range(n_bins):
            index = np.where((lookback_times[np.where(thermals > 0)] >= x_low) & (lookback_times[np.where(thermals > 0)] < x_low + 0.02))[0]
            x_value[j] = np.mean(np.absolute(lookback_times[np.where(thermals > 0)])[index])
            if len(index) > 0:
                sum[j] = np.sum(thermals[np.where(thermals > 0)][index])
            x_low += 0.02
        
        sum00, = plt.plot(x_value, sum, color='black', zorder=5, linestyle='dashed')
        
        # Calculate and plot the mechanical energy sum #
        n_bins = int((max(lookback_times[np.where(mechanicals > 0)]) - min(lookback_times[np.where(mechanicals > 0)])) / 0.02)
        sum = np.empty(n_bins)
        x_value = np.empty(n_bins)
        x_low = min(lookback_times[np.where(mechanicals > 0)])
        for j in range(n_bins):
            index = np.where((lookback_times[np.where(mechanicals > 0)] >= x_low) & (lookback_times[np.where(mechanicals > 0)] < x_low + 0.02))[0]
            x_value[j] = np.mean(np.absolute(lookback_times[np.where(mechanicals > 0)])[index])
            if len(index) > 0:
                sum[j] = np.sum(mechanicals[np.where(mechanicals > 0)][index])
            x_low += 0.02
        
        sum02, = plt.plot(x_value, sum, color='black', zorder=5)
        
        # Create the legends and save the figure #
        axis.legend([plot3, plot2, plot1], [r'$\mathrm{Hot\;gas}$', r'$\mathrm{Warm\;gas}$', r'$\mathrm{Cold\;gas}$'], loc='upper left', fontsize=16,
                    frameon=False, numpoints=1)
        axis2.legend([sum00, sum02], [r'$\mathrm{Thermal}$', r'$\mathrm{Mechanical}$'], loc='upper center', fontsize=16, frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmg-' + date + '.png', bbox_inches='tight')
        plt.close()
    
    return None


def gas_stars_sfr(pdf, data, read):
    """
    Plot the evolution of gas mass, stellar mass and star formation rate inside 500, 750 and 1000 pc.
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    radial_limits = (5e-4, 7.5e-4, 1e-3)
    # Check if a folder to save the data exists, if not create one #
    for radial_limit in radial_limits:
        path = '/u/di43/Auriga/plots/data/' + 'gsse/' + str(radial_limit) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
    
    # Read the data #
    if read is True:
        particle_type = [4]
        attributes = ['age', 'mass', 'pos']
        data.select_haloes(level, 0, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
        
        # Loop over all haloes #
        for s in data:
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            
            # Loop over different radial limits #
            for radial_limit in radial_limits:
                path = '/u/di43/Auriga/plots/data/' + 'gsse/' + str(radial_limit) + '/'
                
                spherical_distance = np.max(np.abs(s.data['pos'] - s.center[None, :]), axis=1)
                stellar_mask, = np.where((spherical_distance < radial_limit) & (
                    s.data['age'] > 0))  # Mask the data: select stellar particles within a 500pc radius sphere.
                
                time_bin = (13 - 0) / 100
                age = s.cosmology_get_lookback_time_from_a(s.data['age'][stellar_mask], is_flat=True)
                weights = s.data['gima'][stellar_mask] * 1e10 / 1e9 / time_bin
                
                # Save data for each halo in numpy arrays #
                np.save(path + 'age_' + str(s.haloname), age)
                np.save(path + 'weights_' + str(s.haloname), weights)
                np.save(path + 'name_' + str(s.haloname), s.haloname)
        
        redshift_cut = 7.0
        gas_masses, stellar_masses, sfg_ratios, wg_ratios, hg_ratios = [], [], [], [], []  # Declare lists to store the data.
        # Get all available redshifts #
        haloes = data.get_haloes(level)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()
        
        for redshift in redshifts[np.where(redshifts <= redshift_cut)]:
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [0, 4]
            attributes = ['age', 'mass', 'pos']
            data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
            
            # Loop over all haloes #
            for s in data:
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
                s.calc_sf_indizes(s.subfind)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
                
                # Loop over different radial limits #
                for radial_limit in radial_limits:
                    path = '/u/di43/Auriga/plots/data/' + 'gsse/' + str(radial_limit) + '/'
                    
                    # Mask the data: select stellar particles and gas cells within a 500pc radius sphere #
                    age = np.zeros(s.npartall)
                    age[s.data['type'] == 4] = s.data['age']
                    spherical_distance = np.max(np.abs(s.data['pos'] - s.center[None, :]), axis=1)
                    gas_mask, = np.where((spherical_distance < radial_limit) & (s.data['type'] == 0))
                    stellar_mask, = np.where((spherical_distance < radial_limit) & (s.data['type'] == 4) & (age > 0))
                    
                    # Calculate the stellar and gas mass of each halo #
                    gas_mass = s.data['mass'][gas_mask]
                    stellar_mass = s.data['mass'][stellar_mask]
                    gas_masses.append(np.sum(gas_mass))
                    stellar_masses.append(np.sum(stellar_mass))
                    
                    # Calculate the temperature of the gas cells #
                    ne = s.data['ne'][gas_mask]
                    metallicity = s.data['gz'][gas_mask]
                    XH = s.data['gmet'][gas_mask, element['H']]
                    yhelium = (1 - XH - metallicity) / (4. * XH)
                    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
                    temperature = GAMMA_MINUS1 * s.data['u'][gas_mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN
                    
                    # Calculate the mass of the gas cells within three temperatures regimes #
                    sfg_ratios.append(np.sum(gas_mass[np.where((temperature < 2e4))]) / np.sum(gas_mass))
                    wg_ratios.append(np.sum(gas_mass[np.where((temperature >= 2e4) & (temperature < 5e5))]) / np.sum(gas_mass))
                    hg_ratios.append(np.sum(gas_mass[np.where((temperature >= 5e5))]) / np.sum(gas_mass))
        
        # Save data for each halo in numpy arrays #
        lookback_times = satellite_utilities.return_lookbacktime_from_a((redshifts[np.where(redshifts <= redshift_cut)] + 1.0) ** (-1.0))  # In Gyr.
        np.save(path + 'wg_ratios_' + str(s.haloname), wg_ratios)
        np.save(path + 'hg_ratios_' + str(s.haloname), hg_ratios)
        np.save(path + 'sfg_ratios_' + str(s.haloname), sfg_ratios)
        np.save(path + 'gas_masses_' + str(s.haloname), gas_masses)
        np.save(path + 'stellar_masses_' + str(s.haloname), stellar_masses)
        np.save(path + 'lookback_times_' + str(s.haloname), lookback_times)
    
    radial_limits = (5e-4, 7.5e-4, 1e-3)
    for radial_limit in radial_limits:
        path = '/u/di43/Auriga/plots/data/' + 'gsse/' + str(radial_limit) + '/'
        path_modes = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
        
        # Get the names and sort them #
        names = glob.glob(path + '/name_18NOA*')
        names.sort()
        
        # Loop over all available haloes #
        for i in range(len(names)):
            # Generate the figure and define its parameters #
            figure = plt.figure(figsize=(10, 7.5))
            gs = gridspec.GridSpec(3, 1, hspace=0.06, height_ratios=[1, 0.5, 0.5])
            axis00 = plt.subplot(gs[0, 0])
            axis10 = plt.subplot(gs[1, 0])
            axis20 = plt.subplot(gs[2, 0])
            for axis in [axis00, axis10, axis20]:
                axis.grid(True, color='gray', linestyle='-')
                axis.tick_params(direction='out', which='both', top='on', right='on')
            axis00.set_ylim(0, 15)
            axis00.set_xticklabels([])
            axis002 = axis00.twinx()
            axis002.set_yscale('log')
            axis002.set_ylim(1e55, 1e61)
            axis002.set_ylabel(r'$\mathrm{Feedback\;energy\;[ergs]}$', size=16)
            axis003 = axis00.twiny()
            plot_tools.set_axis_evo(axis00, axis003, ylabel=r'$\mathrm{Sfr\;[M_\odot\;yr^{-1}]}$')
            axis10.set_xlim(13, 0)
            axis10.set_yscale('log')
            axis10.set_ylim(1e6, 1e11)
            axis10.set_xticklabels([])
            axis10.set_ylabel(r'$\mathrm{Mass\;[M_{\odot}]}$', size=16)
            axis10.set_ylabel(r'$\mathrm{Mass\;[M_{\odot}]}$', size=16)
            axis20.set_xlim(13, 0)
            axis20.set_ylim(-0.1, 1.1)
            axis20.set_xlabel(r'$\mathrm{t_{look}\;[Gyr]}$', size=16)
            
            figure.text(0.01, 0.85, r'$\mathrm{Au-%s}$' '\n' r'$\mathrm{r\;<\;%.0f\;pc}$' % (
                str(re.split('_|.npy', names[i])[1]), (np.float(radial_limit) * 1e6)), fontsize=16, transform=axis00.transAxes)
            
            # Load and plot the data #
            # thermals = np.load(path_modes + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            # mechanicals = np.load(path_modes + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            # lookback_times_modes = np.load(path_modes + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            age = np.load(path + 'age_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            weights = np.load(path + 'weights_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            gas_masses = np.load(path + 'gas_masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            stellar_masses = np.load(path + 'stellar_masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            wg_ratios = np.load(path + 'wg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            hg_ratios = np.load(path + 'hg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            sfg_ratios = np.load(path + 'sfg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            
            # Transform the arrays to comma separated strings and convert each element to float #
            # thermals = ','.join(thermals)
            # mechanicals = ','.join(mechanicals)
            # thermals = np.fromstring(thermals, dtype=np.float, sep=',')
            # mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
            
            axis00.hist(age, weights=weights, histtype='step', bins=100, range=[0, 13], edgecolor='k')  # Plot the sfr history.
            
            # Calculate and plot the thermals energy sum #
            # n_bins = int((max(lookback_times_modes[np.where(thermals > 0)]) - min(lookback_times_modes[np.where(thermals > 0)])) / 0.02)
            # sum = np.empty(n_bins)
            # x_value = np.empty(n_bins)
            # x_low = min(lookback_times_modes[np.where(thermals > 0)])
            # for j in range(n_bins):
            #     index = \
            #         np.where((lookback_times_modes[np.where(thermals > 0)] >= x_low) & (lookback_times_modes[np.where(thermals > 0)] < x_low +
            #         0.02))[
            #             0]
            #     x_value[j] = np.mean(np.absolute(lookback_times_modes[np.where(thermals > 0)])[index])
            #     if len(index) > 0:
            #         sum[j] = np.sum(thermals[np.where(thermals > 0)][index])
            #     x_low += 0.02
            # plot20, = axis002.plot(x_value, sum, color='orange', zorder=5)
            
            # # Calculate and plot the mechanical energy sum #
            # n_bins = int((max(lookback_times_modes[np.where(mechanicals > 0)]) - min(lookback_times_modes[np.where(mechanicals > 0)])) / 0.02)
            # sum = np.empty(n_bins)
            # x_value = np.empty(n_bins)
            # x_low = min(lookback_times_modes[np.where(mechanicals > 0)])
            # for j in range(n_bins):
            #     index = np.where(
            #         (lookback_times_modes[np.where(mechanicals > 0)] >= x_low) & (lookback_times_modes[np.where(mechanicals > 0)] < x_low +
            #         0.02))[0]
            #     x_value[j] = np.mean(np.absolute(lookback_times_modes[np.where(mechanicals > 0)])[index])
            #     if len(index) > 0:
            #         sum[j] = np.sum(mechanicals[np.where(mechanicals > 0)][index])
            #     x_low += 0.02
            # plot21, = axis002.plot(x_value, sum, color='magenta', zorder=5)
            
            # Plot the stellar and gaseous masses #
            plot100, = axis10.plot(lookback_times, gas_masses * 1e10, c='b')
            plot101, = axis10.plot(lookback_times, stellar_masses * 1e10, c='r')
            
            # Plot the gas fractions #
            plot1, = axis20.plot(lookback_times, sfg_ratios, color='blue')
            plot2, = axis20.plot(lookback_times, wg_ratios, color='green')
            plot3, = axis20.plot(lookback_times, hg_ratios, color='red')
            
            # Create the legends and save the figure #
            axis10.legend([plot100, plot101], [r'$\mathrm{Gas}$', r'$\mathrm{Stars}$'], loc='lower center', fontsize=16, frameon=False, numpoints=1,
                          ncol=2)
            # axis002.legend([plot20], [r'$\mathrm{Thermal}$'], loc='upper center', fontsize=16, frameon=False, numpoints=1, ncol=2)
            axis00.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1, ncol=2)
            axis20.legend([plot3, plot2, plot1], [r'$\mathrm{Hot\;gas}$', r'$\mathrm{Warm\;gas}$', r'$\mathrm{Cold\;gas}$'], loc='upper left',
                          fontsize=16, frameon=False, numpoints=1)
            
            pdf.savefig(figure, bbox_inches='tight')
            plt.close()
    return None


def AGN_feedback_kernel(pdf, data, redshift, read, ds):
    """
    Calculate the energy deposition on gas cells from the AGN feedback.
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :param ds: boolean to downsample halo_18_3000 data.
    :return:
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNfk/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        redshift_cut = 7
        gas_volumes, sf_gas_volumes, nsf_gas_volumes, redshifts_mask, blackhole_hsmls = [], [], [], [], []  # Declare lists to store the data.
        
        # Get all available redshifts #
        haloes = data.get_haloes(level)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()
        
        if ds is True:
            # Downsample the 3000 snapshots run #
            for name, halo in haloes.items():
                if name == '18':
                    redshifts = halo.get_redshifts()
        
        for redshift in redshifts[np.where(redshifts <= redshift_cut)]:
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [0, 4, 5]
            attributes = ['age', 'bhhs', 'id', 'mass', 'pos', 'sfr', 'vol']
            data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
            
            # Loop over all haloes #
            for s in data:
                # Check if any of the haloes' data already exists, if not then read and save it #
                names = glob.glob(path + '/name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                if str(s.haloname) in names:
                    continue
                
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
                s.calc_sf_indizes(s.subfind)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
                spherical_distance = np.max(np.abs(s.data['pos'] - s.center[None, :]), axis=1)
                
                # Mask the data: select the black hole inside the halo #
                blackhole_mask, = np.where((s.data['type'] == 5) & (spherical_distance < 0.1 * s.subfind.data['frc2'][0]))
                
                # Check that only one (the closest) black hole is selected #
                if len(blackhole_mask) == 1:
                    blackhole_hsml = s.data['bhhs'][0]
                # In mergers select the most massive black hole #
                elif len(blackhole_mask) > 1:
                    blackhole_id = s.data['id'][s.data['type'] == 5]
                    id_mask, = np.where(blackhole_id == s.data['id'][s.data['mass'].argmax()])
                    blackhole_hsml = s.data['bhhs'][id_mask[0]]
                else:
                    continue
                
                # Mask the data: select star-forming and not gas cells within the black hole's radius #
                gas_mask, = np.where(spherical_distance[s.data['type'] == 0] < blackhole_hsml)
                sf_gas_mask, = np.where((s.data['sfr'] > 0.0) & (spherical_distance[s.data['type'] == 0] < blackhole_hsml))
                nsf_gas_mask, = np.where((s.data['sfr'] <= 0.0) & (spherical_distance[s.data['type'] == 0] < blackhole_hsml))
                
                # Compute the total volume of cells with SFR == 0 and compare it to the total volume of all cells within this
                redshifts_mask.append(redshift)
                blackhole_hsmls.append(blackhole_hsml)
                gas_volumes.append(s.data['vol'][gas_mask].sum() * 1e9)  # In kpc^-3.
                sf_gas_volumes.append(s.data['vol'][sf_gas_mask].sum() * 1e9)  # In kpc^-3.
                nsf_gas_volumes.append(s.data['vol'][nsf_gas_mask].sum() * 1e9)  # In kpc^-3.
        
        # Save data for each halo in numpy arrays #
        if ds is True:
            name = str(s.haloname) + '_ds' + str(len(redshifts))
        else:
            name = str(s.haloname)
        
        lookback_times = satellite_utilities.return_lookbacktime_from_a([(z + 1) ** (-1.0) for z in redshifts_mask])  # In Gyr.
        np.save(path + 'name_' + name, s.haloname)
        np.save(path + 'gas_volumes_' + name, gas_volumes)
        np.save(path + 'sf_gas_volumes_' + name, sf_gas_volumes)
        np.save(path + 'lookback_times_' + name, lookback_times)
        np.save(path + 'nsf_gas_volumes_' + name, nsf_gas_volumes)
        np.save(path + 'blackhole_hsmls_' + name, blackhole_hsmls)
    
    # Load and plot the data #
    names = glob.glob(path + '/name_18.*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        axis3 = axis.twinx()
        axis3.yaxis.label.set_color('red')
        axis3.spines['right'].set_color('red')
        axis3.tick_params(axis='y', direction='out', colors='red')
        plot_tools.set_axis(axis, ylim=[0, 1], xlabel=r'$\mathrm{t_{look}/Gyr}$', ylabel=r'$\mathrm{V_{nSFR}(r<BH_{sml})/V_{all}(r<BH_{sml})}$',
                            aspect=None)
        plot_tools.set_axis(axis3, ylim=[0, 1], xlabel=r'$\mathrm{t_{look}/Gyr}$', ylabel=r'$\mathrm{BH_{sml}/kpc}$', aspect=None)
        plot_tools.set_axis_evo(axis, axis2, ylabel=r'$\mathrm{V_{nSFR}(r<BH_{sml})/V_{all}(r<BH_{sml})}$')
        figure.text(0.0, 0.95, 'Au-' + str(re.split('_|.npy', names[i])[1]), color='k', fontsize=16, transform=axis.transAxes)
        
        # Load and plot the data #
        gas_volumes = np.load(path + 'gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        nsf_gas_volumes = np.load(path + 'nsf_gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        blackhole_hsmls = np.load(path + 'blackhole_hsmls_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        axis3.scatter(lookback_times, blackhole_hsmls * 1e3, c=colors[1], edgecolor='None')
        axis.scatter(lookback_times, nsf_gas_volumes / gas_volumes, c=colors[0], edgecolor='None')
        
        # Plot median and 1-sigma lines #
        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(lookback_times, nsf_gas_volumes / gas_volumes, bin_width=1, log=False)
        axis.plot(x_value, median, color=colors[0], linewidth=3, zorder=5)
        axis.fill_between(x_value, shigh, slow, color=colors[0], alpha='0.3', zorder=5)
        
        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(lookback_times, blackhole_hsmls * 1e3, bin_width=1, log=False)
        axis3.plot(x_value, median, color=colors[1], linewidth=3, zorder=5)
        axis3.fill_between(x_value, shigh, slow, color=colors[1], alpha='0.3', zorder=5)
        
        pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def AGN_feedback_smoothed(pdf):
    """
    Calculate the energy deposition on gas cells from the AGN feedback.
    :param pdf: path to save the pdf from main.make_pdf
    :return:
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNfs/'
    path_modes = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    path_kernel = '/u/di43/Auriga/plots/data/' + 'AGNfk/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Load and plot the data #
    names = glob.glob(path_kernel + '/name_18.*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        plot_tools.set_axis(axis, ylim=[1e52, 1e62], yscale='log', xlabel=r'$\mathrm{t_{look}/Gyr}$',
                            ylabel=r'$\mathrm{(AGN\;feedback\;energy)/ergs}$', aspect=None)
        plot_tools.set_axis_evo(axis, axis2, ylabel=r'$\mathrm{(AGN\;feedback\;energy)/ergs}$')
        figure.text(0.0, 0.95, 'Au-' + str(re.split('_|.npy', names[i])[1]), color='k', fontsize=16, transform=axis.transAxes)
        
        # Load and plot the data #
        thermals = np.load(path_modes + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        gas_volumes = np.load(path_kernel + 'gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path_kernel + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        nsf_gas_volumes = np.load(path_kernel + 'nsf_gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times_modes = np.load(path_modes + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Transform the arrays to comma separated strings and convert each element to float #
        thermals = ','.join(thermals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')
        
        # Calculate and plot the thermals energy sum #
        y_data, edges = np.histogram(lookback_times_modes[np.where(thermals > 0)], weights=thermals[np.where(thermals > 0)],
                                     bins=np.sort(lookback_times))
        y_data /= edges[1:] - edges[:-1]  # Normalise the values wrt the bin width.
        axis.plot(0.5 * (edges[1:] + edges[:-1]), y_data, color=colors[1])
        axis.plot(0.5 * (edges[1:] + edges[:-1]), y_data * np.flip(nsf_gas_volumes[1:] / gas_volumes[1:]), color=colors[4])
        # axis2.hist(lookback_times_modes[np.where(thermals > 0)], weights=thermals[np.where(thermals > 0)], histtype='step',
        #            bins=lookback_times.sort())
        
        pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None
