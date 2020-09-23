import os
import re
import glob
import plot_tools

import numpy as np
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from loadmodules import *
from matplotlib import gridspec
from parallel_decorators import vectorize_parallel
from scripts.gigagalaxy.util import satellite_utilities

res = 512
boxsize = 0.06
default_level = 4
default_redshift = 0.0
colors = ['black', 'tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']
element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}


def sfr(pdf, data, read):
    """
    Plot the evolution of star formation rate for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking sfr")
    n_bins = 130
    time_bin_width = (13 - 0) / n_bins  # In Gyr.
    path = '/u/di43/Auriga/plots/data/' + 'sh/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'gima', 'mass', 'pos']
        data.select_haloes(default_level, default_redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then create it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))

            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            # Get the lookback times and calculate the initial masses #
            stellar_mask, = np.where(
                (s.data['age'] > 0.) & (s.r() < 0.1 * s.subfind.data['frc2'][0]))  # Mask the data: select stellar particles inside 0.1*R200c.
            lookback_times = s.cosmology_get_lookback_time_from_a(s.data['age'][stellar_mask], is_flat=True)  # In Gyr.
            weights = s.data['gima'][stellar_mask] * 1e10 / 1e9 / time_bin_width  # In Msun yr^-1.

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'weights_' + str(s.haloname), weights)
            np.save(path + 'lookback_times_' + str(s.haloname), lookback_times)

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[0, 45], ylabel='$\mathrm{Sfr/(M_\odot\;yr^{-1})}$', aspect=None)
        figure.text(0.0, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis.transAxes)

        # Load the data #
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        weights = np.load(path + 'weights_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        axis.hist(lookback_times, weights=weights, histtype='step', bins=n_bins, range=[0, 13], color=colors[0])  # Plot the evolution of SFR.

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_bs_data(snapshot_ids, halo):
    """
    Parallelised method to get bar strength from Fourier modes of surface density.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :return: lookback time, max_A2s.
    """
    print("Invoking get_bs_data")
    # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
    particle_type = [4]
    attributes = ['age', 'mass', 'pos']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    # Rotate the particle positions so the bar is along the x-axis #
    stellar_mask, = np.where((s.data['age'] > 0.0) & (s.r() < 0.1 * s.subfind.data['frc2'][0]))  # Mask the data: select stellar particles.
    z_rotated, y_rotated, x_rotated = plot_tools.rotate_bar(s.data['pos'][stellar_mask, 0] * 1e3, s.data['pos'][stellar_mask, 1] * 1e3,
                                                            s.data['pos'][stellar_mask, 2] * 1e3)  # Distances are in Mpc.
    s.data['pos'] = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.data['pos'] attribute in kpc.
    x, y = s.data['pos'][:, 2] * 1e3, s.data['pos'][:, 1] * 1e3  # Load positions and convert from Mpc to Kpc.

    # Declare arrays to store the data #
    n_bins = 40  # Number of radial bins.
    r_m = np.zeros(n_bins)
    beta_2 = np.zeros(n_bins)
    alpha_0 = np.zeros(n_bins)
    alpha_2 = np.zeros(n_bins)

    # Split up galaxy in radius bins and calculate the Fourier components #
    r = np.sqrt(x[:] ** 2 + y[:] ** 2)  # Radius of each particle.
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
    max_A2s = max(np.divide(np.sqrt(alpha_2[:] ** 2 + beta_2[:] ** 2), alpha_0[:]))

    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), max_A2s


def bar_strength(pdf, data, read):
    """
    Plot the evolution of bar strength from Fourier modes of surface density for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking bar_strength")
    redshift_cut = 2
    path = '/u/di43/Auriga/plots/data/' + 'bse/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Check if any of the haloes' data already exists, if not then create it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if name in names:
                continue
            else:
                print("Analysing halo:", name)

            # Get all snapshots with redshift less than the redshift cut #
            redshifts = halo.get_redshifts()
            redshift_mask, = np.where(redshifts <= redshift_cut)
            snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]

            bs_data = np.array(get_bs_data(snapshot_ids, halo))  # Get bar strength data.

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(name), name)
            np.save(path + 'lookback_times_' + str(name), bs_data[:, 0])
            np.save(path + 'max_A2s_' + str(name), bs_data[:, 1])

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-0.1, 1.1], ylabel=r'$\mathrm{A_{2}}$', aspect=None)
        figure.text(0.0, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis.transAxes)

        # Load the data #
        max_A2s = np.load(path + 'max_A2s_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        plt.plot(lookback_times, max_A2s, color=colors[0])  # Plot the evolution of bar strength.

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_gtr_data(snapshot_ids, halo):
    """
    Parallelised method to get gas fractions in different temperature regimes.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :return: lookback time, sfg_ratio, wg_ratio, hg_ratio.
    """
    print("Invoking get_gtr_data")
    # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
    particle_type = [0, 4]
    attributes = ['age', 'mass', 'ne', 'pos', 'rho', 'u']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    # Calculate the temperature of the gas cells #
    gas_mask, = np.where(
        (s.data['type'] == 0) & (s.r() < s.subfind.data['frc2'][0]))  # Mask the data: select gas cells inside the virial radius R200c.
    ne = s.data['ne'][gas_mask]
    metallicity = s.data['gz'][gas_mask]
    XH = s.data['gmet'][gas_mask, element['H']]
    yhelium = (1 - XH - metallicity) / (4. * XH)
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    temperature = GAMMA_MINUS1 * s.data['u'][gas_mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN

    # Calculate the mass of the gas cells in temperatures regimes #
    mass = s.data['mass'][gas_mask]
    sfg_ratio = np.sum(mass[np.where(temperature < 2e4)]) / np.sum(mass)
    wg_ratio = np.sum(mass[np.where((temperature >= 2e4) & (temperature < 5e5))]) / np.sum(mass)
    hg_ratio = np.sum(mass[np.where(temperature >= 5e5)]) / np.sum(mass)

    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), sfg_ratio, wg_ratio, hg_ratio


def gas_temperature_regimes(pdf, data, read):
    """
    Plot the evolution of gas fractions in different temperature regimes for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking gas_temperature_regimes")
    redshift_cut = 7
    path = '/u/di43/Auriga/plots/data/' + 'gtr/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Check if any of the haloes' data already exists, if not then create it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if name in names:
                continue
            else:
                print("Analysing halo:", name)

            # Get all snapshots with redshift less than the redshift cut #
            redshifts = halo.get_redshifts()
            redshift_mask, = np.where(redshifts <= redshift_cut)
            snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]

            gtr_data = np.array(get_gtr_data(snapshot_ids, halo))  # Get gas temperature regimes data.

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(name), name)
            np.save(path + 'lookback_times_' + str(name), gtr_data[:, 0])
            np.save(path + 'sfg_ratios_' + str(name), gtr_data[:, 1])
            np.save(path + 'wg_ratios_' + str(name), gtr_data[:, 2])
            np.save(path + 'hg_ratios_' + str(name), gtr_data[:, 3])

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-0.1, 1.1], ylabel=r'$\mathrm{Gas\;fraction}$', aspect=None)
        figure.text(0.0, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis.transAxes)

        # Load the data #
        wg_ratios = np.load(path + 'wg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        hg_ratios = np.load(path + 'hg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sfg_ratios = np.load(path + 'sfg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the evolution of gas fraction in different temperature regimes #
        plt.plot(lookback_times, hg_ratios, color='red', label=r'$\mathrm{Hot\;gas}$')
        plt.plot(lookback_times, sfg_ratios, color='blue', label=r'$\mathrm{Cold\;gas}$')
        plt.plot(lookback_times, wg_ratios, color='green', label=r'$\mathrm{Warm\;gas}$')

        # Create the legend, save and close the figure #
        plt.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_dsr_data(snapshot_ids, halo, radial_limit_min, radial_limit_max):
    """
    Parallelised method to get star formation rate for different spatial regimes.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :param radial_limit_min: inner radial limit.
    :param radial_limit_max: outer radial limit.
    :return: lookback time, SFR.
    """
    print("Invoking get_dsr_data")
    # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
    particle_type = [0, 4]
    attributes = ['age', 'gima', 'mass', 'pos', 'sfr']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    # Calculate the age and sfr #
    # stellar_mask, = np.where((s.data['age'] > 0.0) & (s.r() > radial_limit_min) & (s.r() < radial_limit_max) & (
    #     s.data['pos'][:, 2] < 0.003))  # Mask the data: select stellar particles inside different (physical) spatial regimes
    # time_mask, = np.where(((s.data['age'][stellar_mask] - s.time) < time_bin_width) & (s.data['age'][stellar_mask] > s.time))
    # SFR = s.data['gima'][stellar_mask][time_mask] / time_bin_width * 10  # In Msun yr^-1.

    # Convert radial limits to physical units i.e., keep them constant in co-moving space. #
    a = 1 / (1 + s.redshift)
    gas_mask, = np.where((s.data['sfr'] > 0.0) & (s.r()[s.data['type'] == 0] > radial_limit_min) & (
        s.r()[s.data['type'] == 0] <= radial_limit_max))  # Mask the data: select gas cells inside different physical spatial regimes.
    SFR = np.sum(s.data['sfr'][gas_mask])

    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), SFR


def delta_sfr_regimes(pdf, data, region, read):
    """
    Plot the evolution of star formation rate for different spatial regimes and the difference between Auriga haloes.
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param region: inner or outer.
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking delta_sfr_regimes")
    n_bins = 130
    time_bin_width = (13 - 0) / n_bins  # In Gyr.

    # Get limits based on the region #
    if region == 'outer':
        radial_limits_min, radial_limits_max = (0.0, 1e-3, 5e-3), (1e-3, 5e-3, 15e-3)
    elif region == 'inner':
        radial_limits_min, radial_limits_max = (0.0, 2.5e-4, 5e-4), (2.5e-4, 5e-4, 7.5e-4)

    # Read the data #
    if read is True:
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'gima', 'mass', 'pos']
        data.select_haloes(default_level, default_redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            # Loop over all spatial regimes #
            for radial_limit_min, radial_limit_max in zip(radial_limits_min, radial_limits_max):
                # Check if a folder to save the data exists, if not create one #
                path = '/u/di43/Auriga/plots/data/' + 'dsr/' + str(radial_limit_max) + '/'
                if not os.path.exists(path):
                    os.makedirs(path)

                # Check if any of the haloes' data already exists, if not then create it #
                names = glob.glob(path + '/name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                if str(s.haloname) in names:
                    continue
                else:
                    print("Analysing halo:", str(s.haloname))

                # Get the lookback times and calculate the initial masses #
                stellar_mask, = np.where((s.data['age'] > 0.) & (s.r() > radial_limit_min) & (
                    s.r() <= radial_limit_max))  # Mask the data: select stellar particles inside radial limits.
                lookback_times = s.cosmology_get_lookback_time_from_a(s.data['age'][stellar_mask], is_flat=True)  # In Gyr.
                weights = s.data['gima'][stellar_mask] * 1e10 / 1e9 / time_bin_width  # In Msun yr^-1.

                # Save data for each halo in numpy arrays #
                np.save(path + 'name_' + str(s.haloname), s.haloname)
                np.save(path + 'weights_' + str(s.haloname), weights)
                np.save(path + 'lookback_times_' + str(s.haloname), lookback_times)

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.05)
    axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
    axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])

    for axis in [axis10, axis11, axis12]:
        axis.set_yscale('symlog', subsy=[2, 3, 4, 5, 6, 7, 8, 9], linthreshy=1, linscaley=0.1)
    for axis in [axis01, axis02, axis11, axis12]:
        axis.set_yticklabels([])
    for axis in [axis00, axis01, axis02]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[0, 15], aspect=None)
    for axis in [axis10, axis11, axis12]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=(-1.1, 2e1), aspect=None)
    axis00.set_ylabel(r'$\mathrm{Sfr/(M_\odot\;yr^{-1})}$', size=16)
    axis10.set_ylabel(r'$\mathrm{(\delta Sfr)_{norm}}$', size=16)

    # Loop over all radial limits #
    top_axes, bottom_axes = [axis00, axis01, axis02], [axis10, axis11, axis12]
    for radial_limit_min, radial_limit_max, top_axis, bottom_axis in zip(radial_limits_min, radial_limits_max, top_axes, bottom_axes):
        # Get the names and sort them #
        path = '/u/di43/Auriga/plots/data/' + 'dsr/' + str(radial_limit_max) + '/'
        names = glob.glob(path + '/name_06*')
        names.sort()

        # Loop over all available haloes #
        for i in range(len(names)):
            # Load the data #
            weights = np.load(path + 'weights_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

            # Plot the evolution of SFR and the normalised delta SFR #
            counts, bins, bars = top_axis.hist(lookback_times, weights=weights, histtype='step', bins=n_bins, range=[0, 13], color=colors[i],
                                               label="Au-" + (str(re.split('_|.npy', names[i])[1])))
            if i == 0:
                original_bins, original_counts = bins, counts
            else:
                bottom_axis.plot(original_bins[:-1], (np.divide(counts - original_counts, original_counts)), color=colors[i],
                                 label="Au-" + (str(re.split('_|.npy', names[i])[1])))

        # Add the text and create the legend #
        figure.text(0.01, 0.92, r'$\mathrm{%.0f<r/kpc\leq%.0f}$' % ((np.float(radial_limit_min) * 1e3), (np.float(radial_limit_max) * 1e3)),
                    fontsize=16, transform=top_axis.transAxes)
        top_axis.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
        bottom_axis.legend(loc='upper center', fontsize=12, frameon=False, numpoints=1, ncol=2)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_ssgr_data(snapshot_ids, halo, radial_limit_min, radial_limit_max):
    """
    Parallelised method to get star formation rate, stellar mass and gas mass for different spatial regimes.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :param radial_limit_min: inner radial limit.
    :param radial_limit_max: outer radial limit.
    :return: lookback time, SFR, sfg_ratio, wg_ratio, hg_ratio, np.sum(gas_mass), stellar_mass.
    """
    print("Invoking get_ssgr_data")
    # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
    particle_type = [0, 4]
    attributes = ['age', 'gima', 'mass', 'pos', 'sfr']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    # Convert radial limits to physical units i.e., keep them constant in co-moving space. #
    a = 1 / (1 + s.redshift)
    age = np.zeros(s.npartall)
    age[s.data['type'] == 4] = s.data['age']
    gas_mask, = np.where((s.data['type'] == 0) & (s.r() > radial_limit_min) & (
        s.r() <= radial_limit_max))  # Mask the data: select gas cells inside different physical spatial regimes.
    stellar_mask, = np.where((s.data['type'] == 4) & (age > 0.0) & (s.r() > radial_limit_min) & (
        s.r() <= radial_limit_max))  # Mask the data: select stellar particles inside different physical spatial regimes.

    # Calculate the temperature of the gas cells #
    ne = s.data['ne'][gas_mask]
    metallicity = s.data['gz'][gas_mask]
    XH = s.data['gmet'][gas_mask, element['H']]
    yhelium = (1 - XH - metallicity) / (4. * XH)
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    temperature = GAMMA_MINUS1 * s.data['u'][gas_mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN

    # Calculate the mass of the gas cells in temperatures regimes #
    gas_mass = s.data['mass'][gas_mask]
    hg_ratio = np.sum(gas_mass[np.where(temperature >= 5e5)]) / np.sum(gas_mass)
    sfg_ratio = np.sum(gas_mass[np.where(temperature < 2e4)]) / np.sum(gas_mass)
    wg_ratio = np.sum(gas_mass[np.where((temperature >= 2e4) & (temperature < 5e5))]) / np.sum(gas_mass)

    # Calculate star formation rate and stellar mass #
    SFR = np.sum(s.data['sfr'][gas_mask])
    stellar_mass = np.sum(s.data['mass'][stellar_mask])

    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), SFR, sfg_ratio, wg_ratio, hg_ratio, np.sum(gas_mass), stellar_mass


def sfr_stars_gas_regimes(pdf, data, region, read):
    """
    Plot the evolution of star formation rate, stellar mass and gas mass for different spatial regimes for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param region: inner or outer.
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking sfr_stars_gas_regimes")
    redshift_cut = 7
    n_bins = 130
    time_bin_width = (13 - 0) / n_bins  # In Gyr.

    # Get limits based on the region #
    if region == 'outer':
        radial_limits_min, radial_limits_max = (0.0, 1e-3, 5e-3), (1e-3, 5e-3, 15e-3)
    elif region == 'inner':
        radial_limits_min, radial_limits_max = (0.0, 2.5e-4, 5e-4), (2.5e-4, 5e-4, 7.5e-4)

    # Read the data #
    if read is True:
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'gima', 'mass', 'pos']
        data.select_haloes(default_level, default_redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            # Loop over all spatial regimes #
            for radial_limit_min, radial_limit_max in zip(radial_limits_min, radial_limits_max):
                # Check if a folder to save the data exists, if not create one #
                path = '/u/di43/Auriga/plots/data/' + 'ssgr/' + str(radial_limit_max) + '/'
                if not os.path.exists(path):
                    os.makedirs(path)

                # Check if any of the haloes' data already exists, if not then create it #
                names = glob.glob(path + '/name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                if str(s.haloname) in names:
                    continue
                else:
                    print("Analysing halo:", str(s.haloname))

                # Get the lookback times and calculate the initial masses #
                stellar_mask, = np.where((s.data['age'] > 0.) & (s.r() > radial_limit_min) & (
                    s.r() <= radial_limit_max))  # Mask the data: select stellar particles inside radial limits.
                lookback_times = s.cosmology_get_lookback_time_from_a(s.data['age'][stellar_mask], is_flat=True)  # In Gyr.
                weights = s.data['gima'][stellar_mask] * 1e10 / 1e9 / time_bin_width  # In Msun yr^-1.

                # Save data for each halo in numpy arrays #
                np.save(path + 'name_' + str(s.haloname), s.haloname)
                np.save(path + 'weights_' + str(s.haloname), weights)
                np.save(path + 'lookback_times_SFR_' + str(s.haloname), lookback_times)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Loop over all spatial regimes #
            for radial_limit_min, radial_limit_max in zip(radial_limits_min, radial_limits_max):
                # Check if a folder to save the data exists, if not create one #
                path = '/u/di43/Auriga/plots/data/' + 'ssgr/' + str(radial_limit_max) + '/'

                # Get all snapshots with redshift less than the redshift cut #
                redshifts = halo.get_redshifts()
                redshift_mask, = np.where(redshifts <= redshift_cut)
                snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]

                ssgr_data = np.array(get_ssgr_data(snapshot_ids, halo, radial_limit_min, radial_limit_max))  # Get SFR stars gas regimes data.

                # Save data for each halo in numpy arrays #
                np.save(path + 'lookback_times_' + str(name), ssgr_data[:, 0])
                np.save(path + 'sfg_ratios_' + str(name), ssgr_data[:, 2])
                np.save(path + 'wg_ratios_' + str(name), ssgr_data[:, 3])
                np.save(path + 'hg_ratios_' + str(name), ssgr_data[:, 4])
                np.save(path + 'gas_masses_' + str(name), ssgr_data[:, 5])
                np.save(path + 'stellar_masses_' + str(name), ssgr_data[:, 6])

    # Loop over all radial limits #
    for radial_limit_min, radial_limit_max in zip(radial_limits_min, radial_limits_max):
        path = '/u/di43/Auriga/plots/data/' + 'ssgr/' + str(radial_limit_max) + '/'
        path_modes = '/u/di43/Auriga/plots/data/' + 'AGNmd/'

        # Get the names and sort them #
        names = glob.glob(path + '/name_17.*')
        names.sort()

        # Loop over all available haloes #
        for i in range(len(names)):
            # Generate the figure and set its parameters #
            figure = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(3, 1, hspace=0.06, height_ratios=[1, 0.5, 0.5])
            axis00 = plt.subplot(gs[0, 0])
            axis10 = plt.subplot(gs[1, 0])
            axis20 = plt.subplot(gs[2, 0])

            for axis in [axis00, axis10, axis20]:
                axis2 = axis.twiny()
                plot_tools.set_axes_evolution(axis, axis2, ylabel=r'$\mathrm{Gas\;fraction}$', aspect=None, size=20)
                if axis in [axis10, axis20]:
                    axis2.set_xlabel('')
                    axis2.set_xticklabels([])
            for axis in [axis00, axis10]:
                axis.set_xlabel('')
                axis.set_xticklabels([])

            # axis002 = axis00.twiny()
            # plot_tools.set_axes_evolution(axis00, axis002, ylim=(0, 8), ylabel=r'$\mathrm{Sfr/(M_\odot\;yr^{-1})}$', aspect=None)
            # axis00.set_xticklabels([])
            axis003 = axis00.twinx()
            plot_tools.set_axis(axis003, ylim=(1e55, 1e61), yscale='log', ylabel=r'$\mathrm{(Feedback\;energy)/ergs}$', aspect=None, which='major')
            # plot_tools.set_axis(axis10, xlim=(13, 0), ylim=(1e6, 1e11), yscale='log', ylabel=r'$\mathrm{Mass/M_{\odot}}$', aspect=None,
            # which='major')
            # axis10.set_xticklabels([])
            # plot_tools.set_axis(axis20, xlim=(13, 0), ylim=(-0.1, 1.1), ylabel=r'$\mathrm{Gas\;fraction}$', xlabel=r'$\mathrm{t_{look}/Gyr}$',
            #                     aspect=None)
            figure.text(0.01, 0.85, r'$\mathrm{Au-%s}$' '\n' r'$\mathrm{%.0f<r/kpc\leq%.0f}$' % (
                str(re.split('_|.npy', names[i])[1]), (np.float(radial_limit_min) * 1e3), (np.float(radial_limit_max) * 1e3)), fontsize=16,
                        transform=axis00.transAxes)

            # Load and plot the data #
            weights = np.load(path + 'weights_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            wg_ratios = np.load(path + 'wg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            hg_ratios = np.load(path + 'hg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            sfg_ratios = np.load(path + 'sfg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            gas_masses = np.load(path + 'gas_masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            thermals = np.load(path_modes + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            mechanicals = np.load(path_modes + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            stellar_masses = np.load(path + 'stellar_masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            lookback_times_SFR = np.load(path + 'lookback_times_SFR_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            lookback_times_modes = np.load(path_modes + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

            # Transform the arrays to comma separated strings and convert each element to float #
            thermals = ','.join(thermals)
            mechanicals = ','.join(mechanicals)
            thermals = np.fromstring(thermals, dtype=np.float, sep=',')
            mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')

            # Plot the evolution of SFR #
            axis00.hist(lookback_times_SFR, weights=weights, histtype='step', bins=n_bins, range=[0, 13], color=colors[i],
                        label="Au-" + (str(re.split('_|.npy', names[i])[1])))

            # Plot the feedback modes binned sum line #
            for mode, label, color in zip([mechanicals, thermals], [r'$\mathrm{Mechanical}$', r'$\mathrm{Thermal}$'], [colors[4], colors[5]]):
                x_value, sum = plot_tools.binned_sum(lookback_times_modes[np.where(mode > 0)], mode[np.where(mode > 0)], n_bins=n_bins)
                axis003.plot(x_value, sum / time_bin_width, color=color, label=label)
                axis003.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1, ncol=2)

            # Plot the stellar and gaseous masses #
            axis10.plot(lookback_times, gas_masses * 1e10, color=colors[3], label=r'$\mathrm{Gas}$')
            axis10.plot(lookback_times, stellar_masses * 1e10, color=colors[1], label=r'$\mathrm{Stars}$')

            # Plot the gas fractions #
            axis20.plot(lookback_times, sfg_ratios, color=colors[3], label=r'$\mathrm{Cold\;gas}$')
            axis20.plot(lookback_times, wg_ratios, color=colors[2], label=r'$\mathrm{Warm\;gas}$')
            axis20.plot(lookback_times, hg_ratios, color=colors[1], label=r'$\mathrm{Hot\;gas}$')

            # Create the legends, save and close the figure #
            axis10.legend(loc='lower center', fontsize=16, frameon=False, numpoints=1, ncol=2)
            axis20.legend(loc='upper left', fontsize=16, frameon=False, numpoints=1)
            pdf.savefig(figure, bbox_inches='tight')
            plt.close()
    return None


def AGN_modes_distribution(date, data, read):
    """
    Get the energy of different black hole feedback modes from log files and plot its evolution for Auriga halo(es).
    :param date: .
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking AGN_modes_distribution")
    path = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    n_bins = 130
    time_bin_width = (13 - 0) / n_bins  # In Gyr.

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        data.select_haloes(default_level, default_redshift, loadonlyhalo=0, loadonlytype=None, loadonly=None)

        # Loop over all available haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then create it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))

            # Declare arrays to store the desired words and lines that contain these words #
            redshift_lines, redshifts, feedback_lines, thermals, mechanicals = [], [], [], [], []

            with open('/u/di43/Auriga/output/halo_' + str(s.haloname) + '/Au-' + str(s.haloname) + '.txt') as file:
                # Loop over all lines #
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

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 7.5))
    gs = gridspec.GridSpec(2, 3, wspace=0.22, hspace=0.2, height_ratios=[0.05, 1])
    axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
    axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])

    axis102, axis112, axis122 = axis10.twiny(), axis11.twiny(), axis12.twiny()
    plot_tools.set_axes_evolution(axis10, axis102, yscale='log', ylim=[1e51, 1e61], ylabel=r'$\mathrm{(Mechanical\;feedback\;energy)/ergs}$',
                                  aspect=None, which='major')
    plot_tools.set_axes_evolution(axis11, axis112, ylim=[1e51, 1e61], ylabel=r'$\mathrm{(Thermal\;feedback\;energy)/ergs}$', aspect=None,
                                  which='major')
    plot_tools.set_axes_evolution(axis12, axis122, ylim=[1e51, 1e61], ylabel=r'$\mathrm{(Thermal\;feedback\;energy)/ergs}$', aspect=None,
                                  which='major')

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Load the data #
        if i == 0:
            mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            # Transform the arrays to comma separated strings and convert each element to float #
            mechanicals = ','.join(mechanicals)
            mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
        thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        # Transform the arrays to comma separated strings and convert each element to float #
        thermals = ','.join(thermals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')

        # Plot 2D distribution of the modes and their binned sum line #
        if i == 0:
            axes = [axis10, axis11]
            axescbar = [axis00, axis01]
            modes = [mechanicals, thermals]
        else:
            axes = [axis12]
            modes = [thermals]
            axescbar = [axis02]

        for axis, axiscbar, mode in zip(axes, axescbar, modes):
            hb = axis.hexbin(lookback_times[np.where(mode > 0)], mode[np.where(mode > 0)], bins='log', yscale='log', cmap='hot_r')
            plot_tools.create_colorbar(axiscbar, hb, label=r'$\mathrm{Counts\;per\;hexbin}$', orientation='horizontal')
            x_value, sum = plot_tools.binned_sum(lookback_times[np.where(mode > 0)], mode[np.where(mode > 0)], n_bins=n_bins)
            axis.plot(x_value, sum / time_bin_width, color=colors[0], label=r'$\mathrm{Sum}$')
            figure.text(0.0, 0.95, 'Au-' + str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis.transAxes)

            # Set colour-bar parameters #
            axiscbar.xaxis.tick_top()
            axiscbar.xaxis.set_label_position("top")
            axiscbar.tick_params(direction='out', which='both', top='on', right='on')

    # Create the legends, save and close the figure #
    axis10.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
    axis11.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
    axis12.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
    plt.savefig('/u/di43/Auriga/plots/' + 'AGNmd-' + date + '.png', bbox_inches='tight')
    plt.close()

    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_afk_data(snapshot_ids, halo):
    """
    Parallelised method to get the black hole radius and the volume of gas cells within that.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :return: lookback time, , gas_volumes, sf_gas_volumes, nsf_gas_volumes, blackhole_hsml.
    """
    print("Invoking get_afk_data")
    # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
    particle_type = [0, 4, 5]
    attributes = ['age', 'bhhs', 'id', 'mass', 'pos', 'sfr', 'vol']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    # Check that only one (the closest) black hole is selected #
    blackhole_mask, = np.where(
        (s.data['type'] == 5) & (s.r() < 0.1 * s.subfind.data['frc2'][0]))  # Mask the data: select the black hole inside 0.1*R200c.
    if len(blackhole_mask) == 1:
        blackhole_hsml = s.data['bhhs'][0]

    # In mergers select the most massive black hole #
    elif len(blackhole_mask) > 1:
        blackhole_id = s.data['id'][s.data['type'] == 5]
        id_mask, = np.where(blackhole_id == s.data['id'][s.data['mass'].argmax()])
        blackhole_hsml = s.data['bhhs'][id_mask[0]]
    else:
        blackhole_hsml = 0

    # Convert black hole hsml to physical h-free units i.e., keep it constant in co-moving space. #
    a = 1 / (1 + s.redshift)
    blackhole_hsml = blackhole_hsml * a / 0.6777

    # Mask the data: select star-forming and not gas cells inside the black hole's radius #
    gas_mask, = np.where(s.r()[s.data['type'] == 0] < blackhole_hsml)
    sf_gas_mask, = np.where((s.data['sfr'] > 0.0) & (s.r()[s.data['type'] == 0] < blackhole_hsml))
    nsf_gas_mask, = np.where((s.data['sfr'] <= 0.0) & (s.r()[s.data['type'] == 0] < blackhole_hsml))

    # Compute the total volume of cells with SFR == 0 and compare it to the total volume of all cells inside the black hole's radius #
    gas_volumes = s.data['vol'][gas_mask].sum() * 1e9 * a ** 3 / 0.6777 ** 3  # In kpc^-3.
    sf_gas_volumes = s.data['vol'][sf_gas_mask].sum() * 1e9 * a ** 3 / 0.6777 ** 3  # In kpc^-3.
    nsf_gas_volumes = s.data['vol'][nsf_gas_mask].sum() * 1e9 * a ** 3 / 0.6777 ** 3  # In kpc^-3.

    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), gas_volumes, sf_gas_volumes, nsf_gas_volumes, blackhole_hsml


def AGN_feedback_kernel(pdf, data, ds, read):
    """
    Plot the evolution of black hole radius and the volume of gas cells within that for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param ds: boolean to downsample halo_18_3000 data.
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking AGN_feedback_kernel")
    redshift_cut = 7
    path = '/u/di43/Auriga/plots/data/' + 'AGNfk/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Check if any of the haloes' data already exists, if not then create it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if name in names:
                continue
            else:
                print("Analysing halo:", name)

            # Get all snapshots with redshift less than the redshift cut #
            redshifts = halo.get_redshifts()
            if ds is True:
                # Downsample the 3000 snapshots run #
                for name, halo in haloes.items():
                    if name == '18':
                        redshifts = halo.get_redshifts()
            redshift_mask, = np.where(redshifts <= redshift_cut)
            snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]

            ssgr_data = np.array(get_afk_data(snapshot_ids, halo))  # Get AGN feedback kernel data.

            # Save data for each halo in numpy arrays #
            if ds is True:
                name = name + '_ds'
            np.save(path + 'name_' + str(name), name)
            np.save(path + 'lookback_times_' + str(name), ssgr_data[:, 0])
            np.save(path + 'gas_volumes_' + str(name), ssgr_data[:, 1])
            np.save(path + 'sf_gas_volumes_' + str(name), ssgr_data[:, 2])
            np.save(path + 'nsf_gas_volumes_' + str(name), ssgr_data[:, 3])
            np.save(path + 'blackhole_hsmls_' + str(name), ssgr_data[:, 4])

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        axis3 = axis.twinx()
        axis3.yaxis.label.set_color('red')
        axis3.spines['right'].set_color('red')
        plot_tools.set_axis(axis3, ylim=[-0.1, 1.1], xlabel=r'$\mathrm{t_{look}/Gyr}$', ylabel=r'$\mathrm{BH_{sml}/kpc}$', aspect=None)
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-0.1, 1.1], ylabel=r'$\mathrm{V_{nSFR}(r<BH_{sml})/V_{all}(r<BH_{sml})}$', aspect=None)
        axis3.tick_params(axis='y', direction='out', left='off', colors='red')
        figure.text(0.0, 0.95, 'Au-' + str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis.transAxes)

        # Load and plot the data #
        gas_volumes = np.load(path + 'gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        nsf_gas_volumes = np.load(path + 'nsf_gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        blackhole_hsmls = np.load(path + 'blackhole_hsmls_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the evolution of black hole radius and the volume ratio of gas cells within that #
        axis3.scatter(lookback_times, blackhole_hsmls * 1e3, edgecolor='None', color=colors[1])
        axis.scatter(lookback_times, nsf_gas_volumes / gas_volumes, edgecolor='None', color=colors[0])

        # Plot median and 1-sigma lines #
        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(lookback_times, nsf_gas_volumes / gas_volumes, bin_type='equal_number',
                                                                       n_bins=len(lookback_times) / 2, log=False)
        axis.plot(x_value, median, color=colors[0], linewidth=3)
        axis.fill_between(x_value, shigh, slow, color=colors[0], alpha='0.3')

        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(lookback_times, blackhole_hsmls * 1e3, bin_type='equal_width', n_bins=10,
                                                                       log=False)
        axis3.plot(x_value, median, color=colors[1], linewidth=3)
        axis3.fill_between(x_value, shigh, slow, color=colors[1], alpha='0.3')

        # Save and close the figure #
        plt.savefig('/u/di43/Auriga/plots/' + 'Auriga-' + pdf + '.png', bbox_inches='tight')
        # pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def AGN_feedback_smoothed(pdf):
    """
    Calculate the energy deposition on gas cells from the AGN feedback for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking AGN_feedback_smoothed")
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNfs/'
    path_modes = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    path_kernel = '/u/di43/Auriga/plots/data/' + 'AGNfk/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Get the names and sort them #
    names = glob.glob(path_kernel + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[1e52, 1e62], yscale='log', ylabel=r'$\mathrm{(AGN\;feedback\;energy)/ergs}$')
        figure.text(0.0, 0.95, 'Au-' + str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis.transAxes)

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
                                     bins=np.quantile(np.sort(lookback_times), np.linspace(0, 1, len(lookback_times) + 1)))
        y_data /= edges[1:] - edges[:-1]  # Normalise the values wrt the bin width.
        axis.plot(0.5 * (edges[1:] + edges[:-1]), y_data, color=colors[1])
        axis.plot(0.5 * (edges[1:] + edges[:-1]), y_data * np.flip(nsf_gas_volumes / gas_volumes), color=colors[4])
        # axis2.hist(lookback_times_modes[np.where(thermals > 0)], weights=thermals[np.where(thermals > 0)], histtype='step',
        #            bins=lookback_times.sort())

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_bm_data(snapshot_ids, halo, blackhole_id):
    """
    Parallelised method to get black hole properties.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :param blackhole_id: id of the black hole particle.
    :return: black hole properties.
    """
    print("Invoking get_bm_data")
    # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
    particle_type = [5]
    attributes = ['bhma', 'bcmr', 'bcmq', 'bhmd', 'bhmr', 'bhmq', 'id']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Check if there is a black hole particle, if not return only lookback times #
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
    Plot the evolution of black hole mass, accretion rate, cumulative mass accreted onto the BH in the low and high accretion-state for Auriga
    halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking blackhole_masses")
    redshift_cut = 7
    path = '/u/di43/Auriga/plots/data/' + 'bm/'
    path_modes = '/u/di43/Auriga/plots/data/' + 'AGNmd/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Check if any of the haloes' data already exists, if not then create it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if name in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))

            # Get all snapshots with redshift less than the redshift cut #
            redshifts = halo.get_redshifts()
            redshift_mask, = np.where(redshifts <= redshift_cut)
            snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]

            # Find the black hole's id and use it to get black hole data #
            s = halo.snaps[snapshot_ids.argmax()].loadsnap(loadonlyhalo=0, loadonlytype=[5])
            blackhole_id = s.data['id'][s.data['mass'].argmax()]

            # Get blackhole data #
            bm_data = np.array(get_bm_data(snapshot_ids, halo, blackhole_id))  # Get blackhole masses data.
            lookback_times = bm_data[:, 0]
            black_hole_masses = bm_data[:, 1]
            black_hole_cmasses_radio, black_hole_cmasses_quasar = bm_data[:, 2], bm_data[:, 3]
            black_hole_dmasses, black_hole_dmasses_radio, black_hole_dmasses_quasar = bm_data[:, 4], bm_data[:, 5], bm_data[:, 6]

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(name), name)
            np.save(path + 'lookback_times_' + str(name), lookback_times)
            np.save(path + 'black_hole_masses_' + str(name), black_hole_masses)
            np.save(path + 'black_hole_dmasses_' + str(name), black_hole_dmasses)
            np.save(path + 'black_hole_cmasses_radio_' + str(name), black_hole_cmasses_radio)
            np.save(path + 'black_hole_dmasses_radio_' + str(name), black_hole_dmasses_radio)
            np.save(path + 'black_hole_cmasses_quasar_' + str(name), black_hole_cmasses_quasar)

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.2)
        axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
        axis10, axis11 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])
        axis002, axis012 = axis00.twiny(), axis01.twiny()
        axis102, axis112 = axis10.twiny(), axis11.twiny()

        figure.text(0.0, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis00.transAxes)

        for axis in [axis01, axis11]:
            axis.yaxis.set_label_position("right")
            axis.yaxis.tick_right()

        plot_tools.set_axes_evolution(axis00, axis002, ylim=[1e2, 1e9], yscale='log', ylabel=r'$\mathrm{M_{BH}/M_\odot}$')
        plot_tools.set_axes_evolution(axis01, axis012, ylim=[1e2, 1e9], yscale='log', ylabel=r'$\mathrm{M_{BH,mode}/M_\odot}$')
        plot_tools.set_axes_evolution(axis10, axis102, yscale='log', ylabel=r'$\mathrm{\dot{M}_{BH}/(M_\odot\;Gyr^{-1})}$')
        plot_tools.set_axes_evolution(axis11, axis112, yscale='log', ylabel=r'$\mathrm{(AGN\;feedback\;energy)/ergs}$')

        # Load the data #
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

        # Plot black hole mass and accretion rates #
        axis00.plot(lookback_times, black_hole_masses * 1e10, color=colors[0])
        axis01.plot(lookback_times, black_hole_imasses_radio * 1e10, color=colors[1])
        axis01.plot(lookback_times, black_hole_imasses_quasar * 1e10, color=colors[2])
        plt10, = axis10.plot(lookback_times, black_hole_dmasses, color=colors[0])
        plt101, = axis10.plot(lookback_times, black_hole_dmasses_radio, color=colors[1])
        plt102, = axis10.plot(lookback_times, black_hole_dmasses_quasar, color=colors[2])

        # Transform the arrays to comma separated strings and convert each element to float #
        thermals = ','.join(thermals)
        mechanicals = ','.join(mechanicals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')
        mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')

        # Calculate and plot the thermals energy sum #
        modes = [mechanicals, thermals]
        for i, mode in enumerate(modes):
            n_bins = int((max(lookback_times_modes[np.where(mode > 0)]) - min(lookback_times_modes[np.where(mode > 0)])) / 0.02)
            sum = np.zeros(n_bins)
            x_value = np.zeros(n_bins)
            x_low = min(lookback_times_modes[np.where(mode > 0)])
            for j in range(n_bins):
                index = np.where((lookback_times_modes[np.where(mode > 0)] >= x_low) & (lookback_times_modes[np.where(mode > 0)] < x_low + 0.02))[0]
                x_value[j] = np.mean(np.absolute(lookback_times_modes[np.where(mode > 0)])[index])
                if len(index) > 0:
                    sum[j] = np.sum(mode[np.where(mode > 0)][index])
                x_low += 0.02
            axis11.plot(x_value, sum, color=colors[1 + i])

        axis10.legend([plt10, plt102, plt101], [r'$\mathrm{BH}$', r'$\mathrm{Thermal}$', r'$\mathrm{Mechanical}$'], loc='lower right', fontsize=16,
                      frameon=False, numpoints=1)

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None
