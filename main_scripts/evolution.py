import os
import re
import glob
import plot_tools

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.style as style

from const import *
from sfigure import *
from loadmodules import *
from matplotlib import gridspec
from parallel_decorators import vectorize_parallel
from scripts.gigagalaxy.util import satellite_utilities

style.use("classic")
plt.rcParams.update({'font.family':'serif'})

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
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s)
        # for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'gima', 'mass', 'pos']
        data.select_haloes(default_level, default_redshift, loadonlyhalo=0, loadonlytype=particle_type,
                           loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + 'name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Reading data for halo:", str(s.haloname))

            # Select the halo and rotate it based on its principal axes so
            # galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            # Get the lookback times and calculate the initial masses #
            stellar_mask, = np.where((s.data['age'] > 0.) & (
                s.r() < 0.1 * s.subfind.data['frc2'][0]))  # Mask the data: select stellar particles inside
            # 0.1*R200c.
            lookback_times = s.cosmology_get_lookback_time_from_a(s.data['age'][stellar_mask], is_flat=True)  # In Gyr.
            weights = s.data['gima'][stellar_mask] * 1e10 / 1e9 / time_bin_width  # In Msun yr^-1.

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'weights_' + str(s.haloname), weights)
            np.save(path + 'lookback_times_' + str(s.haloname), lookback_times)

    # Get the names and sort them #
    names = glob.glob(path + 'name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[0, 45], ylabel='$\mathrm{SFR/(M_\odot\;yr^{'
                                                                        '-1})}$', aspect=None)
        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis.transAxes)

        # Load the data #
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        weights = np.load(path + 'weights_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        axis.hist(lookback_times, weights=weights, histtype='step', bins=n_bins, range=[0, 13],
                  color=colors[0])  # Plot the evolution of SFR.

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_gf_data(snapshot_ids, halo):
    """
    Parallelised method to get gas fractions.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :return: lookback time, gas_fraction
    """
    print("Invoking get_gf_data")
    # Read desired galactic property(ies) for specific particle type(s) for
    # Auriga halo(es) #
    particle_type = [0, 4]
    attributes = ['age', 'mass']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's
    # spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=False)

    # Mask the data: stellar particles inside 0.1*R200c #
    age = np.zeros(s.npartall)
    age[s.data['type'] == 4] = s.data['age']
    stellar_mask, = np.where(
        (s.data['type'] == 4) & (s.r() < 0.1 * s.subfind.data['frc2'][0]) & (age > 0.)) - s.nparticlesall[:4].sum()

    stellar_mask, = np.where((s.data['type'] == 4) & (s.r() < 0.1 * s.subfind.data['frc2'][0]) & (age > 0.))
    gas_mask, = np.where((s.data['type'] == 0) & (s.r() < 0.1 * s.subfind.data['frc2'][0]))
    gas_fraction = np.sum(s.data['mass'][gas_mask]) / (
        np.sum(s.data['mass'][gas_mask]) + np.sum(s.data['mass'][stellar_mask]))

    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), gas_fraction


def gas_fraction(pdf, data, read):
    """
    Plot the evolution of gas fraction for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking gas_fraction")
    redshift_cut = 7
    path = '/u/di43/Auriga/plots/data/' + 'gfe/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + 'name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if name in names:
                continue
            else:
                print("Reading data for halo:", name)

            # Get all snapshots with redshift less than the redshift cut #
            redshifts = halo.get_redshifts()
            redshift_mask, = np.where(redshifts <= redshift_cut)
            snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]

            gf_data = np.array(get_gf_data(snapshot_ids, halo))  # Get gas temperature regimes data.

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(name), name)
            np.save(path + 'lookback_times_' + str(name), gf_data[:, 0])
            np.save(path + 'gas_fraction_' + str(name), gf_data[:, 1])

    # Get the names and sort them #
    names = glob.glob(path + 'name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-0.1, 1.1], ylabel=r'$\mathrm{Gas\;fraction}$', aspect=None)
        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis.transAxes)

        # Load the data #
        gas_fractions = np.load(path + 'gas_fraction_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the evolution of gas fraction in different temperature regimes #
        plt.plot(lookback_times, gas_fractions, color=colors[0])

        # Create the legend, save and close the figure #
        plt.legend(loc='upper right', fontsize=20, frameon=False, numpoints=1)
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_bs_data(snapshot_ids, halo):
    """
    Parallelised method to get bar strength from Fourier modes of surface
    density.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :return: lookback time, max_A2s.
    """
    print("Invoking get_bs_data")
    # Read desired galactic property(ies) for specific particle type(s) for
    # Auriga halo(es) #
    particle_type = [4]
    attributes = ['age', 'mass', 'pos']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's
    # spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    # Rotate the particle positions so the bar is along the x-axis #
    stellar_mask, = np.where(
        (s.data['age'] > 0.0) & (s.r() < 0.1 * s.subfind.data['frc2'][0]))  # Mask the data: select stellar particles.
    z_rotated, y_rotated, x_rotated = plot_tools.rotate_bar(s.data['pos'][stellar_mask, 0] * 1e3,
                                                            s.data['pos'][stellar_mask, 1] * 1e3,
                                                            s.data['pos'][stellar_mask, 2] * 1e3)  #
    # Distances are in Mpc.
    s.data['pos'] = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.data['pos']
    # attribute in kpc.
    x, y = s.data['pos'][:, 2] * 1e3, s.data['pos'][:, 1] * 1e3  # Load positions and convert
    # from Mpc to Kpc.

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
    Plot the evolution of bar strength from Fourier modes of surface density
    for Auriga halo(es).
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
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + 'name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if name in names:
                continue
            else:
                print("Reading data for halo:", name)

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
    names = glob.glob(path + 'name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-0.1, 1.1], ylabel=r'$\mathrm{A_{2}}$', aspect=None)
        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis.transAxes)

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
    # Read desired galactic property(ies) for specific particle type(s) for
    # Auriga halo(es) #
    particle_type = [0, 4]
    attributes = ['age', 'mass', 'ne', 'pos', 'rho', 'u', 'vol']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's
    # spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    # Calculate the temperature of the gas cells #
    gas_mask, = np.where((s.data['type'] == 0) & (
        s.r() < s.subfind.data['frc2'][0]))  # Mask the data: select gas cells inside the virial radius R200c.
    ne = s.data['ne'][gas_mask]
    metallicity = s.data['gz'][gas_mask]
    XH = s.data['gmet'][gas_mask, element['H']]
    yhelium = (1 - XH - metallicity) / (4. * XH)
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    temperature = GAMMA_MINUS1 * s.data['u'][gas_mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN

    # Calculate the mass and volume of gas cells in temperatures regimes #
    mass = s.data['mass'][gas_mask]
    volume = s.data['vol'][gas_mask]
    sfg_mass_ratio = np.sum(mass[np.where(temperature < 2e4)]) / np.sum(mass)
    wg_mass_ratio = np.sum(mass[np.where((temperature >= 2e4) & (temperature < 5e5))]) / np.sum(mass)
    hg_mass_ratio = np.sum(mass[np.where(temperature >= 5e5)]) / np.sum(mass)

    sfg_volume_ratio = np.sum(volume[np.where(temperature < 2e4)]) / np.sum(volume)
    wg_volume_ratio = np.sum(volume[np.where((temperature >= 2e4) & (temperature < 5e5))]) / np.sum(volume)
    hg_volume_ratio = np.sum(volume[np.where(temperature >= 5e5)]) / np.sum(volume)
    return s.cosmology_get_lookback_time_from_a(s.time,
                                                is_flat=True), sfg_mass_ratio, wg_mass_ratio, hg_mass_ratio, \
           sfg_volume_ratio, wg_volume_ratio, hg_volume_ratio


def gas_temperature_regimes(pdf, data, read):
    """
    Plot the evolution of gas fractions in different temperature regimes for
    Auriga halo(es).
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
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + 'name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            # if name in names:
            #     continue
            # else:
            #     print("Reading data for halo:", name)

            # Get all snapshots with redshift less than the redshift cut #
            redshifts = halo.get_redshifts()
            redshift_mask, = np.where(redshifts <= redshift_cut)
            snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]

            gtr_data = np.array(get_gtr_data(snapshot_ids, halo))  # Get gas temperature
            # regimes data.

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(name), name)
            np.save(path + 'lookback_times_' + str(name), gtr_data[:, 0])
            np.save(path + 'sfg_mass_ratios_' + str(name), gtr_data[:, 1])
            np.save(path + 'wg_mass_ratios_' + str(name), gtr_data[:, 2])
            np.save(path + 'hg_mass_ratios_' + str(name), gtr_data[:, 3])
            np.save(path + 'sfg_volume_ratios_' + str(name), gtr_data[:, 4])
            np.save(path + 'wg_volume_ratios_' + str(name), gtr_data[:, 5])
            np.save(path + 'hg_volume_ratios_' + str(name), gtr_data[:, 6])

    # Get the names and sort them #
    names = glob.glob(path + 'name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-0.1, 1.1], ylabel=r'$\mathrm{Gas\;fraction}$', aspect=None)
        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis.transAxes)

        # Load the data #
        wg_ratios = np.load(path + 'wg_mass_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        hg_ratios = np.load(path + 'hg_mass_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sfg_ratios = np.load(path + 'sfg_mass_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the evolution of gas fraction in different temperature regimes #
        plt.plot(lookback_times, hg_ratios, color=colors[1], label=r'$\mathrm{Hot\;gas}$')
        plt.plot(lookback_times, sfg_ratios, color=colors[2], label=r'$\mathrm{Cold\;gas}$')
        plt.plot(lookback_times, wg_ratios, color=colors[3], label=r'$\mathrm{Warm\;gas}$')

        # Create the legend, save and close the figure #
        plt.legend(loc='upper right', fontsize=20, frameon=False, numpoints=1)
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_dsr_data(snapshot_ids, halo, radial_cut_min, radial_cut_max):
    """
    Parallelised method to get star formation rate for different spatial
    regimes.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :param radial_cut_min: inner radial limit.
    :param radial_cut_max: outer radial limit.
    :return: lookback time, SFR.
    """
    print("Invoking get_dsr_data")
    # Read desired galactic property(ies) for specific particle type(s) for
    # Auriga halo(es) #
    particle_type = [0, 4]
    attributes = ['age', 'gima', 'mass', 'pos', 'sfr']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's
    # spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    # Calculate the age and sfr #
    # stellar_mask, = np.where((s.data['age'] > 0.0) & (s.r() >
    # radial_cut_min) & (s.r() < radial_cut_max) & (
    #     s.data['pos'][:, 2] < 0.003))  # Mask the data: select stellar
    #     particles inside different (physical) spatial regimes
    # time_mask, = np.where(((s.data['age'][stellar_mask] - s.time) <
    # time_bin_width) & (s.data['age'][stellar_mask] > s.time))
    # SFR = s.data['gima'][stellar_mask][time_mask] / time_bin_width * 10  #
    # In Msun yr^-1.

    # Convert radial limits to physical units i.e. keep them constant in
    # co-moving space. #
    a = 1 / (1 + s.redshift)
    gas_mask, = np.where((s.data['sfr'] > 0.0) & (s.r()[s.data['type'] == 0] > radial_cut_min) & (
        s.r()[s.data['type'] == 0] <= radial_cut_max))  # Mask the data: select gas cells inside
    # different physical spatial regimes.
    SFR = np.sum(s.data['sfr'][gas_mask])
    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), SFR


def delta_sfr_regimes(pdf, data, region, read):
    """
    Plot the evolution of star formation rate for different spatial regimes
    and the difference between Auriga haloes.
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
        radial_cuts_min, radial_cuts_max = (0.0, 1e-3, 5e-3), (1e-3, 5e-3, 15e-3)
    elif region == 'inner':
        radial_cuts_min, radial_cuts_max = (0.0, 2.5e-4, 5e-4), (2.5e-4, 5e-4, 7.5e-4)

    # Read the data #
    if read is True:
        # Read desired galactic property(ies) for specific particle type(s)
        # for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'gima', 'mass', 'pos']
        data.select_haloes(default_level, default_redshift, loadonlyhalo=0, loadonlytype=particle_type,
                           loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Select the halo and rotate it based on its principal axes so
            # galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            # Loop over all spatial regimes #
            for radial_cut_min, radial_cut_max in zip(radial_cuts_min, radial_cuts_max):
                # Check if a folder to save the data exists, if not then
                # create one #
                path = '/u/di43/Auriga/plots/data/' + 'dsr/' + str(radial_cut_max) + '/'
                if not os.path.exists(path):
                    os.makedirs(path)

                # Check if halo's data already exists, if not then read it #
                names = glob.glob(path + 'name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                if str(s.haloname) in names:
                    continue
                else:
                    print("Reading data for halo:", str(s.haloname))

                # Get the lookback times and calculate the initial masses #
                stellar_mask, = np.where((s.data['age'] > 0.) & (s.r() > radial_cut_min) & (
                    s.r() <= radial_cut_max))  # Mask the data: select
                # stellar particles inside radial limits.
                lookback_times = s.cosmology_get_lookback_time_from_a(s.data['age'][stellar_mask],
                                                                      is_flat=True)  # In Gyr.
                weights = s.data['gima'][stellar_mask] * 1e10 / 1e9 / time_bin_width  # In Msun yr^-1.

                # Save data for each halo in numpy arrays #
                np.save(path + 'name_' + str(s.haloname), s.haloname)
                np.save(path + 'weights_' + str(s.haloname), weights)
                np.save(path + 'lookback_times_' + str(s.haloname), lookback_times)

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 7.5))
    axis00, axis01, axis02, axis10, axis11, axis12 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3,
                                                                                         multiple10=True)
    for axis in [axis01, axis02, axis11, axis12]:
        axis.set_yticklabels([])
    for axis in [axis00, axis01, axis02]:
        axis.set_xlabel('')
        axis.set_xticklabels([])
        axis2 = axis.twiny()
        # Au-06: [0,11] Au-17: [0,25]
        plot_tools.set_axes_evolution(axis, axis2, ylim=[0, 25], aspect=None)
    for axis in [axis10, axis11, axis12]:
        axis2 = axis.twiny()
        # Au-06: [-1.1,14] Au-17: [-1.1, 12]
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-1.1, 12], aspect=None)
        axis2.set_xlabel('')
        axis2.set_xticklabels([])
        axis2.set_yticks([0, 5, 10])
        axis2.tick_params(top=False)
    axis00.set_ylabel(r'$\mathrm{SFR/(M_\odot\;yr^{-1})}$', size=20)
    axis10.set_ylabel(r'$\mathrm{(\delta SFR)_{norm}}$', size=20)

    # Loop over all radial limits #
    top_axes, bottom_axes = [axis00, axis01, axis02], [axis10, axis11, axis12]
    for radial_cut_min, radial_cut_max, top_axis, bottom_axis in zip(radial_cuts_min, radial_cuts_max, top_axes,
                                                                     bottom_axes):
        # Get the names and sort them #
        path = '/u/di43/Auriga/plots/data/' + 'dsr/' + str(radial_cut_max) + '/'
        names = glob.glob(path + 'name_18*')
        names.sort()

        # Loop over all available haloes #
        for i in range(len(names)):
            print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
            # Load the data #
            weights = np.load(path + 'weights_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

            # Plot the evolution of SFR and the normalised delta SFR #
            counts, bins, bars = top_axis.hist(lookback_times, weights=weights, histtype='step', bins=n_bins,
                                               range=[0, 13], color=colors[i], lw=1.5,
                                               label=r'$\mathrm{Au-%s}$' % (str(re.split('_|.npy', names[i])[1])))
            if i == 0:
                original_bins, original_counts = bins, counts
            else:
                bottom_axis.plot(original_bins[:-1], np.divide(counts - original_counts, original_counts),
                                 color=colors[i], lw=1.5,
                                 label=r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]))

        # Create the legend #
        axis00.legend(loc='upper left', fontsize=15, frameon=False, numpoints=1)
        axis10.legend(loc='upper left', fontsize=15, frameon=False, numpoints=1)

        # Add the text #
        figure.text(0.6, 0.92, r'$\mathrm{%.0f<r/kpc\leq%.0f}$' % (
            (np.float(radial_cut_min) * 1e3), (np.float(radial_cut_max) * 1e3)), fontsize=15,
                    transform=top_axis.transAxes)
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_ssgr_data(snapshot_ids, halo, radial_cut_min, radial_cut_max):
    """
    Parallelised method to get star formation rate, stellar mass and gas
    mass for different spatial regimes.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :param radial_cut_min: inner radial limit.
    :param radial_cut_max: outer radial limit.
    :return: lookback time, SFR, sfg_ratio, wg_ratio, hg_ratio, np.sum(
    gas_mass), stellar_mass.
    """
    print("Invoking get_ssgr_data")
    # Read desired galactic property(ies) for specific particle type(s) for
    # Auriga halo(es) #
    particle_type = [0, 4]
    attributes = ['age', 'gima', 'mass', 'pos', 'sfr']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's
    # spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    # Convert radial limits to physical units i.e. keep them constant in
    # co-moving space. #
    a = 1 / (1 + s.redshift)
    age = np.zeros(s.npartall)
    age[s.data['type'] == 4] = s.data['age']
    gas_mask, = np.where((s.data['type'] == 0) & (s.r() > radial_cut_min) & (
        s.r() <= radial_cut_max))  # Mask the data: select gas cells inside
    # different physical spatial regimes.
    stellar_mask, = np.where((s.data['type'] == 4) & (age > 0.0) & (s.r() > radial_cut_min) & (
        s.r() <= radial_cut_max))  # Mask the data: select stellar
    # particles inside different physical spatial regimes.

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
    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), SFR, sfg_ratio, wg_ratio, hg_ratio, np.sum(
        gas_mass), stellar_mass


def sfr_stars_gas_regimes(pdf, data, region, read):
    """
    Plot the evolution of star formation rate, stellar mass and gas mass for
    different spatial regimes for Auriga halo(es).
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
        radial_cuts_min, radial_cuts_max = (0.0, 1e-3, 5e-3), (1e-3, 5e-3, 15e-3)
    elif region == 'inner':
        radial_cuts_min, radial_cuts_max = (0.0, 2.5e-4, 5e-4), (2.5e-4, 5e-4, 7.5e-4)

    # Read the data #
    if read is True:
        # Read desired galactic property(ies) for specific particle type(s)
        # for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'gima', 'mass', 'pos']
        data.select_haloes(default_level, default_redshift, loadonlyhalo=0, loadonlytype=particle_type,
                           loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Select the halo and rotate it based on its principal axes so
            # galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            # Loop over all spatial regimes #
            for radial_cut_min, radial_cut_max in zip(radial_cuts_min, radial_cuts_max):
                # Check if a folder to save the data exists, if not then
                # create one #
                path = '/u/di43/Auriga/plots/data/' + 'ssgr/' + str(radial_cut_max) + '/'
                if not os.path.exists(path):
                    os.makedirs(path)

                # Check if halo's data already exists, if not then read it #
                names = glob.glob(path + 'name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                if str(s.haloname) in names:
                    continue
                else:
                    print("Reading data for halo:", str(s.haloname))

                # Get the lookback times and calculate the initial masses #
                stellar_mask, = np.where((s.data['age'] > 0.) & (s.r() > radial_cut_min) & (
                    s.r() <= radial_cut_max))  # Mask the data: select
                # stellar particles inside radial limits.
                lookback_times = s.cosmology_get_lookback_time_from_a(s.data['age'][stellar_mask],
                                                                      is_flat=True)  # In Gyr.
                weights = s.data['gima'][stellar_mask] * 1e10 / 1e9 / time_bin_width  # In Msun yr ^ -1.

                # Save data for each halo in numpy arrays #
                np.save(path + 'name_' + str(s.haloname), s.haloname)
                np.save(path + 'weights_' + str(s.haloname), weights)
                np.save(path + 'lookback_times_SFR_' + str(s.haloname), lookback_times)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Loop over all spatial regimes #
            for radial_cut_min, radial_cut_max in zip(radial_cuts_min, radial_cuts_max):
                path = '/u/di43/Auriga/plots/data/' + 'ssgr/' + str(radial_cut_max) + '/'

                # Get all snapshots with redshift less than the redshift cut #
                redshifts = halo.get_redshifts()
                redshift_mask, = np.where(redshifts <= redshift_cut)
                snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]

                ssgr_data = np.array(
                    get_ssgr_data(snapshot_ids, halo, radial_cut_min, radial_cut_max))  # Get SFR stars gas
                # regimes data.

                # Save data for each halo in numpy arrays #
                np.save(path + 'lookback_times_' + str(name), ssgr_data[:, 0])
                np.save(path + 'sfg_ratios_' + str(name), ssgr_data[:, 2])
                np.save(path + 'wg_ratios_' + str(name), ssgr_data[:, 3])
                np.save(path + 'hg_ratios_' + str(name), ssgr_data[:, 4])
                np.save(path + 'gas_masses_' + str(name), ssgr_data[:, 5])
                np.save(path + 'stellar_masses_' + str(name), ssgr_data[:, 6])

    # Loop over all radial limits #
    for radial_cut_min, radial_cut_max in zip(radial_cuts_min, radial_cuts_max):
        path = '/u/di43/Auriga/plots/data/' + 'ssgr/' + str(radial_cut_max) + '/'
        path_modes = '/u/di43/Auriga/plots/data/' + 'AGNmd/'

        # Get the names and sort them #
        names = glob.glob(path + 'name_*')
        names.sort()

        # Loop over all available haloes #
        for i in range(len(names)):
            print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
            # Generate the figure and set its parameters #
            figure = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(3, 1, hspace=0.06, height_ratios=[1, 0.5, 0.5])
            axis00 = plt.subplot(gs[0, 0])
            axis10 = plt.subplot(gs[1, 0])
            axis20 = plt.subplot(gs[2, 0])

            for axis in [axis00, axis10, axis20]:
                axis2 = axis.twiny()
                plot_tools.set_axes_evolution(axis, axis2, ylabel=r'$\mathrm{'
                                                                  r'Gas\;fraction}$', aspect=None, size=20)
                if axis in [axis10, axis20]:
                    axis2.set_xlabel('')
                    axis2.set_xticklabels([])
            for axis in [axis00, axis10]:
                axis.set_xlabel('')
                axis.set_xticklabels([])

            # axis002 = axis00.twiny()
            # plot_tools.set_axes_evolution(axis00, axis002, ylim=[0, 8],
            # ylabel=r'$\mathrm{SFR/(M_\odot\;yr^{-1})}$', aspect=None)
            # axis00.set_xticklabels([])
            axis003 = axis00.twinx()
            plot_tools.set_axes(axis003, ylim=[1e55, 1e61], yscale='log', ylabel=r'$\mathrm{(Feedback\;energy)/ergs}$',
                                aspect=None, which='major')
            # plot_tools.set_axes(axis10, xlim=(13, 0), ylim=[1e6, 1e11],
            # yscale='log', ylabel=r'$\mathrm{Mass/M_{\odot}}$', aspect=None,
            # which='major')
            # axis10.set_xticklabels([])
            # plot_tools.set_axes(axis20, xlim=(13, 0), ylim=[-0.1, 1.1],
            # ylabel=r'$\mathrm{Gas\;fraction}$', xlabel=r'$\mathrm{t_{
            # look}/Gyr}$',
            #                     aspect=None)
            figure.text(0.01, 0.85, r'$\mathrm{Au-%s}$' '\n' r'$\mathrm{'
                                    r'%.0f<r/kpc\leq%.0f}$' % (
                            str(re.split('_|.npy', names[i])[1]), (np.float(radial_cut_min) * 1e3),
                            (np.float(radial_cut_max) * 1e3)), fontsize=20, transform=axis00.transAxes)

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
            lookback_times_modes = np.load(
                path_modes + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

            # Transform the arrays to comma separated strings and convert
            # each element to float #
            thermals = ','.join(thermals)
            mechanicals = ','.join(mechanicals)
            thermals = np.fromstring(thermals, dtype=np.float, sep=',')
            mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')

            # Plot the evolution of SFR #
            axis00.hist(lookback_times_SFR, weights=weights, histtype='step', bins=n_bins, range=[0, 13],
                        color=colors[i], label=r'$\mathrm{Au-%s}$' + (str(re.split('_|.npy', names[i])[1])))

            # Plot the feedback modes binned sum line #
            for mode, label, color in zip([mechanicals, thermals], [r'$\mathrm{Mechanical}$', r'$\mathrm{Thermal}$'],
                                          [colors[4], colors[5]]):
                x_value, sum = plot_tools.binned_sum(lookback_times_modes[np.where(mode > 0)], mode[np.where(mode > 0)],
                                                     n_bins=n_bins)
                axis003.plot(x_value, sum / time_bin_width, color=color, label=label)
                axis003.legend(loc='upper right', fontsize=20, frameon=False, numpoints=1, ncol=2)

            # Plot the stellar and gaseous masses #
            axis10.plot(lookback_times, gas_masses * 1e10, color=colors[3], label=r'$\mathrm{Gas}$')
            axis10.plot(lookback_times, stellar_masses * 1e10, color=colors[1], label=r'$\mathrm{Stars}$')

            # Plot the gas fractions #
            axis20.plot(lookback_times, sfg_ratios, color=colors[3], label=r'$\mathrm{Cold\;gas}$')
            axis20.plot(lookback_times, wg_ratios, color=colors[2], label=r'$\mathrm{Warm\;gas}$')
            axis20.plot(lookback_times, hg_ratios, color=colors[1], label=r'$\mathrm{Hot\;gas}$')

            # Create the legends, save and close the figure #
            axis10.legend(loc='lower center', fontsize=20, frameon=False, numpoints=1, ncol=2)
            axis20.legend(loc='upper left', fontsize=20, frameon=False, numpoints=1)
            pdf.savefig(figure, bbox_inches='tight')
            plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_gfml_data(snapshot_ids, halo):
    """
    Parallelised method to the gas inflow/outflow.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :return: lookback time, SFR, sfg_ratio, wg_ratio, hg_ratio, np.sum(gas_mass), stellar_mass.
    """
    print("Invoking get_gfml_data")
    # Read desired galactic property(ies) for specific particle type(s) for
    # Auriga halo(es) #
    particle_type = [0, 4]
    attributes = ['mass', 'pos', 'id', 'sfr', 'vel']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's
    # spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    age = np.zeros(s.npartall)
    age[s.data['type'] == 4] = s.data['age']
    stellar_mask, = np.where((s.data['type'] == 4) & (age > 0.0))  # Mask the data: select stellar particles.
    wind_mask, = np.where((s.data['type'] == 4) & (age < 0.0))  # Mask the data: select wind particles.
    gas_mask, = np.where(s.data['type'] == 0)  # Mask the data: select gas cells.

    # Calculate the temperature of the gas cells #
    ne = s.data['ne'][gas_mask]
    metallicity = s.data['gz'][gas_mask]
    XH = s.data['gmet'][gas_mask, element['H']]
    yhelium = (1 - XH - metallicity) / (4. * XH)
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    temperature = GAMMA_MINUS1 * s.data['u'][gas_mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN

    # Calculate the radial velocity #
    spherical_radius = np.sqrt(np.sum(s.data['pos'][gas_mask, :] ** 2, axis=1))
    CoM_velocity = np.sum(s.data['vel'][gas_mask, :] * s.data['mass'][gas_mask][:, None], axis=0) / np.sum(
        s.data['mass'][gas_mask])
    radial_velocity = np.divide(np.sum((s.data['vel'][gas_mask] - CoM_velocity) * s.data['pos'][gas_mask], axis=1),
                                spherical_radius)
    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), s.subfind.data['frc2'][0], s.data['sfr'][
        gas_mask], s.data['mass'][gas_mask], s.data['id'][gas_mask], temperature, spherical_radius, radial_velocity, \
           s.data['mass'][stellar_mask], s.data['mass'][wind_mask], s.data['pos'][gas_mask]


def gas_flow_mass_loading(pdf, data, read, method):
    """
    Plot the evolution of gas flow and mass loading for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :param method: method to calculate flows.
    :return: None
    """
    print("Invoking gas_flow")
    redshift_cut = 7
    dT = 250  # In Myr.
    path = '/u/di43/Auriga/plots/data/' + 'gfml/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + 'name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if name in names:
                continue
            else:
                print("Reading data for halo:", name)

            # Get all snapshots with redshift less than the redshift cut #
            redshifts = halo.get_redshifts()
            redshift_mask, = np.where(redshifts <= redshift_cut)
            snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]

            gf_data = np.array(get_gfml_data(snapshot_ids, halo))  # Get gas flow data.

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(name), name)
            np.save(path + 'lookback_times_' + str(name), gf_data[:, 0])
            np.save(path + 'Rvirs_' + str(name), gf_data[:, 1])
            np.save(path + 'sfrs_' + str(name), gf_data[:, 2])
            np.save(path + 'gas_masses_' + str(name), gf_data[:, 3])
            np.save(path + 'ids_' + str(name), gf_data[:, 4])
            np.save(path + 'temperatures_' + str(name), gf_data[:, 5])
            np.save(path + 'spherical_radii_' + str(name), gf_data[:, 6])
            np.save(path + 'radial_velocities_' + str(name), gf_data[:, 7])
            np.save(path + 'stellar_masses_' + str(name), gf_data[:, 8])
            np.save(path + 'wind_masses_' + str(name), gf_data[:, 9])
            np.save(path + 'positions_' + str(name), gf_data[:, 10])

    # Get the names and sort them #
    names = glob.glob(path + 'name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Generate the figure and set its parameters #
        figure = plt.figure(figsize=(20, 7.5))
        gs = gridspec.GridSpec(1, 2, wspace=0.3)
        axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
        axis002 = axis00.twiny()
        axis012 = axis01.twiny()
        axis003 = axis00.twinx()
        axis003.yaxis.label.set_color('tab:red')
        axis003.spines['right'].set_color('tab:red')
        plot_tools.set_axes(axis003, ylim=[0, 250], xlabel=r'$\mathrm{t_{look}/Gyr}$',
                            ylabel=r'$\mathrm{0.5R_{vir}/kpc}$', aspect=None)
        axis003.tick_params(axis='y', direction='out', left='off', colors='tab:red')
        plot_tools.set_axes_evolution(axis00, axis002, ylim=[-250, 0], ylabel=r'$\mathrm{Net\;flow/('r'M_\odot/yr)}$',
                                      aspect=None)
        plot_tools.set_axes_evolution(axis01, axis012, ylim=[1e-3, 1e2], yscale='log', ylabel=r'$\mathrm{Loading}$',
                                      aspect=None)
        axis00.tick_params(axis='y', direction='out', colors='black')
        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis00.transAxes)

        # Load and plot the data #
        sfrs = np.load(path + 'sfrs_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
        gas_masses = np.load(path + 'gas_masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
        stellar_masses = np.load(path + 'stellar_masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy',
                                 allow_pickle=True)
        wind_masses = np.load(path + 'wind_masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
        Rvirs = np.load(path + 'Rvirs_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
        ids = np.load(path + 'ids_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
        # temperatures = np.load(path + 'temperatures_' + str(re.split('_|.npy', names[i])[1]) + '.npy',
        #     allow_pickle=True)

        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy',
                                 allow_pickle=True)
        spherical_radii = np.load(path + 'spherical_radii_' + str(re.split('_|.npy', names[i])[1]) + '.npy',
                                  allow_pickle=True)
        radial_velocities = np.load(path + 'radial_velocities_' + str(re.split('_|.npy', names[i])[1]) + '.npy',
                                    allow_pickle=True)

        # Declare arrays to store the data #
        mass_outflows, mass_inflows, mass_loading, wind_loading = np.zeros(len(lookback_times)), np.zeros(
            len(lookback_times)), np.zeros(len(lookback_times)), np.zeros(len(lookback_times))

        if method == 'ids':
            common_ids = set(ids[0])
            # Loop over all redshifts and compare ids #
            for ids_in_snap in ids[1:]:
                common_ids.intersection_update(ids_in_snap)
            print(len(common_ids))
            x = [8796187878443, 8796221432877, 5466789209133, 8796221432879, 8796240307248, 5475303646257,
                 8796181586994, 8796221432881, 8796238210097, 8796236112953, 5463649772607, 8796238210112,
                 8796231918660, 8796240307271, 8796240307272, 8796223530058, 8796240307277, 8796231918670,
                 8796240307278, 8796225627216, 8796236112974, 8796206752852, 5475303646293, 8796208850006,
                 5463574275157, 8796206752854, 8796240307285, 8796240307286, 8796236112987, 8796240307287,
                 8796236112989, 8796150129757, 8796236112991, 8796240307294, 5463549109346, 8796240307302,
                 8796240307307, 8796236113004, 8796240307309, 8796240307310, 5471491023984, 5475911820403,
                 5463435863160, 8796221432963, 8796217238666, 8796240307339, 8796240307349, 8796240307350,
                 5463540720794, 87962403073]
            print(np.where(ids[0] == x))
            print(ids[0] in x)
            for i in range(len(lookback_times)):
                plt.scatter(spherical_radii[i][np.where(ids[i] == x), 2], spherical_radii[i][np.where(ids[i] == x), 1])

        elif method == 'time_interval':
            # Loop over all radial limits #
            for k, radial_cut in enumerate([0.5]):
                for j in range(len(lookback_times)):
                    outflow_mask, = np.where((spherical_radii[j] < radial_cut * Rvirs[j]) & (spherical_radii[j] + (
                        radial_velocities[j] * u.km.to(u.Mpc) / u.second.to(u.Myr)) * dT > radial_cut * Rvirs[j]))
                    inflow_mask, = np.where((spherical_radii[j] > radial_cut * Rvirs[j]) & (spherical_radii[j] + (
                        radial_velocities[j] * u.km.to(u.Mpc) / u.second.to(u.Myr)) * dT < radial_cut * Rvirs[j]))
                    mass_outflows[j] = np.divide(np.sum(gas_masses[j][outflow_mask]) * 1e10, dT * 1e6)
                    mass_inflows[j] = np.divide(np.sum(gas_masses[j][inflow_mask]) * 1e10, dT * 1e6)
                    mass_loading[j] = mass_outflows[j] / np.sum(sfrs[j])
                    wind_loading[j] = np.sum(wind_masses[j]) / np.sum(stellar_masses[j])

        elif method == 'percentage':
            # Loop over all radial limits #
            for k, radial_cut in enumerate([0.5]):
                for j in range(len(lookback_times)):
                    outflow_mask, = np.where((spherical_radii[j] < radial_cut * Rvirs[j]) & (
                        spherical_radii[j] > 0.95 * radial_cut * Rvirs[j]) & (radial_velocities[j] > 0))
                    inflow_mask, = np.where((spherical_radii[j] > radial_cut * Rvirs[j]) & (
                        spherical_radii[j] < 1.05 * radial_cut * Rvirs[j]) & (radial_velocities[j] < 0))
                    mass_outflows[j] = np.sum(gas_masses[j][outflow_mask]) * 1e10
                    mass_inflows[j] = np.sum(gas_masses[j][inflow_mask]) * 1e10
                    mass_loading[j] = mass_outflows[j] / np.sum(sfrs[j])
                    wind_loading[j] = np.sum(wind_masses[j]) / np.sum(stellar_masses[j])

        elif method == 'shell':
            # Loop over all radial limits #
            for k, radial_cut in enumerate([0.5]):
                # for k, radial_cut in enumerate([0.01, 0.1, 0.5, 1]):
                for j in range(len(lookback_times)):
                    outflow_mask, = np.where((spherical_radii[j] > radial_cut * Rvirs[j]) & (
                        spherical_radii[j] < 1e-3 + radial_cut * Rvirs[j]) & (radial_velocities[j] > 0))
                    inflow_mask, = np.where((spherical_radii[j] > radial_cut * Rvirs[j]) & (
                        spherical_radii[j] < 1e-3 + radial_cut * Rvirs[j]) & (radial_velocities[j] < 0))
                    mass_outflows[j] = np.divide(np.sum(gas_masses[j][outflow_mask] * np.abs(
                        radial_velocities[j][outflow_mask] * u.km.to(u.Mpc) / u.second.to(u.yr))), 1e-3) * 1e10
                    mass_inflows[j] = np.divide(np.sum(gas_masses[j][inflow_mask] * np.abs(
                        radial_velocities[j][inflow_mask] * u.km.to(u.Mpc) / u.second.to(u.yr))), 1e-3) * 1e10
                    mass_loading[j] = mass_outflows[j] / np.sum(sfrs[j])
                    wind_loading[j] = np.sum(wind_masses[j]) / np.sum(stellar_masses[j])

        # Plot the evolution of gas flow and mass loading #
        net_flow = mass_inflows - mass_outflows
        axis00.plot(lookback_times, net_flow, color=colors[-3])
        axis003.plot(lookback_times, radial_cut * Rvirs * 1e3, c=colors[1], linestyle='dashed')
        axis01.plot(lookback_times, mass_loading, c=colors[2], label=r'$\mathrm{Mass\;loading}$')
        axis01.plot(lookback_times, wind_loading, c=colors[3], label=r'$\mathrm{Wind\;loading}$')

        # Create the legends, save and close the figure #
        axis00.legend(loc='upper right', fontsize=20, frameon=False, numpoints=1)
        axis01.legend(loc='upper right', fontsize=20, frameon=False, numpoints=1)
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def AGN_modes_distribution(date, data, read):
    """
    Get the energy of different black hole feedback modes from log files and
    plot its evolution for Auriga halo(es).
    :param date: date.
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
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s)
        # for Auriga halo(es) #
        data.select_haloes(default_level, default_redshift, loadonlyhalo=0, loadonlytype=None, loadonly=None)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + 'name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Reading data for halo:", str(s.haloname))

            # Declare arrays to store the desired words and lines that
            # contain these words #
            redshift_lines, redshifts, feedback_lines, thermals, mechanicals = [], [], [], [], []

            with open('/u/di43/Auriga/output/halo_' + str(s.haloname) + '/Au-' + str(s.haloname) + '.txt') as file:
                # Loop over all lines #
                for line in file:
                    # Convert the characters in the line to lowercase and
                    # split the line into words #
                    line = line.lower()
                    line = line.strip()  # Remove '\n' at end of line.
                    words = line.split(" ")

                    # Search for the word 'redshift:' and get the next word
                    # which is the redshift value #
                    if 'redshift:' in words:
                        redshift_lines.append(line)
                        redshifts.append(words[words.index('redshift:') + 1])

                    # Search for the words 'thermal' and 'mechanical' and
                    # get the words after next which are the energy values
                    # in ergs #
                    if 'black_holes:' in words and '(step)' in words and 'is' in words and 'thermal' in words and \
                        'mechanical' in words:
                        feedback_lines.append(line)
                        thermals.append(words[words.index('thermal') + 2])
                        mechanicals.append(words[words.index('mechanical') + 2])

            file.close()  # Close the opened file.

            #  Convert redshifts to lookback times in Gyr #
            redshifts = [re.sub(',', '', i) for i in redshifts]  # Remove the commas at the end of each
            # redshift string.
            redshifts = ','.join(redshifts)  # Transform the arrays to comma separated strings.
            redshifts = np.fromstring(redshifts, dtype=np.float, sep=',')  # Convert each element to
            # float.
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
    plot_tools.set_axes_evolution(axis10, axis102, yscale='log', ylim=[1e51, 1e61], ylabel=r'$\mathrm{('
                                                                                           r'Mechanical\;feedback'
                                                                                           r'\;energy)/ergs}$',
                                  aspect=None, which='major')
    plot_tools.set_axes_evolution(axis11, axis112, ylim=[1e51, 1e61], ylabel=r'$\mathrm{('
                                                                             r'Thermal\;feedback\;energy)/ergs}$',
                                  aspect=None, which='major')
    plot_tools.set_axes_evolution(axis12, axis122, ylim=[1e51, 1e61], ylabel=r'$\mathrm{('
                                                                             r'Thermal\;feedback\;energy)/ergs}$',
                                  aspect=None, which='major')

    # Get the names and sort them #
    names = glob.glob(path + 'name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        if i == 0:
            mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            # Transform the arrays to comma separated strings and convert
            # each element to float #
            mechanicals = ','.join(mechanicals)
            mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
        thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        # Transform the arrays to comma separated strings and convert each
        # element to float #
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
            hb = axis.hexbin(lookback_times[np.where(mode > 0)], mode[np.where(mode > 0)], bins='log', yscale='log',
                             cmap='hot_r')
            plot_tools.create_colorbar(axiscbar, hb, label=r'$\mathrm{Counts\;per\;hexbin}$', orientation='horizontal')
            x_value, sum = plot_tools.binned_sum(lookback_times[np.where(mode > 0)], mode[np.where(mode > 0)],
                                                 n_bins=n_bins)
            axis.plot(x_value, sum / time_bin_width, color=colors[0], label=r'$\mathrm{Sum}$')
            figure.text(0.01, 0.95, 'Au-' + str(re.split('_|.npy', names[i])[1]), fontsize=20, transform=axis.transAxes)

            # Set colour-bar parameters #
            axiscbar.xaxis.tick_top()
            axiscbar.xaxis.set_label_position("top")
            axiscbar.tick_params(direction='out', which='both', top='on', right='on')

    # Create the legends, save and close the figure #
    axis10.legend(loc='upper right', fontsize=20, frameon=False, numpoints=1)
    axis11.legend(loc='upper right', fontsize=20, frameon=False, numpoints=1)
    axis12.legend(loc='upper right', fontsize=20, frameon=False, numpoints=1)
    plt.savefig('/u/di43/Auriga/plots/' + 'AGNmd-' + date + '.png', bbox_inches='tight')
    plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_afk_data(snapshot_ids, halo):
    """
    Parallelised method to get the black hole radius and the volume of gas
    cells within that.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :return: lookback time, , gas_volumes, sf_gas_volumes, nsf_gas_volumes,
    blackhole_hsml.
    """
    print("Invoking get_afk_data")
    # Read desired galactic property(ies) for specific particle type(s) for
    # Auriga halo(es) #
    particle_type = [0, 4, 5]
    attributes = ['age', 'bhhs', 'id', 'mass', 'pos', 'sfr', 'vol']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's
    # spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    # Check that only one (the closest) black hole is selected #
    blackhole_mask, = np.where((s.data['type'] == 5) & (
        s.r() < 0.1 * s.subfind.data['frc2'][0]))  # Mask the data: select the black hole inside 0.1*R200c.
    if len(blackhole_mask) == 1:
        blackhole_hsml = s.data['bhhs'][0]

    # In mergers select the most massive black hole #
    elif len(blackhole_mask) > 1:
        blackhole_id = s.data['id'][s.data['type'] == 5]
        id_mask, = np.where(blackhole_id == s.data['id'][s.data['mass'].argmax()])
        blackhole_hsml = s.data['bhhs'][id_mask[0]]
    else:
        blackhole_hsml = 0

    # Convert black hole hsml to physical h-free units i.e. keep it
    # constant in co-moving space. #
    a = 1 / (1 + s.redshift)
    blackhole_hsml = blackhole_hsml * a / 0.6777

    # Mask the data: select star-forming and not gas cells inside the black
    # hole's radius #
    gas_mask, = np.where(s.r()[s.data['type'] == 0] < blackhole_hsml)
    sf_gas_mask, = np.where((s.data['sfr'] > 0.0) & (s.r()[s.data['type'] == 0] < blackhole_hsml))
    nsf_gas_mask, = np.where((s.data['sfr'] <= 0.0) & (s.r()[s.data['type'] == 0] < blackhole_hsml))

    # Compute the total volume of cells with SFR == 0 and compare it to the
    # total volume of all cells inside the black hole's radius #
    gas_volumes = s.data['vol'][gas_mask].sum() * 1e9 * a ** 3 / 0.6777 ** 3  # In
    # kpc^-3.
    sf_gas_volumes = s.data['vol'][sf_gas_mask].sum() * 1e9 * a ** 3 / 0.6777 ** 3  #
    # In kpc^-3.
    nsf_gas_volumes = s.data['vol'][nsf_gas_mask].sum() * 1e9 * a ** 3 / 0.6777 ** 3
    # In kpc^-3.
    return s.cosmology_get_lookback_time_from_a(s.time,
                                                is_flat=True), gas_volumes, sf_gas_volumes, nsf_gas_volumes, \
           blackhole_hsml


def AGN_feedback_kernel(pdf, data, ds, read):
    """
    Plot the evolution of black hole radius and the volume of gas cells
    within that for Auriga halo(es).
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
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + 'name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if name in names:
                continue
            else:
                print("Reading data for halo:", name)

            # Get all snapshots with redshift less than the redshift cut #
            redshifts = halo.get_redshifts()
            if ds is True:
                # Downsample the 3000 snapshots run #
                for name, halo in haloes.items():
                    if name == '18':
                        redshifts = halo.get_redshifts()
            redshift_mask, = np.where(redshifts <= redshift_cut)
            snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]

            ssgr_data = np.array(get_afk_data(snapshot_ids, halo))  # Get AGN feedback
            # kernel data.

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
    names = glob.glob(path + 'name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        axis3 = axis.twinx()
        axis3.yaxis.label.set_color('tab:red')
        axis3.spines['right'].set_color('tab:red')
        plot_tools.set_axes(axis3, ylim=[-0.1, 1.1], xlabel=r'$\mathrm{t_{look}/Gyr}$',
                            ylabel=r'$\mathrm{BH_{sml}/kpc}$', aspect=None)
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-0.1, 1.1], ylabel=r'$\mathrm{V_{nSFR}(r<BH_{'
                                                                            r'sml})/V_{all}(r<BH_{sml})}$', aspect=None)
        axis3.tick_params(axis='y', direction='out', left='off', colors='tab:red')
        figure.text(0.01, 0.95, 'Au-' + str(re.split('_|.npy', names[i])[1]), fontsize=20, transform=axis.transAxes)

        # Load and plot the data #
        gas_volumes = np.load(path + 'gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        nsf_gas_volumes = np.load(path + 'nsf_gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        blackhole_hsmls = np.load(path + 'blackhole_hsmls_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the evolution of black hole radius and the volume ratio of
        # gas cells within that #
        axis3.scatter(lookback_times, blackhole_hsmls * 1e3, edgecolor='None', color=colors[1])
        axis.scatter(lookback_times, nsf_gas_volumes / gas_volumes, edgecolor='None', color=colors[0])

        # Plot median and 1-sigma lines #
        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(lookback_times, nsf_gas_volumes / gas_volumes,
                                                                       bin_type='equal_number',
                                                                       n_bins=len(lookback_times) / 5, log=False)
        axis.plot(x_value[x_value > 0], median[x_value > 0], color=colors[0], linewidth=3)
        axis.fill_between(x_value[x_value > 0], shigh[x_value > 0], slow[x_value > 0], color=colors[0], alpha='0.3')

        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(lookback_times, blackhole_hsmls * 1e3,
                                                                       bin_type='equal_number',
                                                                       n_bins=len(lookback_times) / 5, log=False)
        axis3.plot(x_value[x_value > 0], median[x_value > 0], color=colors[1], linewidth=3)
        axis3.fill_between(x_value[x_value > 0], shigh[x_value > 0], slow[x_value > 0], color=colors[1], alpha='0.3')

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def AGN_feedback_active(pdf):
    """
    Calculate the energy deposition on active gas cells from the AGN
    feedback for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking AGN_feedback_smoothed")
    # Check if a folder to save the data exists, if not then create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNfs/'
    path_modes = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    path_kernel = '/u/di43/Auriga/plots/data/' + 'AGNfk/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Get the names and sort them #
    names = glob.glob(path_kernel + 'name_06.*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[1e56, 1e62], yscale='log', ylabel=r'$\mathrm{('
                                                                                           r'AGN\;feedback\;energy'
                                                                                           r')/ergs}$', aspect=None)
        figure.text(0.01, 0.95, 'Au-' + str(re.split('_|.npy', names[i])[1]), fontsize=20, transform=axis.transAxes)

        # Load and plot the data #
        thermals = np.load(path_modes + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        gas_volumes = np.load(path_kernel + 'gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path_kernel + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        nsf_gas_volumes = np.load(path_kernel + 'nsf_gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times_modes = np.load(path_modes + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Transform the arrays to comma separated strings and convert each
        # element to float #
        thermals = ','.join(thermals)
        thermals = np.fromstring(thermals, dtype=np.float, sep=',')

        # Calculate and plot the thermals energy sum #
        n_bins = len(lookback_times)
        time_bin_width = (13 - 0) / n_bins  # In Gyr.
        x_value, sum = plot_tools.binned_sum(lookback_times_modes[np.where(thermals > 0)],
                                             thermals[np.where(thermals > 0)], n_bins=int(len(lookback_times)) + 1)
        axis.plot(x_value, sum / time_bin_width, color=colors[1], label=r'$\mathrm{Sum}$')

        x, median, shigh, slow = plot_tools.binned_median_1sigma(lookback_times, nsf_gas_volumes / gas_volumes,
                                                                 bin_type='equal_number', n_bins=len(lookback_times),
                                                                 log=False)
        axis.plot(x_value, sum / time_bin_width * median, color=colors[0], label=r'$\mathrm{Sum}$')

        # axis.plot(lookback_times, sum / time_bin_width * (nsf_gas_volumes
        # / gas_volumes), color=colors[0])

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_bm_data(snapshot_ids, halo):
    """
    Parallelised method to get black hole properties.
    :param snapshot_ids: ids of the snapshots.
    :param halo: data for the halo.
    :return: black hole properties.
    """
    print("Invoking get_bm_data")
    # Read desired galactic property(ies) for specific particle type(s) for
    # Auriga halo(es) #
    particle_type = [4, 5]
    attributes = ['bhma', 'bcmr', 'bcmq', 'bhmd', 'bhmr', 'bhmq', 'id', 'pos']
    s = halo.snaps[snapshot_ids].loadsnap(loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

    # Select the halo and rotate it based on its principal axes so galaxy's
    # spin is aligned with the z-axis #
    s.calc_sf_indizes(s.subfind)
    s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

    # Check that only one (the closest) black hole is selected #
    blackhole_mask, = np.where(s.data['type'] == 5)
    if len(blackhole_mask) == 1:
        black_hole_mass = s.data['bhma'][0]
        black_hole_cmass_radio = s.data['bcmr'][0]
        black_hole_cmass_quasar = s.data['bcmq'][0]
        black_hole_dmass = s.data['bhmd'][0]
        black_hole_dmass_radio = s.data['bhmr'][0]
        black_hole_dmass_quasar = s.data['bhmq'][0]

    # In mergers select the most massive black hole #
    elif len(blackhole_mask) > 1:
        blackhole_id = s.data['id'][s.data['type'] == 5]
        id_mask, = np.where(blackhole_id == s.data['id'][s.data['mass'].argmax()])
        black_hole_mass = s.data['bhma'][id_mask[0]]
        black_hole_cmass_radio = s.data['bcmr'][id_mask[0]]
        black_hole_cmass_quasar = s.data['bcmq'][id_mask[0]]
        black_hole_dmass = s.data['bhmd'][id_mask[0]]
        black_hole_dmass_radio = s.data['bhmr'][id_mask[0]]
        black_hole_dmass_quasar = s.data['bhmq'][id_mask[0]]
    else:
        black_hole_mass, black_hole_cmass_radio, black_hole_cmass_quasar, black_hole_dmass, black_hole_dmass_radio, \
        black_hole_dmass_quasar = 0, 0, 0, 0, 0, 0
    return s.cosmology_get_lookback_time_from_a(s.time,
                                                is_flat=True), black_hole_mass, black_hole_cmass_radio, \
           black_hole_cmass_quasar, black_hole_dmass, black_hole_dmass_radio, black_hole_dmass_quasar


def blackhole_masses(pdf, data, read):
    """
    Plot the evolution of black hole mass, accretion rate, cumulative mass
    accreted onto the BH in the low and high accretion-state for Auriga
    halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking blackhole_masses")
    redshift_cut = 7
    n_bins = 130
    time_bin_width = (13 - 0) / n_bins  # In Gyr.
    path = '/u/di43/Auriga/plots/data/' + 'bm/'
    path_modes = '/u/di43/Auriga/plots/data/' + 'AGNmd/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Loop over all available haloes #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + 'name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if name in names:
                continue
            else:
                print("Reading data for halo:", name)

            # Get all snapshots with redshift less than the redshift cut #
            redshifts = halo.get_redshifts()
            redshift_mask, = np.where(redshifts <= redshift_cut)
            snapshot_ids = np.array(list(halo.snaps.keys()))[redshift_mask]

            # Get blackhole data #
            bm_data = np.array(get_bm_data(snapshot_ids, halo))  # Get blackhole masses data.
            lookback_times = bm_data[:, 0]
            black_hole_masses = bm_data[:, 1]
            black_hole_cmasses_radio, black_hole_cmasses_quasar = bm_data[:, 2], bm_data[:, 3]
            black_hole_dmasses, black_hole_dmasses_radio, black_hole_dmasses_quasar = bm_data[:, 4], bm_data[:,
                                                                                                     5], bm_data[:, 6]

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(name), name)
            np.save(path + 'lookback_times_' + str(name), lookback_times)
            np.save(path + 'black_hole_masses_' + str(name), black_hole_masses)
            np.save(path + 'black_hole_cmasses_radio_' + str(name), black_hole_cmasses_radio)
            np.save(path + 'black_hole_cmasses_quasar_' + str(name), black_hole_cmasses_quasar)
            np.save(path + 'black_hole_dmasses_' + str(name), black_hole_dmasses)
            np.save(path + 'black_hole_dmasses_radio_' + str(name), black_hole_dmasses_radio)
            np.save(path + 'black_hole_dmasses_quasar_' + str(name), black_hole_dmasses_quasar)

    # Get the names and sort them #
    names = glob.glob(path + 'name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Generate the figure and set its parameters #
        figure = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.2)
        axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
        axis10, axis11 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])
        axis002, axis012 = axis00.twiny(), axis01.twiny()
        axis102, axis112 = axis10.twiny(), axis11.twiny()

        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis00.transAxes)

        for axis in [axis01, axis11]:
            axis.yaxis.set_label_position("right")
            axis.yaxis.tick_right()

        plot_tools.set_axes_evolution(axis00, axis012, ylim=[1e2, 1e9], yscale='log',
                                      ylabel=r'$\mathrm{M_{BH,cumu}/M_\odot}$', which='major')
        plot_tools.set_axes_evolution(axis01, axis002, ylim=[1e2, 1e9], yscale='log',
                                      ylabel=r'$\mathrm{M_{BH,growth}/M_\odot}$', which='major')
        plot_tools.set_axes_evolution(axis10, axis102, yscale='log',
                                      ylabel=r'$\mathrm{\dot{M}_{BH}/(M_\odot\;Gyr^{-1})}$', which='major')
        plot_tools.set_axes_evolution(axis11, axis112, yscale='log', ylabel=r'$\mathrm{(AGN\;feedback\;energy)/ergs}$',
                                      which='major')

        # Load the data #
        thermals = np.load(path_modes + 'thermals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        mechanicals = np.load(path_modes + 'mechanicals_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times_modes = np.load(path_modes + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_masses = np.load(path + 'black_hole_masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_cmasses_radio = np.load(
            path + 'black_hole_cmasses_radio_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_cmasses_quasar = np.load(
            path + 'black_hole_cmasses_quasar_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_dmasses = np.load(path + 'black_hole_dmasses_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_dmasses_radio = np.load(
            path + 'black_hole_dmasses_radio_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        black_hole_dmasses_quasar = np.load(
            path + 'black_hole_dmasses_quasar_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Convert cumulative masses to instantaneous measurements #
        black_hole_imasses = np.insert(np.diff(black_hole_masses), 0, 0)
        black_hole_imasses_radio = np.insert(np.diff(black_hole_cmasses_radio), 0, 0)
        black_hole_imasses_quasar = np.insert(np.diff(black_hole_cmasses_quasar), 0, 0)

        # Plot black hole mass and accretion rates #
        axis00.plot(lookback_times, black_hole_masses * 1e10, color=colors[0])
        axis00.plot(lookback_times, black_hole_cmasses_radio * 1e10, color=colors[1])
        axis00.plot(lookback_times, black_hole_cmasses_quasar * 1e10, color=colors[2])
        axis01.plot(lookback_times, black_hole_imasses * 1e10, color=colors[0])
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
            x_value, sum = plot_tools.binned_sum(lookback_times_modes[np.where(mode > 0)], mode[np.where(mode > 0)],
                                                 n_bins=n_bins)
            axis.plot(x_value, sum / time_bin_width, color=colors[1 + i], label=r'$\mathrm{Sum}$')

        axis01.legend([plt10, plt102, plt101], [r'$\mathrm{BH}$', r'$\mathrm{Thermal}$', r'$\mathrm{Mechanical}$'],
                      loc='upper left', fontsize=20, frameon=False, numpoints=1)

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None
