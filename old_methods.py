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
from scripts.gigagalaxy.util import satellite_utilities

res = 512
boxsize = 0.06
default_level = 4
default_redshift = 0.0
colors = ['black', 'tab:red', 'tab:green', 'tab:blue', 'tab:orange']
element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}


def AGN_modes_cumulative(date, data, read):
    """
    Get information about different black hole modes from log files and plot the evolution of the cumulative feedback
    for Auriga halo(es).
    :param date: date.
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    path = '/u/di43/Auriga/plots/data/' + 'AGNmc/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        data.select_haloes(default_level, default_redshift, loadonlytype=None, loadonlyhalo=0, loadonly=None)

        # Loop over all available haloes #
        for s in data:
            # Declare arrays to store the desired words and lines that contain these words #
            redshift_lines, redshifts, feedback_lines, thermals, mechanicals = [], [], [], [], []

            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue

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

                    # Search for the words 'thermal' and 'mechanical' and get the words after next which are the
                    # energy values in ergs #
                    if 'black_holes:' in words and '(cumulative)' in words and 'is' in words and 'thermal' in words \
                        and 'mechanical' in words:
                        feedback_lines.append(line)
                        thermals.append(words[words.index('thermal') + 2])
                        mechanicals.append(words[words.index('mechanical') + 2])

            file.close()  # Close the opened file.

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'thermals_' + str(s.haloname), thermals)
            np.save(path + 'redshifts_' + str(s.haloname), redshifts)
            np.save(path + 'mechanicals_' + str(s.haloname), mechanicals)

    # Get the names and sort them #
    names = glob.glob(path + '/name_06NOR*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
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
        figure.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), fontsize=16, transform=axis00.transAxes)

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

        feedback_mask, = np.where((mechanicals != 0) | (thermals != 0))
        sc = axis00.scatter(thermals[feedback_mask], mechanicals[feedback_mask], edgecolor='None', s=50,
                            c=lookback_times[feedback_mask], vmin=0, vmax=max(lookback_times), cmap='jet')
        cb = plt.colorbar(sc, cax=axiscbar)
        cb.set_label(r'$\mathrm{t_{look}\;[Gyr]}$', size=16)
        axiscbar.tick_params(direction='out', which='both', right='on', left='on')
        axiscbar.yaxis.tick_left()

        # Create the legends, save and close the figure #
        axis00.legend([plot000, plot001], [r'$\mathrm{1:10}$', r'$\mathrm{1:50}$'], loc='upper center', fontsize=16,
                      frameon=False, numpoints=1)
        axis00.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmc-' + date + '.png', bbox_inches='tight')
        plt.close()
    return None


def AGN_modes_histogram(date, data, read):
    """
    Get information about different black hole modes from log files and plot a histogram of the evolution of the step
    feedback for Auriga halo(es).
    :param date: date.
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        path = '/u/di43/Auriga/plots/data/' + 'AGNmh/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        data.select_haloes(default_level, default_redshift, loadonlytype=None, loadonlyhalo=0, loadonly=None)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
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

                    # Search for the words 'thermal' and 'mechanical' and get the words after next which are the
                    # energy values in ergs #
                    if 'black_holes:' in words and '(step)' in words and 'is' in words and 'thermal' in words and \
                        'mechanical' in words:
                        feedback_lines.append(line)
                        thermals.append(words[words.index('thermal') + 2])
                        mechanicals.append(words[words.index('mechanical') + 2])

            file.close()  # Close the opened file.

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'thermals_' + str(s.haloname), thermals)
            np.save(path + 'redshifts_' + str(s.haloname), redshifts)
            np.save(path + 'mechanicals_' + str(s.haloname), mechanicals)

    # Get the names and sort them #
    names = glob.glob(path + '/name_06NOR*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):

        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        axis.grid(True, color='gray', linestyle='-')
        axis2 = axis.twiny()
        axis.set_yscale('log')
        plot_tools.set_axes_evo(axis, axis2, ylabel=r'$\mathrm{AGN\;feedback\;energy\;[ergs]}$')
        axis.text(0.01, 0.95, 'Au-' + str(re.split('_|.npy', names[0])[1]), fontsize=16, transform=axis.transAxes)

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

        # Create the legends, save and close the figure #
        axis.legend([hist0, hist1], [r'$\mathrm{Thermal}$', r'$\mathrm{Mechanical}$'], loc='upper right', fontsize=16,
                    frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmh-' + date + '.png', bbox_inches='tight')
        plt.close()

    return None


def AGN_modes_step(date, data, read):
    """
    Get information about different black hole modes from log files and plot the step feedback for Auriga halo(es).
    :param date: date.
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        path = '/u/di43/Auriga/plots/data/' + 'AGNms/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        data.select_haloes(default_level, default_redshift, loadonlytype=None, loadonlyhalo=0, loadonly=None)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
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

                    # Search for the words 'thermal' and 'mechanical' and get the words after next which are the
                    # energy values in ergs #
                    if 'black_holes:' in words and '(step)' in words and 'is' in words and 'thermal' in words and \
                        'mechanical' in words:
                        feedback_lines.append(line)
                        thermals.append(words[words.index('thermal') + 2])
                        mechanicals.append(words[words.index('mechanical') + 2])

            file.close()  # Close the opened file.

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'thermals_' + str(s.haloname), thermals)
            np.save(path + 'mechanicals_' + str(s.haloname), mechanicals)

    # Get the names and sort them #
    names = glob.glob(path + '/name_17NOR*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):

        # Generate the figure and set its parameters #
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
        figure.text(1.01, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), fontsize=16, transform=axis10.transAxes)

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
        axis00.hist(mechanicals, bins=np.linspace(0, 2e56, 100), histtype='step', weights=weights,
                    orientation='vertical', color='k')
        weights = np.ones_like(thermals) / float(len(thermals))
        axis11.hist(thermals, bins=np.linspace(0, 5e57, 100), histtype='step', weights=weights,
                    orientation='horizontal', color='k')

        # Save and close the figure #
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNms-' + date + '.png', bbox_inches='tight')
        plt.close()

    return None


def gas_temperature_fraction(pdf, data, read):
    """
    Calculate the evolution of gas fraction in different temperature regimes for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    redshift_cut = 7

    # Read the data #
    if read is True:
        sfg_ratios, wg_ratios, hg_ratios, masses = [], [], [], []  # Declare lists to store the data.

        # Check if a folder to save the data exists, if not then create one #
        path = '/u/di43/Auriga/plots/data/' + 'gtfe/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Get all available redshifts #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()

        # Loop over all desired redshifts #

        for redshift in redshifts[np.where(redshifts <= redshift_cut)]:
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [0, 4]
            attributes = ['age', 'mass', 'ne', 'pos', 'rho', 'u']
            data.select_haloes(default_level, default_redshift, loadonlytype=particle_type, loadonlyhalo=0,
                               loadonly=attributes)

            # Loop over all available haloes #
            for s in data:
                # Check if halo's data already exists, if not then read it #
                names = glob.glob(path + '/name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                if str(s.haloname) in names:
                    continue

                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the
                # z-axis #
                s.calc_sf_indizes(s.subfind)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

                # Calculate the temperature of the gas cells #
                gas_mask, = np.where((s.data['type'] == 0) & (s.r() < s.subfind.data['frc2'][
                    0]))  # Mask the data: select gas cells inside the virial radius R200.
                ne = s.data['ne'][gas_mask]
                metallicity = s.data['gz'][gas_mask]
                XH = s.data['gmet'][gas_mask, element['H']]
                yhelium = (1 - XH - metallicity) / (4. * XH)
                mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
                temperature = GAMMA_MINUS1 * s.data['u'][gas_mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN

                # Calculate the mass of the gas cells inside three temperatures regimes #
                mass = s.data['mass'][gas_mask]
                masses.append(np.sum(mass))
                sfg_ratios.append(np.sum(mass[np.where(temperature < 2e4)]) / np.sum(mass))
                wg_ratios.append(np.sum(mass[np.where((temperature >= 2e4) & (temperature < 5e5))]) / np.sum(mass))
                hg_ratios.append(np.sum(mass[np.where(temperature >= 5e5)]) / np.sum(mass))

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
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 10))
        plt.grid(True, color='gray', linestyle='-')
        plt.ylim(-0.2, 1.2)
        plt.ylabel(r'$\mathrm{Gas\;fraction}$', size=16)
        plt.xlabel(r'$\mathrm{Redshift}$', size=16)
        axis.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16,
                  transform=axis.transAxes)
        axis2 = axis.twiny()
        plot_tools.set_axes_evo(axis, axis2, ylabel=r'$\mathrm{A_{2}}$')

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

        # for j in range(len(redshifts) - 1):  # b1, = plt.bar(redshifts[j], sfg_ratios[j], width=redshifts[j + 1] -
        # redshifts[j], alpha=0.6,
        # color='blue', align='edge', edgecolor='none')  # b2, = plt.bar(redshifts[j], wg_ratios[j],
        # bottom=sfg_ratios[j], width=redshifts[j + 1] -
        # redshifts[j], alpha=0.6, color='green',  #               align='edge', edgecolor='none')  # b3,
        # = plt.bar(redshifts[j], hg_ratios[j],
        # bottom=np.sum(np.vstack([sfg_ratios[j], wg_ratios[j]]).T),  #               width=redshifts[j + 1] -
        # redshifts[j], alpha=0.6,
        # align='edge', color='red', edgecolor='none')

        b1, = plt.plot(lookback_times, sfg_ratios, color='blue')
        b2, = plt.plot(lookback_times, wg_ratios, color='green')
        b3, = plt.plot(lookback_times, hg_ratios, color='red')

    # Create the legends, save and close the figure #
    plt.legend([b3, b2, b1], [r'$\mathrm{Hot\;gas}$', r'$\mathrm{Warm\;gas}$', r'$\mathrm{Cold\;gas}$'],
               loc='upper left', fontsize=16, frameon=False, numpoints=1)
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def AGN_modes_gas(date):
    """
    # Plot the evolution of gas fraction in different temperature regimes for Auriga halo(es).
    :param date: date.
    :return: None
    """
    path_gas = '/u/di43/Auriga/plots/data/' + 'gtfe/'
    path_modes = '/u/di43/Auriga/plots/data/' + 'AGNmd/'

    # Get the names and sort them #
    names = glob.glob(path_modes + '/name_06*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):

        # Generate the figure and set its parameters #
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
        axis.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), fontsize=16, transform=axis.transAxes)

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
        sum = np.zeros(n_bins)
        x_value = np.zeros(n_bins)
        x_low = min(lookback_times[np.where(thermals > 0)])
        for j in range(n_bins):
            index = np.where((lookback_times[np.where(thermals > 0)] >= x_low) & (
                    lookback_times[np.where(thermals > 0)] < x_low + 0.02))[0]
            x_value[j] = np.mean(np.absolute(lookback_times[np.where(thermals > 0)])[index])
            if len(index) > 0:
                sum[j] = np.sum(thermals[np.where(thermals > 0)][index])
            x_low += 0.02

        sum00, = plt.plot(x_value, sum, color='black', zorder=5, linestyle='dashed')

        # Calculate and plot the mechanical energy sum #
        n_bins = int(
            (max(lookback_times[np.where(mechanicals > 0)]) - min(lookback_times[np.where(mechanicals > 0)])) / 0.02)
        sum = np.zeros(n_bins)
        x_value = np.zeros(n_bins)
        x_low = min(lookback_times[np.where(mechanicals > 0)])
        for j in range(n_bins):
            index = np.where((lookback_times[np.where(mechanicals > 0)] >= x_low) & (
                    lookback_times[np.where(mechanicals > 0)] < x_low + 0.02))[0]
            x_value[j] = np.mean(np.absolute(lookback_times[np.where(mechanicals > 0)])[index])
            if len(index) > 0:
                sum[j] = np.sum(mechanicals[np.where(mechanicals > 0)][index])
            x_low += 0.02

        sum02, = plt.plot(x_value, sum, color='black', zorder=5)

        # Create the legends, save and close the figure #
        axis.legend([plot3, plot2, plot1], [r'$\mathrm{Hot\;gas}$', r'$\mathrm{Warm\;gas}$', r'$\mathrm{Cold\;gas}$'],
                    loc='upper left', fontsize=16, frameon=False, numpoints=1)
        axis2.legend([sum00, sum02], [r'$\mathrm{Thermal}$', r'$\mathrm{Mechanical}$'], loc='upper center', fontsize=16,
                     frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmg-' + date + '.png', bbox_inches='tight')
        plt.close()

    return None


def gas_temperature_fraction2(pdf, data, read):
    """
    Plot the fraction of cold, warm and hot gas at z=0.
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        path = '/u/di43/Auriga/plots/data/' + 'gtf/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['age', 'mass', 'ne', 'pos', 'rho', 'u']
        data.select_haloes(level, 0., loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue

            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            mask, = np.where((s.r() < s.subfind.data['frc2'][0]) & (
                    s.data['type'] == 0))  # Mask the data: select gas cells within the virial radius R200 #

            # Calculate the temperature of the gas cells #
            ne = s.data['ne'][mask]
            metallicity = s.data['gz'][mask]
            XH = s.data['gmet'][mask, element['H']]
            yhelium = (1 - XH - metallicity) / (4. * XH)
            mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
            temperature = GAMMA_MINUS1 * s.data['u'][mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN

            # Calculate the mass of the gas cells within three temperatures regimes #
            mass = s.data['mass'][mask]
            sfgmass = mass[np.where(temperature < 2e4)]
            warmgmass = mass[np.where((temperature >= 2e4) & (temperature < 5e5))]
            hotgmass = mass[np.where(temperature >= 5e5)]

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'sfg_ratio_' + str(s.haloname), np.sum(sfgmass) / np.sum(mass))
            np.save(path + 'wg_ratio_' + str(s.haloname), np.sum(warmgmass) / np.sum(mass))
            np.save(path + 'hg_ratio_' + str(s.haloname), np.sum(hotgmass) / np.sum(mass))

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 7.5))
    plt.grid(True, color='gray', linestyle='-')
    plt.ylim(-0.2, 1.2)
    plt.xlim(-0.2, 1.4)
    plt.ylabel(r'Gas fraction', size=16)

    # Loop over all available haloes #
    for i in range(len(names)):
        # Load and plot the data #
        sfg_ratio = np.load(path + 'sfg_ratio_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        wg_ratio = np.load(path + 'wg_ratio_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        hg_ratio = np.load(path + 'hg_ratio_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        b1, = plt.bar(np.divide(i, 5), sfg_ratio, width=0.1, alpha=0.6, color='blue')
        b2, = plt.bar(np.divide(i, 5), wg_ratio, bottom=sfg_ratio, width=0.1, alpha=0.6, color='green')
        b3, = plt.bar(np.divide(i, 5), hg_ratio, bottom=np.sum(np.vstack([sfg_ratio, wg_ratio]).T), width=0.1,
                      alpha=0.6, color='red')
    axis.set_xticklabels(np.append('', ['Au-' + re.split('_|.npy', halo)[1] for halo in names]))
    plt.legend([b3, b2, b1], [r'Hot gas', r'Warm gas', r'Cold gas'], loc='upper left', fontsize=12, frameon=False,
               numpoints=1)
    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


def delta_sfr_history(pdf, data, redshift, region, read):
    """
    Plot star formation rate history difference between Auriga haloes for three different spatial regimes (<1,
    1<5 and 5<15 kpc).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param region: inner or outer.
    :param read: boolean to read new data.
    :return: None
    """
    # Read the data #
    if read is True:
        if region == 'outer':
            radial_limits_min, radial_limits_max = (7.5e-4, 1e-3, 5e-3), (1e-3, 5e-3, 15e-3)
            texts = [r'$\mathrm{0<r/kpc<1}$', r'$\mathrm{1<r/kpc<5}$', r'$\mathrm{5<r/kpc<15}$']
        elif region == 'inner':
            radial_limits_min, radial_limits_max = (0.0, 2.5e-4, 5e-4), (2.5e-4, 5e-4, 7.5e-4)
            texts = [r'$\mathrm{0.00<r/kpc<0.25}$', r'$\mathrm{0.25<r/kpc<0.50}$', r'$\mathrm{0.50<r/kpc<0.75}$']

        # Get all available redshifts #
        haloes = data.get_haloes(level)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()

        # Loop over all desired redshifts #
        for redshift in redshifts:
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [4]
            attributes = ['age', 'gima', 'mass', 'pos']
            data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

            # Loop over all available haloes #
            for s in data:
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the
                # z-axis #
                # s.centerat(s.subfind.data['fpos'][0, :])  # Centre halo at the potential minimum.
                s.calc_sf_indizes(s.subfind)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

                # Loop over different radial limits #
                for radial_limit_min, radial_limit_max in zip(radial_limits_min, radial_limits_max):
                    SFRs, stellar_masses, redshifts_mask = [], [], []  # Declare lists to store the data.

                    # Check if a folder to save the data exists, if not then create one #
                    path = '/u/di43/Auriga/plots/data/' + 'dsh/' + str(radial_limit_max) + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    # Check if halo's data already exists, if not then read it #
                    # names = glob.glob(path + '/name_*')
                    # names = [re.split('_|.npy', name)[1] for name in names]
                    # if str(s.haloname) in names:
                    #     continue

                    # Mask the data and calculate the age and sfr for stellar particles within different spatial
                    # regimes #
                    a = 1 / (1 + redshift)  # Used to convert radial limits to physical.
                    stellar_mask, = np.where(
                        (s.data['age'] > 0.) & (s.r() > radial_limit_min * a) & (s.r() < radial_limit_max * a) & (
                                s.data['pos'][:, 2] < 0.003 * a))

                    stellar_mass = s.data['mass'][stellar_mask].sum()
                    time_lim = 0.5
                    SFR = s.data['gima'][stellar_mask][
                              (s.data['age'][stellar_mask] - s.time) < time_lim].sum() / time_lim * 10.

                    # Append the properties for all redshifts #
                    SFRs.append(SFR)
                    redshifts_mask.append(redshift)
                    stellar_masses.append(stellar_mass)

        # Save data for each halo in numpy arrays #
        np.save(path + 'SFRs_' + str(s.haloname), SFRs)
        np.save(path + 'name_' + str(s.haloname), s.haloname)
        np.save(path + 'redshifts_mask_' + str(s.haloname), redshifts_mask)
        np.save(path + 'stellar_masses_' + str(s.haloname), stellar_masses)

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.05)
    axis00 = plt.subplot(gs[0, 0])
    axis01 = plt.subplot(gs[0, 1])
    axis02 = plt.subplot(gs[0, 2])
    axis10 = plt.subplot(gs[1, 0])
    axis11 = plt.subplot(gs[1, 1])
    axis12 = plt.subplot(gs[1, 2])

    for axis in [axis00, axis01, axis02]:
        axis.set_ylim(0, 2e2)
        axis.set_yscale('log')
        axis.grid(True, color='gray', linestyle='-')
    for axis in [axis10, axis11, axis12]:
        axis.set_ylim(-1.1, 5e1)
        axis.grid(True, color='gray', linestyle='-')
        axis.set_yscale('symlog', subsy=[2, 3, 4, 5, 6, 7, 8, 9], linthreshy=1, linscaley=0.1)
    for axis in [axis01, axis02, axis11, axis12]:
        axis.set_yticklabels([])
    axis10.set_ylabel('$\mathrm{(\delta Sfr)_{norm}}$')
    axis00.set_ylabel('$\mathrm{Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$')

    top_axes, bottom_axes = [axis00, axis01, axis02], [axis10, axis11, axis12]
    for radial_limit_max, top_axis, bottom_axis, text in zip(radial_limits_max, top_axes, bottom_axes, texts):
        # Get the names and sort them #
        path = '/u/di43/Auriga/plots/data/' + 'dsh/' + str(radial_limit_max) + '/'
        names = glob.glob(path + '/name_06*')
        names.sort()

        # Loop over all available haloes #
        for i in range(len(names)):
            age = np.load(path + 'age_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            weights = np.load(path + 'weights_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

            counts, bins, bars = top_axis.hist(age, weights=weights, histtype='step', bins=n_bins, range=[tmin, tmax],
                                               color=colors[i], label="Au-" + (str(re.split('_|.npy', names[i])[1])))

            axis2 = top_axis.twiny()
            set_axes_evo(top_axis, axis2)
            top_axis.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
            top_axis.text(0.05, 0.92, text, color='k', fontsize=12, transform=top_axis.transAxes)

            if i == 0:
                original_bins, original_counts = bins, counts
            else:
                bottom_axis.plot(original_bins[:-1], (np.divide(counts - original_counts, original_counts)),
                                 color=colors[i], )
                axis2 = bottom_axis.twiny()
                set_axes_evo(bottom_axis, axis2)

    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


def central_bfld(pdf, data, levels):
    nlevels = len(levels)

    figure = plt.figure(figsize=(8.2, 8.2))
    plt.grid(True, color='gray', linestyle='-')
    axis = figure.iaxes(1.0, 1.0, 6.8, 6.8, top=True)
    axis.set_xlabel("$\\rm{M_{stars}\\,[M_\\odot]}$")
    axis.set_ylabel("$B_\mathrm{r<1\,kpc}\,\mathrm{[\mu G]}$")

    for il in range(nlevels):
        default_level = levels[il]

        data.select_haloes(default_level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))

        mstar = np.zeros(nhalos)
        bfld = np.zeros(nhalos)

        data.select_haloes(default_level, 0., loadonlyhalo=0, loadonlytype=[0, 4])

        ihalo = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])

            i, = np.where((s.r() < 0.001) & (s.data['type'] == 0))
            bfld[ihalo] = np.sqrt(
                ((s.data['bfld'][i, :] ** 2.).sum(axis=1) * s.data['vol'][i]).sum() / s.data['vol'][i].sum()) * bfac

            age = np.zeros(s.npartall)
            age[s.data['type'] == 4] = s.data['age']
            stellar_mask, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.data['type'] == 4) & (age > 0.))
            mstar[ihalo] = s.data['mass'][stellar_mask].sum()

            axis.loglog(mstar[ihalo] * 1e10, bfld[ihalo] * 1e6, color=next(colors), linestyle="None", marker='*',
                        ms=15.0, label="Au%s-%d" % (s.haloname, levels[0]))
            axis.legend(loc='lower right', fontsize=16, frameon=False, numpoints=1)

            ihalo += 1

    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
    # plt.savefig('/u/di43/Auriga/plots/' + 'Auriga-' + pdf + '.png', bbox_inches='tight')
    plt.close()
    return None


def gas_temperature_histogram(pdf, data, read):
    """
    Mass- and volume-weighted histograms of gas temperature
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Read the data #
    if read is True:
        volumes, masses, temperatures = [], [], []  # Declare empty arrays to store the data.

        # Check if a folder to save the data exists, if not then create one #
        path = '/u/di43/Auriga/plots/data/' + 'gth/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Get all available redshifts #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()

        for i in range(0, 1):
            redshift = np.flip(redshifts)[i]

            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [0, 4]
            attributes = ['age', 'mass', 'ne', 'pos', 'rho', 'u', 'vol']
            data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

            # Loop over all available haloes #
            for s in data:
                # Check if halo's data already exists, if not then read it #
                names = glob.glob(path + '/name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                # if str(s.haloname) in names:
                #     continue

                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the
                # z-axis #
                s.calc_sf_indizes(s.subfind)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

                # Mask the data: select gas cells within the virial radius R200 #
                mask, = np.where((s.data['type'] == 0) & (s.r() < s.subfind.data['frc2'][0]))

                # Calculate the temperature of the gas cells #
                ne = s.data['ne'][mask]
                metallicity = s.data['gz'][mask]
                XH = s.data['gmet'][mask, element['H']]
                yhelium = (1 - XH - metallicity) / (4 * XH)
                mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
                temperature = GAMMA_MINUS1 * s.data['u'][mask] * 1e10 * mu * PROTONMASS / BOLTZMANN

                # Append the properties for all redshifts #
                temperatures.append(temperature)
                volumes.append(s.data['vol'][mask])
                masses.append(s.data['mass'][mask])

        # Save data for each halo in numpy arrays #
        np.save(path + 'masses_' + str(s.haloname), masses)
        np.save(path + 'volumes_' + str(s.haloname), volumes)
        np.save(path + 'name_' + str(s.haloname), s.haloname)
        np.save(path + 'temperatures_' + str(s.haloname), temperatures)

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 7.5))
    plt.grid(True, color='gray', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Temperature [K]')
    plt.tick_params(direction='out', which='both', top='on', right='on')

    # Load and plot the data #
    names = glob.glob(path + '/name_06*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        masses = np.load(path + 'masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
        volumes = np.load(path + 'volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
        temperatures = np.load(path + 'temperatures_' + str(re.split('_|.npy', names[i])[1]) + '.npy',
                               allow_pickle=True)

        # for i in range(len(masses)):
        #     print(len(masses[i]))
        # average_masses = np.sum(masses, axis=0) / 10
        # average_volumes = np.sum(volumes, axis=0) / 10
        # average_temperatures = np.sum(temperatures,axis=0) / 10

        # y_data, edges = np.histogram(average_temperatures, weights=average_masses / np.sum(average_masses), bins=100)
        # plt.plot(0.5 * (edges[1:] + edges[:-1]), y_data, label='Mass-weighted')

        # Convert bin centres to log space and plot #
        l = np.sort(temperatures)
        print(l[0, :100])
        y_data, edges = np.histogram(temperatures, weights=masses / np.sum(masses), bins=100)
        print(np.sort(edges))
        bin_centres = 10 ** (0.5 * (np.log10(edges[1:]) + np.log10(edges[:-1])))
        plt.plot(bin_centres, y_data, label='Mass-weighted', color=colors[3])

        y_data, edges = np.histogram(temperatures, weights=volumes / np.sum(volumes), bins=100)
        bin_centres = 10 ** (0.5 * (np.log10(edges[1:]) + np.log10(edges[:-1])))
        plt.plot(bin_centres, y_data, label='Volume-weighted', color=colors[2])

    axis.legend(loc='upper left', fontsize=16, frameon=False, numpoints=1)
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


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
                names = glob.glob(path + '/name_*')
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
    figure = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.05)
    axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
    axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
    axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])

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
    for axis in [axis20, axis21, axis22]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, aspect=None)
    axis00.set_ylabel(r'$\mathrm{Sfr/(M_\odot\;yr^{-1})}$', size=20)
    axis10.set_ylabel(r'$\mathrm{(\delta Sfr)_{norm}}$', size=20)
    axis20.set_ylabel(r'$\mathrm{Mass\;loading}$', size=20)

    # Loop over all radial limits #
    top_axes, middle_axes, bottom_axes = [axis00, axis01, axis02], [axis10, axis11, axis12], [axis20, axis21, axis22]
    for radial_cut_min, radial_cut_max, top_axis, middle_axis, bottom_axis in zip(radial_cuts_min, radial_cuts_max,
                                                                                  top_axes, middle_axes, bottom_axes):
        # Get the names and sort them #
        path = '/u/di43/Auriga/plots/data/' + 'dsr/' + str(radial_cut_max) + '/'
        path_gfml = '/u/di43/Auriga/plots/data/' + 'gfml/'
        names = glob.glob(path + '/name_06*')
        names.sort()

        # Loop over all available haloes #
        for i in range(len(names)):
            # Load the data #
            weights = np.load(path + 'weights_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

            sfrs = np.load(path_gfml + 'sfrs_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
            gas_masses = np.load(path_gfml + 'gas_masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy',
                                 allow_pickle=True)
            lookback_times_gfml = np.load(path_gfml + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy',
                                          allow_pickle=True)
            spherical_radii = np.load(path_gfml + 'spherical_radii_' + str(re.split('_|.npy', names[i])[1]) + '.npy',
                                      allow_pickle=True)
            radial_velocities = np.load(
                path_gfml + 'radial_velocities_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)

            # Declare arrays to store the data #
            mass_outflows, mass_inflows, mass_loading, wind_loading = np.zeros(len(lookback_times_gfml)), np.zeros(
                len(lookback_times_gfml)), np.zeros(len(lookback_times_gfml)), np.zeros(len(lookback_times_gfml))

            # Plot the evolution of SFR and the normalised delta SFR #
            counts, bins, bars = top_axis.hist(lookback_times, weights=weights, histtype='step', bins=n_bins,
                                               range=[0, 13], color=colors[i],
                                               label="Au-" + (str(re.split('_|.npy', names[i])[1])))
            if i == 0:
                original_bins, original_counts = bins, counts
            else:
                middle_axis.plot(original_bins[:-1], np.divide(counts - original_counts, original_counts),
                                 color=colors[i], label="Au-" + (str(re.split('_|.npy', names[i])[1])))

            for l in range(len(lookback_times_gfml)):
                outflow_mask, = np.where(
                    (spherical_radii[l] > radial_cut_min) & (spherical_radii[l] < radial_cut_max) & (
                        radial_velocities[l] > 0))
                inflow_mask, = np.where(
                    (spherical_radii[l] > radial_cut_min) & (spherical_radii[l] < radial_cut_max) & (
                        radial_velocities[l] < 0))
                mass_outflows[l] = np.divide(np.sum(gas_masses[l][outflow_mask] * np.abs(
                    radial_velocities[l][outflow_mask] * u.km.to(u.Mpc) / u.second.to(u.yr))), 1e-3) * 1e10
                mass_inflows[l] = np.divide(np.sum(gas_masses[l][inflow_mask] * np.abs(
                    radial_velocities[l][inflow_mask] * u.km.to(u.Mpc) / u.second.to(u.yr))), 1e-3) * 1e10
                mass_loading[l] = mass_outflows[l] / np.sum(sfrs[l])

            # Downsample the runs which have more snapshots #
            if i == 0:
                original_net_flow = mass_inflows - mass_outflows
            else:
                # Plot the evolution of mass loading #
                net_flow = mass_inflows - mass_outflows
                net_flow = plot_tools.linear_resample(net_flow, len(original_net_flow))
                lookback_times_gfml = plot_tools.linear_resample(lookback_times_gfml, len(original_net_flow))
                bottom_axis.plot(lookback_times_gfml, np.divide(net_flow - original_net_flow, original_net_flow),
                                 color=colors[i])

            # Create the legend #
            top_axis.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
            middle_axis.legend(loc='upper center', fontsize=16, frameon=False, numpoints=1, ncol=2)

        # Add the text #
        figure.text(0.01, 0.92, r'$\mathrm{%.0f<r/kpc\leq%.0f}$' % (
        (np.float(radial_cut_min) * 1e3), (np.float(radial_cut_max) * 1e3)), fontsize=20, transform=top_axis.transAxes)
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def AGN_modes_distribution_combination(date):
    """
    Plot a combination of the energy of different black hole feedback modes from log files and plot its evolution for
    Auriga halo(es).
    :param date: date.
    :return: None
    """
    print("Invoking AGN_modes_distribution_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    names = glob.glob(path + 'name_*')
    names.sort()
    n_bins = 130
    time_bin_width = (13 - 0) / n_bins  # In Gyr.

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, \
    axis42, axis50, axis51, axis52 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple7=True)

    for axis in [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[1e51, 2e61], yscale='log', aspect=None, which='major', size=20)
    for axis in [axis10, axis11, axis12, axis30, axis31, axis32]:
        axis.set_xlabel('')
        axis.set_xticklabels([])
    axis10.set_ylabel(r'$\mathrm{(Mechanical\;feedback\;energy)/ergs}$', size=20)
    axis30.set_ylabel(r'$\mathrm{(Thermal\;feedback\;energy)/ergs}$', size=20)
    axis50.set_ylabel(r'$\mathrm{(Thermal\;feedback\;energy)/ergs}$', size=20)

    # Get the names and sort them #
    names = glob.glob(path + 'name_*')
    names.sort()

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e. original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)
    for i in range(len(names_groups)):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        for j in range(len(names_flavours)):
            # Load the data #
            if j == 0:
                mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
                # Transform the arrays to comma separated strings and convert each element to float #
                mechanicals = ','.join(mechanicals)
                mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
            thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            # Transform the arrays to comma separated strings and convert each element to float #
            thermals = ','.join(thermals)
            thermals = np.fromstring(thermals, dtype=np.float, sep=',')

            # Define the plot and colorbar axes #
            if j == 0:
                if i == 0:
                    axes = [axis10, axis30]
                    axescbar = [axis00, axis20]
                    modes = [mechanicals, thermals]
                if i == 1:
                    axes = [axis11, axis31]
                    axescbar = [axis01, axis21]
                    modes = [mechanicals, thermals]
                if i == 2:
                    axes = [axis12, axis32]
                    axescbar = [axis02, axis22]
                    modes = [mechanicals, thermals]
            else:
                if i == 0:
                    axes = [axis50]
                    modes = [thermals]
                    axescbar = [axis40]
                if i == 1:
                    axes = [axis51]
                    modes = [thermals]
                    axescbar = [axis41]
                if i == 2:
                    axes = [axis52]
                    modes = [thermals]
                    axescbar = [axis42]

            # Plot 2D distribution of the modes and their binned sum line #
            for axis, axiscbar, mode in zip(axes, axescbar, modes):
                hb = axis.hexbin(lookback_times[np.where(mode > 0)], mode[np.where(mode > 0)], bins='log', yscale='log',
                                 cmap='hot_r')
                plot_tools.create_colorbar(axiscbar, hb, label=r'$\mathrm{Counts\;per\;hexbin}$',
                                           orientation='horizontal', size=20)
                x_value, sum = plot_tools.binned_sum(lookback_times[np.where(mode > 0)], mode[np.where(mode > 0)],
                                                     n_bins=n_bins)
                axis.plot(x_value, sum / time_bin_width, color=colors[0], label=r'$\mathrm{Sum}$')
                figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                            transform=axis.transAxes)

            for axis in [axis11, axis12, axis31, axis32, axis51, axis52]:
                axis.set_yticklabels([])

    # Create the legends, save and close the figure #
    axis10.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
    axis11.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
    axis12.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
    plt.savefig('/u/di43/Auriga/plots/' + 'AGNmdc-' + date + '.png', bbox_inches='tight')

    return None


def gas_density_combination(pdf, redshift):
    """
    Plot a combination of the gas density projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking gas_density_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gd/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, \
    axis42, axis50, axis51, axis52, axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40,
                 axis41, axis42, axis50, axis51, axis52]:
        plot_tools.set_axis(axis, xlim=[-30, 30], ylim=[-30, 30], aspect=None)
    for axis in [axis10, axis11, axis12, axis30, axis31, axis32, axis50, axis51, axis52]:
        axis.set_ylim([-15, 15])
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22, axis31, axis32, axis41, axis42]:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    for axis in [axis00, axis10, axis20, axis30, axis40]:
        axis.set_xticklabels([])
    for axis in [axis51, axis52]:
        axis.set_yticklabels([])
    for axis in [axis50, axis51, axis52]:
        labels = ['', '-20', '', '0', '', '20', '']
        axis.set_xticklabels(labels)
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=20)
    for axis in [axis00, axis20, axis40]:
        labels = ['', '-20', '', '0', '', '20', '']
        axis.set_yticklabels(labels)
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=20)
    for axis in [axis10, axis30, axis50]:
        labels = ['', '-10', '', '0', '', '10', '']
        axis.set_yticklabels(labels)
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=20)

    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the gas density projections #
        pcm = axis_face_on.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10),
                                      rasterized=True, cmap='inferno')
        axis_edge_on.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10),
                                rasterized=True, cmap='inferno')
        plot_tools.create_colorbar(axiscbar, pcm, label='$\mathrm{\Sigma_{gas}/(M_\odot\;kpc^{-2})}$')

        figure.text(0.01, 0.9, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis_face_on.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


flavours_text = [(r'$\mathrm{Au-06:radio}$', r'$\mathrm{Au-06:quasar,\;quasar_{eff}}$',
                  r'$\mathrm{Au-06NoR:quasar,\;quasar_{eff}}$'), (
                     r'$\mathrm{Au-17:radio}$', r'$\mathrm{Au-17:quasar,\;quasar_{eff}}$',
                     r'$\mathrm{Au-17NoR:quasar,\;quasar_{eff}}$'), (
                     r'$\mathrm{Au-18:radio}$', r'$\mathrm{Au-18:quasar,\;quasar_{eff}}$',
                     r'$\mathrm{Au-18NoR:quasar,\;quasar_{eff}}$')]

custom_lines = [Line2D([0], [0], color=colors[0], ls='--'),
                (Line2D([0], [0], color=colors[0], ls=':'), Line2D([0], [0], color=colors[0], ls='-')),
                (Line2D([0], [0], color=colors[1], ls=':'), Line2D([0], [0], color=colors[1], ls='-'))]


def AGN_modes_distribution_combination(pdf):
    """
    Plot a combination of the energy of different black hole feedback modes from log files and plot its evolution for
    Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking AGN_modes_distribution_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    path_kernel = '/u/di43/Auriga/plots/data/' + 'AGNfk/'
    names = glob.glob(path + 'name_*')
    names.sort()
    n_bins = 89  # len(gas_volumes)
    time_bin_width = ((13 - 0) / n_bins) * u.Gyr.to(u.second)  # In second

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 5))
    axis00, axis01, axis02 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple4=True)
    axis002 = axis00.twiny()
    plot_tools.set_axes_evolution(axis00, axis002, ylim=[1e41, 1e47], yscale='log',
                                  ylabel=r'$\mathrm{Energy\;rate/(ergs\;s^{-1}})$', which='major', aspect=None)
    for axis in [axis01, axis02]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[1e41, 1e47], yscale='log', which='major', aspect=None)
        axis.set_yticklabels([])

    # Get the names and sort them #
    names = glob.glob(path + 'name_*')
    names.sort()

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e. original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)
    for i, axis in zip(range(len(names_groups)), [axis00, axis01, axis02]):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        for j in range(len(names_flavours)):
            print("Plotting data for halo:", str(re.split('_|.npy', names_flavours[j])[1]))
            # Load the data #
            if j == 0:
                mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
                # Transform the arrays to comma separated strings and convert each element to float #
                mechanicals = ','.join(mechanicals)
                mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
            thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            # Transform the arrays to comma separated strings and convert each element to float #
            thermals = ','.join(thermals)
            thermals = np.fromstring(thermals, dtype=np.float, sep=',')

            gas_volumes = np.load(path_kernel + 'gas_volumes_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            nsf_gas_volumes = np.load(
                path_kernel + 'nsf_gas_volumes_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')

            # Downsample the runs which have more snapshots #
            gas_volumes = plot_tools.linear_resample(gas_volumes, n_bins)
            nsf_gas_volumes = plot_tools.linear_resample(nsf_gas_volumes, n_bins)

            # Define the plotting style #
            if j == 0:  # Original halo.
                mode_style = [':', '--']
                modes = [thermals, mechanicals]
            else:  # NoR flavour.
                mode_style = [':']
                modes = [thermals]

            # Plot 2D distribution of the modes and their binned sum line #
            for k, mode in enumerate(modes):
                x_values, sum = plot_tools.binned_sum(lookback_times[np.where(mode > 0)], mode[np.where(mode > 0)],
                                                      n_bins=n_bins)
                # Plot the effective thermal energies #
                if k == 0:
                    efficiency = np.flip(nsf_gas_volumes / gas_volumes)  # binned_sum flips the x_values
                    mask, = np.where(efficiency > 0)
                    axis.plot(x_values[mask], np.divide(efficiency[mask] * sum[mask], time_bin_width), color=colors[j],
                              linestyle='-')

                axis.plot(x_values[mask], sum[mask] / time_bin_width, color=colors[j], linestyle=mode_style[k])

        # Create the legend #
        axis.legend(custom_lines, flavours_text[i], handler_map={tuple:HandlerTuple(ndivide=None)}, numpoints=1,
                    frameon=False, fontsize=15, loc='upper left')

        # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None

def gas_temperature_regimes_combination(pdf):
    """
    Plot a combination of the evolution of gas fractions in different temperature regimes for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking gas_temperature_regimes_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gtr/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple5=True)
    for axis in [axis00, axis01, axis02]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[0, 1.1], aspect=None, size=25)
        axis.set_xlabel('')
        axis.set_xticklabels([])
    for axis in [axis10, axis11, axis12, axis20, axis21, axis22]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[0, 1.1], aspect=None, size=25)
        axis.set_xlabel('')
        axis.set_xticklabels([])
        axis2.set_xlabel('')
        axis2.set_xticklabels([])
        axis2.tick_params(top=False)
    for axis in [axis20, axis21, axis22]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[0, 1.1], aspect=None, size=25)
        axis2.set_xlabel('')
        axis2.set_xticklabels([])
        axis2.tick_params(top=False)
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22, axis21, axis22]:
        axis.set_ylabel('')
        axis.set_yticklabels([])
    axis00.set_ylabel(r'$\mathrm{M_{cold\;gas} / M_{gas}}$', size=25)
    axis10.set_ylabel(r'$\mathrm{M_{warm\;gas} / M_{gas}}$', size=25)
    axis20.set_ylabel(r'$\mathrm{M_{hot\;gas} / M_{gas}}$', size=25)

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e. original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)
    axes = [[axis00, axis10, axis20], [axis01, axis11, axis21], [axis02, axis12, axis22]]
    for i, axis in zip(range(len(names_groups)), axes):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        for j in range(len(names_flavours)):
            print("Plotting data for halo:", str(re.split('_|.npy', names_flavours[j])[1]))
            # Load the data #
            sfg_ratios = np.load(path + 'sfg_ratios_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            wg_ratios = np.load(path + 'wg_ratios_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            hg_ratios = np.load(path + 'hg_ratios_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')

            # Downsample the runs which have more snapshots #
            if j == 0:
                original_lookback_times = lookback_times
            else:
                sfg_ratios = plot_tools.linear_resample(sfg_ratios, len(original_lookback_times))
                wg_ratios = plot_tools.linear_resample(wg_ratios, len(original_lookback_times))
                hg_ratios = plot_tools.linear_resample(hg_ratios, len(original_lookback_times))
                lookback_times = plot_tools.linear_resample(lookback_times, len(original_lookback_times))

            # Plot the evolution of gas fraction in different temperature regimes #
            axis[0].plot(lookback_times, sfg_ratios, color=colors2[j], lw=3,
                         label=r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names_flavours[j])[1]))
            axis[1].plot(lookback_times, wg_ratios, color=colors2[j], lw=3)
            axis[2].plot(lookback_times, hg_ratios, color=colors2[j], lw=3)

            axis[0].legend(loc='upper right', fontsize=25, frameon=False)  # Create the legend.
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None