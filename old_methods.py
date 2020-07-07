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
    Get information about different black hole modes from log files and plot the evolution of the cumulative feedback for Auriga halo(es).
    :param date: .
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    path = '/u/di43/Auriga/plots/data/' + 'AGNmc/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        data.select_haloes(default_level, default_redshift, loadonlytype=None, loadonlyhalo=0, loadonly=None)
        
        # Loop over all available haloes #
        for s in data:
            # Declare arrays to store the desired words and lines that contain these words #
            redshift_lines, redshifts, feedback_lines, thermals, mechanicals = [], [], [], [], []
            
            # Check if any of the haloes' data already exists, if not then read and save it #
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
    
    # Get the names and sort them #
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
        sc = axis00.scatter(thermals[feedback_mask], mechanicals[feedback_mask], edgecolor='None', s=50, c=lookback_times[feedback_mask], vmin=0,
                            vmax=max(lookback_times), cmap='jet')
        cb = plt.colorbar(sc, cax=axiscbar)
        cb.set_label(r'$\mathrm{t_{look}\;[Gyr]}$', size=16)
        axiscbar.tick_params(direction='out', which='both', right='on', left='on')
        axiscbar.yaxis.tick_left()
        
        # Create the legends, save and close the figure #
        axis00.legend([plot000, plot001], [r'$\mathrm{1:10}$', r'$\mathrm{1:50}$'], loc='upper center', fontsize=16, frameon=False, numpoints=1)
        axis00.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmc-' + date + '.png', bbox_inches='tight')
        plt.close()
    return None


def AGN_modes_histogram(date, data, read):
    """
    Get information about different black hole modes from log files and plot a histogram of the evolution of the step feedback for Auriga halo(es).
    :param date: .
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        path = '/u/di43/Auriga/plots/data/' + 'AGNmh/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        data.select_haloes(default_level, default_redshift, loadonlytype=None, loadonlyhalo=0, loadonly=None)
        
        # Loop over all available haloes #
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
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_06NOR*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        
        # Generate the figure and define its parameters #
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
        axis.legend([hist0, hist1], [r'$\mathrm{Thermal}$', r'$\mathrm{Mechanical}$'], loc='upper right', fontsize=16, frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmh-' + date + '.png', bbox_inches='tight')
        plt.close()
    
    return None


def AGN_modes_step(date, data, read):
    """
    Get information about different black hole modes from log files and plot the step feedback for Auriga halo(es).
    :param date: .
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        path = '/u/di43/Auriga/plots/data/' + 'AGNms/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        data.select_haloes(default_level, default_redshift, loadonlytype=None, loadonlyhalo=0, loadonly=None)
        
        # Loop over all available haloes #
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
    
    # Get the names and sort them #
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
        axis00.hist(mechanicals, bins=np.linspace(0, 2e56, 100), histtype='step', weights=weights, orientation='vertical', color='k')
        weights = np.ones_like(thermals) / float(len(thermals))
        axis11.hist(thermals, bins=np.linspace(0, 5e57, 100), histtype='step', weights=weights, orientation='horizontal', color='k')
        
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
        
        # Check if a folder to save the data exists, if not create one #
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
            data.select_haloes(default_level, default_redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
            
            # Loop over all available haloes #
            for s in data:
                # Check if any of the haloes' data already exists, if not then read and save it #
                names = glob.glob(path + '/name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                if str(s.haloname) in names:
                    continue
                
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
                s.calc_sf_indizes(s.subfind)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
                
                # Calculate the temperature of the gas cells #
                gas_mask, = np.where(
                    (s.data['type'] == 0) & (s.r() < s.subfind.data['frc2'][0]))  # Mask the data: select gas cells inside the virial radius R200.
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
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 10))
        plt.grid(True, color='gray', linestyle='-')
        plt.ylim(-0.2, 1.2)
        plt.ylabel(r'$\mathrm{Gas\;fraction}$', size=16)
        plt.xlabel(r'$\mathrm{Redshift}$', size=16)
        axis.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=axis.transAxes)
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
        
        # for j in range(len(redshifts) - 1):  # b1, = plt.bar(redshifts[j], sfg_ratios[j], width=redshifts[j + 1] - redshifts[j], alpha=0.6,
        # color='blue', align='edge', edgecolor='none')  # b2, = plt.bar(redshifts[j], wg_ratios[j], bottom=sfg_ratios[j], width=redshifts[j + 1] -
        # redshifts[j], alpha=0.6, color='green',  #               align='edge', edgecolor='none')  # b3, = plt.bar(redshifts[j], hg_ratios[j],
        # bottom=np.sum(np.vstack([sfg_ratios[j], wg_ratios[j]]).T),  #               width=redshifts[j + 1] - redshifts[j], alpha=0.6,
        # align='edge', color='red', edgecolor='none')
        
        b1, = plt.plot(lookback_times, sfg_ratios, color='blue')
        b2, = plt.plot(lookback_times, wg_ratios, color='green')
        b3, = plt.plot(lookback_times, hg_ratios, color='red')
    
    # Create the legends, save and close the figure #
    plt.legend([b3, b2, b1], [r'$\mathrm{Hot\;gas}$', r'$\mathrm{Warm\;gas}$', r'$\mathrm{Cold\;gas}$'], loc='upper left', fontsize=16, frameon=False,
               numpoints=1)
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None