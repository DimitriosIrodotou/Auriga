import os
import re
import glob
import matplotlib
import plot_tools
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from loadmodules import *
from matplotlib import gridspec

res = 512
level = 4
boxsize = 0.06
element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}


def stellar_light_combination(pdf, redshift):
    """
    Plot a combination of stellar light projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/''sl/' + str(redshift)
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, axis52,\
    x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple2=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50,
                 axis51, axis52]:
        axis.set_yticklabels([])
        axis.set_xticklabels([])
    plt.rcParams['savefig.facecolor'] = 'black'
    
    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        
        # Load the data #
        face_on = np.load(path + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the stellar light projections #
        axis_face_on.imshow(face_on, interpolation='nearest', aspect='equal')
        axis_edge_on.imshow(edge_on, interpolation='nearest', aspect='equal')
        
        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), color='w', fontsize=16, transform=axis_face_on.transAxes)
    
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def stellar_density_combination(pdf, redshift):
    """
    Plot a combination of stellar density projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'sd/' + str(redshift)
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, axis52,\
    axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    cmap = matplotlib.cm.get_cmap('twilight')
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, \
                 axis51, axis52]:
        axis.set_facecolor(cmap(0))
        plot_tools.set_axis(axis, xlim=[-30, 30], ylim=[-30, 30], size=12)
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
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=12)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=12)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=12)
    
    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        
        # Load the data #
        face_on = np.load(path + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the stellar density projections #
        pcm = axis_face_on.pcolormesh(x, y, face_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), rasterized=True, cmap=cmap)
        axis_edge_on.pcolormesh(x, y2, edge_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), rasterized=True, cmap=cmap)
        plot_tools.create_colorbar(axiscbar, pcm, label=r'$\mathrm{\Sigma_{\bigstar}/(M_\odot\;kpc^{-2})}$')
        
        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis_face_on.transAxes)
    
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_density_combination(pdf, redshift):
    """
    Plot a combination of gas density projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gd/' + str(redshift)
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, axis52,\
    axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, \
                 axis51, axis52]:
        plot_tools.set_axis(axis, xlim=[-30, 30], ylim=[-30, 30], size=12)
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
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=12)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=12)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=12)
    
    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        
        # Load the data #
        face_on = np.load(path + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the gas density projections #
        pcm = axis_face_on.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), rasterized=True, cmap='magma')
        axis_edge_on.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), rasterized=True, cmap='magma')
        plot_tools.create_colorbar(axiscbar, pcm, label='$\mathrm{\Sigma_{gas}/(M_\odot\;kpc^{-2})}$')
        
        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis_face_on.transAxes)
    
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_temperature_combination(pdf, redshift):
    """
    Plot a combination of gas temperature projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gt/' + str(redshift)
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, axis52,\
    axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, \
                 axis51, axis52]:
        plot_tools.set_axis(axis, xlim=[-30, 30], ylim=[-30, 30], size=12)
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
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=12)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=12)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=12)
    
    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        
        # Load the data #
        face_on = np.load(path + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        face_on_rho = np.load(path + '/face_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on_rho = np.load(path + '/edge_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the density-weighted gas temperature projections #
        pcm = axis_face_on.pcolormesh(x, y, (face_on / face_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), rasterized=True,
                                      cmap='viridis')
        axis_edge_on.pcolormesh(x, 0.5 * y, (edge_on / edge_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), rasterized=True,
                                cmap='viridis')
        plot_tools.create_colorbar(axiscbar, pcm, label=r'$\mathrm{T/K}$')
        
        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis_face_on.transAxes)
    
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_metallicity_combination(pdf, redshift):
    """
    Plot a combination of gas metallicity projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gm/' + str(redshift)
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, axis52,\
    axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, \
                 axis51, axis52]:
        plot_tools.set_axis(axis, xlim=[-30, 30], ylim=[-30, 30], size=12)
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
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=12)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=12)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=12)
    
    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        
        # Load the data #
        # Load the data #
        face_on = np.load(path + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the gas metallicity projections #
        pcm = axis_face_on.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), rasterized=True, cmap='viridis')
        axis_edge_on.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), rasterized=True, cmap='viridis')
        plot_tools.create_colorbar(axiscbar, pcm, label=r'$\mathrm{Z/Z_\odot}$')
        
        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis_face_on.transAxes)
    
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def AGN_modes_distribution(date, data):
    """
    Get information about different black hole modes from log files and plot the evolution of the step feedback.
    :param date: .
    :param data: data from main.make_pdf
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Generate the figure and set its parameters #
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
    
    # Load and plot the data #
    names = glob.glob(path + '/name_18.*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        
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
        
        hb = axis02.hexbin(lookback_times[np.where(thermals > 0)], thermals[np.where(thermals > 0)], yscale='log', cmap='gist_heat_r',
                           gridsize=(100, 50 * np.int(len(np.where(thermals > 0)[0]) / len(np.where(mechanicals > 0)[0]))))
        
        cb2 = plt.colorbar(hb, cax=axiscbar2, orientation='horizontal')
        cb2.set_label(r'$\mathrm{Counts\;per\;hexbin}$', size=16)
        
        for axis in [axiscbar, axiscbar2]:
            axis.xaxis.tick_top()
            axis.xaxis.set_label_position("top")
            axis.tick_params(direction='out', which='both', top='on', right='on')
        
        # Calculate and plot the mechanical energy sum #
        n_bins = int((max(lookback_times[np.where(mechanicals > 0)]) - min(lookback_times[np.where(mechanicals > 0)])) / 0.02)
        x_value = np.zeros(n_bins)
        sum = np.zeros(n_bins)
        x_low = min(lookback_times[np.where(mechanicals > 0)])
        for j in range(n_bins):
            index = np.where((lookback_times[np.where(mechanicals > 0)] >= x_low) & (lookback_times[np.where(mechanicals > 0)] < x_low + 0.02))[0]
            x_value[j] = np.mean(np.absolute(lookback_times[np.where(mechanicals > 0)])[index])
            if len(index) > 0:
                sum[j] = np.sum(mechanicals[np.where(mechanicals > 0)][index])
            x_low += 0.02
        
        sum00, = axis00.plot(x_value, sum, color='black', zorder=5)
        
        # Calculate and plot the mechanical energy sum #
        n_bins = int((max(lookback_times[np.where(thermals > 0)]) - min(lookback_times[np.where(thermals > 0)])) / 0.02)
        x_value = np.zeros(n_bins)
        sum = np.zeros(n_bins)
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
        axis00.legend([sum00], [r'$\mathrm{Sum}$'], loc='upper left', fontsize=16, frameon=False, numpoints=1)
        axis02.legend([sum02], [r'$\mathrm{Sum}$'], loc='upper left', fontsize=16, frameon=False, numpoints=1)
        figure.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=axis00.transAxes)
        
        # Save and close the figure #
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmd-' + date + '.png', bbox_inches='tight')
        plt.close()
    
    return None


def central_combination(pdf, data, redshift, read):
    """
    Plot a combination of projections for Auriga halo(es) for the central 2kpc.
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    boxsize = 0.004  # Decrease the boxsize.
    
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'm/' + str(redshift) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['mass', 'ne', 'pos', 'rho', 'u', 'bfld', 'sfr']
        data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
        
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
            
            # Get the gas density projections #
            density_face_on = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                                  "grid"] * boxsize * 1e3
            density_edge_on = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                                  "grid"] * boxsize * 1e3
            
            # Get the gas temperature projections #
            meanweight = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * s.data['ne']) * 1.67262178e-24
            temperature = (5.0 / 3.0 - 1.0) * s.data['u'] / KB * (1e6 * parsec) ** 2.0 / (1e6 * parsec / 1e5) ** 2 * meanweight
            s.data['temprho'] = s.rho * temperature
            
            temperature_face_on = s.get_Aslice("temprho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                "grid"]
            temperature_face_on_rho = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                "grid"]
            temperature_edge_on = s.get_Aslice("temprho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                "grid"]
            temperature_edge_on_rho = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                "grid"]
            
            # Get the magnetic field projections #
            s.data['b2'] = (s.data['bfld'] ** 2.).sum(axis=1)
            bfld_face_on = np.sqrt(
                s.get_Aslice("b2", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"] / res) * bfac * 1e6
            bfld_edge_on = np.sqrt(
                s.get_Aslice("b2", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"] / res) * bfac * 1e6
            
            # Get the gas sfr projections #
            sfr_face_on = s.get_Aslice("sfr", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]
            sfr_edge_on = s.get_Aslice("sfr", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]
            
            # Get the gas total pressure projections #
            elements_mass = [1.01, 4.00, 12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33]
            meanweight = np.sum(s.gmet[s.data['type'] == 0, 0:9], axis=1) / (
                np.sum(s.gmet[s.data['type'] == 0, 0:9] / elements_mass[0:9], axis=1) + s.data['ne'] * s.gmet[s.data['type'] == 0, 0])
            Tfac = 1. / meanweight * (1.0 / (5. / 3. - 1.)) * KB / PROTONMASS * 1e10 * msol / 1.989e53
            
            # Un megabars (10**12dyne/cm**2)
            s.data['T'] = s.u / Tfac
            s.data['dens'] = s.rho / (1e6 * parsec) ** 3. * msol * 1e10
            s.data['Ptherm'] = s.data['dens'] * s.data['T'] / (meanweight * PROTONMASS)
            
            pressure_face_on = s.get_Aslice("Ptherm", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]
            pressure_edge_on = s.get_Aslice("Ptherm", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]
            
            # Get the radial velocity projections #
            gas_mask, = np.where(s.data['type'] == 0)
            spherical_radius = np.sqrt(np.sum(s.data['pos'][gas_mask, :] ** 2, axis=1))
            CoM_velocity = np.sum(s.data['vel'][gas_mask, :] * s.data['mass'][gas_mask][:, None], axis=0) / np.sum(s.data['mass'][gas_mask])
            s.data['vrad'] = np.sum((s.data['vel'][gas_mask] - CoM_velocity) * s.data['pos'][gas_mask], axis=1) / spherical_radius
            
            vrad_face_on = s.get_Aslice("vrad", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]
            vrad_edge_on = s.get_Aslice("vrad", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'density_face_on_' + str(s.haloname), density_face_on)
            np.save(path + 'density_edge_on_' + str(s.haloname), density_edge_on)
            np.save(path + 'temperature_face_on_' + str(s.haloname), temperature_face_on)
            np.save(path + 'temperature_edge_on_' + str(s.haloname), temperature_edge_on)
            np.save(path + 'temperature_face_on_rho_' + str(s.haloname), temperature_face_on_rho)
            np.save(path + 'temperature_edge_on_rho_' + str(s.haloname), temperature_edge_on_rho)
            np.save(path + 'bfld_face_on_' + str(s.haloname), bfld_face_on)
            np.save(path + 'bfld_edge_on_' + str(s.haloname), bfld_edge_on)
            np.save(path + 'sfr_face_on_' + str(s.haloname), sfr_face_on)
            np.save(path + 'sfr_edge_on_' + str(s.haloname), sfr_edge_on)
            np.save(path + 'pressure_face_on_' + str(s.haloname), pressure_face_on)
            np.save(path + 'pressure_edge_on_' + str(s.haloname), pressure_edge_on)
            np.save(path + 'vrad_face_on_' + str(s.haloname), vrad_face_on)
            np.save(path + 'vrad_edge_on_' + str(s.haloname), vrad_edge_on)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_06.*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure = plt.figure(figsize=(16, 9))
        axis00, axiscbar, axis20, axis01, axiscbar1, axis21, axis02, axiscbar2, axis22, axis03, axiscbar3, axis23, axis04, axiscbar4, axis24, \
        axis05, axiscbar5, axis25, x, y, area = create_axes(
            res=res, boxsize=boxsize * 1e3, multiple=True)
        tick_labels = np.array(['', '-1.5', '', '', '0', '', '', '1.5', ''])
        for axis in [axis00, axis01, axis02, axis03, axis04, axis05]:
            axis.set_xlim(-2, 2)
            axis.set_ylim(-2, 2)
            axis.set_aspect('equal')
            axis.set_xticklabels([])
            axis.tick_params(direction='out', which='both', top='on', right='on')
        for axis in [axis20, axis21, axis22, axis23, axis24, axis25]:
            axis.set_xlim(-2, 2)
            axis.set_ylim(-2, 2)
            axis.set_aspect('equal')
            axis.set_xticklabels(tick_labels)
            axis.set_xlabel(r'$x\;\mathrm{[kpc]}$', size=12)
            axis.tick_params(direction='out', which='both', top='on', right='on')
            for label in axis.xaxis.get_ticklabels():
                label.set_size(12)
        for axis in [axis01, axis21, axis02, axis22, axis03, axis23, axis04, axis24, axis05, axis25]:
            axis.set_yticklabels([])
        
        axis00.set_ylabel(r'$y\;\mathrm{[kpc]}$', size=12)
        axis20.set_ylabel(r'$z\;\mathrm{[kpc]}$', size=12)
        for axis in [axis00, axis20]:
            for label in axis.yaxis.get_ticklabels():
                label.set_size(12)
                axis.set_yticklabels(tick_labels)
        
        # Load and plot the data #
        density_face_on = np.load(path + 'density_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        density_edge_on = np.load(path + 'density_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_face_on = np.load(path + 'temperature_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_edge_on = np.load(path + 'temperature_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_face_on_rho = np.load(path + 'temperature_face_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_edge_on_rho = np.load(path + 'temperature_edge_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        bfld_face_on = np.load(path + 'bfld_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        bfld_edge_on = np.load(path + 'bfld_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sfr_face_on = np.load(path + 'sfr_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sfr_edge_on = np.load(path + 'sfr_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        pressure_face_on = np.load(path + 'pressure_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        pressure_edge_on = np.load(path + 'pressure_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        vrad_face_on = np.load(path + 'vrad_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        vrad_edge_on = np.load(path + 'vrad_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the gas density projections #
        pcm = axis00.pcolormesh(x, y, density_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='magma', rasterized=True)
        axis20.pcolormesh(x, y, density_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='magma', rasterized=True)
        plot_tools.create_colorbar(axiscbar, pcm, "$\mathrm{\Sigma_{gas}\;[M_\odot\;kpc^{-2}]}$", orientation='horizontal')
        
        # Plot the gas temperature projections #
        pcm = axis01.pcolormesh(x, y, (temperature_face_on / temperature_face_on_rho).T, norm=matplotlib.colors.LogNorm(), cmap='viridis',
                                rasterized=True)
        axis21.pcolormesh(x, y, (temperature_edge_on / temperature_edge_on_rho).T, norm=matplotlib.colors.LogNorm(), cmap='viridis', rasterized=True)
        plot_tools.create_colorbar(axiscbar1, pcm, "$\mathrm{T\;[K]}$", orientation='horizontal')
        
        # Plot the magnetic field projections #
        pcm = axis02.pcolormesh(x, y, bfld_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='CMRmap', rasterized=True)
        axis22.pcolormesh(x, y, bfld_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='CMRmap', rasterized=True)
        plot_tools.create_colorbar(axiscbar2, pcm, "$\mathrm{B\;[\mu G]}$", orientation='horizontal')
        
        # Plot the sfr projections #
        pcm = axis03.pcolormesh(x, y, sfr_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='gist_heat', rasterized=True)
        axis23.pcolormesh(x, y, sfr_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='gist_heat', rasterized=True)
        plot_tools.create_colorbar(axiscbar3, pcm, "$\mathrm{SFR\;[M_\odot\;yr^{-1}]}$", orientation='horizontal')
        
        # Plot the gas total pressure projections #
        pcm = axis04.pcolormesh(x, y, pressure_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='cividis', rasterized=True)
        axis24.pcolormesh(x, y, pressure_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='cividis', rasterized=True)
        plot_tools.create_colorbar(axiscbar4, pcm, "$\mathrm{P\;[K\;cm^{-3}]}$", orientation='horizontal')
        
        pcm = axis05.pcolormesh(x, y, vrad_face_on.T, cmap='coolwarm', vmin=-7000, vmax=7000, rasterized=True)
        axis25.pcolormesh(x, y, vrad_edge_on.T, cmap='coolwarm', vmin=-7000, vmax=7000, rasterized=True)
        plot_tools.create_colorbar(axiscbar5, pcm, "$\mathrm{Velocity\;[km\;s^{-1}]}$", orientation='horizontal')
        
        for axis in [axiscbar, axiscbar1, axiscbar2, axiscbar3, axiscbar4, axiscbar5]:
            axis.xaxis.tick_top()
            for label in axis.xaxis.get_ticklabels():
                label.set_size(12)
            axis.tick_params(direction='out', which='both', top='on', right='on')
        
        figure.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[i])[1]) + ' redshift = ' + str(redshift), fontsize=12, transform=axis00.transAxes)
        
        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None
