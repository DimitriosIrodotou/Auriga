import os
import re
import glob
import matplotlib
import projections
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from loadmodules import *
from matplotlib import gridspec
from scripts.gigagalaxy.util import satellite_utilities

res = 512
level = 4
boxsize = 0.06
element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}


def create_axes(res=res, boxsize=boxsize, contour=False, colorbar=False, velocity_vectors=False, multiple=False, multiple2=False, multiple3=False):
    """
    Generate plot axes.
    :param res: resolution
    :param boxsize: boxsize
    :param contour: contour
    :param colorbar: colorbar
    :param velocity_vectors: velocity_vectors
    :param multiple: multiple
    :return: axes
    """
    
    # Set the axis values #
    x = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y2 = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res / 2 + 1)
    
    area = (boxsize / res) ** 2  # Calculate the area.
    
    # Generate the panels #
    if contour is True:
        gs = gridspec.GridSpec(2, 3, hspace=0.05, wspace=0.05, height_ratios=[1, 0.5], width_ratios=[1, 1, 0.05])
        axis00 = plt.subplot(gs[0, 0])
        axis01 = plt.subplot(gs[0, 1])
        axis10 = plt.subplot(gs[1, 0])
        axis11 = plt.subplot(gs[1, 1])
        axiscbar = plt.subplot(gs[:, 2])
        
        return axis00, axis01, axis10, axis11, axiscbar, x, y, y2, area
    
    elif colorbar is True:
        gs = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.05, height_ratios=[1, 0.5], width_ratios=[1, 0.05])
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])
        axiscbar = plt.subplot(gs[:, 1])
        
        return axis00, axis10, axiscbar, x, y, y2, area
    
    elif velocity_vectors is True:
        gs = gridspec.GridSpec(2, 1, hspace=0.05)
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])
        
        return axis00, axis10, x, y, y2, area
    
    elif multiple is True:
        gs = gridspec.GridSpec(3, 6, hspace=0.0, wspace=0.05, height_ratios=[1, 0.05, 1])
        axis00 = plt.subplot(gs[0, 0])
        axiscbar = plt.subplot(gs[1, 0])
        axis20 = plt.subplot(gs[2, 0])
        axis01 = plt.subplot(gs[0, 1])
        axiscbar1 = plt.subplot(gs[1, 1])
        axis21 = plt.subplot(gs[2, 1])
        axis02 = plt.subplot(gs[0, 2])
        axiscbar2 = plt.subplot(gs[1, 2])
        axis22 = plt.subplot(gs[2, 2])
        axis03 = plt.subplot(gs[0, 3])
        axiscbar3 = plt.subplot(gs[1, 3])
        axis23 = plt.subplot(gs[2, 3])
        axis04 = plt.subplot(gs[0, 4])
        axiscbar4 = plt.subplot(gs[1, 4])
        axis24 = plt.subplot(gs[2, 4])
        axis05 = plt.subplot(gs[0, 5])
        axiscbar5 = plt.subplot(gs[1, 5])
        axis25 = plt.subplot(gs[2, 5])
        
        return axis00, axiscbar, axis20, axis01, axiscbar1, axis21, axis02, axiscbar2, axis22, axis03, axiscbar3, axis23, axis04, axiscbar4, axis24, axis05, axiscbar5, axis25, x, y, area
    
    elif multiple2 is True:
        gs = gridspec.GridSpec(4, 3, hspace=0, wspace=0, height_ratios=[1, 0.5, 1, 0.5])
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])
        axis20 = plt.subplot(gs[2, 0])
        axis30 = plt.subplot(gs[3, 0])
        axis01 = plt.subplot(gs[0, 1])
        axis11 = plt.subplot(gs[1, 1])
        axis21 = plt.subplot(gs[2, 1])
        axis31 = plt.subplot(gs[3, 1])
        axis02 = plt.subplot(gs[0, 2])
        axis12 = plt.subplot(gs[1, 2])
        axis22 = plt.subplot(gs[2, 2])
        axis32 = plt.subplot(gs[3, 2])
        return axis00, axis10, axis20, axis30, axis01, axis11, axis21, axis31, axis02, axis12, axis22, axis32, x, y, y2, area
    
    elif multiple3 is True:
        gs = gridspec.GridSpec(4, 4, hspace=0.05, wspace=0, height_ratios=[1, 0.5, 1, 0.5], width_ratios=[1, 1, 1, 0.1])
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])
        axis20 = plt.subplot(gs[2, 0])
        axis30 = plt.subplot(gs[3, 0])
        axis01 = plt.subplot(gs[0, 1])
        axis11 = plt.subplot(gs[1, 1])
        axis21 = plt.subplot(gs[2, 1])
        axis31 = plt.subplot(gs[3, 1])
        axis02 = plt.subplot(gs[0, 2])
        axis12 = plt.subplot(gs[1, 2])
        axis22 = plt.subplot(gs[2, 2])
        axis32 = plt.subplot(gs[3, 2])
        axiscbar = plt.subplot(gs[:, 3])
        return axis00, axis10, axis20, axis30, axis01, axis11, axis21, axis31, axis02, axis12, axis22, axis32, axiscbar, x, y, y2, area
    
    else:
        gs = gridspec.GridSpec(2, 1, hspace=0.05, height_ratios=[1, 0.5])
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])
        
        return axis00, axis10, x, y, y2, area


def create_colorbar(axis, pcm, label, orientation='vertical'):
    """
    Generate a colorbar.
    :param axis: colorbar axis from create_axes
    :param pcm: pseudocolor plot
    :param label: colorbar label
    :param orientation: colorbar orientation
    :return: None
    """
    # Set the colorbar axes #
    cb = plt.colorbar(pcm, cax=axis, orientation=orientation)
    
    # Set the colorbar parameters #
    cb.set_label(label, size=12)
    # cb.tick_params(direction='out', which='both')
    
    return None


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
    
    # Load and plot the data #
    names = glob.glob(path + '/name_18.*')
    names.sort()
    # Load and plot the data #
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
        nbin = int((max(lookback_times[np.where(mechanicals > 0)]) - min(lookback_times[np.where(mechanicals > 0)])) / 0.02)
        x_value = np.empty(nbin)
        sum = np.empty(nbin)
        x_low = min(lookback_times[np.where(mechanicals > 0)])
        for j in range(nbin):
            index = np.where((lookback_times[np.where(mechanicals > 0)] >= x_low) & (lookback_times[np.where(mechanicals > 0)] < x_low + 0.05))[0]
            x_value[j] = np.mean(np.absolute(lookback_times[np.where(mechanicals > 0)])[index])
            if len(index) > 0:
                sum[j] = np.sum(mechanicals[np.where(mechanicals > 0)][index])
            x_low += 0.05
        
        sum00, = axis00.plot(x_value, sum, color='black', zorder=5)
        
        # Calculate and plot the mechanical energy sum #
        nbin = int((max(lookback_times[np.where(thermals > 0)]) - min(lookback_times[np.where(thermals > 0)])) / 0.02)
        x_value = np.empty(nbin)
        sum = np.empty(nbin)
        x_low = min(lookback_times[np.where(thermals > 0)])
        for j in range(nbin):
            index = np.where((lookback_times[np.where(thermals > 0)] >= x_low) & (lookback_times[np.where(thermals > 0)] < x_low + 0.05))[0]
            x_value[j] = np.mean(np.absolute(lookback_times[np.where(thermals > 0)])[index])
            if len(index) > 0:
                sum[j] = np.sum(thermals[np.where(thermals > 0)][index])
            x_low += 0.05
        
        # Plot sum #
        sum02, = axis02.plot(x_value, sum, color='black', zorder=5)
        
        # Create the legends and save the figure #
        axis00.legend([sum00], [r'$\mathrm{Sum}$'], loc='upper left', fontsize=16, frameon=False, numpoints=1)
        axis02.legend([sum02], [r'$\mathrm{Sum}$'], loc='upper left', fontsize=16, frameon=False, numpoints=1)
        plt.savefig('/u/di43/Auriga/plots/' + 'AGNmd-' + date + '.png', bbox_inches='tight')
        figure.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=axis00.transAxes)
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
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(16, 9))
        axis00, axiscbar, axis20, axis01, axiscbar1, axis21, axis02, axiscbar2, axis22, axis03, axiscbar3, axis23, axis04, axiscbar4, axis24, axis05, axiscbar5, axis25, x, y, \
        area = create_axes(
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
        create_colorbar(axiscbar, pcm, "$\mathrm{\Sigma_{gas}\;[M_\odot\;kpc^{-2}]}$", orientation='horizontal')
        
        # Plot the gas temperature projections #
        pcm = axis01.pcolormesh(x, y, (temperature_face_on / temperature_face_on_rho).T, norm=matplotlib.colors.LogNorm(), cmap='viridis',
                              rasterized=True)
        axis21.pcolormesh(x, y, (temperature_edge_on / temperature_edge_on_rho).T, norm=matplotlib.colors.LogNorm(), cmap='viridis', rasterized=True)
        create_colorbar(axiscbar1, pcm, "$\mathrm{T\;[K]}$", orientation='horizontal')
        
        # Plot the magnetic field projections #
        pcm = axis02.pcolormesh(x, y, bfld_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='CMRmap', rasterized=True)
        axis22.pcolormesh(x, y, bfld_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='CMRmap', rasterized=True)
        create_colorbar(axiscbar2, pcm, "$\mathrm{B\;[\mu G]}$", orientation='horizontal')
        
        # Plot the sfr projections #
        pcm = axis03.pcolormesh(x, y, sfr_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='gist_heat', rasterized=True)
        axis23.pcolormesh(x, y, sfr_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='gist_heat', rasterized=True)
        create_colorbar(axiscbar3, pcm, "$\mathrm{SFR\;[M_\odot\;yr^{-1}]}$", orientation='horizontal')
        
        # Plot the gas total pressure projections #
        pcm = axis04.pcolormesh(x, y, pressure_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='cividis', rasterized=True)
        axis24.pcolormesh(x, y, pressure_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='cividis', rasterized=True)
        create_colorbar(axiscbar4, pcm, "$\mathrm{P\;[K\;cm^{-3}]}$", orientation='horizontal')
        
        pcm = axis05.pcolormesh(x, y, vrad_face_on.T, cmap='coolwarm', vmin=-7000, vmax=7000, rasterized=True)
        axis25.pcolormesh(x, y, vrad_edge_on.T, cmap='coolwarm', vmin=-7000, vmax=7000, rasterized=True)
        create_colorbar(axiscbar5, pcm, "$\mathrm{Velocity\;[km\;s^{-1}]}$", orientation='horizontal')
        
        for axis in [axiscbar, axiscbar1, axiscbar2, axiscbar3, axiscbar4, axiscbar5]:
            axis.xaxis.tick_top()
            for label in axis.xaxis.get_ticklabels():
                label.set_size(12)
            axis.tick_params(direction='out', which='both', top='on', right='on')
        
        figure.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[i])[1]) + ' redshift = ' + str(redshift), fontsize=12, transform=axis00.transAxes)
        pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def stellar_light_combination(pdf, redshift):
    """
    Plot a combination of stellar light projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/'
    names = glob.glob(path + 'sl/' + str(redshift) + '/name_*')
    names.sort()
    
    # Generate the figure and define its parameters #
    figure = plt.figure(figsize=(10, 10))
    axis00, axis10, axis20, axis30, axis01, axis11, axis21, axis31, axis02, axis12, axis22, axis32, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3,
                                                                                                         multiple2=True)
    axes_face_on = [axis00, axis20, axis01, axis21, axis02, axis22]
    axes_edge_on = [axis10, axis30, axis11, axis31, axis12, axis32]
    # Loop over all available haloes #
    for i, a, a2 in zip(range(len(names)), axes_face_on, axes_edge_on):
        for axis in [a, a2]:
            axis.set_yticklabels([])
            axis.set_xticklabels([])
        
        # Load and plot the data #
        face_on = np.load(path + 'sl/' + str(redshift) + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'sl/' + str(redshift) + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        axis.imshow(face_on, interpolation='nearest', aspect='equal')
        a2.imshow(edge_on, interpolation='nearest', aspect='equal')
        
        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), color='w', fontsize=12, transform=axis.transAxes)
    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
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
    path = '/u/di43/Auriga/plots/data/'
    names = glob.glob(path + 'gd/' + str(redshift) + '/name_*')
    names.sort()
    
    # Generate the figure and define its parameters #
    figure = plt.figure(figsize=(10, 10))
    axis00, axis10, axis20, axis30, axis01, axis11, axis21, axis31, axis02, axis12, axis22, axis32, axiscbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3,
                                                                                                                 multiple3=True)
    axes_face_on = [axis00, axis20, axis01, axis21, axis02, axis22]
    axes_edge_on = [axis10, axis30, axis11, axis31, axis12, axis32]
    for axis in [axis00, axis10, axis20, axis30, axis01, axis11, axis21, axis31, axis02, axis12, axis22, axis32]:
        axis.tick_params(direction='out', which='both', top='on', right='on')
    for axis in [axis01, axis11, axis21, axis02, axis12, axis22]:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    for axis in [axis00, axis10, axis20]:
        axis.set_xticklabels([])
    for axis in [axis30, axis31, axis32]:
        axis.set_xlabel(r'$x\;\mathrm{[kpc]}$', size=16)
    for axis in [axis00, axis20]:
        axis.set_ylabel(r'$y\;\mathrm{[kpc]}$', size=16)
    for axis in [axis10, axis30]:
        axis.set_ylabel(r'$z\;\mathrm{[kpc]}$', size=16)
    
    # Loop over all available haloes #
    for i, axis, a2 in zip(range(len(names)), axes_face_on, axes_edge_on):
        
        # Load and plot the data #
        face_on = np.load(path + 'gd/' + str(redshift) + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'gd/' + str(redshift) + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the projections #
        pcm = axis.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma', rasterized=True)
        a2.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma', rasterized=True)
        create_colorbar(axiscbar, pcm, "$\mathrm{\Sigma_{gas}\;[M_\odot\;kpc^{-2}]}$")
        
        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), color='w', fontsize=12, transform=axis.transAxes)
    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None
