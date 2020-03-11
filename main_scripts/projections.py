from __future__ import print_function, division

import os
import re
import glob
import pysph
import calcGrid
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from loadmodules import *
from matplotlib import gridspec

res = 512
boxsize = 0.06
element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}


def create_axes(res=res, boxsize=boxsize, contour=False, colorbar=False, velocity_vectors=False, multiple=False):
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
        ax00 = plt.subplot(gs[0, 0])
        ax01 = plt.subplot(gs[0, 1])
        ax10 = plt.subplot(gs[1, 0])
        ax11 = plt.subplot(gs[1, 1])
        axcbar = plt.subplot(gs[:, 2])
        
        return ax00, ax01, ax10, ax11, axcbar, x, y, y2, area
    
    elif colorbar is True:
        gs = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.05, height_ratios=[1, 0.5], width_ratios=[1, 0.05])
        ax00 = plt.subplot(gs[0, 0])
        ax10 = plt.subplot(gs[1, 0])
        axcbar = plt.subplot(gs[:, 1])
        
        return ax00, ax10, axcbar, x, y, y2, area
    
    elif velocity_vectors is True:
        gs = gridspec.GridSpec(2, 1, hspace=0.05)
        ax00 = plt.subplot(gs[0, 0])
        ax10 = plt.subplot(gs[1, 0])
        
        return ax00, ax10, x, y, y2, area
    
    elif multiple is True:
        gs = gridspec.GridSpec(3, 5, hspace=0.05, wspace=0.05, height_ratios=[1, 0.05, 1])
        axcbar = plt.subplot(gs[1, 0])
        ax10 = plt.subplot(gs[0, 0])
        ax20 = plt.subplot(gs[2, 0])
        axcbar2 = plt.subplot(gs[1, 1])
        ax11 = plt.subplot(gs[0, 1])
        ax21 = plt.subplot(gs[2, 1])
        axcbar3 = plt.subplot(gs[1, 2])
        ax12 = plt.subplot(gs[0, 2])
        ax22 = plt.subplot(gs[2, 2])
        axcbar4 = plt.subplot(gs[1, 3])
        ax13 = plt.subplot(gs[0, 3])
        ax23 = plt.subplot(gs[2, 3])
        axcbar5 = plt.subplot(gs[1, 4])
        ax14 = plt.subplot(gs[0, 4])
        ax24 = plt.subplot(gs[2, 4])
        
        return axcbar, ax10, ax20, axcbar2, ax11, ax21, axcbar3, ax12, ax22, axcbar4, ax13, ax23, axcbar5, ax14, ax24, x, y, area
    
    else:
        gs = gridspec.GridSpec(2, 1, hspace=0.05, height_ratios=[1, 0.5])
        ax00 = plt.subplot(gs[0, 0])
        ax10 = plt.subplot(gs[1, 0])
        
        return ax00, ax10, x, y, y2, area


def create_colorbar(ax, pcm, label, orientation='vertical'):
    """
    Generate a colorbar.
    :param ax: colorbar axis from create_axes
    :param pcm: pseudocolor plot
    :param label: colorbar label
    :param orientation: colorbar orientation
    :return: None
    """
    # Set the colorbar axes #
    cb = plt.colorbar(pcm, cax=ax, orientation=orientation)
    
    # Set the colorbar parameters #
    cb.set_label(label, size=36)
    for label in ax.xaxis.get_ticklabels():
        label.set_size(36)
    # cb.tick_params(direction='out', which='both')
    
    return None


def set_axes(ax00, ax10, xlabel=None, ylabel=None, y2label=None, ticks=False):
    """
    Set axes' parameters.
    :param ax00: top plot axes from create_axes
    :param ax10: bottom plot axes from create_axes
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param y2label: y2-axis label
    :param ticks: y2-axis label
    :return: None
    """
    
    # Set x-axis labels and ticks #
    if xlabel is None:
        ax00.set_xticks([])
        ax10.set_xticks([])
        ax00.set_xticklabels([])
        ax10.set_xticklabels([])
    else:
        ax00.set_xticklabels([])
        ax10.set_xlabel(xlabel, size=16)
    
    # Set y-axis labels and ticks #
    if ylabel is None:
        ax00.set_yticks([])
        ax00.set_yticklabels([])
    else:
        ax00.set_ylabel(ylabel, size=16)
    
    # Set y2-axis labels and ticks #
    if y2label is None:
        ax10.set_yticks([])
        ax10.set_yticklabels([])
    else:
        ax10.set_ylabel(y2label, size=16)
    
    # Set x- and y-axis ticks size #
    if ticks is True:
        for a in [ax00, ax10]:
            a.tick_params(direction='out', which='both', top='on', right='on')
            
            for label in a.xaxis.get_ticklabels():
                label.set_size(16)
            for label in a.yaxis.get_ticklabels():
                label.set_size(16)
    
    return None


def get_projection(pos_orig, mass, data, idir, res, boxsize, type, maxHsml=False):
    """
    Calculate particle projections.
    :param pos_orig: positions of particles
    :param mass: masses of particles
    :param data: projection of light or mass data
    :param idir: direction
    :param res: resolution
    :param boxsize: boxsize
    :param type: 'light' or 'mass'
    :param maxHsml:
    :return:
    """
    
    pos = np.zeros((np.size(mass), 3))  # Declare array to store the new positions of particles.
    
    # Generate projection planes #
    if idir == 0:  # XY plane
        pos[:, 0] = pos_orig[:, 1]  # y-axis.
        pos[:, 1] = pos_orig[:, 2]  # x-axis.
        pos[:, 2] = pos_orig[:, 0]  # 3rd dimension.
        
        xres = res
        yres = res
        boxx = boxsize
        boxy = boxsize
    
    elif idir == 1:  # XZ plane
        pos[:, 0] = pos_orig[:, 0]  # y-axis.
        pos[:, 1] = pos_orig[:, 2]  # x-axis.
        pos[:, 2] = pos_orig[:, 1]  # 3rd dimension.
        
        xres = res // 2
        yres = res
        boxx = boxsize / 2.0
        boxy = boxsize
    
    elif idir == 2:  # YZ plane
        pos[:, 0] = pos_orig[:, 1]
        pos[:, 1] = pos_orig[:, 0]
        pos[:, 2] = pos_orig[:, 2]
        
        xres = res
        yres = res // 2
        boxx = boxsize
        boxy = boxsize / 2.0
    
    tree = pysph.makeTree(pos)
    hsml = tree.calcHsmlMulti(pos, pos, mass, 48, numthreads=8)
    if maxHsml is True:
        hsml = np.minimum(hsml, 4.0 * boxsize / res)
    hsml = np.maximum(hsml, 1.001 * boxsize / res * 0.5)
    rho = np.ones(np.size(mass))
    
    datarange = np.array([[4003.36, 800672.], [199.370, 132913.], [133.698, 200548.]])
    fac = (512.0 / res) ** 2 * (0.5 * boxsize / 0.025) ** 2
    datarange *= fac
    
    boxz = max(boxx, boxy)
    
    # Calculate light projection #
    if type == 'light':
        proj = np.zeros((xres, yres, 3))
        for k in range(3):
            iband = [3, 1, 0][k]
            band = 10 ** (-2.0 * data[:, iband] / 5.0)
            
            grid = calcGrid.calcGrid(pos, hsml, band, rho, rho, xres, yres, 256, boxx, boxy, boxz, 0., 0., 0., 1, 1, numthreads=8)
            
            drange = datarange[k]
            grid = np.minimum(np.maximum(grid, drange[0]), drange[1])
            loggrid = np.log10(grid)
            logdrange = np.log10(drange)
            
            proj[:, :, k] = (loggrid - logdrange[0]) / (logdrange[1] - logdrange[0])
    
    # Calculate mass projection #
    elif type == 'mass':
        proj = calcGrid.calcGrid(pos, hsml, data, rho, rho, xres, yres, 256, boxx, boxy, boxz, 0., 0., 0., 1, 1, numthreads=8)
    
    else:
        print("Type %s not found." % type)
        
        return None
    
    return proj


def stellar_light(pdf, data, redshift, read):
    """
    Plot stellar light projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'sl/' + str(redshift) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'gsph', 'mass', 'pos']
        data.select_haloes(4, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
        
        # Loop over all haloes #
        for s in data:
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned to the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            
            # Mask and rotate the data and plot the projections #
            mask, = np.where((s.data['age'] > 0.0) & (s.r() * 1e3 < 30))  # Distances are in kpc.
            
            z_rotated, y_rotated, x_rotated = rotate_bar(s.pos[mask, 0] * 1e3, s.pos[mask, 1] * 1e3, s.pos[mask, 2] * 1e3)  # Distances are in kpc.
            s.pos = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.pos attribute in kpc.
            
            face_on = get_projection(s.pos.astype('f8'), s.mass[mask].astype('f8'), s.data['gsph'][mask].astype('f8'), 0, res, boxsize, 'light',
                                     maxHsml=True)
            edge_on = get_projection(s.pos.astype('f8'), s.mass[mask].astype('f8'), s.data['gsph'][mask].astype('f8'), 1, res, boxsize, 'light',
                                     maxHsml=True)
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'face_on_' + str(s.haloname), face_on)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_06.*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 10), dpi=300)
        ax00, ax10, x, y, y2, area = create_axes(res=res, boxsize=boxsize)
        for a in [ax00, ax10]:
            a.set_yticks([])
            a.set_xticks([])
            a.set_xticklabels([])
            a.set_aspect('equal')
        
        # Load and plot the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        ax00.imshow(face_on, interpolation='nearest', aspect='equal')
        ax10.imshow(edge_on, interpolation='nearest', aspect='equal')
        
        f.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[i])[1]) + ' redshift = ' + str(redshift), color='k', fontsize=16,
               transform=ax00.transAxes)
        
        pdf.savefig(f, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def stellar_density(pdf, data, redshift, read):
    """
    Plot stellar density projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'sd/' + str(redshift) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['mass', 'pos']
        data.select_haloes(4, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
        
        # Loop over all haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            
            # Generate the axes #
            ax00, ax01, ax10, ax11, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, contour=True)
            
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned to the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            
            # Mask and rotate the data and get the projections #
            mask, = np.where((s.data['age'] > 0.0) & (s.r() < 2.0 * boxsize))
            z_rotated, y_rotated, x_rotated = rotate_bar(s.pos[mask, 0] * 1e3, s.pos[mask, 1] * 1e3, s.pos[mask, 2] * 1e3)  # Distances are in kpc.
            s.pos = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.pos attribute in kpc.
            
            face_on = get_projection(s.pos.astype('f8'), s.mass[mask].astype('f8'), s.data['mass'][mask].astype('f8'), 0, res, boxsize,
                                     'mass') / area * 1e10
            edge_on = get_projection(s.pos.astype('f8'), s.mass[mask].astype('f8'), s.data['mass'][mask].astype('f8'), 1, res, boxsize, 'mass') / (
                0.5 * area) * 1e10
            
            # Get the contour lines #
            face_on_count, face_on_xedges, face_on_yedges = np.histogram2d(s.pos[:, 2] * 1e3, s.pos[:, 1] * 1e3, bins=70,
                                                                           range=[[-30, 30], [-30, 30]])
            edge_on_count, edge_on_xedges, edge_on_yedges = np.histogram2d(s.pos[:, 2] * 1e3, s.pos[:, 0] * 1e3, bins=20,
                                                                           range=[[-30, 30], [-30, 30]])
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'face_on_' + str(s.haloname), face_on)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
            np.save(path + 'face_on_count_' + str(s.haloname), face_on_count)
            np.save(path + 'edge_on_count_' + str(s.haloname), edge_on_count)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_06.*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 7.5), dpi=300)
        ax00, ax01, ax10, ax11, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, contour=True)
        for a in [ax00, ax01, ax10, ax11]:
            a.set_xlim(-30, 30)
            a.set_ylim(-30, 30)
            a.tick_params(direction='out', which='both', top='on', right='on')
        ax00.set_xticklabels([])
        ax01.set_xticklabels([])
        ax01.set_yticklabels([])
        ax11.set_yticklabels([])
        ax10.set_xlabel(r'$x\,\mathrm{[kpc]}$', size=16)
        ax11.set_xlabel(r'$x\,\mathrm{[kpc]}$', size=16)
        ax00.set_ylabel(r'$y\,\mathrm{[kpc]}$', size=16)
        ax10.set_ylabel(r'$z\,\mathrm{[kpc]}$', size=16)
        
        # Load and plot the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        face_on_count = np.load(path + 'face_on_count_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on_count = np.load(path + 'edge_on_count_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the projections #
        pcm = ax01.pcolormesh(x, y, face_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='twilight', rasterized=True)
        ax11.pcolormesh(x, y2, edge_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='twilight', rasterized=True)
        create_colorbar(axcbar, pcm, "$\Sigma_\mathrm{stars}\,\mathrm{[M_\odot\,kpc^{-2}]}$")
        
        # Plot the contour lines #
        ax00.contour(np.log10(face_on_count).T, colors="k", extent=[-30, 30, -30, 30], levels=np.arange(0.0, 5.0 + 0.5, 0.25))
        ax10.contour(np.log10(edge_on_count).T, colors="k", extent=[-30, 30, -30, 30], levels=np.arange(0.0, 5.0 + 0.5, 0.25))
        
        f.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[i])[1]) + ' redshift = ' + str(redshift), color='k', fontsize=16,
               transform=ax00.transAxes)
        pdf.savefig(f, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def gas_density(pdf, data, redshift, read):
    """
    Plot gas density projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'gd/' + str(redshift) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['mass', 'pos', 'rho']
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
            
            # Plot the projections #
            face_on = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] * boxsize * 1e3
            edge_on = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.0], proj=True, numthreads=8)["grid"] * boxsize * 1e3
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'face_on_' + str(s.haloname), face_on)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_06.*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 10), dpi=300)
        ax00, ax10, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        for a in [ax00, ax10]:
            a.tick_params(direction='out', which='both', top='on', right='on')
        ax00.set_xticklabels([])
        ax10.set_xlabel(r'$x\,\mathrm{[kpc]}$', size=16)
        ax00.set_ylabel(r'$y\,\mathrm{[kpc]}$', size=16)
        ax10.set_ylabel(r'$z\,\mathrm{[kpc]}$', size=16)
        
        # Load and plot the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the projections #
        pcm = ax00.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma', rasterized=True)
        ax10.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma', rasterized=True)
        create_colorbar(axcbar, pcm, "$\Sigma_\mathrm{gas}\,\mathrm{[M_\odot\,kpc^{-2}]}$")
        
        f.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[i])[1]) + ' redshift = ' + str(redshift), color='k', fontsize=16,
               transform=ax00.transAxes)
        pdf.savefig(f, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def gas_temperature(pdf, data, redshift, read):
    """
    Plot gas temperature projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'gt/' + str(redshift) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    # Read the data #
    if read is True:
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['mass', 'ne', 'pos', 'rho', 'u']
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
            
            # Plot the projections #
            meanweight = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * s.data['ne']) * 1.67262178e-24
            temperature = (5.0 / 3.0 - 1.0) * s.data['u'] / KB * (1e6 * parsec) ** 2.0 / (1e6 * parsec / 1e5) ** 2 * meanweight
            s.data['temprho'] = s.rho * temperature
            
            face_on = s.get_Aslice("temprho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]
            face_on_rho = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]
            edge_on = s.get_Aslice("temprho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.0], proj=True, numthreads=8)["grid"]
            edge_on_rho = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.0], proj=True, numthreads=8)["grid"]
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'face_on_' + str(s.haloname), face_on)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
            np.save(path + 'face_on_rho_' + str(s.haloname), face_on_rho)
            np.save(path + 'edge_on_rho_' + str(s.haloname), edge_on_rho)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 10), dpi=300)
        ax00, ax10, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        for a in [ax00, ax10]:
            a.set_xlim(-30, 30)
            a.tick_params(direction='out', which='both', top='on', right='on')
        ax00.set_ylim(-30, 30)
        ax10.set_ylim(-15, 15)
        ax00.set_xticklabels([])
        ax10.set_xlabel(r'$x\,\mathrm{[kpc]}$', size=16)
        ax00.set_ylabel(r'$y\,\mathrm{[kpc]}$', size=16)
        ax10.set_ylabel(r'$z\,\mathrm{[kpc]}$', size=16)
        
        # Load and plot the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        face_on_rho = np.load(path + 'face_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on_rho = np.load(path + 'edge_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the projections #
        pcm = ax00.pcolormesh(x, y, (face_on / face_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap='viridis', rasterized=True)
        ax10.pcolormesh(x, 0.5 * y, (edge_on / edge_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap='viridis', rasterized=True)
        create_colorbar(axcbar, pcm, "$T\,\mathrm{[K]}$")
        
        f.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[i])[1]) + ' redshift = ' + str(redshift), color='k', fontsize=16,
               transform=ax00.transAxes)
        pdf.savefig(f, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def gas_metallicity(pdf, data, level, redshift):
    """
    Plot gas metallicity projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param level: level from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
    particle_type = [0, 4]
    attributes = ['mass', 'pos', 'gz']
    data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
    
    # Loop over all haloes #
    for s in data:
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 10), dpi=300)
        ax00, ax10, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        f.text(0.0, 1.01, 'Au-' + str(s.haloname) + ' redshift = ' + str(redshift), color='k', fontsize=16, transform=ax00.transAxes)
        
        # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned to the z-axis #
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        # Plot the projections #
        face_on = s.get_Aslice("gz", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res / 0.0134
        edge_on = s.get_Aslice("gz", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.0], proj=True, numthreads=8)["grid"] / res / 0.0134
        pcm = ax00.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), cmap='viridis', rasterized=True)
        ax10.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), cmap='viridis', rasterized=True)
        create_colorbar(axcbar, pcm, "$Z/Z_\odot$")
        
        set_axes(ax00, ax10, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$', ticks=True)
        
        pdf.savefig(f, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def gas_slice(pdf, data, redshift, read):
    """
    Plot gas temperature projection for different temperature regimes along with velocity arrows Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean
    :return: None
    """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'gs/' + str(redshift) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    boxsize = 0.08  # Increase the boxsize.
    
    # Read the data #
    if read is True:
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['pos', 'vel', 'mass', 'u', 'ne', 'gz', 'gmet', 'rho', 'id', 'vol']
        data.select_haloes(4, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
        
        # Loop over all haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            
            # Select the halo and rotate it based on Euler's angles #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, remove_bulk_vel=True, use_principal_axis=False, euler_rotation=True, rotate_disk=True, do_rotation=True)
            
            # Mask the data #
            dist = np.max(np.abs(s.pos - s.center[None, :]), axis=1)
            mask, = np.where((s.type == 0) & (dist < 0.5 * boxsize))  # Mask the data: select gas particles inside a 30kpc sphere.
            ngas = np.size(mask)
            
            indy = [[1, 2, 0], [1, 0, 2]]  # Swap the position and velocity indices for the face-on and edge-on projections #
            # Loop over the two projections #
            for j in range(2):
                
                # Initialise arrays to store the data #
                frbs = []
                temp_pos = s.pos[mask, :]
                temp_vel = s.vel[mask, :]
                rgbArray = np.zeros((res, res, 3), 'uint8')
                pos, vel = np.zeros((np.size(mask), 3)), np.zeros((np.size(mask), 3))
                u, rho, z, vol = np.zeros(ngas), np.zeros(ngas), np.zeros(ngas), np.zeros(ngas)
                for i in range(3):
                    pos[:, i] = temp_pos[:, indy[j][i]]
                    vel[:, i] = temp_vel[:, indy[j][i]]
                
                # Calculate the temperature of the gas cells #
                vol[:] = s.data['vol'][mask] * 1e9
                XH = s.data['gmet'][mask, element['H']]
                ne, mass, rho[:], metallicity = s.data['ne'][mask], s.data['mass'][mask], s.data['rho'][mask], s.data['gz'][mask]
                yhelium = (1 - XH - metallicity) / (4. * XH)
                mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
                temperature = GAMMA_MINUS1 * s.data['u'][mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN
                
                # Define the three temperatures regimes #
                sfgas = np.where((temperature < 2e4))
                medgas = np.where((temperature >= 1e4) & (temperature < 6e5))
                hotgas = np.where((temperature >= 5e5))
                
                # Loop over the three temperature regimes and get a gas slice #
                for i in range(3):
                    if i == 0:
                        gpos, gmass, grho = np.zeros((np.size(hotgas), 3)), np.zeros((np.size(hotgas))), np.zeros((np.size(hotgas)))
                        gpos[:, :], gmass[:], grho[:] = pos[hotgas, :], mass[hotgas], u[hotgas]
                    if i == 1:
                        gpos, gmass, grho = np.zeros((np.size(medgas), 3)), np.zeros((np.size(medgas))), np.zeros((np.size(medgas)))
                        gpos[:, :], gmass[:], grho[:] = pos[medgas, :], mass[medgas], rho[medgas]
                    if i == 2:
                        gpos, gmass, grho = np.zeros((np.size(sfgas), 3)), np.zeros((np.size(sfgas))), np.zeros((np.size(sfgas)))
                        gpos[:, :], gmass[:], grho[:] = pos[sfgas, :], mass[sfgas], rho[sfgas]
                    
                    A = calcGrid.calcASlice(gpos, grho, res, res, boxx=boxsize, boxy=boxsize, centerx=s.center[0], centery=s.center[1],
                                            centerz=s.center[2], grad=gpos, proj=True, boxz=boxsize / 8, nz=int(res / 8), numthreads=8)
                    frbs.append(np.array(A["grid"]))
                
                # Create a coloured image from the gas slice #
                for i in range(3):
                    frbs[i][np.where(frbs[i] == 0)] = 1e-10
                    frbs_flat = frbs[i].flatten()
                    asort = np.argsort(frbs_flat)
                    frbs_flat = frbs_flat[asort]
                    CumSum = np.cumsum(frbs_flat)
                    CumSum /= CumSum[-1]
                    halflight_val = frbs_flat[np.where(CumSum > 0.5)[0][0]]
                    
                    # Define the maximum and minimum r, g, b magnitudes #
                    rmax, rmin, gmax, gmin, bmax, bmin = 1.0, 0.0, 1.2, -1.0, 0.8, -0.5
                    if i == 0:
                        Max = np.log10(halflight_val) + rmax
                        Min = np.log10(halflight_val) + rmin
                    elif i == 1:
                        Max = np.log10(halflight_val) + gmax
                        Min = np.log10(halflight_val) + gmin
                    elif i == 2:
                        Max = np.log10(halflight_val) + bmax
                        Min = np.log10(halflight_val) + bmin
                    
                    Max -= 0.5
                    Min -= 0.1
                    
                    Color = (np.log10(frbs[i]) - Min) / (Max - Min)
                    Color[np.where(Color < 0.)] = 0.
                    Color[np.where(Color > 1.0)] = 1.0
                    
                    if i == 0:
                        Color[np.where(Color < 0.15)] = 0.
                    
                    A = np.array(Color * 255, dtype=np.uint8)
                    if j == 0:
                        rgbArray[:, :, i] = A.T
                    elif j == 1:
                        rgbArray[:, :, i] = A.T
                
                # Calculate and plot the arrows of the velocity field #
                pn, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1], bins=30,
                                                    range=[[-0.5 * boxsize, 0.5 * boxsize], [-0.5 * boxsize, 0.5 * boxsize]])
                vxgrid, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1], bins=30, weights=vel[:, 0],
                                                        range=[[-0.5 * boxsize, 0.5 * boxsize], [-0.5 * boxsize, 0.5 * boxsize]])
                vygrid, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1], bins=30, weights=vel[:, 1],
                                                        range=[[-0.5 * boxsize, 0.5 * boxsize], [-0.5 * boxsize, 0.5 * boxsize]])
                vxgrid /= pn
                vygrid /= pn
                vygrid *= (-1)
                
                # Sample a velocity field grid #
                xbin, ybin = np.zeros(len(xedges) - 1), np.zeros(len(yedges) - 1)
                xbin[:] = 0.5 * (xedges[:-1] + xedges[1:])
                ybin[:] = 0.5 * (yedges[:-1] + yedges[1:])
                xbin -= xedges[0]
                xbin *= (10 * res) * (0.1 / boxsize)
                ybin -= yedges[-1]
                ybin *= (-10 * res) * (0.1 / boxsize)
                xc, yc = np.meshgrid(xbin, ybin)
                
                # Save data for each halo in numpy arrays #
                np.save(path + 'name_' + str(s.haloname), s.haloname)
                np.save(path + 'xc_' + str(j) + '_' + str(s.haloname), xc)
                np.save(path + 'yc_' + str(j) + '_' + str(s.haloname), yc)
                np.save(path + 'vxgrid_' + str(j) + '_' + str(s.haloname), vxgrid)
                np.save(path + 'vygrid_' + str(j) + '_' + str(s.haloname), vygrid)
                np.save(path + 'rgbArray_' + str(j) + '_' + str(s.haloname), rgbArray)
    
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 15), dpi=300)
        ax00, ax10, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, velocity_vectors=True)
        f.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', str(names[i]))[1]) + ' redshift = ' + str(redshift), color='k', fontsize=16,
               transform=ax00.transAxes)
        for a in [ax00, ax10]:
            a.set_xticks([])
            a.set_yticks([])
            a.set_xticklabels([])
            a.set_yticklabels([])
        
        # Load and plot the data #
        j = 0
        for a in [ax00, ax10]:
            xc = np.load(path + 'xc_' + str(j) + '_' + str(re.split('_|.npy', str(names[i]))[1]) + '.npy')
            yc = np.load(path + 'yc_' + str(j) + '_' + str(re.split('_|.npy', str(names[i]))[1]) + '.npy')
            vxgrid = np.load(path + 'vxgrid_' + str(j) + '_' + str(re.split('_|.npy', str(names[i]))[1]) + '.npy')
            vygrid = np.load(path + 'vygrid_' + str(j) + '_' + str(re.split('_|.npy', str(names[i]))[1]) + '.npy')
            rgbArray = np.load(path + 'rgbArray_' + str(j) + '_' + str(re.split('_|.npy', str(names[i]))[1]) + '.npy')
            
            a.imshow(rgbArray, rasterized=True, aspect='equal')
            a.quiver(xc, yc, np.rot90(vxgrid), np.rot90(vygrid), scale=4000.0, pivot='middle', color='yellow', alpha=0.8)
            j += 1
        pdf.savefig(f, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def bfld(pdf, data, level, redshift):
    """
    Plot gas metallicity projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param level: level from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
    particle_type = [0, 4]
    attributes = ['mass', 'pos', 'bfld']
    data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
    
    # Loop over all haloes #
    for s in data:
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 10), dpi=300)
        ax00, ax10, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        f.text(0.0, 1.01, 'Au-' + str(s.haloname) + ' redshift = ' + str(redshift), color='k', fontsize=16, transform=ax00.transAxes)
        
        # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned to the z-axis #
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        # Plot the projections #
        s.data['b2'] = (s.data['bfld'] ** 2.).sum(axis=1)
        face_on = np.sqrt(s.get_Aslice("b2", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res) * bfac * 1e6
        edge_on = np.sqrt(s.get_Aslice("b2", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res) * bfac * 1e6
        pcm = ax00.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=5e1), cmap='CMRmap', rasterized=True)
        ax10.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=5e1), cmap='CMRmap', rasterized=True)
        create_colorbar(axcbar, pcm, "$B\,\mathrm{[\mu G]}$")
        
        set_axes(ax00, ax10, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$', ticks=True)
        
        pdf.savefig(f, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def dm_mass(pdf, data, level, redshift):
    """
    Plot gas metallicity projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param level: level from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    boxsize = 0.4  # Increase the boxsize.
    
    # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
    particle_type = [1, 4]
    attributes = ['mass', 'pos']
    data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
    
    # Loop over all haloes #
    for s in data:
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 10), dpi=300)
        ax00, ax10, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        f.text(0.0, 1.01, 'Au-' + str(s.haloname) + ' redshift = ' + str(redshift), color='k', fontsize=16, transform=ax00.transAxes)
        
        # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned to the z-axis #
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        # Mask the data and plot the projections #
        mask, = np.where((s.r() < 2.0 * boxsize) & (s.type == 1))
        face_on = get_projection(s.pos[mask].astype('f8'), s.mass[mask].astype('f8'), s.data['mass'][mask].astype('f8'), 0, res, boxsize,
                                 'mass') / area * 1e10
        edge_on = get_projection(s.pos[mask].astype('f8'), s.mass[mask].astype('f8'), s.data['mass'][mask].astype('f8'), 1, res, boxsize, 'mass') / (
            0.5 * area) * 1e10
        pcm = ax00.pcolormesh(x, y, face_on, norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e9), cmap='Greys', rasterized=True)
        ax10.pcolormesh(x, y2, edge_on, norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e9), cmap='Greys', rasterized=True)
        create_colorbar(axcbar, pcm, "$\Sigma_\mathrm{DM}\,\mathrm{[M_\odot\,kpc^{-2}]}$")
        
        set_axes(ax00, ax10, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$', ticks=True)
        
        pdf.savefig(f, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def rotate_bar(z, y, x):
    """
    Calculate bar strength and rotate bar to horizontal position
    :param z: the z-position of the particles.
    :param y: the y-position of the particles.
    :param x: the x-position of the particles.
    :return:
    """
    nbins = 40  # Number of radial bins.
    r = np.sqrt(x[:] ** 2 + y[:] ** 2)  # Radius of each particle.
    
    # Initialise fourier components
    r_m = np.zeros(nbins)
    beta_2 = np.zeros(nbins)
    alpha_0 = np.zeros(nbins)
    alpha_2 = np.zeros(nbins)
    
    # Split disc in radial bins and calculate Fourier components #
    for i in range(0, nbins):
        r_s = float(i) * 0.25
        r_b = float(i) * 0.25 + 0.25
        r_m[i] = float(i) * 0.25 + 0.125
        xfit = x[(r < r_b) & (r > r_s)]
        yfit = y[(r < r_b) & (r > r_s)]
        l = len(xfit)
        for k in range(0, l):
            th_i = np.arctan2(yfit[k], xfit[k])
            alpha_0[i] = alpha_0[i] + 1
            alpha_2[i] = alpha_2[i] + np.cos(2 * th_i)
            beta_2[i] = beta_2[i] + np.sin(2 * th_i)
    
    # Calculate bar rotation angle for each time by averaging over radii between 1 and 5 kpc #
    r_b = 5  # In kpc.
    r_s = 1  # In kpc.
    k = 0.0
    phase_in = 0.0
    for i in range(0, nbins):
        if (r_m[i] < r_b) & (r_m[i] > r_s):
            k = k + 1.
            phase_in = phase_in + 0.5 * np.arctan2(beta_2[i], alpha_2[i])
    phase_in = phase_in / k
    
    # Transform back -tangle to horizontal position #
    z_pos = z[:]
    y_pos = np.cos(-phase_in) * (y[:]) + np.sin(-phase_in) * (x[:])
    x_pos = np.cos(-phase_in) * (x[:]) - np.sin(-phase_in) * (y[:])
    return z_pos / 1e3, y_pos / 1e3, x_pos / 1e3  # Distances are in kpc.


def gas_temperature_edge_on(pdf, data, redshift, read):
    """
    Plot gas temperature projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean
    :return: None
    """
    boxsize = 0.2  # Increase the boxsize.
    
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'gteo/' + str(redshift) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    # Read the data #
    if read is True:
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['mass', 'ne', 'pos', 'rho', 'u']
        data.select_haloes(4, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
        
        # Loop over all haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            # names = glob.glob(path + '/name_*')
            # names = [re.split('_|.npy', name)[1] for name in names]
            # if str(s.haloname) in names:
            #     continue
            
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned to the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            
            # Plot the density-weighted temperature projections #
            meanweight = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * s.data['ne']) * 1.67262178e-24
            temperature = (5.0 / 3.0 - 1.0) * s.data['u'] / KB * (1e6 * parsec) ** 2.0 / (1e6 * parsec / 1e5) ** 2 * meanweight
            s.data['temprho'] = s.rho * temperature
            
            edge_on = s.get_Aslice("temprho", res=res, axes=[1, 0], box=[boxsize, boxsize], boxz=1e-3, proj=True, numthreads=8)["grid"]
            edge_on_rho = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize], boxz=1e-3, proj=True, numthreads=8)["grid"]
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
            np.save(path + 'edge_on_rho_' + str(s.haloname), edge_on_rho)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(10, 10), dpi=300)
        gs = gridspec.GridSpec(1, 2, wspace=0.05, width_ratios=[1, 0.05])
        ax00 = plt.subplot(gs[0, 0])
        axcbar = plt.subplot(gs[:, 1])
        ax00.set_xlabel(r'$x\,\mathrm{[kpc]}$', size=16)
        ax00.set_ylabel(r'$z\,\mathrm{[kpc]}$', size=16)
        ax00.tick_params(direction='out', which='both', top='on', right='on')
        x = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
        z = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
        
        # Load and plot the data #
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on_rho = np.load(path + 'edge_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the projections #
        pcm = ax00.pcolormesh(x * 1e3, z * 1e3, (edge_on / edge_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=2e7), cmap='Spectral_r',
                              rasterized=True)
        create_colorbar(axcbar, pcm, "$T\,\mathrm{[K]}$")
        
        f.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[i])[1]) + ' redshift = ' + str(redshift), color='k', fontsize=16,
               transform=ax00.transAxes)
        pdf.savefig(f, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def multiple(pdf, data, redshift, read):
    """
    Plot stellar density projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean
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
        data.select_haloes(4, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
        
        # Loop over all haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            # if str(s.haloname) in names:
            #     continue
            
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned to the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            
            # Get the gas density projections #
            density_face_on = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] * boxsize * 1e3
            density_edge_on = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] * boxsize * 1e3
            
            # Get the gas temperature projections #
            meanweight = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * s.data['ne']) * 1.67262178e-24
            temperature = (5.0 / 3.0 - 1.0) * s.data['u'] / KB * (1e6 * parsec) ** 2.0 / (1e6 * parsec / 1e5) ** 2 * meanweight
            s.data['temprho'] = s.rho * temperature
            
            temperature_face_on = s.get_Aslice("temprho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]
            temperature_face_on_rho = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]
            temperature_edge_on = s.get_Aslice("temprho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]
            temperature_edge_on_rho = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]
            
            # Get the magnetic field projections #
            s.data['b2'] = (s.data['bfld'] ** 2.).sum(axis=1)
            bfld_face_on = np.sqrt(
                s.get_Aslice("b2", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res) * bfac * 1e6
            bfld_edge_on = np.sqrt(
                s.get_Aslice("b2", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res) * bfac * 1e6
            
            # Get the gas sfr projections #
            sfr_face_on = s.get_Aslice("sfr", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] * boxsize * 1e3
            sfr_edge_on = s.get_Aslice("sfr", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] * boxsize * 1e3
            
            bfac = np.sqrt(1e10 * msol) / np.sqrt(1e6 * parsec) * 1e5 / (1e6 * parsec)
            elements_mass = [1.01, 4.00, 12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33]
            meanweight = np.sum(s.gmet[s.type == 0, 0:9], axis=1) / (
                np.sum(s.gmet[s.type == 0, 0:9] / elements_mass[0:9], axis=1) + s.data['ne'] * s.gmet[s.type == 0, 0])
            Tfac = 1. / meanweight * (1.0 / (5. / 3. - 1.)) * KB / PROTONMASS * 1e10 * msol / 1.989e53
            
            # Get the gas total pressure projections #
            s.data['T'] = s.u / Tfac
            s.data['dens'] = s.rho / (1e6 * parsec) ** 3. * msol * 1e10
            s.data['n_H'] = s.data['dens'] / PROTONMASS * s.gmet[s.type == 0, 0]
            s.data['Ptherm'] = s.data['dens'] * s.data['T'] / (meanweight * PROTONMASS)
            s.data['B'] = np.sqrt((s.bfld ** 2).sum(axis=1)) * bfac
            s.data['PB'] = s.data['B'] ** 2. / (8 * np.pi) / KB
            s.data['Ptot'] = s.data['Ptherm'] + s.data['PB']
            
            pressure_face_on = s.get_Aslice("Ptot", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] * boxsize * 1e3
            pressure_edge_on = s.get_Aslice("Ptot", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] * boxsize * 1e3
            
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
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_17.*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        f = plt.figure(figsize=(50, 22.5), dpi=300)
        axcbar, ax10, ax20, axcbar2, ax11, ax21, axcbar3, ax12, ax22, axcbar4, ax13, ax23, axcbar5, ax14, ax24, x, y, area = create_axes(res=res,
                                                                                                                                         boxsize=boxsize * 1e3,
                                                                                                                                         multiple=True)
        for a in [ax10, ax11, ax12, ax13, ax14]:
            a.set_xlim(-2, 2)
            a.set_ylim(-2, 2)
            a.set_aspect('equal')
            a.set_xticklabels([])
            a.tick_params(direction='out', which='both', top='on', right='on')
        for a in [ax20, ax21, ax22, ax23, ax24]:
            a.set_xlim(-2, 2)
            a.set_ylim(-2, 2)
            a.set_xlabel(r'$x\,\mathrm{[kpc]}$', size=36)
            a.tick_params(direction='out', which='both', top='on', right='on')
            for label in a.xaxis.get_ticklabels():
                label.set_size(36)
        for a in [ax11, ax21, ax12, ax22, ax13, ax23, ax14, ax24]:
            a.set_yticklabels([])
        
        ax10.set_ylabel(r'$y\,\mathrm{[kpc]}$', size=36)
        for label in ax20.yaxis.get_ticklabels():
            label.set_size(36)
        ax20.set_ylabel(r'$z\,\mathrm{[kpc]}$', size=36)
        for label in ax10.yaxis.get_ticklabels():
            label.set_size(36)
        
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
        
        # Plot the gas density projections #
        pcm = ax10.pcolormesh(x, y, density_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='magma', rasterized=True)
        ax20.pcolormesh(x, y, density_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='magma', rasterized=True)
        create_colorbar(axcbar, pcm, "$\Sigma_\mathrm{gas}\,\mathrm{[M_\odot\,kpc^{-2}]}$", orientation='horizontal')
        
        # Plot the gas temperature projections #
        pcm = ax11.pcolormesh(x, y, (temperature_face_on / temperature_face_on_rho).T, norm=matplotlib.colors.LogNorm(), cmap='viridis',
                              rasterized=True)
        ax21.pcolormesh(x, y, (temperature_edge_on / temperature_edge_on_rho).T, norm=matplotlib.colors.LogNorm(), cmap='viridis', rasterized=True)
        create_colorbar(axcbar2, pcm, "$T\,\mathrm{[K]}$", orientation='horizontal')
        
        # Plot the magnetic field projections #
        pcm = ax12.pcolormesh(x, y, bfld_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='CMRmap', rasterized=True)
        ax22.pcolormesh(x, y, bfld_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='CMRmap', rasterized=True)
        create_colorbar(axcbar3, pcm, "$B\,\mathrm{[\mu G]}$", orientation='horizontal')
        
        # Plot the sfr projections #
        pcm = ax13.pcolormesh(x, y, sfr_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='gist_heat', rasterized=True)
        ax23.pcolormesh(x, y, sfr_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='gist_heat', rasterized=True)
        create_colorbar(axcbar4, pcm, "$SFR\,\mathrm{[M_\odot\,yr^{-1}]}$", orientation='horizontal')
        
        # Plot the gas total pressure projections #
        pcm = ax14.pcolormesh(x, y, pressure_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='twilight', rasterized=True)
        ax24.pcolormesh(x, y, pressure_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='twilight', rasterized=True)
        create_colorbar(axcbar5, pcm, "$P\,\mathrm{[M_\odot\,yr^{-1}]}$", orientation='horizontal')
        
        for a in [axcbar, axcbar2, axcbar3, axcbar4, axcbar5]:
            a.xaxis.tick_top()
            a.xaxis.set_label_position("top")
            a.tick_params(direction='out', which='both', top='on', right='on')
        
        f.text(0, 0, 'Au-' + str(re.split('_|.npy', names[i])[1]) + ' redshift = ' + str(redshift), color='white', fontsize=36,
               transform=ax20.transAxes)
        pdf.savefig(f, bbox_inches='tight')  # S ave the figure.
        plt.close()
    return None