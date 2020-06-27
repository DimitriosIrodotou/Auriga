import os
import re
import glob
import const
import pysph
import calcGrid
import matplotlib
import plot_tools

import numpy as np
import matplotlib.pyplot as plt

from sfigure import *
from loadmodules import *
from matplotlib import gridspec

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
        gs = gridspec.GridSpec(2, 3, hspace=0.05, wspace=0.0, height_ratios=[1, 0.5], width_ratios=[1, 1, 0.05])
        axis00 = plt.subplot(gs[0, 0])
        axis01 = plt.subplot(gs[0, 1])
        axis10 = plt.subplot(gs[1, 0])
        axis11 = plt.subplot(gs[1, 1])
        axiscbar = plt.subplot(gs[:, 2])
        
        return axis00, axis01, axis10, axis11, axiscbar, x, y, y2, area
    
    elif colorbar is True:
        gs = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.0, height_ratios=[1, 0.5], width_ratios=[1, 0.05])
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
        axis10 = plt.subplot(gs[1, 0])
        axis20 = plt.subplot(gs[2, 0])
        axis01 = plt.subplot(gs[0, 1])
        axis11 = plt.subplot(gs[1, 1])
        axis21 = plt.subplot(gs[2, 1])
        axis02 = plt.subplot(gs[0, 2])
        axis12 = plt.subplot(gs[1, 2])
        axis22 = plt.subplot(gs[2, 2])
        axis03 = plt.subplot(gs[0, 3])
        axis13 = plt.subplot(gs[1, 3])
        axis23 = plt.subplot(gs[2, 3])
        axis04 = plt.subplot(gs[0, 4])
        axis14 = plt.subplot(gs[1, 4])
        axis24 = plt.subplot(gs[2, 4])
        axis05 = plt.subplot(gs[0, 5])
        axis15 = plt.subplot(gs[1, 5])
        axis25 = plt.subplot(gs[2, 5])
        
        return axis00, axis10, axis20, axis01, axis11, axis21, axis02, axis12, axis22, axis03, axis13, axis23, axis04, axis14, axis24, axis05, \
               axis15, axis25, x, y, area
    
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
        gs = gridspec.GridSpec(2, 1, hspace=0.05)
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])
        
        return axis00, axis10, x, y, y2, area


def set_axes(axis00, axis10, xlabel=None, ylabel=None, y2label=None, ticks=False):
    """
    Set axes' parameters.
    :param axis00: top plot axes from create_axes
    :param axis10: bottom plot axes from create_axes
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param y2label: y2-axis label
    :param ticks: y2-axis label
    :return: None
    """
    
    # Set x-axis labels and ticks #
    if xlabel is None:
        axis00.set_xticks([])
        axis10.set_xticks([])
        axis00.set_xticklabels([])
        axis10.set_xticklabels([])
    else:
        axis00.set_xticklabels([])
        axis10.set_xlabel(xlabel, size=16)
    
    # Set y-axis labels and ticks #
    if ylabel is None:
        axis00.set_yticks([])
        axis00.set_yticklabels([])
    else:
        axis00.set_ylabel(ylabel, size=16)
    
    # Set y2-axis labels and ticks #
    if y2label is None:
        axis10.set_yticks([])
        axis10.set_yticklabels([])
    else:
        axis10.set_ylabel(y2label, size=16)
    
    # Set x- and y-axis ticks size #
    if ticks is True:
        for axis in [axis00, axis10]:
            axis.tick_params(direction='out', which='both', top='on', right='on')
            
            for label in axis.xaxis.get_ticklabels():
                label.set_size(16)
            for label in axis.yaxis.get_ticklabels():
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
    :return: proj
    """
    pos = np.zeros((np.size(mass), 3))  # Declare array to store the data.
    
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
    
    boxz = max(boxx, boxy)
    tree = pysph.makeTree(pos)
    hsml = tree.calcHsmlMulti(pos, pos, mass, 48, numthreads=8)
    
    if maxHsml is True:
        hsml = np.minimum(hsml, 4.0 * boxsize / res)
    hsml = np.maximum(hsml, 1.001 * boxsize / res * 0.5)
    rho = np.ones(np.size(mass))
    
    datarange = np.array([[4003.36, 800672.], [199.370, 132913.], [133.698, 200548.]])
    fac = (512.0 / res) ** 2 * (0.5 * boxsize / 0.025) ** 2
    datarange *= fac
    
    # Calculate light projection #
    if type == 'light':
        boxx = boxy
        xres = yres
        proj = np.zeros((xres, yres, 3))
        for k in range(3):
            iband = [3, 1, 0][k]  # bands = ['U', 'B', 'V', 'K', 'g', 'r', 'i', 'z']
            band = 10 ** (-2.0 * data[:, iband] / 5.0)
            grid = calcGrid.calcGrid(pos, hsml, band, rho, rho, xres, yres, 256, boxx, boxy, boxz, 0., 0., 0., 1, 1, numthreads=8)
            
            drange = datarange[k]
            grid = np.minimum(np.maximum(grid, drange[0]), drange[1])
            loggrid = np.log10(grid)
            logdrange = np.log10(drange)
            
            proj[:, :, k] = (loggrid - logdrange[0]) / (logdrange[1] - logdrange[0])
    
    # Get mass projection #
    elif type == 'mass':
        proj = calcGrid.calcGrid(pos, hsml, data, rho, rho, xres, yres, 256, boxx, boxy, boxz, 0., 0., 0., 1, 1, numthreads=8)
    
    return proj


def stellar_light(pdf, data, redshift, read):
    """
    Plot stellar light projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    path = '/u/di43/Auriga/plots/data/' + 'sl/' + str(redshift) + '/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'gsph', 'mass', 'pos']
        data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
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
            
            # Rotate the data and plot the projections #
            stellar_mask, = np.where((s.data['age'] > 0.0) & (s.r() * 1e3 < 30))  # Mask the data: select stellar particles inside a 30kpc sphere.
            z_rotated, y_rotated, x_rotated = plot_tools.rotate_bar(s.data['pos'][stellar_mask, 0] * 1e3, s.data['pos'][stellar_mask, 1] * 1e3,
                                                                    s.data['pos'][stellar_mask, 2] * 1e3)  # Distances are in kpc.
            s.data['pos'] = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.data['pos'] attribute in kpc.
            
            face_on = get_projection(s.data['pos'].astype('f8'), s.data['mass'][stellar_mask].astype('f8'), s.data['gsph'][stellar_mask].astype('f8'),
                                     0, res, boxsize, 'light', maxHsml=True)
            edge_on = get_projection(s.data['pos'].astype('f8'), s.data['mass'][stellar_mask].astype('f8'), s.data['gsph'][stellar_mask].astype('f8'),
                                     1, res, boxsize, 'light', maxHsml=True)
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'face_on_' + str(s.haloname), face_on)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(10, 10))
        axis00, axis10, x, y, y2, area = create_axes(res=res, boxsize=boxsize)
        for axis in [axis00, axis10]:
            axis.set_yticks([])
            axis.set_xticks([])
            plot_tools.set_axis(axis00)
        figure.text(0.0, 1.01, r'$\mathrm{Au-%s\;z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), color='k', fontsize=16,
                    transform=axis00.transAxes)
        
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the stellar light projections #
        axis00.imshow(face_on, interpolation='nearest', aspect='equal')
        axis10.imshow(edge_on, interpolation='nearest', aspect='equal')
        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def stellar_density(pdf, data, redshift, read):
    """
    Plot stellar density projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    path = '/u/di43/Auriga/plots/data/' + 'sd/' + str(redshift) + '/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['mass', 'pos']
        data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
        # Loop over all available haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            
            axis00, axis01, axis10, axis11, axiscbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, contour=True)  # Generate the axes.
            
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            
            # Rotate the data and get the projections #
            stellar_mask, = np.where(s.data['age'] > 0.0)  # Mask the data: select stellar particles.
            z_rotated, y_rotated, x_rotated = plot_tools.rotate_bar(s.data['pos'][stellar_mask, 0] * 1e3, s.data['pos'][stellar_mask, 1] * 1e3,
                                                                    s.data['pos'][stellar_mask, 2] * 1e3)  # Distances are in kpc.
            s.data['pos'] = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.data['pos'] attribute in kpc.
            
            face_on = get_projection(s.data['pos'].astype('f8'), s.data['mass'][stellar_mask].astype('f8'), s.data['mass'][stellar_mask].astype('f8'),
                                     0, res, boxsize, 'mass') / area * 1e10
            edge_on = get_projection(s.data['pos'].astype('f8'), s.data['mass'][stellar_mask].astype('f8'), s.data['mass'][stellar_mask].astype('f8'),
                                     1, res, boxsize, 'mass') / (0.5 * area) * 1e10
            
            # Get the contour lines #
            face_on_count, face_on_xedges, face_on_yedges = np.histogram2d(s.data['pos'][:, 2] * 1e3, s.data['pos'][:, 1] * 1e3, bins=70,
                                                                           range=[[-30, 30], [-30, 30]])
            edge_on_count, edge_on_xedges, edge_on_yedges = np.histogram2d(s.data['pos'][:, 2] * 1e3, s.data['pos'][:, 0] * 1e3, bins=20,
                                                                           range=[[-30, 30], [-15, 15]])
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'face_on_' + str(s.haloname), face_on)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
            np.save(path + 'face_on_count_' + str(s.haloname), face_on_count)
            np.save(path + 'edge_on_count_' + str(s.haloname), edge_on_count)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(16, 9))
        axis00, axis01, axis10, axis11, axiscbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, contour=True)
        plot_tools.set_axis(axis00, xlim=[-30, 30], ylim=[-30, 30], ylabel=r'$\mathrm{y/kpc}$')
        plot_tools.set_axis(axis01, xlim=[-30, 30], ylim=[-30, 30], ylabel=r'$\mathrm{y/kpc}$')
        plot_tools.set_axis(axis10, xlim=[-30, 30], ylim=[-15, 15], xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{z/kpc}$')
        plot_tools.set_axis(axis11, xlim=[-30, 30], ylim=[-15, 15], xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{z/kpc}$')
        cmap = matplotlib.cm.get_cmap('twilight')
        axis01.set_facecolor(cmap(0))
        figure.text(0.0, 1.01, r'$\mathrm{Au-%s\;z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), color='k', fontsize=16,
                    transform=axis00.transAxes)
        
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        face_on_count = np.load(path + 'face_on_count_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on_count = np.load(path + 'edge_on_count_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the stellar density projections #
        pcm = axis01.pcolormesh(x, y, face_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='twilight', rasterized=True)
        axis11.pcolormesh(x, y2, edge_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='twilight', rasterized=True)
        plot_tools.create_colorbar(axiscbar, pcm, r'$\mathrm{\Sigma_{\bigstar}/(M_\odot\;kpc^{-2})}$')
        
        # Plot the contour lines #
        axis00.contour(np.log10(face_on_count).T, colors='black', extent=[-30, 30, -30, 30], levels=np.arange(0.0, 5.0 + 0.5, 0.25))
        axis10.contour(np.log10(edge_on_count).T, colors='black', extent=[-30, 30, -15, 15], levels=np.arange(0.0, 5.0 + 0.5, 0.25))
        
        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def stellar_light_fit(data, redshift, read):
    """
    Plot stellar light projection for Auriga halo(es).
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    path = '/u/di43/Auriga/plots/data/' + 'slf/' + str(redshift) + '/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'gsph', 'mass', 'pos']
        data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
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
            
            # Rotate the data and plot the projections #
            stellar_mask, = np.where((s.data['age'] > 0.0) & (s.r() * 1e3 < 30))  # Mask the data: select stellar particles inside a 30kpc sphere.
            z_rotated, y_rotated, x_rotated = plot_tools.rotate_bar(s.data['pos'][stellar_mask, 0] * 1e3, s.data['pos'][stellar_mask, 1] * 1e3,
                                                                    s.data['pos'][stellar_mask, 2] * 1e3)  # Distances are in kpc.
            s.data['pos'] = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.data['pos'] attribute in kpc.
            
            face_on = get_projection(s.data['pos'].astype('f8'), s.data['mass'][stellar_mask].astype('f8'), s.data['gsph'][stellar_mask].astype('f8'),
                                     0, res, boxsize, 'light', maxHsml=True)
            edge_on = get_projection(s.data['pos'].astype('f8'), s.data['mass'][stellar_mask].astype('f8'), s.data['gsph'][stellar_mask].astype('f8'),
                                     1, res, boxsize, 'light', maxHsml=True)
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'face_on_' + str(s.haloname), face_on)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Loop over all projections #
        projections = ['face_on', 'edge_on']
        for projection in projections:
            # Generate the figure and define its parameters #
            figure, axis = plt.subplots(1, figsize=(10, 10), frameon=False)
            plt.axis('off')
            axis.set_aspect('equal')
            
            proj = np.load(path + projection + '_' + str(re.split('_|.npy', names[i])[1]) + '.npy')  # Load the data.
            
            # Save and close the figure #
            plt.imsave('/u/di43/Auriga/plots/slf/' + 'Au-' + str(re.split('_|.npy', names[i])[1]) + '_' + str(projection) + '.png', proj, cmap='gray')
            plt.close()
    return None


def r_band_magnitude(data, redshift, read):
    """
    Plot the 2D distribution of the r-band magnitude for Auriga halo(es).
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    path = '/u/di43/Auriga/plots/data/' + 'rbm/' + str(redshift) + '/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'gsph', 'mass', 'pos']
        data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
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
            
            # Rotate the data and plot the projections #
            stellar_mask, = np.where((s.data['age'] > 0.0) & (s.r() * 1e3 < 30))  # Mask the data: select stellar particles inside a 30kpc sphere.
            z_rotated, y_rotated, x_rotated = plot_tools.rotate_bar(s.data['pos'][stellar_mask, 0] * 1e3, s.data['pos'][stellar_mask, 1] * 1e3,
                                                                    s.data['pos'][stellar_mask, 2] * 1e3)  # Distances are in kpc.
            s.data['pos'] = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.data['pos'] attribute in kpc.
            
            band = 10 ** (-2.0 * s.data['gsph'][stellar_mask, 5] / 5.0)  # r-band magnitude.
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'band_' + str(s.haloname), band)
            np.save(path + 'pos_' + str(s.haloname), s.data['pos'])
            np.save(path + 'name_' + str(s.haloname), s.haloname)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 10))
        axis.set_aspect('equal')
        axis.set_facecolor('k')
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        plt.rcParams['savefig.facecolor'] = 'black'
        
        # Load the data #
        pos = np.load(path + 'pos_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        band = np.load(path + 'band_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the r-band 2D histogram #
        plt.hist2d(pos[:, 2], pos[:, 1], weights=band, bins=300, norm=matplotlib.colors.LogNorm(), cmap='gray', range=[[-0.03, 0.03], [-0.03, 0.03]])
        
        # Save and close the figure #
        plt.savefig('/u/di43/Auriga/plots/rbm/' + 'Au-' + str(re.split('_|.npy', names[i])[1]), bbox_inches='tight')
        plt.close()
    return None


def gas_density(pdf, data, redshift, read):
    """
    Plot gas density projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    path = '/u/di43/Auriga/plots/data/' + 'gd/' + str(redshift) + '/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['mass', 'pos', 'rho']
        data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
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
            face_on = s.get_Aslice('rho', res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)['grid'] * boxsize * 1e3
            edge_on = s.get_Aslice('rho', res=res, axes=[1, 0], box=[boxsize, 0.5 * boxsize], proj=True, numthreads=8)['grid'] * boxsize * 1e3
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'face_on_' + str(s.haloname), face_on)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(7.5, 10))
        axis00, axis10, axiscbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        plot_tools.set_axis(axis00, xlim=[-30, 30], ylim=[-30, 30], ylabel=r'$\mathrm{y/kpc}$')
        plot_tools.set_axis(axis10, xlim=[-30, 30], ylim=[-15, 15], xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{z/kpc}$')
        figure.text(0.0, 1.01, r'$\mathrm{Au-%s\;z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), color='k', fontsize=16,
                    transform=axis00.transAxes)
        
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the gas density projections #
        pcm = axis00.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma', rasterized=True)
        axis10.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma', rasterized=True)
        plot_tools.create_colorbar(axiscbar, pcm, '$\mathrm{\Sigma_{gas}/(M_\odot\;kpc^{-2})}$')
        
        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def gas_temperature(pdf, data, redshift, read):
    """
    Plot gas temperature projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    path = '/u/di43/Auriga/plots/data/' + 'gt/' + str(redshift) + '/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['mass', 'ne', 'pos', 'rho', 'u']
        data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
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
            
            # Get the density-weighted gas temperature projections #
            mean_weight = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * s.data['ne']) * 1.67262178e-24
            temperature = (5.0 / 3.0 - 1.0) * s.data['u'] / const.KB * (1e6 * const.parsec) ** 2.0 / (1e6 * const.parsec / 1e5) ** 2 * mean_weight
            s.data['temprho'] = s.rho * temperature
            
            face_on = s.get_Aslice('temprho', res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)['grid']
            face_on_rho = s.get_Aslice('rho', res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)['grid']
            edge_on = s.get_Aslice('temprho', res=res, axes=[1, 0], box=[boxsize, 0.5 * boxsize], proj=True, numthreads=8)['grid']
            edge_on_rho = s.get_Aslice('rho', res=res, axes=[1, 0], box=[boxsize, 0.5 * boxsize], proj=True, numthreads=8)['grid']
            
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
        figure = plt.figure(figsize=(7.5, 10))
        axis00, axis10, axiscbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        plot_tools.set_axis(axis00, xlim=[-30, 30], ylim=[-30, 30], ylabel=r'$\mathrm{y/kpc}$')
        plot_tools.set_axis(axis10, xlim=[-30, 30], ylim=[-15, 15], xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{z/kpc}$')
        figure.text(0.0, 1.01, r'$\mathrm{Au-%s\;z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), color='k', fontsize=16,
                    transform=axis00.transAxes)
        
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        face_on_rho = np.load(path + 'face_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on_rho = np.load(path + 'edge_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the density-weighted gas temperature projections #
        pcm = axis00.pcolormesh(x, y, (face_on / face_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap='viridis', rasterized=True)
        axis10.pcolormesh(x, 0.5 * y, (edge_on / edge_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap='viridis', rasterized=True)
        plot_tools.create_colorbar(axiscbar, pcm, r'$\mathrm{T/K}$')
        
        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def gas_metallicity(pdf, data, redshift, read):
    """
    Plot gas metallicity projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    path = '/u/di43/Auriga/plots/data/' + 'gm/' + str(redshift) + '/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['mass', 'pos', 'gz']
        data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
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
            
            # Get the gas metallicity projections #
            face_on = s.get_Aslice('gz', res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)['grid'] / res / 0.0134
            edge_on = s.get_Aslice('gz', res=res, axes=[1, 0], box=[boxsize, 0.5 * boxsize], proj=True, numthreads=8)['grid'] / res / 0.0134
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'face_on_' + str(s.haloname), face_on)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(7.5, 10))
        axis00, axis10, axiscbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        plot_tools.set_axis(axis00, xlim=[-30, 30], ylim=[-30, 30], ylabel=r'$\mathrm{y/kpc}$')
        plot_tools.set_axis(axis10, xlim=[-30, 30], ylim=[-15, 15], xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{z/kpc}$')
        figure.text(0.0, 1.01, r'$\mathrm{Au-%s\;z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), color='k', fontsize=16,
                    transform=axis00.transAxes)
        
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the gas metallicity projections #
        pcm = axis00.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), cmap='viridis', rasterized=True)
        axis10.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), cmap='viridis', rasterized=True)
        plot_tools.create_colorbar(axiscbar, pcm, r'$\mathrm{Z/Z_\odot}$')
        
        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def gas_slice(pdf, data, redshift, read):
    """
    Plot gas temperature projection for different temperature regimes along with velocity arrows for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    boxsize = 0.08  # Increase the boxsize.
    path = '/u/di43/Auriga/plots/data/' + 'gs/' + str(redshift) + '/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['pos', 'vel', 'mass', 'u', 'ne', 'gz', 'gmet', 'rho', 'vol']
        data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
        # Loop over all available haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            # if str(s.haloname) in names:
            #     continue
            
            # Select the halo and rotate it based on Euler's angles #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, remove_bulk_vel=True, use_principal_axis=False, euler_rotation=True, rotate_disk=True, do_rotation=True)
            
            # Loop over the two projections #
            gas_mask, = np.where(s.data['type'] == 0)  # Mask the data: select gas particles.
            ngas = np.size(gas_mask)
            indy = [[1, 2, 0], [1, 0, 2]]  # Swap the position and velocity indices for the face-on and edge-on projections #
            for j in range(2):
                # Declare lists/arrays to store the data #
                frbs = []
                temp_pos = s.data['pos'][gas_mask, :]
                temp_vel = s.data['vel'][gas_mask, :]
                rgbArray = np.zeros((res, res, 3), 'uint8')
                pos, vel = np.zeros((np.size(gas_mask), 3)), np.zeros((np.size(gas_mask), 3))
                u, rho, z, vol = np.zeros(ngas), np.zeros(ngas), np.zeros(ngas), np.zeros(ngas)
                for i in range(3):
                    pos[:, i] = temp_pos[:, indy[j][i]]
                    vel[:, i] = temp_vel[:, indy[j][i]]
                
                # Calculate the temperature of the gas cells #
                vol[:] = s.data['vol'][gas_mask] * 1e9
                XH = s.data['gmet'][gas_mask, element['H']]
                ne, mass, rho[:], metallicity = s.data['ne'][gas_mask], s.data['mass'][gas_mask], s.data['rho'][gas_mask], s.data['gz'][gas_mask]
                yhelium = (1 - XH - metallicity) / (4. * XH)
                mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
                temperature = GAMMA_MINUS1 * s.data['u'][gas_mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN
                
                # Define the three temperatures regimes #
                sfgas = np.where(temperature < 2e4)
                medgas = np.where((temperature >= 1e4) & (temperature < 6e5))
                hotgas = np.where(temperature >= 5e5)
                
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
                    frbs.append(np.array(A['grid']))
                
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
        figure = plt.figure(figsize=(10, 15))
        axis00, axis10, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, velocity_vectors=True)
        figure.text(0.0, 1.01, r'$\mathrm{Au-%s\;z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), color='k', fontsize=16,
                    transform=axis00.transAxes)
        for axis in [axis00, axis10]:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xticklabels([])
            axis.set_yticklabels([])
        
        j = 0
        for axis in [axis00, axis10]:
            # Load the data #
            xc = np.load(path + 'xc_' + str(j) + '_' + str(re.split('_|.npy', str(names[i]))[1]) + '.npy')
            yc = np.load(path + 'yc_' + str(j) + '_' + str(re.split('_|.npy', str(names[i]))[1]) + '.npy')
            vxgrid = np.load(path + 'vxgrid_' + str(j) + '_' + str(re.split('_|.npy', str(names[i]))[1]) + '.npy')
            vygrid = np.load(path + 'vygrid_' + str(j) + '_' + str(re.split('_|.npy', str(names[i]))[1]) + '.npy')
            rgbArray = np.load(path + 'rgbArray_' + str(j) + '_' + str(re.split('_|.npy', str(names[i]))[1]) + '.npy')
            
            # Plot the gas slice and the velocity arrows #
            axis.imshow(rgbArray, rasterized=True, aspect='equal')
            axis.quiver(xc, yc, np.rot90(vxgrid), np.rot90(vygrid), scale=4000.0, pivot='middle', color='yellow', alpha=0.8)
            j += 1
        
        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def gas_temperature_edge_on(pdf, data, redshift, read):
    """
    Plot gas temperature projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    boxsize = 0.2  # Increase the boxsize.
    path = '/u/di43/Auriga/plots/data/' + 'gteo/' + str(redshift) + '/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['mass', 'ne', 'pos', 'rho', 'u']
        data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
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
            
            # Get the density-weighted gas temperature projections #
            mean_weight = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * s.data['ne']) * 1.67262178e-24
            temperature = (5.0 / 3.0 - 1.0) * s.data['u'] / const.KB * (1e6 * const.parsec) ** 2.0 / (1e6 * const.parsec / 1e5) ** 2 * mean_weight
            s.data['temprho'] = s.rho * temperature
            edge_on = s.get_Aslice('temprho', res=res, axes=[1, 0], box=[boxsize, boxsize], boxz=1e-3, proj=True, numthreads=8)['grid']
            edge_on_rho = s.get_Aslice('rho', res=res, axes=[1, 0], box=[boxsize, boxsize], boxz=1e-3, proj=True, numthreads=8)['grid']
            
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
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(1, 2, wspace=0.05, width_ratios=[1, 0.05])
        axis00 = plt.subplot(gs[0, 0])
        axiscbar = plt.subplot(gs[:, 1])
        x = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
        z = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
        plot_tools.set_axis(axis00, xlim=[-100, 100], ylim=[-100, 100], xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{z/kpc}$')
        figure.text(0.0, 1.01, r'$\mathrm{Au-%s\;z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), color='k', fontsize=16,
                    transform=axis00.transAxes)
        
        # Load the data #
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on_rho = np.load(path + 'edge_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the density-weighted gas temperature projections #
        pcm = axis00.pcolormesh(x * 1e3, z * 1e3, (edge_on / edge_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=2e7), cmap='Spectral_r',
                                rasterized=True)
        plot_tools.create_colorbar(axiscbar, pcm, r'$\mathrm{T/K}$')
        
        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def magnetic_field(pdf, data, redshift, read):
    """
    Plot magnetic field strength projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    path = '/u/di43/Auriga/plots/data/' + 'mf/' + str(redshift) + '/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['mass', 'pos', 'bfld']
        data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
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
            
            # Get the magnetic field projections #
            s.data['b2'] = (s.data['bfld'] ** 2.).sum(axis=1)
            face_on = np.sqrt(
                s.get_Aslice('b2', res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)['grid'] / res) * const.bfac * 1e6
            edge_on = np.sqrt(
                s.get_Aslice('b2', res=res, axes=[1, 0], box=[boxsize, 0.5 * boxsize], proj=True, numthreads=8)['grid'] / res) * const.bfac * 1e6
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'face_on_' + str(s.haloname), face_on)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(7.5, 10))
        axis00, axis10, axiscbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        plot_tools.set_axis(axis00, xlim=[-30, 30], ylim=[-30, 30], ylabel=r'$\mathrm{y/kpc}$')
        plot_tools.set_axis(axis10, xlim=[-30, 30], ylim=[-15, 15], xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{z/kpc}$')
        figure.text(0.0, 1.01, r'$\mathrm{Au-%s\;z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), color='k', fontsize=16,
                    transform=axis00.transAxes)
        
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the magnetic field projections #
        pcm = axis00.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=1e2), cmap='CMRmap', rasterized=True)
        axis10.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=1e2), cmap='CMRmap', rasterized=True)
        plot_tools.create_colorbar(axiscbar, pcm, '$\mathrm{B/\mu G}$')
        
        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def dark_matter_density(pdf, data, redshift, read):
    """
    Plot dark matter projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    boxsize = 0.4  # Increase the boxsize.
    path = '/u/di43/Auriga/plots/data/' + 'dmd/' + str(redshift) + '/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [1, 4]
        attributes = ['mass', 'pos']
        data.select_haloes(level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
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
            
            axis00, axis10, axiscbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)  # Generate the axes.
            
            # Get the dark matter density projections #
            dark_matter_mask, = np.where(
                (s.data['type'] == 1) & (s.r() < 2.0 * boxsize))  # Mask the data: select dark matter particles inside a 0.8Mpc sphere.
            face_on = get_projection(s.data['pos'][dark_matter_mask].astype('f8'), s.data['mass'][dark_matter_mask].astype('f8'),
                                     s.data['mass'][dark_matter_mask].astype('f8'), 0, res, boxsize, 'mass') / area * 1e10
            edge_on = get_projection(s.data['pos'][dark_matter_mask].astype('f8'), s.data['mass'][dark_matter_mask].astype('f8'),
                                     s.data['mass'][dark_matter_mask].astype('f8'), 1, res, boxsize, 'mass') / (0.5 * area) * 1e10
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'face_on_' + str(s.haloname), face_on)
            np.save(path + 'edge_on_' + str(s.haloname), edge_on)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(7.5, 10))
        axis00, axis10, axiscbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        plot_tools.set_axis(axis00, xlim=[-200, 200], ylim=[-200, 200], ylabel=r'$\mathrm{y/kpc}$')
        plot_tools.set_axis(axis10, xlim=[-200, 200], ylim=[-100, 100], xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{z/kpc}$')
        figure.text(0.0, 1.01, r'$\mathrm{Au-%s\;z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), color='k', fontsize=16,
                    transform=axis00.transAxes)
        
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot the dark matter density projection #
        pcm = axis00.pcolormesh(x, y, face_on, norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e9), cmap='Greys', rasterized=True)
        axis10.pcolormesh(x, y2, edge_on, norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e9), cmap='Greys', rasterized=True)
        plot_tools.create_colorbar(axiscbar, pcm, '$\mathrm{\Sigma_{DM}/(M_\odot\;kpc^{-2})}$')
        
        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None
