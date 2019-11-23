from __future__ import print_function, division

import pysph
import calcGrid
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from matplotlib import gridspec

res = 512
boxsize = 0.05


def create_axes(res=res, boxsize=boxsize, colorbar=False):
    """
    Generate plot axes.
    :param res: resolution
    :param boxsize: boxsize
    :param colorbar: colorbar
    :return: axtop, axbot, axcbar, x, y, y2, area
    """
    
    # Set the axis values #
    x = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y2 = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res / 2 + 1)
    
    area = (boxsize / res) ** 2  # Calculate the area.
    
    # Generate the two panels #
    if colorbar is True:
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 0.05])
        gs.update(hspace=0.05, wspace=0.05)
        axtop = plt.subplot(gs[0, 0])
        axbot = plt.subplot(gs[1, 0])
        axcbar = plt.subplot(gs[:, 1])
        
        return axtop, axbot, axcbar, x, y, y2, area
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        gs.update(hspace=0.05)
        axtop = plt.subplot(gs[0, 0])
        axbot = plt.subplot(gs[1, 0])
        
        return axtop, axbot, x, y, y2, area


def create_colorbar(axcbar, pcm, label):
    """
    Generate a colorbar.
    :param axcbar: colorbar axis from create_axes
    :param pcm: pseudocolor plot
    :param label: colorbar label
    :return: None
    """
    # Set the colorbar axes #
    cb = plt.colorbar(pcm, cax=axcbar)
    
    # Set the colorbar parameters #
    cb.set_label(label, size=16)
    cb.ax.tick_params(labelsize=16)
    
    return None


def set_axes(axtop, axbot, xlabel=None, ylabel=None, y2label=None, ticks=False):
    """
    Set axes' parameters.
    :param axtop: top plot axes from create_axes
    :param axbot: bottom plot axes from create_axes
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param y2label: y2-axis label
    :param ticks: y2-axis label
    :return: None
    """
    # Set x-axis labels and ticks #
    if xlabel is None:
        axtop.set_xticks([])
        axbot.set_xticks([])
        axtop.set_xticklabels([])
        axbot.set_xticklabels([])
    else:
        axtop.set_xticklabels([])
        axbot.set_xlabel(xlabel, size=16)
    
    # Set y-axis labels and ticks #
    if ylabel is None:
        axtop.set_yticks([])
        axtop.set_yticklabels([])
    else:
        axtop.set_ylabel(ylabel, size=16)
    
    # Set y2-axis labels and ticks #
    if y2label is None:
        axbot.set_yticks([])
        axbot.set_yticklabels([])
    else:
        axbot.set_ylabel(y2label, size=16)
    
    # Set x- and y-axis ticks size #
    if ticks is True:
        for a in [axtop, axbot]:
            a.axis('tight')
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
    
    pos = np.zeros((np.size(mass), 3))  # Define array to hold the new positions of particles.
    
    # Generate projection planes #
    if idir == 0:  # XY plane
        pos[:, 0] = pos_orig[:, 2]
        pos[:, 1] = pos_orig[:, 1]
        pos[:, 2] = pos_orig[:, 0]
        
        xres = res
        yres = res
        boxx = boxsize
        boxy = boxsize
    
    elif idir == 1:  # ZY plane
        pos[:, 0] = pos_orig[:, 0]
        pos[:, 1] = pos_orig[:, 2]
        pos[:, 2] = pos_orig[:, 1]
        
        xres = res // 2
        yres = res
        boxx = boxsize / 2.
        boxy = boxsize
    
    elif idir == 2:  # YZ plane
        pos[:, 0] = pos_orig[:, 1]
        pos[:, 1] = pos_orig[:, 0]
        pos[:, 2] = pos_orig[:, 2]
        
        xres = res
        yres = res // 2
        boxx = boxsize
        boxy = boxsize / 2.
    
    tree = pysph.makeTree(pos)
    hsml = tree.calcHsmlMulti(pos, pos, mass, 48, numthreads=8)
    if maxHsml is True:
        hsml = np.minimum(hsml, 4. * boxsize / res)
    hsml = np.maximum(hsml, 1.001 * boxsize / res * 0.5)
    rho = np.ones(np.size(mass))
    
    datarange = np.array([[4003.36, 800672.], [199.370, 132913.], [133.698, 200548.]])
    fac = (512. / res) ** 2 * (0.5 * boxsize / 0.025) ** 2
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


def stellar_light(pdf, data, level, redshift):
    """
    Plot stellar light projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param level: level from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    # Read desired galactic property(ies) for specific particle type(s) for Auriga haloes #
    particle_type = [4]
    attributes = ['age', 'gsph', 'mass', 'pos']
    data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
    
    # Loop over all haloes #
    for s in data:
        # Generate the figure #
        plt.close()
        f = plt.figure(figsize=(8, 8), dpi=300)
        axtop, axbot, x, y, y2, area = create_axes(res=res, boxsize=boxsize)
        axtop.text(0.0, 1.01, 'Au' + str(s.haloname) + ' redshift = ' + str(redshift), color='k', fontsize=16, transform=axtop.transAxes)
        
        # Rotate halo based on principal axes #
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        # Mask the data and plot the projections #
        mask, = np.where((s.data['age'] > 0.0) & (s.r() * 1e3 < 30) & (abs(s.pos[:, 0] * 1e3) < 5.0))  # Distances are in Mpc.
        
        z_rotated, y_rotated, x_rotated = rotate_bar(s.pos[mask, 0] * 1e3, s.pos[mask, 1] * 1e3, s.pos[mask, 2] * 1e3)  # Distances are in Mpc.
        s.pos = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.pos attribute in kpc.
        
        face_on = get_projection(s.pos.astype('f8'), s.mass[mask].astype('f8'), s.data['gsph'][mask].astype('f8'), 0, res, boxsize, 'light',
                                 maxHsml=True)
        edge_on = get_projection(s.pos.astype('f8'), s.mass[mask].astype('f8'), s.data['gsph'][mask].astype('f8'), 1, res, boxsize, 'light',
                                 maxHsml=True)
        axtop.imshow(face_on, interpolation='nearest')
        axbot.imshow(edge_on, interpolation='nearest')
        
        set_axes(axtop, axbot)
        pdf.savefig(f)  # Save figure.
    
    return None


def stellar_density(pdf, data, level, redshift):
    """
    Plot stellar density projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param level: level from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    # Read desired galactic property(ies) for specific particle type(s) for Auriga haloes #
    particle_type = [4]
    attributes = ['mass', 'pos']
    data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
    
    # Loop over all haloes #
    for s in data:
        # Generate the figure #
        plt.close()
        f = plt.figure(figsize=(8, 8), dpi=300)
        axtop, axbot, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        axtop.text(0.0, 1.01, 'Au' + str(s.haloname) + ' redshift = ' + str(redshift), color='k', fontsize=16, transform=axtop.transAxes)
        
        # Rotate halo based on principal axes #
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        # Mask the data and plot the projections #
        mask, = np.where((s.r() < 2. * boxsize))
        z_rotated, y_rotated, x_rotated = rotate_bar(s.pos[mask, 0] * 1e3, s.pos[mask, 1] * 1e3, s.pos[mask, 2] * 1e3)  # Distances are in Mpc.
        s.pos = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.pos attribute in kpc.
        
        face_on = get_projection(s.pos.astype('f8'), s.mass[mask].astype('f8'), s.data['mass'][mask].astype('f8'), 0, res, boxsize,
                                 'mass') / area * 1e10
        edge_on = get_projection(s.pos.astype('f8'), s.mass[mask].astype('f8'), s.data['mass'][mask].astype('f8'), 1, res, boxsize, 'mass') / (
            0.5 * area) * 1e10
        pcm = axtop.pcolormesh(x, y, face_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='Oranges', rasterized=True)
        axbot.pcolormesh(x, y2, edge_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='Oranges', rasterized=True)
        create_colorbar(axcbar, pcm, "$\Sigma_\mathrm{stars}\,\mathrm{[M_\odot\,kpc^{-2}]}$")
        
        # count, xedges, yedges = np.histogram2d(s.pos[:, 2] * 1e3, s.pos[:, 1] * 1e3, bins=70, range=[[-25, 25], [-25, 25]])
        # countlog = np.log10(count)
        # axtop.contour(countlog.T, colors="k", linewidth=2, linestyle="-", extent=[-25, 25, -25, 25], levels=np.arange(0.0, 5.0 + 0.5, 0.25))
        
        set_axes(axtop, axbot, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$', ticks=True)
        
        pdf.savefig(f, bbox_inches='tight')  # Save figure.
    
    return None


def gas_density(pdf, data, level, redshift):
    """
    Plot gas density projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param level: level from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    
    # Read desired galactic property(ies) for specific particle type(s) for Auriga haloes #
    particle_type = [0, 4]
    attributes = ['mass', 'pos', 'rho']
    data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
    
    # Loop over all haloes #
    for s in data:
        # Generate the figure #
        plt.close()
        f = plt.figure(figsize=(8, 8), dpi=300)
        axtop, axbot, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        axtop.text(0.0, 1.01, 'Au' + str(s.haloname) + ' redshift = ' + str(redshift), color='k', fontsize=16, transform=axtop.transAxes)
        
        # Rotate halo based on principal axes #
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        # Plot the projections #
        face_on = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] * boxsize * 1e3
        edge_on = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"] * boxsize * 1e3
        pcm = axtop.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma', rasterized=True)
        axbot.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma', rasterized=True)
        create_colorbar(axcbar, pcm, "$\Sigma_\mathrm{gas}\,\mathrm{[M_\odot\,kpc^{-2}]}$")
        
        set_axes(axtop, axbot, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$', ticks=True)
        
        pdf.savefig(f, bbox_inches='tight')  # Save figure.
    return None


def gas_temperature(pdf, data, level, redshift):
    """
    Plot gas temperature projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param level: level from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    # Read desired galactic property(ies) for specific particle type(s) for Auriga haloes #
    particle_type = [0, 4]
    attributes = ['mass', 'ne', 'pos', 'rho', 'u']
    data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
    
    # Loop over all haloes #
    for s in data:
        # Generate the figure #
        plt.close()
        f = plt.figure(figsize=(8, 8), dpi=300)
        axtop, axbot, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        axtop.text(0.0, 1.01, 'Au' + str(s.haloname) + ' redshift = ' + str(redshift), color='k', fontsize=16, transform=axtop.transAxes)
        
        # Rotate halo based on principal axes #
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        # Plot the projections #
        meanweight = 4.0 / (1. + 3. * 0.76 + 4. * 0.76 * s.data['ne']) * 1.67262178e-24;
        temp = (5. / 3. - 1.) * s.data['u'] / KB * (1e6 * parsec) ** 2. / (1e6 * parsec / 1e5) ** 2 * meanweight;
        s.data['temprho'] = s.rho * temp
        
        face_on = s.get_Aslice("temprho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]
        rho = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]
        pcm = axtop.pcolormesh(x, y, (face_on / rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap='viridis', rasterized=True)
        edge_on = s.get_Aslice("temprho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"]
        rho = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"]
        axbot.pcolormesh(x, 0.5 * y, (edge_on / rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap='viridis', rasterized=True)
        create_colorbar(axcbar, pcm, "$T\,\mathrm{[K]}$")
        
        set_axes(axtop, axbot, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$', ticks=True)
        
        pdf.savefig(f, bbox_inches='tight')  # Save figure.
    
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
    # Read desired galactic property(ies) for specific particle type(s) for Auriga haloes #
    particle_type = [0, 4]
    attributes = ['mass', 'pos', 'gz']
    data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
    
    # Loop over all haloes #
    for s in data:
        # Generate the figure #
        plt.close()
        f = plt.figure(figsize=(8, 8), dpi=300)
        axtop, axbot, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        axtop.text(0.0, 1.01, 'Au' + str(s.haloname) + ' redshift = ' + str(redshift), color='k', fontsize=16, transform=axtop.transAxes)
        
        # Rotate halo based on principal axes #
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        # Plot the projections #
        face_on = s.get_Aslice("gz", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res / 0.0134
        edge_on = s.get_Aslice("gz", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"] / res / 0.0134
        pcm = axtop.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), cmap='viridis', rasterized=True)
        axbot.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), cmap='viridis', rasterized=True)
        create_colorbar(axcbar, pcm, "$Z/Z_\odot$")
        
        set_axes(axtop, axbot, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$', ticks=True)
        
        pdf.savefig(f, bbox_inches='tight')  # Save figure.
    
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
    # Read desired galactic property(ies) for specific particle type(s) for Auriga haloes #
    particle_type = [0, 4]
    attributes = ['mass', 'pos', 'bfld']
    data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
    
    # Loop over all haloes #
    for s in data:
        # Generate the figure #
        plt.close()
        f = plt.figure(figsize=(8, 8), dpi=300)
        axtop, axbot, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        axtop.text(0.0, 1.01, 'Au' + str(s.haloname) + ' redshift = ' + str(redshift), color='k', fontsize=16, transform=axtop.transAxes)
        
        # Rotate halo based on principal axes #
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        # Plot the projections #
        s.data['b2'] = (s.data['bfld'] ** 2.).sum(axis=1)
        face_on = np.sqrt(s.get_Aslice("b2", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res) * bfac * 1e6
        edge_on = np.sqrt(s.get_Aslice("b2", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res) * bfac * 1e6
        pcm = axtop.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=5e1), cmap='CMRmap', rasterized=True)
        axbot.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=5e1), cmap='CMRmap', rasterized=True)
        create_colorbar(axcbar, pcm, "$B\,\mathrm{[\mu G]}$")
        
        set_axes(axtop, axbot, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$', ticks=True)
        
        pdf.savefig(f, bbox_inches='tight')  # Save figure.
    
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
    boxsize = 0.4  # Increase the boxsize
    
    # Read desired galactic property(ies) for specific particle type(s) for Auriga haloes #
    particle_type = [1, 4]
    attributes = ['mass', 'pos']
    data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
    
    # Loop over all haloes #
    for s in data:
        # Generate the figure #
        plt.close()
        f = plt.figure(figsize=(8, 8), dpi=300)
        axtop, axbot, axcbar, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3, colorbar=True)
        axtop.text(0.0, 1.01, 'Au' + str(s.haloname) + ' redshift = ' + str(redshift), color='k', fontsize=16, transform=axtop.transAxes)
        
        # Rotate halo based on principal axes #
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        # Mask the data and plot the projections #
        mask, = np.where((s.r() < 2. * boxsize) & (s.type == 1))
        face_on = get_projection(s.pos[mask].astype('f8'), s.mass[mask].astype('f8'), s.data['mass'][mask].astype('f8'), 0, res, boxsize,
                                 'mass') / area * 1e10
        edge_on = get_projection(s.pos[mask].astype('f8'), s.mass[mask].astype('f8'), s.data['mass'][mask].astype('f8'), 1, res, boxsize, 'mass') / (
                0.5 * area) * 1e10
        pcm = axtop.pcolormesh(x, y, face_on, norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e9), cmap='Greys', rasterized=True)
        axbot.pcolormesh(x, y2, edge_on, norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e9), cmap='Greys', rasterized=True)
        create_colorbar(axcbar, pcm, "$\Sigma_\mathrm{DM}\,\mathrm{[M_\odot\,kpc^{-2}]}$")
        
        set_axes(axtop, axbot, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$', ticks=True)
        
        pdf.savefig(f, bbox_inches='tight')  # Save figure.
    
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