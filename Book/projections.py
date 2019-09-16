from __future__ import print_function, division

import calcGrid
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pysph
from const import *
from matplotlib import gridspec
from sfigure import *

res = 512
boxsize = 0.05


def create_axes(res=res, boxsize=boxsize, ):
    """
    Method to create plot axes.
    :param res: resolution
    :param boxsize: boxsize
    :return: ax, ax2, x, y, y2, area
    """

    # Set the axis values #
    x = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y2 = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res / 2 + 1)

    area = (boxsize / res) ** 2  # Calculate the area.

    # Create the two panels #
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=(20, 1))
    gs.update(hspace=0.1)
    ax = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])

    return ax, ax2, x, y, y2, area


def set_axes(ax, ax2, xlabel=None, ylabel=None, y2label=None):
    """
    Method to set axes' parameters.
    :param ax: see create_axes
    :param ax2: see create_axes
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param y2label: y2-axis label
    :return: None
    """
    # Set x-axis labels and ticks #
    if xlabel is None:
        ax.set_xticks([])
        ax2.set_xticks([])
    else:
        ax.set_xticklabels([])
        ax2.set_xlabel(xlabel, size=16)

    # Set y-axis labels and ticks #
    if ylabel is None:
        ax.set_yticks([])
    else:
        ax.set_ylabel(ylabel, size=16)

    # Set y2-axis labels and ticks #
    if y2label is None:
        ax2.set_yticks([])
    else:
        ax2.set_ylabel(y2label, size=16)

    # Set x- and y-axis ticks' size #
    for a in [ax, ax2]:
        a.axis('tight')
        for label in a.xaxis.get_ticklabels():
            label.set_size(16)
        for label in a.yaxis.get_ticklabels():
            label.set_size(16)

    return None


def create_colorbar(f, pc, label):
    """
    Method to create a colorbar.
    :param f: figure
    :param pc: pseudocolor plot
    :param label: colorbar label
    :return: None
    """
    # Set the colorbar axes #
    cax = f.add_axes([0.8, 0.1, 0.03, 0.8])
    cb = plt.colorbar(pc, cax=cax)

    # Set the colorbar parameters #
    cb.set_label(label, size=16)
    cb.ax.tick_params(labelsize=16)

    return None


def get_projection(pos_orig, mass, data, idir, res, boxsize, type, maxHsml=True):
    """
    Method to calculate particle projections.
    :param pos_orig: positions of particles
    :param mass: masses of particles
    :param data: projection of light or mass data
    :param idir: direction
    :param res: resolution
    :param boxsize: boxsize
    :param type: light or mass
    :param maxHsml:
    :return:
    """

    pos = np.zeros((np.size(mass), 3))  # Define array to hold the new positions of particles.

    # Create projection planes #
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
    if maxHsml:
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


def stellar_light(pdf, data, levels, z):
    """
    Method to plot stellar light.
    :param pdf: path to save the pdf (see book.make_book_overview)
    :param data: data (see book.make_book_overview)
    :param levels: level (see book.make_book_overview)
    :param z: redshift (see book.make_book_overview)
    :return: None
    """

    # Count the number of haloes #
    nlevels = len(levels)
    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    # Loop over all levels #
    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[4], loadonlyhalo=0, loadonly=['pos', 'vel', 'mass', 'age', 'gsph'])  # Read galactic properties.

        # Loop over all haloes #
        for s in data:
            # Create the figure #
            plt.close()
            f = plt.figure(figsize=(8, 8), dpi=100)
            ax, ax2, x, y, y2, area = create_axes(res=res, boxsize=boxsize)
            set_axes(ax, ax2, xlabel='$x$', ylabel='$y$', y2label='$z$')
            ax.text(0.0, 1.0, 'Au' + str(s.haloname) + ' level' + str(level) + ' z = ' + str(z), color='k', fontsize=16, transform=ax.transAxes)

            s.calc_sf_indizes(s.subfind)

            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)  # Read halo properties.

            # Mask the data and plot the projections #
            mask, = np.where((s.r() < 2. * boxsize) & (s.data['age'] > 0.))

            face_on = get_projection(s.pos[mask, :].astype('f8'), s.mass[mask].astype('f8'), s.data['gsph'][mask, :].astype('f8'), 0, res, boxsize,
                                     'light')

            edge_on = get_projection(s.pos[mask, :].astype('f8'), s.mass[mask].astype('f8'), s.data['gsph'][mask, :].astype('f8'), 1, res, boxsize,
                                     'light')
            ax.imshow(face_on, interpolation='nearest')
            ax2.imshow(edge_on, interpolation='nearest')

            pdf.savefig(f)  # Save figure.

    return None


def stellar_mass(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[4])

        for s in data:
            plt.close()
            f = plt.figure(figsize=(8, 8), dpi=100)

            s.calc_sf_indizes(s.subfind)

            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3)

            ism, = np.where((s.r() < 2. * boxsize))

            proj = get_projection(s.pos[ism, :].astype('f8'), s.mass[ism].astype('f8'), s.data['mass'][ism].astype('f8'), 0, res, boxsize, 'mass',
                                  maxHsml=False) / area * 1e10

            pc = ax.pcolormesh(x, y, proj, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap=matplotlib.cm.Oranges, rasterized=True)

            create_colorbar(f, pc, "$\Sigma_\mathrm{stars}\,\mathrm{[M_\odot\,kpc^{-2}]}$")

            proj = get_projection(s.pos[ism, :].astype('f8'), s.mass[ism].astype('f8'), s.data['mass'][ism].astype('f8'), 1, res, boxsize, 'mass',
                                  maxHsml=False) / (0.5 * area) * 1e10

            ax2.pcolormesh(x, y2, proj, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap=matplotlib.cm.Oranges, rasterized=True)

            ax.text(0.0, 1.0, 'Au' + str(s.haloname) + ' level' + str(level) + ' z = ' + str(z), color='k', fontsize=16, transform=ax.transAxes)

            set_axes(ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            pdf.savefig(f)

    return None


def gas_density(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[0, 4], loadonly=['pos', 'vol', 'rho', 'mass', 'vel'])

        for s in data:
            plt.close()
            f = plt.figure(figsize=(8, 8), dpi=100)

            s.calc_sf_indizes(s.subfind)

            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3)

            proj = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] * boxsize * 1e3

            pc = ax.pcolormesh(x, y, proj.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap=matplotlib.cm.magma, rasterized=True)

            create_colorbar(f, pc, "$\Sigma_\mathrm{gas}\,\mathrm{[M_\odot\,kpc^{-2}]}$")

            proj = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"] * boxsize * 1e3

            ax2.pcolormesh(x, 0.5 * y, proj.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap=matplotlib.cm.magma, rasterized=True)

            ax.text(0.0, 1.0, 'Au' + str(s.haloname) + ' level' + str(level) + ' z = ' + str(z), color='k', fontsize=16, transform=ax.transAxes)

            set_axes(ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            pdf.savefig(f)
    return None


def gas_temperature(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[0, 4], loadonly=['pos', 'vol', 'rho', 'mass', 'vel', 'ne', 'u'])

        for s in data:
            plt.close()
            f = plt.figure(figsize=(8, 8), dpi=100)

            s.calc_sf_indizes(s.subfind)

            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3)

            meanweight = 4.0 / (1. + 3. * 0.76 + 4. * 0.76 * s.data['ne']) * 1.67262178e-24;

            temp = (5. / 3. - 1.) * s.data['u'] / KB * (1e6 * parsec) ** 2. / (1e6 * parsec / 1e5) ** 2 * meanweight;

            s.data['temprho'] = s.rho * temp

            proj = s.get_Aslice("temprho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]

            rho = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]

            pc = ax.pcolormesh(x, y, (proj / rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap=matplotlib.cm.viridis, rasterized=True)

            create_colorbar(f, pc, "$T\,\mathrm{[K]}$")

            proj = s.get_Aslice("temprho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"]

            rho = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"]

            ax2.pcolormesh(x, 0.5 * y, (proj / rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap=matplotlib.cm.viridis,
                           rasterized=True)

            ax.text(0.0, 1.0, 'Au' + str(s.haloname) + ' level' + str(level) + ' z = ' + str(z), color='k', fontsize=16, transform=ax.transAxes)

            set_axes(ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            pdf.savefig(f)

    return None


def gas_metallicity(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[0, 4], loadonly=['pos', 'vol', 'rho', 'mass', 'vel', 'gz'])

        for s in data:
            plt.close()
            f = plt.figure(figsize=(8, 8), dpi=100)

            s.calc_sf_indizes(s.subfind)

            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3)

            proj = s.get_Aslice("gz", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res / 0.0134

            pc = ax.pcolormesh(x, y, proj.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), cmap=matplotlib.cm.viridis, rasterized=True)

            create_colorbar(f, pc, "$Z/Z_\odot$")

            proj = s.get_Aslice("gz", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"] / res / 0.0134

            ax2.pcolormesh(x, 0.5 * y, proj.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), cmap=matplotlib.cm.viridis, rasterized=True)

            ax.text(0.0, 1.0, 'Au' + str(s.haloname) + ' level' + str(level) + ' z = ' + str(z), color='k', fontsize=16, transform=ax.transAxes)

            set_axes(ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            pdf.savefig(f)

    return None


def bfld(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[0, 4], loadonly=['pos', 'vol', 'rho', 'mass', 'vel', 'bfld'])

        for s in data:
            plt.close()
            f = plt.figure(figsize=(8, 8), dpi=100)

            s.calc_sf_indizes(s.subfind)

            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3)

            s.data['b2'] = (s.data['bfld'] ** 2.).sum(axis=1)

            proj = np.sqrt(s.get_Aslice("b2", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res) * bfac * 1e6

            pc = ax.pcolormesh(x, y, proj.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=5e1), cmap=matplotlib.cm.CMRmap, rasterized=True)

            create_colorbar(f, pc, "$B\,\mathrm{[\mu G]}$")

            proj = np.sqrt(s.get_Aslice("b2", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res) * bfac * 1e6

            ax2.pcolormesh(x, 0.5 * y, proj.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=5e1), cmap=matplotlib.cm.CMRmap, rasterized=True)

            ax.text(0.0, 1.0, 'Au' + str(s.haloname) + ' level' + str(level) + ' z = ' + str(z), color='k', fontsize=16, transform=ax.transAxes)

            set_axes(ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            pdf.savefig(f)

    return None


def dm_mass(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    boxsize = 0.4

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[1, 4], loadonly=['pos', 'vol', 'rho', 'mass', 'vel'])

        for s in data:
            plt.close()
            f = plt.figure(figsize=(8, 8), dpi=100)

            s.calc_sf_indizes(s.subfind)

            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axes(res=res, boxsize=boxsize * 1e3)

            idm, = np.where((s.r() < 2. * boxsize) & (s.type == 1))

            proj = get_projection(s.pos[idm, :].astype('f8'), s.mass[idm].astype('f8'), s.data['mass'][idm].astype('f8'), 0, res, boxsize, 'mass',
                                  maxHsml=False) / area * 1e10

            pc = ax.pcolormesh(x, y, proj, norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e9), cmap=matplotlib.cm.Greys, rasterized=True)

            create_colorbar(f, pc, "$\Sigma_\mathrm{DM}\,\mathrm{[M_\odot\,kpc^{-2}]}$")

            proj = get_projection(s.pos[idm, :].astype('f8'), s.mass[idm].astype('f8'), s.data['mass'][idm].astype('f8'), 1, res, boxsize, 'mass',
                                  maxHsml=False) / (0.5 * area) * 1e10

            ax2.pcolormesh(x, y2, proj, norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e9), cmap=matplotlib.cm.Greys, rasterized=True)

            ax.text(0.0, 1.0, 'Au' + str(s.haloname) + ' level' + str(level) + ' z = ' + str(z), color='k', fontsize=16, transform=ax.transAxes)

            set_axes(ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            pdf.savefig(f)

    return None