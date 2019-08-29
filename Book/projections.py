from __future__ import print_function, division

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from const import *
from sfigure import *

res = 512
boxsize = 0.05


def create_axis(f, idx, ncol=5, cb=None, res=res, boxsize=boxsize):
    x = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y2 = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res / 2 + 1)
    area = (boxsize / res) ** 2

    ix = idx % ncol
    iy = idx // ncol

    s = 1.3

    if cb is None:
        yoff = 0.2
    else:
        yoff = 0.7

    ax = f.iaxes(0.5 + ix * (s + 0.2), yoff + s + iy * (1.5 * s + 0.6), s, s, top=False)
    ax2 = f.iaxes(0.5 + ix * (s + 0.2), yoff + s + iy * (1.5 * s + 0.6) + 0.5 * s + 0.1, s, s * 0.5, top=False)

    return ax, ax2, x, y, y2, area


def create_colorbar(f, pc, label, ncol=5):
    s = 1.3
    cax = f.iaxes(0.5, 0.3, (s + 0.2) * (ncol - 1) + s, 0.15, top=False)
    plt.colorbar(pc, cax=cax, orientation='horizontal')

    cax.set_xlabel(label, size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_size(6)
    return


def set_axis(idx, ax, ax2, xlabel=None, ylabel=None, y2label=None, ncol=5):
    ix = idx % ncol
    iy = idx // ncol

    if xlabel is None:
        ax.set_xticks([])
        ax2.set_xticks([])
    else:
        ax.set_xticklabels([])
        ax2.set_xlabel(xlabel, size=6)

    for a in [ax, ax2]:
        a.axis('tight')

    if ix == 0:
        if ylabel is None:
            ax.set_yticks([])
        else:
            ax.set_ylabel(ylabel, size=6)

        if y2label is None:
            ax2.set_yticks([])
        else:
            ax2.set_ylabel(y2label, size=6)
    else:
        if xlabel is None:
            ax.set_yticks([])
            ax2.set_yticks([])
        else:
            ax.set_yticklabels([])
            ax2.set_yticklabels([])

    for a in [ax, ax2]:
        for label in a.xaxis.get_ticklabels():
            label.set_size(6)
        for label in a.yaxis.get_ticklabels():
            label.set_size(6)

    return


def stellar_light(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 2.7 * ((nhalos - 1) // 5 + 1) + 0.5))

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[4], loadonlyhalo=0, loadonly=['pos', 'vel', 'mass', 'age', 'gsph'])

        isnap = 0
        for s in data:
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axis(f, isnap * nlevels + il, res=res, boxsize=boxsize)

            istars, = np.where((s.r() < 2. * boxsize) & (s.data['age'] > 0.))

            proj = get_projection(s.pos[istars, :].astype('f8'), s.mass[istars].astype('f8'), s.data['gsph'][istars, :].astype('f8'), 0, res, boxsize,
                                  'light')
            ax.imshow(proj, interpolation='nearest')

            proj = get_projection(s.pos[istars, :].astype('f8'), s.mass[istars].astype('f8'), s.data['gsph'][istars, :].astype('f8'), 1, res, boxsize,
                                  'light')
            ax2.imshow(proj, interpolation='nearest')

            ax.text(0.05, 0.92, "Au%s-%d" % (s.haloname, level), color='w', fontsize=6, transform=ax.transAxes)
            set_axis(isnap * nlevels + il, ax, ax2)

            isnap += 1

    pdf.savefig(f)
    plt.close()
    return


def get_projection(pos_orig, mass, data, idir, res, boxsize, type, maxHsml=True):
    pos = np.zeros((np.size(mass), 3))

    if idir == 0:
        pos[:, 0] = pos_orig[:, 1]
        pos[:, 1] = pos_orig[:, 2]
        pos[:, 2] = pos_orig[:, 0]

        xres = res
        yres = res

        boxx = boxsize
        boxy = boxsize
    elif idir == 1:
        pos[:, 0] = pos_orig[:, 0]
        pos[:, 1] = pos_orig[:, 2]
        pos[:, 2] = pos_orig[:, 1]

        xres = res // 2
        yres = res

        boxx = boxsize / 2.
        boxy = boxsize
    elif idir == 2:
        pos[:, 0] = pos_orig[:, 1]
        pos[:, 1] = pos_orig[:, 0]
        pos[:, 2] = pos_orig[:, 2]

        xres = res
        yres = res // 2

        boxx = boxsize
        boxy = boxsize / 2.

    import pysph

    tree = pysph.makeTree(pos)
    hsml = tree.calcHsmlMulti(pos, pos, mass, 48, numthreads=8)
    if maxHsml:
        hsml = np.minimum(hsml, 4. * boxsize / res)
    hsml = np.maximum(hsml, 1.001 * boxsize / res * 0.5)
    rho = np.ones(np.size(mass))

    datarange = np.array([[4003.36, 800672.], [199.370, 132913.], [133.698, 200548.]])
    fac = (512. / res) ** 2 * (0.5 * boxsize / 0.025) ** 2
    datarange *= fac

    import calcGrid

    boxz = max(boxx, boxy)

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
    elif type == 'mass':
        proj = calcGrid.calcGrid(pos, hsml, data, rho, rho, xres, yres, 256, boxx, boxy, boxz, 0., 0., 0., 1, 1, numthreads=8)
    else:
        print("Type %s not found." % type)
        return None

    return proj


def dm_mass(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 2.3 * ((nhalos - 1) // 5 + 1) + 1.2), dpi=300)
    boxsize = 0.4

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[1, 4], loadonly=['pos', 'vol', 'rho', 'mass', 'vel'])

        isnap = 0
        for s in data:
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axis(f, isnap * nlevels + il, res=res, boxsize=boxsize * 1e3, cb=True)

            idm, = np.where((s.r() < 2. * boxsize) & (s.type == 1))

            proj = get_projection(s.pos[idm, :].astype('f8'), s.mass[idm].astype('f8'), s.data['mass'][idm].astype('f8'), 0, res, boxsize, 'mass',
                                  maxHsml=False) / area * 1e10
            pc = ax.pcolormesh(x, y, proj, norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e9), cmap=matplotlib.cm.Greys, rasterized=True)
            if isnap == 0:
                create_colorbar(f, pc, "$\Sigma_\mathrm{DM}\,\mathrm{[M_\odot\,kpc^{-2}]}$")

            proj = get_projection(s.pos[idm, :].astype('f8'), s.mass[idm].astype('f8'), s.data['mass'][idm].astype('f8'), 1, res, boxsize, 'mass',
                                  maxHsml=False) / (0.5 * area) * 1e10
            ax2.pcolormesh(x, y2, proj, norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e9), cmap=matplotlib.cm.Greys, rasterized=True)

            ax.text(0.05, 0.92, "Au%s-%d" % (s.haloname, level), color='k', fontsize=6, transform=ax.transAxes)

            set_axis(isnap * nlevels + il, ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            isnap += 1

    pdf.savefig(f)
    plt.close()
    return


def stellar_mass(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 2.3 * ((nhalos - 1) // 5 + 1) + 1.2), dpi=300)

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[4])

        isnap = 0
        for s in data:
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axis(f, isnap * nlevels + il, res=res, boxsize=boxsize * 1e3, cb=True)

            idm, = np.where((s.r() < 2. * boxsize))

            proj = get_projection(s.pos[idm, :].astype('f8'), s.mass[idm].astype('f8'), s.data['mass'][idm].astype('f8'), 0, res, boxsize, 'mass',
                                  maxHsml=False) / area * 1e10
            pc = ax.pcolormesh(x, y, proj, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap=matplotlib.cm.Oranges, rasterized=True)
            if isnap == 0:
                create_colorbar(f, pc, "$\Sigma_\mathrm{stars}\,\mathrm{[M_\odot\,kpc^{-2}]}$")

            proj = get_projection(s.pos[idm, :].astype('f8'), s.mass[idm].astype('f8'), s.data['mass'][idm].astype('f8'), 1, res, boxsize, 'mass',
                                  maxHsml=False) / (0.5 * area) * 1e10
            ax2.pcolormesh(x, y2, proj, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap=matplotlib.cm.Oranges, rasterized=True)

            ax.text(0.05, 0.92, "Au%s-%d" % (s.haloname, level), color='k', fontsize=6, transform=ax.transAxes)

            set_axis(isnap * nlevels + il, ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            isnap += 1

    pdf.savefig(f)
    plt.close()
    return


def gas_density(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 2.3 * ((nhalos - 1) // 5 + 1) + 1.2), dpi=300)
    res = 256

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[0, 4], loadonly=['pos', 'vol', 'rho', 'mass', 'vel'])

        isnap = 0
        for s in data:
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axis(f, isnap * nlevels + il, res=res, boxsize=boxsize * 1e3, cb=True)

            proj = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] * boxsize * 1e3
            pc = ax.pcolormesh(x, y, proj.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap=matplotlib.cm.magma, rasterized=True)
            if isnap == 0:
                create_colorbar(f, pc, "$\Sigma_\mathrm{gas}\,\mathrm{[M_\odot\,kpc^{-2}]}$")

            proj = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"] * boxsize * 1e3
            ax2.pcolormesh(x, 0.5 * y, proj.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap=matplotlib.cm.magma, rasterized=True)

            ax.text(0.05, 0.92, "Au%s-%d" % (s.haloname, level), color='k', fontsize=6, transform=ax.transAxes)

            set_axis(isnap * nlevels + il, ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            isnap += 1

    pdf.savefig(f)
    plt.close()
    return


def gas_temperature(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 2.3 * ((nhalos - 1) // 5 + 1) + 1.2), dpi=300)
    res = 256

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[0, 4], loadonly=['pos', 'vol', 'rho', 'mass', 'vel', 'ne', 'u'])

        isnap = 0
        for s in data:
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axis(f, isnap * nlevels + il, res=res, boxsize=boxsize * 1e3, cb=True)

            meanweight = 4.0 / (1. + 3. * 0.76 + 4. * 0.76 * s.data['ne']) * 1.67262178e-24;
            temp = (5. / 3. - 1.) * s.data['u'] / KB * (1e6 * parsec) ** 2. / (1e6 * parsec / 1e5) ** 2 * meanweight;
            s.data['temprho'] = s.rho * temp

            proj = s.get_Aslice("temprho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]
            rho = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"]
            pc = ax.pcolormesh(x, y, (proj / rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap=matplotlib.cm.viridis, rasterized=True)
            if isnap == 0:
                create_colorbar(f, pc, "$T\,\mathrm{[K]}$")

            proj = s.get_Aslice("temprho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"]
            rho = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"]
            ax2.pcolormesh(x, 0.5 * y, (proj / rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap=matplotlib.cm.viridis,
                           rasterized=True)

            ax.text(0.05, 0.92, "Au%s-%d" % (s.haloname, level), color='k', fontsize=6, transform=ax.transAxes)

            set_axis(isnap * nlevels + il, ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            isnap += 1

    pdf.savefig(f)
    plt.close()
    return


def gas_metallicity(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 2.3 * ((nhalos - 1) // 5 + 1) + 1.2), dpi=300)
    res = 256

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[0, 4], loadonly=['pos', 'vol', 'rho', 'mass', 'vel', 'gz'])

        isnap = 0
        for s in data:
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axis(f, isnap * nlevels + il, res=res, boxsize=boxsize * 1e3, cb=True)

            proj = s.get_Aslice("gz", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res / 0.0134
            pc = ax.pcolormesh(x, y, proj.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), cmap=matplotlib.cm.viridis, rasterized=True)
            if isnap == 0:
                create_colorbar(f, pc, "$Z/Z_\odot$")

            proj = s.get_Aslice("gz", res=res, axes=[1, 0], box=[boxsize, boxsize / 2.], proj=True, numthreads=8)["grid"] / res / 0.0134
            ax2.pcolormesh(x, 0.5 * y, proj.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), cmap=matplotlib.cm.viridis, rasterized=True)

            ax.text(0.05, 0.92, "Au%s-%d" % (s.haloname, level), color='k', fontsize=6, transform=ax.transAxes)

            set_axis(isnap * nlevels + il, ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            isnap += 1

    pdf.savefig(f)
    plt.close()
    return


def bfld(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 2.3 * ((nhalos - 1) // 5 + 1) + 1.2), dpi=300)
    res = 256

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlytype=[0, 4], loadonly=['pos', 'vol', 'rho', 'mass', 'vel', 'bfld'])

        isnap = 0
        for s in data:
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            ax, ax2, x, y, y2, area = create_axis(f, isnap * nlevels + il, res=res, boxsize=boxsize * 1e3, cb=True)

            s.data['b2'] = (s.data['bfld'] ** 2.).sum(axis=1)

            proj = np.sqrt(s.get_Aslice("b2", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res) * bfac * 1e6
            pc = ax.pcolormesh(x, y, proj.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=5e1), cmap=matplotlib.cm.CMRmap, rasterized=True)
            if isnap == 0:
                create_colorbar(f, pc, "$B\,\mathrm{[\mu G]}$")

            proj = np.sqrt(s.get_Aslice("b2", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, numthreads=8)["grid"] / res) * bfac * 1e6
            ax2.pcolormesh(x, 0.5 * y, proj.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=5e1), cmap=matplotlib.cm.CMRmap, rasterized=True)

            ax.text(0.05, 0.92, "Au%s-%d" % (s.haloname, level), color='k', fontsize=6, transform=ax.transAxes)

            set_axis(isnap * nlevels + il, ax, ax2, xlabel='$x\,\mathrm{[kpc]}$', ylabel='$y\,\mathrm{[kpc]}$', y2label='$z\,\mathrm{[kpc]}$')

            isnap += 1

    pdf.savefig(f)
    plt.close()
    return