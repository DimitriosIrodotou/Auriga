from __future__ import division

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from const import *
from parallel_decorators import vectorize_parallel
from sfigure import *


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


def create_axis(f, idx, ncol=5):
    ix = idx % ncol
    iy = idx // ncol

    s = 1.

    ax = f.iaxes(0.5 + ix * (s + 0.5), 0.3 + s + iy * (s + 0.6), s, s, top=False)
    ax2 = ax.twiny()
    return ax, ax2


def set_axis(s, ax, ax2, ylabel, ylim=None, ncol=5):
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

    ax.set_xlim(0., 13.)
    ax.invert_xaxis()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(times)
    ax2.set_xticklabels(lb)

    ax.set_ylabel(ylabel, size=6)
    ax.set_xlabel("$t_\mathrm{look}\,\mathrm{[Gyr]}$", size=6)
    ax2.set_xlabel("$z$", size=6)

    for a in [ax, ax2]:
        for label in a.xaxis.get_ticklabels():
            label.set_size(6)
        for label in a.yaxis.get_ticklabels():
            label.set_size(6)

    if ylim is not None:
        ax.set_ylim(ylim)

    return


def sfr(pdf, data, levels):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], 0.)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))

    nbins = 100
    tmin = 0
    tmax = 13.
    timebin = (tmax - tmin) / nbins

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, 0., loadonlytype=[4], loadonlyhalo=0)

        isnap = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])

            i, = np.where((s.data['age'] > 0.) & (s.r() < 0.05))
            age = s.cosmology_get_lookback_time_from_a(s.data['age'][i], is_flat=True)

            ax, ax2 = create_axis(f, isnap)
            ax.hist(age, weights=s.data['gima'][i] * 1e10 / 1e9 / timebin, histtype='step', bins=nbins, range=[tmin, tmax])
            set_axis(s, ax, ax2, "$\\mathrm{Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$")

            ax.text(0.05, 0.92, "Au%s-%d" % (s.haloname, level), color='w', fontsize=6, transform=ax.transAxes)

            isnap += 1

    pdf.savefig(f)
    plt.close()
    return


@vectorize_parallel(method='processes', num_procs=8)
def compute_bfld_halo(snapid, halo, offset):
    s = halo.snaps[snapid].loadsnap(loadonlytype=[0], loadonlyhalo=0)
    s.centerat(s.subfind.data['fpos'][0, :])

    time = s.cosmology_get_lookback_time_from_a(s.time, is_flat=True)
    i, = np.where(s.r() < 0.001)
    bfld = np.sqrt(((s.data['bfld'][i, :] ** 2.).sum(axis=1) * s.data['vol'][i]).sum() / s.data['vol'][i].sum()) * bfac
    return time, bfld


def bfld(pdf, data, levels):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], 0.)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))

    nbins = 100
    tmin = 0
    tmax = 13.
    timebin = (tmax - tmin) // nbins

    for il in range(nlevels):
        level = levels[il]

        res = {}
        rpath = "../plots/data/bfld_%d.npy" % level
        if os.path.exists(rpath):
            with open(rpath, 'rb') as ff:
                res = pickle.load(ff)

        halos = data.get_halos(level)
        for name, halo in halos.items():
            if not name in res:
                redshifts = halo.get_redshifts()

                i, = np.where(redshifts < 10.)
                snapids = np.array(list(halo.snaps.keys()))[i]

                dd = np.array(compute_bfld_halo(snapids, halo, snapids.min()))
                res[name] = {}
                res[name]["time"] = dd[:, 0]
                res[name]["bfld"] = dd[:, 1]

        with open(rpath, 'wb') as ff:
            pickle.dump(res, ff, pickle.HIGHEST_PROTOCOL)

        names = get_names_sorted(res.keys())
        for idx, name in enumerate(names):
            ax, ax2 = create_axis(f, idx)

            ax.semilogy(res[name]["time"], res[name]["bfld"] * 1e6)
            set_axis(list(halos[name].snaps.values())[0].loadsnap(), ax, ax2, "$B_\mathrm{r<1\,kpc}\,\mathrm{[\mu G]}$", [1e-2, 1e2])
            ax.set_xlim([13., 11.])

    pdf.savefig(f)
    plt.close()

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))

    nbins = 100
    tmin = 0
    tmax = 13.
    timebin = (tmax - tmin) // nbins

    for il in range(nlevels):
        level = levels[il]

        res = {}
        rpath = "../plots/data/bfld_%d.npy" % level
        if os.path.exists(rpath):
            with open(rpath, 'rb') as ff:
                res = pickle.load(ff)

        names = get_names_sorted(res.keys())
        for idx, name in enumerate(names):
            ax, ax2 = create_axis(f, idx)

            redshifts = halos[name].get_redshifts()
            ax.plot(res[name]["time"], res[name]["bfld"] * 1e6)
            set_axis(list(halos[name].snaps.values())[0].loadsnap(), ax, ax2, "$B_\mathrm{r<1\,kpc}\,\mathrm{[\mu G]}$", [0., 100.])

    pdf.savefig(f)
    plt.close()
    return


@vectorize_parallel(method='processes', num_procs=8)
def get_halo_mass(snapid, halo):
    s = halo.snaps[snapid].loadsnap(loadonlytype=[1], loadonlyhalo=0)
    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), s.subfind.data['fmc2'][0]


def galaxy_mass(pdf, data, levels):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], 0.)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))

    nbins = 100
    tmin = 0
    tmax = 13.
    timebin = (tmax - tmin) // nbins

    for il in range(nlevels):
        level = levels[il]

        res = {}
        rpath = "../plots/data/mvir_%d.npy" % level
        if os.path.exists(rpath):
            with open(rpath, 'rb') as ff:
                res = pickle.load(ff)

        halos = data.get_halos(level)
        for name, halo in halos.items():
            if not name in res:
                redshifts = halo.get_redshifts()

                i, = np.where(redshifts < 10.)
                snapids = np.array(list(halo.snaps.keys()))[i]
                nsnaps = len(snapids)

                dd = np.array(get_halo_mass(snapids, halo))
                res[name] = {}
                res[name]["time"] = dd[:, 0]
                res[name]["mass"] = dd[:, 1]

        with open(rpath, 'wb') as ff:
            pickle.dump(res, ff, pickle.HIGHEST_PROTOCOL)

        names = get_names_sorted(res.keys())
        for idx, name in enumerate(names):
            ax, ax2 = create_axis(f, idx)

            ax.plot(res[name]["time"], res[name]["mass"])
            set_axis(list(halos[name].snaps.values())[0].loadsnap(), ax, ax2, "$M_\mathrm{vir}\,\mathrm{[10^{10}\,M_\odot]}$")

    pdf.savefig(f)
    plt.close()
    return


@vectorize_parallel(method='processes', num_procs=8)
def get_bh_mass(snapid, halo, bhid):
    s = halo.snaps[snapid].loadsnap(loadonlytype=[5], loadonly=['mass', 'id'])
    if not 'id' in s.data:
        return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), 0.

    i, = np.where(s.data['id'] == bhid)
    if len(i) > 0:
        mass = s.mass[i[0]]
    else:
        mass = 0.
    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), mass


def bh_mass(pdf, data, levels):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], 0.)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))

    nbins = 100
    tmin = 0
    tmax = 13.
    timebin = (tmax - tmin) // nbins

    for il in range(nlevels):
        level = levels[il]

        res = {}
        rpath = "../plots/data/bhmass_%d.npy" % level
        if os.path.exists(rpath):
            with open(rpath, 'rb') as ff:
                res = pickle.load(ff)

        halos = data.get_halos(level)
        for name, halo in halos.items():
            if not name in res:
                redshifts = halo.get_redshifts()

                i, = np.where(redshifts < 10.)
                snapids = np.array(list(halo.snaps.keys()))[i]
                nsnaps = len(snapids)

                s = halo.snaps[snapids.argmax()].loadsnap(loadonlytype=[5], loadonlyhalo=0)
                bhid = s.data['id'][s.mass.argmax()]

                dd = np.array(get_bh_mass(snapids, halo, bhid))
                res[name] = {}
                res[name]["time"] = dd[:, 0]
                res[name]["mass"] = dd[:, 1]

        with open(rpath, 'wb') as ff:
            pickle.dump(res, ff, pickle.HIGHEST_PROTOCOL)

        names = get_names_sorted(res.keys())
        for idx, name in enumerate(names):
            ax, ax2 = create_axis(f, idx)

            ax.semilogy(res[name]["time"], res[name]["mass"] * 1e10)
            set_axis(list(halos[name].snaps.values())[0].loadsnap(), ax, ax2, "$M_\mathrm{BH}\,\mathrm{[M_\odot]}$")

    pdf.savefig(f)
    plt.close()
    return