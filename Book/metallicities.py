from __future__ import division

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from const import *
from sfigure import *


def create_axis(f, idx, ncol=5):
    ix = idx % ncol
    iy = idx // ncol

    s = 1.

    ax = f.iaxes(0.5 + ix * (s + 0.5), 0.2 + s + iy * (s + 0.4), s, s, top=False)
    return ax


def set_axis(isnap, ax, xlabel=None, ylabel=None, title=None, ylim=None, ncol=5):
    if ylabel is None:
        ax.set_yticks([])
    else:
        ax.set_ylabel(ylabel, size=6)

    if xlabel is None:
        ax.set_xticks([])
    else:
        ax.set_xlabel(xlabel, size=6)

    for label in ax.xaxis.get_ticklabels():
        label.set_size(6)
    for label in ax.yaxis.get_ticklabels():
        label.set_size(6)

    if ylim is not None:
        ax.set_ylim(ylim)

    if isnap == 0 and title is not None:
        ax.set_title(title, size=7)

    return None


def ratios(pdf, data, levels, z):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], z)
        nhalos += data.selected_current_nsnaps

    elements = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe', 'Y', 'Sr', 'Zr', 'Ba']
    elements_mass = [1.01, 4.00, 12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33]
    elements_solar = [12.0, 10.93, 8.43, 7.83, 8.69, 7.93, 7.60, 7.51, 7.50, 2.21, 2.87, 2.58, 2.18]
    nelements = len(elements)

    mass_eu = 151.96
    solar_eu = 0.52

    zsolar = 0.0127
    rows_per_element = (nhalos - 1) // 5 + 1

    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.4 * (12 + nelements - 2) * rows_per_element + 0.7))

    for il in range(nlevels):
        level = levels[il]

        for ele in range(2, nelements):
            data.select_haloes(level, z, loadonlytype=[4], loadonlyhalo=0)
            ihalo = 0
            for s in data:
                if np.shape(s.data['gmet'])[1] > ele:
                    s.data['gmet'] = np.maximum(s.data['gmet'], 1e-40)
                    s.centerat(s.subfind.data['fpos'][0, :])
                    istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.data['age'] > 0.))

                    nfe = np.log10((s.data['gmet'][istars, 8] / elements_mass[8]) / (s.data['gmet'][istars, 0] / elements_mass[0]))
                    nele = np.log10((s.data['gmet'][istars, ele] / elements_mass[ele]) / (s.data['gmet'][istars, 8] / elements_mass[8]))

                    nfe -= elements_solar[8] - elements_solar[0]
                    nele -= elements_solar[ele] - elements_solar[8]

                    iax = (ele - 2) * rows_per_element * 5 + ihalo
                    ax = create_axis(f, iax)
                    ax.hist2d(nfe, nele, bins=(160, 80), range=([-2.5, 1.1], [-1.0, 1.0]), weights=s.mass[istars], normed=False, rasterized=True,
                              norm=matplotlib.colors.LogNorm(), cmap=matplotlib.cm.viridis)

                    set_axis(iax, ax, "$\mathrm{[Fe/H]}$", "$\mathrm{[%s/Fe]}$" % elements[ele])
                    ax.text(0.05, 0.92, "Au%s-%d" % (s.haloname, level), color='w', fontsize=6, transform=ax.transAxes)

                ihalo += 1

        for ele in range(3):
            data.select_haloes(level, z, loadonlytype=[4], loadonlyhalo=0)
            ihalo = 0
            for s in data:
                if 'gmrp' in s.data and np.shape(s.data['gmrp'])[1] >= ele:
                    s.centerat(s.subfind.data['fpos'][0, :])
                    s.data['gmet'] = np.maximum(s.data['gmet'], 1e-40)
                    s.data['gmrp'] = np.maximum(s.data['gmrp'], 1e-40)

                    istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.data['age'] > 0.))

                    nfe = np.log10((s.data['gmet'][istars, 8] / elements_mass[8]) / (s.data['gmet'][istars, 0] / elements_mass[0]))
                    nele = np.log10((s.data['gmrp'][istars, ele] / mass_eu) / (s.data['gmet'][istars, 8] / elements_mass[8]))

                    nfe -= elements_solar[8] - elements_solar[0]
                    nele -= solar_eu - elements_solar[8]

                    i, = np.where((nfe > -0.2) & (nfe < 0.2))
                    nele -= nele[i].sum() / np.size(i)

                    iax = (nelements - 2 + ele) * rows_per_element * 5 + ihalo
                    ax = create_axis(f, iax)
                    ax.hist2d(nfe, nele, bins=(160, 80), range=([-2.5, 1.1], [-1.0, 2.0]), weights=s.mass[istars], normed=False, rasterized=True,
                              norm=matplotlib.colors.LogNorm(), cmap=matplotlib.cm.viridis)

                    set_axis(iax, ax, "$\mathrm{[Fe/H]}$", "$\mathrm{[Eu_{%d}/Fe]}$" % ele)
                    ax.text(0.05, 0.92, "Au%s-%d" % (s.haloname, level), color='w', fontsize=6, transform=ax.transAxes)

                ihalo += 1

    pdf.savefig(f)
    return None