from __future__ import division

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from const import *
from sfigure import *

mass_proton = 1.6726219e-27


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

    return


def phasediagram(pdf, data, levels):
    nlevels = len(levels)

    redshifts = [4., 3., 2., 1., 0.]

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], 0)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.4 * nhalos + 0.7))

    for il in range(nlevels):
        level = levels[il]

        for iz in range(5):
            data.select_halos(level, redshifts[iz], loadonlytype=[0])

            isnap = 0
            for s in data:
                mu = 4. / (1. + 3. * s.data['gmet'][:, 0] + 4. * s.data['gmet'][:, 0]) * mass_proton
                temp = (5. / 3. - 1.) * s.data['u'] / KB * 1e10 * mu
                rho = (s.data['gmet'][:, 0] / mass_proton + s.data['gmet'][:, 1] / (4. * mass_proton)) * s.data['rho'].astype('f8') * 1e10 * msol / (
                    1e6 * parsec) ** 3

                H, x, y = np.histogram2d(np.log10(rho), np.log10(temp), bins=100, weights=s.rho * s.vol)

                iax = isnap * 5 + iz
                ax = create_axis(f, iax)
                ax.pcolormesh(x, y, np.log10(H).T)
                set_axis(iax, ax, "$\log \\rho\,\mathrm{[cm^{-3}]}$", "$\log T\,\mathrm{[K]}$")
                ax.set_ylim(-2, 6)
                ax.set_xlim(-6, 6)
                ax.text(0.05, 0.90, "Au%s-%d, z=%3.1f" % (s.haloname, level, redshifts[iz]), color='k', fontsize=6, transform=ax.transAxes)

                isnap += 1

    pdf.savefig(f)
    plt.close()
    return


def circularity(pdf, data, levels):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], 0.)
        nhalos += data.selected_current_nsnaps

    Gcosmo = 43.0071
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.4 * ((nhalos - 1) // 5 + 1) + 0.7))

    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, 0., loadonlyhalo=0)

        isnap = 0
        for s in data:
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, do_rotation=True)

            age = np.zeros(s.npartall)
            age[s.type == 4] = s.data['age']

            galrad = 0.1 * s.subfind.data['frc2'][0]
            iall, = np.where((s.r() < galrad) & (s.r() > 0.))
            istars, = np.where((s.r() < galrad) & (s.r() > 0.) & (s.type == 4) & (age > 0.))
            nall = np.size(iall)
            nstars = np.size(istars)

            rsort = s.r()[iall].argsort()
            msum = np.zeros(nall)
            msum[rsort] = np.cumsum(s.mass[iall][rsort])

            nn, = np.where((s.type[iall] == 4) & (age[iall] > 0.))
            smass = s.mass[iall][nn]
            jz = np.cross(s.pos[iall, :][nn, :].astype('f8'), s.data['vel'][iall, :][nn, :])[:, 0]
            ene = 0.5 * (s.vel[iall, :][nn, :].astype('f8') ** 2.).sum(axis=1) + s.data['pot'][iall][nn].astype('f8')
            esort = ene.argsort()

            jz = jz[esort]
            smass = smass[esort]

            jcmax = np.zeros(nstars)
            for nn in range(nstars):
                if nn < 50:
                    left = 0
                    right = 100
                elif nn > nstars - 50:
                    left = nstars - 100
                    right = nstars
                else:
                    left = nn - 50
                    right = nn + 50

                jcmax[nn] = np.max(jz[left:right])

            eps = jz / jcmax

            ax = create_axis(f, isnap)
            ydata, edges = np.histogram(eps, weights=smass / smass.sum(), bins=100, range=[-1.7, 1.7])
            ydata /= edges[1:] - edges[:-1]
            ax.plot(0.5 * (edges[1:] + edges[:-1]), ydata, 'k')

            set_axis(isnap, ax, "$\\epsilon$", "$f\\left(\\epsilon\\right)$", None)
            ax.text(0.05, 0.90, "Au%s-%d" % (s.haloname, level), color='k', fontsize=6, transform=ax.transAxes)
            ax.set_xlim(-2., 2.)
            ax.set_xticks([-1.5, 0., 1.5])

            isnap += 1

    pdf.savefig(f)
    plt.close()
    return


# see http://www.astro.umd.edu/~ssm/ASTR620/mags.html
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]
photo_band = ['U', 'B', 'V', 'K', 'g', 'r', 'i', 'z']


# The observational fit is taken from equation 24 of Dutton et al. 2011
def obs_tullyfisher_fit(masses):
    vel = 10 ** (2.179 + 0.259 * np.log10(masses / 10 ** 0.3))
    return vel


# see Bell et al. 2003. Table in the appendix
def pizagno_convert_color_to_mass(color, magnitude, band=5):
    mass_to_light = 10 ** (-0.306 + 1.097 * color)
    luminosity = 10 ** (2.0 * (Msunabs[band] - magnitude) / 5.0)
    stellar_mass = np.log10(mass_to_light * luminosity * 1.0e-10)  # in 10^10 M_sun
    # this converts in the MPA stellar masses eq. 20 of Dutton+11. There is an offset
    # of -0.1 dex that takes into account the Chabrier IMF see sect. 3.5
    stellar_mass = (stellar_mass - 0.230) / 0.922
    return 10 ** stellar_mass


def verheijen_convert_color_to_mass(color, magnitude, band=1):
    mass_to_light = 10 ** (-0.976 + 1.111 * color)
    luminosity = 10 ** (2.0 * (Msunabs[band] - magnitude) / 5.0)
    stellar_mass = np.log10(mass_to_light * luminosity * 1.0e-10)  # in 10^10 M_sun
    # this converts in the MPA stellar masses eq. 20 of Dutton+11. There is an offset
    # of -0.1 dex that takes into account the Chabrier IMF see sect. 3.5
    stellar_mass = (stellar_mass - 0.230) / 0.922
    return 10 ** stellar_mass


# from Dutton et al. (2007) eq. 28
def courteau_convert_luminosity_to_mass(loglum):
    mass_to_light = 10 ** (0.172 + 0.144 * (loglum - 10.3))
    luminosity = 10 ** (loglum)
    stellar_mass = np.log10(mass_to_light * luminosity * 1.0e-10)  # in 10^10 M_sun
    # this converts in the MPA stellar masses eq. 20 of Dutton+11. There is an offset
    # of -0.1 dex that takes into account the Chabrier IMF see sect. 3.5
    stellar_mass = (stellar_mass - 0.230) / 0.922
    return 10 ** stellar_mass


def tullyfisher(pdf, data, levels):
    nlevels = len(levels)

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 7., 7., top=True)

    # plot Pizagno et al. (2007) sample
    tablename = "./data/pizagno.txt"

    rmag_p = np.genfromtxt(tablename, comments='#', usecols=3)
    gmag_p = np.genfromtxt(tablename, comments='#', usecols=2)
    vcirc_p = np.genfromtxt(tablename, comments='#', usecols=9)
    color_p = np.genfromtxt(tablename, comments='#', usecols=5)
    mass_p = pizagno_convert_color_to_mass(color_p, rmag_p)

    ax.semilogx(1.0e10 * mass_p, np.log10(vcirc_p), '^', mfc='lightgray', ms=2.5, mec='None')

    # plot Verheijen (2001) sample
    tablename = "./data/verheijen.txt"

    Bmag_v = np.genfromtxt(tablename, comments='#', usecols=1)
    Rmag_v = np.genfromtxt(tablename, comments='#', usecols=2)
    vcirc_v = np.genfromtxt(tablename, comments='#', usecols=7)
    color_v = Bmag_v - Rmag_v
    mass_v = verheijen_convert_color_to_mass(color_v, Bmag_v)

    ax.semilogx(1.0e10 * mass_v, np.log10(vcirc_v), 's', mfc='lightgray', ms=2.5, mec='None')

    # plot Courteau+ (2007) sample
    tablename = "./data/courteau.txt"

    loglum_c = np.genfromtxt(tablename, comments='#', usecols=6)
    vcirc_c = np.genfromtxt(tablename, comments='#', usecols=8)
    mass_c = courteau_convert_luminosity_to_mass(loglum_c)

    ax.semilogx(1.0e10 * mass_c, vcirc_c, 'o', mfc='lightgray', ms=2.5, mec='None')

    # plot best fit from Dutton et al. (2011)
    masses = np.arange(0.1, 50.)
    ax.semilogx(1.0e10 * masses, np.log10(obs_tullyfisher_fit(masses)), ls='--', color='darkgray', lw=0.8)
    label = ['Pizagno+ 07', 'Verheijen 01', 'Courteau+ 07']
    l1 = ax.legend(label, loc='upper left', fontsize=12, frameon=False, numpoints=1)

    for il in range(nlevels):
        level = levels[il]

        data.select_halos(level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))

        mstar = np.zeros(nhalos)
        vtot = np.zeros(nhalos)

        data.select_halos(level, 0., loadonlyhalo=0)

        ihalo = 0
        for s in data:
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=False)

            age = np.zeros(s.npartall)
            age[s.type == 4] = s.data['age']
            istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.type == 4) & (age >= 0.))
            mstars = s.mass[istars].sum()

            mass, edges = np.histogram(s.r(), weights=s.mass, bins=100, range=[0., 0.025])
            mcum = np.cumsum(mass)
            vel = np.sqrt(G * mcum * 1e10 * msol / (edges[1:] * 1e6 * parsec)) / 1e5

            smass, edges = np.histogram(s.r()[istars], weights=s.mass[istars], bins=100, range=[0., 0.025])
            smcum = np.cumsum(smass)

            mstar[ihalo] = mstars
            vtot[ihalo] = vel[np.abs(smcum - 0.8 * mstars).argmin()]

            ax.semilogx(mstar[ihalo] * 1e10, np.log10(vtot[ihalo]), color=next(colors), linestyle="None", marker='*', ms=15.0,
                        label='Au-' + s.haloname)
            ax.legend(loc='lower right', fontsize=12, frameon=False, numpoints=1)
            ax.add_artist(l1)
            ihalo += 1

    ax.set_xlabel("$\\rm{M_{stars}\\,[M_\\odot]}$")
    ax.set_ylabel("$\\rm{log_{10}\\,\\,v\\,[km\\,\\,s^{-1}]}$")

    pdf.savefig(f)
    plt.close()
    return


def guo_abundance_matching(mass):
    # equation 3 of Guo et al. 2010. Mass MUST be given in 10^10 M_sun
    c = 0.129
    M_zero = 10 ** 1.4
    alpha = -0.926
    beta = 0.261
    gamma = -2.44

    val = c * mass * ((mass / M_zero) ** alpha + (mass / M_zero) ** beta) ** gamma
    return val


def stellarvstotal(pdf, data, levels):
    nlevels = len(levels)

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 7., 7., top=True)

    masses = np.arange(15., 300.)
    cosmic_baryon_frac = 0.048 / 0.307

    ax.loglog(1.0e10 * masses, 1.0e10 * masses * cosmic_baryon_frac, 'k--', )

    guo_high = guo_abundance_matching(masses) * 10 ** (+0.2)
    guo_low = guo_abundance_matching(masses) * 10 ** (-0.2)

    ax.fill_between(1.0e10 * masses, 1.0e10 * guo_low, 1.0e10 * guo_high, color='lightgray', edgecolor='None')

    ax.loglog(1.0e10 * masses, 1.0e10 * guo_abundance_matching(masses), 'k:')

    labels = ["$\\rm{M_{200}\\,\\,\\,\\Omega_b / \\Omega_m}$", "Guo+ 10"]
    l1 = ax.legend(labels, loc='upper right', fontsize=12, frameon=False)

    for il in range(nlevels):
        level = levels[il]

        data.select_halos(level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))

        mstar = np.zeros(nhalos)
        mhalo = np.zeros(nhalos)

        data.select_halos(level, 0., loadonlyhalo=0)

        ihalo = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])

            age = np.zeros(s.npartall)
            age[s.type == 4] = s.data['age']
            istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.type == 4) & (age > 0.))
            mstar[ihalo] = s.mass[istars].sum()

            iall, = np.where(s.r() < s.subfind.data['frc2'][0])
            mhalo[ihalo] = s.mass[iall].sum()

            ax.loglog(mhalo[ihalo] * 1e10, mstar[ihalo] * 1e10, color=next(colors), linestyle="None", marker='*', ms=15.0, label='Au-' + s.haloname)
            ax.legend(loc='lower right', fontsize=12, frameon=False, numpoints=1)
            ax.add_artist(l1)

            ihalo += 1

    ax.set_xlabel("$\\rm{M_{halo}\\,[M_\\odot]}$")
    ax.set_ylabel("$\\rm{M_{stars}\\,[M_\\odot]}$")

    pdf.savefig(f)
    plt.close()
    return


# for this formula see Appendix of Windhorst+ 1991
def convert_rband_to_Rband_mag(r, g):
    R = r - 0.51 - 0.15 * (g - r)
    return R


def gasfraction(pdf, data, levels):
    nlevels = len(levels)

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 6.8, 6.8, top=True)

    for il in range(nlevels):
        level = levels[il]

        data.select_halos(level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))

        MR = np.zeros(nhalos)
        fgas = np.zeros(nhalos)

        data.select_halos(level, 0., loadonlyhalo=0)

        ihalo = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])

            age = np.zeros(s.npartall)
            age[s.type == 4] = s.data['age']
            istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.type == 4) & (age > 0.)) - s.nparticlesall[:4].sum()
            Rband = convert_rband_to_Rband_mag(s.data['gsph'][istars, 5], s.data['gsph'][istars, 4])
            MR[ihalo] = -2.5 * np.log10((10. ** (- 2.0 * Rband / 5.0)).sum())

            igas, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.type == 0))
            fgas[ihalo] = s.mass[igas].sum() / (s.mass[igas].sum() + s.mass[istars].sum())

            ax.plot(MR[ihalo], fgas[ihalo], color=next(colors), linestyle="None", marker='*', ms=15.0, label='Au-' + s.haloname)
            ax.legend(loc='lower right', fontsize=12, frameon=False, numpoints=1)

            ihalo += 1

    ax.set_xlabel("$\\rm{M_{R}\\,[mag]}$")
    ax.set_ylabel("$\\rm{f_{gas}}$")

    pdf.savefig(f)
    plt.close()
    return


def centralbfld(pdf, data, levels):
    nlevels = len(levels)

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 6.8, 6.8, top=True)

    for il in range(nlevels):
        level = levels[il]

        data.select_halos(level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))

        mstar = np.zeros(nhalos)
        bfld = np.zeros(nhalos)

        data.select_halos(level, 0., loadonlytype=[0, 4], loadonlyhalo=0)

        ihalo = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])

            i, = np.where((s.r() < 0.001) & (s.type == 0))
            bfld[ihalo] = np.sqrt(((s.data['bfld'][i, :] ** 2.).sum(axis=1) * s.data['vol'][i]).sum() / s.data['vol'][i].sum()) * bfac

            age = np.zeros(s.npartall)
            age[s.type == 4] = s.data['age']
            istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.type == 4) & (age > 0.))
            mstar[ihalo] = s.mass[istars].sum()

            ax.loglog(mstar[ihalo] * 1e10, bfld[ihalo] * 1e6, color=next(colors), linestyle="None", marker='*', ms=15.0, label='Au-' + s.haloname)
            ax.legend(loc='lower right', fontsize=12, frameon=False, numpoints=1)

            ihalo += 1

    ax.set_xlabel("$\\rm{M_{stars}\\,[M_\\odot]}$")
    ax.set_ylabel("$B_\mathrm{r<1\,kpc}\,\mathrm{[\mu G]}$")

    pdf.savefig(f)
    plt.close()
    return


def table(pdf, data, levels):
    nlevels = len(levels)

    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], 0.)
        nhalos += data.selected_current_nsnaps

    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 7., 7., top=True)

    text = []
    names = []

    ihalo = 0
    for il in range(nlevels):
        level = levels[il]

        data.select_halos(level, 0., loadonlyhalo=0)

        ihalo = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])

            mass = s.subfind.data['fmc2'][0]

            age = np.zeros(s.npartall)
            age[s.type == 4] = s.data['age']
            istars, = np.where((s.r() < s.subfind.data['frc2'][0]) & (s.type == 4) & (age > 0.))
            mstar = s.mass[istars].sum()

            igas, = np.where((s.r() < s.subfind.data['frc2'][0]) & (s.type == 0))
            mgas = s.mass[igas].sum()

            text.append(["%d" % level, "%g" % mass, "%g" % mstar, "%g" % mgas])
            names += [s.haloname]

            ihalo += 1

    ax.table(cellText=text, rowLabels=names, colLabels=["level", "Mhalo", "Mstar", "Mgas"], loc='center')
    ax.axis("off")

    pdf.savefig(f)
    plt.close()
    return