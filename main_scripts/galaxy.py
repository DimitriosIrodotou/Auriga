from __future__ import division

import main_scripts.projections

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

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
    
    return None


def set_axis_evo(s, ax, ax2, ylabel, ylim=None, ncol=5):
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
    
    ax.set_ylabel(ylabel, size=12)
    ax.set_xlabel("$t_\mathrm{look}\,\mathrm{[Gyr]}$", size=12)
    ax2.set_xlabel("$z$", size=12)
    
    for a in [ax, ax2]:
        for label in a.xaxis.get_ticklabels():
            label.set_size(12)
        for label in a.yaxis.get_ticklabels():
            label.set_size(12)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    return None


def circularity(pdf, data, levels):
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], 0.)
        nhalos += data.selected_current_nsnaps
    
    Gcosmo = 43.0071
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.4 * ((nhalos - 1) // 5 + 1) + 0.7))
    
    for il in range(nlevels):
        level = levels[il]
        data.select_haloes(level, 0., loadonlyhalo=0)
        
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
            
            jj, = np.where((eps > 0.7) & (eps < 1.7))
            ll, = np.where((eps > -1.7) & (eps < 1.7))
            disc_frac = smass[jj].sum() / smass[ll].sum()
            
            ax = create_axis(f, isnap)
            ydata, edges = np.histogram(eps, weights=smass / smass.sum(), bins=100, range=[-1.7, 1.7])
            ydata /= edges[1:] - edges[:-1]
            ax.plot(0.5 * (edges[1:] + edges[:-1]), ydata, 'k')
            
            set_axis(isnap, ax, "$\\epsilon$", "$f\\left(\\epsilon\\right)$", None)
            ax.text(0.05, 0.90, "Au%s-%d" % (s.haloname, level), color='k', fontsize=6, transform=ax.transAxes)
            ax.text(0.05, 0.8, "D/T = %.2f" % disc_frac, color='k', fontsize=6, transform=ax.transAxes)
            ax.set_xlim(-2., 2.)
            ax.set_xticks([-1.5, 0., 1.5])
            
            isnap += 1
    
    pdf.savefig(f)
    return None


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


def tully_fisher(pdf, data, levels):
    nlevels = len(levels)
    
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 7., 7., top=True)
    ax.set_xlabel("$\\rm{M_{stars}\\,[M_\\odot]}$")
    ax.set_ylabel("$\\rm{log_{10}\\,\\,v\\,[km\\,\\,s^{-1}]}$")
    
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
        
        data.select_haloes(level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
        
        vtot = np.zeros(nhalos)
        mstar = np.zeros(nhalos)
        
        data.select_haloes(level, 0., loadonlyhalo=0)
        
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
                        label="Au%s-%d" % (s.haloname, levels[0]))
            ax.legend(loc='lower right', fontsize=12, frameon=False, numpoints=1)
            ax.add_artist(l1)
            ihalo += 1
    
    pdf.savefig(f)
    
    return None


def guo_abundance_matching(mass):
    # equation 3 of Guo et al. 2010. Mass MUST be given in 10^10 M_sun
    c = 0.129
    M_zero = 10 ** 1.4
    alpha = -0.926
    beta = 0.261
    gamma = -2.44
    
    val = c * mass * ((mass / M_zero) ** alpha + (mass / M_zero) ** beta) ** gamma
    return val


def stellar_vs_total(pdf, data, levels):
    nlevels = len(levels)
    
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 7., 7., top=True)
    ax.set_xlabel("$\\rm{M_{halo}\\,[M_\\odot]}$")
    ax.set_ylabel("$\\rm{M_{stars}\\,[M_\\odot]}$")
    
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
        
        data.select_haloes(level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
        
        mstar = np.zeros(nhalos)
        mhalo = np.zeros(nhalos)
        
        data.select_haloes(level, 0., loadonlyhalo=0)
        
        ihalo = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])
            
            age = np.zeros(s.npartall)
            age[s.type == 4] = s.data['age']
            istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.type == 4) & (age > 0.))
            mstar[ihalo] = s.mass[istars].sum()
            
            iall, = np.where(s.r() < s.subfind.data['frc2'][0])
            mhalo[ihalo] = s.mass[iall].sum()
            
            ax.loglog(mhalo[ihalo] * 1e10, mstar[ihalo] * 1e10, color=next(colors), linestyle="None", marker='*', ms=15.0,
                      label="Au%s-%d" % (s.haloname, levels[0]))
            ax.legend(loc='lower right', fontsize=12, frameon=False, numpoints=1)
            ax.add_artist(l1)
            
            ihalo += 1
    
    pdf.savefig(f)
    return None


# for this formula see Appendix of Windhorst+ 1991
def convert_rband_to_Rband_mag(r, g):
    R = r - 0.51 - 0.15 * (g - r)
    return R


def gas_fraction(pdf, data, levels):
    nlevels = len(levels)
    
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 6.8, 6.8, top=True)
    ax.set_xlabel("$\\rm{M_{R}\\,[mag]}$")
    ax.set_ylabel("$\\rm{f_{gas}}$")
    
    for il in range(nlevels):
        level = levels[il]
        
        data.select_haloes(level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
        
        MR = np.zeros(nhalos)
        fgas = np.zeros(nhalos)
        
        data.select_haloes(level, 0., loadonlyhalo=0)
        
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
            
            ax.plot(MR[ihalo], fgas[ihalo], color=next(colors), linestyle="None", marker='*', ms=15.0, label="Au%s-%d" % (s.haloname, levels[0]))
            ax.legend(loc='lower right', fontsize=12, frameon=False, numpoints=1)
            
            ihalo += 1
    
    pdf.savefig(f)
    return None


def central_bfld(pdf, data, levels):
    nlevels = len(levels)
    
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 6.8, 6.8, top=True)
    ax.set_xlabel("$\\rm{M_{stars}\\,[M_\\odot]}$")
    ax.set_ylabel("$B_\mathrm{r<1\,kpc}\,\mathrm{[\mu G]}$")
    
    for il in range(nlevels):
        level = levels[il]
        
        data.select_haloes(level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
        
        mstar = np.zeros(nhalos)
        bfld = np.zeros(nhalos)
        
        data.select_haloes(level, 0., loadonlytype=[0, 4], loadonlyhalo=0)
        
        ihalo = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])
            
            i, = np.where((s.r() < 0.001) & (s.type == 0))
            bfld[ihalo] = np.sqrt(((s.data['bfld'][i, :] ** 2.).sum(axis=1) * s.data['vol'][i]).sum() / s.data['vol'][i].sum()) * bfac
            
            age = np.zeros(s.npartall)
            age[s.type == 4] = s.data['age']
            istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.type == 4) & (age > 0.))
            mstar[ihalo] = s.mass[istars].sum()
            
            ax.loglog(mstar[ihalo] * 1e10, bfld[ihalo] * 1e6, color=next(colors), linestyle="None", marker='*', ms=15.0,
                      label="Au%s-%d" % (s.haloname, levels[0]))
            ax.legend(loc='lower right', fontsize=12, frameon=False, numpoints=1)
            
            ihalo += 1
    
    pdf.savefig(f)
    return None


def bar_strength(pdf, data, level):
    """
    Calculate bar strength from Fourier modes of surface density
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param level: level from main.make_pdf
    :return:
    """
    
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 6.8, 6.8, top=True)
    ax.set_ylabel("$A_{2}/A_{0}$")
    ax.set_xlabel("$r\,\mathrm{[kpc]}$")
    
    data.select_haloes(level, 0., loadonlytype=[4], loadonlyhalo=0)
    nhalos = data.selected_current_nsnaps
    colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
    
    for s in data:
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        mask, = np.where((s.type == 4) & (s.data['age'] > 0.))  # Select stars.
        z_rotated, y_rotated, x_rotated = main_scripts.projections.rotate_bar(s.pos[mask, 0] * 1e3, s.pos[mask, 1] * 1e3,
                                                                              s.pos[mask, 2] * 1e3)  # Distances are in Mpc.
        s.pos = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.pos attribute in kpc.
        x, y = s.pos[:, 2] * 1e3, s.pos[:, 1] * 1e3  # Load positions and convert from Mpc to Kpc.
        
        nbins = 40  # Number of radial bins.
        r = np.sqrt(x[:] ** 2 + y[:] ** 2)  # Radius of each particle.
        
        # Initialise Fourier components #
        r_m = np.zeros(nbins)
        beta_2 = np.zeros(nbins)
        alpha_0 = np.zeros(nbins)
        alpha_2 = np.zeros(nbins)
        
        # Split up galaxy in radius bins and calculate Fourier components #
        for i in range(0, nbins):
            r_s = float(i) * 0.25
            r_b = float(i) * 0.25 + 0.25
            r_m[i] = float(i) * 0.25 + 0.125
            xfit = x[(r < r_b) & (r > r_s)]
            yfit = y[(r < r_b) & (r > r_s)]
            for k in range(0, len(xfit)):
                th_i = np.arctan2(yfit[k], xfit[k])
                alpha_0[i] = alpha_0[i] + 1
                alpha_2[i] = alpha_2[i] + np.cos(2 * th_i)
                beta_2[i] = beta_2[i] + np.sin(2 * th_i)
        
        # Calculate bar strength A_2
        a2 = np.divide(np.sqrt(alpha_2[:] ** 2 + beta_2[:] ** 2), alpha_0[:])
        
        # Plot bar strength as a function of radius plot r_m versus a2
        ax.plot(r_m, a2, color=next(colors), label="Au%s-%d bar strength: %.2f" % (s.haloname, level, max(a2)))
        ax.legend(loc='upper left', fontsize=12, frameon=False, numpoints=1)
    
    pdf.savefig(f)
    return None


def sfr(pdf, data, levels):
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], 0.)
        nhalos += data.selected_current_nsnaps
    
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 6.8, 6.8, top=True)
    ax.set_ylabel("$\\mathrm{Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$")
    ax.set_xlabel("$r\,\mathrm{[kpc]}$")
    
    nbins = 100
    tmin = 0
    tmax = 13.
    timebin = (tmax - tmin) / nbins
    
    for il in range(nlevels):
        level = levels[il]
        data.select_haloes(level, 0., loadonlytype=[4], loadonlyhalo=0)
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
        
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])
            
            mask, = np.where((s.data['age'] > 0.) & (s.r() > 0.005) & (s.r() < 0.015) & (s.pos[:, 2] < 0.003))
            age = s.cosmology_get_lookback_time_from_a(s.data['age'][mask], is_flat=True)
            
            ax.hist(age, weights=s.data['gima'][mask] * 1e10 / 1e9 / timebin, color=next(colors), histtype='step', bins=nbins, range=[tmin, tmax],
                    label="Au%s-%d" % (s.haloname, levels[0]))
            
            ax2 = ax.twiny()
            set_axis_evo(s, ax, ax2, "$\\mathrm{Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$")
            ax.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
            ax.text(0.05, 0.92, "5kpc < r < 15kpc", color='k', fontsize=12, transform=ax.transAxes)
    
    pdf.savefig(f)
    return None


def delta_sfr(pdf, data, levels):
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], 0.)
        nhalos += data.selected_current_nsnaps
    
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 6.8, 6.8, top=True)
    ax.set_ylabel("$\\mathrm{Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$")
    ax.set_xlabel("$r\,\mathrm{[kpc]}$")
    
    nbins = 100
    tmin = 0
    tmax = 13.
    timebin = (tmax - tmin) / nbins
    
    for il in range(nlevels):
        level = levels[il]
        data.select_haloes(level, 0., loadonlytype=[4], loadonlyhalo=0)
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
        i = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])
            
            mask, = np.where((s.data['age'] > 0.) & (s.r() > 0.005) & (s.r() < 0.015) & (s.pos[:, 2] < 0.003))
            age = s.cosmology_get_lookback_time_from_a(s.data['age'][mask], is_flat=True)
            
            counts, bins, bars = ax.hist(age, weights=s.data['gima'][mask] * 1e10 / 1e9 / timebin, color=next(colors), histtype='step', bins=nbins,
                                         range=[tmin, tmax], label="Au%s-%d" % (s.haloname, levels[0]))
            if i == 0:
                tmp_counts = counts
            i += 1
        plt.close()
        f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
        ax = f.iaxes(1.0, 1.0, 6.8, 6.8, top=True)
        ax.set_ylabel("$\\mathrm{Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$")
        ax.set_xlabel("$r\,\mathrm{[kpc]}$")
        
        bins = bins[np.where(bins < 13.0)]
        
        ax.plot(bins, (counts - tmp_counts))
        ax2 = ax.twiny()
        set_axis_evo(s, ax, ax2, "$\\mathrm{\delta Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$")
        ax.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
        ax.text(0.05, 0.92, "5kpc < r < 15kpc", color='k', fontsize=12, transform=ax.transAxes)
    
    pdf.savefig(f)
    return None


def hot_cold_gas_fraction(pdf, data, level):
    
    
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 6.8, 6.8, top=True)
    ax.set_ylabel("$\\mathrm{Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$")
    ax.set_xlabel("$r\,\mathrm{[kpc]}$")
    
    nbins = 100
    tmin = 0
    tmax = 13.
    timebin = (tmax - tmin) / nbins
    
    nhalos = 0
    data.select_haloes(level, 0., loadonlytype=[4], loadonlyhalo=0)
    nhalos += data.selected_current_nsnaps
    colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
    i = 0
    for s in data:
        s.centerat(s.subfind.data['fpos'][0, :])
        
        mask, = np.where((s.data['age'] > 0.) & (s.r() > 0.005) & (s.r() < 0.015) & (s.pos[:, 2] < 0.003))
        age = s.cosmology_get_lookback_time_from_a(s.data['age'][mask], is_flat=True)
        
        counts, bins, bars = ax.hist(age, weights=s.data['gima'][mask] * 1e10 / 1e9 / timebin, color=next(colors), histtype='step', bins=nbins,
                                     range=[tmin, tmax], label="Au%s-%d" % (s.haloname, level))
        if i == 0:
            tmp_counts = counts
        i += 1
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 6.8, 6.8, top=True)
    ax.set_ylabel("$\\mathrm{Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$")
    ax.set_xlabel("$r\,\mathrm{[kpc]}$")
    
    bins = bins[np.where(bins < 13.0)]
    
    ax.plot(bins, (counts - tmp_counts))
    ax2 = ax.twiny()
    set_axis_evo(s, ax, ax2, "$\\mathrm{\delta Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$")
    ax.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
    ax.text(0.05, 0.92, "5kpc < r < 15kpc", color='k', fontsize=12, transform=ax.transAxes)
    
    pdf.savefig(f)
    return None


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx


def table(pdf, data, levels):
    """

    :param pdf:
    :param data:
    :param levels:
    :return:
    """
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], 0.)
        nhalos += data.selected_current_nsnaps
    
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 7., 7., top=True)
    
    text = []
    names = []
    
    for il in range(nlevels):
        level = levels[il]
        
        data.select_haloes(level, 0., loadonlyhalo=0)
        
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
    
    ax.table(cellText=text, rowLabels=names, colLabels=["level", "Mhalo", "Mstar", "Mgas"], loc='center')
    ax.axis("off")
    
    pdf.savefig(f)
    return None