from const import *
from gadget import *
from gadget_subfind import *
from util import multipanel_layout
from util.label_format import UserLogFormatterMathtext

toinch = 0.393700787

toinch = 0.393700787
BOLTZMANN = 1.38065e-16
PROTONMASS = 1.67262178e-24
GAMMA = 5.0 / 3.0
GAMMA_MINUS1 = GAMMA - 1.0
MSUN = 1.989e33
MPC = 3.085678e24
G = 6.6738e-8 / MPC * MSUN  # converted in code units (Mpc, 10 Msun, kms)

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}


def truncated_flat_potential(mass, scale_radius, radius):
    pot = -G * mass / scale_radius * log((sqrt(radius * radius + scale_radius * scale_radius) + scale_radius) / radius)
    return pot


def isothermal_profile(temperature, rho_zero, r_zero, rmin, rmax, nbins=200):
    # see Gatto et al. (2013)
    mass = 1.9e2
    scale_radius = 0.170
    
    phi_zero = truncated_flat_potential(mass, scale_radius, r_zero)
    # the gas is fully ionized
    XH = 0.76
    yhelium = (1 - XH) / (4. * XH);
    ne = 1.0 + 2.0 * yhelium
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    csound = temperature * BOLTZMANN / (mu * PROTONMASS * 1.0e10)
    
    radii = rmin + np.arange(nbins + 1) * (rmax - rmin) / nbins
    phi = truncated_flat_potential(mass, scale_radius, radii[:])
    densities = rho_zero * exp(-(phi[:] - phi_zero) / csound)
    
    return radii, densities


def plot_densprofiles_werk(runs, dirs, outpath, snap, suffix, nrows, ncols, nshells=200):
    panels = len(runs)
    
    # plot Werk et al. (2014) sample
    tablename = "./data/werk.txt"
    
    R = np.genfromtxt(tablename, comments='#', usecols=1)
    
    minU = np.genfromtxt(tablename, comments='#', usecols=5)
    maxU = np.genfromtxt(tablename, comments='#', usecols=6)
    
    nH = 1.21e4 / 10 ** (0.5 * (minU + maxU)) / 3e10
    minU = 1.21e4 / 10 ** (minU) / 3e10
    maxU = 1.21e4 / 10 ** (maxU) / 3e10
    
    # correction for He abundance assuming primordial composition
    X = 0.76
    correction = (3 * X + 1) / (4 * X)
    nH *= correction
    minU *= correction
    maxU *= correction
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, aspect_ratio=True)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[1.0, 1.0e3], ylim=[1.0e-6, 1.0e-1], logaxis=True)
    figure.set_axis_locators(xmajloc=4, ymajloc=6, logxaxis=True, logyaxis=True)
    figure.set_axis_labels(xlabel="$\\rm{r\\,[kpc]}$", ylabel="$\\rm{n\\,[cm^{-3}]}$")
    figure.set_fontsize()
    
    save_snap = snap
    
    conversion_fact = 10.0  # to convert from inernal units to M_sun kpc^-3
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax = figure.axes[d]
        
        # if runs[d] == 'Aq-G_5':
        #	snap = 60
        # else:
        #	snap = save_snap
        
        print
        "Doing dir %s. snapshot %d" % (dd, snap)
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0], loadonly=['pos', 'mass', 'u', 'ne', 'gz', 'sfr', 'gmet'])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
        
        s.center = sf.data['fpos'][0, :]
        rad = sf.data['frc2'][0]
        
        print
        'selecting gas particles'
        # this selects the diffuse gas particles and center them
        i, = np.where((s.r() < 2.0 * rad) & (s.type == 0))
        u = s.data['u'][i].astype('float64')
        u *= 1.0e10  # it's a velocity squared to be converted in cgs
        ne = s.data['ne'][i].astype('float64')
        metallicity = s.data['gz'][i].astype('float64')
        sfr = s.data['sfr'][i].astype('float64')
        XH = s.data['gmet'][i, element['H']].astype('float64')
        yhelium = (1 - XH - metallicity) / (4. * XH);
        mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
        temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
        gpos = s.pos[i, :].astype('float64')
        gmass = s.data['mass'][i].astype('float64')
        
        igas, = np.where(sfr <= 0.0)
        gpos = gpos[igas, :]
        gmass = gmass[igas]
        gmu = mu[igas]
        temp = temp[igas]
        gpos[:, 0] -= s.center[0]
        gpos[:, 1] -= s.center[1]
        gpos[:, 2] -= s.center[2]
        gr = np.sqrt(gpos[:, 0] * gpos[:, 0] + gpos[:, 1] * gpos[:, 1] + gpos[:, 2] * gpos[:, 2])
        
        temp_cut = 1.0e5
        i, = np.where(temp < temp_cut)
        gmasscold = gmass[i]
        grcold = gr[i]
        gmucold = gmu[i]
        
        temp_cut_min = 1.0e5
        temp_cut_max = 1.0e6
        i, = np.where((temp < temp_cut_max) & (temp >= temp_cut_min))
        gmasswarm = gmass[i]
        grwarm = gr[i]
        gmuwarm = gmu[i]
        
        temp_cut = 1.0e6
        i, = np.where(temp >= temp_cut)
        gmasshot = gmass[i]
        grhot = gr[i]
        gmuhot = gmu[i]
        
        Rcut = log10(2.0 * rad)
        min_radius = log10(0.001)
        
        print
        'Computing profiles'
        gasdensity = np.zeros(nshells)
        gasdensitycold = np.zeros(nshells)
        gasdensitywarm = np.zeros(nshells)
        gasdensityhot = np.zeros(nshells)
        radius = np.zeros(nshells + 1)
        hradius = np.zeros(nshells)
        
        gasdensity[:], edge = np.histogram(np.log10(gr[:]), bins=nshells, range=(min_radius, Rcut), weights=(gmass[:] / (gmu[:] * PROTONMASS)))
        print
        'Total density done'
        
        gasdensitycold[:], edge = np.histogram(np.log10(grcold[:]), bins=nshells, range=(min_radius, Rcut),
                                               weights=(gmasscold[:] / (gmucold[:] * PROTONMASS)))
        print
        'Cold gas done'
        
        gasdensitywarm[:], edge = np.histogram(np.log10(grwarm[:]), bins=nshells, range=(min_radius, Rcut),
                                               weights=(gmasswarm[:] / (gmuwarm[:] * PROTONMASS)))
        print
        'Warm gas done'
        
        gasdensityhot[:], edge = np.histogram(np.log10(grhot[:]), bins=nshells, range=(min_radius, Rcut),
                                              weights=(gmasshot[:] / (gmuhot[:] * PROTONMASS)))
        print
        'Hot gas done'
        
        hradius[:] = 10 ** (0.5 * (edge[1:] + edge[:-1]))
        radius[:] = 10 ** (edge[:])
        
        dv = 4.0 * pi * (radius[1:] ** 3.0 - radius[:-1] ** 3.0) / 3.0
        
        gasdensity[:] /= dv[:]
        gasdensitycold[:] /= dv[:]
        gasdensitywarm[:] /= dv[:]
        gasdensityhot[:] /= dv[:]
        
        gasdensity[:] *= (1.0e10 * MSUN / MPC ** 3)  # convert in cm^-3
        gasdensitycold[:] *= (1.0e10 * MSUN / MPC ** 3)  # convert in cm^-3
        gasdensitywarm[:] *= (1.0e10 * MSUN / MPC ** 3)  # convert in cm^-3
        gasdensityhot[:] *= (1.0e10 * MSUN / MPC ** 3)  # convert in cm^-3
        
        # r, n = isothermal_profile(1.8e6, 1.75e-4, 0.068, 1.0e-3, 0.5, nbins=300)
        r, nup = isothermal_profile(1.8e6, 3.6e-4, 0.068, 1.0e-3, 0.5, nbins=300)
        r, ndown = isothermal_profile(1.8e6, 1.5e-4, 0.068, 1.0e-3, 0.5, nbins=300)
        
        # for the legend of the fill_between command
        p1 = plt.Rectangle((0, 0), 1, 1, fc="darkgray", edgecolor='none', lw=0.0)
        # l1 = ax.legend([p1], ['Gatto+ 13'], loc='lower left', frameon=False, prop={'size':5}, fontsize=5)
        handles = [p1]
        
        # to scale the DM only results to the same particle mass of the sims with baryons
        mass_factor = 1.0 - 0.04 / s.omega0
        
        ax.fill_between(r * 1.0e3, ndown, nup, color='darkgray', edgecolor='None', label='Gatto+ 13', zorder=1)
        p2, = ax.loglog(hradius * 1e3, gasdensity, '-', lw=1.0, color='k', label='total', zorder=10)
        p3, = ax.loglog(hradius * 1e3, gasdensitycold, '-', lw=1.0, color='b', label='cold gas', zorder=10)
        p4, = ax.loglog(hradius * 1e3, gasdensitywarm, '-', lw=1.0, color='g', label='warm gas', zorder=10)
        p5, = ax.loglog(hradius * 1e3, gasdensityhot, '-', lw=1.0, color='r', label='hot gas', zorder=10)
        
        ylim = figure.ylim
        ax.vlines(rad * 1.0e3, ylim[0], ylim[1], linestyle=':', color='darkgray', lw=0.7, zorder=2)
        ax.vlines(0.1 * rad * 1.0e3, ylim[0], ylim[1], linestyle=':', color='darkgray', lw=0.7, zorder=2)
        
        p6, = ax.loglog(R, nH, linestyle='None', marker='s', markersize=2, mfc='gray', mec='None', zorder=2)
        ax.vlines(R, minU, maxU, colors='gray', zorder=2)
        
        # fit gas component with power law
        # i, = np.where( (hradius > 0.01) & (hradius < 0.1) )
        # (slope, offset) = polyfit( np.log(hradius[i] * 1.0e3), np.log(conversion_fact * gasdensity[i]), deg=1 )
        
        # dashes = [3, 3]
        # l, = ax.loglog( hradius*1e3, np.exp(offset + slope * np.log(hradius * 1.0e3)), '--', color='k', lw=0.3)
        # l.set_dashes(dashes)
        # text( 0.33, 0.23, "$\\rm{r^{%.5s}}$" % slope, size=5, transform=ax.transAxes )
        
        handles = [p1, p2, p3, p4, p5, p6]
        labels = ['Gatto+ 13', 'total', 'cold gas', 'warm gas', 'hot gas', 'Werk+14']
        
        ax.legend(handles, labels, loc='lower left', frameon=False, prop={'size': 5}, numpoints=1)
        
        ax.xaxis.set_major_formatter(UserLogFormatterMathtext())
        # figure.twinyaxes[d].yaxis.set_major_formatter(UserLogFormatterMathtext())
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        
        figure.set_panel_title(panel=d, title="$\\rm{%s-{%s}}$" % (runs[d].split('_')[0], runs[d].split('_')[1]), position='top right')
    
    figure.reset_axis_limits()
    
    figure.fig.savefig(
        "%s/densprofile_multi%03d.%s" % (outpath, snap, suffix))  # figure.fig.savefig( "%s/densprofile_talk%03d.%s" % (outpath,snap,suffix) )