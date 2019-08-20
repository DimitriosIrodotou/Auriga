from const import *
from gadget import *
from gadget_subfind import *
from util import multipanel_layout
from util.label_format import UserLogFormatterMathtext

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
    halfradii = 0.5 * (radii[1:] + radii[:-1])
    phi = truncated_flat_potential(mass, scale_radius, halfradii[:])
    densities = mu * PROTONMASS * rho_zero * exp(-(phi[:] - phi_zero) / csound)
    
    return radii, halfradii, densities


def cumulative_isothermal_mass(temperature, rho_zero, r_zero=0.068, rmin=1.0e-3, rmax=0.5, nbins=500):
    r, hr, n = isothermal_profile(temperature, rho_zero, r_zero, rmin, rmax, nbins)
    
    masses = np.zeros(len(hr))
    masses[:] = 4.0 * pi * n[:] * hr[:] ** 2.0 * (r[1:] - r[:-1])
    
    masses = np.cumsum(masses)
    
    return r[1:], masses[:] * MPC ** 3.0 / MSUN


def plot_massprofiles(runs, dirs, outpath, snap, suffix, nrows, ncols):
    panels = len(runs)
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[1.0, 1.0e3], ylim=[1.0e8, 1.0e12], logaxis=True)
    figure.set_axis_locators(xmajloc=4, ymajloc=5, logxaxis=True, logyaxis=True)
    figure.set_axis_labels(xlabel="$\\rm{r\\,[kpc]}$", ylabel="$\\rm{M(<r)\\,[M_{\\odot}]}$")
    figure.set_fontsize()
    
    save_snap = snap
    
    conversion_fact = 10.0  # to convert from inernal units to M_sun kpc^-3
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax = figure.fig.axes[d]
        
        # if runs[d] == 'Aq-G_5':
        #	snap = 60
        # else:
        #	snap = save_snap
        
        print
        "Doing dir %s. snapshot %d" % (dd, snap)
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1], loadonly=['pos', 'mass', 'u', 'ne', 'sfr', 'gz', 'gmet'])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'smas'])
        
        s.center = sf.data['fpos'][0, :]
        rad = sf.data['frc2'][0]
        virmass = sf.data['smas'][0]
        omegabar = 0.048
        omegamat = s.omega0
        # barmass = 1.0e10 * virmass * omegabar / omegamat
        # print barmass, omegabar / omegamat
        
        print
        'Select gas properties'
        # this selects the diffuse gas particles and center them
        i, = np.where((s.r() < 2.0 * rad) & (s.type == 0))
        u = s.data['u'][i].astype('float64')
        u *= 1.0e10  # it's a velocity squared to be converted in cgs
        ne = s.data['ne'][i].astype('float64')
        sfr = s.data['sfr'][i].astype('float64')
        metallicity = s.data['gz'][i].astype('float64')
        XH = s.data['gmet'][i, element['H']].astype('float64')
        yhelium = (1 - XH - metallicity) / (4. * XH);
        mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
        temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
        gpos = s.pos[i, :].astype('float64')
        gmass = s.data['mass'][i].astype('float64')
        
        # only the diffuse gas is selected
        igas, = np.where((sfr <= 0.0))
        gpos = gpos[igas, :]
        gmass = gmass[igas]
        temp = temp[igas]
        gpos[:, 0] -= s.center[0]
        gpos[:, 1] -= s.center[1]
        gpos[:, 2] -= s.center[2]
        gr = np.sqrt(gpos[:, 0] * gpos[:, 0] + gpos[:, 1] * gpos[:, 1] + gpos[:, 2] * gpos[:, 2])
        
        iradius = np.argsort(gr)
        gr = gr[iradius]
        gmass = gmass[iradius]
        temp = temp[iradius]
        
        i, = np.where(gr < rad)
        barfrac = 1.0e10 * gmass[i].sum()
        
        temp_cut = 1.0e5
        i, = np.where(temp < temp_cut)
        grcold = gr[i]
        gmasscold = gmass[i]
        
        temp_cut_min = 1.0e5
        temp_cut_max = 1.0e6
        i, = np.where((temp < temp_cut_max) & (temp >= temp_cut_min))
        grwarm = gr[i]
        gmasswarm = gmass[i]
        
        temp_cut = 1.0e6
        i, = np.where(temp >= temp_cut)
        grhot = gr[i]
        gmasshot = gmass[i]
        
        print
        'Computing mass profiles'
        # compute cumulative mass
        gmass[:] = np.cumsum(gmass)
        
        # compute cumulative mass
        gmasscold[:] = np.cumsum(gmasscold)
        
        # compute cumulative mass
        gmasswarm[:] = np.cumsum(gmasswarm)
        
        # compute cumulative mass
        gmasshot[:] = np.cumsum(gmasshot)
        
        # compute DM mass
        i, = np.where((s.r() < rad) & (s.type == 1))
        dmmass = s.data['mass'][i].astype('float64')
        dmtotmass = dmmass.sum()
        barmass = 1.0e10 * dmtotmass * omegabar / (omegamat - omegabar)
        
        radius, massesup = cumulative_isothermal_mass(1.8e6, 3.6e-4, 0.068, 1.0e-3, 0.5)
        radius, massesdown = cumulative_isothermal_mass(1.8e6, 1.5e-4, 0.068, 1.0e-3, 0.5)
        
        # for the legend of the fill_between command
        p1 = plt.Rectangle((0, 0), 1, 1, fc="darkgray", edgecolor='none', lw=0.0)
        l1 = ax.legend([p1], ['Gatto+ 13'], loc='upper left', frameon=False, prop={'size': 5}, fontsize=5)
        
        ax.fill_between(radius * 1.0e3, massesdown, massesup, color='darkgray', edgecolor='None')
        ax.loglog(gr * 1e3, gmass * 1.0e10, '-', lw=1.0, color='k', label='total')
        ax.loglog(grcold * 1e3, gmasscold * 1.0e10, '-', lw=1.0, color='b', label='cold gas')
        ax.loglog(grwarm * 1e3, gmasswarm * 1.0e10, '-', lw=1.0, color='g', label='warm gas')
        ax.loglog(grhot * 1e3, gmasshot * 1.0e10, '-', lw=1.0, color='r', label='hot gas')
        # ax.loglog( radius*1e3, masses, '-', color='gray', label='Gatto+ 13')
        
        ax.axhline(y=barfrac, linewidth=0.5, color='gray')
        
        ylim = figure.ylim
        ax.vlines(rad * 1.0e3, ylim[0], ylim[1], linestyle=':', color='darkgray', lw=0.7)
        ax.vlines(0.1 * rad * 1.0e3, ylim[0], ylim[1], linestyle=':', color='darkgray', lw=0.7)
        
        # fit gas component with power law
        # i, = np.where( (hradius > 0.01) & (hradius < 0.1) )
        # (slope, offset) = polyfit( np.log(hradius[i] * 1.0e3), np.log(conversion_fact * gasdensity[i]), deg=1 )
        
        # dashes = [3, 3]
        # l, = ax.loglog( hradius*1e3, np.exp(offset + slope * np.log(hradius * 1.0e3)), '--', color='k', lw=0.3)
        # l.set_dashes(dashes)
        # text( 0.33, 0.23, "$\\rm{r^{%.5s}}$" % slope, size=5, transform=ax.transAxes )
        
        ax.text(2, 0.4 * barfrac, "$\\rm{%.2f\\,\,f_{b}}$" % (barfrac / barmass), size=5, transform=ax.transData)
        
        ax.legend(loc='lower right', frameon=False, prop={'size': 5})
        ax.add_artist(l1)
        # gca().add_artist(l1)
        
        format = UserLogFormatterMathtext()
        
        ax.xaxis.set_major_formatter(format)
        ax.yaxis.set_major_formatter(format)
        
        figure.set_panel_title(panel=d, title="$\\rm{%s-{%s}}$" % (runs[d].split('_')[0], runs[d].split('_')[1]), position='top right')
    
    figure.reset_axis_limits()
    
    figure.fig.savefig("%s/massprofile_multi%03d.%s" % (outpath, snap, suffix), dpi=300)