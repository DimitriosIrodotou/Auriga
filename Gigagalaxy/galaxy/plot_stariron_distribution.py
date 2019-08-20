from const import *
from gadget import *
from gadget_subfind import *
from util import multipanel_layout

toinch = 0.393700787
Gcosmo = 43.0071
ZSUN = 0.0127

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}

# from Asplund et al. (2009) Table 5
SUNABUNDANCES = {'H': 12.0, 'He': 10.98, 'C': 8.47, 'N': 7.87, 'O': 8.73, 'Ne': 7.97, 'Mg': 7.64, 'Si': 7.55, 'Fe': 7.54}


def plot_starmetal_distribution_data(ax, normalize_bins, norm_factor=1.0):
    tablename = "data/Holmberg.txt"
    iron = np.genfromtxt(tablename, comments='#', usecols=0)
    weight = np.ones(len(iron))
    ydatatot, edges = histogram(iron, bins=100, weights=weight, range=[-2, 2])
    xdatatot = 0.5 * (edges[1:] + edges[:-1])
    
    # such that \int f(\epsilon) d\epsilon = 1
    if normalize_bins:
        norm = len(iron)
        binwidth = edges[1:] - edges[:-1]
        ydatatot[:] /= norm
        ydatatot[:] /= binwidth[:]
    
    # plot( xdatatot, ydatatot, 'b-', lw=0.7, label='Holmberg+' )
    # ax.plot( xdatatot, ydatatot * norm_factor, 'g--', lw=0.7, label='Holmberg+' )
    ax.plot(xdatatot, ydatatot * norm_factor, 'g--', lw=0.7)


# legend( loc='upper right', frameon=False )

def plot_starmetal_distribution(runs, dirs, outpath, snap, suffix, nrows, ncols, normalize_bins=True, SN=False):
    panels = len(runs)
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[-2.0, 2.0], ylim=[0.0, 2.5], logaxis=False)
    
    if normalize_bins:
        figure.set_axis_locators(xminloc=0.25, xmajloc=1.0, yminloc=0.1, ymajloc=0.5)
    else:
        figure.set_axis_locators(xminloc=0.25, xmajloc=1.0, yminloc=0.01, ymajloc=0.05)
    
    figure.set_axis_labels(xlabel="$\\rm{[Fe/H]}$", ylabel="$\\rm{f([Fe/H])}$")
    figure.set_fontsize()
    
    save_snap = snap
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        # wpath = outpath+runs[0]+'/mdf/'
        wpath = outpath + '/plotall/'
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        
        ax = figure.fig.axes[d]
        
        rup_sn = 0.009
        rlo_sn = 0.007
        print
        "Doing dir %s snap %d." % (dd, snap)
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonly=['pos', 'vel', 'mass', 'age', 'gmet', 'pot'])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        na = s.nparticlesall
        
        galrad = 0.1 * sf.data['frc2'][0]
        print
        'galrad:', galrad
        s.select_halo(sf, use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        mass = s.data['mass'].astype('float64')
        
        st = na[:4].sum();
        en = st + na[4]
        age = np.zeros(s.npartall)
        metallicity = np.zeros([s.npartall, 2])
        age[st:en] = s.data['age']
        metallicity[st:en, 0] = s.data['gmet'][na[0]:, element['Fe']]
        metallicity[st:en, 1] = s.data['gmet'][na[0]:, element['H']]
        
        iall, = np.where((s.r() < galrad) & (s.r() > 0.))
        istars, = np.where((s.r() < galrad) & (s.r() > 0.) & (s.type == 4) & (age > 0.))
        
        nstars = size(istars)
        nall = size(iall)
        rr = pylab.sqrt((s.pos[iall, :] ** 2).sum(axis=1))
        msort = rr.argsort()
        mass_all = mass[iall]
        
        msum = pylab.zeros(nall)
        msum[msort[:]] = np.cumsum(mass_all[msort[:]])
        
        pos = s.pos[iall, :].astype('float64')
        vel = s.vel[iall, :].astype('float64')
        mass = s.data['mass'][iall].astype('float64')
        ptype = s.data['type'][iall]
        pot = s.data['pot'][iall].astype('float64')
        radius = pylab.sqrt((s.pos[iall, 1:] ** 2).sum(axis=1))
        age = age[iall]
        metallicity = metallicity[iall, :]
        
        star_radius = np.zeros(nstars)
        star_age = np.zeros(nstars)
        eps = pylab.zeros(nstars)
        eps2 = pylab.zeros(nstars)
        smass = pylab.zeros(nstars)
        jcmax = pylab.zeros(nstars)
        spec_energy = pylab.zeros(nstars)
        star_metallicity = pylab.zeros([nstars, 2])
        
        print
        'Compute stellar properties'
        
        nn, = np.where((ptype[:] == 4) & (age[:] > 0.))
        star_radius[:] = radius[nn]
        star_age[:] = s.cosmology_get_lookback_time_from_a(age[nn], is_flat=True)
        print
        "starage=", star_age
        j = pylab.cross(pos[nn, :], vel[nn, :])
        jc = radius[nn] * pylab.sqrt(Gcosmo * msum[nn] / radius[nn])
        jz = j[:, 0]
        
        spec_energy[:] = 0.5 * (vel[nn, :] ** 2).sum(axis=1) + pot[nn]
        eps[:] = jz / jc
        eps2[:] = jz
        smass[:] = mass[nn]
        star_metallicity[:, :] = metallicity[nn, :]
        
        # sort particle by specific energy
        iensort = np.argsort(spec_energy)
        eps = eps[iensort]
        eps2 = eps2[iensort]
        spec_energy = spec_energy[iensort]
        smass = smass[iensort]
        star_metallicity = star_metallicity[iensort, :]
        star_radius = star_radius[iensort]
        star_age = star_age[iensort]
        
        for nn in range(nstars):
            nn0 = nn - 50
            nn1 = nn + 50
            
            if nn0 < 0:
                nn1 += -nn0
                nn0 = 0
            if nn1 >= nstars:
                nn0 -= (nn1 - (nstars - 1))
                nn1 = nstars - 1
            
            jcmax[nn] = np.max(eps2[nn0:nn1])
        
        smass /= smass.sum()
        eps2[:] /= jcmax[:]
        
        print
        'original values'
        print
        'iron abundances: max', max(star_metallicity[:, 0]), 'min', min(star_metallicity[:, 0])
        print
        'hydrogen abundances: max', max(star_metallicity[:, 1]), 'min', min(star_metallicity[:, 1])
        
        # fix negative metallicities, if any, to very small value
        k, = np.where(star_metallicity[:, 0] <= 0.0)
        star_metallicity[k, 0] = 1.0e-40
        k, = np.where(star_metallicity[:, 1] <= 0.0)
        star_metallicity[k, 1] = 1.0e-40
        
        print
        'metallicity fix applied'
        print
        'iron abundances: max', max(star_metallicity[:, 0]), 'min', min(star_metallicity[:, 0])
        print
        'hydrogen abundances: max', max(star_metallicity[:, 1]), 'min', min(star_metallicity[:, 1])
        print
        
        ironabundance = log10(star_metallicity[:, 0] / star_metallicity[:, 1] / 56.)
        ironabundance -= (SUNABUNDANCES['Fe'] - SUNABUNDANCES['H'])
        
        # compute total new (it is used for bulge disk decomposition and plotted afterwards)
        ydatatot, edges = histogram(ironabundance, bins=200, weights=smass, range=[-2, 2])
        xdatatot = 0.5 * (edges[1:] + edges[:-1])
        
        # such that \int f(\epsilon) d\epsilon = 1
        if normalize_bins:
            binwidth = edges[1:] - edges[:-1]
            ydatatot[:] /= binwidth[:]
        
        if SN == False:
            # disk part
            jj, = np.where((eps2 > 0.7) & (eps2 < 1.7))
            ydatadisk, edges = histogram(ironabundance[jj], bins=200, weights=smass[jj], range=[-2, 2])
            xdatadisk = 0.5 * (edges[1:] + edges[:-1])
            # such that \int f(\epsilon) d\epsilon = 1
            if normalize_bins:
                binwidth = edges[1:] - edges[:-1]
                ydatadisk[:] /= binwidth[:]
            
            # bulge part
            kk, = np.where((eps2 > -1.7) & (eps2 <= 0.7))
            ydatabulge, edges = histogram(ironabundance[kk], bins=200, weights=smass[kk], range=[-2, 2])
            xdatabulge = 0.5 * (edges[1:] + edges[:-1])
            # such that \int f(\epsilon) d\epsilon = 1
            if normalize_bins:
                binwidth = edges[1:] - edges[:-1]
                ydatabulge[:] /= binwidth[:]
            
            ax.fill(xdatabulge, ydatabulge, fc='r', alpha=0.5, lw=0, fill=True, label='spheroid')
            ax.fill(xdatadisk, ydatadisk, fc='b', alpha=0.5, lw=0, fill=True, label='disc')
            
            ax.plot(xdatatot, ydatatot, 'k-', lw=1.0, label='total')
            ax.legend(loc='upper left', frameon=False, prop={'size': 5})
        
        elif SN == True:
            # rup_sn = 0.025
            # rlo_sn = 0.0
            ii1, = np.where((star_radius < rup_sn) & (star_radius > rlo_sn) & (star_age < 3.0) & (star_age > 0.0) & (eps2 > 0.7))
            ii2, = np.where((star_radius < rup_sn) & (star_radius > rlo_sn) & (star_age < 8.0) & (star_age > 3.0) & (eps2 > 0.7))
            ii3, = np.where((star_radius < rup_sn) & (star_radius > rlo_sn) & (star_age > 8.0) & (eps2 > 0.7))
            ii4, = np.where((star_radius < rup_sn) & (star_radius > rlo_sn) & (eps2 > 0.7))
            
            ydata1, edges = np.histogram(ironabundance[ii1], bins=40, range=[-2, 2], normed=False)
            ydata2, edges = np.histogram(ironabundance[ii2], bins=40, range=[-2, 2], normed=False)
            ydata3, edges = np.histogram(ironabundance[ii3], bins=40, range=[-2, 2], normed=False)
            ydata4, edges = np.histogram(ironabundance[ii4], bins=40, range=[-2, 2], normed=False)
            xdata = 0.5 * (edges[1:] + edges[:-1])
            print
            "xdata, ydata1, ydata4=", xdata, ydata1, ydata4
            yngfrac = float(np.sum(ydata1)) / float(np.sum(ydata4))
            intfrac = float(np.sum(ydata2)) / float(np.sum(ydata4))
            oldfrac = float(np.sum(ydata3)) / float(np.sum(ydata4))
            
            print
            "fractions=", yngfrac, intfrac, oldfrac
            
            ax.hist(ironabundance[ii1], 40, ec='b', fc='none', lw=0.5, histtype='step', normed=1, range=[-2, 2], label='frac = %.2f' % (yngfrac))
            ax.hist(ironabundance[ii2], 40, ec='c', fc='none', lw=0.5, histtype='step', normed=1, range=[-2, 2], label='frac = %.2f' % (intfrac))
            ax.hist(ironabundance[ii3], 40, ec='r', fc='none', lw=0.5, histtype='step', normed=1, range=[-2, 2], label='frac = %.2f' % (oldfrac))
            ax.hist(ironabundance[ii4], 40, ec='k', fc='none', lw=0.5, histtype='step', normed=1, range=[-2, 2], label='')
            ax.legend(loc='upper left', frameon=False, prop={'size': 4}, markerscale=0.5)
        
        figure.set_panel_title(panel=d, title="$\\rm{%s-{%s}}$" % (runs[d].split('_')[0], runs[d].split('_')[1]), position='bottom right',
                               fontsize=5.)
    
    # plot_starmetal_distribution_data( ax, normalize_bins, norm_factor=smass[kk].sum() )
    
    figure.reset_axis_limits()
    
    if SN == False:
        figure.fig.savefig('%s/starirondistribution%03d.%s' % (wpath, snap, suffix), dpi=300)
    elif SN == True:
        figure.fig.savefig('%s/starirondistribution_SN-agedecomp%03d.%s' % (wpath, snap, suffix), dpi=300)