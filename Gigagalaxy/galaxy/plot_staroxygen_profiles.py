from const import *
from loadmodules import *
from pylab import *
from util import *

# from Asplund et al. (2009) Table 5
SUNABUNDANCES = {'H': 12.0, 'He': 10.98, 'C': 8.47, 'N': 7.87, 'O': 8.73, 'Ne': 7.97, 'Mg': 7.64, 'Si': 7.55, 'Fe': 7.54}


def plot_staroxygen_profiles(runs, dirs, outpath, outputlistfile, redshift, suffix, nrows, ncols, nshells=35, reltosun=False):
    panels = len(runs)
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[0.0, 24.0], ylim=[7.0, 10.0], logaxis=False)
    figure.set_axis_locators(xminloc=1.0, xmajloc=5.0, yminloc=0.1, ymajloc=0.5)
    if reltosun:
        figure.set_axis_labels(xlabel="$\\rm{R [kpc]}$", ylabel="$\\rm{[O/H]}$")
    else:
        figure.set_axis_labels(xlabel="$\\rm{R [kpc]}$", ylabel="$\\rm{12 + \\log(O/H)}$")
    
    figure.set_fontsize()
    
    p = plot_helper.plot_helper()
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax = figure.fig.axes[d]
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], redshift))
        
        print("Doing dir %s snap %d." % (dd, snap))
        
        attrs = ['pos', 'vel', 'mass', 'age', 'gmet', 'pot']
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonly=attrs, loadonlytype=[0, 1, 4, 5])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        galrad = 0.1 * sf.data['frc2'][0]
        Rcut = 0.025  # radial cut 25 kpc
        
        g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
        g.prep_data()
        sdata = g.sgdata['sdata']
        
        smass = sdata['mass']
        age = sdata['age']
        oxygen = sdata['O']
        eps = sdata['eps2']
        pos = sdata['pos']
        
        star_radius = np.sqrt((pos[:, 1:] ** 2).sum(axis=1))
        
        oxygenabundance, edge = np.histogram(star_radius, bins=nshells, range=(0.0, Rcut), weights=(smass * oxygen))
        mass_bin, edge = np.histogram(star_radius, bins=nshells, range=(0.0, Rcut), weights=smass)
        oxygenabundance /= mass_bin
        
        ii, = np.where(eps > 0.7)
        oxygenabundance_disk, edge = np.histogram(star_radius[ii], bins=nshells, range=(0.0, Rcut), weights=(smass[ii] * oxygen[ii]))
        mass_bin, edge = np.histogram(star_radius[ii], bins=nshells, range=(0.0, Rcut), weights=smass[ii])
        oxygenabundance_disk /= mass_bin
        
        ii, = np.where(eps <= 0.7)
        oxygenabundance_spher, edge = np.histogram(star_radius[ii], bins=nshells, range=(0.0, Rcut), weights=(smass[ii] * oxygen[ii]))
        mass_bin, edge = np.histogram(star_radius[ii], bins=nshells, range=(0.0, Rcut), weights=smass[ii])
        oxygenabundance_spher /= mass_bin
        
        pradius = np.zeros(len(edge) - 1)
        pradius[:] = 0.5 * (edge[1:] + edge[:-1]) * 1e3
        
        if not reltosun:
            oxygenabundance += (SUNABUNDANCES['O'] - SUNABUNDANCES['H'])
            oxygenabundance_disk += (SUNABUNDANCES['O'] - SUNABUNDANCES['H'])
            oxygenabundance_spher += (SUNABUNDANCES['O'] - SUNABUNDANCES['H'])
            oxygenabundance += 12.
            oxygenabundance_disk += 12.
            oxygenabundance_spher += 12.
        
        # plot Rudolph et al. (2006) sample
        tablename = "./data/Rudolph_O.txt"
        radius = np.genfromtxt(tablename, comments='#', usecols=1)
        abundance_R = np.genfromtxt(tablename, comments='#', usecols=2)
        obs = np.genfromtxt(tablename, comments='#', usecols=5)
        # select only FIR observations
        k, = np.where(obs == 0)
        radius_FIR = radius[k]
        abundance_FIR = abundance_R[k]
        ax.plot(radius_FIR, abundance_FIR, '^', mfc='lightgray', ms=1.5, mec='None')
        # select only OPT observations
        k, = np.where(obs == 1)
        radius_OPT = radius[k]
        abundance_OPT = abundance_R[k]
        ax.plot(radius_OPT, abundance_OPT, 'o', mfc='lightgray', ms=1.5, mec='None')
        
        radius = arange(0.0, 25.)
        ax.plot(radius, p.bestfit_rudolph_FIR(radius), ls='-', color='darkgray', lw=1.0, label='Rudolph+06')
        ax.plot(radius, p.bestfit_rudolph_OPT(radius), ls='--', color='darkgray', lw=1.0)
        
        ax.plot(pradius, oxygenabundance, 'k-', lw=1.0, label='total')
        ax.plot(pradius, oxygenabundance_disk, 'b-', lw=1.0, label='disc')
        ax.plot(pradius, oxygenabundance_spher, 'r-', lw=1.0, label='spheroid')
        
        limits = figure.ylim
        ax.vlines(galrad * 1.0e3, limits[0], limits[1], linestyle=':', color='darkgray', lw=0.7)
        
        ax.legend(loc='upper right', frameon=False, prop={'size': 5}, numpoints=1)
        
        figure.set_panel_title(panel=d, title="$\\rm{%s-{%s}}$" % (runs[d].split('_')[0], runs[d].split('_')[1]), position='bottom left')
    
    figure.reset_axis_limits()
    
    if reltosun:
        figure.fig.savefig('%s/staroxygenprofile%03d.%s' % (outpath, snap, suffix))
    else:
        figure.fig.savefig('%s/staroxygenprofilespec%03d.%s' % (outpath, snap, suffix))