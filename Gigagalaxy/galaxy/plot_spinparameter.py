import matplotlib.cm as cmx
import matplotlib.colors as colors
from const import *
from loadmodules import *
from util import multipanel_layout

# colors = ['g','b','r','k','c','y','m','purple']
lines = ['-', ':']
marker = ['o', '^', 'd', 's']
toinch = 0.393700787
Gcosmo = 43.0071

col1 = 'SlateBlue'
col2 = 'tomato'

colors = {'halo_1': 'k', 'halo_2': col1, 'halo_3': col1, 'halo_4': 'k', 'halo_5': 'k', 'halo_6': 'k', 'halo_7': 'k', 'halo_8': col1, 'halo_9': 'k',
    'halo_10':      col2, 'halo_11': col2, 'halo_12': 'k', 'halo_13': 'k', 'halo_14': 'k', 'halo_15': 'k', 'halo_16': col1, 'halo_17': 'k',
    'halo_18':      'k', 'halo_19': 'k', 'halo_20': col1, 'halo_21': 'k', 'halo_22': col2, 'halo_23': 'k', 'halo_24': 'k', 'halo_25': col1,
    'halo_26':      col2, 'halo_27': 'k', 'halo_28': col2, 'halo_29': col2, 'halo_30': 'k'}

mrk1 = '^'
mrk2 = 's'

markers = {'halo_1': 'o', 'halo_2': mrk1, 'halo_3': mrk1, 'halo_4': 'o', 'halo_5': 'o', 'halo_6': 'o', 'halo_7': 'o', 'halo_8': mrk1, 'halo_9': 'o',
    'halo_10':       mrk2, 'halo_11': mrk2, 'halo_12': 'o', 'halo_13': 'o', 'halo_14': 'o', 'halo_15': 'o', 'halo_16': mrk1, 'halo_17': 'o',
    'halo_18':       'o', 'halo_19': 'o', 'halo_20': mrk1, 'halo_21': 'o', 'halo_22': mrk2, 'halo_23': 'o', 'halo_24': 'o', 'halo_25': mrk1,
    'halo_26':       mrk2, 'halo_27': 'o', 'halo_28': mrk2, 'halo_29': mrk2, 'halo_30': 'o'}

outliers = ['halo_1', 'halo_11']

dtkinhi = {'halo_1':  0.75, 'halo_2': 0.8, 'halo_3': 0.72, 'halo_4': 0.4, 'halo_5': 0.66, 'halo_6': 0.81, 'halo_7': 0.62, 'halo_8': 0.8,
           'halo_9':  0.67, 'halo_10': 0.75, 'halo_11': 0.23, 'halo_12': 0.73, 'halo_13': 0.7, 'halo_14': 0.64, 'halo_15': 0.72, 'halo_16': 0.87,
           'halo_17': 0.76, 'halo_18': 0.82, 'halo_19': 0.47, 'halo_20': 0.7, 'halo_21': 0.81, 'halo_22': 0.69, 'halo_23': 0.78, 'halo_24': 0.68,
           'halo_25': 0.88, 'halo_26': 0.72, 'halo_27': 0.75, 'halo_28': 0.62, 'halo_29': 0.16, 'halo_30': 0.59}

dtkinlo = {'halo_1':  0.4, 'halo_2': 0.58, 'halo_3': 0.64, 'halo_4': 0.25, 'halo_5': 0.55, 'halo_6': 0.62, 'halo_7': 0.33, 'halo_8': 0.45,
           'halo_9':  0.43, 'halo_10': 0.39, 'halo_11': 0.11, 'halo_12': 0.5, 'halo_13': 0.33, 'halo_14': 0.49, 'halo_15': 0.53, 'halo_16': 0.65,
           'halo_17': 0.39, 'halo_18': 0.53, 'halo_19': 0.39, 'halo_20': 0.29, 'halo_21': 0.58, 'halo_22': 0.39, 'halo_23': 0.51, 'halo_24': 0.48,
           'halo_25': 0.69, 'halo_26': 0.42, 'halo_27': 0.58, 'halo_28': 0.35, 'halo_29': 0.12, 'halo_30': 0.30}


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    
    return map_index_to_rgb_color


def plot_spinparameter(runs, dirs, outpath, snap, ncols=1, nrows=1):
    # fac = 1.0
    # fig = figure(figsize=np.array([16.1*fac,16.1])*toinch*0.5, dpi=300 )
    # ax = axes( [0.17/fac,0.13,0.8/fac,0.8] )
    
    # fig2 = figure(figsize=np.array([16.1*fac,16.1])*toinch*0.5, dpi=300 )
    # ax2 = axes( [0.17/fac,0.13,0.8/fac,0.8] )
    
    nruns = len(runs)
    lencol = len(colors)
    
    panels = nrows * ncols
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(8)
    figure.set_figure_layout()
    figure.set_fontsize(8)
    figure.set_axis_locators(xminloc=0.04, xmajloc=0.04, yminloc=2., ymajloc=2.)
    figure.set_axis_labels(xlabel="$\\rm{\lambda}$", ylabel="$\\rm{R_d}\\,[kpc]}$")
    figure.set_axis_limits_and_aspect(xlim=[0., 0.16], ylim=[0., 12.])
    
    figure2 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure2.set_fontsize(8)
    figure2.set_figure_layout()
    figure2.set_fontsize(8)
    figure2.set_axis_locators(xminloc=0.04, xmajloc=0.04, yminloc=0.2, ymajloc=0.2)
    figure2.set_axis_labels(xlabel="$\\rm{\lambda}$", ylabel="$\\rm{D/T}$")
    figure2.set_axis_limits_and_aspect(xlim=[0., 0.16], ylim=[0., 1.2])
    
    spin = np.zeros(nruns)
    rd = np.zeros(nruns)
    dtrat = np.zeros(nruns)
    
    for d in range(nruns):
        dd = dirs[d] + runs[d]
        ax = figure.axes[0]
        ax2 = figure2.axes[0]
        print("Doing dir %s snap %d." % (dd, snap))
        
        rpath = outpath + runs[d] + '/radprof/'
        file1 = rpath + '/fit_table_%s.txt' % runs[d]
        
        f = open(file1, 'r')
        data = np.loadtxt(f, delimiter=None, skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        f.close()
        
        rd[d] = data[-1, 5]
        dtrat[d] = data[-1, 9]
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonly=['pos', 'vel', 'age', 'pot', 'mass', 'sfr'], loadonlytype=[0, 1, 4])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'fmc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        na = s.nparticlesall
        
        rad = sf.data['frc2'][0]
        galrad = sf.data['frc2'][0]
        # galrad = 0.1
        
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        mass = s.data['mass'].astype('float64')
        
        st = na[:4].sum();
        en = st + na[4]
        age = np.zeros(s.npartall)
        age[st:en] = s.data['age']
        
        iall, = np.where((s.r() > 0.) & (s.r() < galrad))
        # iall, = np.where( (s.r() > 0.) & ( (s.type == 4) | (s.type == 0) | (s.type == 1) ) & (s.r() < galrad) )
        # iall2, = np.where( (s.r() > 0.) & ( (s.type == 0) | (s.type == 1) ) & (s.r() < galrad) )
        
        # nstars = size( istars )
        nall = size(iall)
        rr = pylab.sqrt((s.pos[iall, :] ** 2).sum(axis=1))
        rsort = rr.argsort()
        
        pos = s.pos[iall, :].astype('float64')
        vel = s.vel[iall, :].astype('float64')
        mass = s.data['mass'][iall].astype('float64')
        ptype = s.data['type'][iall]
        radius = pylab.sqrt((s.pos[iall, 1:] ** 2).sum(axis=1))
        age = age[iall]
        pot = s.data['pot'][iall].astype('float64')
        
        j = mass[:, None] * pylab.cross(pos[:, :], vel[:, :])
        
        # js = ( pylab.sqrt( (j**2).sum(axis=1) ) ).sum() / mass[:].sum()
        js = (pylab.sqrt((j[:, 0] ** 2))).sum() / mass[:].sum()
        
        mtot = mass.sum()
        v200 = pylab.sqrt(Gcosmo * mtot / galrad)  # / 1e5
        
        msum = np.sum(mass)
        
        # Calculate spin parameter
        # spin[d] = ( js * (abs(esum)**0.5) ) / (Gcosmo * (msum**1.5) )
        spin[d] = js / (sqrt(2.) * v200 * galrad)
        
        js = j[:, 0].sum() / mass[:].sum()
        spin[d] = js / (sqrt(2.) * v200 * galrad)
        
        ax.plot(spin[d], rd[d], marker=markers[runs[d]], mfc='k', mec=colors[runs[d]], ms=5., mew=1.)
        if runs[d] in outliers:
            ax.plot(spin[d], rd[d], marker='o', mfc='none', mec='g', ms=10., mew=0.7)
        
        ax2.plot(spin[d], dtkinlo[runs[d]], marker=markers[runs[d]], mfc='k', mec=colors[runs[d]], ms=5., mew=1.)
        if runs[d] in outliers:
            ax2.plot(spin[d], dtkinlo[runs[d]], marker='o', mfc='none', mec='g', ms=10., mew=0.7)
        
        # ax2.plot( spin[d], dtrat[d], 'o', color='k', ms=4., label="$\\rm{%s\, %s}$" % ('Au', runs[d].split('_')[1]) )  # ax2.plot( spin[  # d],
        # dtkinhi[runs[d]], '^', color='b', ms=4., label="$\\rm{%s\, %s}$" % ('Au', runs[d].split('_')[1]) )  # ax2.plot( spin[d],
        # dtkinlo[runs[d]], 's', color='r', ms=4., label="$\\rm{%s\, %s}$" % ('Au', runs[d].split('_')[1]) )  # ax2.plot( spin[d],
        # 0.5*(dtkinlo[runs[d]]+dtkinhi[runs[d]]), 'd', color='g', ms=4., label="$\\rm{%s\, %s}$" % ('Au', runs[d].split('_')[1]) )
    
    # ax.set_xlim( 0.08, 0.21 )
    # ax.set_ylim( 0., 12. )
    # ax.set_xlabel( "$\\rm{\lambda}$", fontsize=10 )
    # ax.set_ylabel( "$\\rm{R_d}\\,[kpc]}$", fontsize=10 )
    # ax.xaxis.set_tick_params(labelsize=7)
    # ax.yaxis.set_tick_params(labelsize=7)
    
    # ax.legend(loc='upper right', fontsize=5, frameon=False, numpoints=1, scatterpoints=1, ncol=5)
    
    figure.fig.savefig('%s/plotall/spin_vs_discsize_new.pdf' % (outpath))
    
    # ax2.scatter( spin, dtrat, marker[int(d / lencol)], color=colors[d % lencol], ms=4., label="$\\rm{%s\, %s}$" % ('Au', runs[d].split('_')[
    # 1]) )
    
    # ax2.set_xlim( 0.08, 0.21 )
    # ax2.set_ylim( 0., 1.2 )
    # ax2.set_xlabel( "$\\rm{\lambda}$", fontsize=10 )
    # ax2.set_ylabel( "$\\rm{D/T}$", fontsize=10 )
    # ax2.xaxis.set_tick_params(labelsize=7)
    # ax2.yaxis.set_tick_params(labelsize=7)
    
    # figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ("Au", runs[d].split('_')[1]), position='bottom right' )
    
    # ax2.legend(loc='upper left', fontsize=5, frameon=False, numpoints=1, scatterpoints=1, ncol=5)
    
    figure2.fig.savefig('%s/plotall/spin_vs_dtratio_new.pdf' % (outpath))