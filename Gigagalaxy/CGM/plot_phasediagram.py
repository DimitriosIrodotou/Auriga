import matplotlib as mpl
from const import *
from gadget import *
from gadget_subfind import *
from matplotlib.colors import LogNorm
from util import multipanel_layout

mpl.rcParams['image.cmap'] = 'gray_r'

toinch = 0.393700787
BOLTZMANN = 1.38065e-16
PROTONMASS = 1.67262178e-24
GAMMA = 5.0 / 3.0
GAMMA_MINUS1 = GAMMA - 1.0
MSUN = 1.989e33
MPC = 3.085678e24
ZSUN = 0.0127
Gcosmo = 43.0071
# ZSUN = 0.02

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}


def plot_phasediagram(runs, dirs, outpath, snap, suffix, nrows, ncols):
    panels = len(runs)
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[-6.0, 0.0], ylim=[3.0, 7.0], logaxis=False)
    figure.set_axis_locators(xminloc=0.25, yminloc=0.1)
    figure.set_axis_labels(ylabel="$\\rm{log\\,T\\,[K]}$", xlabel="$\\rm{log\\,n\\,[cm^{-3}]}$")
    figure.set_fontsize()
    
    save_snap = snap
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax = figure.fig.axes[d]
        
        # if runs[d] == 'Aq-G_5':
        #	snap = 60
        # else:
        #	snap = save_snap
        
        print
        "Doing dir %s. snapshot %d" % (dd, snap)
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0],
                            loadonly=['pos', 'mass', 'u', 'ne', 'rho', 'sfr', 'gz', 'gmet'])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
        
        s.center = sf.data['fpos'][0, :]
        rad = sf.data['frc2'][0]
        
        print
        'Select gas cells'
        # this selects the diffuse gas particles and center them
        i, = np.where((s.r() < 2.0 * rad) & (s.type == 0))
        u = s.data['u'][i].astype('float64')
        u *= 1.0e10  # it's a velocity squared to be converted in cgs
        ne = s.data['ne'][i].astype('float64')
        # mass = s.data['mass'][i].astype('float64') * 1.0e10
        mass = s.data['mass'][i].astype('float64')
        sfr = s.data['sfr'][i].astype('float64')
        dens = s.data['rho'][i].astype('float64') * 1.0e10 * MSUN / MPC ** 3.0
        metallicity = s.data['gz'][i].astype('float64')
        XH = s.data['gmet'][i, element['H']].astype('float64')
        yhelium = (1 - XH - metallicity) / (4. * XH);
        mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
        temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
        dens /= (mu * PROTONMASS)
        
        print
        'Computing histogram'
        igas, = np.where(sfr <= 0.0)
        gasmass = mass[igas]
        gasdensity = log10(dens[igas])
        gastemperature = log10(temp[igas])
        
        ax.hist2d(gasdensity, gastemperature, bins=(140, 80), range=([-6., 1.0], [3., 7.]), weights=gasmass, normed=False, rasterized=True,
                  norm=LogNorm())
        
        figure.set_panel_title(panel=d, title="$\\rm{%s-{%s}}$" % (runs[d].split('_')[0], runs[d].split('_')[1]), position='bottom left')
    
    figure.reset_axis_limits()
    
    figure.fig.savefig("%s/phasediagram_multi%03d.%s" % (outpath, snap, suffix), dpi=300)