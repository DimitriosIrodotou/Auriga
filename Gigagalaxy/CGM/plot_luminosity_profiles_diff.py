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
ZSUN = 0.0127
Gcosmo = 43.0071
DENS = 1.0e10 * MSUN / MPC ** 3.0
YR = 365. * 24 * 3600
GYR = 1.0e9 * 365 * 24 * 3600

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}


def plot_luminosity_profiles_diff_decomp(runs, dirs, outpath, snap, suffix, nrows, ncols, nshells=35):
    panels = len(runs)
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[1.0, 1.0e3], ylim=[1.0e38, 1.0e45], logaxis=True)
    figure.set_axis_locators(xmajloc=4, ymajloc=8, logxaxis=True, logyaxis=True)
    figure.set_axis_labels(xlabel="$\\rm{r\\,[kpc]}$", ylabel="$\\rm{dL/dlog\\,r\\,[erg\\,s^{-1}\\,dex^{-1}]}$")
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
                            loadonly=['pos', 'mass', 'rho', 'u', 'sfr', 'ne', 'gz', 'gmet', 'gcol', 'sfr'])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'fmc2', 'svdi'])
        
        s.center = sf.data['fpos'][0, :]
        rad = sf.data['frc2'][0]
        m200 = sf.data['fmc2'][0]
        v200 = Gcosmo * m200 / rad
        # trying to compute Tvir
        vdisp = sf.data['svdi'][0]
        # assuming total ionisation and X = 0.76, Y = 0.24, Z = 0
        yhelium = (1.0 - 0.76) / (4.0 * 0.76)
        mu = (1 + 4 * yhelium) / (2 + 3 * yhelium)
        print
        np.sqrt(v200), vdisp
        tvir2 = 1.0e10 * vdisp * vdisp / BOLTZMANN * mu * PROTONMASS
        tvir = 0.5 * 1.0e10 * v200 / BOLTZMANN * mu * PROTONMASS
        
        print
        'Select gas particles'
        # this selects the diffuse gas particles and center them
        i, = np.where((s.r() < 2.0 * rad) & (s.type == 0))
        u = s.data['u'][i].astype('float64')
        u *= 1.0e10  # it's a velocity squared to be converted in cgs
        sfr = s.data['sfr'][i].astype('float64')
        ne = s.data['ne'][i].astype('float64')
        metallicity = s.data['gz'][i].astype('float64')
        XH = s.data['gmet'][i, element['H']].astype('float64')
        yhelium = (1 - XH - metallicity) / (4. * XH);
        mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
        temp = GAMMA_MINUS1 / BOLTZMANN * u * PROTONMASS * mu;
        gpos = s.pos[i, :].astype('float64')
        gmass = s.data['mass'][i].astype('float64')
        gcoolrate = s.data['gcol'][i].astype('float64')
        gdens = s.data['rho'][i].astype('float64') * DENS
        SNerg = 1.095 * 1.73e49 * s.data['sfr'][i].astype('float64') / YR  # in M_sun /s CHECK THIS WITH THE NEW SN POWER!!!
        gpos[:, 0] -= s.center[0]
        gpos[:, 1] -= s.center[1]
        gpos[:, 2] -= s.center[2]
        SNradius = np.sqrt(gpos[:, 0] * gpos[:, 0] + gpos[:, 1] * gpos[:, 1] + gpos[:, 2] * gpos[:, 2])
        
        igas, = np.where(sfr <= 0.0)
        gpos = gpos[igas, :]
        gmass = gmass[igas] * MSUN * 1.0e10
        temp = temp[igas]
        gu = u[igas]
        gcoolrate = gcoolrate[igas]
        gdens = gdens[igas]
        XH = XH[igas]
        gcoolrate[:] = gcoolrate[:] / gdens[:] * (XH[:] * gdens[:] / PROTONMASS) ** 2.0
        gr = np.sqrt(gpos[:, 0] * gpos[:, 0] + gpos[:, 1] * gpos[:, 1] + gpos[:, 2] * gpos[:, 2])
        
        temp_cut = 1.0e5
        i, = np.where(temp < temp_cut)
        gmasscold = gmass[i]
        gcoolratecold = gcoolrate[i]
        grcold = gr[i]
        
        temp_cut_min = 1.0e5
        temp_cut_max = 1.0e6
        i, = np.where((temp < temp_cut_max) & (temp >= temp_cut_min))
        gmasswarm = gmass[i]
        gcoolratewarm = gcoolrate[i]
        grwarm = gr[i]
        
        temp_cut = 1.0e6
        i, = np.where(temp >= temp_cut)
        gmasshot = gmass[i]
        gcoolratehot = gcoolrate[i]
        grhot = gr[i]
        
        print
        'Computing histograms'
        Rcut = log10(2.0 * rad)
        min_radius = log10(0.001)
        dr = (Rcut - min_radius) / nshells
        
        gaslum = np.zeros(nshells)
        gaslumcold = np.zeros(nshells)
        gaslumwarm = np.zeros(nshells)
        gaslumhot = np.zeros(nshells)
        SNlum = np.zeros(nshells)
        hradius = np.zeros(nshells)
        
        SNlum[:], edge = np.histogram(np.log10(SNradius[:]), bins=nshells, range=(min_radius, Rcut), weights=SNerg[:])
        
        ind, = np.where(gcoolrate < 0.0)
        gaslum[:], edge = np.histogram(np.log10(gr[ind]), bins=nshells, range=(min_radius, Rcut), weights=-(gmass[ind] * gcoolrate[ind]))
        
        ind, = np.where(gcoolratecold < 0.0)
        gaslumcold[:], edge = np.histogram(np.log10(grcold[ind]), bins=nshells, range=(min_radius, Rcut),
                                           weights=-(gmasscold[ind] * gcoolratecold[ind]))
        
        ind, = np.where(gcoolratewarm < 0.0)
        gaslumwarm[:], edge = np.histogram(np.log10(grwarm[ind]), bins=nshells, range=(min_radius, Rcut),
                                           weights=-(gmasswarm[ind] * gcoolratewarm[ind]))
        
        ind, = np.where(gcoolratehot < 0.0)
        gaslumhot[:], edge = np.histogram(np.log10(grhot[ind]), bins=nshells, range=(min_radius, Rcut), weights=-(gmasshot[ind] * gcoolratehot[ind]))
        
        hradius[:] = 10 ** (0.5 * (edge[1:] + edge[:-1]))
        
        gaslum[:] /= dr
        SNlum[:] /= dr
        gaslumcold[:] /= dr
        gaslumwarm[:] /= dr
        gaslumhot[:] /= dr
        
        ax.loglog(hradius * 1e3, gaslum, '-', lw=1.0, color='k')
        ax.loglog(hradius * 1e3, gaslumcold, '-', lw=1.0, color='b')
        ax.loglog(hradius * 1e3, gaslumwarm, '-', lw=1.0, color='g')
        
        labels = ['total', 'cold gas', 'warm gas']
        l1 = ax.legend(labels, loc='upper left', frameon=False, prop={'size': 5})
        
        ax.loglog(hradius * 1e3, gaslumhot, '-', color='r', label='hot gas')
        ax.loglog(hradius * 1e3, SNlum, '-', color='gray', label='SN erg')
        
        ax.legend(loc='lower left', frameon=False, prop={'size': 5})
        ax.add_artist(l1)
        
        ylim = figure.ylim
        ax.vlines(rad * 1.0e3, ylim[0], ylim[1], linestyle=':', color='darkgray', lw=0.7)
        ax.vlines(0.1 * rad * 1.0e3, ylim[0], ylim[1], linestyle=':', color='darkgray', lw=0.7)
        
        format = UserLogFormatterMathtext()
        ax.xaxis.set_major_formatter(format)
        
        figure.set_panel_title(panel=d, title="$\\rm{%s-{%s}}$" % (runs[d].split('_')[0], runs[d].split('_')[1]), position='top right')
    
    figure.reset_axis_limits()
    
    figure.fig.savefig("%s/luminosityprofile_diff_decomp%03d.%s" % (outpath, snap, suffix), dpi=300)