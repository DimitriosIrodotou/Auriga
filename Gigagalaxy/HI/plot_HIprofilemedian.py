import multipanel_layout
from HIutils import *
from const import *
from gadget import *
from gadget_subfind import *
from pylab import *

toinch = 0.393700787

PROTONMASS = 1.67262178e-24
MSUN = 1.989e33
MPC = 3.085678e24
GAMMA = 5. / 3.
GAMMA_MINUS1 = GAMMA - 1
HYDROGEN_MASSFRAC = 0.76
BOLTZMANN = 1.38065e-16
ZSUN = 0.0127
GRAVITY = 6.6738e-8

MSUNPCSQ = 1.0e12 * MSUN / (PROTONMASS * MPC ** 2)

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}


def get_bluediskprof():
    file = 'Bluediskradprof.txt'
    id = np.loadtxt(file, usecols=[0])
    totrad = np.loadtxt(file, usecols=[1])
    totsigma = np.loadtxt(file, usecols=[2])
    
    file = 'Bluediskdataset.txt'
    idcheck = np.loadtxt(file, usecols=[0])
    
    R_HI = np.zeros(idcheck.shape)
    intRad = np.linspace(0.0, 2.0, 21)
    
    sigmatab = np.zeros((len(idcheck), len(intRad)))
    
    for d in range(len(idcheck)):
        j, = np.where(id == idcheck[d])
        
        if len(j) == 0:
            continue
        
        rad = totrad[j]
        sigma = totsigma[j]
        
        index = np.max(np.where(sigma > 1.0))
        # linear interpolation to find R_HI
        R_HI[d] = rad[index] + (1.0 - sigma[index]) * (rad[index + 1] - rad[index]) / (sigma[index + 1] - sigma[index])
        
        # ax.plot( rad/R_HI[d], sigma, linestyle='solid', color='lightgray', lw=1.5 )
        
        sigmatab[d, :] = np.interp(intRad, rad / R_HI[d], sigma)
    
    sigmamed = np.maximum(np.median(sigmatab, axis=0), 1.0e-8)
    sigmalow = np.maximum(np.percentile(sigmatab, 16, axis=0), 1.0e-8)
    sigmahigh = np.maximum(np.percentile(sigmatab, 84, axis=0), 1.0e-8)
    
    return sigmamed, sigmalow, sigmahigh


def plot_HIprofile(runs, dirs, outpath, snap, suffix, nshells=120, zmax=5.0e-3):
    figure = multipanel_layout.multipanel_layout(nrows=1, ncols=1, npanels=1, hspace=0.07, wspace=0.07, left=0.15, right=0.95, bottom=0.14, top=0.95,
                                                 scale=1.7)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[0.0, 2.0], ylim=[0.05, 15.0])
    figure.set_fontsize(8.)
    figure.set_axis_labels(xlabel="$\\rm{R/R_{HI}}$", ylabel="$\\rm{\\Sigma_{HI}\\,[M_\\odot\\,pc^{-2}]}$")
    
    R_HI = np.zeros(len(runs))
    R_HI_GK = np.zeros(len(runs))
    intRad = np.linspace(0.0, 2.0, 21)
    
    sigmatab = np.zeros((len(R_HI), len(intRad)))
    sigmatab_GK = np.zeros((len(R_HI_GK), len(intRad)))
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        ax = figure.axes[0]
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4],
                            loadonly=['pos', 'vel', 'mass', 'gmet', 'nh', 'u', 'rho', 'sfr', 'vol', 'gz'], forcesingleprec=True)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True)
        
        print
        'calc ind'
        s.calc_sf_indizes(sf, halolist=[0], dosubhalos=True)
        print
        'done'
        
        print
        'selecting halo'
        s.select_halo(sf, use_principal_axis=False, use_cold_gas_spin=True, do_rotation=True)
        print
        'done'
        
        s.pos = s.pos[:s.nparticlesall[0], :]
        s.mass = s.mass[:s.nparticlesall[0]]
        
        s.data['pos'] = s.pos
        s.data['mass'] = s.mass
        
        XH = s.data['gmet'][:s.nparticlesall[0], element['H']].astype('float64')
        NH = s.data['nh'][:s.nparticlesall[0]].astype('float64')
        u = s.data['u'][:s.nparticlesall[0]].astype('float64')
        rho = s.data['rho'][:s.nparticlesall[0]].astype('float64') / s.hubbleparam ** 2
        
        Z = np.maximum(s.data['gz'][:s.nparticlesall[0]].astype('float64'), 1.e-8 * ZSUN)
        
        P = GAMMA_MINUS1 * rho * u * s.hubbleparam * s.hubbleparam / BOLTZMANN * (1e20 * MSUN / MPC ** 3.)
        
        index = s.data['sfr'][:s.nparticlesall[0]] > 0
        print
        '<<<<<<', rho[index].max(), rho[index].min()
        NH[index], ucold = xfactor(u[index], rho[index])
        u[index] = ucold
        
        n = XH * NH * rho * s.hubbleparam * s.hubbleparam * (1e10 * MSUN / MPC ** 3.)
        L = np.sqrt(GAMMA * GAMMA_MINUS1 * u * 1.0e10 / (GRAVITY * n)) / MPC
        SigmaSFR = 1.0e-6 * (s.data['sfr'][:s.nparticlesall[0]] / s.data['vol'][:s.nparticlesall[0]]).astype('float64') * L
        n *= 1.0e-12 * (MPC ** 3) / MSUN * L  # MSUN/pc^2
        
        fHI = HIfrac(NH * P, P0=10 ** 4.23, alpha=0.8)
        mass = fHI * NH * XH * s.mass.astype('float64') * 1.0e10
        
        print
        '>>>>>>>>', Z / ZSUN
        fHI = HIfrac_GK(n, Z, SigmaSFR)
        
        massGK = fHI * NH * XH * s.mass.astype('float64') * 1.0e10
        print
        '>>>>>>>>nnn', n.max(), n.min()
        print
        '>>>>>>>>', L.max(), L.min()
        print
        '>>>>>>>>', fHI
        print
        fHI.min(), fHI.max()
        
        maxrad = 0.120
        zcut = 0.5 * maxrad  # vertical cut in Mpc
        
        pos = s.pos.astype('float64')
        
        # after the rotation the galaxy's spin is aligned to the x axis
        r = np.sqrt(pos[:, 1] * pos[:, 1] + pos[:, 2] * pos[:, 2])
        
        # select gas particles ...
        igas, = np.where((r < maxrad) & (np.abs(pos[:, 0]) < zcut))
        
        sigma, edges = np.histogram(r[igas], bins=nshells, range=[0, maxrad], weights=mass[igas])
        sigmaGK, edges = np.histogram(r[igas], bins=nshells, range=[0, maxrad], weights=massGK[igas])
        
        rad = 0.5 * (edges[1:] + edges[:-1])
        area = 1.0e12 * np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)  # in pc^2
        sigma /= area
        sigmaGK /= area
        
        index = np.max(np.where(sigma > 1.0))
        # linear interpolation to find R_HI
        R_HI[d] = rad[index] + (1.0 - sigma[index]) * (rad[index + 1] - rad[index]) / (sigma[index + 1] - sigma[index])
        
        index = np.max(np.where(sigmaGK > 1.0))
        # linear interpolation to find R_HI
        R_HI_GK[d] = rad[index] + (1.0 - sigmaGK[index]) * (rad[index + 1] - rad[index]) / (sigmaGK[index + 1] - sigmaGK[index])
        
        sigmatab[d, :] = np.interp(intRad, rad / R_HI[d], sigma)
        sigmatab_GK[d, :] = np.interp(intRad, rad / R_HI_GK[d], sigmaGK)
    
    # ax.semilogy( intRad, sigmatab[d,:], linestyle='solid', color='lightgray', lw=1.5 )  # ax.semilogy( intRad, sigmatab_GK[d,:],
    # linestyle='solid', color='gray', lw=1.5 )
    
    sigmamed, sigmalow, sigmahigh = get_bluediskprof()
    
    ax.fill_between(intRad, sigmalow, sigmahigh, color='lightgray', alpha=0.3)
    ax.semilogy(intRad, sigmalow, linestyle='solid', color='lightgray', lw=0.8)
    ax.semilogy(intRad, sigmahigh, linestyle='solid', color='lightgray', lw=0.8)
    ax.semilogy(intRad, sigmamed, linestyle='solid', color='lightgray', lw=1.4, label='Wang+ 14')
    
    sigmamed = np.median(sigmatab, axis=0)
    sigmalow = np.percentile(sigmatab, 16, axis=0)
    sigmahigh = np.percentile(sigmatab, 84, axis=0)
    
    ax.errorbar(intRad[1:20], sigmamed[1:20], yerr=((sigmamed - sigmalow)[1:20], (sigmahigh - sigmamed)[1:20]), ecolor='blue', fmt='None', lw=1.5,
                zorder=3)
    ax.semilogy(intRad[1:20], sigmamed[1:20], linestyle='None', mec='None', mfc='blue', ms=5, marker='s', label='L08', zorder=4)
    
    sigmamed = np.median(sigmatab_GK, axis=0)
    sigmalow = np.percentile(sigmatab_GK, 16, axis=0)
    sigmahigh = np.percentile(sigmatab_GK, 84, axis=0)
    
    ax.errorbar(intRad[1:20], sigmamed[1:20], yerr=((sigmamed - sigmalow)[1:20], (sigmahigh - sigmamed)[1:20]), ecolor='red', fmt='None', lw=1.,
                zorder=4)
    ax.semilogy(intRad[1:20], sigmamed[1:20], linestyle='None', mfc='red', mec='None', ms=5, marker='s', label='GK11', zorder=5)
    
    ax.legend(loc='upper right', frameon=False, numpoints=1, fontsize=figure.fontsize)
    
    minorLocator = MultipleLocator(0.1)
    ax.xaxis.set_minor_locator(minorLocator)
    majorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(FormatStrFormatter('$%g$'))
    
    figure.fig.savefig("%s/HIprofile_median_%03d.%s" % (outpath, snap, suffix), dpi=300)


snap = 127  # 63
# snap = 63

prefix = '/hits/universe/GigaGalaxy/level4_MHD/'
# prefix = '/hits/universe/GigaGalaxy/level3/'
outpath = '../plots'
runs = ['halo_%s' % i for i in range(1, 31)]
# runs = ['halo_%s' % i for i in range(1, 5)]
# runs = ['halo16_MHD', 'halo24_MHD']
dirs = [prefix] * len(runs)
suffix = 'pdf'

plot_HIprofile(runs, dirs, outpath, snap, suffix, nshells=120)