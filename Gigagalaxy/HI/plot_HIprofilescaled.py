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


def plot_HIprofile(runs, dirs, outpath, snap, suffix, nshells=120, zmax=5.0e-3):
    figure = multipanel_layout.multipanel_layout(nrows=5, ncols=3, npanels=len(runs), hspace=0.07, wspace=0.07, left=0.09, right=0.975, bottom=0.06,
                                                 top=0.975)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[0.0, 1.9], ylim=[0., 20.0])
    figure.set_fontsize(6.)
    figure.set_axis_labels(xlabel="$\\rm{R/R_{HI}}$", ylabel="$\\rm{\\Sigma_{HI}\\,[M_\\odot\\,pc^{-2}]}$")
    
    if runs[0] == 'halo_16':
        f = open("mass_diameter_scal.txt", "a")
    else:
        f = open("mass_diameter_scal.txt", "w")
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        ax = figure.axes[d]
        
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
        # s.select_halo( sf, use_principal_axis=True, do_rotation=True )
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
        R_HI = rad[index] + (1.0 - sigma[index]) * (rad[index + 1] - rad[index]) / (sigma[index + 1] - sigma[index])
        
        index = np.max(np.where(sigmaGK > 1.0))
        # linear interpolation to find R_HI
        R_HI_GK = rad[index] + (1.0 - sigmaGK[index]) * (rad[index + 1] - rad[index]) / (sigmaGK[index + 1] - sigmaGK[index])
        
        kk = np.argsort(r[igas])
        r = r[igas[kk]]
        mass = mass[igas[kk]]
        massGK = massGK[igas[kk]]
        kk = r < R_HI
        M_HI = mass[kk].sum()
        kk = r < R_HI_GK
        M_HI_GK = massGK[kk].sum()
        
        header = "%12s%10.1f%10.1f%10.1f%10.1f%10.1f%10.1f\n" % (
        runs[d], R_HI * 1.0e3, M_HI * 1.0e-9, 1.0e-12 * M_HI / (np.pi * R_HI * R_HI), R_HI_GK * 1.0e3, M_HI_GK * 1.0e-9,
        1.0e-12 * M_HI_GK / (np.pi * R_HI_GK * R_HI_GK))
        f.write(header)
        
        ax.plot(rad / R_HI, sigma, linestyle='solid', color='b', lw=1.5)
        ax.plot(rad / R_HI_GK, sigmaGK, linestyle='solid', color='r', lw=1.2)
        
        # string = "$\\rm{R^{L}_{HI}}=%.1f$" % (1.0e3 * R_HI)
        # ax.text(0.94, 0.68, string, fontsize=6, transform=ax.transAxes, horizontalalignment='right' )
        # string = "$\\rm{R^{GK}_{HI}}=%.1f$" % (1.0e3 * R_HI_GK)
        # ax.text(0.94, 0.57, string, fontsize=6, transform=ax.transAxes, horizontalalignment='right' )
        # string = "$\\rm{M_{HI}}=%.1fx10^9$" % (1.0e-9 * M_HI_GK)
        # ax.text(0.45, 0.64, string, fontsize=6, transform=ax.transAxes )
        
        majorLocator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(majorLocator)
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)
        minorLocator = MultipleLocator(1.0)
        ax.yaxis.set_minor_locator(minorLocator)
        
        # limits = ax.get_ylim()
        # ax.vlines( R_HI*1.0e3, limits[0], limits[1], linestyle='--', color='darkgray', lw=1.2 )
        
        titlestr = 'Au' + runs[d].split('_')[1]
        ax.text(0.94, 0.84, titlestr, fontsize=8, transform=ax.transAxes, horizontalalignment='right')
    
    # savefig( "%s/HIprofile_scal_%03dfirst.%s" % (outpath,snap,suffix), dpi=300 )
    savefig("%s/HIprofile_scal_%03dsecond.%s" % (outpath, snap, suffix), dpi=300)
    
    f.close()


snap = 127  # 63
# snap = 63

prefix = '/hits/universe/GigaGalaxy/level4_MHD/'
# prefix = '/hits/universe/GigaGalaxy/level3/'
outpath = '../plots'
runs = ['halo_%s' % i for i in range(1, 16)]
runs = ['halo_%s' % i for i in range(16, 31)]
# runs = ['halo16_MHD', 'halo24_MHD']
dirs = [prefix] * len(runs)
suffix = 'pdf'

plot_HIprofile(runs, dirs, outpath, snap, suffix, nshells=120)