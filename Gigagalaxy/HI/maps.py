from HIutils import *

PROTONMASS = 1.67262178e-24
MSUN = 1.989e33
MPC = 3.085678e24
GAMMA = 5. / 3.
GAMMA_MINUS1 = GAMMA - 1
HYDROGEN_MASSFRAC = 0.76
BOLTZMANN = 1.38065e-16
GRAVITY = 6.6738e-8

MSUNPCSQ = 1.0e12 * MSUN / (PROTONMASS * MPC ** 2)

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}

fields = ['pos', 'vel', 'mass', 'rho', 'vol', 'u', 'sfr', 'nh', 'gz', 'gmet']


def plot_gasprojection(ax1, ax2, cax, s, boxsize, res, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.vel = s.vel[:s.nparticlesall[0], :]
    
    s.data['pos'] = s.pos
    s.data['rho'] = s.rho
    
    XH = s.data['gmet'][:s.nparticlesall[0], element['H']].astype('float64')
    NH = s.data['nh'][:s.nparticlesall[0]].astype('float64')
    u = s.data['u'][:s.nparticlesall[0]].astype('float64')
    rho = s.data['rho'][:s.nparticlesall[0]].astype('float64') / s.hubbleparam ** 2
    
    Z = np.maximum(s.data['gz'][:s.nparticlesall[0]].astype('float64'), 1.e-8 * ZSUN)
    
    P = GAMMA_MINUS1 * rho * u * s.hubbleparam * s.hubbleparam / BOLTZMANN * (1e20 * MSUN / MPC ** 3.)
    
    index = s.data['sfr'][:s.nparticlesall[0]] > 0
    NH[index], ucold = xfactor(u[index], rho[index])
    u[index] = ucold
    
    # NH[:] = 1.0
    
    n = XH * NH * rho * s.hubbleparam * s.hubbleparam * (1e10 * MSUN / MPC ** 3.)
    L = np.sqrt(GAMMA * GAMMA_MINUS1 * u * 1.0e10 / (GRAVITY * n)) / MPC
    SigmaSFR = 1.0e-6 * (s.data['sfr'][:s.nparticlesall[0]] / s.data['vol'][:s.nparticlesall[0]]).astype('float64') * L
    n *= 1.0e-12 * (MPC ** 3) / MSUN * L  # MSUN/pc^2
    
    # fHI = 1.0
    fHI = HIfrac(NH * P)
    # fHI = HIfrac_GK(n, Z, SigmaSFR)
    
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    s.data['rho'] = fHI * NH * XH * s.rho.astype('float64') * 1.0e10 * MSUN / MPC ** 2.0 / PROTONMASS * boxsize / res
    
    levels = [i * MSUNPCSQ for i in [1., 5., 10.]]
    
    print
    levels
    
    axes(ax1)
    s.plot_Aslice("rho", colorbar=False, res=res / 2, proj=1, axes=[2, 1], box=[boxsize, boxsize], vrange=[5.e18, 1.e22], levels=levels, logplot=True,
                  contour=True, rasterized=True, cmap='afmhot_r', newfig=False, numthreads=numthreads)
    axis('image')
    
    cax.set_visible(False)
    
    # colorbar( cax=cax, orientation='horizontal', ticks=[0,50,100,150,200,250] )
    # cax.set_title( '$\\rm{v_{plane}\\,[km\\,s^{-1}]}$', fontsize=10 )
    # for label in cax.xaxis.get_ticklabels():
    #	label.set_fontsize(8)
    
    axes(ax2)
    s.plot_Aslice("rho", colorbar=False, res=res / 2, proj=1, axes=[2, 0], box=[boxsize, boxsize / 2.], vrange=[5.e18, 1.e22], levels=levels,
                  logplot=True, contour=True, rasterized=True, cmap='afmhot_r', newfig=False, numthreads=numthreads)
    axis('image')
    
    return


toinch = 0.393700787
snap = 127  # 63
snap = 63

prefix = '/hits/universe/GigaGalaxy/level4_MHD/'
runs = ['halo_%s' % i for i in range(1, 31)]
# prefix = './level3_MHD/'
# runs = ['halo_6', 'halo_16', 'halo_24']
prefix = './level5_MHD/'
# runs = ['halo_6', 'halo_16', 'halo_24']
runs = ['halo_6']
# prefix = './level5/'
# runs = ['halo_16', 'halo_24']
# prefix = '/home/marinafo/universe/tap/Aquarius/'
# runs = ['Aq-%s_5' % i for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']]
suffix = 'pdf'
suffix = 'jpg'

functions = [plot_gasprojection]


def plot_disk(path, run, snap, suffix):
    path = prefix + run + '/output/'
    s = gadget_readsnap(snap, snappath=path, hdf5=True, loadonly=fields, loadonlytype=[0])
    sf = load_subfind(snap, dir=path, hdf5=True)
    s.calc_sf_indizes(sf)
    s.select_halo(sf, use_principal_axis=False, use_cold_gas_spin=True, do_rotation=True)
    
    # pxsize = 15.
    pxsize = 4.
    pysize = 6.
    
    psize = 1.9
    offset = 0.1
    
    fig = pylab.figure(figsize=(np.array([pxsize, pysize]) * toinch * 0.75), dpi=300)
    
    # res = 512
    res = 1024
    boxsize = 0.09
    
    for iplot in range(len(functions)):
        ix = iplot % 2
        x = ix * (2. * psize + offset) / pxsize + offset / pysize
        
        y = offset / pysize
        y = (psize + 2. * offset) / pysize
        ax1 = axes([x, y, 2. * psize / pxsize, 2. * psize / pysize], frameon=False)
        
        y = offset / pysize
        ax2 = axes([x, y, 2. * psize / pxsize, psize / pysize], frameon=False)
        
        y = (3. * psize + 3. * offset) / pysize + 0.15 * psize / pysize
        cax = axes([x, y, 2. * psize / pxsize, psize / pysize / 15.], frameon=False)
        
        functions[iplot](ax1, ax2, cax, s, boxsize, res, numthreads=4)
        
        for label in cax.xaxis.get_ticklabels():
            label.set_fontsize(7)
        
        num = run.split('_')[1]
        ax1.text(0.05, 0.9, "Au%s" % (num), color='k', transform=ax1.transAxes, fontsize=7)
        ax1.plot([0.95 - 10. / 70, 0.95], [0.085, 0.085], 'k', lw=1.4, transform=ax1.transAxes)
        ax1.text(0.85, 0.02, "%s" % ('10 kpc'), color='k', transform=ax1.transAxes, fontsize=5, horizontalalignment='center')
        
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    # fig.savefig( '../plots/projections_90kpc_%s.%s' % (run, suffix), transparent=True, dpi=300 )
    # fig.savefig( '../plots/projections_GK_90kpc_%s.%s' % (run, suffix), transparent=True, dpi=300 )
    # fig.savefig( '../plotslv3/projections_90kpc_%s.%s' % (run, suffix), transparent=True, dpi=300 )
    # fig.savefig( '../plotslv3/projections_GK_90kpc_%s.%s' % (run, suffix), transparent=True, dpi=300 )
    fig.savefig('../plotslv5/projections_90kpc_%s.%s' % (run, suffix), transparent=True, dpi=300)
    # fig.savefig( '../plotslv5/projections_GK_90kpc_%s.%s' % (run, suffix), transparent=True, dpi=300 )
    # fig.savefig( '../plotslv5HD/projections_90kpc_%s.%s' % (run, suffix), transparent=True, dpi=300 )
    # fig.savefig( '../plotslv5HD/projections_GK_90kpc_%s.%s' % (run, suffix), transparent=True, dpi=300 )
    # fig.savefig( '../plotsAq/projections_90kpc_%s.%s' % (run, suffix), transparent=True, dpi=300 )
    # fig.savefig( '../plotsAq/projections_GK_90kpc_%s.%s' % (run, suffix), transparent=True, dpi=300 )
    plt.close(fig)


for run in runs:
    plot_disk(path, run, snap, suffix)