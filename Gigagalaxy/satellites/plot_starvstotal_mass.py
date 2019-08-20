from const import *
from gadget import *
from gadget_subfind import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = '/fof_subhalo_tab_'


def guo_abundance_matching(mass):
    # equation 3 of Guo et al. 2010. Mass MUST be given in 10^10 M_sun
    c = 0.129
    M_zero = 10 ** 1.4
    alpha = -0.926
    beta = 0.261
    gamma = -2.44
    
    val = c * mass * ((mass / M_zero) ** alpha + (mass / M_zero) ** beta) ** gamma
    return val


def moster_abundance_matching(mass, afac):
    # equation 2 of Moster et al. 2013. Mass units in 10^10 Msun.
    m0 = 11.59
    m1 = 1.195
    n0 = 0.0351
    n1 = -0.0247
    beta0 = 1.376
    beta1 = -0.826
    gamma0 = 0.608
    gamma1 = 0.329
    
    m = 10. ** (m0 + (1 - afac) * m1) * 1e-10
    n = n0 + (1 - afac) * n1
    beta = beta0 + (1 - afac) * beta1
    gamma = gamma0 + (1 - afac) * gamma1
    
    val = 2. * n / ((mass / m) ** (-beta) + (mass / m) ** gamma)
    
    return val * mass


def plot_stellarvstotalmass(runs, dirs, outpath, snap, suffix, guo=False, fgr=True):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    axes([0.22 / fac, 0.18, 0.75 / fac, 0.75])
    
    masses = arange(0.01, 1000.)
    cosmic_baryon_frac = 0.04 / 0.25
    
    loglog(1.0e10 * masses, 1.0e10 * masses * cosmic_baryon_frac, 'k--', )
    
    time = 1.
    
    if guo:
        m_high = guo_abundance_matching(masses) * 10 ** (+0.2)
        m_low = guo_abundance_matching(masses) * 10 ** (-0.2)
        m_ab = guo_abundance_matching(masses)
        lab = r"$\rm Guo+ 10$"
    else:
        m_high = moster_abundance_matching(masses, time) * 10 ** (+0.2)
        m_low = moster_abundance_matching(masses, time) * 10 ** (-0.2)
        m_ab = moster_abundance_matching(masses, time)
        lab = r"$\rm Moster+ 13$"
    
    fill_between(1.0e10 * masses, 1.0e10 * m_low, 1.0e10 * m_high, color='lightgray', edgecolor='None')
    
    loglog(1.0e10 * masses, 1.0e10 * m_ab, 'k:')
    
    labels = ["$\\rm{M_{200}\\,\\,\\,\\Omega_b / \\Omega_m}$", lab]
    l1 = legend(labels, loc='lower right', fontsize=6, frameon=False)
    
    if fgr == True:
        for d in range(len(runs)):
            print
            'doing halo', runs[d]
            dd = dirs[d] + runs[d] + '/output/'
            subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
            subhalos.set_subhalos_properties(parent_group=0)
            print
            "here"
            totalmass, stellarmass = subhalos.get_masses_vs_stellar_masses(rcut=0.3, rvcirc_masses=True)
            
            loglog(1.0e10 * totalmass, 1.0e10 * stellarmass, 'o', ms=3.0, color=colors[d], label="$\\rm{%s}$" % runs[d])
    
    xlim(1.0e8, 1.0e13)
    ylim(1.0e5, 1.0e12)
    
    legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
    gca().add_artist(l1)
    
    xlabel("$\\rm{M_{tot}\\,[M_\\odot]}$")
    ylabel("$\\rm{M_{stars}\\,[M_\\odot]}$")
    
    savefig("%s/stellarvstotalmass.%s" % (outpath, suffix), dpi=300)