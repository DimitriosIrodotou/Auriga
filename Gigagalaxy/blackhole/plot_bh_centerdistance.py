from const import *
from gadget import *
from gadget_subfind import *
from pylab import *

toinch = 0.393700787
colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']


def plot_bh_centerdistance(runs, dirs, outpath, firstsnap, lastsnap, suffix, check_subhalo=True, most_massive=False, closest=False):
    # check_subhalo: the BH has to be in the main subhalo
    # most_massive: take most massive BH (dynamical mass), rather than first by index
    # closest: take closest BH
    
    assert not (most_massive and closest)
    
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    ax1 = axes([0.17 / fac, 0.13, 0.74 / fac, 0.74])
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        nsnaps = lastsnap - firstsnap + 1
        time = np.zeros(nsnaps)
        bhdist = time.copy()
        for isnap in range(nsnaps):
            snap = firstsnap + isnap
            
            print
            "Doing dir %s, snap %d." % (dd, snap)
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, onlyHeader=True)
            
            if s.nparticlesall[5] == 0:
                # do not plot anything for this snap
                time[isnap] = np.nan
                bhdist[isnap] = np.nan
                continue
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[5], loadonly=['pos', 'mass'])
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['flty', 'fnsh', 'ffsh', 'frc2', 'spos', 'slty'])
            s.calc_sf_indizes(sf, halolist=[0], dosubhalos=True, absolutesubnum=False)
            
            s.center = sf.data['spos'][sf.data['ffsh'][0], :]
            galradfac = 1.0
            galrad = galradfac * sf.data['frc2'][0]
            
            part_rad = s.r()
            ibh = (part_rad < galrad) & (s.type == 5)
            if check_subhalo:
                ibh &= (s.data['subhalo'] == 0)
            nbh = ibh.sum()
            print
            nbh, isnap
            
            if nbh == 0:
                # flag for missing BH
                bhdist[isnap] = -1.0
            elif most_massive and (nbh > 1):
                print
                'taking most massive BH of', nbh, 'within', galradfac, 'r200'
                bhdist[isnap] = 1.0e3 * part_rad[ibh][s.mass[ibh].argmax()]
            elif closest and (nbh > 1):
                print
                'taking closest BH of', nbh, 'within', galradfac, 'r200'
                bhdist[isnap] = 1.0e3 * part_rad[ibh].min()
            else:
                bhdist[isnap] = 1.0e3 * part_rad[np.nonzero(ibh)[0][0]]
            
            time[isnap] = s.cosmology_get_lookback_time_from_a(s.time)
        
        for isnap in range(nsnaps):
            print
            isnap, time[isnap], bhdist[isnap]
        
        ax1.plot(time, bhdist, color=colors[d], linestyle='-', label="$\\rm{%s-{%s}}$" % (runs[d].split('_')[0], runs[d].split('_')[1]))
        ax1.plot(time, bhdist, marker='s', mfc='black', mec='None', ms=2, linestyle='None')
    
    z = np.array([5., 3., 2., 1., 0.5, 0.3, 0.1, 0.0])
    a = 1. / (1 + z)
    
    times = np.zeros(len(a))
    for i in range(len(a)):
        times[i] = s.cosmology_get_lookback_time_from_a(a[i])
    
    lb = []
    for v in z:
        if v >= 1.0:
            lb += ["%.0f" % v]
        else:
            if v != 0:
                lb += ["%.1f" % v]
            else:
                lb += ["%.0f" % v]
    
    xlim = [14.0, 0.0]
    
    ax1.legend(loc='upper right', frameon=False, numpoints=1)
    ylim = ax1.get_ylim()
    if ylim[0] < 0:
        ax1.hlines(0.0, xlim[1], xlim[0])
    ax1.set_xlim(xlim)
    ax1.set_xlabel("$\\rm{t_{look}\\,[Gyr]}$")
    ax1.set_ylabel("$\\rm{d\\,[kpc]}$")
    
    ax2 = ax1.twiny()
    ax2.set_xticks(times)
    ax2.set_xticklabels(lb)
    ax2.set_xlim(xlim)
    ax2.set_xlabel("$\\rm{z}$")
    
    save_name = outpath + '/bh_centerdistance'
    if check_subhalo:
        save_name += '_subhalo'
    if most_massive:
        save_name += '_mostmassive'
    if closest:
        save_name += '_closest'
    save_name += '.' + suffix
    print
    save_name
    
    fig.savefig(save_name, dpi=300)