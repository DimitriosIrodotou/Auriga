from const import *
from gadget import *
from gadget_subfind import *
from pylab import *

toinch = 0.393700787
colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']


def plot_bh_utherm(runs, dirs, outpath, firstsnap, lastsnap, suffix):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    ax1 = axes([0.17 / fac, 0.13, 0.74 / fac, 0.74])
    
    lencol = len(colors)
    
    nsnaps = lastsnap - firstsnap + 1
    time = zeros(nsnaps)
    bhutherm = zeros(nsnaps)
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        for isnap in range(nsnaps):
            snap = firstsnap + isnap
            
            print
            "Doing dir %s, snap %d." % (dd, snap)
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, onlyHeader=True)
            if s.nparticlesall[5] == 0:
                continue
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[5], loadonly=['pos', 'bhu'])
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
            
            s.center = sf.data['fpos'][0, :]
            galrad = 0.1 * sf.data['frc2'][0]
            
            ibh, = np.where((s.r() < galrad) & (s.type == 5))
            
            print
            len(ibh), isnap
            
            if len(ibh) == 0:
                continue
            
            bhutherm[isnap] = s.data['bhu'][ibh[0]]
            time[isnap] = s.cosmology_get_lookback_time_from_a(s.time)
        
        # time = time[::-1]
        # bhhsml = bhhsml[::-1]
        
        ibh, = np.where(bhutherm > 0.0)
        
        ax1.semilogy(time[ibh], bhutherm[ibh], color=colors[d], label="$\\rm{%s-{%s}}$" % (runs[d].split('_')[0], runs[d].split('_')[1]))
        
        time[:] = 0.0
        bhutherm[:] = 0.0
    
    z = pylab.array([5., 3., 2., 1., 0.5, 0.3, 0.1, 0.0])
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
    
    ax1.legend(loc='upper right', frameon=False, numpoints=1)
    ax1.set_xlim(0, 14)
    ax1.invert_xaxis()
    ax1.set_xlabel("$\\rm{t_{look}\\,[Gyr]}$")
    ax1.set_ylabel("$\\rm{u\\,[code units]}$")
    
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(times)
    ax2.set_xticklabels(lb)
    ax2.set_xlabel("$\\rm{z}$")
    
    fig.savefig('%s/bh_utherm.%s' % (outpath, suffix), dpi=300)