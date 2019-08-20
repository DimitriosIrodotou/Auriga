from const import *
from gadget import *
from gadget_subfind import *
from pylab import *

toinch = 0.393700787
colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']


def plot_bh_massdifference(runs, dirs, outpath, firstsnap, lastsnap, suffix):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    ax1 = axes([0.17 / fac, 0.13, 0.74 / fac, 0.74])
    
    lencol = len(colors)
    
    nsnaps = lastsnap - firstsnap + 1
    time = ones(nsnaps) * 14.0
    bhmass = zeros(nsnaps)
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        for isnap in range(nsnaps):
            snap = firstsnap + isnap
            
            print
            "Doing dir %s, snap %d." % (dd, snap)
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, onlyHeader=True)
            if s.nparticlesall[5] == 0:
                continue
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[5], loadonly=['pos', 'mass', 'bhma'])
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
            
            s.center = sf.data['fpos'][0, :]
            galrad = 0.1 * sf.data['frc2'][0]
            
            ibh, = np.where((s.r() < galrad) & (s.type == 5))
            print
            len(ibh), isnap
            
            if len(ibh) == 0:
                continue
            
            bhmass[isnap] = s.data['bhma'][ibh[0]] - s.data['mass'][ibh[0]]
            print
            s.data['bhma'][ibh[0]], s.data['mass'][ibh[0]], bhmass[isnap]
            time[isnap] = s.cosmology_get_lookback_time_from_a(s.time)
        
        # time = time[::-1]
        # bhhsml = bhhsml[::-1]
        
        # ibh, = np.where(bhmass > 0.0)
        
        bhmass[:] *= 1.0e10
        
        ax1.plot(time, bhmass, color=colors[d], linestyle='-', label="$\\rm{%s-{%s}}$" % (runs[d].split('_')[0], runs[d].split('_')[1]))
        
        time[:] = 0.0
        bhmass[:] = 0.0
    
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
    
    ax1.legend(loc='lower right', frameon=False, numpoints=1)
    ax1.set_xlim(0, 14)
    ax1.invert_xaxis()
    ax1.set_xlabel("$\\rm{t_{look}\\,[Gyr]}$")
    ax1.set_ylabel("$\\rm{\\Delta\\,M}$")
    
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(times)
    ax2.set_xticklabels(lb)
    ax2.set_xlabel("$\\rm{z}$")
    
    fig.savefig('%s/bh_mass_diff.%s' % (outpath, suffix), dpi=300)