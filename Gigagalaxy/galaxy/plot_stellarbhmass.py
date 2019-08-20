from const import *
from loadmodules import *
from pylab import *
from util import multipanel_layout
from util import select_snapshot_number

toinch = 0.393700787


def plot_stellarbhmass(runs, dirs, outpath, outputlistfile, zfirst, zlast, suffix, nrows, ncols, subhalo=0):
    panels = len(runs)
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, twinxaxis=True)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[0.0, 14.0], ylim=[1.0e6, 1.0e11], logyaxis=True)
    figure.set_axis_locators(xminloc=0.5, xmajloc=2.0, ymajloc=6.0, logyaxis=True)
    
    figure.set_axis_labels(xlabel="Time [Gyr]", ylabel="$\\rm{M\\,[M_\\odot]}$", x2label='z')
    figure.set_fontsize()
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax1 = figure.axes[d]
        ax2 = figure.twinxaxes[d]
        
        lastsnap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlast))
        firstsnap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zfirst))
        
        selected_z = [3.0, 2.0, 1.0, 0.5, 0.1]
        sel_snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile, selected_z))
        
        nsnaps = lastsnap - firstsnap + 1
        stellarmass = zeros(nsnaps)
        bhmass = zeros(nsnaps)
        time = zeros(nsnaps)
        
        loadptype = [4, 5]
        loadedpart = np.zeros(6, dtype=np.int64)
        
        for isnap in range(nsnaps):
            snap = firstsnap + isnap
            
            print
            "Doing dir %s, snap %d." % (dd, snap)
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=loadptype, loadonly=['pos', 'mass', 'age'])
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['frc2', 'fnsh', 'spos', 'smty'])
            
            # loadedpart[:] = 0
            # for type in loadptype:
            #	loadedpart[type] = s.nparticlesall[type]
            
            subhalostarmass = sf.data['smty'][0:sf.data['fnsh'][0], 4]
            jj = subhalostarmass.argmax()
            s.center = sf.data['spos'][jj, :]
            rad = 0.1 * sf.data['frc2'][0]
            
            g = parse_particledata(s, sf, attrs, radialcut=rad)
            g.prep_data()
            sdata = g.sgdata['sdata']
            gdata = g.sgdata['gdata']
            
            time[isnap] = s.cosmology_get_lookback_time_from_a(s.time)
            stellarmass[isnap] = sdata['mass'].sum()
            
            ibh, = np.where((s.r() < rad) & (s.type == 5))
            
            if size(ibh) > 0:
                bhmass[isnap] = s.data['mass'][ibh].max()
        
        time = time[::-1]
        bhmass = bhmass[::-1]
        stellarmass = stellarmass[::-1]
        
        ax1.semilogy(time, 1.0e10 * stellarmass, color='r', label="stars")
        ax1.semilogy(time, 1.0e10 * bhmass, color='k', label='BH')
        
        for j in sel_snap:
            if j > lastsnap:
                continue
            
            i = nsnaps - 1 - (j - firstsnap)
            print(i, j, len(bhmass), len(stellarmass))
            ypos = 10 ** (0.5 * (np.log10(1.0e10 * bhmass[i]) + np.log10(1.0e10 * stellarmass[i])))
            ax1.vlines(time[i], 1.0e10 * bhmass[i], 1.0e10 * stellarmass[i], linestyle='-', color='gray', lw=0.5)
            ax1.text(time[i] + 1.2, ypos, "%.0f" % (stellarmass[i] / bhmass[i]), size=4, transform=ax1.transData)
        
        ax1.legend(loc='lower right', frameon=False, prop={'size': 5})
        
        figure.set_panel_title(panel=d, title="$\\rm{%s-%s}$" % (runs[d].split('_')[0], runs[d].split('_')[1]), position='top left')
        
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
        
        ax1.invert_xaxis()
        
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(times)
        ax2.set_xticklabels(lb)
    
    figure.fig.savefig('%s/stellarbhmass.%s' % (outpath, suffix), dpi=300)