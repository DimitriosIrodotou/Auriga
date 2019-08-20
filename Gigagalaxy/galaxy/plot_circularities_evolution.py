from const import *
from loadmodules import *
from pylab import *
from util import *
from util import multipanel_layout

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']


def plot_circularities_evolution(runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols):
    panels = nrows * ncols
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(3)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[13.5, 0.], ylim=[0.0, 1.1])
    figure.set_axis_locators(xminloc=1., xmajloc=2.0, yminloc=0.1, ymajloc=0.2)
    figure.set_fontsize(10)
    figure.set_axis_labels(xlabel="$\\rm{t_{lookback}\,[Gyr]}$", ylabel="$\\rm{D/T \, , B/T}$")
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        filename = outpath + runs[d] + '/DTBTevo_%s.txt' % runs[d]
        f = open(filename, 'w')
        header = "%12s%12s%12s%12s%12s\n" % ("Time", "D/T-0.5", "D/T-0.6", "D/T-0.7", "B/T")
        f.write(header)
        
        ax = figure.axes[d]
        
        snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], list(zlist)))
        
        dtrat5 = np.zeros(len(snaps))
        dtrat6 = np.zeros(len(snaps))
        dtrat7 = np.zeros(len(snaps))
        btrat = np.zeros(len(snaps))
        time = np.zeros(len(snaps))
        
        for i, snap in enumerate(snaps):
            print("Doing dir %s, snap %d" % (dd, snap))
            
            attrs = ['pos', 'vel', 'mass', 'age', 'id', 'pot']
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
            s.calc_sf_indizes(sf)
            s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            time[i] = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True)
            
            g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
            g.prep_data()
            
            sdata = g.sgdata['sdata']
            eps2 = sdata['eps2']
            smass = sdata['mass']
            star_age = sdata['age']
            
            # simple D/T ratio cut
            
            jj, = np.where((eps2 > 0.5))
            dtrat5[i] = smass[jj].sum() / smass.sum()
            jj, = np.where((eps2 > 0.6))
            dtrat6[i] = smass[jj].sum() / smass.sum()
            jj, = np.where((eps2 > 0.7))
            dtrat7[i] = smass[jj].sum() / smass.sum()
            kk, = np.where((eps2 < 0.))
            btrat[i] = smass[kk].sum() * 2. / smass.sum()
            
            param = "%12.3f%12.3f%12.3f%12.3f%12.3f\n" % (time[i], dtrat5[i], dtrat6[i], dtrat7[i], btrat[i])
            f.write(param)
        
        ax.plot(time, dtrat5, 'b-')
        ax.plot(time, dtrat6, 'b', dashes=(4, 4))
        ax.plot(time, dtrat7, 'b.')
        ax.plot(time, btrat, 'r-')
        
        figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ("Au", runs[d].split('_')[1]), position='bottom left', color='k', fontsize=10)
        
        ax.legend(loc='upper left', frameon=False, prop={'size': 7})
        
        f.close()
    
    figure.fig.savefig('%s/DTkinematic_evo.%s' % (outpath, suffix))