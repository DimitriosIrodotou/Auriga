from const import *
from loadmodules import *
from parallel_decorators import vectorize_parallel
from pylab import *
from util import *
from util import plot_helper

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple', 'gray', 'hotpink']
markers = ['o', '^', 'd', 's', 'v', '>']


def plot_stellarvstotalmass_evolution(runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, suffix, mergertree=True, dat=None, subhalo=0):
    panels = nrows * ncols
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(8)
    figure.set_figure_layout()
    figure.set_fontsize(8)
    figure.set_axis_labels(xlabel="$\\rm{M_{tot}\\,[M_\\odot]}$", ylabel="$\\rm{M_{stars}\\,[M_\\odot]}$")
    figure.set_axis_limits_and_aspect(xlim=[2.e10, 3.e12], ylim=[5.e8, 1.e12], logaxis=True)
    
    figure2 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure2.set_fontsize(8)
    figure2.set_figure_layout()
    figure2.set_fontsize(8)
    figure2.set_axis_labels(xlabel="$\\rm{t_{lookback}\\,[Gyr]}$", ylabel="$\\rm{M_{tot}\\,[M_\\odot]}$")
    figure2.set_axis_limits_and_aspect(xlim=[13.5, 0.], ylim=[2.e10, 3.e12], logyaxis=True)
    
    lencol = len(colors)
    lenmrk = len(markers)
    
    masses = arange(1., 300.)
    cosmic_baryon_frac = 0.048 / 0.307
    
    for i in range(panels):
        figure.axes[i].loglog(1.0e10 * masses, 1.0e10 * masses * cosmic_baryon_frac, ':', color='gray')
        figure.axes[i].fill_between(1.0e10 * masses, 1.0e10 * masses * cosmic_baryon_frac, [1e12] * len(masses), color='gray', alpha=0.2,
                                    edgecolor='none')
    
    if dat:
        # initialise plot helper class
        ph = plot_helper.plot_helper()
    
    if dat == 'Guo':
        for i in range(panels):
            ax = figure.axes[i]
            guo_high = ph.guo_abundance_matching(masses) * 10 ** (+0.2)
            guo_low = ph.guo_abundance_matching(masses) * 10 ** (-0.2)
            ax.fill_between(1.0e10 * masses, 1.0e10 * guo_low, 1.0e10 * guo_high, color='g', hatch='///', hatchcolor='g', alpha=0.2)
            ax.loglog(1.0e10 * masses, 1.0e10 * ph.guo_abundance_matching(masses), linestyle='-', color='g')
            labels = ["$\\rm{M_{200}\\,\\,\\,\\Omega_b / \\Omega_m}$", "Guo+ 10"]
            l1 = legend(labels, loc='upper left', fontsize=9, frameon=False)
    
    # add Moster abundance matching curve
    if dat == 'Moster':
        for i in range(panels):
            ax = figure.axes[i]
            ax.loglog(1.0e10 * masses, 1.0e10 * ph.moster_abundance_matching(masses, 1.), linestyle='-', linewidth=1., color='royalblue')
            moster_up = ph.moster_abundance_matching(masses, 1.) * 10 ** (+0.2)
            moster_lo = ph.moster_abundance_matching(masses, 1.) * 10 ** (-0.2)
            ax.fill_between(1.0e10 * masses, 1.0e10 * moster_lo, 1.0e10 * moster_up, facecolor='royalblue', alpha=0.2, edgecolor='none')
            labels = ["$\\rm{M_{200}\\,\\,\\,\\Omega_b / \\Omega_m}$", "Moster+ 13"]
            l1 = ax.legend(labels, loc='upper left', fontsize=6, frameon=False)
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        print("<< Doing ", runs[d])
        
        snap0 = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], [0.]))
        treepath = '%s/mergertrees/Au-%s/trees_sf1_%03d' % (dirs[d], runs[d].split('_')[1], snap0)
        t = load_tree(0, 0, base=treepath)
        snap_numbers_main, redshifts_main, subfind_indices_main, first_progs_indices_main, ff_tree_indices_main, fof_indices_main, prog_mass_main, \
        next_prog_indices = t.return_first_next_mass_progenitors(
            0)
        
        print("fof_indices_main,subfind_indices_main=", fof_indices_main, subfind_indices_main)
        
        # check if we need to trace ID msot bound particles
        idmb_flag = 0
        snaphz = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], [3.]))
        nin = snap0 - snaphz
        flag_indx, = np.where((fof_indices_main[:nin] > 0) & (subfind_indices_main[:nin] > 0))
        if size(flag_indx) >= 1:
            print("we have found %d snaps out of %d that have FoF and SubFind > 0" % (size(flag_indx), nin))  # idmb_flag = 1
        
        snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
        if isinstance(snaps, int):
            snaps = [snaps]
        
        snaps = list(range(50, array(snaps).max() + 1))
        snaps = snaps[::-1]
        # snaps = snaps[::8]
        
        nsnaps = len(snaps)
        
        filename = outpath + '/%s/massgrowth_%s.txt' % (runs[d], runs[d])
        f2 = open(filename, 'w')
        header = "%18s%18s%18s\n" % ("Time", "Totalmass", "Stellarmass")
        f2.write(header)
        
        ax = figure.axes[d % panels]
        ax2 = figure2.axes[d % panels]
        
        if idmb_flag:
            attrs = ['pos', 'vel', 'mass', 'age', 'id']
            s = gadget_readsnap(snap0, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4], loadonly=attrs)
            sf = load_subfind(snap0, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'spos', 'ffsh', 'fnsh'])
            s.centerat(sf.data['fpos'][0])
            idmb = s.get_most_bound_dm_particles()
            
            arr = []
            arr.append(get_stellarandtotmass_idmb(snaps, dd, attrs, idmb, snap0))
        else:
            attrs = ['pos', 'vel', 'mass', 'age']
            arr = []
            arr.append(get_stellarandtotmass(snaps, dd, attrs, fof_indices_main, subfind_indices_main, snap0))
        
        arr = np.array(arr).ravel()
        times = arr[::3]
        totalmass = arr[1::3]
        stellarmass = arr[2::3]
        print("times,totalmass,stellarmass", times, totalmass, stellarmass)
        label = "$\\rm{%s\, %s}$" % ('Au', runs[d].split('_')[1])
        color = colors[d % lencol]
        marker = markers[int(d / lencol)]
        
        ax.loglog(1.0e10 * totalmass, 1.0e10 * stellarmass, color=color, linestyle='-', lw=1.)
        ax.loglog(1.0e10 * totalmass[0], 1.0e10 * stellarmass[0], marker=marker, mfc='none', mec=color, ms=5., mew=1., linestyle='None', label=label)
        
        ax2.semilogy(times, 1e10 * totalmass, color=color, linestyle='-', lw=1., label=label)
        
        ax.legend(loc='upper left', frameon=False, prop={'size': 5}, numpoints=1, ncol=1)
        ax2.legend(loc='upper left', frameon=False, prop={'size': 5}, numpoints=1, ncol=1)
        np.savetxt(f2, np.column_stack((times, totalmass, stellarmass)))
        f2.close()
    
    figure.fig.savefig("%s/stellarvstotalmass_evolution.%s" % (outpath, suffix), dpi=300)
    figure2.fig.savefig("%s/totalmass_evolution.%s" % (outpath, suffix), dpi=300)


@vectorize_parallel(method='processes', num_procs=20)
def get_stellarandtotmass(snap, dd, attrs, fof_indices_main, subfind_indices_main, snap0):
    s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4], loadonly=attrs)
    sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'spos', 'ffsh', 'fnsh'])
    
    fof = fof_indices_main[snap0 - snap]
    sub = subfind_indices_main[snap0 - snap]
    shind = sub - sf.data['fnsh'][:fof].sum()
    s.centerat(sf.data['spos'][shind])
    g = parse_particledata(s, sf, attrs, radialcut=sf.data['frc2'][fof])
    g.prep_data()
    sdata = g.sgdata['sdata']
    gdata = g.sgdata['gdata']
    
    totalmass = s.data['mass'][(s.r() < g.sf.data['frc2'][fof])].sum()
    stellarmass = sdata['mass'].sum()
    times = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True)
    
    arr = np.array([times, totalmass, stellarmass])
    
    return arr


@vectorize_parallel(method='processes', num_procs=20)
def get_stellarandtotmass_idmb(snap, dd, attrs, idmb, snap0):
    s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4], loadonly=attrs)
    sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'spos', 'ffsh', 'fnsh'])
    centre = s.pos[np.where(np.in1d(s.id, [idmb[0]]) == True), :].ravel()
    s.centerat(centre)
    g = parse_particledata(s, sf, attrs, radialcut=sf.data['frc2'][0])
    g.prep_data()
    sdata = g.sgdata['sdata']
    gdata = g.sgdata['gdata']
    
    totalmass = s.data['mass'][(s.r() < g.sf.data['frc2'][0])].sum()
    stellarmass = sdata['mass'].sum()
    times = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True)
    
    arr = np.array([times, totalmass, stellarmass])
    
    return arr