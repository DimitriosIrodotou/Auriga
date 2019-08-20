from const import *
from loadmodules import *
from pylab import *
from util import *

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
lencol = len(colors)


def plot_decompsfr_multi(runs, dirs, outpath, symlink, outputlistfile, suffix, nrows, ncols, subhalo=0, restest=False):
    if restest:
        panels = len(runs) / len(restest)
    else:
        panels = len(runs)
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, twinxaxis=True)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[0.0, 14.0], ylim=[0.0, 34.])
    figure.set_axis_locators(xminloc=0.5, xmajloc=2.0, yminloc=1.0, ymajloc=5.0)
    
    figure.set_axis_labels(xlabel=r"${\rmTime [Gyr]}$", ylabel=r'${\rm{SFR\,[M_\odot\,yr^{-1}}]}$', x2label=r'$\rm{z}$')
    figure.set_fontsize()
    
    for run in range(len(runs)):
        dd = dirs[run] + runs[run]
        if not restest:
            ax1 = figure.axes[run]
            ax2 = figure.twinxaxes[run]
        else:
            ax1 = figure.axes[int(run / len(restest))]
            ax2 = figure.twinxaxes[int(run / len(restest))]
        
        wpath = dirs[run] + '/lists/'
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[run], [0.]))
        print("Doing dir %s, snap %d." % (dd, snap))
        
        if not restest:
            filename = '%s/%sstarID_accreted_all_newmtree.dat' % (wpath, runs[run])
            print("reading", filename)
            fin = open(filename, 'rb')
            nacc = struct.unpack('i', fin.read(4))[0]
            idacc = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
            subacc = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
            snapacc = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
            aflag = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)))
            fofflag = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)))
            rootid = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
            pkmassid = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
            nins = struct.unpack('i', fin.read(4))[0]
            idins = numpy.array(struct.unpack('%sd' % nins, fin.read(nins * 8)), dtype=int64)
            fin.close()
            print("nacc,nins=", nacc, nins)
            fin.close()
        
        attrs = ['pos', 'vel', 'mass', 'age', 'id', 'gima']
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4, 5], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
        s.centerat(sf.data['fpos'][subhalo, :])
        g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][subhalo])
        g.prep_data()
        
        sdata = g.sgdata['sdata']
        starid = sdata['id']
        pos = sdata['pos']
        rad = np.sqrt((pos[:, :] ** 2).sum(axis=1))
        mass = sdata['gima']
        age = sdata['age']
        
        if not restest:
            acc_index, = np.where(np.in1d(starid, idacc) == True)
            mass_acc = mass[acc_index]
            age_acc = age[acc_index]
            ins_index, = np.where(np.in1d(starid, idins) == True)
            mass_ins = mass[ins_index]
            age_ins = age[ins_index]
        
        nstars = size(age)
        print("nstars=", nstars)
        
        tmax = 14.
        time_range = [0, tmax]
        time_nbins = 100
        time_binsize = 1.0 * (max(time_range) - min(time_range)) / time_nbins
        
        tfac = 1e10 / 1e9 / time_binsize
        
        z = pylab.array([5., 3., 2., 1., 0.5, 0.3, 0.1, 0.0])
        a = 1. / (1 + z)
        
        times = np.zeros(len(a))
        times[:] = s.cosmology_get_lookback_time_from_a(a[:], is_flat=True)
        
        lb = []
        for v in z:
            if v >= 1.0:
                lb += ["%.0f" % v]
            else:
                if v != 0:
                    lb += ["%.1f" % v]
                else:
                    lb += ["%.0f" % v]
        
        if not restest:
            figure.set_panel_title(panel=run, title="$\\rm{%s\,{%s}}$" % ("Au", runs[run].split('_')[1]), position='top left')
            ax1.hist(age_ins, weights=mass_ins * tfac, histtype='step', bins=100, range=time_range, label="in halo", alpha=0.5, fill=True, fc='b',
                     lw=0.0, stacked=False)
            ax1.hist(age_acc, weights=mass_acc * tfac, histtype='step', bins=100, range=time_range, label="accreted", alpha=0.5, fill=True, fc='r',
                     lw=0.0, stacked=False)
        
        if restest:
            print("restest=", restest)
            for j, level in enumerate(restest):
                lst = 'level%01d' % level
                if lst in dirs[run]:
                    rlab = " $\\rm{Au %s lvl %01d}$" % (runs[run].split('_')[1], level)
                else:
                    rlab = " $\\rm{Au %s}$" % (runs[run].split('_')[1])
            color = colors[run % lencol]
        else:
            color = 'k'
            rlab = "total"
        
        ax1.hist(age, color=color, weights=mass * tfac, histtype='step', bins=100, range=time_range, label=rlab)
        
        # this is 250 times the true accretion rate
        if restest:
            color = colors[run]
            label = ''
        else:
            color = 'g'
            label = "BH"
        
        try:
            filename = wpath + '/sfr/mbhev.dat'
            f1 = open(filename, 'rb')
            npoints = struct.unpack('i', f1.read(4))[0]
            tlook = numpy.array(struct.unpack('%sd' % npoints, f1.read(npoints * 8)), dtype=float)
            mbdot = numpy.array(struct.unpack('%sd' % npoints, f1.read(npoints * 8)), dtype=float)
            f1.close()
            dt = [tlook[:-1] - tlook[1:]]
            tcen = 0.5 * (tlook[:-1] + tlook[1:])
            ax1.plot(tlook, mbdot, color=color, lw=0.7, label=label)
        except:
            print("We are not plotting black hole growth")
            pass
        ax1.legend(loc='upper right', prop={'size': 5}, frameon=False)
        
        ax1.invert_xaxis()
        
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(times)
        ax2.set_xticklabels(lb)
    
    if restest:
        figure.fig.savefig('%s/decompsfr_restest.%s' % (outpath, suffix))
    else:
        figure.fig.savefig('%s/decompsfr.%s' % (outpath, suffix))