import util.multipanel_layout as multipanel_layout
from const import *
from gadget import *
from gadget_subfind import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

toinch = 0.393700787
base = '/output/fof_subhalo_tab_'
band = 'V'
band_array = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}


def plot_sfr_multi(runs, dirs, outpath, snap, suffix):
    # aspect ratio 2:1
    fac = 2.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1 * fac]) * toinch * 0.5, dpi=300)
    fig_width = 0.17
    fig_height = 0.33 / fac
    low_cornerx = 0.1
    low_cornery = 0.4
    offsetx = 0.07
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        print
        "Doing dir %s." % dd
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True)
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, photometry=True)
        
        n = 0
        filecounter = 1
        i, = np.where(subhalos.subhaloslentype[:, 4] > 0)
        lastsubhalowithstars = len(i)
        
        print
        'number of subhalos', lastsubhalowithstars
        
        for i in range(subhalos.numbersubhalos[subhalos.groupid]):
            if subhalos.subhaloslentype[i, 4] > 0:
                index = n % 16
                istarsbeg = subhalos.particlesoffsets[i, 4]
                istarsend = istarsbeg + subhalos.subhaloslentype[i, 4]
                mass = s.data['gima'][istarsbeg:istarsend]
                age = s.data['age'][istarsbeg:istarsend]
                
                istars, = np.where(age > 0.)
                
                nstars = size(istars)
                if nstars == 0:
                    continue
                
                age = age[istars]
                mass = mass[istars]
                
                n += 1
                
                tmax = s.cosmology_get_lookback_time_from_a(np.float64(0.0001), is_flat=True)
                
                time_range = [0, tmax]
                time_nbins = 100
                time_binsize = 1.0 * (max(time_range) - min(time_range)) / time_nbins
                
                tfac = 1e10 / 1e9 / time_binsize
                
                age[:] = s.cosmology_get_lookback_time_from_a(age[:], is_flat=True)
                
                z = pylab.array([5., 3., 2., 1., 0.5, 0.3, 0.1, 0.0])
                a = 1. / (1 + z)
                
                times = np.zeros(len(a))
                for k in range(len(a)):
                    times[k] = s.cosmology_get_lookback_time_from_a(a[k], is_flat=True)
                
                print
                times
                
                lb = []
                for v in z:
                    if v >= 1.0:
                        lb += ["%.0f" % v]
                    else:
                        if v != 0:
                            lb += ["%.1f" % v]
                        else:
                            lb += ["%.0f" % v]
                
                # set the position of the plots
                if index < 4:
                    low_cornerx = 0.07 + index * (fig_width + offsetx)
                    low_cornery = 0.78
                elif index < 8:
                    low_cornerx = 0.07 + (index - 4) * (fig_width + offsetx)
                    low_cornery = 0.70 - fig_height
                elif index < 12:
                    low_cornerx = 0.07 + (index - 8) * (fig_width + offsetx)
                    low_cornery = 0.62 - 2.0 * fig_height
                else:
                    low_cornerx = 0.07 + (index - 12) * (fig_width + offsetx)
                    low_cornery = 0.54 - 3.0 * fig_height
                
                ax1 = axes([low_cornerx, low_cornery, fig_width, fig_height])
                ax2 = ax1.twiny()
                
                text(0.05, 0.88, "$\\rm{subhalo-%d}$" % (i + 1), size=6, transform=ax1.transAxes)
                
                minorLocator = MultipleLocator(0.5)
                ax1.xaxis.set_minor_locator(minorLocator)
                minorLocator = MultipleLocator(1.0)
                ax1.yaxis.set_minor_locator(minorLocator)
                ax2.yaxis.set_minor_locator(minorLocator)
                majorLocator = MultipleLocator(5.0)
                ax1.yaxis.set_major_locator(majorLocator)
                ax2.yaxis.set_major_locator(majorLocator)
                
                for label in ax1.xaxis.get_ticklabels():
                    label.set_fontsize(5)
                for label in ax2.xaxis.get_ticklabels():
                    label.set_fontsize(5)
                for label in ax1.yaxis.get_ticklabels():
                    label.set_fontsize(5)
                
                ax1.set_xlabel("$\\rm{t_{look}\\,[Gyr]}$", size=5)
                ax1.set_ylabel(r'${\rm{SFR\,[M_\odot\,yr^{-1}}]}$', size=5)
                ax2.set_xlabel('z', size=5)
                
                ax1.hist(age, color='b', weights=mass * tfac, histtype='step', bins=100, range=time_range, log=True)
                
                ax1.invert_xaxis()
                
                handles = [matplotlib.patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)]
                labels = ["$\\rm{M_{%s} = %.1f}$" % (band, subhalos.subhalosphotometry[i, band_array[band]])]
                ax1.legend(handles, labels, loc='upper right', fontsize=5, frameon=False, handlelength=0, handleheight=0)
                
                ax2.set_xlim(ax1.get_xlim())
                ax2.set_xticklabels(lb)
                ax2.set_xticks(times)
                
                if (n % 16) == 0 or n == lastsubhalowithstars:
                    savefig('%s/sfr_multi_%s_file%d.%s' % (outpath, runs[d], filecounter, suffix), dpi=300)
                    filecounter += 1
                    fig.clf()


def plot_sfr_multi_cumulative(runs, dirs, outpath, snap, suffix):
    nrows = 3
    ncols = 3
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, twinxaxis=True, left=0.095)
    figure.set_figure_layout()
    # the "strange" limits on the x axis are necessary to get all the tick labels"
    figure.set_axis_limits_and_aspect(xlim=[-0.09, 14.09], ylim=[0.0, 1.0], logaxis=False)
    figure.set_axis_locators(xminloc=0.5, yminloc=0.1, ymajloc=0.2)
    figure.set_axis_labels(xlabel="$\\rm{t_{look}\\,[Gyr]}$", ylabel=r'${\rm{f\,(> t_{look}})}$', x2label="z")
    figure.set_fontsize(fontsize=5)
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        print
        "Doing dir %s." % dd
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True)
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, photometry=True)
        
        n = 0
        filecounter = 1
        i, = np.where(subhalos.subhaloslentype[:, 4] > 0)
        lastsubhalowithstars = len(i)
        
        print
        'number of subhalos', lastsubhalowithstars
        
        for i in range(subhalos.numbersubhalos[subhalos.groupid]):
            if subhalos.subhaloslentype[i, 4] > 0:
                index = n % (nrows * ncols)
                ax = figure.axes[index]
                istarsbeg = subhalos.particlesoffsets[i, 4]
                istarsend = istarsbeg + subhalos.subhaloslentype[i, 4]
                mass = s.data['gima'][istarsbeg:istarsend]
                age = s.data['age'][istarsbeg:istarsend]
                
                istars, = np.where(age > 0.)
                
                nstars = size(istars)
                if nstars == 0:
                    continue
                
                age = age[istars]
                mass = mass[istars]
                
                n += 1
                
                tmax = s.cosmology_get_lookback_time_from_a(np.float64(0.0001), is_flat=True)
                
                time_range = [0, tmax]
                time_nbins = 100
                time_binsize = 1.0 * (max(time_range) - min(time_range)) / time_nbins
                
                tfac = 1e10 / 1e9 / time_binsize
                
                age[:] = s.cosmology_get_lookback_time_from_a(age[:], is_flat=True)
                
                z = pylab.array([5., 3., 2., 1., 0.5, 0.3, 0.1, 0.0])
                a = 1. / (1 + z)
                
                times = np.zeros(len(a))
                for k in range(len(a)):
                    times[k] = s.cosmology_get_lookback_time_from_a(a[k], is_flat=True)
                
                print
                times
                
                lb = []
                for v in z:
                    if v >= 1.0:
                        lb += ["%.0f" % v]
                    else:
                        if v != 0:
                            lb += ["%.1f" % v]
                        else:
                            lb += ["%.0f" % v]
                
                figure.twinxaxes[index].set_xticklabels(lb)
                figure.twinxaxes[index].set_xticks(times)
                figure.twinxaxes[index].invert_xaxis()
                
                figure.set_panel_title(index, "$\\rm{subhalo-%d}$" % (i + 1), position='bottom right', fontsize=5)
                
                ax.hist(age, color='b', weights=mass * tfac, histtype='step', bins=100, range=time_range, normed=True, cumulative=-1)
                
                # constant SFH
                sftimes = arange(0.0, 14.0 + 1.0, 1.0)
                print
                'list of sf times', sftimes
                ax.plot(sftimes, 1.0 - sftimes / sftimes.max(), color='gray', ls=':')
                
                handles = [matplotlib.patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)]
                labels = ["$\\rm{M_{%s} = %.1f}$" % (band, subhalos.subhalosphotometry[i, band_array[band]])]
                ax.legend(handles, labels, loc='upper right', prop={'size': 5}, frameon=False, handlelength=0, handleheight=0)
                
                ax.invert_xaxis()
                
                for l in ax.get_xticklines() + ax.get_yticklines() + figure.twinxaxes[index].get_xticklines():
                    l.set_markersize(3)
                    l.set_markeredgewidth(0.6)
                
                if (n % (nrows * ncols)) == 0 or n == lastsubhalowithstars:
                    figure.fig.savefig('%s/sfr_multi_cumulative_%s_file%d.%s' % (outpath, runs[d], filecounter, suffix), dpi=300)
                    filecounter += 1
                    
                    if lastsubhalowithstars - n > nrows * ncols:
                        figure.set_panel_number(nrows * ncols)
                    elif lastsubhalowithstars - n > 0:
                        figure.set_panel_number(lastsubhalowithstars - n)
                    
                    # the "strange" limits on the x axis are necessary to get all the tick labels"
                    figure.set_axis_limits_and_aspect(xlim=[-0.09, 14.09], ylim=[0.0, 1.0], logaxis=False)
                    figure.set_axis_locators(xminloc=0.5, yminloc=0.1, ymajloc=0.2)
                    figure.set_axis_labels(xlabel="$\\rm{t_{look}\\,[Gyr]}$", ylabel=r'${\rm{f\,(> t_{look}})}$', x2label="z")
                    figure.set_fontsize(fontsize=5)