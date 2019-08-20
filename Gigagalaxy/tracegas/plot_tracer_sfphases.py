import matplotlib

matplotlib.use('Agg')
from loadmodules import *
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from util import *

toinch = 0.393700787

colors = ['b', 'c', 'k', 'gray', 'g', 'y', 'r', 'purple', 'm', 'hotpink']
symbols = ['o', 's', '^', '*', 'v', 'd']
lws = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]


def plot_tracer_sfphases(dirs, runs, snaplist, name, outpath, outputlistfile, suffix='', nrec=6, nreclim=7, plot2dhist=False, ploteachsim=False):
    if nrec >= nreclim:
        raise ValueError('nrec cannot be equal or larger to nreclim for the purposes of this script.')
    
    lzsim = np.zeros((len(dirs), nrec))
    ttsim = np.zeros((len(dirs), nrec))
    rmaxsim = np.zeros((len(dirs), nrec - 1))
    ttsimrmax = np.zeros((len(dirs), nrec - 1))
    
    nrows = 2
    ncols = 5
    panels = len(runs)
    
    xran = [0.5, 5.5]  # for phases
    xran = [12.5, 2.5]  # lookback time
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, bottom=0.2, top=0.98, aspect_fac=0.7)  # ,
    # twinyaxis=True)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=xran, ylim=[-0.4, 1.6], logaxis=False)
    figure.set_axis_locators(xminloc=1., xmajloc=2., ymajloc=1., yminloc=0.5)
    figure.set_axis_labels(xlabel="$\\rm{t_{lookback} \,[Gyr]}$",
                           ylabel="$\\rm{l_z \, [10^3 \, kpc\,km\,s^{-1}]}$")  # , y2label="$\\rm{R_{max} \, [kpc]}$" )
    figure.set_fontsize(8)
    
    figure1 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, bottom=0.2, top=0.98, aspect_fac=0.7)
    figure1.set_figure_layout()
    figure1.set_axis_limits_and_aspect(xlim=xran, ylim=[0., 80.], logaxis=False)
    figure1.set_axis_locators(xminloc=1., xmajloc=2., ymajloc=20., yminloc=10.)
    figure1.set_axis_labels(xlabel="$\\rm{t_{lookback} \,[Gyr]}$", ylabel="$\\rm{R_{max} \,[kpc]}$")
    figure1.set_fontsize(8)
    
    figure2 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, bottom=0.2, top=0.98)
    figure2.set_figure_layout()
    figure2.set_axis_limits_and_aspect(xlim=xran, ylim=[0., 2.1], logaxis=False)
    figure2.set_axis_locators(xminloc=2., xmajloc=2., ymajloc=1., yminloc=0.5)
    figure2.set_axis_labels(xlabel="$n\\rm{-th \, star-forming \, phase}$", ylabel="$\\rm{Z[Z_{\odot}]}$")
    figure2.set_fontsize(8)
    
    cmap = plt.get_cmap('viridis')
    parameters = np.linspace(1., nreclim, nreclim)
    norm = mpl.colors.Normalize(vmin=np.min(parameters), vmax=np.max(parameters))
    s_m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    snaplist_reset = snaplist
    
    for d in range(len(runs)):
        
        ax = figure.axes[d]
        # axt = figure.twinyaxes[0]
        ax1 = figure1.axes[d]
        ax2 = figure2.axes[d]
        
        figure3 = multipanel_layout.multipanel_layout(nrows=1, ncols=nreclim - 1, npanels=nreclim - 1, bottom=0.2, top=0.98)
        figure3.set_figure_layout()
        figure3.set_axis_limits_and_aspect(xlim=[13.5, 0.], ylim=[0., 0.4], logaxis=False)
        figure3.set_axis_locators(xminloc=2., xmajloc=2., ymajloc=0.1, yminloc=0.1)
        figure3.set_axis_labels(xlabel="$\\rm{t_{lookback} [Gyr]}$", ylabel="$\\rm{df}$")
        figure3.set_fontsize(8)
        
        figure4 = multipanel_layout.multipanel_layout(nrows=1, ncols=nreclim - 1, npanels=nreclim - 1, bottom=0.2, top=0.98)
        figure4.set_figure_layout()
        figure4.set_axis_limits_and_aspect(xlim=[-8., 8.], ylim=[0., 0.4], logaxis=False)
        figure4.set_axis_locators(xminloc=2., xmajloc=2., ymajloc=0.1, yminloc=0.1)
        figure4.set_axis_labels(xlabel="$\\rm{l_z}$", ylabel="$\\rm{df}$")
        figure4.set_fontsize(8)
        
        run = runs[d]
        dir = dirs[d]
        
        path = '%s/%s%s/' % (dir, run, suffix)
        
        wpath = '%s/%s/tracers/' % (outpath, run)
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        wpath = str(wpath)
        
        loadonlytype = [0, 4, 6]
        sidmallnow = []
        
        nb = 0
        nbt = 0
        
        # if runs[d] == 'halo_L2':
        #        snaplist = snaplist[6:]
        # elif runs[d] == 'halo_L4' or runs[d] == 'halo_L5':
        #        snaplist = snaplist[2:]
        # else:
        #        snaplist = snaplist_reset
        
        lbtime = np.zeros(len(snaplist) - 1)
        tlist = np.arange(len(snaplist) - 1)
        # now look for them in previous snapshots
        print("snaplist=", snaplist)
        rvirial = np.zeros(len(snaplist) - 1)
        
        for isnap, snap in enumerate(snaplist[:-1]):
            if runs[d] == 'halo_L6':  # or runs[d] == 'halo_L5':
                fname = '%s/tracers_stars_snap%d_dmcen.dat' % (wpath, snap)
            else:
                fname = '%s/tracers_stars_snap%d.dat' % (wpath, snap)
            print("Reading %s" % fname)
            fin = open(fname, 'rb')
            time = struct.unpack('d', fin.read(8))[0]
            rvir = struct.unpack('d', fin.read(8))[0]
            ntracers = struct.unpack('i', fin.read(4))[0]
            trid_all = numpy.array(struct.unpack('%sd' % ntracers, fin.read(ntracers * 8)), dtype=int64)
            htype = numpy.array(struct.unpack('%si' % ntracers, fin.read(ntracers * 4)), dtype=int32)
            ttype = numpy.array(struct.unpack('%si' % ntracers, fin.read(ntracers * 4)), dtype=int32)
            ctype = numpy.array(struct.unpack('%si' % ntracers, fin.read(ntracers * 4)), dtype=int32)
            prid_all = numpy.array(struct.unpack('%sd' % ntracers, fin.read(ntracers * 8)), dtype=int64)
            rad_all = numpy.array(struct.unpack('%sf' % ntracers, fin.read(ntracers * 4)), dtype=float64)
            height_all = numpy.array(struct.unpack('%sf' % ntracers, fin.read(ntracers * 4)), dtype=float64)
            lz_all = numpy.array(struct.unpack('%sf' % ntracers, fin.read(ntracers * 4)), dtype=float64)
            age_all = numpy.array(struct.unpack('%sf' % ntracers, fin.read(ntracers * 4)), dtype=float64)
            metallicity_all = numpy.array(struct.unpack('%sf' % ntracers, fin.read(ntracers * 4)), dtype=float64)
            fin.close()
            
            if isnap == 0:
                ctype_evo = np.zeros((ntracers, (len(snaplist))))
                prid_evo = np.zeros((ntracers, (len(snaplist))))
                ttype_evo = np.zeros((ntracers, (len(snaplist))))
                rad_evo = np.zeros((ntracers, (len(snaplist))))
                met_evo = np.zeros((ntracers, (len(snaplist))))
                lz_evo = np.zeros((ntracers, (len(snaplist))))
            
            ctype_evo[:, isnap] = ctype
            ttype_evo[:, isnap] = ttype
            prid_evo[:, isnap] = prid_all
            rad_evo[:, isnap] = rad_all  # * rvir
            met_evo[:, isnap] = metallicity_all
            lz_evo[:, isnap] = lz_all
            lbtime[isnap] = time
            rvirial[isnap] = rvir
            
            if isnap == 0:
                age = age_all
                rad = rad_all  # * rvir
                prid_now = prid_all
                met = metallicity_all
                height = height_all
                lz = lz_all
        
        istars, = np.where((age > 0.) & (rad < 0.03))  # & (abs(height) < 0.005) )
        print("nstars=", size(istars))
        
        star_age = age[istars]
        star_rad = rad[istars]
        star_met = met[istars]
        star_prid = prid_now[istars]
        ctype_evo = ctype_evo[istars, :]
        ttype_evo = ttype_evo[istars, :]
        prid_evo = prid_evo[istars, :]
        rad_evo = rad_evo[istars, :]
        met_evo = met_evo[istars, :]
        lz_evo = lz_evo[istars, :]
        
        lzsf = {}
        metsf = {}
        ttime = {}
        rmax = {}
        nsf = {}
        nw = {}
        for i in range(0, nreclim):
            metsf[i] = []
            lzsf[i] = []
            ttime[i] = []
            rmax[i] = []
            nsf[i] = 0
            nw[i] = 0
        
        wind = False
        for i in range(len(ctype_evo)):
            if wind:
                # do wind phases
                it = np.where((ctype_evo[i, :] == 31) | (ctype_evo[i, :] == 34))
                it = np.clip(it, 0, len(snaplist) - 1)
                
                if (size(it) < nreclim):
                    nsf[size(it)] += 1
                    lzsf[size(it)].extend(lz_evo[i, it].ravel())
                    metsf[size(it)].extend(met_evo[i, it].ravel())
                    ttime[size(it)].extend(lbtime[it])
            
            else:
                # do sf phases
                it, = np.where((ctype_evo[i, :] == 21) | (ctype_evo[i, :] == 24) & (ctype_evo[i, :] == 23))
                
                if size(it) == 0:
                    nsf[0] += 1
                
                if ((size(it) > 0) & (size(it) < nreclim)):
                    nsf[size(it)] += 1
                    lzsf[size(it)].extend(lz_evo[i, it].ravel())
                    metsf[size(it)].extend(met_evo[i, it].ravel())
                    ttime[size(it)].extend(lbtime[it])
                    
                    for j in range(size(it) - 1):
                        rmax[size(it)].extend([rad_evo[i, it[j]:it[j + 1]].max()])
        
        print("rvirial=", rvirial)
        print("nsf=", nsf)
        print("nw=", nw)
        tplot = np.array([])
        lzplot = np.array([])
        for i in range(1, nreclim):
            
            ax3 = figure3.axes[i - 1]
            ax4 = figure4.axes[i - 1]
            if nsf[i] > 0 and (i == 1 or i < nreclim):  # (i == 1 or i == nreclim-1):
                lzsf[i] = np.array(lzsf[i])
                lzsf[i] = np.reshape(lzsf[i], (nsf[i], i))
                metsf[i] = np.array(metsf[i])
                metsf[i] = np.reshape(metsf[i], (nsf[i], i))
                ttime[i] = np.array(ttime[i])
                ttime[i] = np.reshape(ttime[i], (nsf[i], i))
                rmax[i] = np.array(rmax[i])
                rmax[i] = np.reshape(rmax[i], (nsf[i], i - 1))
                
                lzmed = np.median(lzsf[i], axis=0)
                lzmean = np.mean(lzsf[i], axis=0)
                metmed = np.median(metsf[i], axis=0) / 0.0127
                ttmed = np.median(ttime[i], axis=0)
                rmed = np.median(rmax[i], axis=0)
                
                lzstd = np.std(lzsf[i], axis=0)
                rstd = np.std(rmax[i], axis=0)
                
                print("lzmed=", lzmed)
                print("lzmean=", lzmean)
                print("lzstd=", lzstd)
                print("metmed=", metmed)
                print("ttmed=", ttmed)
                print("rmed=", rmed)
                print("rstd=", rstd)
                
                if d == 0:
                    label = '$\\rm{nrec=%d}$' % (i - 1)
                else:
                    label = ''
                
                # axt.set_ylim( [1.,200.] )
                # axt.set_yticks( np.logspace(0., 2., 6) )
                # axt.set_yticklabels( [0, 20, 40, 60, 80, 100] )
                
                # ax.plot( ttmed[::-1], lzmed[::-1], color=s_m.to_rgba(i), marker=symbols[i-1], markersize=6, lw=1., label=label )
                # ax2.plot( ttmed[::-1], metmed[::-1], color=s_m.to_rgba(i), marker=symbols[i-1], markersize=6, lw=1., label=label )
                tplot = np.concatenate((tplot, ttime[i].ravel()))
                lzplot = np.concatenate((lzplot, lzsf[i].ravel()))
                
                if i == nrec:
                    print("ttime[i],lzsf[i]=", ttime[i].ravel(), lzsf[i].ravel())
                    lzsim[d, :] = lzmed[::-1] - lzmed[-1]
                    rmaxsim[d, :] = rmed[::-1] * 1e3
                    ttsim[d, :] = ttmed[::-1]
                    ttsimrmax[d, :] = 0.5 * (ttsim[d, :-1] + ttsim[d, 1:])
                    # tplot = np.array(ttime[i].ravel())
                    # lzplot = np.array(lzsf[i].ravel())
                    print("min, max ttime lzsf=", np.min(tplot), np.max(tplot), np.min(lzplot), np.max(lzplot))
                    
                    if plot2dhist:
                        n, xedges, yedges = np.histogram2d(tplot, lzplot, bins=(20, 20), range=([-0.1, 10.7], [-0.3, 3.2]))
                        xbin = 0.5 * (xedges[:-1] + xedges[1:])
                        ybin = 0.5 * (yedges[:-1] + yedges[1:])
                        xc, yc = np.meshgrid(xbin, ybin)
                        levels = np.linspace(0., np.log10(n.max()), 30)
                        norm = BoundaryNorm(levels[15:], ncolors=cm.get_cmap('Greys').N, clip=True)
                        
                        ax.pcolormesh(xc, yc, np.log10(n.T), cmap=cm.get_cmap('Greys'), rasterized=True, norm=norm)
                    
                    if ploteachsim:
                        ax.plot(ttsim[d, :], lzsim[d, :], color=colors[d], marker=symbols[i - 1], markersize=0, mew=0., lw=1.)
                        ax1.plot(ttsimrmax[d, :], rmaxsim[d, :], color=colors[d], marker=symbols[i - 1], markersize=6, mew=1., mec='w', lw=1.)
                
                # ax2.plot( np.arange(1,i+1), metmed[::-1], color=colors[d], marker=symbols[i-1], markersize=6, lw=1., mew=0.,
                # label=label )
                
                for j in range(np.shape(lzsf[i])[1]):
                    n, bins = np.histogram(ttime[i][:, j], bins=20, range=(0., 13.5), normed=True)
                    tx = 0.5 * (bins[:-1] + bins[1:])
                    ax3.plot(tx, n, lw=lws[j], color=s_m.to_rgba(i))
                    
                    n, bins = np.histogram(lzsf[i][:, j], bins=20, range=(-12., 12.), normed=True)
                    tx = 0.5 * (bins[:-1] + bins[1:])
                    ax4.plot(tx, n, lw=lws[j], color=s_m.to_rgba(i))
                
                # figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ('Au', runs[d].split('_')[1]), position='top right')
                # figure1.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ('Au', runs[d].split('_')[1]), position='top left')
                figure2.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ('Au', runs[d].split('_')[1]), position='bottom right')
                
                # ax.legend( loc='bottom left', frameon=False, prop={'size':5}, ncol=2, numpoints=1 )  # ax1.legend( loc='top   # right',
                # frameon=False, prop={'size':5}, ncol=2, numpoints=1 )  # ax2.legend( loc='top right', frameon=False,   # prop={'size':6}, ncol=1,
                # numpoints=1 )
        
        figure3.fig.savefig('%s/%s/sftracers_timedistrib.pdf' % (outpath, runs[d]), dpi=300)
        figure4.fig.savefig('%s/%s/sftracers_lzdistrib.pdf' % (outpath, runs[d]), dpi=300)
    
    ax.fill_between(np.median(ttsim, axis=0), np.median(lzsim, axis=0) - np.std(lzsim, axis=0), np.median(lzsim, axis=0) + np.std(lzsim, axis=0),
                    alpha=0.2, edgecolor='none', facecolor='k')
    ax.plot(np.median(ttsim, axis=0), np.median(lzsim, axis=0), 'k', lw=2.)
    
    ax1.fill_between(np.median(ttsimrmax, axis=0), np.median(rmaxsim, axis=0) - np.std(rmaxsim, axis=0),
                     np.median(rmaxsim, axis=0) + np.std(rmaxsim, axis=0), alpha=0.2, edgecolor='none', facecolor='k')
    ax1.plot(np.median(ttsimrmax, axis=0), np.median(rmaxsim, axis=0), 'k', lw=2.)
    
    if not ploteachsim:
        i = 0
        for d, halo in enumerate(runs):
            if halo in ['halo_L7', 'halo_L8', 'halo_L4']:
                print("ttsim[d,:], lzsim[d,:]=", ttsim[d, :], lzsim[d, :])
                ax.plot(ttsim[d, :], lzsim[d, :], color=colors[d], marker=symbols[i], markersize=6, mew=1., mec='w', lw=1.,
                        label="$\\rm{Au\,%s}$" % (halo.split('_')[1]))
                i += 1
        
        i = 0
        for d, halo in enumerate(runs):
            if halo in ['halo_L7', 'halo_L8', 'halo_L4']:
                ax1.plot(ttsimrmax[d, :], rmaxsim[d, :], color=colors[d], marker=symbols[i], markersize=6, mew=1., mec='w', lw=1.,
                         label="$\\rm{Au\,%s}$" % (halo.split('_')[1]))
                i += 1
    
    ax.legend(loc='top left', frameon=False, prop={'size': 5}, ncol=1, numpoints=1)
    
    ax1.legend(loc='top left', frameon=False, prop={'size': 5}, ncol=1, numpoints=1)
    
    figure.fig.savefig('%s/plotall/lz_sfphases.pdf' % (outpath), dpi=300)
    figure1.fig.savefig('%s/plotall/rmax_sfphases.pdf' % (outpath), dpi=300)
    figure2.fig.savefig('%s/plotall/gmet_sfphases.pdf' % (outpath), dpi=300)