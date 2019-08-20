from const import *
from loadmodules import *
from matplotlib.colors import LogNorm
from pylab import *
from util import *

mpl.rcParams['image.cmap'] = 'gray_r'

colors = ['k.', 'b.', 'r.', 'g.', 'y.', 'm.', 'c.']


def plot_alphavsiron_stars(runs, dirs, outpath, outputlistfile, suffix, nrows, ncols, alphaelement=None, disc_stars=False, accreted=False,
                           plot_obs=False):
    panels = nrows * ncols
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, left=0.22, right=0.98, bottom=0.2, top=0.96)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[-2., 1.], ylim=[-0.2, 0.5], logaxis=False)
    figure.set_axis_locators(xminloc=0.1, ymajloc=0.2, yminloc=0.05)
    figure.set_axis_labels(xlabel="$\\rm{[Fe/H]}$", ylabel=r"$\rm{[\alpha/Fe]}$")
    figure.set_fontsize(5)
    
    figure2 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure2.set_figure_layout()
    figure2.set_axis_limits_and_aspect(xlim=[0.1, 13.9], ylim=[-0.49, 0.59], logaxis=False)
    figure2.set_axis_locators(xminloc=1., ymajloc=0.2, yminloc=0.05)
    figure2.set_axis_labels(xlabel="$\\rm{Age\; (Gyr)}$", ylabel=r"$\rm{[\alpha/Fe]}$")
    figure2.set_fontsize(5)
    
    figure3 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure3.set_figure_layout()
    figure3.set_axis_limits_and_aspect(xlim=[-0.2, 0.5], ylim=[0.0, 20.], logaxis=False)
    figure3.set_axis_locators(xminloc=0.1, xmajloc=0.1, ymajloc=5., yminloc=1.)
    figure3.set_axis_labels(xlabel="$\\rm{[\alpha/Fe]}$", ylabel="$\\rm{f([\alpha/Fe])}$")
    figure3.set_fontsize(5)
    
    if plot_obs:
        # load all the observational data
        gratton = np.loadtxt('./data/Gratton.txt')
        reddy = np.loadtxt('./data/Reddy.txt')
        bensby = np.loadtxt('./data/Bensby.txt', usecols=(1, 2))  # all star types
        bensby[:, 1] -= bensby[:, 0]
        ramya = np.loadtxt('./data/Ramya.txt')
        obs_dat = np.concatenate([gratton, reddy, bensby, ramya])  # [Fe/H], [O/Fe]
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax = figure.fig.axes[d]
        ax2 = figure2.fig.axes[d]
        ax3 = figure3.fig.axes[d]
        
        snap = select_snapshot_number.select_snapshot_number(outputlistfile[d], [0.])
        
        if plot_obs:
            ax.scatter(obs_dat[:, 0], obs_dat[:, 1], s=1.5, linewidths=0, c='r', marker='o', alpha=0.7, zorder=100)
        
        print("Doing dir %s snap %d." % (dd, snap))
        
        wpath = outpath  # + runs[d]
        
        attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet']
        if disc_stars:
            attrs.appens('pot')
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonly=attrs, loadonlytype=[4])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        
        galrad = 0.1 * sf.data['frc2'][0]
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        g = parse_particledata(s, sf, attrs, radialcut=galrad)
        g.prep_data()
        
        if accreted:
            filename = '%s/lists/%sstarID_accreted_all_newmtree.dat' % (dirs[d], runs[d])
            fin = open(filename, "rb")
            nacc = struct.unpack('i', fin.read(4))[0]
            idacc = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
            subacc = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
            snapacc = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
            aflag = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)))
            fofflag = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)))
            rootid = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
            pkmassid = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
            fin.close()
            
            first_prog = g.return_pkmassprogid_mostmassive(pkmassid)
        
        sdata = g.sgdata['sdata']
        if disc_stars:
            ii, = np.where(sdata['eps2'] > 0.7)
            name = '_disc'
        else:
            ii = np.arange(len(sdata['mass']))
            name = ''
        
        if accreted:
            name += '_acc'
        
        smass = sdata['mass'][ii]
        age_s = sdata['age'][ii]
        iron = sdata['Fe'][ii]
        if alphaelement:
            alphaelem = sdata[alphaelement][ii]
        else:
            alphaelem = sdata['alpha'][ii]
        alphaelem -= iron
        if accreted:
            ids = sdata['id'][ii]
        
        feran = [-2., 1.]
        alran = [-0.2, 0.5]
        # feran = [-1.1, 0.9]
        # alran = [-0.19, 0.39]
        
        nn, xedges, yedges = np.histogram2d(iron, alphaelem, bins=(160, 160), range=(feran, alran))
        ax.hist2d(iron, alphaelem, bins=(100, 200), range=(feran, alran), weights=smass, normed=False, rasterized=True, norm=LogNorm())
        alpha1, edges = np.histogram(iron, bins=40, range=feran, weights=alphaelem * smass)
        alpha1norm, edges = np.histogram(iron, bins=40, range=feran, weights=smass)
        alpha1 /= alpha1norm
        abin = np.zeros(40)
        abin[:] = 0.5 * (edges[1:] + edges[:-1])
        # ax.plot( abin, alpha1, linestyle='-', linewidth=1., color='m' )
        
        ax2.hist2d(age_s, alphaelem, bins=(100, 200), range=([0., 14.], alran), weights=smass, normed=False, rasterized=True, norm=LogNorm())
        ydatatot, edges = np.histogram(alphaelem, bins=40, weights=smass, range=alran, normed=True)
        xdatatot = 0.5 * (edges[1:] + edges[:-1])
        ax3.plot(xdatatot, ydatatot, color='k', lw=1.0)
        
        if disc_stars:
            ii, = np.where((sdata['eps2'] > 0.7))
            ydatadisc, edges = np.histogram(alphaelem[ii], bins=40, weights=smass[ii], range=alran, normed=True)
            xdatadisc = 0.5 * (edges[1:] + edges[:-1])
            ax3.plot(xdatadisc, ydatadisc, color='b', lw=1.0)
        
        if accreted:
            for l in range(len(first_prog)):
                if first_prog[l] == 501:
                    pindy, = np.where((pkmassid == first_prog[l]))
                    sindy = np.in1d(ids, idacc[pindy])
                    ax.hist2d(iron[sindy], alphaelem[sindy], bins=(100, 200), range=(feran, alran), weights=smass[sindy], normed=False,
                              rasterized=True, norm=LogNorm(), cmap=cm.get_cmap('viridis'), alpha=0.5)
        
        figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ('Au', runs[d].split('_')[1]), position='bottom left')
        figure2.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ('Au', runs[d].split('_')[1]), position='bottom left')
        figure3.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ('Au', runs[d].split('_')[1]), position='bottom left')
    
    figure.reset_axis_limits()
    figure2.reset_axis_limits()
    figure3.reset_axis_limits()
    
    figname1 = '%s/alphavsiron_stars_histweight%s.%s' % (wpath, name, suffix)
    print("figname1=", figname1)
    
    figure.fig.savefig(figname1)
    figure2.fig.savefig('%s/agevsmetal_stars_histweight%s.%s' % (wpath, name, suffix))
    figure3.fig.savefig('%s/alpha_stars_1dhistweight%s.%s' % (wpath, name, suffix))