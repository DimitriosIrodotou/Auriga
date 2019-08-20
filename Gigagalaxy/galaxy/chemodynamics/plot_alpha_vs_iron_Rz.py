from const import *
from loadmodules import *
from pylab import *
from util import *

colors = ['r', 'm', 'c', 'b']


def plot_alpha_vs_iron_Rz(runs, dirs, outpath, outputlistfile, redshift, zcut=[[1., 0.5, 0.], [2., 1., 0.5]],
                          rcut=[[3., 7., 11., 15.], [5., 9., 13., 17.]], accretedfiledir=None, birthdatafile=None, alphaelement=None, atbirth=False,
                          disc_stars=False):
    zbinlo = zcut[0]
    zbinhi = zcut[1]
    rbinlo = rcut[0]
    rbinhi = rcut[1]
    
    nrows = len(zbinlo)
    ncols = len(rbinlo)
    panels = nrows * ncols
    
    feran = [-0.9, 0.9]
    alran = [-0.08, 0.25]
    
    for d in range(len(runs)):
        
        figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, bottom=0.2, top=0.98)
        figure.set_figure_layout()
        figure.set_axis_limits_and_aspect(xlim=feran, ylim=alran, logaxis=False)
        figure.set_axis_locators(xminloc=0.1, ymajloc=0.1, yminloc=0.05)
        figure.set_axis_labels(xlabel="$\\rm{[Fe/H]}$", ylabel=r"$\rm{[\alpha/Fe]}$")
        figure.set_fontsize(8)
        
        dd = dirs[d] + runs[d]
        wpath = outpath + runs[d] + '/metals/'
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], redshift))
        
        print("doing %s, snap=%03d" % (wpath, snap))
        
        if atbirth:
            if birthdatafile:  # read birth data from post-processed file
                stardatafile = outpath + runs[d] + '/rm4/' + birthdatafile
                attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'gz']
            else:
                stardatafile = None  # snapshot already contains birth data
                attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'gz', 'bpos', 'bvel']
        else:
            stardatafile = None
            attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'gz']
        
        if disc_stars:
            attrs.append('pot')
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        if accretedfiledir:
            accretedfile = accretedfiledir + runs[d] + '/%sstarID_accreted_all.dat' % runs[d]
        else:
            accretedfile = None
        if birthdatafile and atbirth:
            attrs.append('bpos')
            attrs.append('bvel')
        g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0], accretedfile=accretedfile, stardatafile=stardatafile)
        g.prep_data()
        
        sdata = g.sgdata['sdata']
        
        if accretedfile:
            asind = g.select_accreted_stars(accreted=True)
        else:
            asind = np.arange(len(sdata['mass']))
        
        accreted = True
        if accreted:
            filename = '%s/lists/%sstarID_accreted_all_newmtree.dat' % (dist[d], runs[d])
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
            
            ids = sdata['id'].astype('int64')
            pindy, = np.where((pkmassid == 501))
        
        if disc_stars:
            eps2 = sdata['eps2'][asind]
        smass = sdata['mass'][asind]
        star_age = sdata['age'][asind]
        iron = sdata['Fe'][asind]
        if alphaelement:
            alphaelem = sdata[alphaelement][asind]
        else:
            alphaelem = sdata['alpha'][asind]
        
        alphaelem -= iron
        if atbirth:
            star_pos = sdata['bpos'][asind]
        else:
            star_pos = sdata['pos'][asind]
        
        radius = np.sqrt((star_pos[:, 1:] ** 2).sum(axis=1))
        height = star_pos[:, 0]
        
        binx = 80
        biny = 80
        conv = (binx / (feran[1] - feran[0])) * (biny / (alran[1] - alran[0]))
        vmin = 0.005
        vmax = 0.5
        if accretedfile:
            vmin /= 10.
            vmax /= 10.
        
        pnum = 0
        for i in range(len(zbinlo)):
            for j in range(len(rbinlo)):
                ax = figure.axes[pnum]
                pnum += 1
                
                if disc_stars:
                    jj, = np.where(
                        (radius * 1e3 > rbinlo[j]) & (radius * 1e3 < rbinhi[j]) & (height * 1e3 > zbinlo[i]) & (height * 1e3 < zbinhi[i]) & (
                                    eps2 > 0.7))
                else:
                    jj, = np.where((radius * 1e3 > rbinlo[j]) & (radius * 1e3 < rbinhi[j]) & (height * 1e3 > zbinlo[i]) & (height * 1e3 < zbinhi[i]))
                
                ax.hist2d(iron[jj], alphaelem[jj], bins=(binx, biny), range=(feran, alran), weights=smass[jj] * conv * 1e-1, normed=True,
                          rasterized=True, cmap=cm.get_cmap('viridis'), cmin=0.001)  # , vmin=vmin, vmax=vmax )
                
                figure.set_colorbar([vmin, vmax], r'$\rm{ [10^{10}\,M_{\odot}\, dex^{-2}] }$', [0.1, 0.2, 0.3, 0.4], cmap=cm.get_cmap('viridis'),
                                    fontsize=8, labelsize=8, orientation='vertical')
                
                nn, xedges, yedges = np.histogram2d(iron[jj], alphaelem[jj], bins=(40, 40), range=(feran, alran), weights=star_age[jj])
                n, xedges, yedges = np.histogram2d(iron[jj], alphaelem[jj], bins=(40, 40), range=(feran, alran))
                nn /= n
                nn[nn == 0.] = -1.
                
                if accreted:
                    ind = np.in1d(ids[jj], idacc[pindy])
                
                xbin = 0.5 * (xedges[:-1] + xedges[1:])
                ybin = 0.5 * (yedges[:-1] + yedges[1:])
                xc, yc = np.meshgrid(xbin, ybin)
                
                levels = np.linspace(0., 12., 7)
                cont = ax.contour(xc, yc, nn.T, colors='r', levels=levels)
                ax.clabel(cont, inline=1, fmt='%.1f', fontsize=5)
                
                # ax.scatter( iron[jj][ind], alphaelem[jj][ind], s=2., c="m", marker='.', alpha=0.5, edgecolor='none' )
                
                if pnum == 0:
                    title = "$\\rm{Au \,%s ,\, t=%2.1f}$" % (runs[d].split('_')[1], time)
                    figure.set_panel_title(panel=pnum, title=title, position='bottom left')
                
                ax.text(0.45, 0.77, "$\\rm{%.1f < z < %.1f}$ \n$\\rm{%.0f < R < %.0f}$" % (zbinlo[i], zbinhi[i], rbinlo[j], rbinhi[j]), color='r',
                        transform=ax.transAxes, fontsize=8)
        
        figure.reset_axis_limits()
        
        name = ''
        if atbirth:
            name += '_birth'
        if disc_stars:
            name += '_disc'
        if accretedfiledir:
            name += '_accreted'
        if accreted:
            name += '_acc'
        
        figname1 = '%s/alphavsiron_stars_histweight_Rz%s%03d.pdf' % (wpath, name, snap)
        figure.fig.savefig(figname1)