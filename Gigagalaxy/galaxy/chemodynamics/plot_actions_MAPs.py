import struct

from actions import *
from const import *
from galpy import *
from loadmodules import *
from pylab import *
from util import *

colors = ['r', 'm', 'c', 'b']


def plot_actions(runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols, birthdatafile=None, weight=False, accreted=False, disc_stars=False,
                 atbirth=False, sliceby='age'):
    panels = len(runs)
    
    if sliceby == 'metal':
        upval = array([-0.275, -0.125, 0.025, 0.175, 0.375])
        loval = array([-0.325, -0.175, -0.025, 0.125, 0.325])
        cbname = '[Fe/H]\\,[dex]'
    elif sliceby == 'age':
        loval = array([0., 2., 4., 6., 8., 10.])
        upval = array([2., 4., 6., 8., 10., 13.])
        cbname = 'age\\,[Gyr]'
    else:
        print('No slicing')
    
    if sliceby:
        meanval = 0.5 * (np.array(upval) + np.array(loval))
    
    for d in range(len(runs)):
        
        snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
        if isinstance(snaps, int):
            snaps = [snaps]
        
        print("snaps=", snaps)
        
        dd = dirs[d] + runs[d]
        wpath = outpath + runs[d]
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        
        apath = outpath + runs[d] + '/actions/'
        if not os.path.exists(apath):
            os.makedirs(apath)
        
        fstr = ''
        fa = apath + '/actions_%s%s.dat' % (runs[d], fstr)
        
        fin = open(fa, "rb")
        nst = struct.unpack('i', fin.read(4))[0]
        bid = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64).astype('int64')
        jR = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
        jz = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
        lz = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
        if atbirth:
            jRb = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
            jzb = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
            lzb = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
        fin.close()
        
        if atbirth:
            if birthdatafile:  # read birth data from post-processed file
                stardatafile = outpath + runs[d] + birthdatafile
                attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz']
            else:
                stardatafile = None  # snapshot already contains birth data
                attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz', 'bpos', 'bvel']
        else:
            stardatafile = None
            attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'gz']
        
        s = gadget_readsnap(snaps[-1], snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4, 5], loadonly=attrs)
        sf = load_subfind(snaps[-1], dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'fmc2'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        if birthdatafile and atbirth:
            attrs.append('bpos')
            attrs.append('bvel')
        
        rotmatfile = dd + '/output/rotmatlist_%s.txt' % runs[d]
        galcenfile = dd + '/output/galcen_%s.txt' % runs[d]
        g = parse_particledata(s, sf, attrs, rotmatfile, galcenfile, radialcut=sf.data['frc2'][0], stardatafile=stardatafile)
        g.prep_data()
        
        sdata = g.sgdata['sdata']
        
        nstars = np.int_(g.numpart[4])
        print("number of stars=", nstars)
        
        # plotting
        
        alphafe = sdata['alpha'] - sdata['Fe']
        iron = sdata['Fe']
        star_age = sdata['age']
        # eps2 = sdata['eps2']
        smass = sdata['mass']
        star_height = sdata['pos'][:, 0]
        
        aval = 0.03
        aval2 = 0.1
        zcut = 0.005
        nbin = 20
        sty = ''
        
        bra = [-5000., 5000.]
        yra = [0.01, 0.5]
        figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=2, npanels=2 * nrows, left=0.12, right=0.88, bottom=0.2, top=0.96)
        figure.set_figure_layout()
        figure.set_axis_limits_and_aspect(xlim=bra, ylim=[0., 0.25], logyaxis=False)
        figure.set_axis_locators(xmajloc=2000., xminloc=1000.)  # , ymajloc=0.1, yminloc=0.05 )
        # figure.set_axis_locators( xmajloc=5., xminloc=5.)#, ymajloc=0.1, yminloc=0.05 )
        figure.set_axis_labels(xlabel="$\\rm{L_z\,[kpc \, km \, s^{-1}]}$", ylabel="$\\rm{df}$")
        figure.set_fontsize(5)
        
        figure2 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=2, npanels=2 * nrows, left=0.12, right=0.88, bottom=0.2, top=0.96)
        figure2.set_figure_layout()
        figure2.set_axis_limits_and_aspect(xlim=[0., 250.], ylim=yra, logyaxis=True)
        figure2.set_axis_locators(xmajloc=50., xminloc=25.)  # , ymajloc=0.1, yminloc=0.05 )
        figure2.set_axis_labels(xlabel="$\\rm{J_R\,[kpc \, km \, s^{-1}]}$", ylabel="$\\rm{df}$")
        figure2.set_fontsize(5)
        
        figure3 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=2, npanels=2 * nrows, left=0.12, right=0.88, bottom=0.2, top=0.96)
        figure3.set_figure_layout()
        figure3.set_axis_limits_and_aspect(xlim=[0., 250.], ylim=yra, logyaxis=True)
        figure3.set_axis_locators(xmajloc=50., xminloc=25.)  # , ymajloc=0.1, yminloc=0.05 )
        figure3.set_axis_labels(xlabel="$\\rm{J_z\,[kpc \, km \, s^{-1}]}$", ylabel="$\\rm{df}$")
        figure3.set_fontsize(5)
        
        cmap = plt.get_cmap("viridis")
        s_m = figure.get_scalar_mappable_for_colorbar(meanval, cmap)
        
        if sliceby == 'metal':
            sliceval = iron
        elif sliceby == 'age':
            sliceval = star_age
        
        for j in range(len(meanval)):
            ii, = np.where((alphafe < aval) & (sliceval < upval[j]) & (sliceval > loval[j]) & (star_age < 12.) & (abs(star_height) < zcut))
            # high alpha sequence
            jj, = np.where((alphafe > aval2) & (sliceval < (upval[j])) & (sliceval > (loval[j])) & (star_age < 12.) & (abs(star_height) < zcut))
            weights1 = np.ones(len(ii)) / len(ii)
            weights2 = np.ones(len(jj)) / len(jj)
            # present day
            n1, edges = np.histogram(lz[ii], bins=nbin, range=bra, weights=weights1)
            figure.fig.axes[0].plot(0.5 * (edges[1:] + edges[:-1]), n1, dashes=(2, 2), color=s_m.to_rgba(meanval[j]), linewidth=0.7)
            n2, edges = np.histogram(lz[jj], bins=nbin, range=bra, weights=weights2)
            figure.fig.axes[1].plot(0.5 * (edges[1:] + edges[:-1]), n2, dashes=(2, 2), color=s_m.to_rgba(meanval[j]), linewidth=0.7)
            n3, edges3 = np.histogram(jR[ii], bins=nbin, range=[0., 250.], weights=weights1)
            figure2.fig.axes[0].semilogy(0.5 * (edges3[1:] + edges3[:-1]), n3, dashes=(2, 2), color=s_m.to_rgba(meanval[j]), linewidth=0.7)
            n4, edges3 = np.histogram(jR[jj], bins=nbin, range=[0., 250.], weights=weights2)
            figure2.fig.axes[1].semilogy(0.5 * (edges3[1:] + edges3[:-1]), n4, dashes=(2, 2), color=s_m.to_rgba(meanval[j]), linewidth=0.7)
            n5, edges5 = np.histogram(jz[ii], bins=nbin, range=[0., 250.], weights=weights1)
            figure3.fig.axes[0].semilogy(0.5 * (edges5[1:] + edges5[:-1]), n5, dashes=(2, 2), color=s_m.to_rgba(meanval[j]), linewidth=0.7)
            n6, edges5 = np.histogram(jz[jj], bins=nbin, range=[0., 250.], weights=weights2)
            figure3.fig.axes[1].semilogy(0.5 * (edges5[1:] + edges5[:-1]), n6, dashes=(2, 2), color=s_m.to_rgba(meanval[j]), linewidth=0.7)
            
            if atbirth:
                kk, = np.where((alphafe < aval) & (sliceval < upval[j]) & (sliceval > loval[j]) & (star_age < 12.) & (abs(star_height) < zcut))
                ll, = np.where((alphafe > aval2) & (sliceval < upval[j]) & (sliceval > loval[j]) & (star_age < 12.) & (abs(star_height) < zcut))
                weights3 = np.ones(len(kk)) / len(kk)
                weights4 = np.ones(len(ll)) / len(ll)
                nb1, edges = np.histogram(lzb[kk], bins=nbin, range=bra, weights=weights3)
                figure.fig.axes[0].plot(0.5 * (edges[1:] + edges[:-1]), nb1, linestyle='-', color=s_m.to_rgba(meanval[j]), linewidth=0.7)
                nb2, edges = np.histogram(lzb[ll], bins=nbin, range=bra, weights=weights4)
                figure.fig.axes[1].plot(0.5 * (edges[1:] + edges[:-1]), nb2, linestyle='-', color=s_m.to_rgba(meanval[j]), linewidth=0.7)
                
                nb3, edges3 = np.histogram(jRb[kk], bins=nbin, range=[0., 250.], weights=weights3)
                figure2.fig.axes[0].semilogy(0.5 * (edges3[1:] + edges3[:-1]), nb3, linestyle='-', color=s_m.to_rgba(meanval[j]), linewidth=0.7)
                nb4, edges3 = np.histogram(jRb[ll], bins=nbin, range=[0., 250.], weights=weights4)
                figure2.fig.axes[1].semilogy(0.5 * (edges3[1:] + edges3[:-1]), nb4, linestyle='-', color=s_m.to_rgba(meanval[j]), linewidth=0.7)
                
                nb5, edges5 = np.histogram(jzb[kk], bins=nbin, range=[0., 250.], weights=weights3)
                figure3.fig.axes[0].semilogy(0.5 * (edges5[1:] + edges5[:-1]), nb5, linestyle='-', color=s_m.to_rgba(meanval[j]), linewidth=0.7)
                nb6, edges5 = np.histogram(jzb[ll], bins=nbin, range=[0., 250.], weights=weights4)
                figure3.fig.axes[1].semilogy(0.5 * (edges5[1:] + edges5[:-1]), nb6, linestyle='-', color=s_m.to_rgba(meanval[j]), linewidth=0.7)
        
        # Fig 1 labels
        figure.fig.axes[1].legend(loc='upper right', frameon=False, prop={'size': 7})
        
        n, edges = np.histogram(lzb[kk], bins=nbin, range=bra)
        figure.fig.axes[0].plot([], [], linestyle='-', color='k', linewidth=0.7, label="$\\rm{R_{b}}$")
        n, edges = np.histogram(lz[ii], bins=nbin, range=bra)
        figure.fig.axes[0].plot([], [], dashes=(2, 2), color='k', linewidth=0.7, label="$\\rm{R_{z=0}}$")
        
        figure.fig.axes[0].legend(loc='upper right', frameon=False, prop={'size': 7})
        
        title = "$\\rm{Au \,%s}$" % runs[d].split('_')[1]
        figure.set_panel_title(panel=0, title=title, position='top left')
        
        figure.set_colorbar([meanval[0], meanval[-1]], '$\\rm{%s}$' % cbname, meanval, cmap=cmap, labelsize=3)
        
        figname1 = '%s/lz_MAPs_%s_histnew%s%s_sliceby%s.%s' % (apath, sty, fstr, runs[d], sliceby, suffix)
        print("figname1=", figname1)
        
        figure.fig.savefig(figname1)
        
        # Fig 2 labels
        figure2.fig.axes[1].legend(loc='upper right', frameon=False, prop={'size': 7})
        
        n, edges = np.histogram(jRb[kk], bins=nbin, range=[0., 250.])
        figure2.fig.axes[0].semilogy([], [], linestyle='-', color='k', linewidth=0.7, label="$\\rm{R_{b}}$")
        n, edges = np.histogram(jR[ii], bins=nbin, range=[0., 250.])
        figure2.fig.axes[0].semilogy([], [], dashes=(2, 2), color='k', linewidth=0.7, label="$\\rm{R_{z=0}}$")
        
        figure2.fig.axes[0].legend(loc='upper right', frameon=False, prop={'size': 7})
        
        title = "$\\rm{Au \,%s}$" % runs[d].split('_')[1]
        figure2.set_panel_title(panel=0, title=title, position='top left')
        # figure2.set_colorbar([meanfe[0], meanfe[-1]],'$\\rm{[Fe/H]\\,[dex]}$',meanfe,cmap=cmap,labelsize=3)
        figure2.set_colorbar([meanval[0], meanval[-1]], '$\\rm{%s}$' % cbname, meanval, cmap=cmap, labelsize=3)
        
        figname2 = '%s/jR_MAPs_%s_histnew%s%s_sliceby%s.%s' % (apath, sty, fstr, runs[d], sliceby, suffix)
        print("figname2=", figname2)
        
        figure2.fig.savefig(figname2)
        
        # Fig 3 labels
        figure3.fig.axes[1].legend(loc='upper right', frameon=False, prop={'size': 7})
        
        n, edges = np.histogram(jRb[kk], bins=nbin, range=[0., 250.])
        figure3.fig.axes[0].semilogy([], [], linestyle='-', color='k', linewidth=0.7, label="$\\rm{R_{b}}$")
        
        n, edges = np.histogram(jR[ii], bins=nbin, range=[0., 250.])
        figure3.fig.axes[0].semilogy([], [], dashes=(2, 2), color='k', linewidth=0.7, label="$\\rm{R_{z=0}}$")
        
        figure3.fig.axes[0].legend(loc='upper right', frameon=False, prop={'size': 7})
        
        title = "$\\rm{Au \,%s}$" % runs[d].split('_')[1]
        figure3.set_panel_title(panel=0, title=title, position='top left')
        # figure3.set_colorbar([meanfe[0], meanfe[-1]],'$\\rm{[Fe/H]\\,[dex]}$',meanfe,cmap=cmap,labelsize=3)
        figure3.set_colorbar([meanval[0], meanval[-1]], '$\\rm{%s}$' % cbname, meanval, cmap=cmap, labelsize=3)
        
        figname3 = '%s/jz_MAPs_%s_histnew%s%s_sliceby%s.%s' % (apath, sty, fstr, runs[d], sliceby, suffix)
        print("figname3=", figname3)
        
        figure3.fig.savefig(figname3)


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx