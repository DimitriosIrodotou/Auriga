from const import *
from loadmodules import *
from pylab import *
from util import *

lines = ['-', ':']
marker = ['o', '^', 'd', 's']
toinch = 0.393700787
colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
markers = ['o', '^', 'd', 's']


def plot_AVR(runs, dirs, outpath, nrows, ncols, outputlistfile, suffix, ddir='z', rbinfac=[0.5, 1., 2., 3.], birthdatafile=None, restest=False,
             accreted=False):
    if restest:
        panels = len(runs) / len(restest)
    else:
        panels = nrows * ncols
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[0.0, 11.], ylim=[0., 120.0])
    figure.set_fontsize(8.)
    figure.set_axis_labels(xlabel="$\\rm{Age\\,[Gyr]}$", ylabel="$\\rm \sigma _{%s}\,[km\,s^{-1}]$" % ddir)
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        if not restest:
            ax = figure.axes[d]
        else:
            ax = figure.axes[int(d / len(restest))]
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], [0.]))
        
        print("Doing dir %s, snap %d" % (dd, snap))
        
        if birthdatafile:  # read birth data from post-processed file
            stardatafile = outpath + runs[d] + '/rm4/' + birthdatafile
            attrs = ['pos', 'vel', 'mass', 'age', 'id']
        else:
            stardatafile = None  # snapshot already contains birth data
            attrs = ['pos', 'vel', 'mass', 'age', 'id', 'bpos', 'bvel']
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        attrs = ['pos', 'vel', 'mass', 'age', 'id', 'bpos', 'bvel']
        g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0], stardatafile=stardatafile)
        g.prep_data()
        g.sort_birthIDs_fromfile()
        # if accretedfile:
        #       asind = g.select_accreted_stars(accreted=True)
        
        sdata = g.sgdata['sdata']
        smass = sdata['mass']
        star_age = sdata['age']
        pos = sdata['pos']
        vel = sdata['vel']
        bpos = sdata['bpos']
        bvel = sdata['bvel']
        bid = sdata['bid']
        
        srxy = np.sqrt((pos[:, 1:] ** 2).sum(axis=1))
        bsrxy = np.sqrt((bpos[:, 1:] ** 2).sum(axis=1))
        svel = np.zeros(len(star_age))
        bsvel = np.zeros(len(bid))
        if ddir == 'z':
            svel[:] = vel[:, 0]
            bsvel[:] = bvel[:, 0]
        elif ddir == 'R':
            svel[:] = (pos[:, 1] * vel[:, 1] + pos[:, 2] * vel[:, 2]) / srxy
            bsvel[:] = (bpos[:, 1] * bvel[:, 1] + bpos[:, 2] * bvel[:, 2]) / bsrxy
        elif ddir == 'theta':
            svel[:] = (pos[:, 1] * vel[:, 2] - pos[:, 1] * vel[:, 2]) / srxy
            bsvel[:] = (bpos[:, 1] * bvel[:, 2] - bpos[:, 1] * bvel[:, 2]) / bsrxy
            if np.mean(svel) < 0.:
                svel *= -1.
                bvel *= -1.
        else:
            raise ValueError('invalid choice for ddir keyword! Choose either z, R or theta.')
        
        wpath = outpath + runs[d]
        
        file5 = wpath + '/radprof/fit_table.txt'
        f5 = open(file5, 'r')
        a1 = f5.readline()
        a2 = f5.readline()
        data = a2.split()
        f5.close()
        r_d = float(data[5])
        galradz0 = 0.1 * float(data[2])
        rcen = np.array(rbinfac) * r_d
        rhalfwidth = 1.
        nshells = 14
        
        for j in range(len(rcen)):
            rlo = rcen[j] - rhalfwidth
            rhi = rcen[j] + rhalfwidth
            
            jj, = np.where((srxy * 1e3 > rlo) & (srxy * 1e3 < rhi) & (abs(pos[:, 0]) < 0.0005))
            jb, = np.where((bsrxy * 1e3 > rlo) & (bsrxy * 1e3 < rhi) & (abs(bpos[:, 0]) < 0.0005))
            
            avg, edge = np.histogram(star_age[jj], bins=nshells, range=(-0.5, 13.), weights=(smass[jj] * svel[jj]))
            # avgb, edge = np.histogram(star_age[jb], bins=nshells, range=(-0.5, 13.), weights=(smass[jb] * bsvel[jb]))
            avgb, edge = np.histogram(star_age[jb], bins=nshells, range=(-0.5, 13.), weights=(bsvel[jb]))
            
            masstot, edge = np.histogram(star_age[jj], bins=nshells, range=(-0.5, 13.), weights=smass[jj])
            # masstotb, edge = np.histogram(star_age[jb], bins=nshells, range=(-0.5, 13.), weights=smass[jb])
            masstotb, edge = np.histogram(star_age[jb], bins=nshells, range=(-0.5, 13.))
            
            xbin = np.zeros(len(edge) - 1)
            xbin[:] = 0.5 * (edge[:-1] + edge[1:])
            
            avg /= masstot
            avgb /= masstotb
            
            binind = np.digitize(star_age[jj], xbin) - 1
            sigma, edge = np.histogram(star_age[jj], bins=nshells, range=(-0.5, 13.), weights=(smass[jj] * (svel[jj] - avg[binind]) ** 2))
            binind = np.digitize(star_age[jb], xbin) - 1
            # sigmab, edge = np.histogram(star_age[jb], bins=nshells, range=(-0.5, 13.), weights=(smass[jb] * (bsvel[jb] - avgb[
            # binind])**2))
            sigmab, edge = np.histogram(star_age[jb], bins=nshells, range=(-0.5, 13.), weights=((bsvel[jb] - avgb[binind]) ** 2))
            
            sigma /= masstot
            sigmab /= masstotb
            sigma = np.sqrt(sigma)
            sigmab = np.sqrt(sigmab)
            
            color = colors[j]
            lw = 1.
            
            ax.plot(xbin, sigma, color=color, lw=lw, linestyle='-', label='$\\rm{R = %2.0f R_{d}}$' % (rbinfac[j]))
            ax.plot(xbin, sigmab, color=color, lw=lw, dashes=(2, 2), label='')
            
            ax.legend(loc='upper left', frameon=False, prop={'size': 6}, ncol=1)
            
            if not restest:
                figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ("Au", runs[d].split('_')[1]), position='bottom right', color=color,
                                       fontsize=10)
    
    if restest:
        figure.fig.savefig("%s/%s%03d_%s.%s" % (outpath, "age-veldisprelation_restest", snap, ddir, suffix))
    else:
        figure.fig.savefig("%s/%s%03d_%s.%s" % (outpath, "age-veldisprelation", snap, ddir, suffix))