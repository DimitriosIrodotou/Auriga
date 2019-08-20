from const import *
from loadmodules import *
from pylab import *
from scipy.ndimage.filters import gaussian_filter
from util import *

toinch = 0.393700787
Gcosmo = 43.0071
parsec = 3.08567758e16


def plot_angmom_fluxtensor(runs, dirs, outpath, outputlistfile, redshift, suffix, nrows, ncols):
    panels = ncols * nrows
    
    xlim = [-24., 24.0]
    ylim = [-24.0, 24.0]
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=xlim, ylim=ylim)
    figure.set_axis_locators(xminloc=5., xmajloc=10., yminloc=5., ymajloc=10.)
    figure.set_axis_labels(xlabel="$\\rm{X [kpc]}$", ylabel="$\\rm{Y [kpc]}$")
    figure.set_fontsize(5)
    
    figure2, ax2 = plt.subplots()
    ax2.set_xlim([0., 24.0])
    ax2.set_ylim([-200.0, 200.0])
    ax2.set_xlabel('R \\,[kpc]', fontsize=15)
    ax2.set_ylabel('$\\rm \Lambda _{zr} \, [M_{\odot} Gyr^{-2}]$', fontsize=15)
    
    parameters = np.linspace(0., len(runs), len(runs) + 1)
    cmap = plt.get_cmap('gnuplot')  # mpl.cm.jet
    s_m = figure.get_scalar_mappable_for_colorbar(parameters, cmap)
    
    for d in range(len(runs)):
        wpath = outpath + runs[d] + '/'
        
        dd = dirs[d] + runs[d]
        ax = figure.axes[d]
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], redshift))
        print("Doing dir %s snap %d." % (dd, snap))
        attrs = ['pos', 'vel', 'mass', 'age', 'pot']
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
        g.prep_data()
        
        sdata = g.sgdata['sdata']
        smass = s.data['mass']
        eps2 = sdata['eps2']
        star_age = sdata['age']
        pos = sdata['pos']
        vel = sdata['vel']
        
        jj, = np.where((abs(pos[:, 0]) < 0.005))
        
        y = pos[jj, 2]
        x = pos[jj, 1]
        star_radius = np.sqrt((pos[jj, 1:] ** 2).sum(axis=1))
        svphi = (vel[jj, 1] * pos[jj, 2] - vel[jj, 2] * pos[jj, 1]) / (star_radius)
        svrad = (vel[jj, 1] * pos[jj, 1] + vel[jj, 2] * pos[jj, 2]) / (star_radius)
        
        nstars = len(smass)
        
        lambda_zr = smass[jj] * (svrad * svphi)
        
        num = 40
        nrbin = 30
        xs = np.linspace(xlim[0], xlim[1], num)
        ys = np.copy(xs)
        hrange = [xlim, ylim]
        
        n, xedges, yedges = np.histogram2d(x * 1e3, y * 1e3, bins=num, range=hrange)
        lzrtot, xedges, yedges = np.histogram2d(x * 1e3, y * 1e3, bins=num, weights=lambda_zr, range=hrange)
        
        dx = (xlim[1] - xlim[0]) / float(num)
        dz = 0.01
        da = dx * dx
        dv = da * dz * 1e6
        lzr_mean = (lzrtot * 1e10 / n) / dv
        ax.imshow(np.flipud(lzr_mean.T), aspect='auto', interpolation='nearest', cmap=cmap, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], vmin=-10000.,
                  vmax=10000.)
        # Overplot contours of stellar over-density
        xbin = np.zeros(num)
        xbin[:] = 0.5 * (xedges[:-1] + xedges[1:])
        x, y = np.meshgrid(xbin, xbin)
        # smooth grid
        h_g = np.log10(gaussian_filter(n, .5) * (num / (xedges[-1] - xedges[0])) ** 2)
        h_gl = np.ravel(h_g)
        sdav = np.zeros(len(h_gl))
        
        rxy = np.zeros(shape(h_g))
        rxy[:, :] = sqrt(x[:, :] ** 2 + y[:, :] ** 2)
        
        rs = np.linspace(0., xedges[-1], nrbin)
        rxyl = np.ravel(rxy)
        binind = np.digitize(rxyl, rs)
        for i in range(len(sdav)):
            sdav[i] = np.mean(h_gl[np.where(binind == binind[i])])
        
        h_glsub = h_gl - sdav
        h_gsub = np.reshape(h_glsub, np.shape(h_g))
        
        levels = [0.025, 0.05, 0.1, 0.2, 0.3]
        ax.contour(x, y, np.flipud(h_gsub), levels, colors='w')
        
        nrad, rad = np.histogram(star_radius * 1e3, bins=num, range=[0.0, xlim[1]])
        lzrrad, rad = np.histogram(star_radius * 1e3, bins=num, weights=lambda_zr, range=[0.0, xlim[1]])
        
        da = np.zeros(len(nrad))
        for i in range(len(nrad)):
            da[i] = np.pi * (rad[i + 1] ** 2 - rad[i] ** 2)
        dv = da * dz * 1e6
        
        lzrrad_mean = np.log10(((lzrrad * 1e10 / nrad) / dv))
        
        srad = np.zeros(len(rad) - 1)
        srad[:] = 0.5 * (rad[:-1] + rad[1:])
        ax2.plot(srad, lzrrad_mean, linestyle='-', color=s_m.to_rgba(d), lw=0.5)
        figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ("Au", runs[d].split('_')[1]), position='top right', pef=True)
    
    figure.set_colorbar([-5., 5.], '$\\rm log_{10} \Lambda _{zR} \, [M_{\odot} Gyr^{-2}]$', [-4., -2., 0., 2., 4.], cmap=cmap)
    
    figure.fig.savefig('%s/lzr_fluxtensormap.%s' % (outpath, suffix))
    figure2.savefig('%s/lzr_flux_radius.%s' % (outpath, suffix))