from const import *
from loadmodules import *
from pylab import *
from util import *

colors = ['r', 'b', 'g', 'c', 'y']


def plot_vertical_phasewrap(runs, dirs, outpath, outputlistfile, redshift, agecut=[[0., 0.5, 1., 2., 6.], [0.5, 1., 2., 6., 12.]],
                            vrcut=[[-10., -5., 0., 5.], [-5., 0., 5., 10.]], vpcut=[[190., 200., 210., 220.], [200., 210., 220., 230.]],
                            accretedfiledir=None, birthdatafile=None, alphaelement=None, atbirth=False, disc_stars=False):
    agelo = agecut[0]
    agehi = agecut[1]
    vradlo = vrcut[0]
    vradhi = vrcut[1]
    vphilo = vpcut[0]
    vphihi = vpcut[1]
    
    ncols = 2
    nrows = len(agelo)
    
    panels = nrows * ncols
    
    vzran = [-100., 100.]
    zran = [-2., 2.]
    
    vrmin, vrmax = -60., 60.
    vpmin, vpmax = 180., 260.
    
    for d in range(len(runs)):
        
        figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, bottom=0.2, top=0.98)
        figure.set_figure_layout()
        figure.set_axis_limits_and_aspect(xlim=zran, ylim=vzran, logaxis=False)
        figure.set_axis_locators(xminloc=0.5, xmajloc=1., ymajloc=50., yminloc=50.)
        figure.set_axis_labels(xlabel="$\\rm{z \, [kpc]}$", ylabel=r"$\rm{V_z\,[km\,s^{-1}]}$")
        figure.set_fontsize(8)
        
        dd = dirs[d] + runs[d]
        wpath = outpath + runs[d]
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], redshift))
        
        print("doing %s, snap=%03d" % (wpath, snap))
        
        if atbirth:
            if birthdatafile:  # read birth data from post-processed file
                stardatafile = outpath + runs[d] + '/rm4/' + birthdatafile
                attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz']
            else:
                stardatafile = None  # snapshot already contains birth data
                attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz', 'bpos', 'bvel']
        else:
            stardatafile = None
            attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'gz']
        
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
        
        if 'pot' in attrs:
            eps2 = sdata['eps2'][asind]
        smass = sdata['mass'][asind]
        star_age = sdata['age'][asind]
        
        if atbirth:
            star_pos = sdata['bpos'][asind]
            star_vel = sdata['bvel'][asind]
        else:
            star_pos = sdata['pos'][asind]
            star_vel = sdata['vel'][asind]
        
        star_pos[:, 2] -= 0.008
        
        radius = np.sqrt((star_pos[:, 1:] ** 2).sum(axis=1))
        height = star_pos[:, 0]
        rad3d = np.sqrt(radius ** 2 + height ** 2)
        vrad = (star_pos[:, 1] * star_vel[:, 1] + star_pos[:, 2] * star_vel[:, 2]) / radius
        vphi = (star_pos[:, 1] * star_vel[:, 2] - star_pos[:, 2] * star_vel[:, 1]) / radius
        vz = star_vel[:, 0]
        
        vpsign = np.sum(vphi)
        if (vpsign < 0.):
            vphi *= (-1.)
        
        binx = 30
        biny = 30
        
        pnum = 0
        for i in range(len(agelo)):
            ax = figure.axes[2 * i]
            ax2 = figure.axes[2 * i + 1]
            
            # for j in range(len(vradlo)):
            
            if disc_stars:
                jj, = np.where((rad3d * 1e3 < 3.) & (star_age < agehi[i]) & (star_age > agelo[i]) & (
                            eps2 > 0.7))  # & (vrad <   # vradhi[j]) & (vrad > vradlo[j])  )  # ii, = np.where( (rad3d*1e3 < 3.) & (star_age <
                # agehi[i]) & (star_age > agelo[i]) & (eps2 > 0.7))# & (vphi < vphihi[j]) & (vphi > vphilo[j]) & (eps2 > 0.7) )
            else:
                jj, = np.where((rad3d * 1e3 < 3.) & (star_age < agehi[i]) & (star_age > agelo[
                    i]))  # & (vrad < vradhi[j]) & (vrad   # > vradlo[j]) )  # ii, = np.where( (rad3d*1e3 < 3.) & (star_age < agehi[i]) & (star_age
                # > agelo[i]) )#& (vphi <   # vphihi[j]) & (vphi > vphilo[j]) )
                
                # ax.plot( height[jj]*1e3, vz[jj], color=colors[j], marker='o', mec='none', markersize=0.8, lw=0.)#, vmin=vmin,   # vmax=vmax )
                
                # ax2.plot( height[ii]*1e3, vz[ii], color=colors[j], marker='o', mec='none', markersize=0.8, lw=0.)#, vmin=vmin,   # vmax=vmax )
            
            nn, xedges, yedges = np.histogram2d(height[jj] * 1e3, vz[jj], bins=(binx, biny), range=(zran, vzran), weights=vrad[jj])
            n, xedges, yedges = np.histogram2d(height[jj] * 1e3, vz[jj], bins=(binx, biny), range=(zran, vzran))
            nn /= n
            nn[n < 10] = nan
            xbin = 0.5 * (xedges[:-1] + xedges[1:])
            ybin = 0.5 * (yedges[:-1] + yedges[1:])
            
            xc, yc = np.meshgrid(xbin, ybin)
            
            ax.pcolormesh(xc, yc, nn, cmap=cm.get_cmap('viridis'), vmin=vrmin, vmax=vrmax)
            # ax.imshow( nn, cmap=cm.get_cmap('viridis'), vmin=vrmin, vmax=vrmax, interpolation='gaussian' )
            nn, xedges, yedges = np.histogram2d(height[jj] * 1e3, vz[jj], bins=(binx, biny), range=(zran, vzran), weights=vphi[jj])
            nn /= n
            nn[n < 10] = nan
            ax2.pcolormesh(xc, yc, nn, cmap=cm.get_cmap('viridis'), vmin=vrmin,
                           vmax=vrmax)  # ax2.imshow( nn, cmap=cm.get_cmap(  # 'viridis'), vmin=vrmin, vmax=vrmax, interpolation='gaussian' )
            
            # figure.set_colorbar([vmin,vmax],r'$\rm{ [10^{10}\,M_{\odot}\, dex^{-2}] }$',[0.1, 0.2, 0.3, 0.4], cmap=cm.get_cmap(  # 'viridis'),
            # fontsize=8, labelsize=8, orientation='vertical')
            
            # if pnum == 0:  #       title = "$\\rm{Au \,%s ,\, t=%2.1f}$" % (runs[d].split('_')[1], time)  #       figure.set_panel_title(
            # panel=pnum, title=title, position='bottom left')
            
            # ax.text( 0.45, 0.77, "$\\rm{%.1f < z < %.1f}$ \n$\\rm{%.0f < R < %.0f}$" % (zbinlo[i],zbinhi[i],rbinlo[j],rbinhi[j]), color='r',
            # transform=ax.transAxes, fontsize=8 )
        
        figure.reset_axis_limits()
        
        name = ''
        if atbirth:
            name += '_birth'
        if disc_stars:
            name += '_disc'
        if accretedfiledir:
            name += '_accreted'
        
        figname1 = '%s/vz_z_phasewrap_%s%s%03d.png' % (wpath, runs[d], name, snap)
        figure.fig.savefig(figname1)