from const import *
from loadmodules import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
from util import *

toinch = 0.393700787


def plot_kinematic_map(runs, dirs, outpath, snap, alpha=0, VelocityComponent='z', disc_stars=False, arrows=False):
    for d in range(len(runs)):
        
        if VelocityComponent == 'z':
            nrow = 2
            fig = pylab.figure(figsize=(11 * toinch, 5 * toinch), dpi=300)
        else:
            nrow = 1
            fig = pylab.figure(figsize=(11 * toinch, 2.5 * toinch), dpi=300)
        
        dd = dirs[d] + runs[d]
        wpath = outpath + runs[d] + '/projections/stars/'
        
        indy = [[1, 2, 0], [1, 0, 2]]
        ascale = [6000., 6000.]  # 2000.]
        
        print("doing %s, snap=%03d" % (wpath, snap))
        
        # Get birth data
        attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'gz']  # , 'bpos', 'bvel']
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'flty', 'fnsh', 'slty', 'svel'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        rotmatfile = dd + '/output/rotmatlist_%s.txt' % runs[d]
        galcenfile = dd + '/output/galcen_%s.txt' % runs[d]
        g = parse_particledata(s, sf, attrs)  # , rotmatfile, galcenfile )
        g.prep_data()
        sdata = g.sgdata['sdata']
        
        for j in range(nrow):
            star_bpos = sdata['pos']
            star_bvel = sdata['vel']
            
            star_age = sdata['age']
            alphafe = sdata['alpha']
            iron = sdata['Fe']
            
            jj, = np.where((star_age < 12.))
            
            star_age = star_age[jj]
            alphafe = alphafe[jj]
            iron = iron[jj]
            
            temp_pos = star_bpos[jj, :]
            pos = np.zeros((len(temp_pos), 3))
            pos[:, 0] = temp_pos[:, 0]  # 0:1
            pos[:, 1] = temp_pos[:, 1]  # 1:2
            pos[:, 2] = temp_pos[:, 2]  # 2:0
            
            temp_vel = star_bvel[jj, :]
            vel = np.zeros((len(temp_vel), 3))
            vel[:, 0] = temp_vel[:, 0]
            vel[:, 1] = temp_vel[:, 1]
            vel[:, 2] = temp_vel[:, 2]
            
            if VelocityComponent == 'z':
                if j == 0:
                    vw = pos[:, 0] * 1e3
                    levels = np.linspace(-3., 3., 31)
                elif j == 1:
                    vw = vel[:, 0]
                    levels = np.linspace(-30., 30., 21)
            
            elif VelocityComponent == 'R':
                Radius = np.sqrt((vel[:, 1:] ** 2).sum(axis=1))
                vw = (pos[:, 1] * vel[:, 1] + pos[:, 2] * vel[:, 2]) / Radius
            elif VelocityComponent == 'phi':
                vw = (pos[:, 1] * vel[:, 2] - pos[:, 2] * vel[:, 1]) / Radius
            else:
                raise ValueError('Invalid VelocityComponent %s' % VelocityComponent)
            
            nx = 60
            ny = 60
            
            agebinlo = [5., 0., 0., 1., 2., 3., 4.]
            agebinhi = [6., 6., 1., 2., 3., 4., 5.]
            
            npan = len(agebinlo)
            nrows = 2.
            ncols = 7.
            
            ecut = 0.5
            
            for i in range(npan):
                
                ix = (i - int(nrows)) % int(ncols)
                iy = j
                
                x = ix * (1 / ncols) + 1 / ncols * 0.05 * 0.5
                y = iy * (1 / nrows) + 1 / ncols * 0.05 * 0.5 + 1. / (1. * ncols)
                
                ax = axes([x, y, 1 / ncols * 0.95, 2 / ncols * 0.95], frameon=False)
                
                jj, = np.where((star_age < agebinhi[i]) & (star_age > agebinlo[i]) & (abs(pos[:, 0]) < 0.005))
                
                vc, xedges, yedges = np.histogram2d(pos[jj, 1], pos[jj, 2], weights=vw[jj], bins=(nx, ny), range=[[-.025, .025], [-.025, .025]])
                n, xedges, yedges = np.histogram2d(pos[jj, 1], pos[jj, 2], bins=(nx, ny), range=[[-.025, .025], [-.025, .025]])
                
                vc /= n
                
                xbin = np.zeros(len(xedges) - 1)
                ybin = np.zeros(len(yedges) - 1)
                xbin[:] = 0.5 * (xedges[:-1] + xedges[1:])
                ybin[:] = 0.5 * (yedges[:-1] + yedges[1:])
                
                xc, yc = np.meshgrid(xbin, ybin)
                
                # levels = [0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.5, 2.0, 2.5, 3., 3.5, 4., 4.5, 5.]
                
                plt1 = ax.contourf(xc, yc, vc.T, cmap=cm.get_cmap('magma'), levels=levels)
                
                label = '$\\rm{%1.f < age < %1.f}$' % (agebinlo[i], agebinhi[i])
                if j == 1:
                    text(0.1, 1.01, label, color='g', transform=ax.transAxes, fontsize=6)
                
                if i == 1:
                    # maxtick = 5
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = plt.colorbar(plt1, cax=cax, orientation='vertical')  # , ticks=np.arange(2, maxtick))
                    # fig.subplots_adjust(right=0.8)
                    # cax = fig.add_axes([0.05, 0.45, 0.9, 0.05])
                    # cbar = fig.colorbar(plt1, cax=cax, orientation='horizontal' )
                    if j == 1:
                        cbar.set_label('$\\rm{<v> \\, km\,s^{-1}]}$')
                    else:
                        cbar.set_label('$\\rm{<z> \\, kpc]}$')
                
                # plot2
                x = ix * (1 / ncols) + 1 / ncols * 0.05 * 0.5
                y = iy * (1 / nrows) + 1 / ncols * 0.05 * 0.5
                
                ax2 = axes([x, y, 1 / ncols * 0.95, 2 / (2. * ncols) * 0.95], frameon=False)
                
                vc, xedges, yedges = np.histogram2d(pos[jj, 1], pos[jj, 0], weights=vw[jj], bins=(nx, ny / 2), range=[[-.025, .025], [-.025, .025]])
                n, xedges, yedges = np.histogram2d(pos[jj, 1], pos[jj, 0], bins=(nx, ny / 2), range=[[-0.025, 0.025], [-0.0125, 0.0125]])
                vc /= n
                
                xbin = np.zeros(len(xedges) - 1)
                ybin = np.zeros(len(yedges) - 1)
                xbin[:] = 0.5 * (xedges[:-1] + xedges[1:])
                ybin[:] = 0.5 * (yedges[:-1] + yedges[1:])
                
                xc, yc = np.meshgrid(xbin, ybin)
                
                ax2.contourf(xc, yc, vc.T, cmap=cm.get_cmap('magma'), levels=levels)
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax2.set_xticks([])
                ax2.set_yticks([])
        
        if disc_stars:
            st = 'disc'
        else:
            st = ''
        
        savefig('%s/%svelocitymap_%s_%03d%s.pdf' % (wpath, VelocityComponent, runs[d], snap, st), dpi=300)
        
        plt.close()


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx