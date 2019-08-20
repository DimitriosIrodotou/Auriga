from const import *
from loadmodules import *
from pylab import *
from util import *

toinch = 0.393700787


def plot_birth_dist(runs, dirs, outpath, snap, alpha=0, disc_stars=False, arrows=False):
    for d in range(len(runs)):
        
        pylab.figure(figsize=(10 * toinch, 5 * toinch), dpi=300)
        
        dd = dirs[d] + runs[d]
        wpath = outpath + runs[d] + '/projections/stars/'
        
        indy = [[1, 2, 0], [1, 0, 2]]
        ascale = [6000., 6000.]  # 2000.]
        
        print("doing %s, snap=%03d" % (wpath, snap))
        
        # Get birth data
        attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz', 'bpos', 'bvel']
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
        s.center = sf.data['fpos'][0, :]
        
        rotmatfile = dd + '/output/rotmatlist_%s.txt' % runs[d]
        galcenfile = dd + '/output/galcen_%s.txt' % runs[d]
        g = parse_particledata(s, sf, attrs, rotmatfile, galcenfile)
        g.prep_data()
        sdata = g.sgdata['sdata']
        
        for j in range(2):
            
            if j == 1:
                star_bpos = sdata['bpos']
                star_bvel = sdata['bvel']
                print("USING BIRTH DATA")
            else:
                star_bpos = sdata['pos']
                star_bvel = sdata['vel']
            
            star_age = sdata['age']
            alphafe = sdata['alpha']
            iron = sdata['Fe']
            eps2 = sdata['eps2']
            # cnrat = g.data['cnrat']
            
            jj, = np.where((star_age < 12.))
            
            star_age = star_age[jj]
            alphafe = alphafe[jj]
            iron = iron[jj]
            eps2 = eps2[jj]
            # cnrat = cnrat[jj]
            
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
            
            nx = 60
            ny = 60
            
            agebinlo = [10., 0., 0., 2., 4., 6., 8.]
            agebinhi = [12., 12., 2., 4., 6., 8., 10.]
            
            npan = len(agebinlo)
            nrows = 2.
            ncols = 7.
            
            # aval2 = 0.25
            # aval = 0.2
            
            aval2 = 0.1
            aval = 0.03
            
            # cnrat
            aval2 = 0.2
            aval = 0.15
            
            # aint = 0.03
            # agrad = -0.02
            # aval = aint + agrad*iron
            
            ecut = 0.5
            
            for i in range(npan):
                
                ix = (i - int(nrows)) % int(ncols)
                iy = j
                
                x = ix * (1 / ncols) + 1 / ncols * 0.05 * 0.5
                y = iy * (1 / nrows) + 1 / ncols * 0.05 * 0.5 + 1. / (1. * ncols)
                
                ax = axes([x, y, 1 / ncols * 0.95, 2 / ncols * 0.95], frameon=False)
                
                if alpha == 1:
                    jj, = np.where((star_age < agebinhi[i]) & (star_age > agebinlo[i]) & (cnrat < aval))
                elif alpha == 2:
                    jj, = np.where((star_age < agebinhi[i]) & (star_age > agebinlo[i]) & (cnrat > aval2))
                else:
                    jj, = np.where((star_age < agebinhi[i]) & (star_age > agebinlo[i]))
                
                if disc_stars:
                    jj = jj[eps2[jj] > ecut]
                print("%d stars in age range: %1.1f - %1.1f" % (len(jj), agebinlo[i], agebinhi[i]))
                n, xedges, yedges = np.histogram2d(pos[jj, 1], pos[jj, 2], bins=(nx, ny), range=[[-.025, .025], [-.025, .025]])
                
                xbin = np.zeros(len(xedges) - 1)
                ybin = np.zeros(len(yedges) - 1)
                xbin[:] = 0.5 * (xedges[:-1] + xedges[1:])
                ybin[:] = 0.5 * (yedges[:-1] + yedges[1:])
                
                xc, yc = np.meshgrid(xbin, ybin)
                
                levels = [0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.5, 2.0, 2.5, 3., 3.5, 4., 4.5, 5.]
                
                pylab.contourf(xc, yc, log10(n.T), cmap=cm.get_cmap('magma'), levels=levels)
                
                if arrows:
                    nbin = nx
                    d1, d2 = 1, 2  # 2,1
                    pn, xedges, yedges = np.histogram2d(pos[:, d1], pos[:, d2], bins=(nbin, nbin), range=[[-0.025, 0.025], [-0.025, 0.025]])
                    vxgrid, xedges, yedges = np.histogram2d(pos[:, d1], pos[:, d2], bins=(nbin, nbin), weights=vel[:, d1],
                                                            range=[[-0.025, 0.025], [-0.025, 0.025]])
                    vygrid, xedges, yedges = np.histogram2d(pos[:, d1], pos[:, d2], bins=(nbin, nbin), weights=vel[:, d2],
                                                            range=[[-0.025, 0.025], [-0.025, 0.025]])
                    vxgrid /= pn
                    vygrid /= pn
                    
                    xbin = np.zeros(len(xedges) - 1)
                    ybin = np.zeros(len(yedges) - 1)
                    xbin[:] = 0.5 * (xedges[:-1] + xedges[1:])
                    ybin[:] = 0.5 * (yedges[:-1] + yedges[1:])
                    xc, yc = np.meshgrid(xbin, ybin)
                    
                    vygrid *= -1
                    
                    p = plt.quiver(xc, yc, np.flipud(vxgrid.T), np.flipud(vygrid.T), scale=ascale[0], pivot='middle', color='w', alpha=0.8,
                                   width=0.001)
                
                label = '$\\rm{%1.f < age < %1.f}$' % (agebinlo[i], agebinhi[i])
                if j == 1:
                    text(0.1, 1.01, label, color='g', transform=ax.transAxes, fontsize=6)
                
                # plot2
                x = ix * (1 / ncols) + 1 / ncols * 0.05 * 0.5
                y = iy * (1 / nrows) + 1 / ncols * 0.05 * 0.5
                
                ax2 = axes([x, y, 1 / ncols * 0.95, 2 / (2. * ncols) * 0.95], frameon=False)
                
                n, xedges, yedges = np.histogram2d(pos[jj, 1], pos[jj, 0], bins=(nx, ny / 2), range=[[-0.025, 0.025], [-0.0125, 0.0125]])
                
                xbin = np.zeros(len(xedges) - 1)
                ybin = np.zeros(len(yedges) - 1)
                xbin[:] = 0.5 * (xedges[:-1] + xedges[1:])
                ybin[:] = 0.5 * (yedges[:-1] + yedges[1:])
                
                xc, yc = np.meshgrid(xbin, ybin)
                
                pylab.contourf(xc, yc, log10(n.T), cmap=cm.get_cmap('magma'), levels=levels)
                
                if arrows:
                    nbin = nx
                    d1, d2 = 1, 0  # 2,1
                    pn, xedges, yedges = np.histogram2d(pos[:, d1], pos[:, d2], bins=(nbin, nbin), range=[[-0.025, 0.025], [-0.0125, 0.0125]])
                    vxgrid, xedges, yedges = np.histogram2d(pos[:, d1], pos[:, d2], bins=(nbin, nbin), weights=vel[:, d1],
                                                            range=[[-0.025, 0.025], [-0.0125, 0.0125]])
                    vygrid, xedges, yedges = np.histogram2d(pos[:, d1], pos[:, d2], bins=(nbin, nbin), weights=vel[:, d2],
                                                            range=[[-0.025, 0.025], [-0.0125, 0.0125]])
                    vxgrid /= pn
                    vygrid /= pn
                    
                    xbin = np.zeros(len(xedges) - 1)
                    ybin = np.zeros(len(yedges) - 1)
                    xbin[:] = 0.5 * (xedges[:-1] + xedges[1:])
                    ybin[:] = 0.5 * (yedges[:-1] + yedges[1:])
                    xc, yc = np.meshgrid(xbin, ybin)
                    
                    vygrid *= -1
                    
                    p = plt.quiver(xc, yc, np.flipud(vxgrid.T), np.flipud(vygrid.T), scale=ascale[1], pivot='middle', color='white', alpha=0.8,
                                   width=0.001)
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax2.set_xticks([])
                ax2.set_yticks([])
        
        if disc_stars:
            st = 'disc'
        else:
            st = ''
        
        if alpha == 1:
            savefig('%s/birthxyz_%s_%03d_alphapoor_test%s.pdf' % (wpath, runs[d], snap, st), dpi=300)
        elif alpha == 2:
            savefig('%s/birthxyz_%s_%03d_alpharich_test%s.pdf' % (wpath, runs[d], snap, st), dpi=300)
        else:
            savefig('%s/birthxyz_%s_%03d_test%s.pdf' % (wpath, runs[d], snap, st), dpi=300)
        
        plt.close()


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx