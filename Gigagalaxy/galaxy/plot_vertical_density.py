from const import *
from loadmodules import *
from parse_particledata import parse_particledata
from pylab import *
from scipy.optimize import curve_fit
from util import *


def plot_vertical_density(runs, dirs, outpath, outputlistfile, redshift, suffix, nrows, ncols, logfit=False, disc_stars=True):
    readflag = np.zeros(len(runs))
    lstyle = [':', '--', '-']
    
    figure3 = multipanel_layout.multipanel_layout(nrows=1, ncols=1, npanels=1)
    figure3.set_figure_layout()
    figure3.set_axis_locators(xminloc=1.0, xmajloc=5.0, yminloc=1., ymajloc=1.)
    figure3.set_axis_labels(xlabel=r"$\rm{R [kpc]}$", ylabel=r"$\rm{h_z [kpc]}$")
    figure3.set_fontsize()
    figure3.set_axis_limits_and_aspect(xlim=[0.0, 16.], ylim=[0., 5.])
    
    ax3 = figure3.axes[0]
    
    p = plot_helper.plot_helper()
    
    panels = ncols * nrows
    figure1 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, left=0.14, bottom=0.2)
    figure1.set_figure_layout()
    figure1.set_axis_locators(xminloc=1.0, xmajloc=5.0, yminloc=1., ymajloc=1.)
    figure1.set_axis_labels(xlabel=r"$\rm{z [kpc]}$", ylabel=r"$\rm{\rho [M_{\odot} pc^{-3}]}$")
    figure1.set_fontsize()
    figure1.set_axis_limits_and_aspect(xlim=[0.0, 3.9], ylim=[1e-5, 0.05], logyaxis=True)
    
    figure2 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure2.set_figure_layout()
    figure2.set_axis_locators(xminloc=1.0, xmajloc=5.0, yminloc=0.5, ymajloc=1.)
    figure2.set_axis_labels(xlabel=r"$\rm{R [kpc]}$", ylabel=r"$\rm{h_{z} [kpc]}$")
    figure2.set_fontsize()
    figure2.set_axis_limits_and_aspect(xlim=[0.0, 16.0], ylim=[0., 4.5])
    
    clist = ['r', 'b', 'c', 'g', 'k']
    
    h_all = np.array([])
    h_young = np.array([])
    
    for d in range(len(runs)):
        snap = select_snapshot_number.select_snapshot_number(outputlistfile[d], redshift)
        
        dd = dirs[d] + runs[d]
        ax = figure1.fig.axes[d]
        ax2 = figure2.fig.axes[d]
        
        figure1.set_panel_title(panel=d, title="$\\rm %s\,%s$" % ("Au", runs[d].split('_')[1]), position='bottom left', color='k', fontsize=10)
        figure2.set_panel_title(panel=d, title="$\\rm %s\,%s$" % ("Au", runs[d].split('_')[1]), position='top left', color='k', fontsize=10)
        
        wpath = outpath + runs[d] + '/zprof/'
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        
        if readflag[d] == 0:
            filename = wpath + 'fit_hz_%03d.txt' % snap
            f3 = open(filename, 'w')
            header = "%12s%12s%12s%12s%12s%12s\n" % ("Time", "Radius", "a", "h1", "R0", "h2")
            f3.write(header)
            readflag[d] += 1
        
        nr = 10
        nx = 2 * nr
        nz = 10
        rc = np.linspace(0., 16., nr)
        hz = np.zeros(len(rc))
        hz2 = np.zeros(len(rc))
        
        print
        "Doing dir %s snap %d." % (dd, snap)
        if disc_stars:
            attrs = ['pos', 'vel', 'mass', 'age', 'pot']
        else:
            attrs = ['pos', 'vel', 'mass', 'age']
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'fmc2'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, [], 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        time = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True)
        g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
        g.prep_data()
        
        sdata = g.sgdata['sdata']
        mass = sdata['mass']
        pos = sdata['pos']
        star_age = sdata['age']
        
        galrad = 0.016
        zmax = 0.006
        
        if disc_stars:
            eps = sdata['eps2']
            ii, = np.where((eps > 0.7))
        else:
            ii = np.arange(len(mass))
        
        star_radius = pylab.sqrt((pos[ii, 1:] ** 2).sum(axis=1))
        star_height = pos[ii, 0]
        star_x = pos[ii, 1]
        star_y = pos[ii, 2]
        smass = mass[ii]
        star_age = star_age[ii]
        
        dz = zmax / float(nz)
        rbin = np.zeros(nr)
        zbin = np.zeros(nz)
        
        # Do different ages
        ii, = np.where((star_age < 3.))  # young
        jj, = np.where((star_age > 3.) & (star_age < 8.))  # inter
        kk, = np.where((star_age < 12.) & (star_radius > 0.0075) & (star_radius < 0.0085))  # all
        
        pp = ['young', 'all']
        pp = ['all']
        pdict = {'young': ii, 'all': kk}
        coltab = ['b', 'k']
        symtab = ['o', '^']
        
        for ind, val in enumerate(pp):
            
            mpf = pp[ind]
            pn, edges = np.histogram(abs(star_height[pdict[pp[ind]]]), bins=nz, weights=smass[pdict[mpf]], range=[0., zmax])
            pn /= 2.
            zbin = 0.5 * (edges[:-1] + edges[1:])
            vol = (edges[1:] - edges[:-1]) * zbin
            rho = 1e-8 * pn / vol
            
            guess = (.001, 0.5, .0001, 1.)
            # sigma = [1.]*nz
            sigma = 0.1 * np.log(rho * 1e9 + 0.1) - 1.
            # sigma = 1./sigma
            print
            "sigma=", sigma
            (popt, pcov) = curve_fit(p.total_profilede, zbin * 1e3, rho, guess, sigma=sigma)
            # (popt, pcov) = curve_fit( p.sech2_exp, zbin*1e3, rho, guess, sigma=sigma )
            print
            "double expon fit params=", popt
            
            print
            "zbin =", zbin
            print
            "rho=", rho
            
            pn, edges = np.histogram(abs(star_height[pdict[pp[ind]]]), bins=nz * 2, weights=smass[pdict[mpf]], range=[0., zmax])
            pn /= 2.
            zbin = 0.5 * (edges[:-1] + edges[1:])
            vol = (edges[1:] - edges[:-1]) * zbin
            rho = 1e-8 * pn / vol
            ax.semilogy(zbin * 1e3, rho, linestyle='-', color='k', lw=1.)
            
            if popt[1] < popt[3]:
                ax.semilogy(zbin * 1e3, p.exp_prof(zbin * 1e3, popt[0], popt[1]), linestyle=':', color='b', lw=1.,
                            label="$\\rm{%.1f pc}$" % (popt[1] * 1e3))
                ax.semilogy(zbin * 1e3, p.exp_prof(zbin * 1e3, popt[2], popt[3]), dashes=(2, 2), color='r', lw=1.,
                            label="$\\rm{%.1f pc}$" % (popt[3] * 1e3))
            else:
                ls1 = '--'
                c1 = 'r'
                ls2 = ':'
                c2 = 'b'
                ax.semilogy(zbin * 1e3, p.exp_prof(zbin * 1e3, popt[2], popt[3]), linestyle=':', color='b', lw=1.,
                            label="$\\rm{%.1f pc}$" % (popt[3] * 1e3))
                ax.semilogy(zbin * 1e3, p.exp_prof(zbin * 1e3, popt[0], popt[1]), dashes=(2, 2), color='r', lw=1.,
                            label="$\\rm{%.1f pc}$" % (popt[1] * 1e3))
            
            # ax.semilogy(zbin*1e3,p.exp_prof(zbin*1e3,popt2[0],popt2[1]),linestyle='--',color=s_m.to_rgba(j),lw=0.3)
            ax.xaxis.set_ticks([0, 1, 2, 3, 4])
            ax.yaxis.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.])
            figure1.reset_axis_limits()
            
            ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=1)
            
            """
            mean_height, xedges, yedges = np.histogram2d( star_x[pdict[mpf]], star_y[pdict[mpf]], bins=(nx,nx), weights=star_height[pdict[
            mpf]]*smass[pdict[mpf]], range=[[-galrad, galrad],[-galrad, galrad]] )
            num, xedges, yedges = np.histogram2d( star_x[pdict[mpf]], star_y[pdict[mpf]], bins=(nx,nx), weights=smass[pdict[mpf]], range=[[-galrad,
            galrad],[-galrad, galrad]] )

            mean_height /= num
            mean_height[np.isnan(mean_height)] = 0.
            print "mean_height=",mean_height

            xcen, ycen = np.zeros(nx), np.zeros(nx)
            xcen[:] = 0.5 * (xedges[1:] + xedges[:-1])
            ycen[:] = 0.5 * (yedges[1:] + yedges[:-1])
            xind = np.digitize( star_x, xcen ) - 1
            yind = np.digitize( star_y, ycen ) - 1

            star_height -= mean_height[xind,yind]

            pn, xedges, yedges = np.histogram2d( star_radius[pdict[pp[ind]]], abs(star_height[pdict[pp[ind]]]), bins=(nr,nz), weights=smass[pdict[
            pp[ind]]], range=[[0., galrad],[0., zmax]] )
            rc = np.zeros(nr)
            rc[:] = 0.5 * ( xedges[1:] + xedges[:-1] ) * 1e3

            pn /= 2.

            s = (nr,nz)
            dv = np.zeros( nr )
            rho = np.zeros( s )
            rbin[:] = 0.5 * ( xedges[1:] + xedges[:-1] )
            zbin[:] = 0.5 * ( yedges[1:] + yedges[:-1] )

            # Stellar density
            dv[:] = np.pi *( ( xedges[1:]**2 - xedges[:-1]**2 )*1e6 ) * ( dz * 1e3 )
            dv = [dv] * nz
            dv = np.array(dv)
            dv = dv.T
            rho[:,:] = pn[:,:] * 1e10 / dv[:,:]
            rho[:,:] *= 1e-9   # /pc^3
            

            parameters = np.linspace(0,len(rc),len(rc))
            cmap = mpl.cm.jet
            s_m = figure1.get_scalar_mappable_for_colorbar(parameters, cmap)
        
        
            for j in range(len(rc)):
                flag = 0
                i = int( rc[j] / (galrad*1e3) * float(nr) )

                if logfit:
                        
                    if val == 'young':
                        bounds = [[-10.,-10., 0.,-10.],[30.,2.,1.,2.]]
                        sigma = [1.]*nz
                    elif val == 'all':
                        bounds = [[-5.,-2., 0.,-2.],[30.,2.,2.,2.]]
                        sigma = 0.1*np.log(rho[i,:]*1e9 + 0.1)  

                    y = np.log( rho[j,:]*1e9 + 0.1 )
                    zbin = np.array(zbin)
                    y = np.array(y)
                    sigma = np.array(sigma)

                    try:
                        (popt, pcov) = curve_fit( p.double_exp, zbin[np.where(y>0.)]*1e3, y[np.where(y>0.)], bounds=bounds, sigma=sigma[np.where(
                        y>0.)] )
                        try:
                            perr = np.sqrt((np.diag(pcov))).mean()
                        except:
                            perr = 1e9

                    except RuntimeError:
                        print "Error- curve fit failed (1)"
                        perr = 1e9
                        flag += 1
                        popt[:] = inf

                    hz[j] = -1. / (min(popt[1], popt[3]))
                    hz2[j] = -1. / (max(popt[1], popt[3]))

                    if val == 'all' and len(popt) == 4:
                        with open(filename, "a") as f3:
                            param = "%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f\n" % ( time, rc[j], np.exp(popt[0])*1e-9, (-1. / popt[1]), popt[2]*1e3,
                            (-1. / popt[3]) )
                            f3.write(param)

                else: # if not logfit
                    y = rho[j,:]
                    guess = (.001, 0.5, .001, 2.)
                    if val == 'young':
                        sigma = [1.]*nz
                    elif val == 'all':
                        sigma = 0.1*rho[i,:]

                    try:
                        (popt, pcov) = curve_fit( p.total_profilede, zbin[:]*1e3, y, guess, sigma=sigma )
                        print "double expon fit params=",popt
                        try:
                            perr = np.sqrt((np.diag(pcov))).mean()
                        except:
                            perr = 1e9
                    
                    except RuntimeError:
                        print "Error- curve fit failed (1)"
                    #    perr = 1e9
                    #    flag += 1
                    try:
                        guess = (.1, 2.)
                        (popt2, pcov2) = curve_fit( p.exp_prof, zbin[:]*1e3, y, guess, sigma=sigma )
                        try:
                            perr2 = np.sqrt((np.diag(pcov2))).mean()
                        except:
                            perr2 = 1e9
                                
                    except RuntimeError:
                        print "Error- curve fit failed (2)"
                    #    flag += 2

                    #if flag == 0:
                    #    if perr < perr2:
                    #        print "double expon profile given"
                    #        hz[j] = min(popt[1], popt[3])
                    #    elif perr >= perr2:
                    #        print "Single expon given"
                    #        hz[j] = popt2[1]
                    #    print "popt=",popt
						
                    #elif flag == 2:
                    #    print "double expon profile given"
                    #    hz[j] = popt[1]
                    #elif flag == 1:
                    #    print "Single expon given"
                    #    hz[j] = popt2[1]
                    #elif flag == 3:
                    #    raise ValueError('flag = 3. No fit was successful.')

                    
                    
                    with open(filename, "a") as f3:
                        param = "%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f\n" % (time, rc[j], popt2[0], popt2[1], popt[0], popt[1], popt[3])
                        f3.write(param)		

                ax.semilogy(zbin*1e3,rho[i,:],linestyle='-',color=s_m.to_rgba(j),lw=0.2)

                if logfit:
                    ax.semilogy(zbin*1e3, 1e-9*np.exp(p.double_exp(zbin*1e3, *popt)),linestyle=':',color=s_m.to_rgba(j),lw=0.5)
                else:
                    #if perr < perr2:
                    print "p.exp_prof(zbin*1e3,popt[0],popt[1])=",p.exp_prof(zbin*1e3,popt[0],popt[1])
                    print "p.exp_prof(zbin*1e3,popt[2],popt[3])=",p.exp_prof(zbin*1e3,popt[2],popt[3])
                    ax.semilogy(zbin*1e3,p.exp_prof(zbin*1e3,popt[0],popt[1]),linestyle='--',color=s_m.to_rgba(j),lw=0.3)
                    ax.semilogy(zbin*1e3,p.exp_prof(zbin*1e3,popt[2],popt[3]),linestyle=':',color=s_m.to_rgba(j),lw=0.3)
                    #else:
                    ax.semilogy(zbin*1e3,p.exp_prof(zbin*1e3,popt2[0],popt2[1]),linestyle='--',color=s_m.to_rgba(j),lw=0.3)

                ax.xaxis.set_ticks([0,1,2,3,4])
                ax.yaxis.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.])
                figure1.reset_axis_limits()

            dum = np.zeros(len(rc))
            for i in range(len(rc)):
                dum[i] = p.gaussian_kernel( hz, rc-rc[i], 2. )
            hz = dum
            dum = np.zeros(len(rc))
            for i in range(len(rc)):
                dum[i] = p.gaussian_kernel( hz2, rc-rc[i], 2. )
            hz2 = dum

            if val == 'young':
                h_young = np.append(h_young, hz)
            elif val == 'all':
                h_all = np.append(h_all, hz)

            ax2.plot(rc, hz, linestyle='-', color=coltab[ind], lw=0.7) 

            guess = ( 0.5, -2. ) 
            (popt3, pcov3) = curve_fit( p.exp_prof, rc, hz, guess )
            ax2.plot(rc, p.exp_prof(rc, popt3[0], popt3[1]), linestyle='--', color=coltab[ind], lw=0.7)
            ax2.text( 0.1, 0.7-0.1*float(ind), "$\\rm{-R_{flare}^{-1}= %1.2f\, kpc^{-1}}$" % (-1./popt3[1]), color=coltab[ind],
            transform=ax2.transAxes, fontsize=7 )

        f3.close()

    h_arr = [h_all, h_young]
    colorss = ['k', 'royalblue']

    for j in range(len(h_arr)):
        if j == 0:
            hname = r"${\rm z = %1.1f}$" % redshift[0]
        else:
            hname = ''
        h_median = np.zeros(nr)
        h_sigma = np.zeros(nr)
        for i in range(nr):
            h_median[i] = np.nanmedian( h_arr[j][i::nr] )
            h_sigma[i] = np.nanstd( h_arr[j][i::nr] )
            
            #ax3.fill_between( rc, h_median-h_sigma, h_median+h_sigma, facecolor=colorss[j], alpha=0.2, edgecolor='none' )
        ax3.plot( rc, h_median, linestyle='-', lw=0.7, color=colorss[j], label=hname )
        	
    figure1.fig.savefig( '%s/rho_z_prof%03d_meanzsub.%s' % (outpath, snap, suffix) )
    figure2.fig.savefig( '%s/fitted_hzprof%03d_meanzsub.pdf' % (outpath, snap) )

    ax3.legend( loc='upper left', fontsize=10, frameon=False, numpoints=1, ncol=1 )
    ax3.text( 0.75, 0.87, "$\\rm{all}$", transform=ax3.transAxes, fontsize=12 )        
    ax3.text( 0.75, 0.8, "$\\rm{young}$", color='royalblue', transform=ax3.transAxes, fontsize=12 )

    figure3.fig.savefig( '%s/fitted_hzprof_simmean_meanzsub.pdf' % (outpath) )

    """
    figure1.fig.savefig('%s/rho_z_prof%03d_meanzsub.%s' % (outpath, snap, suffix))