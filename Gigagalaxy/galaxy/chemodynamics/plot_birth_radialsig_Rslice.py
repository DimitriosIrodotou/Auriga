from const import *
from loadmodules import *
from pylab import *
from util import *

colors = ['r', 'm', 'c', 'b']


def plot_birth_radialsig_Rslice(runs, dirs, outpath, outputlistfile, redshift, zcut=0.5, rcut=[[3., 7., 11., 15.], [5., 9., 13., 17.]],
                                accretedfiledir=None, birthdatafile=None, alphaelement=None, disc_stars=False, actions=False):
    rbinlo = rcut[0]
    rbinhi = rcut[1]
    
    nrows = 1
    ncols = len(rbinlo)
    panels = nrows * ncols
    
    xran = [0., 13.]
    yran = [-0.1, 2.5]
    
    for d in range(len(runs)):
        
        figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, bottom=0.1, top=0.98, twinyaxis=True)
        figure.set_figure_layout()
        figure.set_axis_limits_and_aspect(xlim=xran, ylim=yran, logaxis=False)
        figure.set_axis_locators(xminloc=1., ymajloc=0.2, yminloc=0.2)
        figure.set_axis_labels(xlabel="$\\rm{age\,[Gyr]}$", ylabel=r"$\rm{log_{10} <v_r^2>^{1/2}\, [km\,s^{-1}]}$",
                               y2label="$\\rm{log_{10}(\sqrt{J_{R,rms}})}$")
        figure.set_fontsize(8)
        
        figure2 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, bottom=0.1, top=0.98, twinyaxis=True)
        figure2.set_figure_layout()
        figure2.set_axis_limits_and_aspect(xlim=xran, ylim=yran, logaxis=False)
        figure2.set_axis_locators(xminloc=1., ymajloc=0.2, yminloc=0.2)
        figure2.set_axis_labels(xlabel="$\\rm{age\,[Gyr]}$", ylabel=r"$\rm{log_{10} <v_z^2>^{1/2}\, [km\,s^{-1}]}$",
                                y2label="$\\rm{log_{10}(\sqrt{J_{z,rms}})}$")
        figure2.set_fontsize(8)
        
        cblimp = [1., 11.]
        parameters = np.linspace(cblimp[0], cblimp[1], 20)
        cmapl = plt.get_cmap('viridis')  # mpl.cm.jet
        s_m = figure.get_scalar_mappable_for_colorbar(parameters, cmapl)
        
        dd = dirs[d] + runs[d]
        wpath = outpath + runs[d]
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], redshift))
        
        print("doing %s, snap=%03d" % (wpath, snap))
        
        if actions:
            apath = outpath + runs[d] + '/actions/'
            fstr = '_new'
            fa = apath + '/actions_%s%s.dat' % (runs[d], fstr)
            
            fin = open(fa, "rb")
            nst = struct.unpack('i', fin.read(4))[0]
            idact = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64).astype('int64')
            jR = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
            jz = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
            lz = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
            jRb = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
            jzb = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
            lzb = np.array(struct.unpack('%sd' % nst, fin.read(nst * 8)), dtype=float64)
            fin.close()
        
        if birthdatafile:  # read birth data from post-processed file
            stardatafile = outpath + runs[d] + birthdatafile
            rotmatfile = None
            galcenfile = None
            attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz']
        else:
            stardatafile = None  # snapshot already contains birth data
            rotmatfile = dd + '/output/rotmatlist_%s.txt' % (runs[d])
            galcenfile = dd + '/output/galcen_%s.txt' % (runs[d])
            attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz', 'bpos', 'bvel']
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        if accretedfiledir:
            accretedfile = accretedfiledir + runs[d] + '/%sstarID_accreted_all.dat' % runs[d]
        else:
            accretedfile = None
        if birthdatafile:
            attrs.append('bpos')
            attrs.append('bvel')
        g = parse_particledata(s, sf, attrs, rotmatfile=rotmatfile, galcenfile=galcenfile, accretedfile=accretedfile, stardatafile=stardatafile,
                               radialcut=0.1 * sf.data['frc2'][0])
        g.prep_data()
        
        sdata = g.sgdata['sdata']
        
        bradius = np.sqrt((sdata['bpos'][:, 1:] ** 2).sum(axis=1))
        radius = np.sqrt((sdata['pos'][:, 1:] ** 2).sum(axis=1))
        bheight = sdata['bpos'][:, 0]
        height = sdata['pos'][:, 0]
        eps2 = sdata['eps2']
        star_age = sdata['age']
        vrad = (sdata['pos'][:, 1] * sdata['vel'][:, 1] + sdata['pos'][:, 2] * sdata['vel'][:, 2]) / radius
        bvrad = (sdata['bpos'][:, 1] * sdata['bvel'][:, 1] + sdata['bpos'][:, 2] * sdata['bvel'][:, 2]) / bradius
        
        # sort action array with IDs
        if actions:
            aci = np.in1d(idact, sdata['id'].astype('int64'))
            jR = jR[aci]
            jz = jz[aci]
            lz = lz[aci]
            jRb = jRb[aci]
            jzb = jzb[aci]
            lzb = lzb[aci]
            idact = idact[aci]
        
        binx = 80
        biny = 80
        # conv = (binx / (feran[1]-feran[0])) * (biny / (alran[1] - alran[0]))
        vmin = 0.005
        vmax = 0.5
        if accretedfile:
            vmin /= 10.
            vmax /= 10.
        
        agelo = np.array([0., 2., 4., 6., 8., 10.])  # , 0.])
        agehi = np.array([2., 4., 6., 8., 10., 12.])  # , 12.])
        
        print("agelo,agehi=", agelo, agehi)
        pnum = 0
        for j in range(len(rbinlo)):
            ax = figure.axes[pnum]
            axt = figure.twinyaxes[pnum]
            ax2 = figure2.axes[pnum]
            axt2 = figure2.twinyaxes[pnum]
            
            # axt.set_ylim( ax.get_ylim() )
            # axt.set_yticks( np.linspace(yran[0], yran[1], 7) )
            # axt.set_yticklabels( ['0', '10', '20', '30', '40', '50', '60'] )
            # axt2.set_ylim( ax2.get_ylim() )
            # axt2.set_yticks( np.linspace(yran[0], yran[1], 7) )
            # axt2.set_yticklabels( ['0', '10', '20', '30', '40', '50', '60'] )
            
            pnum += 1
            
            for i in range(len(agelo)):
                
                age = 0.5 * (agelo[i] + agehi[i])
                
                if disc_stars:
                    jj, = np.where((radius * 1e3 > rbinlo[j]) & (radius * 1e3 < rbinhi[j]) & (abs(height * 1e3) < zcut) & (eps2 > 0.7) & (
                                star_age < agehi[i]) & (star_age > agelo[i]))
                else:
                    jj, = np.where((radius * 1e3 > rbinlo[j]) & (radius * 1e3 < rbinhi[j]) & (abs(height * 1e3) < zcut) & (star_age < agehi[i]) & (
                                star_age > agelo[i]))
                
                # radial Vdisp
                sigr = np.sqrt(np.mean(vrad[jj] ** 2))
                bsigr = np.sqrt(np.mean(bvrad[jj] ** 2))
                
                sigz = np.sqrt(np.mean(sdata['vel'][jj, 0] ** 2))
                bsigz = np.sqrt(np.mean(sdata['bvel'][jj, 0] ** 2))
                
                # RMS of root J
                if actions:
                    jRsig = np.sqrt(np.mean(jR[jj]))
                    bjRsig = np.sqrt(np.mean(jRb[jj]))
                    
                    jzsig = np.sqrt(np.mean(jz[jj]))
                    bjzsig = np.sqrt(np.mean(jzb[jj]))
                
                ax.plot(age, np.log10(sigr), 'o', color=s_m.to_rgba(age))
                ax.plot(age, np.log10(bsigr), '^', color=s_m.to_rgba(age))
                ax2.plot(age, np.log10(sigz), 'o', color=s_m.to_rgba(age))
                ax2.plot(age, np.log10(bsigz), '^', color=s_m.to_rgba(age))
                
                if actions:
                    axt.plot(age, np.log10(jRsig), 'o', color=s_m.to_rgba(age), markeredgecolor='r')
                    axt.plot(age, np.log10(bjRsig), '^', color=s_m.to_rgba(age), markeredgecolor='r')
                    axt2.plot(age, np.log10(jzsig), 'o', color=s_m.to_rgba(age), markeredgecolor='r')
                    axt2.plot(age, np.log10(bjzsig), '^', color=s_m.to_rgba(age), markeredgecolor='r')
            
            if pnum == 0:
                title = "$\\rm{Au \,%s ,\, t=%2.1f}$" % (runs[d].split('_')[1], time)
                figure.set_panel_title(panel=pnum, title=title, position='bottom left')
                figure2.set_panel_title(panel=pnum, title=title, position='bottom left')
            
            ax.text(0.5, 0.85, "$\\rm{%.0f < R < %.0f}$" % (rbinlo[j], rbinhi[j]), color='gray', transform=ax.transAxes, fontsize=6)
            ax2.text(0.5, 0.85, "$\\rm{%.0f < R < %.0f}$" % (rbinlo[j], rbinhi[j]), color='gray', transform=ax2.transAxes, fontsize=6)
        
        figure.reset_axis_limits()
        figure2.reset_axis_limits()
        
        figure.set_colorbar([1., 11.], r'$\rm{ age \, [Gyr] }$', [1., 3., 5., 7., 9., 11.], cmap=cmapl, fontsize=7, labelsize=7,
                            orientation='horizontal')
        figure2.set_colorbar([1., 11.], r'$\rm{ age \, [Gyr] }$', [1., 3., 5., 7., 9., 11.], cmap=cmapl, fontsize=7, labelsize=7,
                             orientation='horizontal')
        
        name = ''
        
        if disc_stars:
            name += '_disc'
        if accretedfiledir:
            name += '_accreted'
        
        figname1 = '%s/testbirth_sigR_radslice%s%03d.pdf' % (wpath, name, snap)
        figure.fig.savefig(figname1)
        figname2 = '%s/testbirth_sigZ_radslice%s%03d.pdf' % (wpath, name, snap)
        figure2.fig.savefig(figname2)