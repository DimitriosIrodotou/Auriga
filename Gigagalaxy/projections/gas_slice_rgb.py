from gadget import *
from gadget_subfind import *
from pysph import *
import calcGrid
from util import get_halo_centres
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yt
import PIL

toinch = 0.393700787
BOLTZMANN = 1.38065e-16
PROTONMASS = 1.67262178e-24
GAMMA = 5.0 / 3.0
GAMMA_MINUS1 = GAMMA - 1.0
MSUN = 1.989e33
MPC = 3.085678e24
KPC = 3.085678e21
ZSUN = 0.0127
Gcosmo = 43.0071

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}
# from Asplund et al. (2009) Table 5
SUNABUNDANCES = {'H': 12.0, 'He': 10.98, 'C': 8.47, 'N': 7.87, 'O': 8.73, 'Ne': 7.97, 'Mg': 7.64, 'Si': 7.55, 'Fe': 7.54}


def gas_slice(dir, run, snaplist, outpath, suffix='', boxsize=0.05, loadonlytype=[], numthreads=4, arrows=False):
    path = '%s/%s%s/' % (dir, run, suffix)
    print("Doing gas projection for %s" % path)
    
    outpath += ('/%s/projections/gas' % run)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    if not loadonlytype:
        loadonlytype = [i for i in range(6)]
    
    if run == 'halo_1' or run == 'halo_11':
        sidmallnow = get_halo_centres.get_bound_DMparticles(127, path, loadonlytype)
        loadptype = [0, 1, 4, 5]
    else:
        sidmallnow = []
        loadptype = [0, 4]
    
    for snap in snaplist:
        res = 512
        plt.figure(1, figsize=(4, 7))
        
        s = gadget_readsnap(snap, snappath=path + '/output/', loadonlytype=loadptype,
                loadonly=['pos', 'vel', 'mass', 'sfr', 'u', 'ne', 'gz', 'gmet', 'rho', 'age', 'id', 'hsml', 'vol', 'area'], hdf5=True,
                forcesingleprec=True)
        print(s.time, s.redshift)
        sf = load_subfind(snap, dir=path + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'spos'])
        s.calc_sf_indizes(sf, verbose=False)
        age_select = 3.
        s.select_halo(sf, sidmallnow, age_select, remove_bulk_vel=True, use_principal_axis=False, euler_rotation=True, rotate_disk=True,
                do_rotation=True)
        
        time = s.cosmology_get_lookback_time_from_a(s.time, is_flat=True)
        if time < 0.0001:
            time = 0.0
        print("Time =", time)
        
        dist = np.max(np.abs(s.pos - s.center[None, :]), axis=1)
        igas, = np.where((s.type == 0) & (dist < 0.5 * boxsize))
        sfr = s.data['sfr'][igas].astype('float64')
        ngas = size(igas)
        
        jgas, = np.where((sfr > 0.0))
        
        indy = [[1, 2, 0], [1, 0, 2]]
        ascale = [4000., 4000.]  # 2000.]
        
        for j in range(2):
            
            plt.subplot(2, 1, j + 1)
            
            temp_pos = s.pos[igas, :].astype('float64')
            pos = np.zeros((size(igas), 3))
            pos[:, 0] = temp_pos[:, indy[j][0]]  # 0:1
            pos[:, 1] = temp_pos[:, indy[j][1]]  # 1:2
            pos[:, 2] = temp_pos[:, indy[j][2]]  # 2:0
            
            temp_vel = s.vel[igas, :].astype('float64')
            vel = np.zeros((size(igas), 3))
            vel[:, 0] = temp_vel[:, indy[j][0]]
            vel[:, 1] = temp_vel[:, indy[j][1]]
            vel[:, 2] = temp_vel[:, indy[j][2]]
            
            mass = s.data['mass'][igas].astype('float64')
            
            u = np.zeros(ngas)
            
            ne = s.data['ne'][igas].astype('float64')
            metallicity = s.data['gz'][igas].astype('float64')
            XH = s.data['gmet'][igas, element['H']].astype('float64')
            yhelium = (1 - XH - metallicity) / (4. * XH);
            mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
            u[:] = GAMMA_MINUS1 * s.data['u'][igas].astype('float64') * 1.0e10 * mu * PROTONMASS / BOLTZMANN
            
            rho = np.zeros(ngas)
            rho[:] = s.data['rho'][igas].astype('float64')
            vol = np.zeros(ngas)
            vol[:] = s.data['vol'][igas].astype('float64') * 1e9
            # area = np.zeros(ngas)
            # area[:] = s.data['area'][igas].astype('float64') * 1e6
            
            z = np.zeros(ngas)
            z[:] = pos[:, 2]
            gradius = np.zeros(ngas)
            gradius[:] = np.sqrt((pos[:, :] ** 2).sum(axis=1))
            
            hsml = np.zeros(ngas)
            # hsml[:] = s.data['hsml'][igas].astype('float64')
            hsml[:] = np.array([0.00001] * ngas)
            
            # tree = makeTree( pos )
            # hsml = tree.calcHsmlMulti( pos, pos, mass, 48, numthreads=numthreads )
            # hsml = np.minimum( hsml, 4. * boxsize / res )
            # hsml = np.maximum( hsml, 1.001 * boxsize / res * 0.5 )
            
            sfuthresh = u[np.where(sfr > 0.)].min()
            
            if boxsize <= 0.1:
                bmax, bmin = 0.8, -.5
                gmax, gmin = 1.2, -1.
                rmax, rmin = 1., 0.  # 0.7, 0.
                pfac = 8
                sfgas = np.where((u < 2e4))
                hotgas = np.where((u >= 5e5))
                medgas = np.where((u >= 1e4) & (u < 6e5))
            elif boxsize > 0.1:
                bmax, bmin = 1.7, -0.3  # 1.6,-0.5
                gmax, gmin = 2., -2.  # 2,-1
                rmax, rmin = 1., 0.2  # 0.9, 0.2
                pfac = 6
                sfgas = np.where((u < 2e4))
                hotgas = np.where((u >= 4e5))  # >1e5 (early snaps)
                medgas = np.where((u >= 1.5e4) & (u < 5e5))
            
            # print "hot gas area, min, max=",area[hotgas],area[hotgas].min(),area[hotgas].max()
            
            frbs = []
            # frbs = np.zeros( (res, res, 3) )
            for i in range(3):
                if i == 0:
                    gpos = np.zeros((size(hotgas), 3))
                    gmass = np.zeros((size(hotgas)))
                    grho = np.zeros((size(hotgas)))
                    gpos[:, :] = pos[hotgas, :]
                    gmass[:] = mass[hotgas]
                    grho[:] = u[hotgas]
                    ghsml = hsml[hotgas]
                    nref = 4
                if i == 1:
                    gpos = np.zeros((size(medgas), 3))
                    gmass = np.zeros((size(medgas)))
                    grho = np.zeros((size(medgas)))
                    gpos[:, :] = pos[medgas, :]
                    gmass[:] = mass[medgas]
                    grho[:] = rho[medgas]
                    ghsml = hsml[medgas]
                    nref = 1
                if i == 2:
                    gpos = np.zeros((size(sfgas), 3))
                    gmass = np.zeros((size(sfgas)))
                    grho = np.zeros((size(sfgas)))
                    gpos[:, :] = pos[sfgas, :]
                    gmass[:] = mass[sfgas]
                    grho[:] = rho[sfgas]
                    ghsml = hsml[sfgas]
                    nref = 1
                
                A = calcGrid.calcASlice(gpos, grho, res, res, boxx=boxsize, boxy=boxsize, centerx=s.center[0], centery=s.center[1],
                        centerz=s.center[2], grad=gpos, proj=True, boxz=boxsize / pfac, nz=res / pfac, numthreads=8)
                
                slic = A["grid"]
                frbs.append(np.array(slic))
                
                rgbArray = numpy.zeros((res, res, 3), 'uint8')
            
            for i in range(3):
                frbs[i][np.where(frbs[i] == 0)] = 1e-10
                frbs_flat = frbs[i].flatten()
                asort = numpy.argsort(frbs_flat)
                frbs_flat = frbs_flat[asort]
                CumSum = numpy.cumsum(frbs_flat)
                CumSum /= CumSum[-1]
                halflight_val = frbs_flat[numpy.where(CumSum > 0.5)[0][0]]
                
                if i == 0:
                    Max = numpy.log10(halflight_val) + rmax
                    Min = numpy.log10(halflight_val) + rmin
                elif i == 1:
                    Max = numpy.log10(halflight_val) + gmax
                    Min = numpy.log10(halflight_val) + gmin
                elif i == 2:
                    Max = numpy.log10(halflight_val) + bmax
                    Min = numpy.log10(halflight_val) + bmin
                
                print("frbs, max, min=", np.log10(frbs[i]), np.log10(frbs[i]).max(), np.log10(frbs[i]).min())
                
                Min -= 0.1
                Max -= 0.5
                print("Max,Min=", Max, Min)
                
                Color = (numpy.log10(frbs[i]) - Min) / (Max - Min)  # - 0.5
                print("color=", Color)
                Color[numpy.where(Color < 0.)] = 0.
                Color[numpy.where(Color > 1.0)] = 1.0
                
                if i == 0:
                    Color[np.where(Color < 0.15)] = 0.
                
                print("num zero=", size(np.where(Color <= 0.)))
                A = numpy.array(Color * 255, dtype=numpy.uint8)
                if j == 0:
                    rgbArray[:, :, i] = A.T  # np.flipud(A.transpose())
                elif j == 1:
                    rgbArray[:, :, i] = A.T  # np.fliplr(A)
                
                # if i == 0:  #    rgbArray[3*(res/4):,:,i] = 0.  #    rgbArray[:(res/4),:,i] = 0.  #    rgbArray[:,3*(res/4):,i] = 0.  #
                # rgbArray[:,:(res/4),i] = 0.
            
            img = PIL.Image.fromarray(rgbArray)
            plt.imshow(img, rasterized=True)
            
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_xticklabels([], fontsize=8)
            ax.set_yticks([])
            ax.set_yticklabels([], fontsize=8)
            
            # Arrows for velocity field
            if arrows:
                nbin = 30
                d1, d2 = 0, 1  # 2,1
                pn, xedges, yedges = np.histogram2d(pos[:, d1], pos[:, d2], bins=(nbin, nbin),
                                                    range=[[-0.5 * boxsize, 0.5 * boxsize], [-0.5 * boxsize, 0.5 * boxsize]])
                vxgrid, xedges, yedges = np.histogram2d(pos[:, d1], pos[:, d2], bins=(nbin, nbin), weights=vel[:, d1],
                                                        range=[[-0.5 * boxsize, 0.5 * boxsize], [-0.5 * boxsize, 0.5 * boxsize]])
                vygrid, xedges, yedges = np.histogram2d(pos[:, d1], pos[:, d2], bins=(nbin, nbin), weights=vel[:, d2],
                                                        range=[[-0.5 * boxsize, 0.5 * boxsize], [-0.5 * boxsize, 0.5 * boxsize]])
                vxgrid /= pn
                vygrid /= pn
                
                xbin = np.zeros(len(xedges) - 1)
                ybin = np.zeros(len(yedges) - 1)
                xbin[:] = 0.5 * (xedges[:-1] + xedges[1:])
                ybin[:] = 0.5 * (yedges[:-1] + yedges[1:])
                xbin -= xedges[0]
                xbin *= (10 * res) * (0.1 / boxsize)
                ybin -= yedges[-1]
                ybin *= (-10 * res) * (0.1 / boxsize)
                
                xc, yc = np.meshgrid(xbin, ybin)
                
                vygrid *= (-1.)
                # p = plt.quiver(xc, yc, np.flipud(vxgrid.T), np.flipud(vygrid.T), scale=ascale[j], pivot='middle', color='yellow', alpha=0.8)
                p = plt.quiver(xc, yc, np.rot90(vxgrid), np.rot90(vygrid), scale=ascale[j], pivot='middle', color='yellow', alpha=0.8)
            
            if j == 0:
                plt.title("z = %1.2f, t = %1.2f Gyr" % (s.redshift, time))
        
        del s
        del sf
        savefig('%s/gasslice_%s_bx%06d_snap%03d_lres.pdf' % (outpath, run, int(boxsize * 1e3), snap), dpi=300)
        
        plt.close()