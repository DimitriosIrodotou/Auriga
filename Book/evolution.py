from __future__ import division

import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from const import *
from numpy import cos, sin
from sfigure import *

mass_proton = 1.6726219e-27


def create_axis(f, idx, ncol=5):
    ix = idx % ncol
    iy = idx // ncol
    
    s = 1.
    
    ax = f.iaxes(0.5 + ix * (s + 0.5), 0.2 + s + iy * (s + 0.4), s, s, top=False)
    return ax


def set_axis(isnap, ax, xlabel=None, ylabel=None, title=None, ylim=None, ncol=5):
    if ylabel is None:
        ax.set_yticks([])
    else:
        ax.set_ylabel(ylabel, size=6)
    
    if xlabel is None:
        ax.set_xticks([])
    else:
        ax.set_xlabel(xlabel, size=6)
    
    for label in ax.xaxis.get_ticklabels():
        label.set_size(6)
    for label in ax.yaxis.get_ticklabels():
        label.set_size(6)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if isnap == 0 and title is not None:
        ax.set_title(title, size=7)
    
    return None


def bar_strength(pdf, data, levels, z):
    """
        Calculate bar strength from Fourier modes of surface density (see e.g. sec 2.3.2 from Athanassoula et al. 2013)
        :param pdf:
        :param data:
        :param levels:
        :return:
        """
    
    nlevels = len(levels)
    
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 8.2))
    ax = f.iaxes(1.0, 1.0, 6.8, 6.8, top=True)
    ax.set_ylabel("$A_{2}$")
    ax.set_xlabel("$r\,\mathrm{[kpc]}$")
    
    for il in range(nlevels):
        level = levels[il]
        
        data.select_halos(level, z)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
        
        data.select_halos(level, z, loadonlytype=[0, 4], loadonlyhalo=0)
        
        ihalo = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])
            
            galrad = 0.03
            age = np.zeros(s.npartall)
            age[s.type == 4] = s.data['age']  # Get ages of stars.
            istars, = np.where((s.type == 4) & (age > 0.) & (s.r() < galrad))  # Select stars.
            x, y = s.pos[istars, 2] * 1000, s.pos[istars, 1] * 1000  # Load positions and convert from Mpc to Kpc.

            nbins = 40  # Number of radial bins.
            r = np.sqrt(x[:] ** 2 + y[:] ** 2)  # Radius of each particle.

            # Initialise Fourier components #
            r_m = np.zeros(nbins)
            alpha_0 = np.zeros(nbins)
            alpha_2 = np.zeros(nbins)
            beta_2 = np.zeros(nbins)

            # Split up galaxy in radius bins and calculate Fourier components #
            for i in range(0, nbins):
    
                r_s = float(i) * 0.5
                r_b = float(i) * 0.5 + 0.25
                r_m[i] = float(i) * 0.5 + 0.125
    
                xfit = x[(r < r_b) & (r > r_s)]
                yfit = y[(r < r_b) & (r > r_s)]
    
                for k in range(0, len(xfit)):
                    th_i = math.atan2(yfit[k], xfit[k])
                    alpha_0[i] = alpha_0[i] + 1
                    alpha_2[i] = alpha_2[i] + cos(2 * th_i)
                    beta_2[i] = beta_2[i] + sin(2 * th_i)

            # Calculate bar strength A_2
            a2 = np.divide(np.sqrt(alpha_2[:] ** 2 + beta_2[:] ** 2), alpha_0[:])
            
            # Plot bar strength as a function of radius plot r_m versus a2
            ax.plot(r_m, a2, color=next(colors), label="Au%s-%d max(A2): %.2f" % (s.haloname, levels[0], max(a2)))
            ax.legend(loc='top left', fontsize=12, frameon=False, numpoints=1)
            ax.text(0.05, 0.92, "z = %.1f" % z, color='k', fontsize=12, transform=ax.transAxes)
            
            ihalo += 1
    
    pdf.savefig(f)
    return None


def circularity(pdf, data, levels, z):
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_halos(levels[il], z)
        nhalos += data.selected_current_nsnaps
    
    Gcosmo = 43.0071
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.4 * ((nhalos - 1) // 5 + 1) + 0.7))
    
    for il in range(nlevels):
        level = levels[il]
        data.select_halos(level, z, loadonlyhalo=0)
        
        isnap = 0
        for s in data:
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, do_rotation=True)
            
            age = np.zeros(s.npartall)
            age[s.type == 4] = s.data['age']
            
            galrad = 0.1 * s.subfind.data['frc2'][0]
            iall, = np.where((s.r() < galrad) & (s.r() > 0.))
            istars, = np.where((s.r() < galrad) & (s.r() > 0.) & (s.type == 4) & (age > 0.))
            nall = np.size(iall)
            nstars = np.size(istars)
            
            rsort = s.r()[iall].argsort()
            msum = np.zeros(nall)
            msum[rsort] = np.cumsum(s.mass[iall][rsort])
            
            nn, = np.where((s.type[iall] == 4) & (age[iall] > 0.))
            smass = s.mass[iall][nn]
            jz = np.cross(s.pos[iall, :][nn, :].astype('f8'), s.data['vel'][iall, :][nn, :])[:, 0]
            ene = 0.5 * (s.vel[iall, :][nn, :].astype('f8') ** 2.).sum(axis=1) + s.data['pot'][iall][nn].astype('f8')
            esort = ene.argsort()
            
            jz = jz[esort]
            smass = smass[esort]
            
            jcmax = np.zeros(nstars)
            for nn in range(nstars):
                if nn < 50:
                    left = 0
                    right = 100
                elif nn > nstars - 50:
                    left = nstars - 100
                    right = nstars
                else:
                    left = nn - 50
                    right = nn + 50
                
                jcmax[nn] = np.max(jz[left:right])
            
            eps = jz / jcmax
            
            ax = create_axis(f, isnap)
            ydata, edges = np.histogram(eps, weights=smass / smass.sum(), bins=100, range=[-1.7, 1.7])
            ydata /= edges[1:] - edges[:-1]
            ax.plot(0.5 * (edges[1:] + edges[:-1]), ydata, 'k')
            
            set_axis(isnap, ax, "$\\epsilon$", "$f\\left(\\epsilon\\right)$", None)
            ax.text(0.05, 0.90, "Au%s-%d z = %.1f " % (s.haloname, level, z), color='k', fontsize=6, transform=ax.transAxes)
            ax.set_xlim(-2., 2.)
            ax.set_xticks([-1.5, 0., 1.5])
            
            isnap += 1
    
    pdf.savefig(f)
    return None