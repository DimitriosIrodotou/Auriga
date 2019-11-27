from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from scipy import interpolate


def create_axes(f, idx, ncol=5):
    """
    Generate plot axes.
    :param f: figure
    :param idx:
    :param ncol: number of columns
    :return: ax
    """
    ix = idx % ncol
    iy = idx // ncol
    
    ax = f.iaxes(0.5 + ix * 1.5, 1.2 + iy * (1.4), 1.0, 1.0, top=False)
    return ax


def set_axes(isnap, ax, xlabel=None, ylabel=None, title=None, ylim=None):
    """
    Set axes' parameters.
    :param isnap:
    :param ax: axes from from create_axes
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param title: title of the plot
    :param ylim: y-axis limit
    :return: None
    """
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


def radial_profiles(pdf, data, level, redshift):
    """
    Plot radial gas density, gs metallicity, gas energy density, magnetic field strength and gas velocity dispersion profiles for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param level: level from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return:
    """
    
    # Read specific particle type(s) for Auriga haloes #
    data.select_haloes(level, redshift, loadonlytype=[0, 4], loadonlyhalo=0)
    nhalos = data.selected_current_nsnaps
    
    # Generate the figure #
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.4 * nhalos + 0.7))
    
    isnap = 0
    for s in data:
        # Rotate halo based on principal axes #
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        ngas = s.nparticlesall[0]  # Number of gas particles.
        z = np.abs(s.pos[:, 0]) * 1e3  # Convert z direction to kpc.
        rxy = np.sqrt((s.pos[:, 1:] ** 2.).sum(axis=1)) * 1e3  # Distance on the xy plane.
        bfld = np.sqrt((s.data['bfld'] ** 2.).sum(axis=1)) * bfac * 1e6
        mcum, edges = np.histogram(s.r() * 1e3, bins=60, range=[0, 40.], weights=s.mass.astype('f8'))
        for i in range(1, 60):
            mcum[i] += mcum[i - 1]
        
        # Calculate the required quantities for the gas energy densities #
        e_phi = np.zeros((ngas, 3))
        phi = np.arctan2(s.pos[:ngas, 2], s.pos[:ngas, 1])
        e_phi[:, 1] = -np.sin(phi)
        e_phi[:, 2] = +np.cos(phi)
        center = 0.5 * (edges[1:] + edges[:-1])
        vkep = np.sqrt(G * mcum * 1e10 * msol / (center * 1e3 * parsec)) * 1e-5
        fkep = interpolate.interp1d(center, vkep, fill_value=0., bounds_error=False)
        vkep = fkep(rxy)
        
        # Mask the data and calculate the radial velocities #
        i, = np.where(s.r() < 0.1 * s.subfind.data['frc2'][0])
        vel = (s.data['vel'][i, :].astype('f8') * s.mass[i][:, None]).sum(axis=0) / s.mass[i].astype('f8').sum()
        vrad = ((s.data['vel'][:, 1:] - vel[1:]) * s.pos[:, 1:] * 1e3).sum(axis=1) / rxy
        
        # Mask the data and calculate the volume and mass weighted distances for gas particles #
        i, = np.where((rxy < 30.) & (z < 1.) & (s.type == 0))
        mass, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['mass'][i].astype('f8'))
        vol, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['vol'][i].astype('f8'))
        center = 0.5 * (edges[1:] + edges[:-1])
        
        # Mask the data and calculate the mass weighted distances for stellar particles #
        mask, = np.where((rxy < 30.) & (z < 1.) & (s.type == 4))
        np.histogram(rxy[mask], bins=30, range=[0, 30.], weights=s.data['mass'][mask])
        
        for ipanel in range(5):
            ax = create_axes(f, isnap * 5 + ipanel)
            if ipanel == 0:  # Plot gas density.
                ax.semilogy(center, mass / vol / 1e9 * 1e10, 'k')
                set_axes(isnap, ax, "$r\,\mathrm{[kpc]}$", "$\\rho\,\mathrm{[M_\odot\,kpc^{-3}]}$", "$\mathrm{Gas\ density}$", [1e5, 1e8])
            
            elif ipanel == 1:  # Plot gas metallicity.
                metals, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['mass'][i] * s.data['gz'][i])
                ax.semilogy(center, metals / mass / 0.0134, 'k')
                set_axes(isnap, ax, "$r\,\mathrm{[kpc]}$", "$Z\,\mathrm{[Z_\odot]}$", "$\mathrm{Gas\ metallicity}$", [0.5, 20.0])
            
            elif ipanel == 2:  # Plot gas energy densities.
                eB = (s.data['bfld'] ** 2.).sum(axis=1) / (8.0 * np.pi)
                eK = 0.5 * (((s.data['vel'][:ngas, :] - vel) ** 2.0).sum(axis=1) * s.mass[:ngas]) / s.data['vol']
                eT = 0.5 * (((s.data['vel'][:ngas, :] - vel - vkep[:ngas, None] * e_phi) ** 2.).sum(axis=1) * s.mass[:ngas]) / s.data['vol']
                
                u = s.data['u'].astype('f8')
                j, = np.where(s.data['sfr'] == 0.0)
                rad = (s.data['vol'].astype('f8') * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
                u_jeans = 2.0 * 43.0187 * s.mass[:ngas] / np.maximum(5e-4, 2.8 * rad)
                u[j] = np.maximum(u[j], u_jeans[j])
                eU = u * s.mass[:ngas] / s.data['vol']
                
                efac = 1e10 * msol * 1e5 ** 2.0 / 1e18  # Convert to erg / pc^3
                
                reB, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['vol'][i] * eB[i])
                reK, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['vol'][i] * eK[i])
                reU, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['vol'][i] * eU[i])
                reT, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['vol'][i] * eT[i])
                
                ax.semilogy(center, reB / vol * efac, 'k', label='$\\epsilon_\\mathrm{B}$')
                ax.semilogy(center, reK / vol * efac, 'r', label='$\\epsilon_\\mathrm{kin}$')
                ax.semilogy(center, reU / vol * efac, 'b', label='$\\epsilon_\\mathrm{therm}$')
                ax.semilogy(center, reT / vol * efac, 'g', label='$\\epsilon_\\mathrm{turb}$')
                ax.legend(fontsize=4, frameon=False)
                
                set_axes(isnap, ax, "$r\,\mathrm{[kpc]}$", "$\\epsilon\\,\\mathrm{[erg\\ pc^{-3}]}$", "$\mathrm{Gas\ energy\ density}$", [1e42, 1e48])
            
            elif ipanel == 3:  # Plot magnetic field strength.
                bsqr, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['vol'][i] * bfld[i] ** 2.)
                ax.semilogy(center, np.sqrt(bsqr / vol))
                set_axes(isnap, ax, "$r\,\mathrm{[kpc]}$", "$B\,\mathrm{[\mu G]}$", "$\mathrm{Magnetic\ field\ strength}$", [0.1, 100.])
            
            elif ipanel == 4:  # Plot gas velocity dispersion.
                velz, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['mass'][i] * s.data['vel'][i, 0])
                velz /= mass
                velr, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['mass'][i] * vrad[i])
                velr /= mass
                
                binid = np.digitize(rxy[i], edges) - 1
                sigmaz, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['mass'][i] * (s.data['vel'][i, 0] - velz[binid]) ** 2.)
                sigmaz = np.sqrt(sigmaz / mass)
                sigmar, edges = np.histogram(rxy[i], bins=30, range=[0, 30.], weights=s.data['mass'][i] * (vrad[i] - velr[binid]) ** 2.)
                sigmar = np.sqrt(sigmar / mass)
                
                ax.plot(center, sigmaz, 'k', label="$\\sigma_\mathrm{z}$")
                ax.plot(center, sigmar, 'r', label="$\\sigma_\mathrm{r}$")
                ax.legend(fontsize=4, frameon=False)
                
                set_axes(isnap, ax, "$r\,\mathrm{[kpc]}$", "$\\sigma\ \\rm{\\,[km\\,s^{-1}]}$", "$\mathrm{Gas\ velocity\ dispersion}$", [0, 160.])
            else:
                continue
            
            if ipanel == 0:
                ax.text(0.0, 1.01, "Au%s" % s.haloname, color='k', fontsize=6, transform=ax.transAxes)
        
        isnap += 1
    
    pdf.savefig(f)
    return None


def vertical_profiles(pdf, data, levels, z):
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], z)
        nhalos += data.selected_current_nsnaps
    
    plt.close()
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.4 * nhalos + 0.7))
    
    for il in range(nlevels):
        level = levels[il]
        data.select_haloes(level, z, loadonlytype=[0, 4], loadonlyhalo=0)
        
        isnap = 0
        for s in data:
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            
            rxy = np.sqrt((s.pos[:, 1:] ** 2.).sum(axis=1)) * 1e3
            z = np.abs(s.pos[:, 0]) * 1e3
            bfld = np.sqrt((s.data['bfld'] ** 2.).sum(axis=1)) * bfac * 1e6
            
            ngas = s.nparticlesall[0]
            
            mcum, edges = np.histogram(s.r() * 1e3, bins=60, range=[0, 40.], weights=s.mass.astype('f8'))
            for i in range(1, 60):
                mcum[i] += mcum[i - 1]
            
            phi = np.arctan2(s.pos[:ngas, 2], s.pos[:ngas, 1])
            e_phi = np.zeros((ngas, 3))
            e_phi[:, 1] = -np.sin(phi)
            e_phi[:, 2] = +np.cos(phi)
            
            center = 0.5 * (edges[1:] + edges[:-1])
            vkep = np.sqrt(G * mcum * 1e10 * msol / (center * 1e3 * parsec)) * 1e-5
            fkep = interpolate.interp1d(center, vkep, fill_value=0., bounds_error=False)
            vkep = fkep(rxy)
            
            i, = np.where(s.r() < 0.1 * s.subfind.data['frc2'][0])
            vel = (s.data['vel'][i, :].astype('f8') * s.mass[i][:, None]).sum(axis=0) / s.mass[i].astype('f8').sum()
            vrad = ((s.data['vel'][:, 1:] - vel[1:]) * s.pos[:, 1:] * 1e3).sum(axis=1) / rxy
            
            i, = np.where((rxy < 30.) & (z < 10.) & (s.type == 0))
            mass, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['mass'][i].astype('f8'))
            vol, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['vol'][i].astype('f8'))
            center = 0.5 * (edges[1:] + edges[:-1])
            
            for ipanel in range(5):
                ax = create_axes(f, isnap * 5 + ipanel)
                if ipanel == 0:
                    ax.semilogy(center, mass / vol / 1e9 * 1e10, 'k')
                    set_axes(isnap, ax, "$z\,\mathrm{[kpc]}$", "$\\rho\,\mathrm{[M_\odot\,kpc^{-3}]}$", "$\mathrm{Gas\ density}$", [1e5, 1e8])
                elif ipanel == 1:
                    metals, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['mass'][i] * s.data['gz'][i])
                    ax.semilogy(center, metals / mass / 0.0134, 'k')
                    set_axes(isnap, ax, "$z\,\mathrm{[kpc]}$", "$Z\,\mathrm{[Z_\odot]}$", "$\mathrm{Gas\ metallicity}$", [0.3, 10.0])
                elif ipanel == 2:
                    eB = (s.data['bfld'] ** 2.).sum(axis=1) / (8.0 * np.pi)
                    eK = 0.5 * (((s.data['vel'][:ngas, :] - vel) ** 2.).sum(axis=1) * s.mass[:ngas]) / s.data['vol']
                    eT = 0.5 * (((s.data['vel'][:ngas, :] - vel - vkep[:ngas, None] * e_phi) ** 2.).sum(axis=1) * s.mass[:ngas]) / s.data['vol']
                    
                    u = s.data['u'].astype('f8')
                    j, = np.where(s.data['sfr'] == 0.)
                    rad = (s.data['vol'].astype('f8') * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
                    u_jeans = 2.0 * 43.0187 * s.mass[:ngas] / np.maximum(5e-4, 2.8 * rad)
                    u[j] = np.maximum(u[j], u_jeans[j])
                    eU = u * s.mass[:ngas] / s.data['vol']
                    
                    efac = 1e10 * msol * 1e5 ** 2.0 / 1e18  # conversion to erg / pc^3
                    
                    reB, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['vol'][i] * eB[i])
                    reK, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['vol'][i] * eK[i])
                    reU, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['vol'][i] * eU[i])
                    reT, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['vol'][i] * eT[i])
                    
                    ax.semilogy(center, reB / vol * efac, 'k', label='$\\epsilon_\\mathrm{B}$')
                    ax.semilogy(center, reK / vol * efac, 'r', label='$\\epsilon_\\mathrm{kin}$')
                    ax.semilogy(center, reU / vol * efac, 'b', label='$\\epsilon_\\mathrm{therm}$')
                    ax.semilogy(center, reT / vol * efac, 'g', label='$\\epsilon_\\mathrm{turb}$')
                    ax.legend(fontsize=4, frameon=False)
                    
                    set_axes(isnap, ax, "$z\,\mathrm{[kpc]}$", "$\\epsilon\\,\\mathrm{[erg\\ pc^{-3}]}$", "$\mathrm{Gas\ energy\ density}$",
                             [1e42, 1e46])
                elif ipanel == 3:
                    bsqr, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['vol'][i] * bfld[i] ** 2.)
                    ax.semilogy(center, np.sqrt(bsqr / vol))
                    set_axes(isnap, ax, "$z\,\mathrm{[kpc]}$", "$B\,\mathrm{[\mu G]}$", "$\mathrm{Magnetic\ field\ strength}$", [0.1, 30.])
                elif ipanel == 4:
                    velz, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['mass'][i] * s.data['vel'][i, 0])
                    velz /= mass
                    velr, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['mass'][i] * vrad[i])
                    velr /= mass
                    
                    binid = np.digitize(z[i], edges) - 1
                    sigmaz, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['mass'][i] * (s.data['vel'][i, 0] - velz[binid]) ** 2.)
                    sigmaz = np.sqrt(sigmaz / mass)
                    sigmar, edges = np.histogram(z[i], bins=30, range=[0, 10.], weights=s.data['mass'][i] * (vrad[i] - velr[binid]) ** 2.)
                    sigmar = np.sqrt(sigmar / mass)
                    
                    ax.plot(center, sigmaz, 'k', label="$\\sigma_\mathrm{z}$")
                    ax.plot(center, sigmar, 'r', label="$\\sigma_\mathrm{r}$")
                    ax.legend(fontsize=4, frameon=False)
                    
                    set_axes(isnap, ax, "$z\,\mathrm{[kpc]}$", "$\\sigma\ \\rm{\\,[km\\,s^{-1}]}$", "$\mathrm{Gas\ velocity\ dispersion}$", [0, 100.])
                else:
                    continue
                
                if ipanel == 0:
                    ax.text(0.0, 1.0, "Au%s-%d" % (s.haloname, level), color='k', fontsize=6, transform=ax.transAxes)
            
            isnap += 1
    
    pdf.savefig(f)
    return None