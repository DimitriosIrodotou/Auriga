from loadmodules import *
from galpy import *
from galpy import potential
from galpy.potential import MWPotential2014 as pote
from galpy.actionAngle import actionAngleStaeckel
from galpy.actionAngle import estimateDeltaStaeckel
from pylab import *
from const import *
from util import plot_helper
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from lmfit import Model
from scipy import special, integrate
import sys

potdict = {"disk_exp":   potential.DoubleExponentialDiskPotential, "bulge_hernquist": potential.HernquistPotential,
           "dmhalo_NFW": potential.NFWPotential, "gasdisk_exp": potential.DoubleExponentialDiskPotential}


class actions():
    
    def __init__(self, potlist, dobject=None, verbose=False):
        """
        Some initialization
        """
        p = plot_helper.plot_helper()
        self.fdict = {}
        
        self.potlist = potlist
        
        if dobject:
            self.d = dobject
        
        self.verbose = verbose
    
    def build_potential(self, potlist, potargs):
        # Pass in full list of potentials and potential arguments here
        nparams = []
        for potcomp in potlist:
            params = potargs[potcomp]
            nparams.append(len(params))
        
        return params, nparams
    
    def normalise_components(self, paramset, index=None, userotcurve=True):  # , nshells=100, radius=0.025, nr=50):
        ndict = {}
        self.ndict = {}
        # typedict = {"gasdisk":0, "dmhalo":1, "stars":4}
        self.ndict['all'] = 0.
        RSun = 8.
        zSun = 0.
        G = 4.3e-6
        
        if not userotcurve:
            
            for i, key in enumerate(self.potlist):
                if key.split('_')[1] == 'NFW':
                    norm = paramset[key][0]  # *1e9
                    slope = paramset[key][1]  # parameter a in eq 2.64 (BT08)
                    massenc = paramset[key][2] * 1e10
                    print("massenc NFW before=", massenc)
                    massenc = 4. * np.pi * norm * slope * slope * slope * (np.log(1. + RSun / slope) - ((RSun / slope) / (1. + RSun / slope)))
                    print("NFW massenc=", massenc)
                    ndict[key] = G * massenc / (RSun)
                if key.split('_')[1] == 'hernquist':
                    norm = paramset[key][0]
                    slope = paramset[key][1]  # parameter a in eq 2.64 (BT08)
                    massenc = 4. * np.pi * norm * slope * slope * slope * (RSun / slope) ** 2 / (2. * (1. + RSun / slope) ** 2)
                    print("Hern massenc=", massenc)
                    ndict[key] = G * massenc / RSun
                if key.split('_')[1] == 'exp':
                    norm = paramset[key][0]
                    hr = paramset[key][1]  # parameter a in eq 2.165 (BT08)
                    hz = paramset[key][2]
                    y = RSun / hr
                    
                    glx, glw = np.polynomial.legendre.leggauss(10)
                    nzeros = 10
                    j0zeros = np.zeros(nzeros + 1)
                    j0zeros[1:nzeros + 1] = special.jn_zeros(0, nzeros)
                    dj0zeros = j0zeros - np.roll(j0zeros, 1)
                    dj0zeros[0] = j0zeros[0]
                    j1zeros = np.zeros(nzeros + 1)
                    j1zeros[1:nzeros + 1] = special.jn_zeros(1, nzeros)
                    dj1zeros = j1zeros - np.roll(j1zeros, 1)
                    dj1zeros[0] = j1zeros[0]
                    kmaxFac = 2.
                    beta = hz  # / RSun
                    alpha = hr  # / RSun
                    
                    R4max = RSun
                    kmax = kmaxFac * beta
                    kmax = 2. * kmaxFac * beta
                    maxj1zeroIndx = np.argmin((j1zeros - kmax * R4max) ** 2.)  # close enough
                    ks = np.array([0.5 * (glx + 1.) * dj1zeros[ii + 1] + j1zeros[ii] for ii in range(maxj1zeroIndx)]).flatten()
                    weights = np.array([glw * dj1zeros[ii + 1] for ii in range(maxj1zeroIndx)]).flatten()
                    evalInt = ks * special.jn(1, ks * RSun) * (alpha ** 2. + ks ** 2.) ** -1.5 * (
                                beta * np.exp(-ks * np.fabs(zSun)) - ks * np.exp(-beta * np.fabs(zSun))) / (beta ** 2. - ks ** 2.)
                    ndict[key] = (220. ** 2) * 2. * np.pi * alpha * np.sum(
                        weights * evalInt)  # (divided by R in units of RSun: /1.) # multiply by bovy's velocity units (220 km/s)
        
        else:
            nshells = 100
            radius = 0.04
            
            dmass = pylab.zeros((nshells, 6))
            dvel = pylab.zeros((nshells, 6))
            
            dr = radius / float(nshells)
            
            tpos = np.zeros((0, 3))
            tmass = np.zeros(0)
            
            for i, val in enumerate(self.d.loadptype):
                
                pos = self.d.sgdata[self.d.datatypes[val]]['pos']
                mass = self.d.sgdata[self.d.datatypes[val]]['mass']
                
                rp = calcGrid.calcRadialProfile(pos.astype('float64'), mass.astype('float64'), 0, nshells, dr, 0., 0., 0.)
                
                dradius = rp[1, :]
                dmass[:, i] = rp[0, :]
                
                for j in range(1, nshells):
                    dmass[j, i] += dmass[j - 1, i]
                
                dvel[:, i] = pylab.sqrt(6.674e-08 * dmass[:, i] * 1e10 * msol / (dradius * 1e6 * parsec)) / 1e5
                
                tpos = np.vstack((tpos, pos))
                tmass = np.concatenate((tmass, mass))
            
            rp = calcGrid.calcRadialProfile(tpos.astype('float64'), tmass.astype('float64'), 0, nshells, dr, 0., 0., 0.)
            
            dradius = rp[1, :]
            mtot = rp[0, :]
            
            for j in range(1, nshells):
                mtot[j] += mtot[j - 1]
            
            indy = np.int_(find_nearest(dradius, [0.008]).astype('int64'))
            vtot = pylab.sqrt(6.674e-08 * mtot * 1e10 * msol / (dradius * 1e6 * parsec)) / 1e5
            
            velRsun = dvel[indy]
            
            print("vel at Sun=", velRsun)
            print("vtot=", vtot[indy])
            
            # Get hernquist bulge contribution
            norm = paramset['bulge_hernquist'][index][0]
            slope = paramset['bulge_hernquist'][index][1]  # parameter a in eq 2.64 (BT08)
            massenc = 4. * np.pi * norm * slope * slope * slope * (RSun / slope) ** 2 / (2. * (1. + RSun / slope) ** 2)
            ndict['bulge_hernquist'] = G * massenc / RSun
            # Subtract from rest of stellar component (assuming the rest is the disc)
            ndict['disk_exp'] = velRsun[2] ** 2 - ndict['bulge_hernquist']
            
            ndict['dmhalo_NFW'] = velRsun[1] ** 2
            
            ndict['gasdisk_exp'] = velRsun[0] ** 2
        
        for i, key in enumerate(self.potlist):
            print("<<< sqrt ndict=", sqrt(ndict[key]), key)
            
            self.ndict['all'] += ndict[key]
        
        for i, key in enumerate(self.potlist):
            self.ndict[key] = ndict[key]
            self.ndict[key] /= self.ndict['all']
        
        print("ndict=", ndict)
        print("self ndict=", self.ndict)
        
        return ndict
    
    def set_potential_args_arrays(self, potargs, ndict, nparam, times):
        self.potargsall = potargs
        self.ndictall = ndict
        self.nparam = nparam
        self.time = times
    
    def construct_potential(self, potlist, pottype, nparam, filename, galrad=0.025, disc_stars=False, atbirth=False, readparams=None, totfit=False):
        print("Constructing potential")
        
        print("reading parameters:", readparams)
        potargs = {}
        
        for key in self.paramset:
            component = key.split('_')[0]
            potargs[component] = self.paramset[key]
        
        ndict = self.normalise_components()
        
        return potargs, ndict
    
    def get_actions(self, sdata, potlist, pottype, rcut=0.025, zcut=0.005, nsk=1, ferange=[], fa='', disc_stars=False, atbirth=False):
        
        print("Calculating actions...")
        
        ageinduni = np.unique(ageindex)
        for j, t in enumerate(ageinduni[3:]):
            potargsb = {}
            ndictb = {}
            for i, potcomp in enumerate(potlist):
                plist = self.potargsall[potcomp]
                pn = self.nparam[potcomp]
                potargsb[potcomp] = plist[t * pn:t * pn + pn]
                nlist = self.ndictall[potcomp]
                ndictb[potcomp] = nlist[t]
        
        jR, jz, lz = self.calculate_actions(phase_space_dat, potlist, pottype, potargsb, ndictb)
        
        return jR, jz, lz
    
    def galpy_potential(self, paramset, dradius=None, Snhd=False, verbose=False):
        print("Constructing galpy potential...")
        
        pote = []
        
        # galpy scale units:
        _REFR0 = 8.  # [kpc]  --> galpy length unit
        _REFV0 = 220.  # [km/s] --> galpy velocity unit
        
        for potcomp in paramset.keys():
            pargs = paramset[potcomp]
            
            if potcomp == "disk_exp":
                pot = potdict[potcomp](hr=pargs[1] / _REFR0, hz=pargs[2] / _REFR0, normalize=self.ndict[potcomp])
                print("potcomp,pargs=", pargs, potcomp)
            else:
                pot = potdict[potcomp](a=pargs[1] / _REFR0, normalize=self.ndict[potcomp])
            
            pote.append(pot)
        
        return pote
    
    def calculate_actions(self, phase_space_dat, paramset, index, dradius=None, verbose=False):
        
        print("Entering galpy action calculation routines...")
        
        pote = []
        
        # galpy scale units:
        _REFR0 = 8.  # [kpc]  --> galpy length unit
        _REFV0 = 220.  # [km/s] --> galpy velocity unit
        
        for potcomp in paramset.keys():
            pargs = paramset[potcomp][index]
            if potcomp == "disk_exp" or potcomp == "gasdisk_exp":
                pot = potdict[potcomp](hr=pargs[1] / _REFR0, hz=pargs[2] / _REFR0, normalize=self.ndict[potcomp] * (sqrt(self.ndict['all']) / _REFV0))
            else:
                pot = potdict[potcomp](a=pargs[1] / _REFR0, normalize=self.ndict[potcomp] * (sqrt(self.ndict['all']) / _REFV0))
            
            pote.append(pot)
        if self.verbose:
            print("pote=", pote)
            print("paramset=", paramset)
        
        # unpack phase space data
        star_radius, star_vR, star_vT, star_height, star_vz = zip(*phase_space_dat)
        star_radius = np.array(star_radius)
        star_vR = np.array(star_vR)
        star_vT = np.array(star_vT)
        star_height = np.array(star_height)
        star_vz = np.array(star_vz)
        
        print("Calculating actions for %d stars" % size(star_radius))
        R = star_radius * 1e3 / _REFR0
        vR = star_vR / _REFV0
        vT = star_vT / _REFV0
        z = star_height * 1e3 / _REFR0
        vz = star_vz / _REFV0
        
        delta = 0.5
        if self.verbose:
            print("Focal length of Staeckel potential=", delta)
        
        aAS = actionAngleStaeckel(pot=pote, delta=delta, c=True)
        nsk = 1
        jR, lz, jz = aAS(R[::nsk], vR[::nsk], vT[::nsk], z[::nsk], vz[::nsk])
        vT *= _REFV0
        star_radius *= _REFR0
        
        if self.verbose:
            print("Radial   action  J_R = ", jR * _REFR0 * _REFV0, "\t kpc km/s")
            print("Vertical action  J_z = ", jz * _REFR0 * _REFV0, "\t kpc km/s")
            print("Angular momentum L_z = ", lz * _REFR0 * _REFV0, "\t kpc km/s")
        
        jR = jR * _REFR0 * _REFV0
        jz = jz * _REFR0 * _REFV0
        lz = lz * _REFR0 * _REFV0
        
        return (jR, jz, lz)


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx