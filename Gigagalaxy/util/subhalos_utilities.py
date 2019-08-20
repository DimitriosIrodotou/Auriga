from gadget_subfind import *


class subhalos_properties:
    def __init__(self, snap, directory, base='fof_subhalo_tab_', loadonly=False, verbose=False, hdf5=True):
        self.groupid = -1
        self.numbersubhalos = -1
        
        # load subfind catalogue
        self.catalogue = load_subfind(snap, base=base, dir=directory, loadonly=loadonly, verbose=verbose, hdf5=hdf5)
    
    def block_loaded(self, value):
        return self.catalogue.data.has_key(value)
    
    def set_subhalos_properties(self, parent_group=None, photometry=False, sfr=False, main_halo=True, center_to_main_halo=True, verbose=True):
        print("set subhalo props")
        if parent_group == None:
            self.groupid = parent_group
            if self.block_loaded('fnsh'):
                self.numbersubhalos = self.catalogue.data['fnsh'][:].sum()
            self.offsets = 0
            
            if self.block_loaded('smas'):
                self.subhalosmasses = self.catalogue.data['smas'][self.offsets:self.offsets + self.numbersubhalos]
            if self.block_loaded('smty'):
                self.subhalosmassestype = self.catalogue.data['smty'][self.offsets:self.offsets + self.numbersubhalos]
            if self.block_loaded('svmx'):
                self.subhalosvcirc = self.catalogue.data['svmx'][self.offsets:self.offsets + self.numbersubhalos]
            if self.block_loaded('svrx'):
                self.subhalosradvcirc = self.catalogue.data['svrx'][self.offsets:self.offsets + self.numbersubhalos]
            if self.block_loaded('smir'):
                self.subhalosmrad = self.catalogue.data['smir'][self.offsets:self.offsets + self.numbersubhalos]
            if self.block_loaded('smit'):
                self.subhalosmradtype = self.catalogue.data['smit'][self.offsets:self.offsets + self.numbersubhalos]
            if self.block_loaded('shmr'):
                self.subhalossize = self.catalogue.data['shmr'][self.offsets:self.offsets + self.numbersubhalos]
            if self.block_loaded('spos'):
                self.subhalospos = self.catalogue.data['spos'][self.offsets:self.offsets + self.numbersubhalos]
            
            if center_to_main_halo:
                center = self.catalogue.data['fpos'][0]
                self.subhalospos -= center
            
            if photometry:
                if self.block_loaded('ssph'):
                    self.subhalosphotometry = self.catalogue.data['ssph'][self.offsets:self.offsets + self.numbersubhalos]
            
            if sfr:
                if self.block_loaded('ssfr'):
                    self.subhalossfr = self.catalogue.data['ssfr'][self.offsets:self.offsets + self.numbersubhalos]
        else:
            self.groupid = parent_group
            if self.block_loaded('frc2'):
                self.rvir = self.catalogue.data['frc2'][self.groupid]
            if self.block_loaded('fnsh'):
                self.numbersubhalos = self.catalogue.data['fnsh'][:]
                self.offsets = np.zeros(len(self.catalogue.data['fnsh'][:]), dtype=int64)
                self.offsets[1:] = np.cumsum(self.numbersubhalos[:-1])
                self.particlesoffsets = np.zeros((self.numbersubhalos[self.groupid], 6), dtype=int64)
                self.particlesoffsets[1:, :] = np.cumsum(
                        self.catalogue.data['slty'][self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid] - 1, :],
                        axis=0)
                
                groupoffsets = np.zeros((len(self.offsets), 6), dtype=int32)
                groupoffsets[1:] = np.cumsum(self.catalogue.data['flty'][:-1, :], axis=0)
                self.particlesoffsets[:, :] += groupoffsets[self.groupid, :]
            
            if self.block_loaded('spos'):
                self.subhalospos = self.catalogue.data['spos'][
                                   self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            if self.block_loaded('svel'):
                self.subhalosvel = self.catalogue.data['svel'][
                                   self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            
            if center_to_main_halo:
                center = self.catalogue.data['fpos'][self.groupid]
                self.subhalospos -= center
            
            if self.block_loaded('smas'):
                self.subhalosmasses = self.catalogue.data['smas'][
                                      self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            if self.block_loaded('smty'):
                self.subhalosmassestype = self.catalogue.data['smty'][
                                          self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            if self.block_loaded('svmx'):
                self.subhalosvcirc = self.catalogue.data['svmx'][
                                     self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            if self.block_loaded('svrx'):
                self.subhalosradvcirc = self.catalogue.data['svrx'][
                                        self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            if self.block_loaded('smir'):
                self.subhalosmrad = self.catalogue.data['smir'][
                                    self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            if self.block_loaded('smit'):
                self.subhalosmradtype = self.catalogue.data['smit'][
                                        self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            if self.block_loaded('shmr'):
                self.subhalossize = self.catalogue.data['shmr'][
                                    self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            if self.block_loaded('slty'):
                self.subhaloslentype = self.catalogue.data['slty'][
                                       self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            if self.block_loaded('shmt'):
                self.subhaloshalfmassradiustype = self.catalogue.data['shmt'][
                                                  self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            
            if photometry:
                if self.block_loaded('ssph'):
                    self.subhalosphotometry = self.catalogue.data['ssph'][
                                              self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
            
            if sfr:
                if self.block_loaded('ssfr'):
                    self.subhalossfr = self.catalogue.data['ssfr'][
                                       self.offsets[self.groupid]:self.offsets[self.groupid] + self.numbersubhalos[self.groupid]]
        
        if self.groupid is not None:
            # order properties by subhalo mass
            # i = np.argsort(self.subhalosmasses)
            # self.subhalospos = self.subhalospos[i]
            # self.subhalosmasses = self.subhalosmasses[i]
            # self.subhalosmassestype = self.subhalosmassestype[i]
            # self.subhalosvcirc = self.subhalosvcirc[i]
            # self.subhalosradvcirc = self.subhalosradvcirc[i]
            # self.subhalosmrad = self.subhalosmrad[i]
            # self.subhalosmradtype = self.subhalosmradtype[i]
            
            # if photometry:
            #	self.subhalosphotometry = self.subhalosphotometry[i]
            
            # if sfr:
            #	self.subhalossfr = self.subhalossfr[i]
            
            # exclude most massive halo (central galaxy) from the computation
            if not main_halo:
                self.numbersubhalos[self.groupid] -= 1
                self.offsets = self.offsets[1:]
                self.particlesoffsets = self.particlesoffsets[1:]
                if self.block_loaded('spos'):
                    self.subhalospos = self.subhalospos[1:]
                if self.block_loaded('svel'):
                    self.subhalosvel = self.subhalosvel[1:]
                if self.block_loaded('smas'):
                    self.subhalosmasses = self.subhalosmasses[1:]
                if self.block_loaded('smty'):
                    self.subhalosmassestype = self.subhalosmassestype[1:]
                if self.block_loaded('svmx'):
                    self.subhalosvcirc = self.subhalosvcirc[1:]
                if self.block_loaded('svrx'):
                    self.subhalosradvcirc = self.subhalosradvcirc[1:]
                if self.block_loaded('smir'):
                    self.subhalosmrad = self.subhalosmrad[1:]
                if self.block_loaded('smit'):
                    self.subhalosmradtype = self.subhalosmradtype[1:]
                if self.block_loaded('shmr'):
                    self.subhalossize = self.subhalossize[1:]
                if self.block_loaded('slty'):
                    self.subhaloslentype = self.subhaloslentype[1:]
                if self.block_loaded('shmt'):
                    self.subhaloshalfmassradiustype = self.subhaloshalfmassradiustype[1:]
                
                if photometry:
                    if self.block_loaded('ssph'):
                        self.subhalosphotometry = self.subhalosphotometry[1:]
                if sfr:
                    if self.block_loaded('ssfr'):
                        self.subhalossfr = self.subhalossfr[1:]
        
        if verbose:
            print(self.groupid)
            print(self.numbersubhalos)
            print(self.offsets)
    
    def get_masses_vs_stellar_masses(self, rcut=False, rvcirc_masses=False):
        if rvcirc_masses:
            if rcut:
                r2 = (self.subhalospos[:] * self.subhalospos[:]).sum(axis=1)
                j, = np.where(r2 < rcut * rcut)
                return self.subhalosmasses[j], self.subhalosmassestype[j, 4]  # return self.subhalosmasses[j], self.subhalosmradtype[j,4]
            else:
                return self.subhalosmasses, self.subhalosmassestype[:, 4]  # self.subhalosmradtype[:,4]
        else:
            if rcut:
                r2 = (self.subhalospos[:] * self.subhalospos[:]).sum(axis=1)
                j, = np.where(r2 < rcut * rcut)
                return self.subhalosmasses[j], self.subhalosmassestype[:, 4]
            else:
                return self.subhalosmasses, self.subhalosmassestype[:, 4]
    
    def get_differential_subhalo_mass_function(self, bins, range, rcut=False, logscale=False, normed=False, weights=None, density=False):
        
        if logscale:
            masses = log10(self.subhalosmasses)
            if range:
                range = log10(range)
        else:
            masses = self.subhalosmasses
        
        if rcut is not False:
            r2 = (self.subhalospos[:] * self.subhalospos[:]).sum(axis=1)
            
            j, = np.where(r2 < rcut * rcut)
            masses = masses[j]
        
        values, edges = np.histogram(masses, bins=bins, range=range, normed=normed, weights=weights, density=density)
        bin = 0.5 * (edges[1:] + edges[:-1])
        return values, bin, edges
    
    def get_cumulative_subhalo_mass_function(self, bins, range, rcut=False, logscale=False, normed=False, weights=None, density=False, inverse=False):
        values, bin, edges = self.get_differential_subhalo_mass_function(bins, range, rcut, logscale, normed, weights, density)
        
        if inverse:
            # reverse bin and values
            bin = edges[::-1]
            bin = bin[:-1]
            values = np.cumsum(values[::-1])
        else:
            bin = edges[1:]
            values = np.cumsum(values)
        
        return values, bin
    
    def get_differential_subhalo_vcirc_function(self, bins, range, rcut=False, logscale=False, normed=False, weights=None, density=False):
        if logscale:
            vcirc = log10(self.subhalosvcirc)
            if range:
                range = log10(range)
        else:
            vcirc = self.subhalosvcirc
        
        if rcut is not False:
            r2 = (self.subhalospos[:] * self.subhalospos[:]).sum(axis=1)
            j, = np.where(r2 < rcut * rcut)
            vcirc = vcirc[j]
        
        values, edges = histogram(vcirc, bins=bins, range=range, normed=normed, weights=weights, density=density)
        bin = 0.5 * (edges[1:] + edges[:-1])
        
        return values, bin, edges
    
    def get_cumulative_subhalo_vcirc_function(self, bins, range, rcut=False, logscale=False, normed=False, weights=None, density=False,
                                              inverse=False):
        values, bin, edges = self.get_differential_subhalo_vcirc_function(bins, range, rcut, logscale, normed, weights, density)
        
        if inverse:
            # reverse bin and values
            bin = edges[::-1]
            bin = bin[:-1]
            values = np.cumsum(values[::-1])
        else:
            bin = edges[1:]
            values = np.cumsum(values)
        
        return values, bin
    
    def get_differential_subhalo_luminosity_function(self, bins, range, band, rcut=False, normed=False, weights=None, density=False):
        magnitude = self.subhalosphotometry[:, band]
        i, = np.where(magnitude < 1.0e10)
        magnitude = magnitude[i]
        
        if rcut is not False:
            r2 = (self.subhalospos[i] * self.subhalospos[i]).sum(axis=1)
            j, = np.where(r2 < rcut * rcut)
            magnitude = magnitude[j]
        
        values, edges = histogram(magnitude, bins=bins, range=range, normed=normed, weights=weights, density=density)
        bin = 0.5 * (edges[1:] + edges[:-1])
        
        return values, bin, edges
    
    def get_cumulative_subhalo_luminosity_function(self, bins, range, band, rcut=False, normed=False, weights=None, density=False, inverse=False):
        values, bin, edges = self.get_differential_subhalo_luminosity_function(bins, range, band, rcut, normed, weights, density)
        
        if inverse:
            # reverse bin and values
            bin = edges[::-1]
            bin = bin[:-1]
            values = np.cumsum(values[::-1])
        else:
            bin = edges[1:]
            values = np.cumsum(values)
        
        return values, bin
    
    def get_sfr_vs_position(self):
        if self.groupid == None:
            print('Not valid to do that for all the box')
            return
        radii = sqrt((self.subhalospos[:, :] ** 2).sum(axis=1))
        i, = np.where(self.subhalossfr > 0.0)
        return self.subhalossfr[i], radii[i]
    
    def get_mgas_vs_position(self, mradflag=False):
        if self.groupid == None:
            print('Not valid to do that for all the box')
            return
        radii = sqrt((self.subhalospos[:, :] ** 2).sum(axis=1))
        if not mradflag:
            i, = np.where(self.subhalosmassestype[:, 0] > 0.0)
            return self.subhalosmassestype[i, 0], radii[i]
        else:
            i, = np.where(self.subhalosmradtype[:, 0] > 0.0)
            return self.subhalosmradtype[i, 0], radii[i]
    
    def get_subhaloswsf_properties(self, mradflag=False):
        if self.groupid == None:
            print('Not valid to do that for all the box')
            return
        radii = sqrt((self.subhalospos[:, :] ** 2).sum(axis=1))
        if not mradflag:
            i, = np.where(self.subhalossfr > 0.0)
            return self.subhalosmassestype[i, 0], self.subhalosmassestype[i, 4], radii[i]
        else:
            i, = np.where(self.subhalossfr > 0.0)
            return self.subhalosmradtype[i, 0], self.subhalosmradtype[i, 4], radii[i]
    
    def get_gasfrac_vs_position(self, mradflag=False):
        if self.groupid == None:
            print('Not valid to do that for all the box')
            return
        radii = sqrt((self.subhalospos[:, :] ** 2).sum(axis=1))
        if not mradflag:
            i, = np.where(self.subhalosmassestype[:, 4] > 0.0)
            return self.subhalosmassestype[i, 0] / (self.subhalosmassestype[i, 0] + self.subhalosmassestype[i, 4]), radii[i]
        else:
            i, = np.where(self.subhalosmradtype[:, 4] > 0.0)
            return self.subhalosmradtype[i, 0] / (self.subhalosmradtype[i, 0] + self.subhalosmradtype[i, 4]), radii[i]
    
    def get_mstar_vs_position(self, mradflag=False):
        if self.groupid == None:
            print('Not valid to do that for all the box')
            return
        radii = sqrt((self.subhalospos[:, :] ** 2).sum(axis=1))
        if not mradflag:
            i, = np.where((self.subhalosmassestype[:, 4] > 0.0) & (self.subhalosmassestype[:, 0] <= 0.0) & (self.subhalossfr[:] <= 0.0))
            j, = np.where((self.subhalosmassestype[:, 4] > 0.0) & (self.subhalosmassestype[:, 0] > 0.0) & (self.subhalossfr[:] <= 0.0))
            k, = np.where((self.subhalosmassestype[:, 4] > 0.0) & (self.subhalosmassestype[:, 0] > 0.0) & (self.subhalossfr[:] > 0.0))
            return self.subhalosmassestype[:, 4], radii[:], i, j, k
        else:
            i, = np.where((self.subhalosmradtype[:, 4] > 0.0) & (self.subhalosmradtype[:, 0] <= 0.0) & (self.subhalossfr[:] <= 0.0))
            j, = np.where((self.subhalosmradtype[:, 4] > 0.0) & (self.subhalosmradtype[:, 0] > 0.0) & (self.subhalossfr[:] <= 0.0))
            k, = np.where((self.subhalosmradtype[:, 4] > 0.0) & (self.subhalosmradtype[:, 0] > 0.0) & (self.subhalossfr[:] > 0.0))
            return self.subhalosmradtype[:, 4], radii[:], i, j, k
    
    def get_mstar_vs_vcirc(self):
        if self.groupid == None:
            print('Not valid to do that for all the box')
            return
        
        i, = np.where(self.subhalosmassestype[:, 4] > 0)
        return self.subhalosmassestype[i, 4], self.subhalosvcirc[i]
    
    def get_mstar_vs_photometry(self, band):
        if self.groupid == None:
            print('Not valid to do that for all the box')
            return
        
        i, = np.where(self.subhalosmassestype[:, 4] > 0)
        return self.subhalosmassestype[i, 4], self.subhalosphotometry[i, band]
    
    def get_vcirc_vs_photometry(self, band):
        if self.groupid == None:
            print('Not valid to do that for all the box')
            return
        
        i, = np.where(self.subhalosphotometry[:, band] < 1.0e30)
        return self.subhalosvcirc[i], self.subhalosphotometry[i, band]
    
    def get_stellar_masses_vs_sizes(self):
        if self.groupid == None:
            print('Not valid to do that for all the box')
            return
        
        i, = np.where(self.subhalosmassestype[:, 4] > 0)
        return self.subhalosmassestype[i, 4], self.subhalossize[i]
    
    def get_total_masses_vs_sizes(self):
        if self.groupid == None:
            print('Not valid to do that for all the box')
            return
        
        i, = np.where(self.subhalosmasses[:] > 0)
        return self.subhalosmasses[i], self.subhalossize[i]
    
    def get_differential_subhalo_stellar_mass_function(self, bins, range, rcut=False, logscale=False, normed=False, weights=None, density=False):
        
        if logscale:
            i, = np.where(self.subhalosmradtype[:, 4] > 0)
            masses = log10(self.subhalosmradtype[i, 4])
            if range:
                range = log10(range)
        else:
            masses = self.subhalosmradtype[:, 4]
        
        if rcut is not False:
            if logscale:
                r2 = (self.subhalospos[i] * self.subhalospos[i]).sum(axis=1)
            else:
                r2 = (self.subhalospos[:] * self.subhalospos[:]).sum(axis=1)
            
            j, = np.where(r2 < rcut * rcut)
            masses = masses[j]
        
        values, edges = np.histogram(masses, bins=bins, range=range, normed=normed, weights=weights, density=density)
        bin = 0.5 * (edges[1:] + edges[:-1])
        return values, bin, edges
    
    def get_cumulative_subhalo_stellar_mass_function(self, bins, range, rcut=False, logscale=False, normed=False, weights=None, density=False,
                                                     inverse=False):
        values, bin, edges = self.get_differential_subhalo_stellar_mass_function(bins, range, rcut, logscale, normed, weights, density)
        
        if inverse:
            # reverse bin and values
            bin = edges[::-1]
            bin = bin[:-1]
            values = np.cumsum(values[::-1])
        else:
            bin = edges[1:]
            values = np.cumsum(values)
        
        return values, bin
    
    def get_totalmass_vs_position(self):
        if self.groupid == None:
            print('Not valid to do that for all the box')
            return
        radii = sqrt((self.subhalospos[:, :] ** 2).sum(axis=1))
        i, = np.where(self.subhalosmasses[:] > 0)
        
        return self.subhalosmasses[i], radii[i]