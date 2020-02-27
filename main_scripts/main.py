from __future__ import print_function

import os
import time
import glob
import galaxy
import profiles
import evolution
import projections
import metallicities
import time_evolution

import numpy as np

from gadget import gadget_readsnap
from gadget_subfind import load_subfind
from matplotlib.backends.backend_pdf import PdfPages

start_time = time.time()
date = time.strftime("%d_%m_%y_%H%M")


def get_names_sorted(names):
    """
    Sort Auriga haloes based on their names.
    :return: names_sorted
    """
    # Find the number (0-30) in each Auriga halo's name and sort them based on that #
    if list(names)[0].find("_"):
        names_sorted = np.array(list(names))
        names_sorted.sort()
        
        return names_sorted
    
    else:
        values = np.zeros(len(names))
        for i in range(len(names)):
            name = names[i]
            value = 0
            while not name[0].isdigit():
                value = value * 256 + ord(name[0])
                name = name[1:]
            values[i] = value * 256 + np.int32(name)
        isort = values.argsort()
        
        return np.array(names)[isort]


class AurigaSnapshot:
    """
    Perform third-level (snapshot) analysis of the Auriga output.
    """
    
    
    def __init__(self, snapid, snappath):
        """
        Initialise the attributes of the class.
        :param snapid: snapshot number from AurigaHalo.
        :param snappath: output path from AurigaHalo.
        """
        self.snapid = snapid
        self.snappath = snappath
        
        # Read the redshift and time of a snapshot for an Auriga halo.
        s = gadget_readsnap(self.snapid, snappath=self.snappath, onlyHeader=True)
        self.redshift = s.redshift
        self.time = s.time
        
        del s
    
    
    def loadsnap(self, **kwargs):
        """
        Read snapshot and subfind files for an Auriga halo.
        :param kwargs: load desired properties.
        :return: s
        """
        sf = load_subfind(self.snapid, dir=self.snappath + '/')
        s = gadget_readsnap(self.snapid, snappath=self.snappath, lazy_load=True, subfind=sf, **kwargs)
        s.subfind = sf
        
        return s


class AurigaHalo:
    """
    Perform second-level (halo) analysis of the Auriga output.
    """
    
    
    def __init__(self, directory):
        """
        Initialise the attributes of the class.
        :param directory: halo path from AurigaOutput.
        """
        self.snaps = {}
        self.directory = directory
        
        # Find how many snapshots an Auriga halo has #
        snaps = glob.glob("%s/output/snap*" % self.directory)
        snaps.sort()
        self.nsnaps = len(snaps)
        
        print("Found %d snapshots for halo %s" % (self.nsnaps, self.directory))
        
        # Store the names of the snapshots for each Auriga halo and analyse each one's output individually #
        for snap in snaps:
            snapid = np.int32(snap.split("_")[-1])
            self.snaps[snapid] = AurigaSnapshot(snapid, os.path.dirname(snap))
    
    
    def get_redshifts(self):
        """
        Turn snapshot numbers into redshifts.
        :return: redshifts
        """
        redshifts = np.zeros(self.nsnaps)
        for idx, (snapid, snap) in enumerate(self.snaps.items()):
            redshifts[idx] = snap.redshift
        return redshifts
    
    
    def get_snap_redshift(self, redshift):
        """
        Collect snapshot numbers and redshifts for an Auriga halo.
        :param redshift: redshift from get_snapshots.
        :return: self.snaps[(np.abs(redshifts - redshift)).argmin()]
        """
        redshifts = self.get_redshifts()
        
        return self.snaps[(np.abs(redshifts - redshift)).argmin()]


class AurigaOutput:
    """
    Perform first-level (directory) analysis of the Auriga output.
    """
    
    
    def __init__(self, directory, level):
        """
        Initialise the attributes of the class.
        :param directory: path from AurigaPdf.add_directory
        :param level: level of the run.
        """
        self.haloes = {}
        self.level = level
        self.directory = directory
        
        # Find how many Auriga haloes will be used #
        haloes = glob.glob("%s/halo_06L3" % self.directory)
        self.nhalos = len(haloes)
        
        print("Found %d halo(es)" % self.nhalos)
        
        # Store the names of the Auriga haloes and analyse each one's output individually #
        for halo in haloes:
            if not os.path.exists("%s/output" % halo):
                continue
            name = halo.split("_")[-1]
            self.haloes[name] = AurigaHalo(halo)
    
    
    def get_snapshots(self, redshift):
        """
        Collect snapshot(s)'s information for an Auriga halo.
        :param redshift: redshift from select_haloes
        :return: snaps
        """
        snaps = []
        names = get_names_sorted(self.haloes.keys())
        
        for name in names:
            snap = self.haloes[name].get_snap_redshift(redshift)
            snap.haloname = name
            snaps += [snap]
        
        print('Analysing snapdir_' + str(snap.__getattribute__('snapid')) + ' with redshift ' + str(snap.__getattribute__('redshift')))
        return snaps


class AurigaPdf:
    """
    Create a pdf file containing various plots for multiple Auriga haloes.
    """
    
    
    def __init__(self):
        """
        Initialise the attributes of the class.
        """
        self.directories = []
        
        return
    
    
    def add_directory(self, path, level):
        """
        Add a directory to the pdf.
        :param path: path to save the pdf.
        :param level: level of the run.
        :return: None
        """
        self.directories += [AurigaOutput(path, level)]
        
        return None
    
    
    def get_haloes(self, level):
        """
        Create a dictionary with the haloes.
        :param level: level of the run.
        :return: haloes
        """
        haloes = {}
        for d in self.directories:
            if d.level == level:
                for name, halo in d.haloes.items():
                    haloes[name] = halo
        
        return haloes
    
    
    def select_haloes(self, level, redshift, **kwargs):
        """
        Mask haloes and read desired properties.
        :param level: level of the run.
        :param redshift: redshift of each snapshot.
        :param kwargs: select desired properties.
        :return: None
        """
        self.selected_index = 0
        self.selected_snaps = []
        self.selected_arguments = kwargs
        
        for d in self.directories:
            if d.level == level:
                self.selected_snaps += d.get_snapshots(redshift)
        
        self.selected_current_snapshot = None
        self.selected_current_nsnaps = len(self.selected_snaps)
        
        return None
    
    
    def __iter__(self):
        """
        Return a new iterator object that can iterate over all the objects in the container.
        :return: self
        """
        return self
    
    
    def __next__(self):
        """
        Return the next item from the container.
        :return: self.selected_current_snapshot
        """
        if self.selected_current_snapshot is not None:
            del self.selected_current_snapshot
        try:
            snap = self.selected_snaps[self.selected_index]
            self.selected_current_snapshot = snap.loadsnap(**self.selected_arguments)
            self.selected_current_snapshot.haloname = snap.haloname
        except IndexError:
            raise StopIteration
        self.selected_index += 1
        return self.selected_current_snapshot
    
    
    def make_pdf(self, level):
        """
        Create a pdf with the desired plots from
        :param level: level of the run
        :return: None
        """
        redshift = 0.0
        pdf = PdfPages('/u/di43/Auriga/plots/Auriga-' + date + '.pdf')
        
        # TODO remove set_axes and level from projections and astype and add read
        # Projections #
        projections.multiple(pdf, self, redshift, read=False)
        # Stars #
        # projections.stellar_light(pdf, self, level, redshift)
        # projections.stellar_density(pdf, self, redshift, read=False)
        # Gas #
        # projections.gas_density(pdf, self, redshift, read=False)
        # projections.gas_slice(pdf, self, redshift, read=False)
        # projections.gas_metallicity(pdf, self, level, redshift)
        # projections.gas_temperature(pdf, self, redshift, read=False)
        # projections.gas_temperature_edge_on(pdf, self, redshift, read=False)
        # Magnetic fields #
        # projections.bfld(pdf, self, level, redshift)
        # Dark matter #
        # projections.dm_mass(pdf, self, level, redshift)
        
        # Profiles #
        # profiles.radial_profiles(pdf, self, level, redshift)
        # profiles.vertical_profiles(pdf, self, level, redshift)
        # profiles.stellar_profiles(pdf, self, level, redshift)
        
        # TODO fix the rest scripts
        # Time evolution #
        # for redshift in np.linspace(0.0, 2, 21):
        #     print(redshift)
        #     evolution.circularity(pdf, self, [level], redshift)
        # time_evolution.bar_strength_evolution(pdf, self, read=False)
        # main.time_evolution.bfld(pdf, self, level)
        # time_evolution.bh_mass(pdf, self, [level])
        # time_evolution.gas_temperature_fraction_evolution(pdf, self, read=False)
        #  AGN modes #
        # time_evolution.AGN_modes_cumulative(date, self, read=False)
        # time_evolution.AGN_modes_histogram(date, self, read=False)
        # time_evolution.AGN_modes_distribution(date, self, read=False)
        # time_evolution.AGN_modes_step(date, self, read=False)
        # time_evolution.AGN_modes_gas(date)
        
        # Global galactic relations #
        # galaxy.sfr(pdf, self, [level])
        # galaxy.delta_sfr(pdf, self, [level])
        # galaxy.gas_temperature_fraction(pdf, self, level, read=False)
        # for redshift in np.linspace(0.0, 1.0, 11):
        #     print(redshift)
        #     galaxy.circularity(pdf, self, [level], redshift)
        # galaxy.tully_fisher(pdf, self, [level])
        # galaxy.stellar_vs_total(pdf, self, [level])
        # galaxy.gas_fraction(pdf, self, [level])
        # main.galaxy.central_bfld(pdf, self, [level])
        # galaxy.bar_strength(pdf, self, read=True)
        # galaxy.stellar_surface_density_decomposition(pdf, self, redshift)
        # galaxy.circular_velocity_curves(pdf, self, redshift)
        # galaxy.gas_temperature_histogram(pdf, self, redshift, read=False)
        # galaxy.gas_distance_temperature(date, pdf, self, redshift, read=False)
        
        # Metallicities #
        # metallicities.ratios(pdf, self, [level], 0.)
        
        pdf.close()
        file_name = 'Auriga-' + date + '.pdf'
        # file_name = 'AGNmd-' + date + '.png'
        os.system('scp -r ../plots/%s di43@gate.mpcdf.mpg.de:/afs/ipp-garching.mpg.de/home/d/di43/Auriga/plots/' % file_name)
        return None


# Set the path to the simulation data and the level of the run #
b = AurigaPdf()
b.add_directory('/u/di43/Auriga/output/', 4)
b.make_pdf(4)  # Generate the pdf.

# Print total time # hewwoc-dItnub-8fejza
print('–––––––––––––––––––––––––––––––––––––––––––––')
print('Finished main.py in %.4s s' % (time.time() - start_time))