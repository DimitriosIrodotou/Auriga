import os
import time
import glob
import galaxy
import movies
import profiles
import evolution
import projections
import combinations
import metallicities

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
        :param kwargs: load desired attributes.
        :return: s
        """
        print('Found the following arguments: ', kwargs)
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

        print("Found %d snapshots for %s" % (self.nsnaps, self.directory))

        # Store the names of the snapshots for each Auriga halo and analyse each one's output individually #
        snapids = []
        for snap in snaps:
            snapid = np.int32(snap.split("_")[-1])
            snapids.append(snapid)

        for snapid in np.sort(snapids):
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
        haloes = glob.glob("%shalo_" % self.directory)
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
        Mask haloes and read desired attributes.
        :param level: level of the run.
        :param redshift: redshift of each snapshot.
        :param kwargs: select desired attributes.
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


    def make_pdf(self):
        """
        Create a pdf with the desired plots from
        :return: None
        """
        redshift = 0.0
        # for redshift in np.linspace(0.0, 1.0, 11):
        pdf = PdfPages('/u/di43/Auriga/plots/Auriga-' + date + '.pdf')

        # Projections #
        # Stellar #
        # projections.stellar_light(pdf, self, redshift, read=False)
        # projections.stellar_density(pdf, self, redshift, read=False)
        # projections.stellar_light_fit(self, redshift, read=False)
        # projections.r_band_magnitude(self, redshift, read=False)
        # Gas #
        # projections.gas_density(pdf, self, redshift, read=False)
        # projections.gas_temperature(pdf, self, redshift, read=False)
        # projections.gas_metallicity(pdf, self, redshift, read=False)
        # projections.gas_slice(pdf, self, redshift, read=False)
        # projections.gas_temperature_edge_on(pdf, self, redshift, read=False)
        # Magnetic field #
        # projections.magnetic_field(pdf, self, redshift, read=False)
        # Dark matter #
        # projections.dark_matter_density(pdf, self, redshift, read=False)

        # Global galactic relations #
        # galaxy.circularity_distribution(pdf, self, redshift, read=False)
        # galaxy.tully_fisher(pdf, self, redshift, read=False)
        # galaxy.stellar_vs_halo_mass(pdf, self, redshift, read=False)
        # galaxy.gas_fraction_vs_magnitude(pdf, self, redshift, read=False)
        # galaxy.bar_strength_profile(pdf, self, redshift, read=False)
        # galaxy.stellar_surface_density_profiles(pdf, self, redshift, read=False)
        # galaxy.circular_velocity_curves(pdf, self, redshift, read=False)
        # galaxy.gas_temperature_vs_distance(pdf, self, redshift, read=False)
        # galaxy.decomposition_IT20(pdf, self, redshift, read=True)
        galaxy.velocity_dispersion_profiles(pdf, self, redshift, read=False)

        # Evolution #
        # Galaxy #
        # evolution.sfr(pdf, self, read=False)
        # evolution.bar_strength(pdf, self, read=False)
        # evolution.gas_temperature_regimes(pdf, self, read=False)
        # evolution.delta_sfr_regimes(pdf, self, region='outer', read=False)
        # evolution.sfr_stars_gas_regimes(pdf, self, region='outer', read=False)
        # AGN #
        # evolution.AGN_modes_distribution(date, self, read=False)
        # evolution.AGN_feedback_kernel(date, self, ds=False, read=False)
        # evolution.AGN_feedback_smoothed(pdf)
        # evolution.blackhole_masses(pdf, self, read=True)

        # Movies #
        # movies.gas_movie(self, read=True)

        # Combinations #
        # Projections #
        # combinations.stellar_light_combination(pdf, redshift)
        # combinations.stellar_density_combination(pdf, redshift)
        # combinations.gas_density_combination(pdf, redshift)
        # combinations.gas_temperature_combination(pdf, redshift)
        # combinations.gas_metallicity_combination(pdf, redshift)
        # combinations.magnetic_field_combination(pdf, redshift)
        # combinations.central_combination(pdf, self, redshift, read=False)
        # Global galactic relations #
        # combinations.circularity_distribution_combination(pdf)
        # combinations.tully_fisher_combination(pdf)
        # combinations.stellar_vs_halo_mass_combination(pdf)
        # combinations.gas_fraction_vs_magnitude_combination(pdf)
        # combinations.bar_strength_profile_combination(pdf)
        # combinations.stellar_surface_density_profiles_combination(pdf)
        # combinations.circular_velocity_curves_combination(pdf)
        # combinations.gas_temperature_vs_distance_combination(pdf)
        # combinations.decomposition_IT20_combination(pdf, self, redshift)
        # Evolution #
        # combinations.bar_strength_combination(pdf)
        # combinations.gas_temperature_regimes_combination(pdf)
        # combinations.AGN_modes_distribution_combination(date)

        pdf.close()
        # file_name = 'gm/'
        # file_name = '/rbm/Au-*'
        # file_name = 'AGNmdc-' + date + '.png'
        file_name = 'Auriga-' + date + '.pdf'
        os.system('scp -r ../plots/%s di43@gate.mpcdf.mpg.de:/afs/ipp-garching.mpg.de/home/d/di43/Auriga/plots/' % file_name)
        return None


# Set the path to the simulation data and the level of the run #
default_level = 4
b = AurigaPdf()
b.add_directory('/u/di43/Auriga/output/', default_level)
b.make_pdf()  # Generate the pdf.

# Print total time # funzax-wahrIc-miwwe4
print('–––––––––––––––––––––––––––––––––––––––––––––')
print('Finished main.py in %.4s s' % (time.time() - start_time))
