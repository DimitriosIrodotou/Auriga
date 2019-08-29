from __future__ import print_function

import time

import numpy as np
from gadget import gadget_readsnap
from gadget_subfind import load_subfind
from matplotlib.backends.backend_pdf import PdfPages

import book
import book.galaxy
import book.metallicities
import book.profiles
import book.projections
import book.time_evolution

start_time = time.time()
date = time.strftime("%d\%m\%y\%H%M")


def get_names_sorted(names):
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


class auriga_snapshot:
    def __init__(self, snapid, snappath):
        self.snapid = snapid
        self.snappath = snappath

        s = gadget_readsnap(self.snapid, snappath=self.snappath, onlyHeader=True)
        self.redshift = s.redshift
        self.time = s.time
        del s


    def loadsnap(self, **kwargs):
        sf = load_subfind(self.snapid, dir=self.snappath + '/')
        s = gadget_readsnap(self.snapid, snappath=self.snappath, lazy_load=True, subfind=sf, **kwargs)
        s.subfind = sf
        return s


class auriga_halo():
    def __init__(self, directory):
        self.directory = directory
        self.snaps = {}

        import glob
        import os
        snaps = glob.glob("%s/output/snapdir_127" % self.directory)
        snaps.sort()
        self.nsnaps = len(snaps)

        for snap in snaps:
            snapid = np.int32(snap.split("_")[-1])
            self.snaps[snapid] = auriga_snapshot(snapid, os.path.dirname(snap))

        print("Found %d snapshots for halo %s." % (self.nsnaps, self.directory))


    def get_redshifts(self):
        redshifts = np.zeros(self.nsnaps)
        for idx, (snapid, snap) in enumerate(self.snaps.items()):
            redshifts[idx] = snap.redshift
        return redshifts


    def get_snap_redshift(self, redshift):
        redshifts = self.get_redshifts()
        return self.snaps[(np.abs(redshifts - redshift)).argmin()]


class auriga_directory():
    def __init__(self, directory, level):
        self.directory = directory
        self.halos = {}
        self.level = level

        import glob
        import os
        halos = glob.glob("%s/halo_6" % self.directory)
        self.nhalos = len(halos)

        for halo in halos:
            if not os.path.exists("%s/output" % halo):
                self.nhalos
                continue
            name = halo.split("_")[-1]
            self.halos[name] = auriga_halo(halo)


    def get_snapshots(self, redshift):
        snaps = []
        names = get_names_sorted(self.halos.keys())
        for name in names:
            snap = self.halos[name].get_snap_redshift(redshift)
            snap.haloname = name
            snaps += [snap]
        return snaps


class auriga_book():
    def __init__(self):
        self.directories = []
        return


    def add_directory(self, path, level):
        self.directories += [auriga_directory(path, level)]
        return


    def get_halos(self, level):
        halos = {}
        for d in self.directories:
            if d.level == level:
                for name, halo in d.halos.items():
                    halos[name] = halo
        return halos


    def select_halos(self, level, redshift, **kwargs):
        self.selected_arguments = kwargs
        self.selected_index = 0
        self.selected_snaps = []
        for d in self.directories:
            if d.level == level:
                self.selected_snaps += d.get_snapshots(redshift)
        self.selected_current_snapshot = None
        self.selected_current_nsnaps = len(self.selected_snaps)


    def __iter__(self):
        return self


    def __next__(self):
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


    def make_book_overview(self, level):
        pdf = PdfPages('/u/di43/Auriga/plots/auriga_book_lvl%d' % level + '-' + date + '.pdf')

        # overview table??
        # book.galaxy.table( pdf, self, [level] )

        for z in [0.]:
            book.projections.stellar_light(pdf, self, [level], z)
            book.projections.dm_mass(pdf, self, [level], z)
            book.projections.stellar_mass(pdf, self, [level], z)
            book.projections.gas_density(pdf, self, [level], z)
            book.projections.gas_temperature(pdf, self, [level], z)
            book.projections.gas_metallicity(pdf, self, [level], z)
            book.projections.bfld(pdf, self, [level], z)

        book.galaxy.phasediagram(pdf, self, [level])

        book.metallicities.ratios(pdf, self, [level], 0.)

        for z in [0.]:
            book.profiles.radial_profiles(pdf, self, [level], z)  # gas density, energies, velocities?, metallicity
            book.profiles.vertical_profiles(pdf, self, [level], z)

        book.time_evolution.sfr(pdf, self, [level])
        book.time_evolution.bfld(pdf, self, [level])  # early time exponential ( < z=2 ), full time linear
        book.time_evolution.galaxy_mass(pdf, self, [level])
        book.time_evolution.bh_mass(pdf, self, [level])

        # CPU time!

        book.galaxy.circularity(pdf, self, [level])
        book.galaxy.tullyfisher(pdf, self, [level])
        book.galaxy.stellarvstotal(pdf, self, [level])
        book.galaxy.gasfraction(pdf, self, [level])
        book.galaxy.centralbfld(pdf, self, [level])

        pdf.close()
        return


    def make_book_resolution(self, levels):
        return


b = auriga_book()
b.add_directory("/virgo/simulations/Auriga/level4_MHD/", 4)
b.make_book_overview(4)