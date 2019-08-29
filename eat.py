import numpy as np
from gadget import gadget_readsnap
from gadget_subfind import load_subfind
from loadmodules import *
from pylab import *


def eat_snap_and_fof(level, halo_number, snapnr, snappath, loadonlytype=[0, 4], haloid=0, galradfac=0.1, remove_bulk_vel=True, verbose=True):
    """
    Eat an Auriga snapshot, given a level/halo_number/snapnr. Subfind has been executed 'on-the-fly', during the simulation run.

    :param level: level of the Auriga simulation (3=high, 4='normal' or 5=low). Level 3/5 only for halo 6, 16 and 24. See Grand+ 2017 for
    details. Careful when level != 4 because directories may have different names.

    :param halo_number: which Auriga galaxy? See Grand+ 2017 for details. Should be an integer in range(1, 31)

    :param snapnr: which snapshot number? This is an integer, in most cases in range(1, 128) depending on the number of timesteps of the run.
    The last snapshot would then be 127. Snapshots are written at a certain time, but careful because the variable called time is actually the
    cosmological expansion factor a = 1/(1+z). For example, snapnr=127 has s.time = 1, which corresponds to a redshift of ~0. This makes sense
    because this is the last snapshot and the last snapshot is written at redshift zer

    :param snappath: full path to the level/halo directory that contains all of the simulation snapshots

    :param loadonlytype: which particle types should be loaded? This should be a list of integers. If I'm not mistaken, the options are:
    0 (gas), 1 (dark matter), 2 (unused), 3 (tracers), 4 (stars & wind; age > 0. --> stars; age < 0. --> wind), 5 (black holes).

    :param haloid: the ID of the SubFind halo. In case you are interested in the main galaxy in the simulation run: set haloid to zero. This
    was a bit confusing to me at first because a zoom-simulation run of one Auriga galaxy is also referred to as 'halo', see halo_number.

    :param galradfac: the radius of the galaxy is often used to make cuts in the (star) particles. It seems that in general galrad is set to
    10% of the virial radius R200 of the DM halo that the galaxy sits in. The disk does seem to 'end' at 0.1R200.

    :param remove_bulk_vel: boolean to subtract bulk velocity [default True]

    :param verbose: boolean to print some information

    :return: two-tuple (s, sf) where s is an instance of the gadget_snapshot class, and sf is an instance of the subfind class. See
    Arepo-snap-util,gadget_snap.py respectively gadget_subfind.py
    """

    # Eat the subfind friend of friends output
    print("Snapshot:", snapnr, "From:", snappath)
    sf = load_subfind(snapnr, dir=snappath)

    # Eat the Gadget snapshot
    s = gadget_readsnap(snapnr, snappath=snappath, lazy_load=True, subfind=sf, loadonlytype=loadonlytype)
    s.subfind = sf

    try:
        # Sets s.(sub)halo. This allows selecting the halo, e.g. 0 (main 'Galaxy')
        s.calc_sf_indizes(s.subfind)
        has_sf_indizes = True
    except KeyError as e:
        # for example, Au5-24: snapnr 0-3 has empty sf.data; snapnr 4-6 has no stars and no cold-gas spin
        print("WARNING: KeyError encountered in s.calc_sf_indizes")
        if str(e) == "'flty'":
            print("WARNING: KeyError arised because 'flty' was not found in sf.data")
            # print("  sf.data.keys(): {0}".format(sf.data.keys()))
            # print("   s.data.keys(): {0}".format(s.data.keys()))
            has_sf_indizes = False
        else:
            raise

    # For example, Au5-24; snapnr 7-14 breaks due to lack of stars

    s.galrad = None
    if has_sf_indizes:
        # Note that selecting the halo now rotates the disk using the principal axis. Rotate_disk is a general switch which has to be set to True
        # to rotate. To then actually do the rotation, do_rotation has to be True as well. Within rotate_disk there are three methods to handle the
        # rotation. Choose one of them, but see the select_halo method for details.
        try:
            s.select_halo(s.subfind, haloid=haloid, galradfac=galradfac, rotate_disk=True, use_principal_axis=True, euler_rotation=False,
                          use_cold_gas_spin=False, do_rotation=True, remove_bulk_vel=remove_bulk_vel, verbose=verbose)
        except KeyError as e:
            # For example, Au5-24: snapnr 0-3 has empty sf.data; snapnr 4-6 has no stars and no cold-gas spin
            print("WARNING: KeyError encountered in s.select_halo")
            print(str(e))
            if str(e) == "'svel'":
                print("WARNING: KeyError arised because 'svel' was not found in s.data")
            elif str(e) == "'pos'":
                print("WARNING: this particular snapshot has no positions.")
            else:
                raise
        except IndexError as e:
            # For example, Au5-24: snapnr 0-3 has empty sf.data; snapnr 4-6 has no stars and no cold-gas spin
            print("WARNING: IndexError encountered in s.select_halo")
            if str(e) == "index 0 is out of bounds for axis 0 with size 0":
                print("WARNING: IndexError arised possibly in get_principal_axis because there are no stars (yet)")
            else:
                raise

        # This means that galrad is 10 % of R200 (200*rho_crit definition)
        s.galrad = galradfac * sf.data['frc2'][haloid]  # frc2 = Group_R_Crit200

    # Age has different shape than other properties, so force age to be of same shape by shoving a bunch of zeros in-between.
    if "age" in s.data.keys():
        age = np.zeros(s.npartall)
        st = s.nparticlesall[:4].sum()
        en = st + s.nparticlesall[4]
        age[st:en] = s.age
        s.data['age'] = age
        del age

    if "gmet" in s.data.keys():
        # Clean negative and zero values of gmet to avoid RuntimeErrors
        s.gmet = np.maximum(s.gmet, 1e-40)

    # Sneak some more info into the s instance
    s.halo_number = halo_number
    s.level = level
    s.snapnr = snapnr
    s.haloid = haloid
    s.name = "Au{0}-{1}".format(s.level, s.halo_number)

    if verbose:
        print("\n{0}".format(s.name))
        print("galrad  : {0}".format(s.galrad))
        print("redshift: {0}".format(s.redshift))
        print("time    : {0}".format(s.time))
        print("center  : {0}\n".format(s.center))

    return s, sf