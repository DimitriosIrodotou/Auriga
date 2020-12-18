import math

import numpy as np
from cosmological_factors import *
from numpy import *

pi = math.pi
hubbleparam = 0.67769999999999997
omega0 = 0.307
omegalambda = 0.69299999999999995
rho_crit_now = 1.359929e11 * (hubbleparam ** 2)  # Msun Mpc^-3


def get_hist(arr, norm):
    h = np.histogram(arr, bins='auto', normed=False)
    x = np.array([h[1][0]])
    y = np.array([0.0])
    for i in range(len(h[0])):
        xi = np.array([h[1][i]])
        yi = np.array([h[0][i]])
        x = np.concatenate((x, xi))
        y = np.concatenate((y, yi))

        xi = np.array([h[1][i + 1]])
        yi = np.array([h[0][i]])
        x = np.concatenate((x, xi))
        y = np.concatenate((y, yi))

    xi = np.array([h[1][-1]])
    yi = np.array([0.0])
    x = np.concatenate((x, xi))
    y = np.concatenate((y, yi))
    y *= 1.0 / norm
    return x, y


def return_redshifts(snaps):
    a_list, foo = np.genfromtxt('ExpansionList_128', unpack=True)

    redshifts_list = (1. / a_list) - 1.
    snap_list = np.arange(len(a_list))

    redshifts = np.arange(len(snaps), dtype=float)
    for i in range(len(snaps)):
        redshifts[i] = redshifts_list[snaps[i]]
    return redshifts


def return_a(snaps):
    a_list, foo = np.genfromtxt('ExpansionList_128', unpack=True)
    snap_list = np.arange(len(a_list))
    a = np.arange(len(snaps), dtype=float)
    for i in range(len(snaps)):
        a[i] = a_list[snaps[i]]
    return a


def return_lookbacktime_from_a(a_list):
    cosmo = CosmologicalFactors(my_h=hubbleparam, my_OmegaMatter=omega0, my_OmegaLambda=omegalambda)
    cosmo.SetLookbackTimeTable()
    lookbacktime_list = np.arange(len(a_list), dtype=float)
    for i in range(len(a_list)):
        lookbacktime_list[i] = cosmo.LookbackTime_a_in_Gyr(a_list[i])
    return lookbacktime_list


def return_lookbacktime(snaps):
    cosmo = CosmologicalFactors(my_h=hubbleparam, my_OmegaMatter=omega0, my_OmegaLambda=omegalambda)
    cosmo.SetLookbackTimeTable()

    a_list, foo = np.genfromtxt('ExpansionList_128', unpack=True)
    snap_list = np.arange(len(a_list))
    lookbacktime_list = np.arange(len(a_list), dtype=float)
    for i in range(len(a_list)):
        lookbacktime_list[i] = cosmo.LookbackTime_a_in_Gyr(a_list[i])

    lookbacktimes = np.arange(len(snaps), dtype=float)
    for i in range(len(snaps)):
        lookbacktimes[i] = lookbacktime_list[snaps[i]]
    return lookbacktimes


def lookup_tree(isub, path, treebase):
    tc_sbnr, tc_sgnr, tc_cnum, tc_tnum = np.genfromtxt(path, unpack=True)
    tc_isort = np.argsort(tc_sbnr)
    tc_sbnr = tc_sbnr[tc_isort]
    tc_sgnr = tc_sgnr[tc_isort]
    tc_cnum = tc_cnum[tc_isort]
    tc_tnum = tc_tnum[tc_isort]
    # print isub,tc_cnum[isub],tc_tnum[isub]
    t = load_tree(int(tc_cnum[isub]), int(tc_tnum[isub]), base=treebase, verbose=False)

    return t


def return_subfind_value(path, fields, snaps, subids):
    values = np.zeros(len(snaps), len(fields))
    for snap, subid, value in zip(snaps, subids, values):
        sf = load_subfind(snap, dir=path, hdf5=True, loadonly=fields)
        for field, field_value in zip(fields, value):
            field_value = sf.data[field][subid]

    return values


def return_infall_times(isub, t, t0, sf, cfirst_infall=True, cfinal_infall=False, cfirst_peri=False,
    cthreeR200_infall=False):
    if not cfirst_infall and not cfinal_infall and not cthreeR200_infall and not cfirst_peri:
        raise ValueError('no timescales requested')
        return

    output = []
    # t0 = load_tree(0,0,base=treebase,verbose=False)
    # t = lookup_tree(isub,path,treebase)
    snap_numbers0, redshifts0, subfind_indices0, tree_indices0, ff_tree_indices0, fof_indices0 = \
        t0.return_subhalo_first_progenitors(
        0)
    r200 = t0.data['frc2'][ff_tree_indices0] / (1.0 + redshifts0) / hubbleparam

    snap_numbers, redshifts, subfind_indices, tree_indices, ff_tree_indices, fof_indices = \
        t.return_subhalo_first_progenitors(
        isub, target_FOF=sf.data['sgnr'][isub])
    lookbacktimes = return_lookbacktime_from_a((redshifts + 1.0) ** (-1.0))

    ## Find Distance between host and subhalo over time and infall time
    distance = np.zeros_like(redshifts)
    withinhalo = np.zeros_like(snap_numbers, dtype=bool)
    within3R200 = np.zeros_like(snap_numbers, dtype=bool)

    for isnap, itree, red in zip(np.arange(len(snap_numbers)), tree_indices, redshifts):
        # Match current progenitor redshift to redshift in the main halo's tree
        isnap0 = where(snap_numbers0 == t.data['snum'][itree])

        # Sometimes the progenitor is tracked over more snapshots than the main halo.  
        # If this is the case, make a dummy distance and set withinhalo to False
        # Need not continue with this iteration of the loop
        if len(isnap0[0]) == 0:
            distance[isnap] = -99999999
            withinhalo[isnap] = False
            within3R200[isnap] = False
            continue

        # Calculate the subhalo progenitor posistion relative to the position of the main halo's progenitor
        subpos = t.data['spos'][itree] - t0.data['spos'][tree_indices0][isnap0][0]

        # Compute distance and save results in arrays
        d = (subpos[0] ** 2 + subpos[1] ** 2 + subpos[2] ** 2) ** 0.5
        distance[isnap] = d / (1.0 + red) / hubbleparam
        withinhalo[isnap] = (d / (1.0 + red) / hubbleparam < r200[isnap0])
        within3R200[isnap] = (d / (1.0 + red) / hubbleparam < 3.0 * r200[isnap0])

    if cfirst_peri:
        # Compute times of pericentric passages:
        dist_minima = argrelmin(distance, order=3)
        index = where(withinhalo[dist_minima] == True)

        if len(index[0]) > 0:
            first_peri = max(lookbacktimes[dist_minima[0][index]])
        else:
            first_peri = -1
        output.append(first_peri)

    if cfirst_infall or cfinal_infall:
        # Save infall time.  This is defined as the first time when the subhalo is within the main halo
        if len(where(lookbacktimes[withinhalo])[0]) > 0:
            t_infall_i_subhalo = max(lookbacktimes[withinhalo])
            notwithinhalo = np.logical_not(withinhalo)

            if notwithinhalo[0]:
                t_infall_f_subhalo = -1  # max(lookbacktimes[withinhalo])
            else:
                index = np.arange(len(withinhalo))
                t_infall_f_subhalo = lookbacktimes[max(min(index[notwithinhalo]) - 1, 0)]
        else:
            t_infall_i_subhalo = -1
            t_infall_f_subhalo = -1

        if cfirst_infall:
            output.append(t_infall_i_subhalo)
        if cfinal_infall:
            output.append(t_infall_f_subhalo)

    if cthreeR200_infall:
        t_3R200_i_subhalo = max(lookbacktimes[within3R200])
        output.append(t_3R200_i_subhalo)

    return np.array(output)


def return_Lorb_final(t, host_first_progenitors, first_progenitors, hubbleparam):
    snap_numbers0, redshifts0, subfind_indices0, tree_indices0, ff_tree_indices0, fof_indices0 = host_first_progenitors
    snap_numbers, redshifts, subfind_indices, tree_indices, ff_tree_indices, fof_indices = first_progenitors
    # r200 = t.data['frc2'][ff_tree_indices0]/(1.0+redshifts0)/hubbleparam #R200 over time of main halo; note that
    # the tree reader does not correct
    # for cosmological units

    spos_now = sf.data['spos'] - sf.data['fpos'][0]
    svel_now = sf.data['svel'] - sf.data['fvel'][0]

    r_now = (spos_now[:, 0] ** 2 + spos_now[:, 1] ** 2 + spos_now[:, 2] ** 2) ** 0.5
    v_now = (svel_now[:, 0] ** 2 + svel_now[:, 1] ** 2 + svel_now[:, 2] ** 2) ** 0.5

    return


def isnotcontaminated(isub, slowres, sf):
    radius = (sf.data['smas'][isub] * 1e10 / ((4. / 3.) * pi * 200. * rho_crit_now)) ** (1. / 3.)
    radius *= 3.0
    # compute particle posistion relative to the subhalo posisiton
    pos = slowres.data['pos'] - sf.data['spos'][isub]
    # compute distance from subhalo center
    rp = (pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2) ** 0.5

    index = where(rp < radius)
    if len(index[0] > 0):
        print('skipping contaminated halo', len(index[0]), sf.data['slty'][isub])
        return False
    else:
        return True


def return_youngest_star(isub, s_star, sf):
    particle_index = where(
        (s_star.data['subhalo'] == where(isub == where(sf.data['sgnr'] == sf.data['sgnr'][isub])[0])[0][0]) & (
            s_star.data['halo'] == sf.data['sgnr'][isub]))
    return np.max(s_star.data['age'][particle_index])
