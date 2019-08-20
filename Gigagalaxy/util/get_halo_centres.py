from const import *
from gadget import *
from gadget_subfind import *
from pylab import *


def get_bound_DMparticles(snap, path, loadonlytype):
    s = gadget_readsnap(snap, snappath=path + '/output/', loadonlytype=loadonlytype, loadonly=['pos', 'vel', 'mass', 'id'], hdf5=True,
                        forcesingleprec=True)
    sf = load_subfind(snap, dir=path + '/output/', hdf5=True, loadonly=['fpos', 'slty', 'frc2', 'svel', 'sidm', 'smty', 'spos', 'fnsh', 'flty'],
                      forcesingleprec=True)
    s.select_halo(sf, remove_bulk_vel=False, use_principal_axis=True, rotate_disk=False)
    # jj, = np.where( (s.type == 1) & (s.r() < 1.1) )
    # iddm = s.data['id'][jj].astype('int64')
    # sidmallnow = iddm[:20]
    # del s
    # del sf
    
    return list(s.get_most_bound_dm_particles())