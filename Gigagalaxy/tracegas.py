import matplotlib

matplotlib.use('Agg')
from loadmodules import *

from tracegas import plot_tracer_sfphases

MHD = True
level = 4
L = True

rtype = 'MHD'  # DM, MHD, HD

if level == 4:
    #####  For all halos (30) #####
    # runs = ['halo_1','halo_2', 'halo_3','halo_4','halo_5', 'halo_6','halo_7','halo_8', 'halo_9', 'halo_10','halo_11','halo_12', 'halo_13',
    # 'halo_14', 'halo_15', 'halo_16', 'halo_17', 'halo_18', 'halo_19','halo_20', 'halo_21','halo_22', 'halo_23', 'halo_24', 'halo_25',
    # 'halo_26', 'halo_27','halo_28','halo_29','halo_30']
    
    runs = ['halo_L1', 'halo_L2', 'halo_L3', 'halo_L4', 'halo_L5', 'halo_L7', 'halo_L8', 'halo_L9', 'halo_L10']
    
    if rtype is 'MHD':
        if not L:
            dir5 = '/virgo/simulations/Auriga/level4_MHD/'
            snaplist = list(range(60, 128))
        else:
            dir5 = '/virgo/simulations/Auriga/level4_MHD_new/'
            targetgasmass = 3.90662e-6
            runs = ['halo_L10', 'halo_L7', 'halo_6mlf0p6', 'halo_L3', 'halo_L9', 'halo_L6', 'halo_L8', 'halo_L1', 'halo_L5', 'halo_L4']
            snaplist = list(range(51, 128))
        
        outputlistfile = ['/u/rgrand/explists/ExpansionList_128']
        outputlistgyr = ['/u/rgrand/explists/timegyrlist_128']
        outpath = '/u/rgrand/analysis/level4_MHD/'
    elif rtype is 'HD':
        dir5 = '/virgo/simulations/Auriga/level4/'
        outputlistfile = ['/u/rgrand/explists/ExpansionList_hrt']
        outputlistgyr = ['/u/rgrand/explists/timegyrlist_128']  # need to make hrt!
        outpath = '/u/rgrand/analysis/level4/'
    elif rtype is 'DM':
        dir5 = '/virgo/simulations/Auriga/level4_DM/'
        outputlistfile = ['/u/rgrand/explists/ExpansionList_128']
        outputlistgyr = ['/u/rgrand/explists/timegyrlist_128']  # need to make hrt!
        outpath = '/u/rgrand/analysis/level4_DM/'
    else:
        raise ValueError('inappropriate rtype value given. Stopping.')
    
    nruns = len(runs)
    dirs = [dir5] * nruns
    
    dirs[2] = '/virgo/simulations/Auriga/level4_MHD_variants/'
    
    outputlistfile *= nruns
    outputlistgyr *= nruns
    nrows, ncols = 1, nruns
    
    lastsnap = 127

elif level == 3:
    dir5 = '/virgo/simulations/Auriga/level3_MHD/'
    outputlistfile = ['/u/rgrand/explists/ExpansionList_64']
    outputlistgyr = ['/u/rgrand/explists/timegyrlist_64']
    outpath = '/u/rgrand/analysis/level3_MHD/'
    runs = ['halo_6', 'halo_16', 'halo_21', 'halo_23', 'halo_24', 'halo_27']
    dirs = [dir5] * len(runs)
    outputlistfile *= len(runs)
    outputlistgyr *= len(runs)
    nrows, ncols = 1, len(runs)
    nrows, ncols = 2, 3
    lastsnap = 63

suffix = 'pdf'

################ gastrace plots ########################################
# trace_allstellar_tracers.trace_allstellar_tracers(dirs, runs, snaplist[::-1], '', outpath, outputlistfile, suffix='', boxsize=0.05,
# loadonlytype=[0, 4, 6], rcut=0.005, numthreads=1, targetgasmass=targetgasmass, accreted=False, fhcen=False)

# plot_tracer_distribution_category.plot_tracer_distribution_category(dirs, runs, snaplist[::-1], '', outpath, outputlistfile, suffix='',
# targetgasmass=targetgasmass)

# plot_xyz_tracers.plot_xyz_tracers(dirs, runs, snaplist[::-1], '', outpath, suffix='', boxsize=0.05, loadonlytype=[0, 4, 6], rcut=0.005,
# numthreads=1, targetgasmass=targetgasmass, accreted=False, fhcen=False)

plot_tracer_sfphases.plot_tracer_sfphases(dirs, runs, snaplist[::-1], '', outpath, outputlistfile, suffix='', ploteachsim=True)