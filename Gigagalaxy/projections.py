from projections import column_density_projections as colproj
from projections import projection_main as proj

dir = '/hits/universe/marinafo/GigaGalaxy/'
dir2 = '/hits/universe/marinafo/GigaGalaxy/level_4/'
dir3 = '/hits/universe/marinafo/GigaGalaxy/level_5/'
dir4 = '/hits/universe/marinafo/GigaGalaxy/level_3/'

infile1 = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_short'
infile2 = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_64'

# z = [0.0]
# snap = select_snapshot_number.select_snapshot_number( infile2, z )

# sim = 9
# rot = stellar_projections.get_rotation_vector(dir3, sim, snap, '')
# stellar_projections.project_stars(dir3, sim, rot, '', infile1, infile2, '', loadonlytype=[4])

# sim = 9
# z = [0.0]
# snap = select_snapshot_number.select_snapshot_number( infile1, z )
# rot = stellar_projections.get_rotation_vector(dir, sim, snap, '')
# stellar_projections.project_stars(dir, sim, rot, '.old', infile1, infile1, '', loadonlytype=[4])

# sim = 9
# z = [0.0]
# snap = select_snapshot_number.select_snapshot_number( infile1, z )
# rot = stellar_projections.get_rotation_vector(dir2, sim, snap, '')
# stellar_projections.project_stars(dir2, sim, rot, '.lev4old', infile1, infile1, '', loadonlytype=[4])

# sim = 28
# z = [0.0]
# snap = select_snapshot_number.select_snapshot_number( infile2, z )
snap = 45
# rot = stellar_projections.get_rotation_vector(dir4, sim, snap, '', loadonlytype=[0,1,4])
# print rot
# stellar_projections.project_stars(dir4, sim, rot, '', infile1, infile2, '', loadonlytype=[4], numthreads=4)

run = ['halo_28']
dir = [dir2]
functions = [colproj.plot_gasdensity, colproj.plot_hydrogen, colproj.plot_oxygen]
proj.plot_gas_projections(dir[0], run[0], snap, 'pdf', functions, "column-dens", numthreads=4)

# functions = [colproj.plot_NV, colproj.plot_OVI, colproj.plot_OVII]
# proj.plot_gas_projections( dir[0], run[0], snap, 'pdf', functions, "column-highions", numthreads=4 )

# functions = [colproj.plot_SiIII, colproj.plot_SiIV, colproj.plot_CIV]
# proj.plot_gas_projections( dir[0], run[0], snap, 'pdf', functions, "column-lowions", numthreads=4 )

# functions = [tempproj.plot_coldgas, tempproj.plot_warmgas, tempproj.plot_hotgas]
# proj.plot_gas_projections( dir[0], run[0], snap, 'pdf', functions, "column-temperature", numthreads=4 )
daxes = [[1, 0], [1,
                  2]]  # functions = [metproj.plot_gasdensity, metproj.plot_temperature, metproj.plot_metallicity]  # proj.plot_gas_projections(
# dir[0], run[0], snap, '', 'pdf', functions, "column-temperature", daxes, numthreads=4 )