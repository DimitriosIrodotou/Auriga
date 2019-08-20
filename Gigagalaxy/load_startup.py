#!/usr/bin/env python

'''avoid (necessarily) using ipython config to load arepo-snap-util'''

# usage: from load_startup import *

import sys

import matplotlib
import numpy
import scipy

#####################################################################
# for information only

print('python:', sys.version)
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
print('matplotlib:', matplotlib.__version__, matplotlib.get_backend())

#####################################################################


path = '/cosma/home/nhpb14/giga_galaxy/analysis/standard/arepo-snap-util/'

sys.path.insert(1, path)

from startup import *  # arepo-snap-util