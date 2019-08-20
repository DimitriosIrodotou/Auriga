import os
import sys

import numpy as np


def check_whether_interactive():
    if not sys.stdout.isatty():
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)


def percentile(x, indata, q):
    sum = np.cumsum(indata)
    # normalize for percentile extraction
    if sum[-1] <= 0.0:
        return -99.0
    
    sum *= 100. / sum[-1]
    
    down = 0
    up = len(sum) - 1
    i = (up + down) / 2
    
    while True:
        if sum[i] > q:
            up = i
        else:
            down = i
        
        if up - down == 1:
            break
        
        i = (up + down) / 2
    
    res = x[down] + (q - sum[down]) / (sum[up] - sum[down]) * (x[up] - x[down])
    return res