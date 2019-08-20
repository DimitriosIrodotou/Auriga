import numpy as np


def search_snapshot_num(time, expansion_fact_list, rounddown=False, gyr=False):
    start = 0
    end = len(expansion_fact_list)
    search_point = np.int_((end + start) / 2)
    search_len = end - start
    
    expansion_fact_list[end - 1] = 1.  # quick fix for almost redshift zero value
    
    if not gyr:
        if time > expansion_fact_list[end - 1] or time < expansion_fact_list[0]:
            errmsg = 'Error: expansion factor must be between %g and %g' % (expansion_fact_list[end - 1], expansion_fact_list[0])
            errmsg += 'The current value is %d' % (time)
            raise ValueError(errmsg)
    
    while search_len > 1:
        val = expansion_fact_list[np.int_(search_point)]
        if (val > time):
            end = search_point
        else:
            start = search_point
        
        search_point = (end + start) / 2
        search_len = end - start
    
    if time > expansion_fact_list[np.int_(search_point)]:
        if abs(time - expansion_fact_list[np.int_(search_point)]) < abs(time - expansion_fact_list[np.int_(search_point) + 1]) and not rounddown:
            index = search_point
        else:
            index = search_point + 1
    
    else:
        if abs(time - expansion_fact_list[np.int_(search_point)]) < abs(time - expansion_fact_list[np.int_(search_point) - 1]) or rounddown:
            index = search_point
        else:
            index = search_point - 1
    
    return index


# return the closest snapshot to the selected redshift
def select_snapshot_number(outputlistfile, redshift, exp_fact=False, rounddown=False, verbose=False):
    linecount = 0
    skip = 0
    i = 0
    
    if type(redshift) != list:
        raise TypeError("redshift is not of type list")
    
    if (len(redshift) > 1):
        snaplist = []
    
    f = open(outputlistfile, 'r')
    
    # compute line number
    for line in f:
        if line.strip():
            linecount += 1
    
    if verbose:
        print('>>>Total number of data lines read from expansion list file --->', linecount)
    
    f.seek(skip, 0)
    
    expansion_fact = np.zeros(linecount)
    
    for line in f:
        if line.strip():
            # linelist = str.split(line)
            linelist = line.split()
            expansion_fact[i] = float(linelist[0])
            i += 1
    
    f.close()
    
    # redshift --> expansion factor
    if exp_fact:
        alist = redshift
    else:
        alist = [1.0 / (1.0 + i) for i in redshift]
    
    for t in alist:
        # look for the snapshot number
        index = search_snapshot_num(t, expansion_fact, rounddown=rounddown)
        if (len(alist) > 1):
            snaplist.append(index)
        else:
            snaplist = index
        
        if verbose:
            print('>>>Sought redshift', 1.0 / t - 1.0, '  actual redshift', 1.0 / expansion_fact[index] - 1.0)
    
    return snaplist


def select_snapshot_number_gyr(outputlistfile, time, verbose=False):
    linecount = 0
    skip = 0
    i = 0
    
    if (len(time) > 1):
        snaplist = []
    
    f = open(outputlistfile, 'r')
    
    # compute line number
    for line in f:
        if line.strip():
            linecount += 1
    
    if verbose:
        print('>>>Total number of data lines read from expansion list file --->', linecount)
    
    f.seek(skip, 0)
    
    tgyr = np.zeros(linecount)
    
    for line in f:
        if line.strip():
            # linelist = str.split(line)
            linelist = line.split()
            tgyr[i] = float(linelist[0])
            i += 1
    
    f.close()
    
    for t in time:
        # look for the snapshot number
        dumind = search_snapshot_num(t, tgyr[::-1], gyr=True)
        index = abs(dumind - len(tgyr) + 1)
        if (len(time) > 1):
            snaplist.append(index)
        else:
            snaplist = index
        
        if verbose:
            print('>>>Sought time', t, '  actual time', tgyr[index])
    
    return snaplist


# return the greatest snapshot number n closest to the selected redshift
# that is the maximum index n such that z(n) < z
def select_snapshot_number_low_bound(outputlistfile, redshift, exp_fact=False):
    linecount = 0
    skip = 0
    i = 0
    
    if type(redshift) != list:
        raise TypeError("redshift is not of type list")
    
    if (len(redshift) > 1):
        raise ValueError('Error: only a redishift value is allowed')
    
    f = open(outputlistfile, 'r')
    
    # compute line number
    for line in f:
        if line.strip():
            linecount += 1
    
    print('Total number of data lines ', linecount)
    
    f.seek(skip, 0)
    
    expansion_fact = np.zeros(linecount)
    
    for line in f:
        if line.strip():
            # linelist = str.split(line)
            linelist = line.split()
            expansion_fact[i] = float(linelist[0])
            i += 1
    
    f.close()
    
    # redshift --> expansion factor
    if exp_fact:
        alist = redshift
    else:
        alist = 1.0 / (1.0 + redshift)
    
    # look for the snapshot number
    index = search_snapshot_num(t, expansion_fact)
    if expansion_fact[index] < alist and index < len(expansion_fact) - 1:
        index += 1
    
    return index


# return the smallest snapshot number n closest to the selected redshift
# that is the minimum index n such that z(n) > z
def select_snapshot_number_high_bound(outputlistfile, redshift, exp_fact=False):
    linecount = 0
    skip = 0
    i = 0
    
    if type(redshift) != list:
        raise TypeError("redshift is not of type list")
    
    if (len(redshift) > 1):
        raise ValueError('Error: only a redishift value is allowed')
    
    f = open(outputlistfile, 'r')
    
    # compute line number
    for line in f:
        if line.strip():
            linecount += 1
    
    print('Total number of data lines ', linecount)
    
    f.seek(skip, 0)
    
    expansion_fact = np.zeros(linecount)
    
    for line in f:
        if line.strip():
            # linelist = str.split(line)
            linelist = line.split()
            expansion_fact[i] = float(linelist[0])
            i += 1
    
    f.close()
    
    # redshift --> expansion factor
    if exp_fact:
        alist = redshift
    else:
        alist = 1.0 / (1.0 + redshift)
    
    # look for the snapshot number
    index = search_snapshot_num(t, expansion_fact)
    if expansion_fact[index] > alist and index > 0:
        index -= 1
    
    return index


# the function returns a list of snaphosts in file2 (which has more entries) corresponding
# the the list of snapshots in input taken from file1
def match_expansion_factor_files(inputfile1, inputfile2, snaplist):
    skip = 0
    i = 0
    k = 0
    
    if type(snaplist) != list:
        raise TypeError("snaplist is not of type list")
    
    if (len(snaplist) > 1):
        result = []
    
    linecount1 = 0
    file1 = open(inputfile1, 'r')
    # compute line number
    for line in file1:
        if line.strip():
            linecount1 += 1
    
    linecount2 = 0
    file2 = open(inputfile2, 'r')
    # compute line number
    for line in file2:
        if line.strip():
            linecount2 += 1
    
    print('Total number of data lines in file %s ---> %d' % (inputfile1, linecount1))
    print('Total number of data lines in file %s ---> %d' % (inputfile2, linecount2))
    
    if linecount1 > linecount2:
        raise ValueError('second file must have more entries than the first one')
    
    file1.seek(skip, 0)
    file2.seek(skip, 0)
    
    expansion_fact1 = np.zeros(linecount1)
    expansion_fact2 = np.zeros(linecount2)
    
    for line in file1:
        if line.strip():
            # linelist = str.split(line)
            linelist = line.split()
            expansion_fact1[i] = float(linelist[0])
            
            i += 1
    
    for line in file2:
        if line.strip():
            # linelist = str.split(line)
            linelist = line.split()
            expansion_fact2[k] = float(linelist[0])
            
            k += 1
    
    file1.close()
    file2.close()
    
    for snap in snaplist:
        t = expansion_fact1[snap]
        # look for the snapshot number
        index = search_snapshot_num(t, expansion_fact2)
        if (len(snaplist) > 1):
            result.append(index)
        else:
            result = index
    
    return result