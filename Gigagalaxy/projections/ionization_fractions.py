import numpy as np

roman_numerals = {'I':   1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13,
                  'XIV': 14, 'XV': 15}


def get_cie_ionization_fractions(element, ion, temperatures):
    index = roman_numerals[ion]
    # read the Bryans+ (2006) table
    table_name = './data/%s-balance.txt' % (element)
    table = np.genfromtxt(table_name, comments='#')
    k = np.maximum(np.digitize(temperatures, table[:, 0]) - 1, 0)
    ionfrac = table[k, index] + (table[k + 1, index] - table[k, index]) / (table[k + 1, 0] - table[k, 0]) * (temperatures - table[k, 0])
    
    return -ionfrac