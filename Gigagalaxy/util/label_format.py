import math

from matplotlib import rcParams
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.ticker import is_close_to_int
from matplotlib.ticker import nearest_long


# inherit from LogFormatterMathtext all the other properties
class UserLogFormatterMathtext(LogFormatterMathtext):
    """
    Format values for log axis; using ``exponent = log_base(value)``
    """
    
    def __call__(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        b = self._base
        usetex = rcParams['text.usetex']
        
        # only label the decades
        if x == 0:
            if usetex:
                return '$0$'
            else:
                return '$\mathdefault{0}$'
        
        fx = math.log(abs(x)) / math.log(b)
        is_decade = is_close_to_int(fx)
        
        sign_string = '-' if x < 0 else ''
        
        # use string formatting of the base if it is not an integer
        if b % 1 == 0.0:
            base = '%d' % b
        else:
            base = '%s' % b
        
        if not is_decade and self.labelOnlyBase:
            return ''
        elif not is_decade:
            if usetex:
                if abs(x) > 10 or abs(x) < 0.1:
                    return (r'$%s%s^{%.2f}$') % (sign_string, base, fx)
                else:
                    return (r'$%s%g$') % (sign_string, abs(x))
            else:
                if abs(x) > 10 or abs(x) < 0.1:
                    return ('$\mathdefault{%s%s^{%.2f}}$') % (sign_string, base, fx)
                else:
                    return ('$\mathdefault{%s%g}$') % (sign_string, abs(x))
        else:
            if usetex:
                if abs(x) > 10 or abs(x) < 0.1:
                    return (r'$%s%s^{%d}$') % (sign_string, base, nearest_long(fx))
                else:
                    return (r'$%s%g$') % (sign_string, abs(x))
            else:
                if abs(x) > 10 or abs(x) < 0.1:
                    return (r'$\mathdefault{%s%s^{%d}}$') % (sign_string, base, nearest_long(fx))
                else:
                    return (r'$\mathdefault{%s%g}$') % (sign_string, abs(x))