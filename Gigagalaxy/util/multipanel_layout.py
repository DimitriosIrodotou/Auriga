import matplotlib as mpl

mpl.use('Agg')
import pylab as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import LogLocator
import matplotlib.patheffects as path_effects


class multipanel_layout:
    def __init__(self, nrows, ncols, npanels=None, twinxaxis=False, twinyaxis=False, plotdim=16.1, scale=1.0, hspace=0.05, wspace=0.05, left=0.12,
                 right=0.9, bottom=0.18, top=0.96, aspect_ratio=True, aspect_fac=1., fontsize=8., dpi=300):
        
        self.fig = None
        self.nrows = nrows
        self.ncols = ncols
        self.npanels = npanels
        self.xlim = []
        self.ylim = []
        self.hspace = hspace
        self.wspace = wspace
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.aspect_ratio = aspect_ratio
        self.aspect_fac = aspect_fac
        self.fontsize = fontsize
        self.plotdim = plotdim
        self.dpi = dpi
        self.scale = scale
        self.toinch = 0.393700787
        self.twinxaxis = twinxaxis
        self.twinyaxis = twinyaxis
        self.axes = []
        
        if self.nrows == 1 and self.ncols == 1:
            self.top -= 0.1
            self.left += 0.1
        elif (self.nrows * self.ncols) <= 4:
            self.bottom += (0.1 * self.ncols / 4.)
            self.left += (0.1 * self.nrows / 4)
        
        if self.twinxaxis:
            self.twinxaxes = []
            self.x2lim = []
            self.top = 1.0 - self.bottom
            self.right = 1.0 - self.left
        
        if self.twinyaxis:
            self.twinyaxes = []
            self.y2lim = []
            self.top = 1.0 - self.bottom
            self.right = 1.0 - self.left
        
        if self.nrows <= 0 or self.ncols <= 0:
            raise TypeError('please specify a number of rows or columns greater than zero')
    
    def set_figure_layout(self):
        if self.scale <= 0.0:
            raise TypeError('scale must be a positive number.')
        
        # the default value of plotdim is calibrated for a 2x4 plot.
        # In other words each panel occupies only 1/4 this dimension
        self.plotdim *= (0.25 * self.scale)
        if self.ncols == 1 and self.nrows == 1:
            self.fig = plt.figure(
                figsize=np.array([self.plotdim * self.ncols + 0.5 * self.plotdim, self.plotdim * self.nrows + 0.5 * self.plotdim]) * self.toinch,
                dpi=self.dpi)
        else:
            self.fig = plt.figure(figsize=np.array([self.plotdim * self.ncols, self.plotdim * self.nrows]) * self.toinch, dpi=self.dpi)
        
        if not self.npanels:
            self.npanels = self.nrows * self.ncols
        
        if self.npanels > self.nrows * self.ncols:
            raise TypeError('npanels is greater than nrows * ncols. This is not allowed')
        
        for n in range(self.npanels):
            self.fig.add_subplot(self.nrows, self.ncols, n + 1)
        
        # Fine-tune figure; make subplots close to each other and hide x ticks for
        # all but bottom plot.
        self.fig.subplots_adjust(hspace=self.hspace)
        self.fig.subplots_adjust(wspace=self.wspace)
        self.fig.subplots_adjust(left=self.left)
        self.fig.subplots_adjust(right=self.right)
        self.fig.subplots_adjust(bottom=self.bottom)
        self.fig.subplots_adjust(top=self.top)
        
        self.axes = [axis for axis in self.fig.axes]
        
        if self.twinxaxis:
            self.twinxaxes = [axis.twiny() for axis in self.fig.axes]
        
        if self.twinyaxis:
            self.twinyaxes = [axis.twinx() for axis in self.fig.axes]
        
        if self.nrows > 1:
            notfilledpanels = (self.nrows * self.ncols) - self.npanels
            plt.setp([axis.get_xticklabels() for axis in self.axes[:(self.nrows - 1) * self.ncols - notfilledpanels]], visible=False)
            if self.twinxaxis:
                if self.npanels > self.ncols:
                    plt.setp([axis.get_xticklabels() for axis in self.twinxaxes[self.ncols:]], visible=False)
        
        plt.setp([axis.get_yticklabels() for axis in self.axes], visible=False)
        plt.setp([axis.get_yticklabels() for axis in self.axes[::self.ncols]], visible=True)
        
        if self.twinyaxis:
            plt.setp([axis.get_yticklabels() for axis in self.twinyaxes], visible=False)
            plt.setp([axis.get_yticklabels() for axis in self.twinyaxes[self.ncols - 1::self.ncols]], visible=True)
            plt.setp([axis.get_yticklabels() for axis in self.twinyaxes[self.npanels - 1:]], visible=True)
    
    def set_axis_locators(self, xmajloc=None, xminloc=None, ymajloc=None, yminloc=None, x2majloc=None, x2minloc=None, y2majloc=None, y2minloc=None,
                          logxaxis=False, logyaxis=False, logx2axis=None, logy2axis=None):
        if logx2axis == None:
            logx2axis = logxaxis
        if logy2axis == None:
            logy2axis = logyaxis
        
        for axis in self.axes:
            if not logxaxis:
                if xmajloc != None:
                    majorLocator = MultipleLocator(xmajloc)
                    axis.xaxis.set_major_locator(majorLocator)
                if xminloc != None:
                    minorLocator = MultipleLocator(xminloc)
                    axis.xaxis.set_minor_locator(minorLocator)
            else:
                if xmajloc != None:
                    majorLocator = LogLocator(numdecs=xmajloc)
                    axis.xaxis.set_major_locator(majorLocator)
            
            if not logyaxis:
                if ymajloc != None:
                    majorLocator = MultipleLocator(ymajloc)
                    axis.yaxis.set_major_locator(majorLocator)
                if yminloc != None:
                    minorLocator = MultipleLocator(yminloc)
                    axis.yaxis.set_minor_locator(minorLocator)
            else:
                if ymajloc != None:
                    majorLocator = LogLocator(numdecs=ymajloc)
                    axis.yaxis.set_major_locator(majorLocator)
        
        if self.twinxaxis:
            if x2majloc == 'copy':
                x2majloc = xmajloc
            if x2minloc == 'copy':
                x2minloc = xminloc
            
            for axis in self.twinxaxes:
                if not logx2axis:
                    if x2majloc != None:
                        majorLocator = MultipleLocator(x2majloc)
                        axis.xaxis.set_major_locator(majorLocator)
                    if x2minloc != None:
                        minorLocator = MultipleLocator(x2minloc)
                        axis.xaxis.set_minor_locator(minorLocator)
                else:
                    if x2majloc != None:
                        majorLocator = LogLocator(numdecs=x2majloc)
                        axis.xaxis.set_major_locator(majorLocator)
        
        if self.twinyaxis:
            if y2majloc == 'copy':
                y2majloc = ymajloc
            if y2minloc == 'copy':
                y2minloc = yminloc
            
            for axis in self.twinyaxes:
                if not logy2axis:
                    if y2majloc != None:
                        majorLocator = MultipleLocator(y2majloc)
                        axis.yaxis.set_major_locator(majorLocator)
                    if y2minloc != None:
                        minorLocator = MultipleLocator(y2minloc)
                        axis.yaxis.set_minor_locator(minorLocator)
                else:
                    if y2majloc != None:
                        majorLocator = LogLocator(numdecs=y2majloc)
                        axis.yaxis.set_major_locator(majorLocator)
    
    def set_axis_limits_and_aspect(self, xlim, ylim, logaxis=False, logxaxis=False, logyaxis=False, x2lim=None, y2lim=None):
        for axis in self.axes:
            if xlim:
                self.xlim = xlim
                axis.set_xlim(self.xlim)
            if ylim:
                self.ylim = ylim
                axis.set_ylim(self.ylim)
            
            if self.xlim and self.ylim and self.aspect_ratio:
                # square figures
                x0, x1 = axis.get_xlim()
                y0, y1 = axis.get_ylim()
                if (x1 - x0) < 0.:
                    tmp = x0
                    x0 = x1
                    x1 = tmp
                if logaxis:
                    axis.set_aspect(self.aspect_fac * np.log10(x1 / x0) / np.log10(y1 / y0))
                elif logxaxis:
                    axis.set_aspect(self.aspect_fac * np.log10(x1 / x0) / (y1 - y0))
                elif logyaxis:
                    axis.set_aspect(self.aspect_fac * (x1 - x0) / np.log10(y1 / y0))
                else:
                    axis.set_aspect(self.aspect_fac * (x1 - x0) / (y1 - y0))
        if self.twinxaxis:
            if not x2lim:
                self.x2lim = self.xlim
            for axis in self.twinxaxes:
                axis.set_xlim(self.x2lim)
        
        if self.twinyaxis:
            if not y2lim:
                self.y2lim = self.ylim
            for axis in self.twinyaxes:
                axis.set_ylim(self.y2lim)
    
    def set_fontsize(self, fontsize=None):
        if fontsize:
            self.fontsize = fontsize
        
        for axis in self.axes:
            for label in axis.xaxis.get_ticklabels():
                label.set_fontsize(self.fontsize)
            for label in axis.yaxis.get_ticklabels():
                label.set_fontsize(self.fontsize)
            for tick in axis.xaxis.get_major_ticks():
                tick.label.set_fontsize(self.fontsize)
            for tick in axis.yaxis.get_major_ticks():
                tick.label.set_fontsize(self.fontsize)
        
        if self.twinxaxis:
            for axis in self.twinxaxes:
                for label in axis.xaxis.get_ticklabels():
                    label.set_fontsize(self.fontsize)
                for label in axis.yaxis.get_ticklabels():
                    label.set_fontsize(self.fontsize)
        
        if self.twinyaxis:
            for axis in self.twinyaxes:
                for label in axis.xaxis.get_ticklabels():
                    label.set_fontsize(self.fontsize)
                for label in axis.yaxis.get_ticklabels():
                    label.set_fontsize(self.fontsize)
    
    def set_axis_labels(self, xlabel, ylabel, x2label=None, y2label=None):
        if ylabel != None:
            for axis in self.axes[::self.ncols]:
                axis.set_ylabel(ylabel, fontsize=self.fontsize)
            if y2label != None:
                for axis in self.twinyaxes:
                    axis.set_ylabel("", fontsize=self.fontsize)
                self.twinyaxes[self.npanels - 1].set_ylabel(y2label, fontsize=self.fontsize)
                for axis in self.twinyaxes[self.ncols - 1::self.ncols]:
                    axis.set_ylabel(y2label, fontsize=self.fontsize)
        
        if xlabel != None:
            if self.nrows > 1:
                notfilledpanels = (self.nrows * self.ncols) - self.npanels
                for axis in self.axes[(self.nrows - 1) * self.ncols - notfilledpanels:]:
                    axis.set_xlabel(xlabel, fontsize=self.fontsize)
                if x2label != None:
                    for axis in self.twinxaxes:
                        axis.set_xlabel("", fontsize=self.fontsize)
                    if self.npanels > self.ncols:
                        for axis in self.twinxaxes[:self.ncols]:
                            axis.set_xlabel(x2label, fontsize=self.fontsize)
                    else:
                        for axis in self.twinxaxes[:self.npanels]:
                            axis.set_xlabel(x2label, fontsize=self.fontsize)
            else:
                for axis in self.axes:
                    axis.set_xlabel(xlabel, fontsize=self.fontsize)
                if x2label != None:
                    for axis in self.twinxaxes:
                        axis.set_xlabel(x2label, fontsize=self.fontsize)
    
    def reset_axis_limits(self):
        for axis in self.axes:
            axis.set_xlim(self.xlim)
            axis.set_ylim(self.ylim)
        
        if self.twinxaxis:
            for axis in self.twinxaxes:
                axis.set_xlim(self.x2lim)
        
        if self.twinyaxis:
            for axis in self.twinyaxes:
                axis.set_ylim(self.y2lim)
    
    def reset_ticks_visibility(self):
        if self.nrows > 1:
            notfilledpanels = (self.nrows * self.ncols) - self.npanels
            plt.setp([axis.get_xticklabels() for axis in self.axes[:(self.nrows - 1) * self.ncols - notfilledpanels]], visible=False)
            if self.twinxaxis:
                if self.npanels > self.ncols:
                    plt.setp([axis.get_xticklabels() for axis in self.twinxaxes[self.ncols:]], visible=False)
        # I don't want duplicated x-axis
        plt.setp([axis.get_xticklabels() for axis in self.twinxaxes[self.ncols:]], visible=False)
        
        plt.setp([axis.get_yticklabels() for axis in self.axes], visible=False)
        plt.setp([axis.get_yticklabels() for axis in self.axes[::self.ncols]], visible=True)
        
        if self.twinyaxis:
            plt.setp([axis.get_yticklabels() for axis in self.twinyaxes], visible=False)
            plt.setp([axis.get_yticklabels() for axis in self.twinyaxes[self.ncols - 1::self.ncols]], visible=True)
            plt.setp([axis.get_yticklabels() for axis in self.twinyaxes[self.npanels - 1:]], visible=True)
    
    def set_panel_title(self, panel, title, position='top right', fontsize=7.5, color='black', fontweight='normal', pef=False):
        if panel >= self.npanels:
            raise TypeError('panel is greater than npanels. This is not allowed')
        
        axis = self.axes[panel]
        
        if position == 'top left':
            text = axis.text(0.06, 0.94, title, color=color, fontsize=fontsize, fontweight=fontweight, horizontalalignment='left',
                             verticalalignment='top', transform=axis.transAxes)
        elif position == 'top right':
            text = axis.text(0.94, 0.94, title, color=color, fontsize=fontsize, fontweight=fontweight, horizontalalignment='right',
                             verticalalignment='top', transform=axis.transAxes)
        elif position == 'bottom left':
            text = axis.text(0.06, 0.06, title, color=color, fontsize=fontsize, fontweight=fontweight, horizontalalignment='left',
                             verticalalignment='bottom', transform=axis.transAxes)
        elif position == 'bottom right':
            text = axis.text(0.94, 0.06, title, color=color, fontsize=fontsize, fontweight=fontweight, horizontalalignment='right',
                             verticalalignment='bottom', transform=axis.transAxes)
        elif position == 'above middle':
            text = axis.text(0.5 * (self.right + self.left), 1.02, title, color=color, fontsize=fontsize, fontweight=fontweight,
                             horizontalalignment='left', verticalalignment='bottom', transform=axis.transAxes)
        else:
            raise TypeError('invalid title position')
        
        if pef == True:
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])
    
    def set_panel_number(self, npanels):
        if self.scale <= 0.0:
            raise TypeError('scale must be a positive number.')
        
        # the default value of plotdim is calibrated for a 2x4 plot.
        # In other words each panel occupies only 1/4 this dimension
        self.plotdim *= (0.25 * self.scale)
        self.fig = plt.figure(figsize=np.array([self.plotdim * self.ncols, self.plotdim * self.nrows]) * self.toinch, dpi=self.dpi)
        self.npanels = npanels
        
        if not self.npanels:
            self.npanels = self.nrows * self.ncols
        
        if self.npanels > self.nrows * self.ncols:
            raise TypeError('npanels is greater than nrows * ncols. This is not allowed')
        
        self.fig.clear()
        
        # Fine-tune figure; make subplots close to each other and hide x ticks for
        # all but bottom plot.
        self.fig.subplots_adjust(hspace=self.hspace)
        self.fig.subplots_adjust(wspace=self.wspace)
        self.fig.subplots_adjust(left=self.left)
        self.fig.subplots_adjust(right=self.right)
        self.fig.subplots_adjust(bottom=self.bottom)
        self.fig.subplots_adjust(top=self.top)
        
        self.axes = [axis for axis in self.fig.axes]
        
        if self.twinxaxis:
            self.twinxaxes = [axis.twiny() for axis in self.fig.axes]
        
        if self.twinyaxis:
            self.twinyaxes = [axis.twinx() for axis in self.fig.axes]
        
        if self.nrows > 1:
            notfilledpanels = (self.nrows * self.ncols) - self.npanels
            plt.setp([axis.get_xticklabels() for axis in self.axes[:(self.nrows - 1) * self.ncols - notfilledpanels]], visible=False)
            if self.twinxaxis:
                if self.npanels > self.ncols:
                    plt.setp([axis.get_xticklabels() for axis in self.twinxaxes[self.ncols:]], visible=False)
        
        plt.setp([axis.get_yticklabels() for axis in self.axes], visible=False)
        plt.setp([axis.get_yticklabels() for axis in self.axes[::self.ncols]], visible=True)
        
        if self.twinyaxis:
            plt.setp([axis.get_yticklabels() for axis in self.twinyaxes], visible=False)
            plt.setp([axis.get_yticklabels() for axis in self.twinyaxes[self.ncols - 1::self.ncols]], visible=True)
            plt.setp([axis.get_yticklabels() for axis in self.twinyaxes[self.npanels - 1:]], visible=True)
    
    def set_colorbar(self, vval, cblabel, cticks, cmap=None, bounds=False, fontsize=5, labelsize=5, orientation='vertical'):
        
        if cmap == None:
            cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=vval[0], vmax=vval[1])
        
        if orientation == 'vertical':
            cbleft = self.right
            cbbottom = self.bottom
            width = 0.02
            height = self.top - cbbottom
        elif orientation == 'horizontal':
            cbleft = self.left
            cbbottom = self.top
            height = 0.02
            width = self.right - cbleft
        
        axb = self.fig.add_axes([cbleft, cbbottom, width, height])
        
        if bounds == False:
            cb = mpl.colorbar.ColorbarBase(axb, cmap=cmap, norm=norm, ticks=cticks, orientation=orientation)
        elif bounds == True:
            bound = cticks
            cb = mpl.colorbar.ColorbarBase(axb, cmap=cmap, norm=norm, boundaries=bound, ticks=cticks, orientation=orientation)
        
        if orientation == 'vertical':
            cb.set_label(cblabel, fontsize=fontsize, labelpad=2.)
            axb.yaxis.set_tick_params(labelsize=labelsize)
            axb.xaxis.set_label_position('top')
        
        if orientation == 'horizontal':
            cb.set_label(cblabel, fontsize=fontsize, labelpad=2.)
            axb.xaxis.set_tick_params(labelsize=labelsize)
            axb.xaxis.set_ticks_position('top')
            axb.xaxis.set_label_position('top')
    
    def get_scalar_mappable_for_colorbar(self, parameters, cmap):
        norm = mpl.colors.Normalize(vmin=np.min(parameters), vmax=np.max(parameters))
        c_m = cmap  # mpl.cm.jet
        s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        
        return s_m