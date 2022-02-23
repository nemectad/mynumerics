import numpy as np
import os
import time
# import multiprocessing as mp
import shutil
import h5py
import sys
import units
import mynumerics as mn

import re
import glob

import warnings


import matplotlib.pyplot as plt

class figure_driver:
    def __init__(self):
        self.sf = []
        
        self.legend_args = []
        self.legend_kwargs = {}
        
        self.right_axis_legend_args = []
        self.right_axis_legend_kwargs = {}
        
        self.savefig_args = None
        self.savefig_kwargs = {}

        self.savefigs_args = None
        self.savefigs_kwargs = []
        
        self.show_fig = True
        
        self.xlim_args = []; self.xlim_kwargs = {}        
        self.ylim_args = []; self.ylim_kwargs = {}
        self.right_ylim_args = []; self.right_ylim_kwargs = {}
        self.invert_xaxis = False
        
        


class colorbar:
    def __init__(self):
        self.show = False
        self.show_contours = False
        self.kwargs = {}
        
class plotter:
    def __init__(self):
        self.method = plt.plot
        self.args = []
        self.kwargs = {}
        self.colorbar = colorbar()


def plot_preset(i):
    Nsf = len(i.sf)
    
    fig, ax = plt.subplots()
    
    add_legend = False
    
    right_axis_exists = False
    add_right_axis_legend = False
    
    for k1 in range(Nsf):
        # plot
        if (i.sf[k1].method is plt.plot):
            # choose left or right axis
            if hasattr(i.sf[k1], 'axis_location'):
              if i.sf[k1].axis_location == 'right':
                if not(right_axis_exists): ax_right = ax.twinx(); right_axis_exists = True
                
                ax_right.plot(*i.sf[k1].args,**i.sf[k1].kwargs)
                if ('label' in i.sf[k1].kwargs.keys()): add_right_axis_legend = True
                
            else:
                ax.plot(*i.sf[k1].args,**i.sf[k1].kwargs)
                if ('label' in i.sf[k1].kwargs.keys()): add_legend = True
                
        elif (i.sf[k1].method is plt.errorbar):
            # choose left or right axis
            if hasattr(i.sf[k1], 'axis_location'):
              if i.sf[k1].axis_location == 'right':
                if not(right_axis_exists): ax_right = ax.twinx(); right_axis_exists = True
                
                ax_right.errorbar(*i.sf[k1].args,**i.sf[k1].kwargs)
                if ('label' in i.sf[k1].kwargs.keys()): add_right_axis_legend = True
                
            else:
                ax.errorbar(*i.sf[k1].args,**i.sf[k1].kwargs)
                if ('label' in i.sf[k1].kwargs.keys()): add_legend = True
            
        elif (i.sf[k1].method is plt.pcolor):
            map1 = ax.pcolor(*i.sf[k1].args,**i.sf[k1].kwargs)
            if i.sf[k1].colorbar.show:
                colorbar = fig.colorbar(mappable=map1, **i.sf[k1].colorbar.kwargs)
                
        elif (i.sf[k1].method is plt.contour): # if colorbar modified, has to be applied after pcolor
            map2 = ax.contour(*i.sf[k1].args,**i.sf[k1].kwargs)
            if i.sf[k1].colorbar.show_contours:
                try:
                    colorbar.add_lines(map2)
                except:
                    raise(ValueError('cannot add contours to colorbar'))
            
            # if hasattr(i.sf[k1], 'colorbar'):
            #     fig.colorbar(map1)
        elif (i.sf[k1].method is plt.hist):
            ax.hist(*i.sf[k1].args,**i.sf[k1].kwargs)
            
        elif (i.sf[k1].method is None):
            pass
                
        else:
            raise(NotImplementedError())

    if ((len(i.legend_args) > 0) or (len(i.legend_kwargs) > 0)): add_legend = True
    
    if add_legend:
        ax.legend(*i.legend_args, **i.legend_kwargs)
        
    if add_right_axis_legend:
        ax_right.legend(*i.right_axis_legend_args, **i.right_axis_legend_kwargs)
        
        
    if ((len(i.xlim_args) > 0) or (len(i.xlim_kwargs) > 0)):
        ax.set_xlim(*i.xlim_args, **i.xlim_kwargs)
        
    if i.invert_xaxis: ax.invert_xaxis()
        
        
    if ((len(i.ylim_args) > 0) or (len(i.ylim_kwargs) > 0)):
        ax.set_ylim(*i.ylim_args, **i.ylim_kwargs)
        
    if (((len(i.right_ylim_args) > 0) or (len(i.right_ylim_kwargs) > 0)) and right_axis_exists):
        ax_right.set_ylim(*i.right_ylim_args, **i.right_ylim_kwargs)
        
        
    if hasattr(i, 'xlabel'):
        ax.set_xlabel(i.xlabel)
    if hasattr(i, 'right_ylabel'):
        if not(right_axis_exists): ax_right = ax.twinx(); right_axis_exists = True
        ax_right.set_ylabel(i.right_ylabel)
    if hasattr(i, 'ylabel'):   
        ax.set_ylabel(i.ylabel)
    if hasattr(i, 'title'):   
        ax.set_title(i.title)
        
        
    if not(i.savefig_args is None):
        fig.savefig(*i.savefig_args, **i.savefig_kwargs)
        
    if not(i.savefigs_args is None):
        for k1 in range(len(i.savefigs_args)):
            fig.savefig(*i.savefigs_args[k1], **i.savefigs_kwargs[k1])
    
    if i.show_fig:
        plt.show()
        
    return fig
            
        