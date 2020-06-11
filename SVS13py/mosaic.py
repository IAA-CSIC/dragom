import numpy as np
from numpy.linalg import eig, inv

from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.colors import to_hex

from astropy.wcs.utils import pixel_to_skycoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.modeling import models, fitting

from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model, EllipseSample, EllipseFitter, Isophote
from photutils import EllipticalAperture

from scipy.optimize import curve_fit

import warnings

import shelve

import copy

import pandas as pd

from itertools import product

from SVS13py.SVS13py import show_slice_cube, mosaic_slices, calc_moment, collapse_chans, fit_ellipse, create_cube
from SVS13py.ellipse_fitter import EllipseFitter
from SVS13py.ellfit_results import create_table, import_table, plot_fit_results, plot_arc_map
import SVS13py.mf as mf

#fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13_nocont.fits')
#fits_path = fits.open(mf.default_params['path_fits'])
#fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13.fits')
hdu = fits.open(mf.default_params['path_fits'])[0]
hdr = hdu.header
wcs = WCS(hdu.header).celestial
#hdulist['BLENDED'].header, naxis=2
image_data = hdu.data[0]


#vmax=0.064 vmin=0.0036 vcenter=0.0172
box = [[556,786],[855,1164]]
mosaic_slices(image_data, nrow=4, ncol=4, ngrid=8, chan_0=1036, chan_f=1477, wcs=wcs, box=box,
              output_name='slices_divnorm_5width_mean_1',
              header=hdr,
              add_beam=True, beam_nax=12,
              operation='mom0', width_chan=5,
              vmax=0.057060,
              vcenter=0.0161027,
              vmin=0.00357,
              norm='divnorm',
              cmap='jet',
              box_fontsize=35,
              font_size=35,
              markerstar_color='w', markerstar_width=3, markerstar_size=20,
              add_scalebar=True, scalebar_nax=12, scalebar_fontsize=35, scalebar_width=2.5, scalebar_distance=200,
              scalebar_units='au', scalebar_pad=0.5,
              output_format='pdf')
