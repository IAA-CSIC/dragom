import numpy as np
from numpy.linalg import eig, inv

from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.font_manager as fm
from matplotlib.colors import to_hex
import matplotlib.image as mpimg
from matplotlib.patches import Arc, ConnectionPatch
from matplotlib.gridspec import GridSpec

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector, mark_inset

from IPython.display import display, clear_output

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.modeling import models, fitting

from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model, EllipseSample, EllipseFitter, Isophote
from photutils import EllipticalAperture

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import warnings

import shelve

import copy

import pandas as pd

from itertools import product

import casa_proc

import telegram_send

import gc

from SVS13py.SVS13py import show_slice_cube, mosaic_slices, calc_moment, collapse_chans, fit_ellipse, create_cube, casa_immoments, casa_impv, open_fits, casa_exportfits, mosaic_images, plot_pvlines
from SVS13py.ellipse_fitter import EllipseFitter
from SVS13py.ellfit_results import create_table, import_table, plot_fit_results, plot_arc_map, dyn_time, SMA_dist, radial_flux_matrix, radial_flux, angle_velocity_diagram
import SVS13py.mf as mf

# fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13_nocont.fits')
#fits_path = get_pkg_data_filename(mf.default_params['path_fits'])
#fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13.fits')
#hdu = fits.open(fits_path)[0]
hdu = fits.open(mf.default_params['path_fits'])[0]
hdr = hdu.header
wcs = WCS(hdu.header).celestial
#hdulist['BLENDED'].header, naxis=2
image_data = hdu.data[0]



# fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13_nocont.fits')
#fits_path = get_pkg_data_filename(mf.default_params['path_fits'])
#fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13.fits')
#hdu = fits.open(fits_path)[0]
lsr_path = '/home/gblazquez/data/spw-2-9-gb.contsub.lsr.fits'
hdu_lsr = fits.open(lsr_path)[0]
hdr_lsr = hdu_lsr.header
wcs_lsr = WCS(hdu_lsr.header).celestial
#hdulist['BLENDED'].header, naxis=2
image_data_lsr = hdu_lsr.data[0]
v0_lsr=88.7103
vf_lsr=-93.3177

# fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13_nocont.fits')
#fits_path = get_pkg_data_filename(mf.default_params['path_fits'])
#fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13.fits')
#hdu = fits.open(fits_path)[0]
lsr2_path = '/home/gblazquez/data/spw-2-9-gb.contsub.hanning.lsr2.fits'
hdu_lsr2 = fits.open(lsr2_path)[0]
hdr_lsr2 = hdu_lsr2.header
wcs_lsr2 = WCS(hdu_lsr2.header).celestial
#hdulist['BLENDED'].header, naxis=2
image_data_lsr2 = hdu_lsr2.data[0]
v0_lrs2=88.7103
vf_lrs2=-93.2237


# fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13_nocont.fits')
#fits_path = get_pkg_data_filename(mf.default_params['path_fits'])
#fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13.fits')
#hdu = fits.open(fits_path)[0]
spw17_path = '/home/gblazquez/data/spw-17-gb.244kHz.3840chans.contsub.fits'
hdu_spw17 = fits.open(spw17_path)[0]
hdr_spw17 = hdu_spw17.header
wcs_spw17 = WCS(hdu_spw17.header).celestial
image_data_spw17 = hdu_spw17.data[0]
#v0_spw17=88.7103
#vf_spw17=-93.2237



mom0_paths = {'EHVjet':'spw-17-gb.224kHz.3840chans.contsub.image.chans2326~2448.EHVjet_allfield.mom0.fits',
            'EHV':'spw-17-gb.224kHz.3840chans.contsub.image.chans2326~2513.EHV_allfield.mom0.fits',
            'IHV':'spw-2-9-gb.contsub.lsr.image.chans198~316.IHV_allfield.mom0.fits',
            'LV':'spw-2-9-gb.contsub.lsr.image.chans168~198.LV_allfield.mom0.fits',
            'LV-IHV':'spw-2-9-gb.contsub.lsr.image.chans168~316.LV-IHV_allfield.mom0.fits',
            'SVred':'spw-2-9-gb.contsub.lsr.image.chans110~149.SVred_allfield.mom0.fits',
            'IHVred':'spw-2-9-gb.contsub.lsr.image.chans35~110.IHVred_allfield.mom0.fits',
            'allred':'spw-2-9-gb.contsub.lsr.image.chans35~149.allred_allfield.mom0.fits'}


mom1_paths = {'EHVjet':'spw-17-gb.224kHz.3840chans.contsub.image.chans2326~2448.EHVjet_allfield.mom1.fits',
            'EHV':'spw-17-gb.224kHz.3840chans.contsub.image.chans2326~2513.EHV_allfield.mom1.fits',
            'IHV':'spw-2-9-gb.contsub.lsr.image.chans198~316.IHV_allfield.mom1.fits',
            'LV':'spw-2-9-gb.contsub.lsr.image.chans168~198.LV_allfield.mom1.fits',
            'LV-IHV':'spw-2-9-gb.contsub.lsr.image.chans168~316.LV-IHV_allfield.includepix0.015.mom1.fits',
            'SVred':'spw-2-9-gb.contsub.lsr.image.chans110~149.SVred_allfield.mom1.fits',
            'IHVred':'spw-2-9-gb.contsub.lsr.image.chans35~110.IHVred_allfield.mom1.fits',
            'allred':'spw-2-9-gb.contsub.lsr.image.chans35~149.allred_allfield.mom1.fits'}

pv_paths = {'EHVjet':'spw-17-gb.244kHz.3840chans.contsub.3width.0-3839chans.pv.fits',
           'EHVjet2':'spw-17-gb.244kHz.3840chans.contsub.5width.0-3839chans.pv.fits',
           'LV_IHV':'spw-17-gb.244kHz.3840chans.contsub.7width.0-3839chans.pv.fits',
           'LV_IHV_lsr':'spw-2-9-gb.contsub.lsr.5width.0-344chans.pv.fits'}

mom0_hdus = {mom0:fits.open('/home/gblazquez/data/moments/{}'.format(mom0_paths[mom0]))[0] for mom0 in mom0_paths}
mom0_hdrs = {mom0:mom0_hdus[mom0].header for mom0 in mom0_paths}
mom0_wcss = {mom0:WCS(mom0_hdrs[mom0]).celestial for mom0 in mom0_paths}
mom0_datas = {mom0:mom0_hdus[mom0].data[0] for mom0 in mom0_paths}


mom1_hdus = {mom1:fits.open('/home/gblazquez/data/moments/{}'.format(mom1_paths[mom1]))[0] for mom1 in mom1_paths}
mom1_hdrs = {mom1:mom1_hdus[mom1].header for mom1 in mom1_paths}
mom1_wcss = {mom1:WCS(mom1_hdrs[mom1]).celestial for mom1 in mom1_paths}
mom1_datas = {mom1:mom1_hdus[mom1].data[0] for mom1 in mom1_paths}
for mom1 in mom1_datas:
#    mom1_datas[mom1][np.isnan(mom1_datas[mom1])] = 0
    mom1_datas[mom1] = np.abs(mom1_datas[mom1])

pv_hdus = {pv:fits.open('/home/gblazquez/data/pv_diagrams/{}'.format(pv_paths[pv]))[0] for pv in pv_paths}
pv_hdrs = {pv:pv_hdus[pv].header for pv in pv_paths}
pv_wcss = {pv:WCS(pv_hdrs[pv]).celestial for pv in pv_paths}
pv_datas ={pv:pv_hdus[pv].data[0] for pv in pv_paths}

EHVjet_PV = {'start':[52.26580297, 31.26763918],
             'end':[52.26591933, 31.26715587]}

EHVjet_PV2 = {'start':EHVjet_PV['end'],
             'end':[52.26698650, 31.26566336]}

#SkyCoord(ra='03h29m03.852s', dec='31d16m01.01s', frame='icrs')
#This are the coordinates of the end of the jet according to the ellipse fits
doff = 1.4
end_ra = 52.26605417 + doff*abs(mf.default_params['vla4b_deg'][0]-52.26605417)
end_dec = 31.26694722 - doff*abs(mf.default_params['vla4b_deg'][1]-31.26694722)
LV_IHV_PV = {'start':mf.default_params['vla4b_deg'],
            'end':[end_ra, end_dec]}



EHVjet_PV = {'start':[52.26580297, 31.26763918],
             'end':[52.26591933, 31.26715587]}

EHVjet_PV2 = {'start':EHVjet_PV['end'],
             'end':[52.26698650, 31.26566336]}

trans_length_pv = 3
length_pv = 1.8
dists = np.linspace(0,length_pv,25)
trans_dists = np.linspace(0,trans_length_pv,10)
jet_axis, trans_axis, ax_pvline = plot_pvlines(EHVjet_PV,
                                    hdr_spw17,
                                    dists=dists,
                                    trans_dists=trans_dists,
                                    show_plot=False)

trans_length_pv_2 = 5
length_pv_2 = 7
dists_2 = np.linspace(0,length_pv_2,25)
trans_dists_2 = np.linspace(0,trans_length_pv_2,10)
jet_axis_2, trans_axis_2, ax_pvline = plot_pvlines(EHVjet_PV2,
                                    hdr_spw17,
                                    dists=dists_2,
                                    trans_dists=trans_dists_2,
                                    show_plot=False)

chans = [1957,2513]
##chans = [2326,2513]
#n=0
#for xs,ys in zip(jet_axis['center_xs'], jet_axis['center_ys']):
#    telegram_send.send(messages=['PV diagram n={} started jet 1'.format(n)])
#    casa_impv(spw17_path,
#             start=[xs, ys],
#             end=trans_length_pv,
#             width=3,
#             mode='length',
#             pa=trans_axis['pa_PV_trans'],
#             chans=chans,
#             output_name='{}trans_jet1'.format(n))
#    telegram_send.send(messages=['PV diagram n={} finished jet 1!'.format(n)])
#    n+=1
#

n=3
for xs,ys in zip(jet_axis_2['center_xs'][3:], jet_axis_2['center_ys'][3:]):
    telegram_send.send(messages=['PV diagram n={} started jet 2'.format(n)])
    casa_impv(spw17_path,
             start=[xs, ys],
             end=trans_length_pv_2,
             width=3,
             mode='length',
             pa=trans_axis_2['pa_PV_trans'],
             chans=chans,
             output_name='{}trans_jet2'.format(n))
    gc.collect()
    telegram_send.send(messages=['PV diagram n={} finished jet 2!'.format(n)])
    n+=1






