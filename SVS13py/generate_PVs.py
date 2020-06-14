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

from SVS13py.SVS13py import show_slice_cube, mosaic_slices, calc_moment, collapse_chans, fit_ellipse, create_cube, casa_immoments, casa_impv, open_fits, casa_exportfits
from SVS13py.ellipse_fitter import EllipseFitter
from SVS13py.ellfit_results import create_table, import_table, plot_fit_results, plot_arc_map, dyn_time
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



hdr_hodapp = copy.copy(hdr_lsr)
hdr_hodapp['NAXIS'] = 4
hdr_hodapp['NAXIS1'] = 745
hdr_hodapp['NAXIS2'] = 992
hdr_hodapp['NAXIS3'] = 1


hdr_hodapp['CRVAL1'] = mf.default_params['vla4b_deg'][0]
hdr_hodapp['CDELT1'] = -1.2077294685990338e-06
hdr_hodapp['CRPIX1'] = 745-193

hdr_hodapp['CRVAL2'] = mf.default_params['vla4b_deg'][1]
hdr_hodapp['CDELT2'] = 1.2077294685990338e-06
hdr_hodapp['CRPIX2'] = 832

hdr_hodapp['CRVAL3'] = 1
hdr_hodapp['CDELT3'] = 1
hdr_hodapp['CRPIX3'] = 1

wcs_hodapp = WCS(hdr_hodapp).celestial

arc1 = Arc((444.76923076923083,530.2692307692307),
           width= 115.3846153846154*2,
           height= 115.3846153846154*2,
           angle=0.0,
           theta1=190,
           theta2=350,
           color='k',linewidth=4, linestyle='--')

arc2 = Arc((349,326.84615384615387),
           width= 140*2,
           height= 140*2,
           angle=-15,
           theta1=190,
           theta2=350,
           color='k',linewidth=4, linestyle='--')

arc3 = Arc((479,741.5769230769231),
           width= 52.78846153846155*2,
           height= 52.78846153846155*2,
           angle=-80,
           theta1=190,
           theta2=350,
           color='k',linewidth=4, linestyle='--')

arcs = {'1':arc1, '2':arc2, '3':arc3}

arcs_pix_hodapp = {arc:{} for arc in arcs}
for arc in arcs:
    arcs_pix_hodapp[arc]['x0'] = arcs[arc].center[0]
    arcs_pix_hodapp[arc]['y0'] = arcs[arc].center[1]
    arcs_pix_hodapp[arc]['width'] = arcs[arc].width
    arcs_pix_hodapp[arc]['height'] = arcs[arc].height
    arcs_pix_hodapp[arc]['angle'] = arcs[arc].angle
    arcs_pix_hodapp[arc]['theta1'] = arcs[arc].theta1
    arcs_pix_hodapp[arc]['theta2'] = arcs[arc].theta2

arcs_sky_hodapp = mf.arcs2skycoord(arcs_pix_hodapp, hdr_hodapp)
arcs_pix_lsr = mf.arcs2pix(arcs_sky_hodapp, hdr_lsr)
arcs_pix_spw17 = mf.arcs2pix(arcs_sky_hodapp, hdr_spw17)


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
#            'LV':'spw-2-9-gb.contsub.lsr.image.chans168~198.LV_allfield.mom1.fits',
            'LV-IHV':'spw-2-9-gb.contsub.lsr.image.chans168~316.LV-IHV_allfield.includepix0.015.mom1.fits',
            'SVred':'spw-2-9-gb.contsub.lsr.image.chans110~149.SVred_allfield.mom1.fits',
            'IHVred':'spw-2-9-gb.contsub.lsr.image.chans35~110.IHVred_allfield.mom1.fits',
            'allred':'spw-2-9-gb.contsub.lsr.image.chans35~149.allred_allfield.mom1.fits'}

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



#matplotlib.rcParams.update({'font.size': 5})

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

box_bigpic = [[350,550],[1100,1500]]
box_moms_EHV = [[400,560],[855,1164]]
#box_moms_IHV = [[580,790],[875,1164]]
box_moms_LV_IHV = [[590,800],[895,1195]]
box_arcs = [[620,825],[850,1130]]
#box_arcs = [[140,125],[700,945]]

nrow = 1
ncol = 3
ngrid = 2
magical_factor = 23
wspace = 0
hspace = 0

fig = plt.figure(figsize=(nrow*magical_factor,(ncol+1)*ngrid,))
gs1 = GridSpec(nrow, (ncol+1)*ngrid, )
gs1.update(wspace=wspace, hspace=hspace,)

ax = {}
n = 0
for i,j in product(range(nrow), [i for i in range(ncol*ngrid)][::ngrid]):
    ax[n] = plt.subplot(gs1[i,j:j+ngrid], projection=wcs)
    n += 1

sigmas = {}
max_values = {}
contour_levels = {}

mom0_name = 'EHV'
sigmas[mom0_name] = np.std(mom0_datas[mom0_name][0:100,0:100])
max_values[mom0_name] = np.max(mom0_datas[mom0_name])
#contour_levels[mom0_name] = np.linspace(3,max_values[mom0_name]/sigmas[mom0_name],6) * sigmas[mom0_name]
contour_levels[mom0_name] = np.arange(3,44,6) * sigmas[mom0_name]
im_EHV, ax_EHV = show_slice_cube(mom0_datas[mom0_name],
                     channel=0,
                     box=box_bigpic,
                     wcs=mom0_wcss[mom0_name],
                     header=mom0_hdrs[mom0_name],
                     add_beam=True,
                     vmax=10,
                     vmin=0.00,
                     add_scalebar=True,
                     scalebar_loc='lower right',
                     scalebar_distance=200,
                     markerstar_color='k',
                     markerstar_width=3,
                     scalebar_units='au',
                     ax = ax[1],
                     contour_colors='b',
                     show_slice_return='contour_imax',
                     plot_vel=False,
                     render='contours',
                     contour_levels=contour_levels[mom0_name],
                     contour_label=mom0_name,
                     plot_cbar=False,
                     rotate_ticktext_yaxis=90,
                     icrs_ylabel='')


mom0_name = 'LV-IHV'
sigmas[mom0_name] = np.std(mom0_datas[mom0_name][0:100,0:100])
max_values[mom0_name] = np.max(mom0_datas[mom0_name])
#contour_levels[mom0_name] = np.linspace(3,max_values[mom0_name]/sigmas[mom0_name],4) * sigmas[mom0_name]
contour_levels[mom0_name] = np.arange(3, 17, 4) * sigmas[mom0_name]
im_LV_IHV, ax_LV_IHV = show_slice_cube(mom0_datas[mom0_name],
                     channel=0,
                     box=box_bigpic,
                     wcs=mom0_wcss[mom0_name],
                     header=mom0_hdrs[mom0_name],
                     add_beam=True,
                     vmax=1.5,
                     vmin=0.00,
                     add_scalebar=True,
                     scalebar_loc='lower right',
                     scalebar_distance=200,
                     scalebar_color='k',
                     markerstar_color='k',
                     markerstar_width=3,
                     scalebar_units='au',
                     ax=ax[1],
                     contour_colors='g',
                     contour_sigma_filter=1,
                     show_slice_return='contour_imax',
                     plot_vel=False,
                     render='contours',
                     contour_levels=contour_levels[mom0_name],
                     contour_label=mom0_name,
                     plot_cbar=False,
                     icrs_ylabel='')

mom0_name = 'SVred'
sigmas[mom0_name] = np.std(mom0_datas[mom0_name][0:100,0:100])
max_values[mom0_name] = np.max(mom0_datas[mom0_name])
#contour_levels[mom0_name] = np.arange(1.5,13,) * sigmas[mom0_name]
contour_levels[mom0_name] = np.arange(1.5, 13, 3) * sigmas[mom0_name]
im_allred, ax_allred = show_slice_cube(mom0_datas[mom0_name],
                     channel=0,
                     box=box_bigpic,
                     wcs=mom0_wcss[mom0_name],
                     header=mom0_hdrs[mom0_name],
                     add_beam=True,
                     vmax=1.5,
                     vmin=0.00,
                     add_scalebar=True,
                     scalebar_loc='lower right',
                     scalebar_distance=200,
                     scalebar_color='k',
                     markerstar_color='k',
                     markerstar_width=3,
                     scalebar_units='au',
                     ax=ax[1],
                     contour_colors='r',
                     contour_sigma_filter=2,
                     show_slice_return='contour_imax',
                     plot_vel=False,
                     render='contours',
                     contour_levels=contour_levels[mom0_name],
                     contour_label=mom0_name,
                     plot_cbar=False,
                     icrs_ylabel='')

### Mom1
mom1_name = 'EHV'
im_EHV_mom1, ax_EHV_mom1 = show_slice_cube(mom1_datas[mom1_name],
                     channel=0,
                     box=box_moms_EHV,
                     wcs=mom1_wcss[mom1_name],
                     header=mom1_hdrs[mom1_name],
                     add_beam=True,
                     beam_color='k',
                     vmax=115,
                     vcenter=100,
                     vmin=78,
                     norm='divnorm',
                     cmap='jet_r',
                     add_scalebar=True,
                     scalebar_loc='lower right',
                     scalebar_distance=200,
                     scalebar_color='k',
                     scalebar_units='au',
                     ax=ax[0],
                     return_ax=True,
                     plot_vel=False,
                     cbar_unit='km/s',
                     render='raster',
                     contour_levels=None,
                     plot_cbar=True,
                     colorbar_orientation='horizontal',
                     colorbar_fraction=0.1,
                     colorbar_pad=0.1,
                     colorbar_shrink=0.7,
                     colorbar_aspect=20,
                     colorbar_anchor=(0.5,1),
                     colorbar_panchor=(0.5,0.0),
                     cbar_extend='neither',
                     rotate_ticktext_yaxis=90,)

mom0_name = 'EHV'
im_EHV_mom0, ax_EHV_mom0 = show_slice_cube(mom0_datas[mom0_name],
                     channel=0,
                     box=box_moms_EHV,
                     wcs=mom0_wcss[mom0_name],
                     header=mom0_hdrs[mom0_name],
                     add_beam=True,
                     beam_color='k',
                     vmax=10,
                     vmin=0.00,
                     add_scalebar=True,
                     scalebar_loc='lower right',
                     scalebar_distance=200,
                     scalebar_color='k',
                     markerstar_color='k',
                     markerstar_width=3,
                     scalebar_units='au',
                     ax = ax_EHV_mom1,
                     contour_colors='k',
                     show_slice_return='contour_imax',
                     plot_vel=False,
                     render='contours',
                     contour_levels=contour_levels[mom0_name],
                     contour_label=mom0_name,
                     plot_cbar=False)




#mom1_name = 'LV-IHV'
#sigma = np.std(mom1_datas[mom1_name][0:100,0:100])
#max_value = np.max(mom1_datas[mom1_name])
##contour_levels = np.linspace(3,max_value/sigma,6) * sigma
#im_IHV_mom1, ax_IHV_mom1 = show_slice_cube(mom1_datas[mom1_name],
#                     channel=0,
#                     box=box_moms_IHV,
#                     wcs=mom1_wcss[mom1_name],
#                     header=mom1_hdrs[mom1_name],
#                     add_beam=True,
#                     beam_color='k',
#                     vmax=78,
#                     vcenter=50,
#                     vmin=16,
#                     norm='divnorm',
#                     cmap='jet_r',
#                     add_scalebar=True,
#                     scalebar_loc='lower right',
#                     scalebar_distance=200,
#                     scalebar_units='au',
#                     ax = ax[2],
#                     return_ax=True,
#                     plot_vel=False,
#                     cbar_unit='km/s',
#                     render='raster',
#                     contour_levels=None,
#                     plot_cbar=True,
#                     icrs_ylabel='')
#
#mom0_name = 'LV-IHV'
#sigma = np.std(mom0_datas[mom0_name][0:100,0:100])
#max_value = np.max(mom0_datas[mom0_name])
#contour_levels = np.linspace(3,max_value/sigma,6) * sigma
#im_IHV_mom0, ax_IHV_mom0 = show_slice_cube(mom0_datas[mom0_name],
#                     channel=0,
#                     box=box_moms_IHV,
#                     wcs=mom0_wcss[mom0_name],
#                     header=mom0_hdrs[mom0_name],
#                     add_beam=True,
#                     beam_color='k',
#                     vmax=1.5,
#                     vmin=0.00,
#                     add_scalebar=True,
#                     scalebar_loc='lower right',
#                     scalebar_distance=200,
#                     scalebar_color='k',
#                     markerstar_color='k',
#                     markerstar_width=3,
#                     scalebar_units='au',
#                     ax=ax_IHV_mom1,
#                     contour_colors='k',
#                     show_slice_return='contour_imax',
#                     plot_vel=False,
#                     render='contours',
#                     contour_levels=contour_levels,
#                     contour_label=mom0_name,
#                     plot_cbar=False,
#                     icrs_ylabel='')
#

mom1_name = 'LV-IHV'
im_LV_IHV_mom1, ax_LV_IHV_mom1 = show_slice_cube(mom1_datas[mom1_name],
                     channel=0,
                     box=box_moms_LV_IHV,
                     wcs=mom1_wcss[mom1_name],
                     header=mom1_hdrs[mom1_name],
                     add_beam=True,
                     beam_color='k',
                     vmax=78,
                     vcenter=50,
                     vmin=0,
                     norm='divnorm',
                     cmap='jet_r',
                     add_scalebar=True,
                     scalebar_loc='lower right',
                     scalebar_distance=200,
                     scalebar_units='au',
                     ax = ax[2],
                     return_ax=True,
                     plot_vel=False,
                     cbar_unit='km/s',
                     render='raster',
                     contour_levels=None,
                     plot_cbar=True,
                     colorbar_orientation='horizontal',
                     colorbar_fraction=0.1,
                     colorbar_pad=0.1,
                     colorbar_shrink=0.7,
                     colorbar_aspect=20,
                     colorbar_anchor=(0.5,1),
                     colorbar_panchor=(0.5,0.0),
                     cbar_extend='neither',
                     icrs_ylabel='',
                     rotate_ticktext_yaxis=90,)

mom0_name = 'LV-IHV'
im_LV_IHV_mom0, ax_LV_IHV_mom0 = show_slice_cube(mom0_datas[mom0_name],
                     channel=0,
                     box=box_moms_LV_IHV,
                     wcs=mom0_wcss[mom0_name],
                     header=mom0_hdrs[mom0_name],
                     add_beam=True,
                     beam_color='k',
                     vmax=1.5,
                     vmin=0.00,
                     add_scalebar=True,
                     scalebar_loc='lower right',
                     scalebar_distance=200,
                     scalebar_color='k',
                     markerstar_color='k',
                     markerstar_width=3,
                     scalebar_units='au',
                     ax=ax_LV_IHV_mom1,
                     contour_colors='k',
                     show_slice_return='contour_imax',
                     plot_vel=False,
                     render='contours',
                     contour_levels=contour_levels[mom0_name],
                     contour_label=mom0_name,
                     contour_linewidths=1,
                     plot_cbar=False,
                     icrs_ylabel='')

h1,l1 = im_EHV.legend_elements()
h2,l2 = im_LV_IHV.legend_elements()
h3,l3 = im_allred.legend_elements()
ax[1].legend([h1[0],h2[0],h3[0]], ['EHV','LV-IHV','LV-IHVred'], loc='upper left')

arcs_pix_mom0_EHV = mf.arcs2pix(arcs_sky_hodapp, mom0_hdrs['EHV'])
#arcs_sky_hodapp = arcs2skycoord(arcs_pix_hodapp, wcs_hodapp, hdr_hodapp)
#arcs_pix_lsr = arcs2pix(arcs_sky_hodapp, wcs_lsr, hdr_lsr)
#arcs_pix_spw17 = arcs2pix(arcs_sky_hodapp, wcs_spw17, hdr_spw17)

pix_rear_arc = 5
for ax_n in [ax[0], ax[1], ax[2]]:
    for arc in arcs_pix_mom0_EHV:
        patch_arc_black = Arc((arcs_pix_mom0_EHV[arc]['x0'],
                               arcs_pix_mom0_EHV[arc]['y0']),
                   width=arcs_pix_mom0_EHV[arc]['width']-pix_rear_arc,
                   height=arcs_pix_mom0_EHV[arc]['height']-pix_rear_arc,
                   angle=arcs_pix_mom0_EHV[arc]['angle'],
                   theta1=arcs_pix_mom0_EHV[arc]['theta1'],
                   theta2=arcs_pix_mom0_EHV[arc]['theta2'],
                   linewidth=2,
                   linestyle='-',
                   color='xkcd:light gray',
                   fill=False,
                   zorder=5,)

        patch_arc_white = Arc((arcs_pix_mom0_EHV[arc]['x0'],
                               arcs_pix_mom0_EHV[arc]['y0']),
                   width=arcs_pix_mom0_EHV[arc]['width'],
                   height=arcs_pix_mom0_EHV[arc]['height'],
                   angle=arcs_pix_mom0_EHV[arc]['angle'],
                   theta1=arcs_pix_mom0_EHV[arc]['theta1'],
                   theta2=arcs_pix_mom0_EHV[arc]['theta2'],
                   color='k',
                   linewidth=2,
#                   linestyle=':',
                   linestyle='-',
                   fill=False,
                   zorder=6)
        ax_n.add_patch(patch_arc_black)
        ax_n.add_patch(patch_arc_white)


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



ax_LV_IHV_mom1.plot((LV_IHV_PV['start'][0], LV_IHV_PV['end'][0]),
           (LV_IHV_PV['start'][1], LV_IHV_PV['end'][1]),
            'k-.',
           linewidth=2,
           transform=ax_LV_IHV_mom1.get_transform('icrs'))

ax_EHV_mom1.plot((LV_IHV_PV['start'][0], LV_IHV_PV['end'][0]),
           (LV_IHV_PV['start'][1], LV_IHV_PV['end'][1]),
            'k-.',
           linewidth=2,
           transform=ax_EHV_mom1.get_transform('icrs'))

ax_EHV_mom1.plot((EHVjet_PV['start'][0], EHVjet_PV['end'][0]),
           (EHVjet_PV['start'][1], EHVjet_PV['end'][1]),
            'k-.',
           linewidth=2,
           transform=ax_EHV_mom1.get_transform('icrs'))

ax_EHV_mom1.plot((EHVjet_PV2['start'][0], EHVjet_PV2['end'][0]),
           (EHVjet_PV2['start'][1], EHVjet_PV2['end'][1]),
            'k-.',
           linewidth=2,
           transform=ax_EHV_mom1.get_transform('icrs'))



ax[1].set_aspect('equal')
ax[0].set_aspect('equal')
ax[2].set_aspect('equal')



def mark_inset_ext(parent_axes, inset_axes,
                   loc1a=1,
                   loc1b=1,
                   loc2a=2,
                   loc2b=2,
                   **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData, )

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2

def mark_inset_ext2(parent_axes,
                    inset_axes,
                    loc1a=1,
                    loc1b=1,
                    loc2a=2,
                    loc2b=2,
                    **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData, )

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(rect, inset_axes.bbox, loc1=loc1b, loc2=loc1a, **kwargs)
    parent_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(rect, inset_axes.bbox, loc1=loc2b, loc2=loc2a, **kwargs)
    parent_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2

#ax[1].indicate_inset_zoom(ax[0])
#ax[1].indicate_inset_zoom(ax[2])
mark_inset_ext2(ax[1],
                ax[0],
                loc1a=1,
                loc1b=2,
                loc2a=4,
                loc2b=3,
                fc="none",
                ec="0.5")
mark_inset_ext(ax[1],
               ax[2],
               loc1a=3,
               loc1b=4,
               loc2a=2,
               loc2b=1,
               fc="none",
               ec="0.5")


#mark_inset(ax[1], ax[2], loc1=1, loc2=4, fc="none", ec="0.5")
#mark_inset(ax[1], ax[0], loc1=2, loc2=3, fc="none", ec="0.5")
#mark_inset(ax[2], ax[1], loc1=4, loc2=1, fc="none", ec="0.5")

fig.savefig('{}moments_SVS13_horizontal.pdf'.format(mf.default_params['path_save']),
            bbox_inches=mf.default_params['bbox_inches'])

#chans = [0,3839]
#data_EHVjet_PV, wcs_EHVjet_PV, hdr_EHVjet_PV = casa_impv(spw17_path,
#                                                         EHVjet_PV['start'],
#                                                         EHVjet_PV['end'],
#                                                         width=3,
#                                                         chans=chans)
#
#chans = [0,3839]
#data_EHVjet_PV2, wcs_EHVjet_PV2, hdr_EHVjet_PV2 = casa_impv(spw17_path,
#                                                         EHVjet_PV2['start'],
#                                                         EHVjet_PV2['end'],
#                                                         width=5,
#                                                         chans=chans)
#
#chans = [0,3839]
#data_LV_IHV, wcs_LV_IHV_PV, hdr_LV_IHV_PV = casa_impv(spw17_path,
#                                                         LV_IHV_PV['start'],
#                                                         LV_IHV_PV['end'],
#                                                         width=7,
#                                                         chans=chans)
#
chans = [0,344]
data_LV_IHV_lsr, wcs_LV_IHV_PV_lsr, hdr_LV_IHV_PV_lsr = casa_impv(lsr_path,
                                                         LV_IHV_PV['start'],
                                                         LV_IHV_PV['end'],
                                                         width=5,
                                                         chans=chans)



