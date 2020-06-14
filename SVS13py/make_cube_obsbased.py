import numpy as np
from numpy.linalg import eig, inv

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.font_manager as fm
from matplotlib.colors import to_hex
import matplotlib.image as mpimg
from matplotlib.patches import Arc, Ellipse, ConnectionPatch
from matplotlib.gridspec import GridSpec

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector, mark_inset

from IPython.display import display, clear_output

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord, concatenate
from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D

from photutils.isophote import EllipseGeometry, build_ellipse_model, EllipseSample, EllipseFitter, Isophote
from photutils import EllipticalAperture

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import optimize

import warnings

import shelve

import copy

import pandas as pd

from itertools import product

import casa_proc

import telegram_send

import importlib

import SVS13py.SVS13py
from SVS13py.ellipse_fitter import EllipseFitter
import SVS13py.ellfit_results as ellfit_results
import SVS13py.mf as mf
import SVS13py.main_functions as main_functions
import SVS13py.windmodel_correction as wm
import SVS13py.models as models
import SVS13py.bubblepy_general as bbpy

import subprocess
result = subprocess.run(['whoami'], stdout=subprocess.PIPE)
user = result.stdout.decode().strip()



# fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13_nocont.fits')
#fits_path = get_pkg_data_filename(mf.default_params['path_fits'])
#fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13.fits')
#hdu = fits.open(fits_path)[0]
if user=='gblazquez':
    lsr_path = '/home/gblazquez/data/spw-2-9-gb.contsub.lsr.fits'
elif user=='guille':
    lsr_path = '/mnt/hdd/data/spw-2-9-gb.contsub.lsr.fits'
    #lsr_path = '/run/media/guille/4589F60A42CB3786/data/spw-2-9-gb.contsub.lsr.fits'

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
if user == 'gblazquez':
    spw17_path = '/home/gblazquez/data/spw-17-gb.244kHz.3840chans.contsub.fits'
elif user == 'guille':
    spw17_path = '/mnt/hdd/data/spw-17-gb.244kHz.3840chans.contsub.fits'
    #spw17_path = '/run/media/guille/4589F60A42CB3786/data/spw-17-gb.244kHz.3840chans.contsub.fits'

hdu_spw17 = fits.open(spw17_path)[0]
hdr_spw17 = hdu_spw17.header
wcs_spw17 = WCS(hdu_spw17.header).celestial
image_data_spw17 = hdu_spw17.data[0]
#v0_spw17=88.7103
#vf_spw17=-93.2237


# fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13_nocont.fits')
#fits_path = get_pkg_data_filename(mf.default_params['path_fits'])
#fits_path = get_pkg_data_filename('/run/user/1000/gvfs/sftp:host=dragom.iaa.es,user=gblazquez/data/gblazquez/SVS13_data/SVS13.fits')
#hdu = fits.open(fits_path)[0]
if user == 'gblazquez':
    spw17_lsr_path = '/home/gblazquez/data/spw-17-gb.610kHz.chans1536.contsub.fits'
elif user == 'guille':
    spw17_lsr_path = '/mnt/hdd/data/spw-17-gb.610kHz.chans1536.contsub.fits'
#    spw17_lsr_path = '/run/media/guille/4589F60A42CB3786/data/spw-17-gb.610kHz.chans1536.contsub.fits'


hdu_spw17_lsr = fits.open(spw17_lsr_path)[0]
hdr_spw17_lsr = hdu_spw17_lsr.header
wcs_spw17_lsr = WCS(hdu_spw17_lsr.header).celestial
image_data_spw17_lsr = hdu_spw17_lsr.data[0]
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


marcsyr2kms = lambda marcsyr: ((marcsyr*mf.default_params['SVS13_distance']/1000)*u.au/u.yr).to(u.km/u.s).value
inclination_angle_calc = lambda v_lsr, v_sys, vz_proyected: np.arctan(vz_proyected/(v_lsr-v_sys)) * 180 / np.pi

vz_proyected_last_arc = marcsyr2kms(31) #proyected velocity of the bow shock front
vz_proyected_bubble = marcsyr2kms(20)

inclination_angle_SVS13 = inclination_angle_calc(-80,
                                                 mf.default_params['SVS13_vsys'],
                                                 -vz_proyected_last_arc)
#inclination_angle_SVS13 = inclination_angle_calc(-90,mf.default_params['SVS13_vsys'],-vz_proyected_last_arc)

inclination_angle_SVS13


results_fitting_keys = [#'big_cavity',
                        'minor_bubble',
                        'veryfirst_little_bubble',
                        'veryfirst_bigger_bubble',
                        'first_cavity',
                        'second_cavity',
                        'first_cavity_tip',
                        'second_cavity_tip',
                        'first_knot_rings',
                        'first_knot',
                        'second_knot_rings_dubious',]
#                        'main_arcs_annulus',
#                        'first_arcs_annulus',
#                        'nearest_arcs_annulus',
#                        'secondary_arcs_annulus',
#                        'clouds_elliptical',
#                        'last_filled']
inclination_angle=inclination_angle_SVS13
results_fit = {rf: ellfit_results.import_table(rf,
                                hdr_lsr,
                                syst_vel=+8.5,
                                s_p=2.5,
                                a_p=5,
                                deproject_i=inclination_angle*np.pi/180) \
                                for rf in results_fitting_keys}

results_fit['low_res_arcs'] = ellfit_results.import_table('low_res_arcs',
                                           hdr_spw17_lsr,
                                           syst_vel=+8.5,
                                            s_p=2.5,
                                            a_p=14,
                                           deproject_i=inclination_angle*np.pi/180)
results_fit['low_res_arcs_dubious'] = ellfit_results.import_table('low_res_arcs_dubious',
                                                   hdr_spw17_lsr,
                                                   syst_vel=+8.5,
                                                    s_p=2.5,
                                                    a_p=14,
                                                   deproject_i=inclination_angle*np.pi/180)
results_fit['low_res_arcs_tip'] = ellfit_results.import_table('low_res_arcs_tip',
                                               hdr_spw17_lsr,
                                               syst_vel=+8.5,
                                               s_p=2.5,
                                               a_p=14,
                                               deproject_i=inclination_angle*np.pi/180)

#results_fit['little_tip'] = ellfit_results.import_table('little_tip',
#                                               hdr_spw17_lsr,
#                                               syst_vel=+8.5,
#                                               deproject_i=inclination_angle*np.pi/180)
dic_bb = {}

alpha_lowv = 1
#dic_bb['big_cavity'] = {'data':results_fit['big_cavity'],
#                                    'c':'k',
#                                    'sty':'.',
#                                    'fsty':None,
#                                    'regr':None,
#                                    'sty_regr':'k--',
#                                    'linestyle':'-',
#                                    'pos_text':None,
#                                    'plot_centers':True,
#                                    'markersize_centers':10,
#                                    'markeredgewidth_centers':0.5,
#                                    'markerfacecolor_centers':None,
#                                    'markeredgecolor_centers':'k',
#                                    'alpha':alpha_lowv,
#                                    'plot_ellipses':'aper',
#                                    'c_ell':'g',
#                                    'step_chan':1,
#                                    'label':'Rings I',
#                                    'centers_zorder':5,
#                                    'regr_zorder':4,
#                                    'ellipse_zorder':2}

dic_bb['minor_bubble'] = {'data':results_fit['minor_bubble'],
                                    'c':'k',
                                    'sty':'.',
                                    'fsty':None,
                                    'regr':None,
                                    'sty_regr':'k--',
                                    'linestyle':'-',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':alpha_lowv,
                                    'plot_ellipses':'aper',
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':'Rings I',
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':2}

dic_bb['veryfirst_little_bubble'] = {'data':results_fit['veryfirst_little_bubble'],
                                    'c':'k',
                                    'sty':'.',
                                    'fsty':None,
                                    'regr':3,
                                    'sty_regr':'k--',
                                    'linestyle':'-',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':alpha_lowv,
                                    'plot_ellipses':'aper',
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':None,
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':1}

dic_bb['veryfirst_bigger_bubble'] = {'data':results_fit['veryfirst_bigger_bubble'],
                                    'c':'k',
                                    'sty':'.',
                                    'fsty':None,
                                    'linestyle':'-',
                                    'regr':3,
                                    'sty_regr':'k--',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':alpha_lowv,
                                    'plot_ellipses':'aper',
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':None,
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':1}

dic_bb['first_cavity'] = {'data':results_fit['first_cavity'],
                          'c':'g',
                          'sty':'.',
                          'fsty':None,
                          'regr':1,
                          'sty_regr':'k--',
                          'linestyle':'-',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':10,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':alpha_lowv,
                          'plot_ellipses':'patches',
                          'theta1_pa':120,
                          'theta2_pa':40,
                          'c_ell':'b',
                          'step_chan':1,
                          'label':'Rings II',
                          'centers_zorder':5,
                          'regr_zorder':4,
                          'ellipse_zorder':None}

dic_bb['first_cavity2'] = {'data':results_fit['first_cavity'],
                          'c':'g',
                          'sty':'.',
                          'fsty':None,
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'-',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':10,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':0.1,
                          'plot_ellipses':'aper',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':None,
                          'centers_zorder':5,
                          'regr_zorder':4,
                          'ellipse_zorder':None}

dic_bb['first_cavity_tip'] = {'data':results_fit['first_cavity_tip'][results_fit['first_cavity_tip']['vel']>-91],
                          'c':'g',
                          'sty':'x',
                          'fsty':'none',
                          'regr':1,
                          'sty_regr':'k--',
                          'linestyle':'--',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':6,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':alpha_lowv,
                          'theta1_pa':120,
                          'theta2_pa':40,
                          'plot_ellipses':'patches',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':None,
                          'centers_zorder':4,
                          'regr_zorder':3,
                          'ellipse_zorder':1}

dic_bb['first_cavity_tip2'] = {'data':results_fit['first_cavity_tip'][results_fit['first_cavity_tip']['vel']>-91],
                          'c':'g',
                          'sty':'x',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'--',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':6,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':0.05,
                          'plot_ellipses':'filled_ellipse',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':None,
                          'centers_zorder':4,
                          'regr_zorder':3,
                          'ellipse_zorder':1}


dic_bb['second_cavity'] = {'data':results_fit['second_cavity'],
                                    'c':'r',
                                    'sty':'.',
                                    'fsty':None,
                                    'regr':2,
                                    'sty_regr':'k--',
                                    'linestyle':'-',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':alpha_lowv,
                                    'plot_ellipses':'patches',
                                    'theta1_pa':120,
                                    'theta2_pa':-30,
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':'Rings III',
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':None}

dic_bb['second_cavity2'] = {'data':results_fit['second_cavity'],
                                    'c':'r',
                                    'sty':'.',
                                    'fsty':None,
                                    'regr':None,
                                    'sty_regr':'k--',
                                    'linestyle':'-',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':0.1,
                                    'plot_ellipses':'aper',
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':None,
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':None}


dic_bb['second_cavity_tip'] = {'data':results_fit['second_cavity_tip'][results_fit['second_cavity_tip']['vel']>-93],
                          'c':'r',
                          'sty':'x',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'--',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':6,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':alpha_lowv,
                          'theta1_pa':120,
                          'theta2_pa':40,
                          'plot_ellipses':'patches',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':None,
                          'centers_zorder':4,
                          'regr_zorder':3,
                          'ellipse_zorder':2}


dic_bb['second_cavity_tip2'] = {'data':results_fit['second_cavity_tip'][results_fit['second_cavity_tip']['vel']>-93],
                          'c':'r',
                          'sty':'x',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'-',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':6,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':0.05,
                          'plot_ellipses':'filled_ellipse',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':None,
                          'centers_zorder':4,
                          'regr_zorder':3,
                          'ellipse_zorder':2}



dic_bb['first_knot_rings'] = {'data':results_fit['first_knot_rings'],
                          'c':'gray',
                          'sty':'.',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'-',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':10,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':alpha_lowv,
                          'theta1_pa':130,
                          'theta2_pa':-5,
                          'plot_ellipses':'patches',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':'Rings IV',
                          'centers_zorder':4,
                          'regr_zorder':3,
                          'ellipse_zorder':2}


dic_bb['first_knot_rings2'] = {'data':results_fit['first_knot_rings'],
                          'c':'gray',
                          'sty':'.',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'-',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':10,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':0.1,
                          'plot_ellipses':'aper',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':None,
                          'centers_zorder':4,
                          'regr_zorder':3,
                          'ellipse_zorder':2}

dic_bb['first_knot'] = {'data':results_fit['first_knot'],
                          'c':'gray',
                          'sty':'x',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'--',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':6,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':alpha_lowv,
                          'theta1_pa':130,
                          'theta2_pa':-5,
                          'plot_ellipses':'patches',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':None,
                          'centers_zorder':3,
                          'regr_zorder':3,
                          'ellipse_zorder':2}


dic_bb['first_knot2'] = {'data':results_fit['first_knot'],
                          'c':'gray',
                          'sty':'x',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'-',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':6,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':0.05,
                          'plot_ellipses':'filled_ellipse',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':None,
                          'centers_zorder':3,
                          'regr_zorder':3,
                          'ellipse_zorder':2}

dic_bb['second_knot_rings_dubious'] = {'data':results_fit['second_knot_rings_dubious'],
                          'c':'m',
                          'sty':'.',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'-',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':10,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':alpha_lowv,
                          'theta1_pa':130,
                          'theta2_pa':-5,
                          'plot_ellipses':'patches',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':'Rings V',
                          'centers_zorder':4,
                          'regr_zorder':3,
                          'ellipse_zorder':2}


dic_bb['second_knot_rings_dubious2'] = {'data':results_fit['second_knot_rings_dubious'],
                          'c':'m',
                          'sty':'.',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'-',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':10,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':0.1,
                          'plot_ellipses':'aper',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':None,
                          'centers_zorder':4,
                          'regr_zorder':3,
                          'ellipse_zorder':2}

#dic_bb['low_res_arcs_dubious'] = {'data':results_fit['low_res_arcs_dubious'],
#                                    'c':'g', 'sty':'.', 'fsty':None,
#                                    'regr':None, 'sty_regr':'k--',
#                                    'linestyle':'-',
#                                    'pos_text':None,
#                                    'plot_centers':True,
#                                     'markersize_centers':10,
#                                     'markeredgewidth_centers':0.5,
#                                     'markerfacecolor_centers':None,
#                                     'markeredgecolor_centers':'k',
#                                    'alpha':alpha_lowv,
#                                    'plot_ellipses':'patches',
#                                    'theta1_pa':120,
#                                    'theta2_pa':-30,
#                                    'c_ell':'g',
#                                    'step_chan':1,
#                                    'label':True,
#                                    'centers_zorder':1,
#                                    'regr_zorder':1,
#                                    'ellipse_zorder':3}
#
#dic_bb['low_res_arcs_dubious2'] = {'data':results_fit['low_res_arcs_dubious'],
#                                    'c':'g', 'sty':'.', 'fsty':None,
#                                    'regr':None, 'sty_regr':'k--',
#                                    'linestyle':'-',
#                                    'pos_text':None,
#                                    'plot_centers':True,
#                                    'markersizhttps://youtu.be/IBhvat5u9wIe_centers':10,
#                                    'markeredgewidth_centers':0.5,
#                                    'markerfacecolor_centers':None,
#                                    'markeredgecolor_centers':'k',
#                                    'alpha':0.05,
#                                    'plot_ellipses':'aper',
#                                    'c_ell':'g',
#                                    'step_chan':1,
#                                    'label':True,
#                                    'centers_zorder':1,
#                                    'regr_zorder':1,
#                                    'ellipse_zorder':3}

dic_bb['low_res_arcs'] = {'data':results_fit['low_res_arcs'][results_fit['low_res_arcs']['vel']>-106],
                                    'c':'b', 'sty':'.', 'fsty':None,
                                    'regr':None, 'sty_regr':'k--',
                                    'linestyle':'-',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':alpha_lowv,
                                    'plot_ellipses':'patches',
                                    'theta1_pa':120,
                                    'theta2_pa':-30,
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':'Rings VI',
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':4}

dic_bb['low_res_arcs2'] = {'data':results_fit['low_res_arcs'][results_fit['low_res_arcs']['vel']>-106],
                                    'c':'b',
                                    'sty':'.',
                                    'fsty':None,
                                    'regr':None,
                                    'sty_regr':'k--',
                                    'linestyle':'-',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':0.1,
                                    'plot_ellipses':'aper',
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':None,
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':4}



dic_bb['low_res_arcs_tip'] = {'data':results_fit['low_res_arcs_tip'],
                                    'c':'b',
                                    'sty':'x',
                                    'fsty':'none',
                                    'regr':None,
                                    'sty_regr':'k--',
                                    'linestyle':'--',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':6,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':alpha_lowv,
                                    'plot_ellipses':'patches',
                                    'theta1_pa':120,
                                    'theta2_pa':-30,
                                    'c_ell':'k',
                                    'step_chan':1,
                                    'label':None,
                                    'centers_zorder':4,
                                    'regr_zorder':4,
                                    'ellipse_zorder':4}

dic_bb['low_res_arcs_tip2'] = {'data':results_fit['low_res_arcs_tip'],
                                    'c':'b',
                                    'sty':'x',
                                    'fsty':'none',
                                    'regr':None,
                                    'sty_regr':'k--',
                                    'linestyle':'--',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':6,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':0.05,
                                    'plot_ellipses':'filled_ellipse',
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':None,
                                    'centers_zorder':4,
                                    'regr_zorder':4,
                                    'ellipse_zorder':4}



#dic_bb['little_tip'] = {'data':results_fit['little_tip'],
#                                    'c':'c',
#                                    'sty':'.',
#                                    'fsty':None,
#                                    'regr':None,
#                                    'sty_regr':'k--',
#                                    'linestyle':'--',
#                                    'pos_text':[mf.default_params['vla4b_deg'][0]+0.00004, mf.default_params['vla4b_deg'][1]+0.00005],
#                                    'plot_centers':True,
#                                    'markersize_centers':10,
#                                    'markeredgewidth_centers':0.5,
#                                    'markerfacecolor_centers':'None',
#                                    'markeredgecolor_centers':None,
#                                    'alpha':0.5,
#                                    'plot_ellipses':'patches',
#                                    'theta1_pa':120,
#                                    'theta2_pa':-30,
#                                    'c_ell':'g',
#                                    'step_chan':1,
#                                    'label':None,
#                                    'centers_zorder':5,
#                                    'regr_zorder':4,
#                                    'ellipse_zorder':4}
#
#
#bbs_zoom = ['main_arcs_1bb',
#            'main_arcs_1bb2',
#            'first_cavity_tip',
#            'first_cavity_tip2',
#            'second_cavity_tip',
#            'second_cavity_tip2',
#            'secondary_arcs_annulus',
#            'secondary_arcs_annulus2',
#            'minor_bubble',
#            'veryfirst_little_bubble',
#            'veryfirst_bigger_bubble',]

dic_bb_zoom = {}

#dic_bb_zoom['big_cavity'] = {'data':results_fit['big_cavity'],
#                                    'c':'k',
#                                    'sty':'.',
#                                    'fsty':None,
#                                    'regr':None,
#                                    'sty_regr':'k--',
#                                    'linestyle':'-',
#                                    'pos_text':[mf.default_params['vla4a_deg'][0]-0.00004,mf.default_params['vla4a_deg'][1]+0.00003],
#                                    'plot_centers':True,
#                                    'markersize_centers':10,
#                                    'markeredgewidth_centers':0.5,
#                                    'markerfacecolor_centers':None,
#                                    'markeredgecolor_centers':'k',
#                                    'alpha':alpha_lowv,
#                                    'plot_ellipses':'aper',
#                                    'c_ell':'g',
#                                    'step_chan':1,
#                                    'label':True,
#                                    'centers_zorder':5,
#                                    'regr_zorder':4,
#                                    'ellipse_zorder':2}

dic_bb_zoom['minor_bubble'] = {'data':results_fit['minor_bubble'],
                                    'c':'k',
                                    'sty':'.',
                                    'fsty':None,
                                    'regr':3, 'sty_regr':'k--',
                                    'linestyle':'-',
                                    'pos_text':[mf.default_params['vla4a_deg'][0]-0.00004,mf.default_params['vla4a_deg'][1]+0.00003],
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':alpha_lowv,
                                    'plot_ellipses':'aper',
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':True,
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':2}

dic_bb_zoom['veryfirst_little_bubble'] = {'data':results_fit['veryfirst_little_bubble'],
                                    'c':'k',
                                    'sty':'.',
                                    'fsty':None,
                                    'regr':3,
                                    'sty_regr':'k--',
                                    'linestyle':'-',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':alpha_lowv,
                                    'plot_ellipses':'aper',
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':True,
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':1}

dic_bb_zoom['veryfirst_bigger_bubble'] = {'data':results_fit['veryfirst_bigger_bubble'],
                                    'c':'k',
                                    'sty':'.',
                                    'fsty':None,
                                    'linestyle':'-',
                                    'regr':3,
                                    'sty_regr':'k--',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':alpha_lowv,
                                    'plot_ellipses':'aper',
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':True,
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':1}

dic_bb_zoom['first_cavity'] = {'data':results_fit['first_cavity'],
                          'c':'g',
                          'sty':'.',
                          'fsty':None,
                          'regr':1,
                          'sty_regr':'k--',
                          'linestyle':'-',
                          'pos_text':[mf.default_params['vla4a_deg'][0],mf.default_params['vla4a_deg'][1]+0.00006],
                          'plot_centers':True,
                          'markersize_centers':10,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':alpha_lowv,
                          'plot_ellipses':'patches',
                          'theta1_pa':120,
                          'theta2_pa':40,
                          'c_ell':'b',
                          'step_chan':1,
                          'label':True,
                          'centers_zorder':5,
                          'regr_zorder':4,
                          'ellipse_zorder':1}

dic_bb_zoom['first_cavity2'] = {'data':results_fit['first_cavity'],
                          'c':'g',
                          'sty':'.',
                          'fsty':None,
                          'regr':1,
                          'sty_regr':'k--',
                          'linestyle':'-',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':10,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':0.1,
                          'plot_ellipses':'aper',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':True,
                          'centers_zorder':5,
                          'regr_zorder':4,
                          'ellipse_zorder':2}

dic_bb_zoom['first_cavity_tip'] = {'data':results_fit['first_cavity_tip'][results_fit['first_cavity_tip']['vel']>-92],
                          'c':'g',
                          'sty':'x',
                          'fsty':'none',
                          'regr':1,
                          'sty_regr':'k--',
                          'linestyle':'--',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':6,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':0.5,
                          'theta1_pa':120,
                          'theta2_pa':40,
                          'plot_ellipses':'patches',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':True,
                          'centers_zorder':4,
                          'regr_zorder':0,
                          'ellipse_zorder':0}

dic_bb_zoom['first_cavity_tip2'] = {'data':results_fit['first_cavity_tip'][results_fit['first_cavity_tip']['vel']>-93],
                          'c':'g',
                          'sty':'x',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'--',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':6,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':0.1,
                          'plot_ellipses':'filled_ellipse',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':True,
                          'centers_zorder':4,
                          'regr_zorder':0,
                          'ellipse_zorder':1}


dic_bb_zoom['second_cavity'] = {'data':results_fit['second_cavity'],
                                    'c':'r',
                                    'sty':'.',
                                    'fsty':None,
                                    'regr':2,
                                    'sty_regr':'k--',
                                    'linestyle':'-',
                                    'pos_text':[mf.default_params['vla4b_deg'][0]+0.00004, mf.default_params['vla4b_deg'][1]+0.00005],
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':alpha_lowv,
                                    'plot_ellipses':'patches',
                                    'theta1_pa':120,
                                    'theta2_pa':-30,
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':True,
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':2}

dic_bb_zoom['second_cavity2'] = {'data':results_fit['second_cavity'],
                                    'c':'r',
                                    'sty':'.',
                                    'fsty':None,
                                    'regr':None,
                                    'sty_regr':'k--',
                                    'linestyle':'-',
                                    'pos_text':None,
                                    'plot_centers':True,
                                    'markersize_centers':10,
                                    'markeredgewidth_centers':0.5,
                                    'markerfacecolor_centers':None,
                                    'markeredgecolor_centers':'k',
                                    'alpha':0.1,
                                    'plot_ellipses':'aper',
                                    'c_ell':'g',
                                    'step_chan':1,
                                    'label':True,
                                    'centers_zorder':5,
                                    'regr_zorder':4,
                                    'ellipse_zorder':2}


dic_bb_zoom['second_cavity_tip'] = {'data':results_fit['second_cavity_tip'][results_fit['second_cavity_tip']['vel']>-93],
                          'c':'r',
                          'sty':'x',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'--',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':6,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':alpha_lowv,
                          'theta1_pa':120,
                          'theta2_pa':40,
                          'plot_ellipses':'patches',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':None,
                          'centers_zorder':4,
                          'regr_zorder':3,
                          'ellipse_zorder':3}


dic_bb_zoom['second_cavity_tip2'] = {'data':results_fit['second_cavity_tip'][results_fit['second_cavity_tip']['vel']>-93],
                          'c':'r',
                          'sty':'x',
                          'fsty':'none',
                          'regr':None,
                          'sty_regr':'k--',
                          'linestyle':'-',
                          'pos_text':None,
                          'plot_centers':True,
                          'markersize_centers':6,
                          'markeredgewidth_centers':0.5,
                          'markerfacecolor_centers':None,
                          'markeredgecolor_centers':'k',
                          'alpha':0.05,
                          'plot_ellipses':'filled_ellipse',
                          'c_ell':'b',
                          'step_chan':1,
                          'label':None,
                          'centers_zorder':4,
                          'regr_zorder':3,
                          'ellipse_zorder':3}

arcs_sel = ['minor_bubble',
 'veryfirst_little_bubble',
 'veryfirst_bigger_bubble',
 'first_cavity',
 'first_cavity_tip',
 'second_cavity',
 'second_cavity_tip',
 'first_knot_rings',
 'first_knot',
 'second_knot_rings_dubious',
 'low_res_arcs',
 'low_res_arcs_tip',]
#
tip_arcs = ['first_cavity_tip', 'second_cavity_tip', 'first_knot', 'low_res_arcs_tip']
#
dic_bb_sel = {arc_sel: dic_bb[arc_sel] for arc_sel in arcs_sel}
for arc in tip_arcs:
    dic_bb_sel[arc]['markeredgecolor_centers'] = None



mom0_paths = {}
mom1_paths = {}
for pb in ['', '.pbcor']:
    mom0_paths['EHVjet{}'.format(pb)] = 'spw-17-gb.224kHz.3840chans.contsub.image{}.chans2326~2448.EHVjet_allfield.mom0.fits'.format(pb)
    mom0_paths['EHV{}'.format(pb)] = 'spw-17-gb.224kHz.3840chans.contsub.image{}.chans2326~2513.EHV_allfield.mom0.fits'.format(pb)
    mom0_paths['IHV{}'.format(pb)] = 'spw-2-9-gb.contsub.lsr.image{}.chans198~316.IHV_allfield.mom0.fits'.format(pb)
    mom0_paths['LV{}'.format(pb)] = 'spw-2-9-gb.contsub.lsr.image{}.chans168~198.LV_allfield.mom0.fits'.format(pb)
    mom0_paths['LV-IHV{}'.format(pb)] = 'spw-2-9-gb.contsub.lsr.image{}.chans168~316.LV-IHV_allfield.mom0.fits'.format(pb)
    mom0_paths['SVred{}'.format(pb)] = 'spw-2-9-gb.contsub.lsr.image{}.chans110~149.SVred_allfield.mom0.fits'.format(pb)
    mom0_paths['IHVred{}'.format(pb)] = 'spw-2-9-gb.contsub.lsr.image{}.chans35~110.IHVred_allfield.mom0.fits'.format(pb)
    mom0_paths['allred{}'.format(pb)] = 'spw-2-9-gb.contsub.lsr.image{}.chans35~149.allred_allfield.mom0.fits'.format(pb)
    mom0_paths['first_bullet{}'.format(pb)] = 'spw-2-9.subimage{}.chans846-947.allmap.first_bullet.mom0.fits'.format(pb)
    mom0_paths['second_bullet{}'.format(pb)] = 'spw-2-9.subimage{}.chans990-1141.allmap.second_bullet.mom0.fits'.format(pb)
#    mom0_paths['third_bullet{}'.format(pb)] = 'spw-2-9.subimage{}.chans1263-1478.allmap.third_bullet.mom0.fits'.format(pb)
    mom0_paths['third_bullet{}'.format(pb)] = 'spw-2-9.subimage{}.chans1368-1462.allmap.third_bullet.mom0.fits'.format(pb)
    mom0_paths['first_bullet_jet{}'.format(pb)] = 'spw-2-9.subimage{}.chans1636~1722.allmap.first_bullet_jet.mom0.fits'.format(pb)
    mom0_paths['second_bullet_jet{}'.format(pb)] = 'spw-17-gb.224kHz.3840chans.contsub.image{}.chans2402-2503.allmap.second_bullet_jet.mom0.fits'.format(pb)
    mom0_paths['tip_cavity{}'.format(pb)] = 'spw-2-9.subimage{}.chans1554~1631.allmap.tip_cavity.mom0.fits'.format(pb)
    mom0_paths['red_bb{}'.format(pb)] = 'spw-2-9.subimage{}.chans258-650.allmap.red_bb.mom0.fits'.format(pb)
    mom0_paths['red_jet{}'.format(pb)] = 'spw-2-9.subimage{}.chans578-742.allmap.red_jet.mom0.fits'.format(pb)
    mom0_paths['redarms{}'.format(pb)] = 'spw-2-9.subimage{}.chans590~717.redarms_allfield.mom0.fits'.format(pb)
    mom0_paths['redarms2{}'.format('.pbcor')] = 'spw-2-9.subimage{}.chans590~717.redarms_allfield.mom0.fits'.format('.pbcor')
    mom0_paths['bluearms{}'.format(pb)] = 'spw-2-9.subimage{}.chans840~954.bluearms_allfield.mom0.fits'.format(pb)

    mom1_paths['EHVjet{}'.format(pb)] = 'spw-17-gb.224kHz.3840chans.contsub.image{}.chans2326~2448.EHVjet_allfield.mom1.fits'.format(pb)
    mom1_paths['EHV{}'.format(pb)] = 'spw-17-gb.224kHz.3840chans.contsub.image{}.chans2326~2513.EHV_allfield.mom1.fits'.format(pb)
    mom1_paths['IHV'] = 'spw-2-9-gb.contsub.lsr.image.chans198~316.IHV_allfield.includepix0.013.mom1.fits'
    mom1_paths['LV{}'.format(pb)] = 'spw-2-9-gb.contsub.lsr.image{}.chans168~198.LV_allfield.mom1.fits'.format(pb)
    mom1_paths['LV-IHV{}'.format(pb)] = 'spw-2-9-gb.contsub.lsr.image{}.chans168~316.LV-IHV_allfield.mom1.fits'.format(pb)
    mom1_paths['SVred{}'.format(pb)] = 'spw-2-9-gb.contsub.lsr.image{}.chans110~149.SVred_allfield.mom1.fits'.format(pb)
    mom1_paths['IHVred{}'.format(pb)] = 'spw-2-9-gb.contsub.lsr.image{}.chans35~110.IHVred_allfield.mom1.fits'.format(pb)
    mom1_paths['allred{}'.format(pb)] = 'spw-2-9-gb.contsub.lsr.image{}.chans35~149.allred_allfield.mom1.fits'.format(pb)
    mom1_paths['first_bullet{}'.format(pb)] = 'spw-2-9.subimage{}.chans846-947.allmap.first_bullet.mom1.fits'.format(pb)
    mom1_paths['second_bullet{}'.format(pb)] = 'spw-2-9.subimage{}.chans990-1141.allmap.second_bullet.mom1.fits'.format(pb)
    mom1_paths['third_bullet{}'.format(pb)] = 'spw-2-9.subimage{}.chans1263-1478.allmap.third_bullet.mom1.fits'.format(pb)
    mom1_paths['first_bullet_jet{}'.format(pb)] = 'spw-2-9.subimage{}.chans1636~1722.allmap.first_bullet_jet.mom1.fits'.format(pb)
    mom1_paths['second_bullet_jet{}'.format(pb)] = 'spw-17-gb.224kHz.3840chans.contsub.image{}.chans2402-2503.allmap.second_bullet_jet.mom1.fits'.format(pb)
    mom1_paths['tip_cavity{}'.format(pb)] = 'spw-2-9.subimage{}.chans1554~1631.allmap.tip_cavity.mom1.fits'.format(pb)
    mom1_paths['red_bb{}'.format(pb)] = 'spw-2-9.subimage{}.chans258-650.allmap.red_bb.mom1.fits'.format(pb)
    mom1_paths['red_jet{}'.format(pb)] = 'spw-2-9.subimage{}.chans578-742.allmap.red_jet.mom1.fits'.format(pb)
    mom1_paths['redarms{}'.format(pb)] = 'spw-2-9.subimage{}.chans590~717.redarms_allfield.mom1.fits'.format(pb)
    mom1_paths['redarms2{}'.format('.pbcor')] = 'spw-2-9.subimage{}.chans590~700.includepix0.02.redarms_allfield.mom1.fits'.format('.pbcor')
    mom1_paths['bluearms{}'.format(pb)] = 'spw-2-9.subimage{}.chans840~954.bluearms_allfield.mom1.fits'.format(pb)

#mom1_paths = {'EHVjet':'spw-17-gb.224kHz.3840chans.contsub.image.chans2326~2448.EHVjet_allfield.mom1.fits',
#            'EHV':'spw-17-gb.224kHz.3840chans.contsub.image.chans2326~2513.EHV_allfield.mom1.fits',
#            'IHV':'spw-2-9-gb.contsub.lsr.image.chans198~316.IHV_allfield.mom1.fits',
#            'LV':'spw-2-9-gb.contsub.lsr.image.chans168~198.LV_allfield.mom1.fits',
#            'LV-IHV':'spw-2-9-gb.contsub.lsr.image.chans168~316.LV-IHV_allfield.includepix0.015.mom1.fits',
#            'SVred':'spw-2-9-gb.contsub.lsr.image.chans110~149.SVred_allfield.mom1.fits',
#            'IHVred':'spw-2-9-gb.contsub.lsr.image.chans35~110.IHVred_allfield.mom1.fits',
#            'allred':'spw-2-9-gb.contsub.lsr.image.chans35~149.allred_allfield.mom1.fits',
#            'first_bullet':'spw-2-9.subimage.chans846-947.allmap.first_bullet.mom1.fits',
#            'second_bullet':'spw-2-9.subimage.chans990-1141.allmap.second_bullet.mom1.fits',
#            'third_bullet':'spw-2-9.subimage.chans1263-1478.allmap.third_bullet.mom1.fits',
#            'first_bullet_jet':'spw-2-9.subimage.chans1636~1722.allmap.first_bullet_jet.mom1.fits',
#            'second_bullet_jet':'spw-17-gb.224kHz.3840chans.contsub.image.chans2402-2503.allmap.second_bullet_jet.mom1.fits',
#            'tip_cavity':'spw-2-9.subimage.chans1554~1631.allmap.tip_cavity.mom1.fits',
#            'red_bb':'spw-2-9.subimage.chans258-650.allmap.red_bb.mom1.fits',
#            'red_jet':'spw-2-9.subimage.chans578-742.allmap.red_jet.mom1.fits'
#             }

pv_paths = {'EHVjet':'spw-17-gb.244kHz.3840chans.contsub.3width.0-3839chans.pv.fits',
           'EHVjet2':'spw-17-gb.244kHz.3840chans.contsub.5width.0-3839chans.pv.fits',
           'LV_IHV':'spw-17-gb.244kHz.3840chans.contsub.7width.0-3839chans.pv.fits',
           'LV_IHV_lsr':'spw-2-9-gb.contsub.lsr.5width.0-344chans.pv.fits'}

mom0_hdus = {mom0:fits.open('/home/{}/data/moments/{}'.format(user, mom0_paths[mom0]))[0] for mom0 in mom0_paths}
mom0_hdrs = {mom0:mom0_hdus[mom0].header for mom0 in mom0_paths}
mom0_wcss = {mom0:WCS(mom0_hdrs[mom0]).celestial for mom0 in mom0_paths}
mom0_datas = {mom0:mom0_hdus[mom0].data[0] for mom0 in mom0_paths}
for mom0_name in mom0_datas:
    mom0_datas[mom0_name][np.isnan(mom0_datas[mom0_name])] = 0


mom1_hdus = {mom1:fits.open('/home/{}/data/moments/{}'.format(user, mom1_paths[mom1]))[0] for mom1 in mom1_paths}
mom1_hdrs = {mom1:mom1_hdus[mom1].header for mom1 in mom1_paths}
mom1_wcss = {mom1:WCS(mom1_hdrs[mom1]).celestial for mom1 in mom1_paths}
mom1_datas = {mom1:mom1_hdus[mom1].data[0] for mom1 in mom1_paths}
for mom1 in mom1_datas:
#    mom1_datas[mom1][np.isnan(mom1_datas[mom1])] = 0
    mom1_datas[mom1] = np.abs(mom1_datas[mom1])

pv_hdus = {pv:fits.open('/home/{}/data/pv_diagrams/{}'.format(user, pv_paths[pv]))[0] for pv in pv_paths}
pv_hdrs = {pv:pv_hdus[pv].header for pv in pv_paths}
pv_wcss = {pv:WCS(pv_hdrs[pv]).celestial for pv in pv_paths}
pv_datas ={pv:pv_hdus[pv].data[0] for pv in pv_paths}

EHVjet_PV = {'start':[52.26580297, 31.26763918],
             'end':[52.26591933, 31.26715587]}
EHVjet_PV['PA'] = -np.arctan((EHVjet_PV['end'][1]-EHVjet_PV['start'][1])/ \
                             (EHVjet_PV['end'][0]-EHVjet_PV['start'][0])) * 180/np.pi + 90

EHVjet_PV2 = {'start':EHVjet_PV['end'],
             'end':[52.26698650, 31.26566336]}
EHVjet_PV2['PA'] = -np.arctan((EHVjet_PV2['end'][1]-EHVjet_PV2['start'][1])/ \
                             (EHVjet_PV2['end'][0]-EHVjet_PV2['start'][0])) * 180/np.pi + 90

#SkyCoord(ra='03h29m03.852s', dec='31d16m01.01s', frame='icrs')
#These are the coordinates of the end of the jet according to the ellipse fits
doff = 1.4
end_ra = 52.26605417 + doff*abs(mf.default_params['vla4b_deg'][0]-52.26605417)
end_dec = 31.26694722 - doff*abs(mf.default_params['vla4b_deg'][1]-31.26694722)
LV_IHV_PV = {'start':mf.default_params['vla4b_deg'],
            'end':[end_ra, end_dec]}
LV_IHV_PV['PA'] = -np.arctan((LV_IHV_PV['end'][1]-LV_IHV_PV['start'][1])/ \
                             (LV_IHV_PV['end'][0]-LV_IHV_PV['start'][0])) * 180/np.pi + 90


from photutils import EllipticalAperture
from photutils import SkyEllipticalAperture

angle_pvline = np.pi/180 * (160+90)
#angle_pvline = 10
y_pv = lambda xp, m, x0, y0: m * (xp-x0) + y0

vla4b_sky = SkyCoord(*mf.default_params['vla4b_deg'], unit='deg')
vla4b_pixel = skycoord_to_pixel(vla4b_sky,wcs_lsr)
aframe = vla4b_sky.skyoffset_frame()
vla4b_offset = vla4b_sky.transform_to(aframe)

box = [[630,850],[855,1165]]

x_first, y_first = box[1]
x_last, y_last = box[0]
xs_pixel = np.array([x for x in np.linspace(x_first, x_last, 100)])

ys_pixel = np.array([y_pv(x,
                    np.tan(angle_pvline),
                    vla4b_pixel[0],
                    vla4b_pixel[1],
                    ) for x in xs_pixel])

xys_sky = np.array([pixel_to_skycoord(x,y,wcs_lsr) for x,y in zip(xs_pixel, ys_pixel)])
xs_sky = np.array([xy.ra.deg for xy in xys_sky])
ys_sky = np.array([xy.dec.deg for xy in xys_sky])


def ellipse_points_calc(x0, y0, sma, eps, pa, n_points):
    """
    Cannot work in sky coordinates: it would render the wrong pixel,
    remember that distances in RA depends on the latitude. Use pixel
    coordinates instead, and convert later.
    """
    x = lambda theta: sma * np.cos(theta)
    y = lambda theta: sma * (1-eps) * np.sin(theta)
    thetas = np.linspace(0,2*np.pi,n_points)
    xs = np.array([x(theta) for theta in thetas])
    ys = np.array([y(theta) for theta in thetas])
    rot_coords = [main_functions.rot({'x':x, 'y':y, 'z':0}, 'z', pa)
                  for x,y in zip(xs,ys)]
    xs_rot = [rot_coord['x']+x0 for rot_coord in rot_coords]
    ys_rot = [rot_coord['y']+y0 for rot_coord in rot_coords]
    xys_sky = np.array([pixel_to_skycoord(_x,_y,wcs_lsr)
                        for _x,_y in zip(xs_rot,ys_rot)])
    xys_pixel = np.array([[x,y] for x,y in zip(xs_rot, ys_rot)])
    return xys_pixel, xys_sky


bbs = ['first_cavity', 'second_cavity']
xp_phi180 = {}
xp_phi0 = {}
for bb in bbs:
    xp_phi180[bb] = {}
    xp_phi0[bb] = {}
    for chan in dic_bb[bb]['data'].index:
        positions = SkyCoord(ra=dic_bb[bb]['data']['x0_RA'][chan]*u.deg,
                             dec=dic_bb[bb]['data']['y0_DEC'][chan]*u.deg)
        x0 = dic_bb[bb]['data']['x0'][chan]
        y0 = dic_bb[bb]['data']['y0'][chan]
        a = dic_bb[bb]['data']['sma'][chan]
        b = dic_bb[bb]['data']['sma'][chan]*(1.-dic_bb[bb]['data']['eps'][chan])
        eps = dic_bb[bb]['data']['eps'][chan]
        pa = dic_bb[bb]['data']['pa'][chan]
        ellipse_pixel, ellipse_sky = ellipse_points_calc(x0,
                                                         y0,
                                                         a,
                                                         eps,
                                                         pa,
                                                         50)
        ellipse_sky_cat = concatenate(ellipse_sky)
        xys_sky_cat = concatenate(xys_sky)

        idx, d2d, d3d = xys_sky_cat.match_to_catalog_3d(ellipse_sky_cat)

        ellipse_idx_closest = list(set(idx))
        idx_times_sorted = np.argsort([len([i for i in idx==p if i])
                                       for p in ellipse_idx_closest])
        p1_sky = ellipse_sky_cat[ellipse_idx_closest[idx_times_sorted[-1]]]
        p2_sky = ellipse_sky_cat[ellipse_idx_closest[idx_times_sorted[-2]]]

        x1_arcsec = p1_sky.separation(vla4b_sky).arcsec
        x2_arcsec = p2_sky.separation(vla4b_sky).arcsec

        xp_phi180[bb][chan] = np.min([x1_arcsec,x2_arcsec])
        xp_phi0[bb][chan] = np.max([x1_arcsec,x2_arcsec])


plt.figure(figsize=(8,8))
ax = plt.subplot(projection=wcs_lsr)

ax.plot(vla4b_sky.ra.deg,
        vla4b_sky.dec.deg,
        '*',
        transform=ax.get_transform('world'))

ax.plot(xs_sky,
        ys_sky,
        transform=ax.get_transform('world'),
       )


for p in [p1_sky, p2_sky]:
    ax.plot(p.ra.value,
            p.dec.value,
            '+k',
            zorder=3,
            transform=ax.get_transform('world'))

for point in ellipse_sky:
    ax.plot(point.ra.value,
            point.dec.value,
            '.r',
            transform=ax.get_transform('world')
           )


positions_deg = SkyCoord(ra=dic_bb[bb]['data']['x0_RA'][chan]*u.deg,
                     dec=dic_bb[bb]['data']['y0_DEC'][chan]*u.deg)
a_deg = dic_bb[bb]['data']['sma_deg'][chan]*u.deg
b_deg = dic_bb[bb]['data']['sma_deg'][chan]*(1.-dic_bb[bb]['data']['eps'][chan])*u.deg
pa_deg = (dic_bb[bb]['data']['pa'][chan] - np.pi/2.) * 180 / np.pi * u.deg
aper = SkyEllipticalAperture(positions_deg, a_deg, b_deg, pa_deg)
aper2plot = aper.to_pixel(wcs_lsr)
aper2plot.plot(ax,
               color='b',
               alpha=0.5)

ax.set_aspect("equal")


bb = 'second_cavity'
Rs = dic_bb[bb]['data']['mean_radius']
zs = dic_bb[bb]['data']['deprojected_dist_vla4b']

xps_phi180 = [xp_phi180[bb][chan] for chan in dic_bb[bb]['data'].index]
xps_phi0 = [xp_phi0[bb][chan] for chan in dic_bb[bb]['data'].index]
xps_phi90 = [dic_bb[bb]['data']['dist_vla4b'][chan] for chan in dic_bb[bb]['data'].index]
vzps_xp = [-dic_bb[bb]['data']['vel_rel'][chan] for chan in dic_bb[bb]['data'].index]

R_z_inter = interp1d(zs,
               Rs,
               kind='linear',
               fill_value="extrapolate",)

vzp_xp_phi180_inter = interp1d(xps_phi180,
                               vzps_xp,
                               kind='linear',
                               fill_value="extrapolate",)

vzp_xp_phi90_inter = interp1d(xps_phi90,
                             vzps_xp,
                             kind='linear',
                             fill_value="extrapolate",)

vzp_xp_phi0_inter = interp1d(xps_phi0,
                             vzps_xp,
                             kind='linear',
                             fill_value="extrapolate",)


R_z = lambda z: R_z_inter(z) if (z>=np.min(zs)) and (z<=np.max(zs)) else np.nan
vzp_xp_phi180 = lambda xp: vzp_xp_phi180_inter(xp) \
                           if (xp>=np.min(xps_phi180)) and (xp<=np.max(xps_phi180)) \
                           else np.nan

vzp_xp_phi0 = lambda xp: vzp_xp_phi0_inter(xp) \
                           if (xp>=np.min(xps_phi0)) and (xp<=np.max(xps_phi0)) \
                           else np.nan

vzp_xp_phi90 = lambda xp: vzp_xp_phi90_inter(xp) \
                           if (xp>=np.min(xps_phi90)) and (xp<=np.max(xps_phi90)) \
                           else np.nan


xp_phi_z = lambda z, i, phi: R_z(z) * np.cos(i) * np.cos(phi) + z * np.sin(i)
xp_phi_equal_0 = lambda z, xp, i, phi: xp - xp_phi_z(z,i,phi)
z_xp_phi = lambda xp, i, phi: optimize.brentq(
                                       lambda z: xp_phi_equal_0(z,xp,i,phi),
                                       np.min(zs),
                                       np.max(zs))
vzp_z_phi180 = lambda z, i: vzp_xp_phi180(xp_phi_z(z,i,np.pi))
vzp_z_phi0 = lambda z, i: vzp_xp_phi0(xp_phi_z(z,i,0))
vzp_z_phi90 = lambda z, i: vzp_xp_phi90(xp_phi_z(z,i,np.pi/2))

def v_z(z,i):
    return 1/np.sin(2*i) * np.sqrt(vzp_z_phi0(z,i)**2 + vzp_z_phi180(z,i)**2 \
                                   - 2*vzp_z_phi0(z,i)*vzp_z_phi180(z,i)*np.cos(2*i))

def v_z_2(z,i):
    return 1/np.sin(i) * np.sqrt((vzp_z_phi90(z,i)/np.cos(i))**2 \
                                 + vzp_z_phi0(z,i)**2 \
                                 - 2*vzp_z_phi90(z,i)*vzp_z_phi0(z,i))

def alpha(z,i):
    return np.arccos(vzp_z_phi0(z,i)/v_z(z,i)) - i

def alpha_1(z,i):
    return np.arccos(vzp_z_phi90(z,i)/(v_z_2(z,i)*np.cos(i)))

def alpha_2(z,i):
    return np.arccos(vzp_z_phi180(z,i)/v_z(z,i)) + i


alphamodel = models.ObsBasedAlpha(r=lambda z: R_z(z),
                              alpha=lambda z: alpha(z,np.pi/9),
                              v=lambda z: v_z(z,np.pi/9),
                              params={},)

p = bbpy.BuildModelInteractive(alphamodel,
                               box=box,
                               gaussian_filter=2,
                               thetas=np.linspace(np.min(zs),np.max(zs),300),
                               thetas_arrows=np.linspace(0,8,40),
                               theta_0=0,
                               theta_f=10,
                               phis=np.linspace(0, 2*np.pi, 100),
                               indep_variable='z',
                               v_range=[-93.3177-0.5291,88.7103],
                               arcsec_per_bin=0.05,
                               mark_velocities=[20,40,60,70],
                               change_header=True,
                               cube_name='i21',)


plt.show()

#p = bbpy.BuildModel(alphamodel,
#                    thetas=zs,
#                    phis=np.linspace(0,2*np.pi,100),
##                    thetas=np.linspace(0,8.59,200),
#                    thetas_arrows=np.linspace(0,8,40),
##                    phis=np.linspace(0, 2*np.pi, 300),
#                    cmap='jet')
#


#def remove_nans(self, positions=None, velocities=None):
#    positions = positions if positions is not None else p.positions
#    velocities = velocities if velocities is not None else p.velocities
#    id_nans = lambda p,cart: list(np.where(np.isnan([p[i][cart]
#                        for i in range(len(p))]) == True)[0])
#    ids2flat = [id_nans(positions,cart)+id_nans(velocities,cart)
#                    for cart in ['x','y','x']]
#    ids_nans = {i for i in np.array(ids2flat).flat}
#    positions_nonans = [pos for i,pos in enumerate(positions) if i not in ids_nans]
#    velocities_nonans = [vel for i,vel in enumerate(velocities) if i not in ids_nans]
#    return positions_nonans, velocities_nonans
#
#
#
#pos_range = [[-13.2598,13.6562],
#             [-9.72,9.51]]
#v_per_bin=0.5291
#arcsec_per_bin=0.01
#
#poss = p.positions
#vels = p.velocities
#poss, vels = remove_nans(poss, vels)
#
#cube_points = {cart: np.array([pos[cart] for pos in poss]) for cart in ['x','y','z']}
#
#cube_points['v'] = np.array([vel['z'] for vel in vels]) * (-1)**True
#
#v_range =[-93, 88]
##v_range = [-73,0]
#
#
#cube_range = [v_range] + pos_range
#
#v_n_bin = abs(v_range[1]-v_range[0]) / v_per_bin
#y_n_bin = abs(pos_range[0][1]-pos_range[0][0]) / arcsec_per_bin
#x_n_bin = abs(pos_range[1][1]-pos_range[1][0]) / arcsec_per_bin
#
#data, edges = np.histogramdd((np.array(cube_points['v'], dtype=np.float16),
#                             np.array(cube_points['y'], dtype=np.float16),
#                             np.array(cube_points['x'], dtype=np.float16)),
#                             bins=(int(v_n_bin),int(y_n_bin),int(x_n_bin)),
#                             range=cube_range,
#                             density=False,
#                             weights=None)
#
#print('123123')
#
#
