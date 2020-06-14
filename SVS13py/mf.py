from matplotlib import cm

import numpy as np

import os

from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants as const

import subprocess

process = subprocess.Popen(["whoami"], stdout=subprocess.PIPE)
result = process.communicate()[0]
user = result.decode('utf-8').rstrip('\n')

default_params = {'fig_size':8,
                 'origin':'lower',
                 'cmap':'viridis',
                 'vmin':None,
                 'vmax':0.06,
                 'v0':109.678,
                 'vf':-93.3969,
                 'scatter_size':10,
                 'scatter_alpha':0.2,
                 'colorbar_shrink':1,
                 'colorbar_aspect':20,
                 'colorbar_ngrid':0,
                 'colorbar_orientation':'vertical',
                 'colorbar_fraction':0.15,
                 'colorbar_pad':0.05,
                 'colorbar_anchor':(0.0,0.5),
                 'colorbar_panchor':(1.0,0.5),
                 'colorbar_show':True,
                 'colorbar_labelpad':10,
                 'colorbar_ticks':None,
                 'colorbar_format':'%.2f',
                 'colorbar_nticks':None,
                 'x_label':'x',
                 'y_label':'y',
                 'z_label':'z',
                 'textbox':None,
                 'textcolor':'w',
                 'boxstyle':'round',
                 'box_facecolor':'white',
                 'facecolor_background':'w',
                 'plot_vel':True,
                 'cbar_unit':'Jy/beam',
                 'render':'raster',
                 'contour_colors':'k',
                 'contour_levels':5,
                 'contour_linewidths':1,
                 'contour_sigma_filter':None,
                 'plot_cbar':True,
                 'x_box':0.05,
                 'y_box':0.95,
                 'box_fontsize':15,
                 'fraction':0.1,
                 'pad':0.1,
                 'save_fig':False,
                 'show_plot':True,
                 'save_format':'eps',
                 'interpolation':None,
                 'filterrad':4,
                 'norm':'linear',
                 'vcenter':0.05,
                 'linthresh':0.03,
                 'linscale':0.03,
                 'wcs':None,
                 'cube_aspect':'equal',
                 'add_stars':True,
                 'markerstar_size':10,
                 'markerstar_style':'+',
                 'markerstar_color':'w',
                 'markerstar_width':1,
                 'markerbluearm_style':'.',
                 'cbar_label_pad':20,
                 'cbar_pad':0.05,
                 'cbar_fraction':0.1,
                 'cbar_extend':'both',
                 'cbar_shrink':1,
                 'cbar_anchor':(0.0, 0.5),
                 'cbar_panchor':(1.0, 0.5),
                 'cbar_aspect':20,
                 'icrs_xlabel':'ICRS RA',
                 'icrs_ylabel':'ICRS DEC',
                 'pv_xlabel':'Offset (arcsec)',
                 'pv_ylabel':'LSRK radio velocity (km/s)',
                 'rotate_ticktext_yaxis':None,
                 'font_size':20,
                 'alpha_box':0.85,
                 'magical_factor':9.7,
                 'wspace':0.01,
                 'hspace':0.01,
                 'tick_spacing':1.5, #arcsec
                 'tick_direction':'in',
                 'grid_color':'w',
                 'label_color':'k',
                 'grid_linewidth':10,
                 'tick_width':1,
                 'tick_length':10,
                 'output_format':'eps',
                 'bbox_inches':'tight',
                 'path_save':'/home/{}/radio_astronomy/SVS13/paperplots/'.format(user),
                 'path_save_bb_characterization':'/home/{}/radio_astronomy/SVS13/bb_characterization/'.format(user),
                 'path_video':'/home/{}/radio_astronomy/SVS13/paperplots/video/'.format(user),
                 'path_bb_data':'/home/{}/radio_astronomy/SVS13/burbujas_data/'.format(user),
                 'path_database':'/home/{}/radio_astronomy/SVS13/databases/'.format(user),
#                 'path_fits':'/home/{}/data/SVS13_nocont.fits'.format(user),
                  'path_fits':'/home/{}/data/spw-2-9-gb.contsub.allchans.subimage.fits'.format(user),
#                 'path_fits':'/home/{}/data/spw-2-9-gb.contsub.hanning.lsr2.fits'.format(user),
                 'path_folder_points':'/home/{}/radio_astronomy/SVS13/regions_arms/spw-2-9-gb.contsub.lsr.'.format(user),
                 'use_tex':False,
                 'header':None,
                 'add_beam':False,
                 'beam_linewidth':1.5,
                 'beam_color':'w',
                 'beam_nax':0,
                 'xpos_beam':30,
                 'ypos_beam':30,
                 'add_scalebar':False,
                 'scalebar_distance':100,
                 'scalebar_fontsize':15,
                 'scalebar_width':1.,
                 'scalebar_color':'white',
                 'scalebar_loc':'lower right',
                 'scalebar_pad':1,
                 'scalebar_units':'au',
                 'scalebar_nax':0,
                 'scalebar_labeltop':False,
                 'scalebar_sep':5.,
                 'SVS13_distance':235, #pc
                 'SVS13_vsys':8.5,
                 'figtext_x_hor':0.4,
                 'figtext_x_vert':0.08,
                 'figtext_y_hor':0.04,
                 'figtext_y_vert':0.5,
                 'vla4a_arc':[0,0],
                 'vla4a_deg':[52.26560583, 31.26771778],
#SkyCoord(ra='03h29m3.7454s', dec='31d16m3.784s', frame='icrs')
                 'vla4b_arc':[-0.356,-0.007],
                 'vla4b_deg':[52.26570167, 31.26771556],
#SkyCoord(ra='03h29m3.7684s', dec='31d16m3.776s', frame='icrs')
                 'n_sigma':3,
                 'cmap_bg':'gray',
                 'cmap_contours':'coolwarm_r',
                 'cmap_nanized':'cool_r',
                 'cmap_ellipses':'viridis',
                 'color_length':300,
                 'plot_contours':True,
                 'contour_area':True,
                 'contour_alpha':0.25,
                 'contour_levels':0,
                 'theta1_pa':0,
                 'theta2_pa':360,
                 'show_slice_return':None,
                 'bb_centers':None,
                 'markercenter_color':'',
                 'markercenter_style':'*',
                 'markercenter_size':2,
                 'markercenter_width':0.1,
                 'ellipse_color':'r',
                 'markerellfit_color':'b',
                 'markerellfit_style':'.',
                 'markerellfit_size':0.5,
                 'limit_fit':{'+x0':30, '-x0':30, '+y0':30, '-y0':30,
                          '+sma':20, '-sma':20, '+eps':0.5, '-eps':0.5,
                          '+pa':2*np.pi, '-pa':2*np.pi},
                 'hodapp_zorder':6,
                 'stars_zorder':7,
                 'color_ellip_results':'r',
                 'linewidth_ellip_results':2,
                 'av_addjetdir':True,
                 'av_linecolor':'r',
                 'av_linewidth':2,
                 'av_linestyle':'solid',
                 'pv_vmax':0.08,
                 'vcenter_factor':1/3.,
                 'pvline_color':'b',
                 'pvline_width':1,
                 'pvline_style':'--',
                 'pvline_trans_color':'r',
                 'pvline_trans_width':1,
                 'pvline_trans_style':'--',
                 'pvtext_x':0.25,
                 'pvtext_y':-85,
                 'pv_jetline_x':None,
                 'pv_jetlinecolor':'r',
                 'pv_jetlinewidth':2,
                 'pv_jetlinestyle':'solid',
                 'twosided_plot':False,
                 'channels2plot':None,
                 'inverse_plot':False,
                 'plot_legend':False,
                 'n_points_ellipse':50,
                 'n_points_PV':50,}

def make_sequential_cm(ref_cm_name, n_seq=20, n_ref=20):
    ref_cm = cm.get_cmap(ref_cm_name,n_ref)(np.linspace(0,1,n_ref))
    seq_cmap = {}
    for i, color in enumerate(ref_cm):
        seq_cmap[i] = [list(color[:3]) + [alpha]
                       for alpha in list(np.linspace(0,1,n_seq))]
    return seq_cmap

def get_ak_bbcenters():
    nbb = [i+1 for i in range(4)]
    bb_pos_path = {i: default_params['path_bb_data']+'pos_deg_{}.dat'.format(i)
                   for i in nbb}
    bb_files = {i: open(bb_pos_path[i]) for i in nbb}
    make_list_float = lambda l: [float(x) for x in l]
    bb_pos = {i: np.array([make_list_float(line.split('\t')[1:3])
                           for k,line in enumerate(bb_files[i]) if k>0])
              for i in nbb}
    return bb_pos


def vel_from_header(header, vel_axis=''):
    if header['CTYPE3'] == 'FREQ':
        rest_freq = header['RESTFRQ'] * u.Hz
        hz_per_chan = header['CDELT3'] * u.Hz
        chan0_freq = header['CRVAL3'] * u.Hz
        vel_per_chan = (const.c / rest_freq * hz_per_chan).to(u.km/u.s)
        v0 = (const.c / rest_freq * (rest_freq-chan0_freq)).to(u.km/u.s)
        vf = (v0 - vel_per_chan * (header['NAXIS3']-1)).to(u.km/u.s)
        v_channels = np.linspace(v0.value, vf.value, header['NAXIS3'])
    elif header['CTYPE3'] == 'VRAD':
        vel_per_chan = np.abs(header['CDELT3']) * u.m / u.s
        v0 = header['CRVAL3'] * u.m / u.s
        vf = (v0 - vel_per_chan * (header['NAXIS3']-1)).to(u.m/u.s)
        v_channels = np.linspace(v0.to(u.km/u.s).value, vf.to(u.km/u.s).value, header['NAXIS3'])
    elif header['CTYPE2'] == 'FREQ':
        rest_freq = header['RESTFRQ'] * u.Hz
        hz_per_chan = header['CDELT2'] * u.Hz
        chan0_freq = header['CRVAL2'] * u.Hz
        vel_per_chan = (const.c / rest_freq * hz_per_chan).to(u.km/u.s)
        v0 = (const.c / rest_freq * (rest_freq-chan0_freq)).to(u.km/u.s)
        vf = (v0 - vel_per_chan * (header['NAXIS2']-1)).to(u.km/u.s)
        v_channels = np.linspace(v0.value, vf.value, header['NAXIS2'])
    else:
        rest_freq = header['RESTFRQ'] * u.Hz
        hz_per_chan = header['CDELT{}'.format(vel_axis)] * u.Hz
        chan0_freq = header['CRVAL{}'.format(vel_axis)] * u.Hz
        vel_per_chan = (const.c / rest_freq * hz_per_chan).to(u.km/u.s)
        v0 = (const.c / rest_freq * (rest_freq-chan0_freq)).to(u.km/u.s)
        vf = (v0 - vel_per_chan * (header['NAXIS{}'.format(vel_axis)]-1)).to(u.km/u.s)
        v_channels = np.linspace(v0.value, vf.value, header['NAXIS{}'.format(vel_axis)])
    return v_channels

def change_velrange_header(header_new, header_old, chan_0, vel_axis='2'):
    freq_offset =  header_old['CDELT{}'.format(vel_axis)]*chan_0
    header_new['CRVAL{}'.format(vel_axis)] = header_old['CRVAL{}'.format(vel_axis)] + freq_offset

hms_to_deg = lambda h, m, s: 360*(h/24+m/(24*60)+s/(24*3600) )

def deg_to_hms(deg):
    h = int(deg*(24/360))
    h_rest = deg*(24/360)-h
    m = int(h_rest*60)
    m_rest = h_rest*60-m
    s = m_rest*60
    return h, m, s
#SkyCoord(RA*u.deg, DEC*u.deg, frame='icrs')
#c = SkyCoord(ra='03h29m3.7454s', dec='31d16m3.784s', frame='icrs')
#p.ra.hms
#p.dec


def arcs2skycoord(arcs_pix, header):
    wcs = WCS(header).celestial
    arcs_sky = {arc:{} for arc in arcs_pix}
    for arc in arcs_pix:
        arcs_sky[arc]['x0'] = pixel_to_skycoord(arcs_pix[arc]['x0'], arcs_pix[arc]['y0'], wcs).ra
        arcs_sky[arc]['y0'] = pixel_to_skycoord(arcs_pix[arc]['x0'], arcs_pix[arc]['y0'], wcs).dec
        arcs_sky[arc]['width'] = arcs_pix[arc]['width'] * header['CDELT2'] * u.deg
        arcs_sky[arc]['height'] = arcs_pix[arc]['height'] * header['CDELT2'] * u.deg
        arcs_sky[arc]['angle'] = arcs_pix[arc]['angle']
        arcs_sky[arc]['theta1'] = arcs_pix[arc]['theta1']
        arcs_sky[arc]['theta2'] = arcs_pix[arc]['theta2']
    return arcs_sky

def arcs2pix(arcs_sky, header):
    wcs = WCS(header).celestial
    arcs_pix = {arc:{} for arc in arcs_sky}
    for arc in arcs_sky:
        arcs_pix[arc]['x0'] = skycoord_to_pixel(SkyCoord(arcs_sky[arc]['x0'],arcs_sky[arc]['y0']), wcs)[0]
        arcs_pix[arc]['y0'] = skycoord_to_pixel(SkyCoord(arcs_sky[arc]['x0'],arcs_sky[arc]['y0']), wcs)[1]
        arcs_pix[arc]['width'] = (arcs_sky[arc]['width'] / (header['CDELT2'] * u.deg)).value
        arcs_pix[arc]['height'] = (arcs_sky[arc]['height'] / (header['CDELT2'] * u.deg)).value
        params_arc = ['angle', 'theta1', 'theta2']
        for param in params_arc:
            arcs_pix[arc][param] = arcs_sky[arc][param] if param in arcs_sky[arc] else None
            arcs_pix[arc][param] = arcs_sky[arc][param] if param in arcs_sky[arc] else None
            arcs_pix[arc][param] = arcs_sky[arc][param] if param in arcs_sky[arc] else None
    return arcs_pix


def get_nmax(data, n):
    data_nomax = np.copy(data)
    maxes = []
    for i in range(n):
        maxes.append(np.max(data_nomax))
        data_nomax[np.argmax(data_nomax)] = 0
    return maxes

def dbugg():
    pass
