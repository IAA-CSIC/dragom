import numpy as np

from IPython.display import display, clear_output

from scipy.ndimage import gaussian_filter
from scipy import optimize

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import Colorbar
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import to_hex
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from itertools import product

from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord

from photutils.isophote import EllipseGeometry
from photutils.isophote import Ellipse
from photutils.isophote import build_ellipse_model
from photutils import EllipticalAperture, EllipticalAnnulus, CircularAperture

import pandas as pd

import os

import casa_proc

import shelve

import SVS13py.mf as mf
from SVS13py.ellipsefitting import ellipse_fitting


def casa_immoments(operation, central_chans, width_chan, **kwargs):
    """
    moments=-1
    moments=0
    moments=1
    moments=2
    moments=3
    moments=4
    moments=5
    moments=6
    moments=7
    moments=8
    moments=9
    moments=10
    moments=11
    - mean value of the spectrum
    - integrated value of the spectrum
    - intensity weighted coordinate; traditionally used to get
    ’velocity fields’
    - intensity weighted dispersion of the coordinate; traditionally
    used to get ’velocity dispersion’
    - median of I
    - median coordinate
    - standard deviation about the mean of the spectrum
    - root mean square of the spectrum
    - absolute mean deviation of the spectrum
    - maximum value of the spectrum
    - coordinate of the maximum value of the spectrum
    - minimum value of the spectrum
    - coordinate of the minimum value of the spectrum
    """

    for param in mf.default_params:
        kwargs[param] = kwargs[param] \
                if param in kwargs \
                else mf.default_params[param]

    str_chans = '{}~{}'.format(central_chans-width_chan,
                               central_chans+width_chan)
    save_image = '/home/gblazquez/data/{}-{}-im{}'.format(central_chans,
                                                          width_chan,
                                                          operation)
    save_fits =  save_image+'.fits'
    try:
        os.system('rm -r {}'.format(save_image))
    except:
        pass
    casa_proc.casa_task('immoments',
                        imagename=kwargs['path_fits'],
                        chans=str_chans,
                        moments=[operation],
                        outfile=save_image)
#                       region=kwargs['cp_region'],
#                       box=kwargs['cp_box'],
#                       chans=kwargs['cp_chans'],
#                       stokes=kwargs['cp_stokes'],
#                       mask=kwargs['cp_mask'],
#                       includepix=-1,
#                       excludepix=-1,
#                       )
    casa_proc.casa_task('exportfits',
                        imagename=save_image,
                        fitsimage=save_fits,
                        overwrite=True)

    hdu = fits.open(save_fits)[0]
    hdr = hdu.header
    wcs = WCS(hdu.header).celestial
    collapsed_image = hdu.data[0]

    return collapsed_image[0], wcs, hdr

def open_fits(fits_path):
    hdu = fits.open(fits_path)[0]
    hdr = hdu.header
    wcs = WCS(hdu.header).celestial
    data = hdu.data[0]
    return data, hdr, wcs


def casa_exportfits(im_path, return_image=True, overwrite=True):
    fits_path = im_path + '.fits'

    casa_proc.casa_task('exportfits',
                        imagename=im_path,
                        fitsimage=fits_path,
                        overwrite=True)

    if return_image:
        return open_fits(fits_path)


def casa_impv(im_path,
              start,
              end,
              width,
              chans,
              mode='coords',
              pa=None,
              unit='arcsec',
              region='',
              stokes='',
              mask='',
              overwrite=False,
              output_name=''):
    """imagename           =         ''        #  Name of the input image
    outfile             =         ''        #  Output image name. If empty, no image
                                            #   is written.
    mode                =   'coords'        #  If "coords", use start and end
                                            #   values. If "length", use center,
                                            #   length, and pa values.
         start          =         ''        #  The starting pixel in the direction
                                            #   plane (array of two values).
         end            =         ''        #  The ending pixel in the direction
                                            #   plane (array of two values).

    width               =          1        #  Width of slice for averaging pixels
                                            #   perpendicular to the slice. Must be
                                            #   an odd positive integer or valid
                                            #   quantity. See help for details.
    unit                =   'arcsec'        #  Unit for the offset axis in the
                                            #   resulting image. Must be a unit of
                                            #   angular measure.
    chans               =         ''        #  Channels to use.  Channels must be
                                            #   contiguous. Default is to use all
                                            #   channels.
         region         =         ''        #  Region selection. Default is entire
                                            #   image. No selection is permitted in
                                            #   the direction plane.

    stokes              =         ''        #  Stokes planes to use. Planes must be
                                            #   contiguous. Default is to use all
                                            #   stokes.
    mask                =         ''        #  Mask to use. Default is none.
    """

    im_path_save = im_path.rstrip('.fits') + \
            '{}.{}width.{}-{}chans.pv'.format(output_name, width, *chans)

    if overwrite:
        try:
            os.system('rm -r {}'.format(im_path_save))
        except:
            pass

    if mode=='coords':
        casa_proc.casa_task('impv',
                            imagename=im_path,
                            outfile=im_path_save,
                            mode=mode,
                            start=['{}deg'.format(start[0]),
                                   '{}deg'.format(start[1])],
                            end=['{}deg'.format(end[0]),
                                 '{}deg'.format(end[1])],
                            width=width,
                            unit=unit,
                            chans='{}~{}'.format(*chans),
                            region=region,
                            stokes=stokes,
                            mask=mask)
    elif mode=='length':
        pa = 0 if pa is None else pa
        casa_proc.casa_task('impv',
                           imagename=im_path,
                           outfile=im_path_save,
                           mode=mode,
                           center=['{}deg'.format(start[0]),
                                   '{}deg'.format(start[1])],
                           length='{}arcsec'.format(end),
                           pa='{}deg'.format(pa),
                           width=width,
                           unit=unit,
                           chans='{}~{}'.format(*chans),
                           region=region,
                           stokes=stokes,
                           mask=mask)
    pv_diagram, hdr, wcs = casa_exportfits(im_path_save)
    return pv_diagram, wcs, hdr



def collapse_chans(image_cube, operation, central_chans, width_chan, box):
    """
    Combines different channels through an operation (max_pixel, moments, median, etc) to obtain one single image
    central_chan can be an integer specifying the central channel, or a list of different channels to collapse.
    """

    if type(central_chans)==list or type(central_chans)==np.ndarray:
        lists_channels = [np.arange(central_chan-width_chan,central_chan+width_chan) for central_chan in central_chans] \
                        if width_chan!=0 else None
        channels = [channel for list_channels in lists_channels for channel in list_channels] if width_chan!=0 else central_chans
    else:
        channels = np.arange(central_chans-width_chan, central_chans+width_chan) if width_chan!=0 else [central_chans]

    collapsed_array = np.zeros_like(image_cube[0,:,:])
    nx, ny = np.arange(box[0][1], box[1][1]),np.arange(box[0][0], box[1][0])
    if operation == 'max':
        for i,j in product(nx, ny):
            collapsed_array[i,j] = np.max([image_cube[:,i,j][channels]])
    elif operation == 'sum_slow':
        for i,j in product(nx, ny):
            collapsed_array[i,j] = np.sum(image_cube[:,i,j][channels])
    elif operation == 'sum':
        collapsed_array[box[0][1]:box[1][1],box[0][0]:box[1][0]] = \
        sum([image_cube[chan][box[0][1]:box[1][1],box[0][0]:box[1][0]] for chan in channels])
    elif operation == 'mom0_slow':
        for i,j in product(nx, ny):
            collapsed_array[i,j] = np.mean(image_cube[:,i,j][channels])
    elif operation == 'mom0':
        collapsed_array[box[0][1]:box[1][1],box[0][0]:box[1][0]] = \
        sum([image_cube[chan][box[0][1]:box[1][1],box[0][0]:box[1][0]] for chan in channels]) / len(channels)
    elif operation == 'median':
        for i,j in product(nx, ny):
            collapsed_array[i,j] = np.median(image_cube[:,i,j][channels])
    else:
        pass

    return collapsed_array


def create_cube(image_cube,
                operation,
                channels,
                width_chan,
                box,
                output_name=None,
                header=None,
                overwrite=True,
               ):
    channels = channels if channels != 'all' else np.arange(len(image_cube))
#    new_cube = np.copy(image_cube[channels])
    nchan = len(channels)
    nx, ny, nz = np.shape(image_cube)
    new_cube = np.empty([nchan, ny, nz])
    for i, channel in enumerate(channels):
        new_cube[i] = collapse_chans(image_cube, operation, channel, width_chan, box)
        clear_output()
        print('{:.1f} %'.format(i/nchan*100.))

    if output_name is not None:
        fits.writeto('/home/gblazquez/data/{}.fits'.format(output_name), new_cube, header=header, overwrite=overwrite)
    return new_cube


def show_slice_cube(image_cube, channel, width_chan=0, cbax=None,
                    operation=None, box=None, return_ax=False,
                    return_im=False, output_name=None, ax=None, pv=False,
                    results_fit=None, **kwargs):
    """
    Makes a plot for a given spectral slice of the cube
    box = [[x_1,y_1],[x_2,y_2]]
    """

    for param in mf.default_params:
        kwargs[param] = kwargs[param] \
                if param in kwargs \
                else mf.default_params[param]
    im_str = image_cube if type(image_cube)==str else 'None'

    matplotlib.rcParams.update({'font.size': kwargs['font_size']})     #changes the size of labels
    plt.rc('text', usetex=kwargs['use_tex'])                       #set False to unable latex rendering

    if (im_str != 'casa_image') and (not pv):
        v_channels = mf.vel_from_header(kwargs['header'])
        image = collapse_chans(image_cube, operation, channel, width_chan, box) if operation is not None else image_cube[channel]
        extent = None
    elif (im_str != 'casa_image') and (not pv):
        v_channels = np.linspace(kwargs['v0'], kwargs['vf'], 1920)
        image, kwargs['wcs'], kwargs['header'] = casa_immoments(int(operation.split('_')[1]), channel, width_chan)
        extent = None
    else:
        v_channels = mf.vel_from_header(kwargs['header'])
        offsets = [0,kwargs['header']['CDELT1']*kwargs['header']['NAXIS1']]
        extent = [offsets[0],offsets[1],v_channels[0],v_channels[-1]]
        image = image_cube

    vel_str = '{:.2f} km/s'.format(v_channels[channel])

    if (kwargs['wcs'] is not None) and (ax is None):
        fig = plt.figure(figsize=(kwargs['fig_size'], kwargs['fig_size']))
        ax = plt.subplot(projection=kwargs['wcs'])
    elif (kwargs['wcs'] is not None) and (ax is not None):
        pass
    elif (kwargs['wcs'] is None) and (ax is not None):
        pass
    else:
        fig, ax = plt.subplots(figsize=(kwargs['fig_size'],kwargs['fig_size']))

    if kwargs['rotate_ticktext_yaxis'] is not None:
        ax.coords[1].set_ticklabel(rotation=kwargs['rotate_ticktext_yaxis'])

    if kwargs['norm'] == 'linear':
        norm = colors.Normalize(vmax=kwargs['vmax'], vmin=kwargs['vmin'])
    elif kwargs['norm'] == 'log':
        norm = colors.LogNorm(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
    elif kwargs['norm'] == 'symlog':
        norm = colors.SymLogNorm(colors.SymLogNorm(linthresh=kwargs['linthresh'],
                                                   linscale=kwargs['linscale'],
                                                   vmin=kwargs['vmin'],
                                                   vmax=kwargs['vmax']))
    elif kwargs['norm'] == 'divnorm':
        norm = colors.DivergingNorm(vmin=kwargs['vmin'],
                                    vcenter=kwargs['vcenter'],
                                    vmax=kwargs['vmax'])

    if kwargs['contour_sigma_filter'] is not None:
        image_contour = gaussian_filter(image,
                                        sigma=kwargs['contour_sigma_filter'])
    else:
        image_contour = image


    if kwargs['render'] == 'raster':
        im = ax.imshow(image,
                       origin=kwargs['origin'],
                       aspect=kwargs['cube_aspect'],
                       cmap=kwargs['cmap'],
                       norm=norm,
                       extent=extent,
                       interpolation=kwargs['interpolation'],
                       filterrad=kwargs['filterrad'])
    #                    vmin=kwargs['vmin'], vmax=kwargs['vmax'])
    elif kwargs['render'] == 'contours':
        im = ax.contour(image_contour,
                        levels=kwargs['contour_levels'],
                        linewidths=kwargs['contour_linewidths'],
                        origin=kwargs['origin'],
                        extent=extent,
                        colors=kwargs['contour_colors'],
                        norm=norm,)
    else:
        im = ax.imshow(image,
                       origin=kwargs['origin'],
                       aspect=kwargs['cube_aspect'],
                       cmap=kwargs['cmap'],
                       norm=norm,
                       extent=extent,
                       interpolation=kwargs['interpolation'],
                       filterrad=kwargs['filterrad'])
        ax.contour(image_contour,
                   levels=kwargs['contour_levels'],
                   linewidths=kwargs['contour_linewidths'],
                   origin=kwargs['origin'],
                   extent=extent,
                   colors=kwargs['contour_colors'],
                   norm=norm,)
    if pv:
        xaxis = ax.get_xaxis()
        yaxis = ax.get_yaxis()
    else:
        xaxis = ax.coords[0]
        yaxis = ax.coords[1]
        xaxis.set_ticks(spacing=kwargs['tick_spacing'] * u.arcsec)
        xaxis.set_ticklabel(exclude_overlapping=True)
        ax.tick_params(labelsize=kwargs['font_size'])
        props = dict(boxstyle=kwargs['boxstyle'],
                  facecolor=kwargs['box_facecolor'],
                  alpha=kwargs['alpha_box']) if kwargs['textbox'] else None

    if kwargs['plot_vel']:
        ax.text(kwargs['x_box'],
                kwargs['y_box'],
                vel_str,
                fontsize=kwargs['box_fontsize'],
                color=kwargs['textcolor'],
                verticalalignment='top',
                transform=ax.transAxes,
                bbox=props)

    if kwargs['plot_cbar'] and (cbax is None):
        cbar = plt.colorbar(im,
                            ax=ax,
                            orientation=kwargs['colorbar_orientation'],
                            fraction=kwargs['colorbar_fraction'],
                            pad=kwargs['colorbar_pad'],
                            shrink=kwargs['colorbar_shrink'],
                            aspect=kwargs['colorbar_aspect'],
                            anchor=kwargs['colorbar_anchor'],
                            panchor=kwargs['colorbar_panchor'],
                            extend=kwargs['cbar_extend'])
        if kwargs['colorbar_orientation'] == 'vertical':
            cbar.ax.set_ylabel(kwargs['cbar_unit'])
        elif kwargs['colorbar_orientation'] == 'horizontal':
            cbar.ax.set_xlabel(kwargs['cbar_unit'])

    elif kwargs['plot_cbar'] and (cbax is not None):
        cbar = plt.colorbar(im,
                            ax=ax,
                            cmap=kwargs['cmap_ellipses'],
                            cax=cbax,
                            orientation=kwargs['colorbar_orientation'],
                            fraction=kwargs['colorbar_fraction'],
                            pad=kwargs['colorbar_pad'],
                            shrink=kwargs['colorbar_shrink'],
                            aspect=kwargs['colorbar_aspect'],
                            anchor=kwargs['colorbar_anchor'],
                            panchor=kwargs['colorbar_panchor'],
                            extend=kwargs['cbar_extend'],
                            format=kwargs['colorbar_format'])
        cbar_ticklabels = ['-{:.0f}'.format(i) for i in cbar.ax.get_yticks()]
        cbar.ax.set_yticklabels(cbar_ticklabels)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(direction='out', color='k', pad=2.)
        cbar.ax.set_xlabel(kwargs['cbar_unit'],
                           labelpad=kwargs['colorbar_labelpad'])
        if kwargs['colorbar_nticks'] is not None:
            cbar.ax.locator_params(nbins=kwargs['colorbar_nticks'])

    if box is not None:
        ax.set_xlim([box[0][0], box[1][0]])
        ax.set_ylim([box[0][1], box[1][1]])

    if kwargs['add_beam']:
        pa = kwargs['header']['BPA'] * np.pi/180. + np.pi/2   #in radians
        a = kwargs['header']['BMAJ'] / kwargs['header']['CDELT2']   #semi-major axis in pixels
        b = kwargs['header']['BMIN'] / kwargs['header']['CDELT2']   #semi-minor axis in pixels
        xpos = (box[0][0] + kwargs['xpos_beam']) if box is not None else kwargs['xpos_beam']
        ypos = (box[0][1] + kwargs['ypos_beam']) if box is not None else kwargs['ypos_beam']

        geometry = EllipseGeometry(x0=xpos, y0=ypos, sma=b, eps=b/a, pa=pa)
        aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                   geometry.sma*(1 - geometry.eps),
                                   geometry.pa)
        aper.plot(ax, color=kwargs['beam_color'], linewidth=kwargs['beam_linewidth'],)

    if kwargs['add_scalebar']:
        fontprops = fm.FontProperties(size=kwargs['scalebar_fontsize'])
        arcsec_per_pix = kwargs['header']['CDELT1'] * 3600
        scalebar_distance = (kwargs['scalebar_distance']/arcsec_per_pix) \
                * (kwargs['scalebar_units']=='arcsec') \
                + (kwargs['scalebar_distance']/kwargs['SVS13_distance']) \
                / arcsec_per_pix * (kwargs['scalebar_units']=='au')
        scalebar = AnchoredSizeBar(ax.transData,
                           scalebar_distance,
                           str(kwargs['scalebar_distance'])\
                                   +' '+kwargs['scalebar_units'],
                           kwargs['scalebar_loc'],
                           pad=kwargs['scalebar_pad'],
                           color=kwargs['scalebar_color'],
                           frameon=False,
                           size_vertical=kwargs['scalebar_width'],
                           label_top=kwargs['scalebar_labeltop'],
                           sep=kwargs['scalebar_sep'],
                           fontproperties=fontprops)

        ax.add_artist(scalebar)

    if kwargs['add_stars'] and (kwargs['wcs'] is not None):
        xs_star = kwargs['vla4a_deg'][0], kwargs['vla4b_deg'][0]
        ys_star = kwargs['vla4a_deg'][1], kwargs['vla4b_deg'][1]
        ax.plot(xs_star,
                ys_star,
                marker=kwargs['markerstar_style'],
                color=kwargs['markerstar_color'],
                linestyle='',
                transform=ax.get_transform('icrs'),
                markersize=kwargs['markerstar_size'],
                mew=kwargs['markerstar_width'])
    else:
        print('You must include wcs information if add_stars is desired')

    if results_fit is not None:
        bb_types_with_chan = [bb_type for bb_type in results_fit if channel in results_fit[bb_type].index]
        beam_size_pix = (kwargs['header']['BMAJ']+kwargs['header']['BMIN']) * 0.5 / \
        kwargs['header']['CDELT2']
        for bb_type in bb_types_with_chan:
            aper = EllipticalAnnulus((results_fit[bb_type].x0[channel],
                                      results_fit[bb_type].y0[channel]),
                                     results_fit[bb_type].sma[channel]-beam_size_pix/2.,
                                     results_fit[bb_type].sma[channel]+beam_size_pix/2.,
                                     (results_fit[bb_type].sma[channel]+beam_size_pix/2.) * \
                                     (1. - results_fit[bb_type].eps[channel]),
                                     results_fit[bb_type].pa[channel])
            aper.plot(ax, color=kwargs['color_ellip_results'])


    if pv:
        ax.set_xlabel(kwargs['pv_xlabel'])
        ax.set_ylabel(kwargs['pv_ylabel'])

    else:
        ax.set_xlabel(kwargs['icrs_xlabel'])
        ax.set_ylabel(kwargs['icrs_ylabel'])

    if return_ax:
        return image, ax

    if return_im:
        return image

    if kwargs['show_slice_return'] == 'contour_imax':
        return im, ax
    else:
        pass

    if output_name is not None:
        fig.savefig('{}{}.{}'.format(kwargs['path_save'], output_name, kwargs['output_format']),
                    bbox_inches=kwargs['bbox_inches'], )

    if kwargs['show_plot'] is not True:
        plt.close()



def mosaic_slices(image_cube, nrow, ncol, ngrid, chan_0, chan_f, wcs, box,
                  operation=None, width_chan=0, output_name=None,
                  ellip_dic=None, **kwargs):
    """
    Makes a mosaic of spectral slices of the cube
    box = [[x_1,y_1],[x_2,y_2]]
    """

    for param in mf.default_params:
        kwargs[param] = kwargs[param] \
                if param in kwargs \
                else mf.default_params[param]
#   changes the size of labels
    matplotlib.rcParams.update({'font.size': kwargs['font_size']})
#   set False to unable latex rendering
    plt.rc('text', usetex=kwargs['use_tex'])
#   plt.rc('font', family='serif')

    fig = plt.figure(figsize=(nrow*kwargs['magical_factor'], (ncol+1)*ngrid,))

    gs1 = GridSpec(nrow, (ncol+1)*ngrid, )
    gs1.update(wspace=kwargs['wspace'], hspace=kwargs['hspace'],)

    n = 0
    ims = {}
    for i, j in product(range(nrow), [i for i in range(ncol*ngrid)][::ngrid]):
        ims[n] = plt.subplot(gs1[i, j:j+ngrid], projection=wcs)
        ims[n].set_xlim([box[0][0], box[1][0]])
        ims[n].set_ylim([box[0][1], box[1][1]])

        xaxis = ims[n].coords[0]
        yaxis = ims[n].coords[1]
        xaxis.tick_params(direction=kwargs['tick_direction'],
                          grid_color=kwargs['grid_color'],
                          colors=kwargs['grid_color'],
                          labelcolor=kwargs['label_color'],
                          color=kwargs['grid_color'],
                          grid_linewidth=kwargs['grid_linewidth'],
                          width=kwargs['tick_width'],
                          length=kwargs['tick_length'])

        yaxis.tick_params(direction=kwargs['tick_direction'],
                          grid_color=kwargs['grid_color'],
                          colors=kwargs['grid_color'],
                          labelcolor=kwargs['label_color'],
                          color=kwargs['grid_color'],
                          grid_linewidth=kwargs['grid_linewidth'],
                          width=kwargs['tick_width'],
                          length=kwargs['tick_length'])

        xaxis.set_ticks(spacing=kwargs['tick_spacing'] * u.arcsec)
        xaxis.set_ticklabel(exclude_overlapping=True)
        ims[n].set_xlabel(' ')
        ims[n].set_ylabel(' ')

        if kwargs['rotate_ticktext_yaxis'] is not None:
            yaxis.set_ticklabel(rotation=kwargs['rotate_ticktext_yaxis'])

        if (j > 0) and (i < nrow-1):
            xaxis.set_ticklabel_visible(False)
            yaxis.set_ticklabel_visible(False)
        if (i == (nrow-1)) and (j > 0):
            yaxis.set_ticklabel_visible(False)
        if (i < (nrow-1)) and (j == 0):
            xaxis.set_ticklabel_visible(False)
        n += 1

    ax_cbar = plt.subplot(gs1[:, -ngrid+kwargs['colorbar_ngrid']])

    imax = {}
    nchannels = len(image_cube)
    channels = np.arange(0, nchannels)
    v_channels = mf.vel_from_header(kwargs['header'])
    props_dict = dict(boxstyle=kwargs['boxstyle'],
                      facecolor=kwargs['box_facecolor'],
                      alpha=kwargs['alpha_box'])
    props = props_dict if kwargs['textbox'] else None

    if kwargs['channels2plot'] is not None:
        channels2plot = kwargs['channels2plot']
    else:
        channels2plot = np.array([int(round(i)) for i in
                                  np.linspace(chan_0, chan_f, nrow*ncol)])

    for n, channel in enumerate(channels2plot):
        if kwargs['norm'] == 'linear':
            norm = colors.Normalize(vmax=kwargs['vmax'], vmin=kwargs['vmin'])
        elif kwargs['norm'] == 'log':
            norm = colors.LogNorm(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
        elif kwargs['norm'] == 'symlog':
            norm = colors.SymLogNorm(linthresh=kwargs['linthresh'],
                                     linscale=kwargs['linscale'],
                                     vmin=kwargs['vmin'],
                                     vmax=kwargs['vmax'])
        elif kwargs['norm'] == 'divnorm':
            norm = colors.DivergingNorm(vmin=kwargs['vmin'],
                                        vcenter=kwargs['vcenter'],
                                        vmax=kwargs['vmax'])

        image = collapse_chans(image_cube, operation, channel,
                               width_chan, box) \
            if operation is not None else image_cube[channel]
        imax[n] = ims[n].imshow(image,
                                origin=kwargs['origin'],
                                cmap=kwargs['cmap'],
                                aspect=kwargs['cube_aspect'],
                                norm=norm,
                                interpolation=kwargs['interpolation'])
        ims[n].tick_params(labelsize=kwargs['font_size'])

        vel_str = '{:.2f} km/s'.format(v_channels[channel])
        ims[n].text(kwargs['x_box'],
                    kwargs['y_box'],
                    vel_str,
                    fontsize=kwargs['box_fontsize'],
                    color=kwargs['textcolor'],
                    verticalalignment='top',
                    transform=ims[n].transAxes,
                    bbox=props)

        if kwargs['add_stars']:
            xs_star = kwargs['vla4a_deg'][0], kwargs['vla4b_deg'][0]
            ys_star = kwargs['vla4a_deg'][1], kwargs['vla4b_deg'][1]
            ims[n].plot(xs_star,
                        ys_star,
                        marker=kwargs['markerstar_style'],
                        color=kwargs['markerstar_color'],
                        linestyle='',
                        transform=ims[n].get_transform('icrs'),
                        markersize=kwargs['markerstar_size'],
                        mew=kwargs['markerstar_width'])
        if ellip_dic is not None:
            bb_types_with_chan = [bb_type for bb_type in ellip_dic if channel
                                  in ellip_dic[bb_type]['data'].index]
            beam_size_pix = (kwargs['header']['BMAJ']
                             + kwargs['header']['BMIN']) \
                * 0.5 / kwargs['header']['CDELT2']

            for bb_type in bb_types_with_chan:
                if ellip_dic[bb_type]['plot_ellipses'] == 'annulus':
                    aper = EllipticalAnnulus((
                        ellip_dic[bb_type]['data'].x0[channel],
                        ellip_dic[bb_type]['data'].y0[channel]),
                        ellip_dic[bb_type]['data'].sma[channel]
                        - beam_size_pix / 2.,
                        ellip_dic[bb_type]['data'].sma[channel]
                        + beam_size_pix / 2.,
                        (ellip_dic[bb_type]['data'].sma[channel]
                            + beam_size_pix / 2.)
                        * (1. - ellip_dic[bb_type]['data'].eps[channel]),
                        ellip_dic[bb_type]['data'].pa[channel])

                elif ellip_dic[bb_type]['plot_ellipses'] == 'ellipse':
                    aper = EllipticalAperture((
                        ellip_dic[bb_type]['data'].x0[channel],
                        ellip_dic[bb_type]['data'].y0[channel]),
                        ellip_dic[bb_type]['data'].sma[channel],
                        ellip_dic[bb_type]['data'].sma[channel]
                        * (1. - ellip_dic[bb_type]['data'].eps[channel]),
                        ellip_dic[bb_type]['data'].pa[channel])

                elif ellip_dic[bb_type]['plot_ellipses'] == \
                        'ellipse_beam_upper':
                    aper = EllipticalAperture((
                        ellip_dic[bb_type]['data'].x0[channel],
                        ellip_dic[bb_type]['data'].y0[channel]),
                        ellip_dic[bb_type]['data'].sma[channel]
                        + beam_size_pix / 2.,
                        (ellip_dic[bb_type]['data'].sma[channel]
                            + beam_size_pix / 2.)
                        * (1. - ellip_dic[bb_type]['data'].eps[channel]),
                        ellip_dic[bb_type]['data'].pa[channel])

                aper.plot(ims[n],
                          color=ellip_dic[bb_type]['color'],
                          linewidth=ellip_dic[bb_type]['linewidth'],
                          linestyle=ellip_dic[bb_type]['linestyle'])

    cbar = plt.colorbar(imax[0],
                        cax=ax_cbar,
                        extend=kwargs['cbar_extend'],
                        pad=kwargs['cbar_pad'],
                        fraction=kwargs['cbar_fraction'],
                        shrink=kwargs['cbar_shrink'],
                        panchor=kwargs['cbar_panchor'],
                        anchor=kwargs['cbar_anchor'],
                        aspect=kwargs['cbar_aspect'],)
    ax_cbar.tick_params(labelsize=kwargs['font_size'])

    cbar.set_label(r'Jy/beam', labelpad=kwargs['cbar_label_pad'],
                   fontsize=kwargs['font_size'])
    plt.figtext(kwargs['figtext_x_hor'], kwargs['figtext_x_vert'],
                'R.A. (ICRS)')
    plt.figtext(kwargs['figtext_y_hor'], kwargs['figtext_y_vert'],
                'Declination (ICRS)', rotation='vertical')

    if kwargs['add_beam']:
        #  pa in radians
        pa = kwargs['header']['BPA'] * np.pi/180. + np.pi / 2
        #  semi-major axis in pixels
        a = kwargs['header']['BMAJ'] / kwargs['header']['CDELT2']
        #  semi-minor axis in pixels
        a = kwargs['header']['BMAJ'] / kwargs['header']['CDELT2']
        b = kwargs['header']['BMIN'] / kwargs['header']['CDELT2']
        xpos = (box[0][0] + kwargs['xpos_beam']) \
            if box is not None else kwargs['xpos_beam']
        ypos = (box[0][1] + kwargs['ypos_beam']) \
            if box is not None else kwargs['ypos_beam']

        geometry = EllipseGeometry(x0=xpos, y0=ypos, sma=b, eps=b/a, pa=pa)
        aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                  geometry.sma*(1 - geometry.eps),
                                  geometry.pa)
        aper.plot(ims[kwargs['beam_nax']],
                  color=kwargs['beam_color'],
                  linewidth=kwargs['beam_linewidth'])

    if kwargs['add_scalebar']:
        fontprops = fm.FontProperties(size=kwargs['scalebar_fontsize'])
        arcsec_per_pix = kwargs['header']['CDELT1'] * 3600
        scalebar_distance = (kwargs['scalebar_distance']/arcsec_per_pix) \
            * (kwargs['scalebar_units'] == 'arcsec') \
            + (kwargs['scalebar_distance']/kwargs['SVS13_distance']) \
            / arcsec_per_pix * (kwargs['scalebar_units'] == 'au')
        scalebar = AnchoredSizeBar(ims[kwargs['scalebar_nax']].transData,
                                   scalebar_distance,
                                   str(kwargs['scalebar_distance'])
                                   + ' ' + kwargs['scalebar_units'],
                                   kwargs['scalebar_loc'],
                                   pad=kwargs['scalebar_pad'],
                                   color=kwargs['scalebar_color'],
                                   frameon=False,
                                   size_vertical=kwargs['scalebar_width'],
                                   label_top=kwargs['scalebar_labeltop'],
                                   sep=kwargs['scalebar_sep'],
                                   fontproperties=fontprops)
        ims[kwargs['scalebar_nax']].add_artist(scalebar)

    if output_name is not None:
        fig.savefig('{}{}.{}'.format(kwargs['path_save'],
                                     output_name, kwargs['output_format']),
                    bbox_inches=kwargs['bbox_inches'])



def calc_moment(image_cube,
                moment,
                box,
                box_noise,
                channel_step,
                channel_0,
                channel_f,
                add_last_channel=False,
                output_name=None,
                **kwargs):
    """
    Calculates an image in which an operation (moment) has been applied (max, mean, median...) And draws contours over it.
    box = [[x_1,y_1],[x_2,y_2]]
    """

    for param in mf.default_params:
        kwargs[param] = kwargs[param] \
                if param in kwargs \
                else mf.default_params[param]

    if kwargs['wcs'] is not None:
        fig = plt.figure(figsize=(kwargs['fig_size'], kwargs['fig_size']))
        ax = plt.subplot(projection=kwargs['wcs'])
    else:
        fig, ax = plt.subplots(figsize=(kwargs['fig_size'],kwargs['fig_size']))

    channels = list(np.arange(channel_0,channel_f,channel_step))+([len(image_cube)-1] if add_last_channel else [])

    cmap_seq = mf.make_sequential_cm(kwargs['cmap_nanize'], n_seq=1000, n_ref=len(channels)) # this is only used for nanized slices


    if kwargs['norm'] == 'linear':
        norm = colors.Normalize(vmax=kwargs['vmax'], vmin=kwargs['vmin'])
    elif kwargs['norm'] == 'log':
        norm = colors.LogNorm(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
    elif kwargs['norm'] == 'symlog':
        norm = colors.SymLogNorm(linthresh=kwargs['linthresh'],
                                 linscale=kwargs['linscale'],
                                 vmin=kwargs['vmin'], vmax=kwargs['vmax'])
    elif kwargs['norm'] == 'divnorm':
        norm = colors.DivergingNorm(vmin=kwargs['vmin'], vcenter=kwargs['vcenter'], vmax=kwargs['vmax'])

    calc_std = lambda im,bx: np.std(im[bx[0][0]:bx[1][0],bx[0][1]:bx[1][1]])
    if moment == 'nanize':
        for i,channel in enumerate(channels):
#            moment_array = np.copy(image_cube[channel])
            image = collapse_chans(image_cube,
                                   operation,
                                   channel,
                                   width_chan,
                                   box) if operation is not None else image_cube[channel]
            moment_array[image < (kwargs['n_sigma']*calc_std(image,box_noise))] = np.nan
            ax.imshow(moment_array,
                      norm=norm,
                      origin='lower',
                      cmap=ListedColormap(cmap_seq[i]),
                      interpolation=kwargs['interpolation'])
    else:
        moment_array = collapse_chans(image_cube,
                                      operation=moment,
                                      central_chans=channels,
                                      width_chan=0,
                                      box=box,
                                      **kwargs)
        ax.imshow(moment_array,
                  norm=norm,
                  origin='lower',
                  cmap=kwargs['cmap_bg'],
                  interpolation=kwargs['interpolation'])

    cmap_contours = cm.get_cmap(kwargs['cmap_contours'])
    colors_cmap = [to_hex(cmap_contours(i)) for i in np.linspace(0,1,len(channels))]
    if kwargs['plot_contours']:
        for i,channel in enumerate(channels):
            image_cut = np.copy(image_cube[channel])
            if kwargs['sig_chans'] == None:
                n_sigma = kwargs['n_sigma']
            else:
                n_sigma = np.sum([kwargs['n_sigmas'][0]*(channel<kwargs['sig_chans'][0])] + \
                [kwargs['n_sigmas'][k]*((channel>=kwargs['sig_chans'][k-1])and(channel<kwargs['sig_chans'][k]))
                 for k in np.arange(1,len(kwargs['n_sigmas']))])

            if kwargs['contour_area']:
                ax.contourf(image_cut, levels=[n_sigma*np.std(image_cut[box_noise[0][0]:box_noise[1][0],box_noise[0][1]:box_noise[1][1]]), np.max(image_cut)],
                            colors=colors_cmap[i], linewidths=kwargs['contour_linewidth'], alpha=kwargs['contour_alpha'])
            else:
                ax.contour(image_cut, levels=kwargs['contour_levels'],
                            colors=colors_cmap[i], linewidths=kwarg['contour_linewidth'], alpha=kwargs['contour_alpha'])

        velocities_list = np.linspace(kwargs['v0'],kwargs['vf'],len(image_cube))
        v_range = velocities_list[channel_0:channel_f]
        cbar = plt.colorbar(
               cm.ScalarMappable(norm=colors.Normalize(
                   vmax=np.abs(v_range[-1]),
                   vmin=np.abs(v_range[0])),
               cmap=kwargs['cmap_contours']),
               ax=ax)
        cbar_ticklabels = ['-{:.0f}'.format(i) for i in cbar.ax.get_yticks()]
        cbar.ax.set_yticklabels(cbar_ticklabels)
        cbar.ax.set_ylabel('Velocity (km/s)')

    if box is not None:
        ax.set_xlim([box[0][0], box[1][0]])
        ax.set_ylim([box[0][1], box[1][1]])

    if kwargs['bb_centers'] is not None:
        ras = [i[0] for i in kwargs['bb_centers']]
        decs = [i[1] for i in kwargs['bb_centers']]
        ax.plot(ras,
                decs,
                '{}{}'.format(kwargs['markercenter_style'],kwargs['markercenter_color']),
                transform=ax.get_transform('icrs'),
                markersize=kwargs['markercenter_size'],
                mew=kwargs['markercenter_width'])

    if kwargs['add_beam']:
        pa = kwargs['header']['BPA'] * np.pi/180. + np.pi/2   #in radians
        a = kwargs['header']['BMAJ'] / kwargs['header']['CDELT2']   #semi-major axis in pixels
        b = kwargs['header']['BMIN'] / kwargs['header']['CDELT2']   #semi-minor axis in pixels
        xpos = (box[0][0] + kwargs['xpos_beam']) if box is not None else kwargs['xpos_beam']
        ypos = (box[0][1] + kwargs['ypos_beam']) if box is not None else kwargs['ypos_beam']

        geometry = EllipseGeometry(x0=xpos, y0=ypos, sma=b, eps=b/a, pa=pa)
        aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                   geometry.sma*(1 - geometry.eps),
                                   geometry.pa)
        aper.plot(ax, color=kwargs['beam_color'], linewidth=kwargs['beam_linewidth'])

    if kwargs['add_scalebar']:
        fontprops = fm.FontProperties(size=kwargs['scalebar_fontsize'])
        arcsec_per_pix = kwargs['header']['CDELT1'] * 3600
        scalebar_distance = (kwargs['scalebar_distance']/arcsec_per_pix) \
                            * (kwargs['scalebar_units']=='arcsec') \
                            + (kwargs['scalebar_distance']/kwargs['SVS13_distance']) \
                            / arcsec_per_pix * (kwargs['scalebar_units']=='au')
        scalebar = AnchoredSizeBar(ax.transData,
                           scalebar_distance,
                                   str(kwargs['scalebar_distance'])+\
                                   ' '+kwargs['scalebar_units'],
                                   kwargs['scalebar_loc'],
                           pad=kwargs['scalebar_pad'],
                           color=kwargs['scalebar_color'],
                           frameon=False,
                           size_vertical=kwargs['scalebar_width'],
                           label_top=kwargs['scalebar_labeltop'],
                           sep=kwargs['scalebar_sep'],
                           fontproperties=fontprops)

        ax.add_artist(scalebar)

    ax.set_xlabel('R.A. (ICRS) ')
    ax.set_ylabel('Declination (ICRS) ')

    if kwargs['add_stars']:
        xs_star = kwargs['vla4a_deg'][0], kwargs['vla4b_deg'][0]
        ys_star = kwargs['vla4a_deg'][1], kwargs['vla4b_deg'][1]
        ax.plot(xs_star,
                ys_star,
                marker=kwargs['markerstar_style'],
                color=kwargs['markerstar_color'],
                linestyle='',
                transform=ax.get_transform('icrs'),
                markersize=kwargs['markerstar_size'],
                mew=kwargs['markerstar_width'])

    if output_name is not None:
        fig.savefig('{}{}.{}'.format(kwargs['path_save'],
                                     output_name,
                                     kwargs['output_format']),
                    bbox_inches=kwargs['bbox_inches'], )

#    return


def mosaic_images(images,
                  headers,
                  text_dic,
                  box,
                  nrow,
                  ncol,
                  ngrid,
                  output_name,
                  **kwargs):
    for param in mf.default_params:
        kwargs[param] = kwargs[param] \
                if param in kwargs \
                else mf.default_params[param]

    matplotlib.rcParams.update({'font.size': kwargs['font_size']})     #changes the size of labels
    plt.rc('text', usetex=kwargs['use_tex'])                           #set False to unable latex rendering


    fig = plt.figure(figsize=(nrow*kwargs['magical_factor'],
                              (ncol+1)*ngrid,))

    gs1 = GridSpec(nrow, (ncol+1)*ngrid, )
    gs1.update(wspace=kwargs['wspace'], hspace=kwargs['hspace'],)

    n = 0
    ims = {}
    for i,j in product(range(nrow),
                       [i for i in range(ncol*ngrid)][::ngrid]):
        ims[n] = plt.subplot(gs1[i,j:j+ngrid])

        xaxis=ims[n].axes.get_xaxis()
        yaxis=ims[n].axes.get_yaxis()
        ims[n].tick_params(direction=kwargs['tick_direction'],
                          grid_color=kwargs['grid_color'],
                          colors=kwargs['grid_color'],
                          labelcolor=kwargs['label_color'],
                          color=kwargs['grid_color'],
                          grid_linewidth=kwargs['grid_linewidth'],
                          width=kwargs['tick_width'],
                          length=kwargs['tick_length'])

        ims[n].tick_params(direction=kwargs['tick_direction'],
                          grid_color=kwargs['grid_color'],
                          colors=kwargs['grid_color'],
                          labelcolor=kwargs['label_color'],
                          color=kwargs['grid_color'],
                          grid_linewidth=kwargs['grid_linewidth'],
                          width=kwargs['tick_width'],
                          length=kwargs['tick_length'])

        if (j>0) and (i<nrow-1):
            xaxis.set_visible(False)
            yaxis.set_visible(False)
        if (i==(nrow-1)) and (j>0):
            yaxis.set_visible(False)
        if (i<(nrow-1)) and (j==0):
            xaxis.set_visible(False)

        n += 1

    ax_cbar = plt.subplot(gs1[:,-ngrid+kwargs['colorbar_ngrid']])

    im_trans = {}
    ax_trans = {}
    for n in images:
        vmax_trans = np.max(images[n]) if kwargs['pv_vmax'] is None else kwargs['pv_vmax']
        vmin_trans = np.min(images[n]) if kwargs['pv_vmin'] is None else kwargs['pv_vmin']
        try:
            im_trans[n], ax_trans[n] = show_slice_cube(images[n],
                                 ax=ims[n],
                                 channel=0,
                                 box=box,
                                 wcs=None,
                                 header=headers[n],
                                 add_beam=False,
                                 beam_color='k',
                                 vmax=vmax_trans,
                                 vcenter=vmax_trans*kwargs['vcenter_factor'],
                                 vmin=0,
                                 norm='divnorm',
                                 cmap='jet',
                                 add_scalebar=False,
                                 scalebar_loc='lower right',
                                 scalebar_distance=200,
                                 scalebar_color='k',
                                 scalebar_units='au',
                                 return_ax=False,
                                 plot_vel=False,
                                 render='both',
                                 contour_levels=None,
                                 plot_cbar=False,
                                 cbar_extend='both',
                                 cube_aspect='auto',
                                 pv=True,
                                 pv_xlabel=None,
                                 pv_ylabel=None,
                                 show_slice_return='contour_imax')
        except:
            print("Last image was {}".format(n))
            break

        if kwargs['pv_jetline_x'] is not None:
            ymin, ymax = ax_trans[n].get_ylim()
            ax_trans[n].vlines(kwargs['pv_jetline_x'],
                               ymin,
                               ymax,
                               color=kwargs['pv_jetlinecolor'],
                               linewidth=kwargs['pv_jetlinewidth'],
                               linestyle=kwargs['pv_jetlinestyle'])

        if text_dic is not None:
            ax_trans[n].text(kwargs['pvtext_x'],
                            kwargs['pvtext_y'],
                           "{:.2f}''".format(text_dic[n]),
                           color=kwargs['textcolor'],
                           fontsize=kwargs['font_size'],)

        xticks_trans = ax_trans[n].get_xticks()
        ax_trans[n].set_xticks(xticks_trans[1:])
        if box is not None:
            ax_trans[n].set_xlim([box[0][0], box[1][0]])
            ax_trans[n].set_ylim([box[0][1], box[1][1]])

    cbar = plt.colorbar(im_trans[0],
                        cax=ax_cbar,
                        extend=kwargs['cbar_extend'],
                        pad=kwargs['cbar_pad'],
                        fraction=kwargs['cbar_fraction'],
                        shrink=kwargs['cbar_shrink'],
                        panchor=kwargs['cbar_panchor'],
                        anchor=kwargs['cbar_anchor'],
                        aspect=kwargs['cbar_aspect'])

    cbar.set_label(r'Jy/beam', labelpad=kwargs['cbar_label_pad'])

    plt.figtext(kwargs['figtext_x_hor'],
                kwargs['figtext_x_vert'],
                kwargs['pv_xlabel'])
    plt.figtext(kwargs['figtext_y_hor'],
                kwargs['figtext_y_vert'],
                kwargs['pv_ylabel'],
                rotation='vertical')

    if output_name is not None:
        fig.savefig('{}{}.{}'.format(kwargs['path_save'],
                                     output_name, kwargs['output_format']),
                    bbox_inches=kwargs['bbox_inches'])


def image2cloud(im, box, vmin_ell,):
    """
    Transform an image to a cloud of points, where the number of points in pixel size is proportional to the flux found in that pixel of the image.
    """
    im[im < vmin_ell] = 0
    grid_points = im / vmin_ell
    xs = []
    ys = []
    ny = np.arange(box[0][1], box[1][1])
    nx = np.arange(box[0][0], box[1][0])

    for i,j in product(nx, ny):
        xs += [np.random.rand(int(grid_points[j][i])) + i]
        ys += [np.random.rand(int(grid_points[j][i])) + j]

    xs_pos = np.array([x for xpos in xs for x in xpos])
    ys_pos = np.array([y for ypos in ys for y in ypos])

    return xs_pos, ys_pos


def fit_ellipse(image_cube,
                channel,
                box,
                vmin_ell,
                width_chan=0,
                operation=None,
                output_name=None,
                return_params=False,
                show_plot_ell=True,
                **kwargs):
    """
    Fits the ellipse which minimize the distance of a cloud of points.
    """
    for param in mf.default_params:
        kwargs[param] = kwargs[param] \
                if param in kwargs \
                else mf.default_params[param]

    im, ax = show_slice_cube(image_cube,
                             channel,
                             width_chan=width_chan,
                             operation=operation,
                             box=box,
                             return_ax=True,
                             output_name=None,
                             **kwargs)

    xs_pos, ys_pos = image2cloud(im, box, vmin_ell)
    params, center, phi, axes = ellipse_fitting(np.array(xs_pos),
                                                np.array(ys_pos))

    theta = np.arange(0,2*np.pi, 0.01)
    a, b = axes
    xx = center[0] + a*np.cos(theta)*np.cos(phi) - b*np.sin(theta)*np.sin(phi)
    yy = center[1] + a*np.cos(theta)*np.sin(phi) + b*np.sin(theta)*np.cos(phi)

    ax.plot(xx, yy, color=kwargs['ellipse_color'])
    ax.set_aspect('equal')

    fig1 = plt.gcf()

    if box is not None:
        ax.set_xlim([box[0][0], box[1][0]])
        ax.set_ylim([box[0][1], box[1][1]])

    if show_plot_ell:
        plt.figure(figsize=(kwargs['fig_size'],
                            kwargs['fig_size']))
        plt.plot(np.array(xs_pos),
                 np.array(ys_pos),
                 marker=kwargs['markerellfit_style'],
                 color=kwargs['markerellfit_color'],
                 markersize=kwargs['markerellfit_size'],
                 linestyle='')
        plt.plot(xx, yy, color=kwargs['ellipse_color'])
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()

    if return_params:
        return params, center, phi, axes

    if output_name is not None:
        fig1.savefig('{}{}.{}'.format(kwargs['path_save'],
                                      output_name,
                                      kwargs['output_format']),
                     bbox_inches=kwargs['bbox_inches'], )


def jet_axis_dir(x, x0, y0, pa):
    """
    Retunrs y(x) for a direction defined by pa, x0 and y0
    """
    return (x - x0) * np.tan(-np.pi/180*(pa+90)) + y0

def plot_pvlines(points,
                 header,
                 dists,
                 trans_dists=None,
                 ax=None,
                 show_plot=True,
                 **kwargs):
    """
    Draws in a given ax, or in a new ax, the direction defined by the start and ending point. The number
    of points drawn is given de array dists, which are the distances to the starting point. If trans_dists
    is given, perpendicular lines are drawn also. Points must be a dictionary with 'start' and 'end' points.
    Returns a dictionary with the line points and the pa.
    """
    wcs = WCS(header).celestial
    for param in mf.default_params:
        kwargs[param] = kwargs[param] \
                if param in kwargs \
                else mf.default_params[param]

    if ax is None and show_plot:
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(projection=wcs)
    else:
        pass

    pa_PV = -np.arctan((points['start'][1]-points['end'][1]) \
                       / (points['start'][0]-points['end'][0])) * 180 / np.pi + 90
    xs_dists = np.array([np.cos((pa_PV-90)*np.pi/180)*dist for dist in dists])*u.arcsec
    center_xs = np.array([(points['start'][0]*u.deg+arcsec.to(u.deg)).value for arcsec in xs_dists])
    center_ys = [jet_axis_dir(x,
                          x0=points['start'][0],
                          y0=points['start'][1],
                          pa=pa_PV) for x in center_xs]
    jet_axis = {'center_xs':center_xs, 'center_ys':center_ys, 'pa_PV':pa_PV}

    if trans_dists is not None:
        pa_PV_trans = pa_PV - 90
        trans_dists_centered = [-dist/2. for dist in trans_dists] + [dist/2. for dist in trans_dists]
        trans_xs_dists = np.array([np.sin((pa_PV-90)*np.pi/180)*dist for dist in trans_dists_centered])*u.arcsec


        trans_xs = np.array([(points['start'][0]*u.deg+arcsec.to(u.deg)).value for arcsec in trans_xs_dists])
        trans_ys = np.array([jet_axis_dir(x,
                             x0=points['start'][0],
                             y0=points['start'][1],
                             pa=pa_PV_trans) for x in trans_xs])

        trans_axis = {'trans_xs':trans_xs, 'trans_ys':trans_ys, 'pa_PV_trans':pa_PV_trans}

    if show_plot:
        transform_ax = ax.get_transform('icrs')
        ax.plot(center_xs,
                center_ys,
                color=kwargs['pvline_color'],
                linestyle=kwargs['pvline_style'],
                linewidth=kwargs['pvline_width'],
                transform=transform_ax,)
        if trans_dists is not None:
            ax.plot(trans_xs,
                    trans_ys,
                    color=kwargs['pvline_trans_color'],
                    linestyle=kwargs['pvline_trans_style'],
                    linewidth=kwargs['pvline_trans_width'],
                    transform=transform_ax,)
        ax.set_aspect('equal')

    if trans_dists is not None:
        return jet_axis, trans_axis, ax
    else:
        return jet_axis, ax



class FitCurve2DArray(object):
    """
    Fits a curve y = f(x,**params) over the elements of a 2Darray, being y,x the spatial coordinates.
    If x is not the x spatial coordinate, set onlyparams=True
    """
    default_attr = {'res_factor':5,
                   'tol':10**(-8)}
    atribs2save = ['data','params','extent','res_factor','method','xmin','xmax',
                        'ymin','ymax','ny','nx','xedges','yedges','xlim_fit','tol',
                        'op2min','xs','ys','pixels','values','fit_results']
    def __init__(self, f, data, init_params, extent, **kwargs):
        self.f = f
        self.data = np.copy(data)
        self.params = init_params
        self.extent = extent
        self.res_factor = kwargs['res_factor'] if 'res_factor' in kwargs else self.default_attr['res_factor']
        self.ax = kwargs['ax'] if 'ax' in kwargs else None
        self.method = kwargs['method'] if 'method' in kwargs else None
        self.xmin, self.xmax = np.min(self.extent[0]), np.max(self.extent[0])
        self.ymin, self.ymax = np.min(self.extent[1]), np.max(self.extent[1])
        self.ny, self.nx = np.shape(self.data)
        self.xedges = np.linspace(self.extent[0][0], self.extent[0][1], self.nx)
        self.yedges = np.linspace(self.extent[1][0], self.extent[1][1], self.ny)
        self.xlim_fit =  kwargs['xlim_fit'] if 'xlim_fit' in kwargs else [self.xmin,self.xmax]
        self.ylim_fit = kwargs['ylim_fit'] if 'ylim_fit' in kwargs else [self.ymin,self.ymax]
        self.tol = kwargs['tol'] if 'tol' in kwargs else self.default_attr['tol']
        self.op2min = kwargs['op2min'] if 'op2min' in kwargs else 'mean'
        self.db_name = kwargs['db_name'] if 'db_name' in kwargs else 'FitCurve2Darray'
        self.nmax = kwargs['nmax'] if 'nmax' in kwargs else 100
        self.dist_lim = kwargs['dist_lim'] if 'dist_lim' in kwargs else None
        self.wcs = kwargs['wcs'] if 'wcs' in kwargs else None
        self.onlyparams = kwargs['onlyparams'] if 'onlyparams' in kwargs else None
        self.optimize_method = kwargs['optimize_method'] if 'optimize_method' in kwargs else 'minimize'
        self.bounds = kwargs['bounds'] if 'bounds' in kwargs else None

        self._xs = np.linspace(self.xlim_fit[0], self.xlim_fit[1], self.nx*self.res_factor)
        self._ys = None
        self.zs = kwargs['zs'] if 'zs' in kwargs else None
        self.xs = None
        self.ys = None
        self.pixels = None
        self.values = None
        self.fit_results = None
        self.distance = None
        self.dist_lim_reached = False
        self.color_plot_fit = 'r'


    def _in_limits(self, x, y):
            _in_xlim = (x>self.xmin) and (x<self.xmax)
            _in_ylim = (y>self.ymin) and (y<self.ymax)
            return _in_xlim and _in_ylim


    def _calc_distance(self,):
            distance_f = lambda x,y,x0,y0: ((x-x0)**2 + (y-y0)**2)**(0.5)
            self.distance = 0
            for i,j in zip(range(len(self.xs))[1:], range(len(self.ys))[1:]):
                    self.distance += distance_f(self.xs[i],
                                                self.ys[j],
                                                self.xs[i-1],
                                                self.ys[j-1])
                    if self.dist_lim is not None:
                        if self.distance > self.dist_lim:
                            self.xmax = np.max([self.xs[i-1],self.xs[0]])
                            self.xmin = np.min([self.xs[i-1],self.xs[0]])
                            self.ymax = np.max([self.f(self.xmax,**self.params),
                                                self.ymax])
                            self.ymin = np.min([self.f(self.ymin,**self.params),
                                                self.ymin])
                            self.dist_lim_reached = True
                            break
                        else:
                            self.dist_lim_reached = False
            if self.dist_lim_reached:
                self.pick_values()
            else:
                pass

    def whichpix(self, x, y,):
        """
        Returns the pixel coordinates and the value of the element for the physical coordinate (x,y)
        """
        if self.wcs is not None:
            xy_skycoord = SkyCoord(x*u.deg,y*u.deg)
            xpix = int(np.round(skycoord_to_pixel(xy_skycoord, self.wcs)[0])),
            ypix = int(np.round(skycoord_to_pixel(xy_skycoord, self.wcs)[1]))
            xval = pixel_to_skycoord(xpix, ypix, self.wcs).ra.value
            yval = pixel_to_skycoord(xpix, ypix, self.wcs).dec.value
        else:
            xpix = np.argmin(np.abs(self.xedges-x))
            ypix = np.argmin(np.abs(self.yedges-y))
            xval, yval = self.xedges[xpix], self.yedges[ypix]
        return (xpix, ypix), (xval, yval)

    def pick_values(self, params=None):
        """
        Returns the pixels and its values that lies under the curve f(x,**params)
        """
        if self.onlyparams:
             xys = np.array([[x,y] for x,y in zip(*self.f(**self.params)) if self._in_limits(x,y)])
#            xys = np.array([self.f(z,**self.params) for z in self.zs if self._in_limits(*self.f(z,**self.params))])
        else:
            self._ys = np.array([self.f(x,**self.params) for x in self._xs])
            xys = np.array([[x,y] for x,y in zip(self._xs, self._ys) if self._in_limits(x,y)])

        try:
            self.xs, self.ys = xys[:,0], xys[:,1]
        except:
            self.xs, self.ys = [], []
        self._calc_distance()
        self.pixels = {self.whichpix(x, y)[0] for x,y in zip(self.xs,self.ys)}
        self.values = [self.data[ypix,xpix] for xpix,ypix in self.pixels]

    def to_minimize(self,pars):
        """
        The function to minimize
        """
        self.params = {param: pars[i] for i,param in enumerate(self.params)}
        self.pick_values()
        if self.op2min =='mean':
            return -np.mean(self.values)
        elif self.op2min=='median':
            return -np.median(self.values)
        elif self.op2min=='sum':
            return -np.sum(self.values)
        elif self.op2min=='max':
            max_return = -np.max(self.values) if len(self.values)!=0 else 0
            return max_return
        elif self.op2min=='nmax':
            max_return = -np.median(mf.get_nmax(self.values, self.nmax)) if len(self.values)!=0 else 0
            return max_return
        elif self.op2min=='mean_path':
            return -np.sum(self.values)/self.distance

    def fit(self,):
        """
        Calls minimize from scipy.optimize, and minimize the function to_minimize
        """
        pars = [self.params[param] for param in self.params]
        if self.optimize_method == 'minimize':
            self.fit_results = optimize.minimize(self.to_minimize,
                                        pars,
                                        method=self.method,
                                        tol=self.tol)
        elif self.optimize_method == 'brute':
            self.fit_results = optimize.brute(self.to_minimize,
                                        self.bounds,
                                        full_output=True)
        elif self.optimize_method == 'basinhopping':
            self.fit_results = optimize.basinhopping(self.to_minimize,
                                        pars)
        elif self.optimize_method == 'differential_evolution':
            self.fit_results = optimize.differential_evolution(self.to_minimize,
                                                              self.bounds)
        try:
            self.params = {param:self.fit_results['x'][i] for i,param in enumerate(self.params)}
        except:
            pass

    def plot_fit(self, ax=None, c=None):
        """
        Draws the fitting results given an ax
        """
        c = c if c is not None else self.color_plot_fit
        self.ax = ax if ax is not None else self.ax
        self.ax.plot(self.xs, self.ys, c)

    def pickle(self, db_name=None):
        """
        Saves the results in a database with shelve
        """
        db_name = db_name if db_name is not None else self.db_name
        self.path_database = '{}{}.db'.format(mf.default_params['path_database'],db_name)
        database = shelve.open(self.path_database)
        for atrib in self.atribs2save:
            database[atrib] = getattr(self, atrib)
        print('\n Results pickled in {}.db!\n'.format(db_name))
        database.close()

def calc_time_dyn(m, i):
#z' = m * v_los
    t_years = (m * 235*u.au/(u.km/u.s)).to(u.yr) / (np.cos(i) * np.sin(i))
    return t_years
