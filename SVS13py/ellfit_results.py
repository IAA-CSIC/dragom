import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
import matplotlib.colors as colors
import matplotlib.image as mpimg
from matplotlib.patches import Arc, Ellipse

from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord

from photutils import EllipticalAperture
from photutils import SkyEllipticalAperture

from scipy.optimize import curve_fit
from scipy import optimize

from scipy.ndimage.interpolation import rotate
from scipy.interpolate import interp1d

import shelve

import pandas as pd

import SVS13py.mf as mf

import SVS13py.windmodel_correction as wm
import SVS13py.general_deprojection as gd


def create_table(ellip_data, header, set_index=None, **kwargs):
    """
    This function creates a DataFrame of the results of the ellipse fits,
    from ellip_data, which is a dictionary taken from a database. A header
    is necesary, since all results in ellip_data are in pixel coordinates.
    """
    wcs = WCS(header).celestial
    for param in mf.default_params:
        kwargs[param] = kwargs[param] if param in kwargs \
                         else mf.default_params[param]

    geom_params = ['x0',
                   'y0',
                   'sma',
                   'eps',
                   'pa',
                   'sector_angular_width',
                   'initial_polar_angle',
                   'initial_polar_radius']

    bb_data = pd.DataFrame(index=[int(float(param))
                                  for param in ellip_data],
                           columns=geom_params)

    for geom_param in geom_params:
        bb_data[geom_param] = [ellip_data[chan]['geom'].__dict__[geom_param]
                               for chan in ellip_data]

    vla4a_coords = SkyCoord(*kwargs['vla4a_deg'], unit='deg')
    vla4b_coords = SkyCoord(*kwargs['vla4b_deg'], unit='deg')
    vla4b_offset = vla4b_coords.skyoffset_frame()
    aper_sky = {}
    for chan in bb_data.index:
        aper = EllipticalAperture((bb_data['x0'][chan], bb_data['y0'][chan]),
                                  bb_data['sma'][chan],
                                  bb_data['sma'][chan]
                                  * (1.-bb_data['eps'][chan]),
                                  bb_data['pa'][chan])
        aper_sky[chan] = aper.to_sky(wcs)

#   We save in the bb_data DataFrame all the results
#   We start with the ellipse parameters
    bb_data['center_coords'] = [aper_sky[chan].positions
                                for chan in bb_data.index]
    bb_data['x0_offset_arcsec'] = [bb_data['center_coords'][chan].transform_to(
        vla4b_offset).lon.to(u.arcsec).value
                                   for chan in bb_data.index]
    bb_data['y0_offset_arcsec'] = [bb_data['center_coords'][chan].transform_to(
        vla4b_offset).lat.to(u.arcsec).value
                                   for chan in bb_data.index]
    bb_data['x0_RA'] = [aper_sky[chan].positions.ra.value
                        for chan in bb_data.index]
    bb_data['y0_DEC'] = [aper_sky[chan].positions.dec.value
                         for chan in bb_data.index]
    bb_data['sma_arcsec'] = [aper_sky[chan].a.value
                             for chan in bb_data.index]
    bb_data['sma_deg'] = bb_data['sma_arcsec'] / 3600
    bb_data['semiminor_axis'] = [aper_sky[chan].b.value
                                 for chan in bb_data.index]
    bb_data['semiminor_axis_deg'] = bb_data['semiminor_axis'] / 3600
    bb_data['mean_radius'] = [np.mean([bb_data['semiminor_axis'][chan],
                                       bb_data['sma_arcsec'][chan]])
                              for chan in bb_data.index]

#   Distances from the source
    bb_data['dist_vla4a'] = [bb_data['center_coords'][chan].separation(vla4a_coords).arcsec
                             for chan in bb_data.index]
    bb_data['dist_vla4b'] = [bb_data['center_coords'][chan].separation(vla4b_coords).arcsec
                             for chan in bb_data.index]
    bb_data['mean_dist'] = [np.mean([bb_data['dist_vla4a'][chan],
                                    bb_data['dist_vla4b'][chan]])
                            for chan in bb_data.index]

#   PA from the source
    bb_data['pa_center_vla4a'] = [bb_data['center_coords'][chan].position_angle(vla4a_coords)
                                  for chan in bb_data.index]
    bb_data['pa_center_vla4b'] = [bb_data['center_coords'][chan].position_angle(vla4b_coords)
                                  for chan in bb_data.index]

#   We store the velocities
    velocities = mf.vel_from_header(header)
    bb_data['vel'] = np.array([velocities[chan] for chan in bb_data.index])
    if 'syst_vel' in kwargs:
        bb_data['vel_rel'] = bb_data['vel'] - kwargs['syst_vel']

    if set_index is not None:
        bb_data = bb_data.set_index(set_index)

    return bb_data


def write_deprojection(dic_bb, bb, inclination_angle):
    """
    This function writes to the dic_bb the results from the deprojection

    Parameters
    ------
    dic_bb: dic
        Dictionary with results
    bb: str
        Key of the results
    inclination_angle: float
        Inclination angle given in radians
    """
    dic_bb[bb]['inclination_angle'] = inclination_angle
    dic_bb[bb]['data']['deprojected_dist_vla4b'] = \
        dic_bb[bb]['data']['dist_vla4b'] / np.sin(inclination_angle)
    dic_bb[bb]['data']['deprojected_dist_vla4a'] = \
        dic_bb[bb]['data']['dist_vla4a'] / np.sin(inclination_angle)

    dic_bb[bb]['data']['distance_edge'] = \
        np.sqrt(dic_bb[bb]['data']['deprojected_dist_vla4b']**2
                + dic_bb[bb]['data']['mean_radius']**2)

    dic_bb[bb]['data']['theta_angle'] = \
        np.arctan(dic_bb[bb]['data']['mean_radius']
                  / dic_bb[bb]['data']['deprojected_dist_vla4b'])

#    This deprojected vel are taking into account a wind model!
    dic_bb[bb]['data']['deprojected_vel'] = dic_bb[bb]['data']['vel_rel'] \
        / (np.cos(inclination_angle)
           * np.cos(dic_bb[bb]['data']['theta_angle']))


def tdyn_windmodel_calc(dist_edge_arcsec, dist_pc, depr_vel):
    """
    Calculates the dinamical time given the deprojected distance from the
    center to the edge of the jet, and the deprojected velocity. The result is
    in yr result is in yr
    """
    dist_au = (dist_edge_arcsec * dist_pc * u.au).to(u.km)
    vel_kms = abs(depr_vel) * u.km/u.s
    return (dist_au/vel_kms).to(u.yr).value


def write_tdyn_windmodel(dic_bb, bb, SVS13_distance=None):
    SVS13_distance = SVS13_distance if SVS13_distance is not None \
        else mf.default_params['SVS13_distance']

    dic_bb[bb]['data']['tdyn_windmodel'] = \
        [tdyn_windmodel_calc(dic_bb[bb]['data']['distance_edge'][chan],
                             SVS13_distance,
                             dic_bb[bb]['data']['deprojected_vel'][chan])
         for chan in dic_bb[bb]['data'].index]


def write_wind_correction(dic_bb, bb, deproject_i, s_p, a_p=None, **kwargs):
    """
    Writes the wind_model correction due to the distance of the centers
    to the real z-axis

    Parameters
    ----------
    dic_bb: dic
        dictionary of the ellipse results
    bb: str
        Keyword entry of the ellipse result
    deproject_i: float
        Inclination angle in radians
    s_p: float
        Power index of parabola
    a_p: float
        apex of the parabola
    """
    for kwarg in mf.default_params:
        kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
            else mf.default_params[kwarg]

    # The no corrected deprojection assumes that the centers lies over
    # the z-axis
    a_p = a_p if a_p is not None \
        else np.max(dic_bb[bb]['data']['deprojected_dist_vla4b'])

    dic_bb[bb]['data']['dx_zaxis'] = np.array([wm.dx_zaxis_try(
                                    dic_bb[bb]['data']['deprojected_dist_vla4b'][chan],
                                    s_p,
                                    a_p,
                                    deproject_i)
                                    for chan in dic_bb[bb]['data'].index])

    dic_bb[bb]['data']['dz_zaxis'] = np.array([wm.dz_zaxis_try(
                                    dic_bb[bb]['data']['deprojected_dist_vla4b'][chan],
                                    s_p,
                                    a_p,
                                    deproject_i)
                                    for chan in dic_bb[bb]['data'].index])

    dic_bb[bb]['data']['deprojected_dist_vla4b_wcor'] = \
            dic_bb[bb]['data']['deprojected_dist_vla4b'] \
            + dic_bb[bb]['data']['dz_zaxis']
    dic_bb[bb]['data']['distance_edge_wcor'] = \
            np.sqrt(dic_bb[bb]['data']['deprojected_dist_vla4b_wcor']**2 \
                    + dic_bb[bb]['data']['mean_radius']**2)
    dic_bb[bb]['data']['theta_angle_wcor'] = \
            np.arctan(dic_bb[bb]['data']['mean_radius'] / \
                      dic_bb[bb]['data']['deprojected_dist_vla4b_wcor'])
    dic_bb[bb]['data']['deprojected_vel_wcor'] = \
            dic_bb[bb]['data']['vel_rel'] \
            / (np.cos(deproject_i) \
            * np.cos(dic_bb[bb]['data']['theta_angle_wcor']))

    dic_bb[bb]['data']['tdyn_windmodel_wcor'] = \
            [tdyn_windmodel_calc(dic_bb[bb]['data']['distance_edge_wcor'][chan],
                                 kwargs['SVS13_distance'],
                                 dic_bb[bb]['data']['deprojected_vel_wcor'][chan])
             for chan in dic_bb[bb]['data'].index]


def write_xphis(dic_bb, bb, wcs, PA_jet, box_PV, **kwargs):
    """
    Writes to the dic_bb[bb] the information of the x_phis, needed for the
    general deprojection

    Parameters
    ----------
    dic_bb: dict
       A dictionary with the ellipse fit results
    bb: str
       The keyword for the ellipse fit result dataframe
    PA_jet: float
       The PA angle of the PV line in degrees
    """
    for kwarg in mf.default_params:
        kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
           else mf.default_params[kwarg]

    # We store the interesting points of the ellipses:
    xys_sky_PV = gd.generate_pv_line_coordinates(PA_jet, box_PV, wcs,
                                                 kwargs['n_points_PV'])

    spetial_points = {chan:
                      gd.spetial_points_calc(
                          dic_bb[bb]['data']['x0'][chan],
                          dic_bb[bb]['data']['y0'][chan],
                          dic_bb[bb]['data']['sma'][chan],
                          dic_bb[bb]['data']['eps'][chan],
                          dic_bb[bb]['data']['pa'][chan],
                          kwargs['n_points_ellipse'],
                          wcs,
                          dist_center=dic_bb[bb]['data']['dist_vla4b'][chan],
                          xys_sky_PV=xys_sky_PV)
                      for chan in dic_bb[bb]['data'].index}

    dic_bb[bb]['ellipse_pixel'] = pd.Series(
        [spetial_points[chan][0][0]
         for chan in dic_bb[bb]['data'].index],
        index=dic_bb[bb]['data'].index)

    dic_bb[bb]['ellipse_sky'] = pd.Series(
        [spetial_points[chan][0][1]
         for chan in dic_bb[bb]['data'].index],
        index=dic_bb[bb]['data'].index)

    dic_bb[bb]['data']['p180_sky'] = np.array(
        [spetial_points[chan][1][0]
         for chan in dic_bb[bb]['data'].index])

    dic_bb[bb]['data']['p0_sky'] = np.array(
        [spetial_points[chan][1][1]
         for chan in dic_bb[bb]['data'].index])

    dic_bb[bb]['data']['x_phi180'] = np.array(
        [spetial_points[chan][2][0]
         for chan in dic_bb[bb]['data'].index])

    dic_bb[bb]['data']['x_phi90'] = np.array(
        [spetial_points[chan][2][1]
         for chan in dic_bb[bb]['data'].index])

    dic_bb[bb]['data']['x_phi0'] = np.array(
        [spetial_points[chan][2][2]
         for chan in dic_bb[bb]['data'].index])


def write_depr_funcs(dic_bb, bb, i_angle):
    """
    This function generates a dictionary of functions that are useful for the
    deprojection

    Parameters
    ----------
    dic_bb: dict
        Dictionary with ellipse fit results
    bb: str
        Keyword of the entry of the dictionary to write
    i_angle: float
        Inclination angle in radians
    """
    dic_bb[bb]['i_angle'] = i_angle

    depr_funcs = gd.depr_func_gen(dic_bb, bb, i_angle)

    dic_bb[bb]['df'] = {}
    dic_bb[bb]['df']['zs'] = depr_funcs[0]
    dic_bb[bb]['df']['R_z'] = depr_funcs[1]
    dic_bb[bb]['df']['R_z_inter'] = depr_funcs[2]
    dic_bb[bb]['df']['xp_phi_z'] = depr_funcs[3]
    dic_bb[bb]['df']['xps_phis'] = depr_funcs[4]
    dic_bb[bb]['df']['xp_phi_equal_0'] = depr_funcs[5]
    dic_bb[bb]['df']['z_xp_phi'] = depr_funcs[6]
    dic_bb[bb]['df']['vzp_xp_phi_inter'] = depr_funcs[7]
    dic_bb[bb]['df']['vzp_xp_phi'] = depr_funcs[8]
    dic_bb[bb]['df']['vzp_z_phi'] = depr_funcs[9]
    dic_bb[bb]['df']['v_zs'] = depr_funcs[10]
    dic_bb[bb]['df']['alphas'] = depr_funcs[11]


def import_table(database_name, header, set_index=None, **kwargs):
    """
    Calls create_table to create a DataFrame of results from a database.
    """
    database_path = '{}{}.db'.format(mf.default_params['path_database'],
                                     database_name)
    database = shelve.open(database_path)
    ellip_data = {param: database[param] for param in database}
    database.close()
    bb_data = create_table(ellip_data, header=header, set_index=set_index,
                           **kwargs)
    return bb_data


def get_color(vel_range, vel, color_length, cmap):
    cmap_contours = cm.get_cmap(cmap, color_length)
    vs_linear = np.linspace(vel_range[0], vel_range[1], color_length)
    diff_vels = np.abs(vs_linear - vel)
    closest_vel, closest_vel_index = np.min(diff_vels), np.argmin(diff_vels)
    color_vel = to_hex(cmap_contours(closest_vel_index))
    return color_vel


def plot_fit_results(dic_bb, box, header, deproject=False, output_name=None,
                     vel_color=True, source_dist='dist_vla4b',
                     return_axes=False, show_plot=True, **kwargs):
    for param in mf.default_params:
        kwargs[param] = kwargs[param] if param in kwargs \
            else mf.default_params[param]
    wcs = WCS(header).celestial
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    fig4, ax4 = plt.subplots(figsize=(8, 8))
    fig5, ax5 = plt.subplots(figsize=(8, 8))
    ax6 = plt.subplot(projection=wcs)

    if deproject:
        vel_key = 'deprojected_vel'
        source_dist_key = 'deprojected_' + source_dist
    else:
        syst_vel = '' if 'syst_vel' not in kwargs else '_rel'
        vel_key = 'vel' + syst_vel
        source_dist_key = source_dist

    v_range = [np.max([np.max(dic_bb[bb]['data'][vel_key]) for bb in dic_bb]),
               np.min([np.min(dic_bb[bb]['data'][vel_key]) for bb in dic_bb])]
#    chan_range = [np.max([np.max(dic_bb[bb]['data'].index) for bb in dic_bb]),
#               np.min([np.min(dic_bb[bb]['data'].index) for bb in dic_bb])]
#
#    cmap_contours = cm.get_cmap(kwargs['cmap_ellipses'])
#    colors_cmap = [0 for i in range(chan_range[1])] \
#                + [to_hex(cmap_contours(i)) for i in np.linspace(0,1,chan_range[0]-chan_range[1]+1)]
#    mf.dbugg.cmap_contours = cmap_contours
#    mf.dbugg.colors_cmap = colors_cmap
#    mf.dbugg.chan_range = chan_range
    c_ell = lambda vel_color, vel, bb, w: get_color(v_range,
                                                    vel,
                                                    kwargs['color_length'],
                                                    kwargs['cmap_ellipses']) \
            if vel_color else dic_bb[bb][w]

    for bb in dic_bb:
        ax1.plot(dic_bb[bb]['data'][vel_key],
                 dic_bb[bb]['data'][source_dist_key],
                 dic_bb[bb]['sty'],
                 color=dic_bb[bb]['c'],
                 alpha=dic_bb[bb]['alpha'])

        ax2.plot(dic_bb[bb]['data'][source_dist_key],
                 dic_bb[bb]['data'][vel_key],
                 dic_bb[bb]['sty'],
                 color=dic_bb[bb]['c'],
                 alpha=dic_bb[bb]['alpha'])

        ax3.plot(dic_bb[bb]['data'][vel_key],
                 dic_bb[bb]['data']['sma_arcsec'],
                 dic_bb[bb]['sty'],
                 color=dic_bb[bb]['c'],
                 alpha=dic_bb[bb]['alpha'])

        markerfacecolor = dic_bb[bb]['markerfacecolor_centers'] \
                if dic_bb[bb]['markerfacecolor_centers'] is not None \
                else c_ell(vel_color, vel, bb, 'c')
        markeredgecolor = dic_bb[bb]['markeredgecolor_centers'] \
                if dic_bb[bb]['markeredgecolor_centers'] is not None \
                else c_ell(vel_color, vel, bb, 'c')

        for chan in dic_bb[bb]['data'].index:
            vel = dic_bb[bb]['data'][vel_key][chan]
            ax4.plot(dic_bb[bb]['data'][source_dist_key][chan],
                     dic_bb[bb]['data']['sma_arcsec'][chan],
                     dic_bb[bb]['sty'],
                     fillstyle=dic_bb[bb]['fsty'],
                     color=c_ell(vel_color, vel, bb, 'c_ell'),
                     markersize=dic_bb[bb]['markersize_centers'],
                     markerfacecolor=markerfacecolor,
                     markeredgecolor=markeredgecolor,
                     markeredgewidth=dic_bb[bb]['markeredgewidth_centers'],)

        ax5.plot(dic_bb[bb]['data'][vel_key],
                 dic_bb[bb]['data']['eps'],
                 dic_bb[bb]['sty'],)

        xs_star = mf.default_params['vla4a_deg'][0], \
                  mf.default_params['vla4b_deg'][0]
        ys_star = mf.default_params['vla4a_deg'][1], \
                  mf.default_params['vla4b_deg'][1]
        ax6.plot(dic_bb[bb]['data']['x0_RA'],
                 dic_bb[bb]['data']['y0_DEC'],
                 dic_bb[bb]['sty'],
                 transform=ax6.get_transform('icrs'),
                 color='b')
        ax6.plot(xs_star,
                 ys_star,
                 marker=mf.default_params['markerstar_style'],
                 color='k',
                 linestyle='',
                 transform=ax6.get_transform('icrs'),
                 markersize=mf.default_params['markerstar_size'],
                 mew=mf.default_params['markerstar_width'])

    ax1.set_xlabel('Velocity (km/s)')
    ax1.set_ylabel('Distance (arcsec)')
    ax1.invert_xaxis()

    ax2.set_xlabel('Distance (arcsec)')
    ax2.set_ylabel('Velocity (km/s)')
    ax2.invert_yaxis()

    ax3.set_xlabel('Velocity (km/s)')
    ax3.set_ylabel('SMA (arcsec)')

    ax4.set_xlabel('Distance (arcsec)')
    ax4.set_ylabel('SMA (arcsec)')
    ax4.set_aspect('equal')

    ax5.set_xlabel('Velocity (km/s)')
    ax5.set_ylabel('Ellipticity')

    ax6.set_aspect('equal')
    ax6.set_xlim([box[0][0],box[1][0]])
    ax6.set_ylim([box[0][1],box[1][1]])
    ax6.set_xlabel('ICRS RA')
    ax6.set_ylabel('ICRS DEC')

    if vel_color and kwargs['colorbar_show']:
#        velocities_list = np.linspace(kwargs['v0'],kwargs['vf'],channel_f)
        cbar = plt.colorbar(
               cm.ScalarMappable(norm=colors.Normalize(vmax=np.abs(v_range[1]),
                                                       vmin=np.abs(v_range[0])),
                                 cmap=kwargs['cmap_ellipses']),
               ax=ax4,
               orientation=kwargs['colorbar_orientation'],
               fraction=kwargs['colorbar_fraction'],
               pad=kwargs['colorbar_pad'],
               shrink=kwargs['colorbar_shrink'],
               aspect=kwargs['colorbar_aspect'],
               anchor=kwargs['colorbar_anchor'],
               panchor=kwargs['colorbar_panchor'],
               extend=kwargs['cbar_extend'])
        cbar_ticklabels = ['-{:.0f}'.format(i) for i in cbar.ax.get_yticks()]
        cbar.ax.set_yticklabels(cbar_ticklabels)

        if kwargs['colorbar_orientation'] == 'vertical':
            cbar.ax.set_ylabel('Velocity (km/s)')
        elif kwargs['colorbar_orientation'] == 'horizontal':
            cbar.ax.set_xlabel('Velocity (km/s)')

        ax4.set_facecolor(kwargs['facecolor_background'])

    if output_name is not None:
        fig1.savefig('{}{}_vel_sma.pdf'.format(kwargs['path_save'], output_name),
                     bbox_inches=kwargs['bbox_inches'])
        fig4.savefig('{}{}.pdf'.format(kwargs['path_save'], output_name),
                     bbox_inches=kwargs['bbox_inches'])

    if return_axes:
        return [ax1, ax2, ax3, ax4, ax5, ax6]

    if show_plot:
        plt.show()


def dyn_time(dic_bb,
             box,
             wcs,
             output_name=None,
             db_func=None,
             source_dist='dist_vla4b',
             **kwargs):
    for param in mf.default_params:
        kwargs[param] = kwargs[param] if param in kwargs \
        else mf.default_params[param]

    fig1, ax1 = plt.subplots(figsize=(8,8))
    vels = {dic_bb[bb]['regr']: [] for bb in dic_bb}
    dists = {dic_bb[bb]['regr']: [] for bb in dic_bb}

    for bb in dic_bb:
        ax1.plot(dic_bb[bb]['data'][source_dist],
                 dic_bb[bb]['data']['vel'],
                 dic_bb[bb]['sty'],
                 color=dic_bb[bb]['c'],
                 alpha=dic_bb[bb]['alpha'])
        vels[dic_bb[bb]['regr']] += list(dic_bb[bb]['data']['vel'])
        dists[dic_bb[bb]['regr']] += list(dic_bb[bb]['data'][source_dist])

    y_func = lambda x, m, y0: m*x + y0
    x_first, x_last = ax1.get_ylim()
    xs = np.linspace(x_first, x_last, 100)
    popt = {}
    pcov = {}
    t_dyn = {}
    for regr in vels:
        popt[regr], pcov[regr] = curve_fit(y_func, vels[regr], dists[regr])
        ys = y_func(xs, popt[regr][0], popt[regr][1])
        ax1.plot(ys, xs, '--k')
#        t_dyn[regr] = popt[0] * u.s
    ax1.set_ylabel('Velocity (km/s)')
    ax1.set_xlabel('Distance (arcsec)')
    ax1.invert_yaxis()

    if output_name is not None:
        fig1.savefig('{}{}_dyn_time.pdf'.format(kwargs['path_save'],
                                                output_name),
                     bbox_inches=kwargs['bbox_inches'])

    return popt, pcov


def plot_arc_map(dic_bb, box, header, ax=None, cbax=None, show_legend=True,
                 output_name=None, vel_color=True, compare_hodapp=None,
                 return_ax=False, **kwargs):
    wcs = WCS(header).celestial

    for param in mf.default_params:
        kwargs[param] = kwargs[param] if param in kwargs \
            else mf.default_params[param]

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(projection=wcs)
    else:
        pass

    if compare_hodapp == 'background':
        img = mpimg.imread('hodapp.png')
        ax.imshow(img, origin='lower')
    elif compare_hodapp is None:
        pass
    else:
        pix_rear_arc = 5
        for arc in compare_hodapp:
            patch_arc_black = Arc(
                (compare_hodapp[arc]['x0'],
                 compare_hodapp[arc]['y0']),
                width=compare_hodapp[arc]['width'],
                height=compare_hodapp[arc]['height'],
                angle=compare_hodapp[arc]['angle'],
                theta1=compare_hodapp[arc]['theta1'],
                theta2=compare_hodapp[arc]['theta2'],
                color='k',
                linewidth=2,
                linestyle='-',
                zorder=kwargs['hodapp_zorder'])
            patch_arc_white = Arc(
                (compare_hodapp[arc]['x0'],
                 compare_hodapp[arc]['y0']),
                width=compare_hodapp[arc]['width']-pix_rear_arc,
                height=compare_hodapp[arc]['height']-pix_rear_arc,
                angle=compare_hodapp[arc]['angle'],
                theta1=compare_hodapp[arc]['theta1'],
                theta2=compare_hodapp[arc]['theta2'],
                color='w',
                linewidth=2,
                linestyle='-',
                zorder=kwargs['hodapp_zorder'])
            ax.add_patch(patch_arc_black)
            ax.add_patch(patch_arc_white)

    xs_star = kwargs['vla4a_deg'][0], kwargs['vla4b_deg'][0]
    ys_star = kwargs['vla4a_deg'][1], kwargs['vla4b_deg'][1]

    transform_ax = ax.get_transform('icrs')
    ax.plot(xs_star,
            ys_star,
            marker=kwargs['markerstar_style'],
            color=kwargs['markerstar_color'],
            linestyle='',
            transform=transform_ax,
            markersize=kwargs['markerstar_size'],
            mew=kwargs['markerstar_width'],
            zorder=kwargs['stars_zorder'])

    y_func = lambda x, m, y0: m*x + y0
    bb_with_regr = [bb for bb in dic_bb
                    if dic_bb[bb]['regr'] is not None]
    bb_with_text = [bb for bb in dic_bb
                    if (dic_bb[bb]['pos_text'] is not None)
                    and (dic_bb[bb]['regr'] is not None)]
    xs_dic = {dic_bb[bb]['regr']: [] for bb in bb_with_regr}
    ys_dic = {dic_bb[bb]['regr']: [] for bb in bb_with_regr}

#    #sky_coord
#    x_first = pixel_to_skycoord(box[0][0],box[0][1],wcs).ra,
#    y_first = pixel_to_skycoord(box[0][0],box[0][1],wcs).dec
#    x_last = pixel_to_skycoord(box[1][0],box[1][1],wcs).ra,
#    y_last = pixel_to_skycoord(box[1][0],box[1][1],wcs).dec
#    xs = np.array([x.value for x in np.linspace(x_first, x_last, 100)])

#   #pix coord
    x_first, y_first = box[0]
    x_last, y_last = box[1]
    xs = np.array([x for x in np.linspace(x_first, x_last, 100)])

    syst_vel = '' if 'syst_vel' not in kwargs else '_rel'
    v_range = [np.max([np.max(dic_bb[bb]['data']['vel'+syst_vel])
                       for bb in dic_bb]),
               np.min([np.min(dic_bb[bb]['data']['vel'+syst_vel])
                       for bb in dic_bb])]

    c_ell = lambda vel_color, vel, bb, w: get_color(
                                              v_range,
                                              vel,
                                              kwargs['color_length'],
                                              kwargs['cmap_ellipses']) \
                                           if vel_color else dic_bb[bb][w]

    for bb in dic_bb:
        if dic_bb[bb]['regr'] is not None:
            # xs_dic[dic_bb[bb]['regr']] += list(dic_bb[bb]['data']['x0_RA'])
            # ys_dic[dic_bb[bb]['regr']] += list(dic_bb[bb]['data']['y0_DEC'])
            xs_dic[dic_bb[bb]['regr']] += list(dic_bb[bb]['data']['x0'])
            ys_dic[dic_bb[bb]['regr']] += list(dic_bb[bb]['data']['y0'])
        if dic_bb[bb]['plot_centers']:
            # vmin = np.min(abs(dic_bb[bb]['data']['vel'+syst_vel]))
            # vmax = np.max(abs(dic_bb[bb]['data']['vel'+syst_vel]))
            for chan in dic_bb[bb]['data'].index:
                vel = dic_bb[bb]['data']['vel'+syst_vel][chan]
                markerfacecolor = dic_bb[bb]['markerfacecolor_centers'] \
                    if dic_bb[bb]['markerfacecolor_centers'] is not None \
                    else c_ell(vel_color, vel, bb, 'c')
                markeredgecolor = dic_bb[bb]['markeredgecolor_centers'] \
                    if dic_bb[bb]['markeredgecolor_centers'] is not None \
                    else c_ell(vel_color, vel, bb, 'c')
                ax.plot(dic_bb[bb]['data']['x0_RA'][chan],
                        dic_bb[bb]['data']['y0_DEC'][chan],
                        dic_bb[bb]['sty'],
                        fillstyle=dic_bb[bb]['fsty'],
                        transform=transform_ax,
                        markersize=dic_bb[bb]['markersize_centers'],
                        markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor,
                        markeredgewidth=dic_bb[bb]['markeredgewidth_centers'],
                        alpha=dic_bb[bb]['alpha'],
                        zorder=dic_bb[bb]['centers_zorder'])

        if dic_bb[bb]['plot_ellipses'] == 'aper':
            for chan in np.sort(dic_bb[bb]['data'].index)[::dic_bb[bb]['step_chan']]:
                positions = SkyCoord(ra=dic_bb[bb]['data']['x0_RA'][chan]*u.deg,
                                     dec=dic_bb[bb]['data']['y0_DEC'][chan]*u.deg)
                vel = dic_bb[bb]['data']['vel'+syst_vel][chan]
                a = dic_bb[bb]['data']['sma_deg'][chan] * u.deg
                b = dic_bb[bb]['data']['sma_deg'][chan] * \
                    (1.-dic_bb[bb]['data']['eps'][chan]) * u.deg
                pa_deg = (dic_bb[bb]['data']['pa'][chan]-np.pi/2.) \
                    * 180 / np.pi * u.deg
                aper = SkyEllipticalAperture(positions, a, b, pa_deg)
                aper2plot = aper.to_pixel(wcs)
                aper2plot.plot(ax,
                               color=c_ell(vel_color, vel, bb, 'c_ell'),
                               linestyle=dic_bb[bb]['linestyle'],
                               alpha=dic_bb[bb]['alpha'],
                               zorder=dic_bb[bb]['ellipse_zorder'])

        elif dic_bb[bb]['plot_ellipses'] == 'patches':
            chans_sorted = \
                    np.sort(dic_bb[bb]['data'].index)[::dic_bb[bb]['step_chan']]
            arcs_dic_sky = {chan: {} for chan in chans_sorted}
            for chan in chans_sorted:
                a = dic_bb[bb]['data']['sma_deg'][chan] * u.deg
                b = dic_bb[bb]['data']['sma_deg'][chan] * \
                    (1.-dic_bb[bb]['data']['eps'][chan])*u.deg
                pa_deg = (dic_bb[bb]['data']['pa'][chan]) * 180 / np.pi * u.deg
                arcs_dic_sky[chan]['x0'] = \
                    dic_bb[bb]['data']['x0_RA'][chan] * u.deg
                arcs_dic_sky[chan]['y0'] = \
                    dic_bb[bb]['data']['y0_DEC'][chan] * u.deg
                arcs_dic_sky[chan]['width'] = a * 2
                arcs_dic_sky[chan]['height'] = b * 2
                arcs_dic_sky[chan]['angle'] = pa_deg.value
                arcs_dic_sky[chan]['theta1'] = \
                    dic_bb[bb]['theta1_pa'] - pa_deg.value
                arcs_dic_sky[chan]['theta2'] = \
                    dic_bb[bb]['theta2_pa'] - pa_deg.value

            arcs_dic_pix = mf.arcs2pix(arcs_dic_sky, header)
            for chan in chans_sorted:
                vel = dic_bb[bb]['data']['vel'+syst_vel][chan]
                """
                It is very important not to use Arc or Ellipse, which comes
                from matplotlib.patches, in skycoordnates, but in pixel
                coordinates. If it is done in skycoordinates, this would not be
                the right ellipse, since rembember that the angular distances
                in RA depends on the latitude.
                """
                patch_arc = Arc((arcs_dic_pix[chan]['x0'],
                                 arcs_dic_pix[chan]['y0']),
                                width=arcs_dic_pix[chan]['width'],
                                height=arcs_dic_pix[chan]['height'],
                                angle=arcs_dic_pix[chan]['angle'],
                                theta1=arcs_dic_pix[chan]['theta1'],
                                theta2=arcs_dic_pix[chan]['theta2'],
                                color=c_ell(vel_color, vel, bb, 'c_ell'),
                                linewidth=1,
                                linestyle=dic_bb[bb]['linestyle'],
                                alpha=dic_bb[bb]['alpha'],
                                zorder=dic_bb[bb]['ellipse_zorder'])
                ax.add_patch(patch_arc)
        elif dic_bb[bb]['plot_ellipses'] == 'filled_ellipse':
            chans_sorted = np.sort(dic_bb[bb]['data'].index)[::dic_bb[bb]['step_chan']]
            arcs_dic_sky = {chan: {} for chan in chans_sorted}
            for chan in chans_sorted:
                a = dic_bb[bb]['data']['sma_deg'][chan] * u.deg
                b = dic_bb[bb]['data']['sma_deg'][chan] \
                    * (1.-dic_bb[bb]['data']['eps'][chan])*u.deg
                pa_deg = (dic_bb[bb]['data']['pa'][chan]) * 180 / np.pi * u.deg
                arcs_dic_sky[chan]['x0'] = \
                    dic_bb[bb]['data']['x0_RA'][chan] * u.deg
                arcs_dic_sky[chan]['y0'] = \
                    dic_bb[bb]['data']['y0_DEC'][chan] * u.deg
                arcs_dic_sky[chan]['width'] = a * 2
                arcs_dic_sky[chan]['height'] = b * 2
                arcs_dic_sky[chan]['angle'] = pa_deg.value

            arcs_dic_pix = mf.arcs2pix(arcs_dic_sky, header)
            for chan in chans_sorted:
                vel = dic_bb[bb]['data']['vel'+syst_vel][chan]
                patch_arc = Ellipse(
                    (arcs_dic_pix[chan]['x0'],
                     arcs_dic_pix[chan]['y0']),
                    width=arcs_dic_pix[chan]['width'],
                    height=arcs_dic_pix[chan]['height'],
                    angle=arcs_dic_pix[chan]['angle'],
                    color=c_ell(vel_color, vel, bb, 'c_ell'),
                    linewidth=1,
                    linestyle=dic_bb[bb]['linestyle'],
                    alpha=dic_bb[bb]['alpha'],
                    zorder=dic_bb[bb]['ellipse_zorder'])
                ax.add_patch(patch_arc)
        else:
            pass

    rot_angles = {}
    pas = {}
    for regr in xs_dic:
        popt, pcov = curve_fit(y_func, xs_dic[regr], ys_dic[regr])
        ys = y_func(xs, popt[0], popt[1])
        debug.xs, debug.ys = xs, ys
        debug.popt = popt[0]
        ax.plot(xs,
                ys,
                dic_bb[bb]['sty_regr'],
                transform=ax.get_transform('pixel'),
               )
        rot_angles[regr] = np.arctan(popt[0]) * 180./np.pi
        pas[regr] = rot_angles[regr] + 90
    pas


    for bb in bb_with_text:
        ax.text(dic_bb[bb]['pos_text'][0],
                dic_bb[bb]['pos_text'][1],
                "PA = {:.0f}°".format(pas[dic_bb[bb]['regr']]),
                rotation=rot_angles[dic_bb[bb]['regr']],
                rotation_mode='anchor',
                transform=transform_ax,
                zorder=dic_bb[bb]['regr_zorder'])

    if vel_color and (cbax is not None):
        cbar = plt.colorbar(
            cm.ScalarMappable(
                norm=colors.Normalize(
                    vmax=np.abs(v_range[1]),
                    vmin=np.abs(v_range[0])),
                cmap=kwargs['cmap_ellipses']),
            cax=cbax,
            orientation=kwargs['colorbar_orientation'],
            fraction=kwargs['colorbar_fraction'],
            pad=kwargs['colorbar_pad'],
            shrink=kwargs['colorbar_shrink'],
            aspect=kwargs['colorbar_aspect'],
            anchor=kwargs['colorbar_anchor'],
            panchor=kwargs['colorbar_panchor'],
            extend=kwargs['cbar_extend'])
        cbar_ticklabels = ['-{:.0f}'.format(i) for i in cbar.ax.get_xticks()]
        cbar.ax.set_xticklabels(cbar_ticklabels)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(direction='out', color='k', pad=2.)
        if kwargs['colorbar_orientation'] == 'vertical':
            cbar.ax.set_ylabel('Velocity (km/s)')
        elif kwargs['colorbar_orientation'] == 'horizontal':
            cbar.ax.set_xlabel('Velocity (km/s)')

            # cbxy = ax.get_position().get_points().flatten()
            # cbpos = cbxy[3] + (cbxy[1] + cbxy[3])*0.008
            # cbhei = (cbxy[1] + cbxy[3])*0.02
            # cbax = fig.add_axes([cbxy[0], cbpos, cbxy[2]-cbxy[0], cbhei])
            # cb = plt.colorbar(im, cax=cbax, orientation='horizontal', format='%.1f',
            #                   ticks=[si_min, si_min+(np.amax(si)-si_min)/4,
            #                   si_min+2*(np.amax(si)-si_min)/4,
            #                   si_min+3*(np.amax(si)-si_min)/4, np.amax(si)])
            # cb.ax.xaxis.set_ticks_position('top')
            # cb.ax.tick_params(direction='out', color='k', pad=2.)
            # cb.outline.set_linewidth(0.5)
            # cb.ax.set_xlabel(r'{\bf Total Intensity (mJy/pixel)}', labelpad=7.)
            # cb.ax.xaxis.set_label_position('top')

    ax.set_aspect('equal')
    ax.set_xlim([box[0][0], box[1][0]])
    ax.set_ylim([box[0][1], box[1][1]])
    ax.set_xlabel(kwargs['icrs_xlabel'])
    ax.set_ylabel(kwargs['icrs_ylabel'])
    if kwargs['rotate_ticktext_yaxis'] is not None:
        ax.coords[1].set_ticklabel(rotation=kwargs['rotate_ticktext_yaxis'])

    if show_legend:
        plt.legend()

    ax.set_facecolor(kwargs['facecolor_background'])

    if return_ax:
        return ax

    plt.show()
    if output_name is not None:
        fig.savefig('{}{}.pdf'.format(kwargs['path_save'], output_name),
                    bbox_inches=kwargs['bbox_inches'])


def dist_vel(dic_bb,
             box,
             deproject=False,
             ax=None,
             output_name=None,
             vel_color=True,
             source_dist='dist_vla4b',
             return_ax=False,
             show_plot=True,
             **kwargs):

    for param in mf.default_params:
        kwargs[param] = kwargs[param] if param in kwargs \
        else mf.default_params[param]
    if ax == None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot()
    else:
        pass

    if deproject:
        vel_key = 'deprojected_vel'
        source_dist_key = 'deprojected_' + source_dist
    else:
        syst_vel = '' if 'syst_vel' not in kwargs else '_rel'
        vel_key = 'vel' + syst_vel
        source_dist_key = source_dist

    v_range = [np.max([np.max(dic_bb[bb]['data'][vel_key]) for bb in dic_bb]),
               np.min([np.min(dic_bb[bb]['data'][vel_key]) for bb in dic_bb])]
    c_ell = lambda vel_color, vel, bb, w: get_color(v_range,
                                                    vel,
                                                    kwargs['color_length'],
                                                    kwargs['cmap_ellipses']) \
            if vel_color else dic_bb[bb][w]
    for bb in dic_bb:
        for chan in dic_bb[bb]['data'].index:
            vel = dic_bb[bb]['data'][vel_key][chan]
            markerfacecolor = dic_bb[bb]['markerfacecolor_centers'] \
                    if dic_bb[bb]['markerfacecolor_centers'] is not None \
                    else c_ell(vel_color, vel, bb, 'c')
            markeredgecolor = dic_bb[bb]['markeredgecolor_centers'] \
                    if dic_bb[bb]['markeredgecolor_centers'] is not None \
                    else c_ell(vel_color, vel, bb, 'c')
            xs = dic_bb[bb]['data'][vel_key][chan] \
                    if kwargs['inverse_plot'] is not True \
                    else dic_bb[bb]['data'][source_dist_key][chan]
            ys = dic_bb[bb]['data'][source_dist_key][chan] \
                    if kwargs['inverse_plot'] is not True \
                    else dic_bb[bb]['data'][vel_key][chan]
            xlabel = 'Velocity (km/s)' \
                    if kwargs['inverse_plot'] is not True \
                    else 'Distance (arcsec)'
            ylabel = 'Distance (arcsec)' \
                    if kwargs['inverse_plot'] is not True \
                    else 'Velocity (km/s)'
            label = dic_bb[bb]['label'] \
                    if chan==dic_bb[bb]['data'].index[0] \
                    else None
            ax.plot(xs,
                    ys,
                    dic_bb[bb]['sty'],
                    fillstyle=dic_bb[bb]['fsty'],
                    color=c_ell(vel_color, vel, bb, 'c_ell'),
                    markersize=dic_bb[bb]['markersize_centers'],
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    markeredgewidth=dic_bb[bb]['markeredgewidth_centers'],
                    label=label)


    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if kwargs['inverse_plot']:
        ax.invert_yaxis()
    else:
        ax.invert_xaxis()

    if vel_color and kwargs['colorbar_show']:
#        velocities_list = np.linspace(kwargs['v0'],kwargs['vf'],channel_f)
        cbar = plt.colorbar(
               cm.ScalarMappable(norm=colors.Normalize(vmax=np.abs(v_range[1]),
                                                       vmin=np.abs(v_range[0])),
               cmap=kwargs['cmap_ellipses']),
               ax=ax,
               orientation=kwargs['colorbar_orientation'],
               fraction=kwargs['colorbar_fraction'],
               pad=kwargs['colorbar_pad'],
               shrink=kwargs['colorbar_shrink'],
               aspect=kwargs['colorbar_aspect'],
               anchor=kwargs['colorbar_anchor'],
               panchor=kwargs['colorbar_panchor'],
               extend=kwargs['cbar_extend'])
        cbar_ticklabels = ['-{:.0f}'.format(i) for i in cbar.ax.get_yticks()]
        cbar.ax.set_yticklabels(cbar_ticklabels)

        if kwargs['colorbar_orientation'] == 'vertical':
            cbar.ax.set_ylabel('Velocity (km/s)')
        elif kwargs['colorbar_orientation'] == 'horizontal':
            cbar.ax.set_xlabel('Velocity (km/s)')

        ax.set_facecolor(kwargs['facecolor_background'])

    if kwargs['plot_legend']:
        ax.legend()

    if 'inclination_angle' in kwargs:
        ax.text(0.05,0.97,
                'i={:d}°'.format(int(np.round(kwargs['inclination_angle']))),
                transform=ax.transAxes)

    if output_name is not None:
        fig.savefig('{}{}_dist_vel.pdf'.format(kwargs['path_save'],
                                               output_name),
                     bbox_inches=kwargs['bbox_inches'])

    if return_ax:
        return ax

    if show_plot:
        plt.show()


def sma_vel(dic_bb, box, deproject=False, ax=None,
            output_name=None, vel_color=True, source_dist='dist_vla4b',
            return_ax=False, show_plot=True, **kwargs):

    for param in mf.default_params:
        kwargs[param] = kwargs[param] if param in kwargs \
        else mf.default_params[param]
    if ax == None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot()
    else:
        pass

    if deproject:
        vel_key = 'deprojected_vel'
        source_dist_key = 'deprojected_' + source_dist
    else:
        syst_vel = '' if 'syst_vel' not in kwargs else '_rel'
        vel_key = 'vel' + syst_vel
        source_dist_key = source_dist

    v_range = [np.max([np.max(dic_bb[bb]['data'][vel_key]) for bb in dic_bb]),
               np.min([np.min(dic_bb[bb]['data'][vel_key]) for bb in dic_bb])]
    c_ell = lambda vel_color, vel, bb, w: get_color(v_range,vel,kwargs['color_length'],kwargs['cmap_ellipses']) if vel_color else dic_bb[bb][w]
    for bb in dic_bb:
        for chan in dic_bb[bb]['data'].index:
            vel = dic_bb[bb]['data'][vel_key][chan]
            markerfacecolor = dic_bb[bb]['markerfacecolor_centers'] \
                    if dic_bb[bb]['markerfacecolor_centers'] is not None \
                    else c_ell(vel_color, vel, bb, 'c')
            markeredgecolor = dic_bb[bb]['markeredgecolor_centers'] \
                    if dic_bb[bb]['markeredgecolor_centers'] is not None \
                    else c_ell(vel_color, vel, bb, 'c')
            xs = dic_bb[bb]['data'][vel_key][chan] \
                    if kwargs['inverse_plot'] is not True \
                    else dic_bb[bb]['data']['mean_radius'][chan]
            ys = dic_bb[bb]['data']['mean_radius'][chan] \
                    if kwargs['inverse_plot'] is not True \
                    else dic_bb[bb]['data'][vel_key][chan]
            xlabel = 'Velocity (arcsec)' \
                    if kwargs['inverse_plot'] is not True \
                    else 'Mean radius (arcsec)'
            ylabel = 'Mean Radius (arcsec)' \
                    if kwargs['inverse_plot'] is not True \
                    else 'Velocity (arcsec)'
            label = dic_bb[bb]['label'] \
                    if chan==dic_bb[bb]['data'].index[0] \
                    else None
            ax.plot(xs,
                    ys,
                    dic_bb[bb]['sty'],
                    fillstyle=dic_bb[bb]['fsty'],
                    color=c_ell(vel_color, vel, bb, 'c_ell'),
                    markersize=dic_bb[bb]['markersize_centers'],
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    markeredgewidth=dic_bb[bb]['markeredgewidth_centers'],
                    label=label)


    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if kwargs['inverse_plot']:
        ax.invert_yaxis()
    else:
        ax.invert_xaxis()

    if vel_color and kwargs['colorbar_show']:
#        velocities_list = np.linspace(kwargs['v0'],kwargs['vf'],channel_f)
        cbar = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmax=np.abs(v_range[1]),
                                                                    vmin=np.abs(v_range[0])),
                            cmap=kwargs['cmap_ellipses']),
                            ax=ax,
                            orientation=kwargs['colorbar_orientation'],
                            fraction=kwargs['colorbar_fraction'],
                            pad=kwargs['colorbar_pad'],
                            shrink=kwargs['colorbar_shrink'],
                            aspect=kwargs['colorbar_aspect'],
                            anchor=kwargs['colorbar_anchor'],
                            panchor=kwargs['colorbar_panchor'],
                            extend=kwargs['cbar_extend'])
        cbar_ticklabels = ['-{:.0f}'.format(i) for i in cbar.ax.get_yticks()]
        cbar.ax.set_yticklabels(cbar_ticklabels)

        if kwargs['colorbar_orientation'] == 'vertical':
            cbar.ax.set_ylabel('Velocity (km/s)')
        elif kwargs['colorbar_orientation'] == 'horizontal':
            cbar.ax.set_xlabel('Velocity (km/s)')

        ax.set_facecolor(kwargs['facecolor_background'])

    if 'inclination_angle' in kwargs:
        ax.text(0.05,0.97,'i={}°'.format(int(np.round(kwargs['inclination_angle']))),
                transform=ax.transAxes)

    if kwargs['plot_legend']:
        ax.legend()

    if output_name is not None:
        fig.savefig('{}{}_dist_vel.pdf'.format(kwargs['path_save'],output_name),
                     bbox_inches=kwargs['bbox_inches'])

    if return_ax:
        return ax

    if show_plot:
        plt.show()


def SMA_dist(dic_bb, box, deproject=False, ax=None, vel_color=True,
             source_dist='dist_vla4b', show_plot=True, return_ax=False,
             tip_arcs=None, save_fig=None,  **kwargs):
    for param in mf.default_params:
        kwargs[param] = kwargs[param] if param in kwargs else mf.default_params[param]
    if ax == None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot()
    else:
        pass

    if deproject:
        vel_key = 'deprojected_vel'
        source_dist_key = 'deprojected_' + source_dist
    else:
        syst_vel = '' if 'syst_vel' not in kwargs else '_rel'
        vel_key = 'vel' + syst_vel
        source_dist_key = source_dist

    v_range = [np.max([np.max(dic_bb[bb]['data'][vel_key]) for bb in dic_bb]),
               np.min([np.min(dic_bb[bb]['data'][vel_key]) for bb in dic_bb])]
#    chan_range = [np.max([np.max(dic_bb[bb]['data'].index) for bb in dic_bb]),
#               np.min([np.min(dic_bb[bb]['data'].index) for bb in dic_bb])]
#
#    cmap_contours = cm.get_cmap(kwargs['cmap_ellipses'])
#    colors_cmap = [0 for i in range(chan_range[1])] \
#                + [to_hex(cmap_contours(i)) for i in np.linspace(0,1,chan_range[0]-chan_range[1]+1)]
    c_ell = lambda vel_color, vel, bb, w: get_color(v_range,
                                                    vel,
                                                    kwargs['color_length'],
                                                    kwargs['cmap_ellipses']) \
            if vel_color else dic_bb[bb][w]

    for bb in dic_bb:
        for chan in dic_bb[bb]['data'].index:
            vel = dic_bb[bb]['data'][vel_key][chan]
            markerfacecolor = dic_bb[bb]['markerfacecolor_centers'] \
                    if dic_bb[bb]['markerfacecolor_centers'] is not None \
                    else c_ell(vel_color, vel, bb, 'c')
            markeredgecolor = dic_bb[bb]['markeredgecolor_centers'] \
                    if dic_bb[bb]['markeredgecolor_centers'] is not None \
                    else c_ell(vel_color, vel, bb, 'c')
            xs = dic_bb[bb]['data']['mean_radius'][chan] \
                    if kwargs['inverse_plot'] is not True \
                    else dic_bb[bb]['data'][source_dist_key][chan]
            ys = dic_bb[bb]['data'][source_dist_key][chan] \
                    if kwargs['inverse_plot'] is not True \
                    else dic_bb[bb]['data']['mean_radius'][chan]
            ylabel = 'Distance (arcsec)' \
                    if kwargs['inverse_plot'] is not True \
                    else 'Mean radius (arcsec)'
            xlabel = 'Mean Radius (arcsec)' \
                    if kwargs['inverse_plot'] is not True \
                    else 'Distance (arcsec)'
            ax.plot(xs,
                    ys,
                    dic_bb[bb]['sty'],
                    fillstyle=dic_bb[bb]['fsty'],
                    color=c_ell(vel_color, vel, bb, 'c_ell'),
                    markersize=dic_bb[bb]['markersize_centers'],
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    markeredgewidth=dic_bb[bb]['markeredgewidth_centers'])


            if kwargs['twosided_plot']:
                xs = -dic_bb[bb]['data']['sma_arcsec'][chan] \
                        if kwargs['inverse_plot'] is not True \
                        else dic_bb[bb]['data'][source_dist_key][chan]
                ys = dic_bb[bb]['data'][source_dist_key][chan] \
                        if kwargs['inverse_plot'] is not True \
                        else -dic_bb[bb]['data']['sma_arcsec'][chan]
                ax.plot(xs,
                        ys,
                        dic_bb[bb]['sty'],
                        fillstyle=dic_bb[bb]['fsty'],
                        color=c_ell(vel_color, vel, bb, 'c_ell'),
                        markersize=dic_bb[bb]['markersize_centers'],
                        markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor,
                        markeredgewidth=dic_bb[bb]['markeredgewidth_centers'])


        if (tip_arcs is not None) and (bb in tip_arcs):
            ys, xs = [], []
            for chan in dic_bb[bb]['data'].index:
                ys.append(dic_bb[bb]['data']['mean_radius'][chan])
                xs.append(dic_bb[bb]['data']['dist_vla4b'][chan])
            xs, ys = np.array(xs), np.array(ys)
            xs_sorted, ys_sorted = [], []
            for i in xs.argsort():
                ys_sorted.append(ys[i])
                xs_sorted.append(xs[i])

            ax.fill_between(xs_sorted,
                            ys_sorted,
                            color=c_ell(vel_color, vel, bb, 'c_ell'),
                            alpha=0.3,)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_aspect('equal')
    if kwargs['inverse_plot'] is not True:
        ax.invert_yaxis()

    if vel_color and kwargs['colorbar_show']:
#        velocities_list = np.linspace(kwargs['v0'],kwargs['vf'],channel_f)
        cbar = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmax=np.abs(v_range[1]),
                                                       vmin=np.abs(v_range[0])),
               cmap=kwargs['cmap_ellipses']),
               ax=ax,
               orientation=kwargs['colorbar_orientation'],
               fraction=kwargs['colorbar_fraction'],
               pad=kwargs['colorbar_pad'],
               shrink=kwargs['colorbar_shrink'],
               aspect=kwargs['colorbar_aspect'],
               anchor=kwargs['colorbar_anchor'],
               panchor=kwargs['colorbar_panchor'],
               extend=kwargs['cbar_extend'])
        cbar_ticklabels = ['-{:.0f}'.format(i) for i in cbar.ax.get_yticks()]
        cbar.ax.set_yticklabels(cbar_ticklabels)

        if kwargs['colorbar_orientation'] == 'vertical':
            cbar.ax.set_ylabel('Velocity (km/s)')
        elif kwargs['colorbar_orientation'] == 'horizontal':
            cbar.ax.set_xlabel('Velocity (km/s)')

        ax.set_facecolor(kwargs['facecolor_background'])

    if 'inclination_angle' in kwargs:
        ax.text(0.05,0.97,'i={:d}°'.format(int(np.round(kwargs['inclination_angle']))),
                transform=ax.transAxes)

    if save_fig is not None:
        fig.savefig( '{}{}.pdf'.format(mf.default_params['path_save'],
                                      save_fig),
                    bbox_inches=mf.default_params['bbox_inches'])

    if return_ax:
        return ax

    if show_plot:
        plt.show()


def radial_flux_matrix(data_matrix, oper, angle_range,
                       angle_step, width_segment, pa_jet):
    """
    Given a data_matrix grid, performs an operation (sum or mean) over a radial
    segment of data of width width_segment. Returns a dictionary whose keys
    are the angles, and their data is te results of the operation
    """
    angle0, anglef = angle_range
    rot_angles = np.arange(angle0, anglef, angle_step)
    rotated_data = {rot_deg: rotate(data_matrix, angle=-rot_deg-pa_jet)
                    for rot_deg in rot_angles}

    segment_flux = {}
    for angle in rotated_data:
        ny, nx = np.shape(rotated_data[angle])
        pix0_x, pixf_x = int(nx/2-width_segment), int(nx/2+width_segment)
        pix0_y, pixf_y = int(ny/2), -1
        if oper == 'sum':
            segment_flux[angle] = \
                    np.sum(rotated_data[angle][pix0_y:pixf_y,pix0_x:pixf_x])
        elif oper == 'mean':
            segment_data = rotated_data[angle][pix0_y:pixf_y,pix0_x:pixf_x]
            segment_data_non0 = np.where(segment_data!=0)
            segment_flux[angle] = np.mean(segment_data[segment_data_non0])

    return segment_flux


def radial_flux(database_name, oper, angle_range, angle_step,
                width_segment, pa_jet, chans=None):
    """
    Reads a database and perform the operation over a range of angles for every channel.
    Returns a dictionary for the results for every angle and every channel, and the readed
    data from the database.
    """
    database_path = '{}{}.db'.format(mf.default_params['path_database'], database_name)
    database = shelve.open(database_path)
    ellip_data = {param: database[param] for param in database}
    database.close()

    segment_fluxes = {}
    channels = chans if chans is not None else [chan for chan in ellip_data]
    for channel in channels:
        data_matrix = ellip_data[channel]['ellip_data']
        segment_fluxes[channel] = radial_flux_matrix(data_matrix,
                                   oper=oper,
                                   angle_range=angle_range,
                                   angle_step=angle_step,
                                   width_segment=width_segment,
                                   pa_jet=pa_jet)

    return ellip_data, segment_fluxes


def angle_velocity_diagram(database, header, dic4radial_flux=None, return_ax=False,
                           output_name=None, **kwargs):
    """
    Creates an angle-velocity diagram. Database_name can be the name,
    (then a dictionary to pass it to radial_flux must be provided)
    of the database or a tuple of results_fit and segment_fluxes
    """
    for param in mf.default_params:
        kwargs[param] = kwargs[param] if param in kwargs \
        else mf.default_params[param]

    if dic4radial_flux is None:
        results_fit, segment_fluxes = database
    else:
        ellip_data, segment_fluxes = radial_flux(database, **dic4radial_flux)
        results_fit = import_table(database, header)

    sorted_chans = [chan for chan in np.sort([chan
                                              for chan in results_fit.index])]
    xs = [deg for deg in segment_fluxes[str(sorted_chans[0])]]
    ys = [results_fit['vel'][chan] for chan in sorted_chans]

    if 'ax' not in kwargs:
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot()
    else:
        pass

    av_diagram = np.array([[segment_fluxes[str(chan)][deg] for deg in xs]
                           for chan in sorted_chans])
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    im = ax.imshow(av_diagram,
             origin='lower',
             extent=extent,
             aspect='auto',
             cmap='jet',
             interpolation='bilinear')

    if kwargs['plot_cbar']:
        cbar = plt.colorbar(im, ax=ax,
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

    if kwargs['av_addjetdir']:
        ymin, ymax = ax.get_ylim()
        ax.vlines(0, ymin, ymax,
                  color=kwargs['av_linecolor'],
                  linewidth=kwargs['av_linewidth'],
                  linestyle=kwargs['av_linestyle']
                  )

    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Velocity (km/s)')

    plt.show()
    if output_name is not None:
        fig.savefig('{}{}.pdf'.format(kwargs['path_save'],output_name),
                    bbox_inches=kwargs['bbox_inches'])

    if return_ax:
        return ax


def int_emission_vel(database_name, header, chans=None, dic4radial_flux=None,
                     return_ax=False, output_name=None, **kwargs):
    """
    Plots the integrated emission a s a function of velocity. Database_name can be the name,
    (then a dictionary to pass it to radial_flux must be provided) of the database or a tuple of
    results_fit and segment_fluxes
    """

    for param in mf.default_params:
        kwargs[param] = kwargs[param] \
                if param in kwargs \
                else mf.default_params[param]

    database_path = '{}{}.db'.format(mf.default_params['path_database'], database_name)
    database = shelve.open(database_path)
    ellip_data = {param: database[param] for param in database}
    database.close()

    results_fit = import_table(database_name, header)

    channels = chans if chans is not None else [chan for chan in ellip_data]
    data_matrix_int = {channel: np.sum(ellip_data[channel]['ellip_data'])
                       for channel in channels}

    sorted_chans = [chan for chan in np.sort([chan for chan in results_fit.index])]
    xs = np.array([results_fit['vel'][chan] for chan in sorted_chans])
    ys = np.array([data_matrix_int[str(chan)] for chan in sorted_chans])

    if 'ax' not in kwargs:
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot()
    else:
        pass

    ax.plot(xs, ys)


def plot_tdyn(dic_bb,
              deproject=False,
              source_dist='dist_vla4b',
              ax=None,
              bb_list=None,
              wcor=False,
              **kwargs):
    for param in mf.default_params:
        kwargs[param] = kwargs[param] if param in kwargs \
        else mf.default_params[param]

    if ax is not None:
        pass
    else:
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        fig = plt.figure(figsize=(8,8))

        ax = plt.axes(rect_scatter)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False)

    wcor = '_wcor' if wcor else ''
    if deproject:
        vel_key = 'deprojected_vel' + wcor
        source_dist_key = 'deprojected_' + source_dist + wcor
    else:
        syst_vel = '' if 'syst_vel' not in kwargs else '_rel'
        vel_key = 'vel' + syst_vel
        source_dist_key = source_dist

    bb_list = bb_list if bb_list is not None else list(dic_bb)
    for bb in bb_list:
        bb_data = dic_bb[bb]['data']
        chans = bb_data.index
        vels = np.array([bb_data[vel_key][chan] for chan in chans])
        tdyns = np.array([bb_data['tdyn_windmodel'+wcor][chan] for chan in chans])
        markerfacecolor = dic_bb[bb]['c']
        markeredgecolor = dic_bb[bb]['markeredgecolor_centers'] \
                if dic_bb[bb]['markeredgecolor_centers'] is not None \
                else dic_bb[bb]['c']
        ax.plot(vels, tdyns, dic_bb[bb]['sty'],
                        markersize=dic_bb[bb]['markersize_centers'],
                        markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor,
                        markeredgewidth=dic_bb[bb]['markeredgewidth_centers'],
                        label=dic_bb[bb]['label'])
        if ax is None:
            ax_histy.hist(tdyns,
                          orientation='horizontal',
                          histtype='stepfilled',
                          color=dic_bb[bb]['c'])
            ax_histy.hist(tdyns,
                          orientation='horizontal',
                          histtype='step',
                          color=dic_bb[bb]['c'])

    ax.set_xlabel('Velocities (km/s)')
    ax.set_ylabel('Dynamical Time (yr)')
    ax.invert_xaxis()

    if kwargs['plot_legend']:
        ax.legend()

    if 'inclination_angle' in kwargs:
        ax.text(0.05,
                0.95,
                'i={:d}°'.format(int(np.round(kwargs['inclination_angle']))),
                transform=ax.transAxes)

    if kwargs['save_fig']:
        fig.savefig('{}{}.pdf'.format(mf.default_params['path_save'],
                                      'dynamical_times_i{:d}'.format(
                                          int(np.round(kwargs['inclination_angle']))
                                                                    )
                                     ),
                    bbox_inches=mf.default_params['bbox_inches'])

    return ax


def bluearm_points(chans, v_range, wcs=None, cmap='rainbow_r',
                   ax=None, **kwargs):
    for param in mf.default_params:
        kwargs[param] = kwargs[param] if param in kwargs \
        else mf.default_params[param]

    if ax is None:
        plt.figure(figsize=(kwargs['figsize'],kwargs['figsize']))
        ax = plt.subplot(projection=wcs)

    path_points = {chan:kwargs['path_folder_points']+str(chan)
                   for chan in chans}

    get_coords = lambda line: [float(i.rstrip('deg'))
                 for i in line.strip('symbol [[').split(']')[0].split(',')]

    files_points = {}
    points = {}
    for chan in chans:
        files_points[chan] = open(path_points[chan])
        points[chan] = np.array([get_coords(line) for line in
                                 files_points[chan] if line.startswith('symbol')])
        files_points[chan].close()

    vels_chan = {chan: np.linspace(v_range[0], v_range[1], len(chans))[i]
                 for i,chan in enumerate(chans)}
    for chan in points:
        ax.plot(points[chan][:,0], points[chan][:,1],
                marker=kwargs['markerbluearm_style'],
                linestyle='',
                transform=ax.get_transform('icrs'),
                color=get_color(v_range,
                                vels_chan[chan],
                                color_length=300,
                                cmap=cmap))

    cmap_cbar = cmap.rstrip('_r') if cmap[-2:]=='_r' else cmap+'_r'
    cbar = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmax=v_range[1],
                                                                vmin=v_range[0]),
                                          cmap=cmap_cbar,),
                         ax=ax)

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

    ax.set_aspect('equal')
    ax.set_xlabel(kwargs['icrs_xlabel'])
    ax.set_ylabel(kwargs['icrs_ylabel'])
    cbar.ax.set_ylabel('Velocity (km/s)')
    if kwargs['rotate_ticktext_yaxis'] is not None:
        ax.coords[1].set_ticklabel(rotation=kwargs['rotate_ticktext_yaxis'])

    ax.set_facecolor(kwargs['facecolor_background'])


def debug():
    pass
