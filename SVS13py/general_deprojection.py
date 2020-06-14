import numpy as np

from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord, concatenate
from astropy import units as u

from photutils import SkyEllipticalAperture, EllipticalAperture

from scipy.interpolate import interp1d
from scipy import optimize

from scipy.ndimage.filters import gaussian_filter1d

import matplotlib.pyplot as plt

import SVS13py.mf as mf
import SVS13py.main_functions as main_functions


def generate_pv_line_coordinates(angle, box, wcs, n_points):
    """
    This function generates the PV pixel and sky position given
    the its PA in degrees.
    """
    angle_pvline = np.pi/180 * (angle+90)

    def y_pv(xp, m, x0, y0): return m * (xp-x0) + y0

    vla4b_sky = SkyCoord(*mf.default_params['vla4b_deg'], unit='deg')
    vla4b_pixel = skycoord_to_pixel(vla4b_sky, wcs)
#   aframe = vla4b_sky.skyoffset_frame()
#   vla4b_offset = vla4b_sky.transform_to(aframe)

    x_first, y_first = box[1]
    x_last, y_last = box[0]
    xs_pixel = np.array([x for x in np.linspace(x_first, x_last, n_points)])

    ys_pixel = np.array([y_pv(x,
                        np.tan(angle_pvline),
                        vla4b_pixel[0],
                        vla4b_pixel[1],
                              ) for x in xs_pixel])

    xys_sky_PV = np.array([pixel_to_skycoord(x, y, wcs)
                           for x, y in zip(xs_pixel, ys_pixel)])

    return xys_sky_PV


def ellipse_points_calc(x0, y0, sma, eps, pa, n_points, wcs):
    """
    Calculates the points of an ellipse and returns them in pixel and sky
    coordinates. Cannot work with inputs in skycoordinates: it would render the
    wrong pixel, remember that distances in RA depends on the latitud. Use
    pixel coordinates instead, and convert them later.
    """
    def x(theta): return sma * np.cos(theta)
    def y(theta): return sma * (1-eps) * np.sin(theta)
    thetas = np.linspace(0, 2*np.pi, n_points)
    xs = np.array([x(theta) for theta in thetas])
    ys = np.array([y(theta) for theta in thetas])
    rot_coords = [main_functions.rot({'x': x, 'y': y, 'z': 0}, 'z', pa)
                  for x, y in zip(xs, ys)]
    xs_rot = [rot_coord['x']+x0 for rot_coord in rot_coords]
    ys_rot = [rot_coord['y']+y0 for rot_coord in rot_coords]
    xys_sky = np.array([pixel_to_skycoord(_x, _y, wcs)
                        for _x, _y in zip(xs_rot, ys_rot)])
    xys_pixel = np.array([[x, y] for x, y in zip(xs_rot, ys_rot)])
    return xys_pixel, xys_sky


def spetial_points_calc(x0, y0, sma, eps, pa, n_points, wcs, dist_center,
                        xys_sky_PV):
    """
    Calculates the points for the three interesting points in the elliptical
    fits from which you can extract important information from. This are where
    phi is 0, 90 and 180

    Parameters
    ----------
    x0, y0, sma, eps, pa: float
        ellipse parameters
    wcs:
        wcs of the image
    n_points: int
        number of points of the pv line. The higher the more precises the
        value of the x_phis are
    dist_center: float
        this are the values set to x_phi90, the distances of the
        centers of the ellipses
    """

    vla4b_sky = SkyCoord(*mf.default_params['vla4b_deg'], unit='deg')

    ellipse_pixel, ellipse_sky = ellipse_points_calc(x0,
                                                     y0,
                                                     sma,
                                                     eps,
                                                     pa,
                                                     n_points,
                                                     wcs)
    ellipse_sky_cat = concatenate(ellipse_sky)
    xys_sky_PV_cat = concatenate(xys_sky_PV)

    aper = EllipticalAperture((x0, y0), sma, sma*(1.-eps), pa)
    aper_sky = aper.to_sky(wcs)
    ell_center_sky = aper_sky.positions
    sma_arcsec = aper_sky.a.value
    sminora_arcsec = aper_sky.b.value
    xys_PV_inrange = np.array([xys for xys in xys_sky_PV_cat
                               if (xys.separation(ell_center_sky).arcsec
                                   <= 1.1 * sma_arcsec)
                               and (xys.separation(ell_center_sky).arcsec
                                    >= 0.5 * sminora_arcsec)])
    xys_180_PV = np.array([xys for xys in xys_PV_inrange
                           if vla4b_sky.separation(xys).arcsec
                           < vla4b_sky.separation(ell_center_sky).arcsec])
    xys_0_PV = np.array([xys for xys in xys_PV_inrange
                         if vla4b_sky.separation(xys).arcsec
                         > vla4b_sky.separation(ell_center_sky).arcsec])

    distances_180 = np.array([[xys.separation(ell).arcsec
                               for ell in ellipse_sky_cat]
                              for xys in xys_180_PV])
    distances_0 = np.array([[xys.separation(ell).arcsec
                             for ell in ellipse_sky_cat]
                            for xys in xys_0_PV])

    idx_180 = np.where(distances_180 == np.min(distances_180))
    idx_0 = np.where(distances_0 == np.min(distances_0))

    p180_sky = ellipse_sky_cat[idx_180[1]]
    p0_sky = ellipse_sky_cat[idx_0[1]]

    xp_phi180 = p180_sky.separation(vla4b_sky).arcsec
    xp_phi90 = dist_center
    xp_phi0 = p0_sky.separation(vla4b_sky).arcsec

    return [[ellipse_pixel, ellipse_sky], [p180_sky, p0_sky],
            [xp_phi180, xp_phi90, xp_phi0]]


def plot_ellipse_spetial_points(dic_bb, angle, box, wcs, bb, chan, ax=None,
                                **kwargs):
    """
    Plot the ellipse from the fit results and draws the position of x_phi180
    and x_phi0, which intersects the ellipse with the PV line
    """
    for kwarg in mf.default_params:
        kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
           else mf.default_params[kwarg]

    vla4b_sky = SkyCoord(*mf.default_params['vla4b_deg'], unit='deg')
#    vla4b_pixel = skycoord_to_pixel(vla4b_sky, wcs)

    xys_sky = generate_pv_line_coordinates(angle, box, wcs,
                                           kwargs['n_points_PV'])

    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(projection=wcs)
    else:
        pass

    ax.plot(vla4b_sky.ra.deg,
            vla4b_sky.dec.deg,
            '*',
            transform=ax.get_transform('world'))

#   Plots the PV diagram
    xs_sky = np.array([xy.ra.deg for xy in xys_sky])
    ys_sky = np.array([xy.dec.deg for xy in xys_sky])
    ax.plot(xs_sky,
            ys_sky,
            transform=ax.get_transform('world'),
            )

#   Plots the special points
    for p in [dic_bb[bb]['data']['p180_sky'][chan],
              dic_bb[bb]['data']['p0_sky'][chan]]:
        ax.plot(p.ra.value,
                p.dec.value,
                '+g',
                zorder=3,
                transform=ax.get_transform('world'))

#   Plots the point of the ellipse
    for point in dic_bb[bb]['ellipse_sky'][chan]:
        ax.plot(point.ra.value,
                point.dec.value,
                '.w',
                transform=ax.get_transform('world')
                )

#   Plots a line of the elliptical fit
    positions_deg = SkyCoord(ra=dic_bb[bb]['data']['x0_RA'][chan]*u.deg,
                             dec=dic_bb[bb]['data']['y0_DEC'][chan]*u.deg)
    a_deg = dic_bb[bb]['data']['sma_deg'][chan]*u.deg

    b_deg = dic_bb[bb]['data']['sma_deg'][chan] \
        * (1. - dic_bb[bb]['data']['eps'][chan]) * u.deg
    pa_deg = (dic_bb[bb]['data']['pa'][chan] - np.pi/2.) * 180 / np.pi * u.deg
    aper = SkyEllipticalAperture(positions_deg, a_deg, b_deg, pa_deg)
    aper2plot = aper.to_pixel(wcs)
    aper2plot.plot(ax,
                   color='b',
                   alpha=0.5)

    ax.set_aspect("equal")


def depr_func_gen(dic_bb, bb, i_angle, sigmas_xps=None):
    """
    This function generates a dictionary of functions that are useful for the
    deprojection
    """
#   Data from bb_dic:
    zs = np.array(dic_bb[bb]['data']['dist_vla4b'] / np.sin(i_angle))
    Rs = np.array(dic_bb[bb]['data']['mean_radius'])
    rel_vel = np.array(-dic_bb[bb]['data']['vel_rel'])
    phis = ['0', '90', '180']
    phi_rad = {'0': 0, '90': np.pi/2, '180': np.pi}
    xps_phis = {phi: np.array(dic_bb[bb]['data']['x_phi'+phi])
                for phi in phis}

#   Gaussian filtering for smoothing the interpolation:
    sigmas_xps = sigmas_xps if sigmas_xps is not None else \
        {'0': 4, '90': 4, '180': 8}
    xps_sorted_index = {phi: np.argsort(xps_phis[phi]) for phi in phis}
    xps_phis_g = {phi: gaussian_filter1d([xps_phis[phi][i] for i in
                                          xps_sorted_index[phi]],
                                         sigmas_xps[phi])
                  for phi in phis}

    rel_vel_g = {phi: gaussian_filter1d([rel_vel[i] for i in
                                         xps_sorted_index[phi]],
                                        sigmas_xps[phi])
                 for phi in phis}

#   Interpolations
    R_z_inter = interp1d(zs,
                         Rs,
                         kind='linear',
                         fill_value='extrapolate',)

    vzp_xp_phi_inter = {phi: interp1d(xps_phis_g[phi],
                                      rel_vel_g[phi],
                                      kind="linear",
                                      fill_value="extrapolate",)
                        for phi in phis}

    R_z = lambda z: R_z_inter(z) \
        if (z >= np.min(zs)) and (z <= np.max(zs)) \
        else np.nan

#   Dictionary of functions
    vzp_xp_phi = {phi: lambda xp, phi=phi: vzp_xp_phi_inter[phi](xp)
                  if xp >= np.min(xps_phis[phi]) and
                  xp <= np.max(xps_phis[phi])
                  else np.nan for phi in phis}

    xp_phi_z = {phi: lambda z, phi=phi: R_z(z) * np.cos(phi_rad[phi]) *
                np.cos(i_angle) + z * np.sin(i_angle)
                for phi in phis}

#   ---------------------
    xp_phi_vzp_inter = {phi: interp1d(rel_vel_g[phi],
                                      xps_phis_g[phi],
                                      kind='linear',
                                      fill_value='extrapolate',)
                        for phi in phis}

    xp_phi_vzp = {phi: lambda vzp, phi=phi: xp_phi_vzp_inter[phi](vzp)
                  if vzp >= np.min(rel_vel_g[phi]) and
                  vzp <= np.max(rel_vel_g[phi])
                  else np.nan for phi in phis}

    R_xphi_inter = {phi: interp1d(xps_phis_g[phi],
                                  Rs,
                                  kind='linear',
                                  fill_value='extrapolate',)
                    for phi in phis}

    R_xphi = {phi: lambda xp, phi=phi: R_xphi_inter[phi](xp)
              if xp >= np.min(xps_phis[phi]) and
              xp <= np.max(xps_phis[phi])
              else np.nan for phi in phis}

    xp_phi_equal_0 = {phi: lambda z, xp, phi=phi: xp - xp_phi_z[phi](z)
                      for phi in phis}

    z_xp_phi = {phi: lambda xp, phi=phi: optimize.brentq(
                         lambda z, phi=phi: xp_phi_equal_0[phi](z, xp),
                         np.min(zs),
                         np.max(zs)) for phi in phis}

#     def i_equal_0(vzp, i):
#         _zs = np.array(dic_bb[bb]['data']['dist_vla4b'] / np.sin(i))
#         _R_z_inter = interp1d(_zs,
#                               Rs,
#                               kind='linear',
#                               fill_value='extrapolate',)
#
#         _R_z = lambda z: _R_z_inter(z) \
#             if (z >= np.min(_zs)) and (z <= np.max(_zs)) \
#             else np.nan
#
#         _xp_phi_z = {phi: lambda z, phi=phi: _R_z(z) * np.cos(phi_rad[phi]) *
#                      np.cos(i) + z * np.sin(i)
#                      for phi in phis}
#
#         _xp_phi_equal_0 = {phi: lambda z, xp, phi=phi: xp - _xp_phi_z[phi](z)
#                            for phi in phis}
#
#         _z_xp_phi = {phi: lambda xp, phi=phi: optimize.brentq(
#                               lambda z, phi=phi: _xp_phi_equal_0[phi](z, xp),
#                               np.min(_zs),
#                               np.max(_zs)) for phi in phis}
#
#         xp180 = xp_phi_vzp['180'](vzp)
#         xp0 = xp_phi_vzp['0'](vzp)
#         R_xp180 = _R_z(_z_xp_phi['180'](xp180))
#         R_xp0 = _R_z(_z_xp_phi['0'](xp0))
#         equals_0 = (R_xp180 + R_xp0) * np.cos(i) \
#             - (_z_xp_phi['0'](xp0) - _z_xp_phi['180'](xp180)) * np.sin(i) \
#             - xp0 + xp180
#         return equals_0
#
#     i_calc = lambda vzp: optimize.brentq(
#                  lambda i: i_equal_0(vzp, i),
#                  np.min(np.pi / 11),
#                  np.max(np.pi / 2.1))

#   --------------------

    vzp_z_phi = {phi: lambda z, phi=phi: vzp_xp_phi[phi](
                                    xp_phi_z[phi](z))
                 for phi in phis}

    def v_z_180_0(z): return 1/np.sin(2*i_angle) \
        * np.sqrt(vzp_z_phi['0'](z)**2
                  + vzp_z_phi['180'](z)**2
                  - 2 * vzp_z_phi['0'](z)
                  * vzp_z_phi['180'](z)
                  * np.cos(2*i_angle))

    def v_z_180_90(z): return 1/np.sin(i_angle) \
        * np.sqrt(vzp_z_phi['180'](z)**2
                  + (vzp_z_phi['90'](z) / np.cos(i_angle))**2
                  - 2 * vzp_z_phi['180'](z)
                  * vzp_z_phi['90'](z))

    def v_z_90_0(z): return 1/np.sin(i_angle) \
        * np.sqrt((vzp_z_phi['90'](z) / np.cos(i_angle))**2
                  + vzp_z_phi['0'](z)**2
                  - 2 * vzp_z_phi['90'](z)
                  * vzp_z_phi['0'](z))

    v_zs = {'180_0': v_z_180_0, '180_90': v_z_180_90, '90_0': v_z_90_0}

    def alpha_180(z, s=0, v='180_90'):
        return (-1)**s * np.arccos(vzp_z_phi['180'](z) / v_zs[v](z)) + i_angle

    def alpha_90(z, s=0, v='180_90'):
        return (-1)**s * np.arccos(vzp_z_phi['90'](z)
                                   / (v_zs[v](z)*np.cos(i_angle)))

    def alpha_0(z, s=0, v='90_0'):
        return (-1)**s * np.arccos(vzp_z_phi['0'](z) / v_zs[v](z)) - i_angle

    alphas = {'180': alpha_180, '90': alpha_90, '0': alpha_0}

    return zs, R_z, R_z_inter, xp_phi_z, xps_phis, xp_phi_equal_0, z_xp_phi, \
        vzp_xp_phi_inter, vzp_xp_phi, vzp_z_phi, v_zs, alphas,
    # i_equal_0, i_calc
