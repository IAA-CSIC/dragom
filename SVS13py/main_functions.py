import numpy as np

import matplotlib.pyplot as plt

import functools

from astropy.io import fits

from astropy import units as u
from astropy import constants as const

import subprocess

process = subprocess.Popen(["whoami"], stdout=subprocess.PIPE)
result = process.communicate()[0]
user = result.decode('utf-8').rstrip('\n')

power_th = 2.
theta_0 = 0
theta_f = 1/2.

def theta_powerspace(ntheta, power=power_th, theta_0=theta_0, theta_f=theta_f):
    array_thetas = np.array([theta_f * np.pi * x**power
                             for x in np.linspace(theta_0 * np.pi, 1, ntheta)])
    return array_thetas

default_kwargs = {'figsize': 10,
                  'arrowstyle': '->',
                  # 'rs2D': np.arange(-0.1, 0.1, 0.005),
                  'zs2D': np.arange(0, 1, 0.01),
                  'step_arrow': 2,
                  'thetas': theta_powerspace(100,),
                  'power_th': power_th,
                  'theta_0': theta_0,
                  'theta_f': theta_f,
                  'phis': np.linspace(0, 2*np.pi, 100),
                  'phi_0': 0.,
                  'phi_f': 2.,
                  'scatter_size': 10,
                  'scatter_alpha': 0.2,
                  'colorbar_shrink': 1,
                  'colorbar_aspect': 20,
                  'show_arrows': True,
                  # 'zs_arrows': np.linspace(0.3,1.5, 3),
                  'thetas_arrows': np.linspace(theta_0*np.pi, theta_f*np.pi, 8),
                  'color_arrow': 'k',
                  'length_arrow': 0.01,
                  'x_label': 'x (arcsec)',
                  'y_label': 'y (arcsec)',
                  'z_label': 'z (arcsec)',
                  'refgeom': None,
                  'plot_refgeom': False,
                  'rg_scatter_size': 1,
                  'rg_scatter_alpha': 0.1,
                  'density': False,
                  'weights': None,
                  'cube_origin': 'lower',
                  'cube_aspect': 'equal',
                  'boxstyle': 'square',
                  'box_facecolor': 'white',
                  'x_box': 0.05,
                  'y_box': 0.95,
                  'box_fontsize': 10,
                  'interpolation': None,
                  'filterrad': 4,
                  'cmap': 'jet_r',
                  'fraction': 0.1,
                  'pad': 0.1,
                  'save_fig': False,
                  'show_plot': True,
                  'save_format': 'eps',
                  'gaussian_filter': None,
                  'change_header': True,
                  '12CO_J3-2': 3.4579599*10**11,  # Hz
                  'plot_source': True,
                  'mark_velocities': None,
                  'mark_velocities_color_array': np.array([0/256, 0/256, 0/256, 1]),
                  'mark_rel_width': 0.01,
                  'markerstar_size': 10,
                  'indep_variable': 'theta',
                  'path_save_cube': '/home/{}/data/model_cubes/'.format(user),
                  'plot_zaxis': False,
                  'inv_zorder': True,
                  'plot_cbar': True,
                  }




#histogramdd kwargs
#density : bool, optional
#    If False, the default, returns the number of samples in each bin.
#    If True, returns the probability *density* function at the bin,

#normed : bool, optional
#    An alias for the density argument that behaves identically. To avoid
#    confusion with the broken normed argument to `histogram`, `density`
#    should be preferred.

#weights : (N,) array_like, optional
#    An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`.
#    Weights are normalized to 1 if normed is True. If normed is False,
#    the values of the returned histogram are equal to the sum of the
#    weights belonging to the samples falling into each bin.


#imshow kwargs
#interpolation : str, optional
#    The interpolation method used. If *None*
#    :rc:`image.interpolation` is used, which defaults to 'nearest'.
#
#    Supported values are 'none', 'nearest', 'bilinear', 'bicubic',
#    'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
#    'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
#    'lanczos'.
#
#    If *interpolation* is 'none', then no interpolation is performed
#    on the Agg, ps, pdf and svg backends. Other backends will fall back
#    to 'nearest'. Note that most SVG renders perform interpolation at
#    rendering and that the default interpolation method they implement
#    may differ.
#
#    See
#    :doc:`/gallery/images_contours_and_fields/interpolation_methods`
#    for an overview of the supported interpolation methods.
#
#    Some interpolation methods require an additional radius parameter,
#    which can be set by *filterrad*. Additionally, the antigrain image
#    resize filter is controlled by the parameter *filternorm*.

default_header_1920 = {
    'SIMPLE': True,
    'BITPIX': -32,
    'NAXIS': 4,
    'NAXIS1': 1600,  # x
    'NAXIS2': 2240,  # y
    'NAXIS3': 1920,  # v
    'NAXIS4': 1,
    'EXTEND': True,
    'BSCALE': 1.0,
    'BZERO': 0.0,
    'BMAJ': 4.542971236838e-05,
    'BMIN': 2.322266706162e-05,
    'BPA': -2.702255249023,
    'BTYPE': 'Intensity',
    'OBJECT': 'SVS13',
    'BUNIT': 'Jy/beam',
    'RADESYS': 'ICRS',
    'LONPOLE': 180.0,
    'LATPOLE': 31.26777777778,
    'PC1_1': 1.0,
    'PC2_1': 0.0,
    'PC3_1': 0.0,
    'PC4_1': 0.0,
    'PC1_2': 0.0,
    'PC2_2': 1.0,
    'PC3_2': 0.0,
    'PC4_2': 0.0,
    'PC1_3': 0.0,
    'PC2_3': 0.0,
    'PC3_3': 1.0,
    'PC4_3': 0.0,
    'PC1_4': 0.0,
    'PC2_4': 0.0,
    'PC3_4': 0.0,
    'PC4_4': 1.0,
    'CTYPE1': 'RA---SIN',
    'CRVAL1': 52.26562499999,
    'CDELT1': -3.333333333333e-06,
    'CRPIX1': 801.0,
    'CUNIT1': 'deg',
    'CTYPE2': 'DEC--SIN',
    'CRVAL2': 31.26777777778,
    'CDELT2': 3.333333333333e-06,
    'CRPIX2': 1121.0,
    'CUNIT2': 'deg',
    'CTYPE3': 'FREQ',
    'CRVAL3': 3.456936670000E+11,
    'CDELT3': 6.103500000000E+05,
    'CRPIX3': 1.000000000000E+00,
    'CUNIT3': 'Hz',
    'CTYPE4': 'STOKES',
    'CRVAL4': 1.0,
    'CDELT4': 1.0,
    'CRPIX4': 1.0,
    'CUNIT4': '',
    'PV2_1': 0.0,
    'PV2_2': 0.0,
    'RESTFRQ': 345795990000.0,
    'SPECSYS': 'LSRK',
    'ALTRVAL': 345669482138.7,
    'ALTRPIX': 1.0,
    'VELREF': 257,
    'COMMENT': 'casacore non-standard usage: 4 LSD, 5 GEO, 6 SOU, 7 GAL',
    'TELESCOP': 'ALMA',
    'OBSERVER': 'guillem',
    'DATE-OBS': '2016-09-09T08:27:20.928001',
    'TIMESYS': 'UTC',
    'OBSRA': 52.26562499999,
    'OBSDEC': 31.26777777778,
    'OBSGEO-X': 2225142.180269,
    'OBSGEO-Y': -5440307.370349,
    'OBSGEO-Z': -2481029.851874,
    'DATE': '2019-10-18T09:33:43.528000',
    'ORIGIN': 'CASA 5.4.0-68'}


hdr_lsr = {
    'SIMPLE': True,
    'BITPIX': -32,
    'NAXIS': 4,
    'NAXIS1': 1600,
    'NAXIS2': 2240,
    'NAXIS3': 345,
    'NAXIS4': 1,
    'EXTEND': True,
    'BSCALE': 1.0,
    'BZERO': 0.0,
    'BMAJ': 4.539020773437e-05,
    'BMIN': 2.322353008721e-05,
    'BPA': -2.697280883789,
    'BTYPE': 'Intensity',
    'OBJECT': 'SVS13',
    'BUNIT': 'Jy/beam',
    'RADESYS': 'ICRS',
    'LONPOLE': 180.0,
    'LATPOLE': 31.26777777778,
    'PC1_1': 1.0,
    'PC2_1': 0.0,
    'PC3_1': 0.0,
    'PC4_1': 0.0,
    'PC1_2': 0.0,
    'PC2_2': 1.0,
    'PC3_2': 0.0,
    'PC4_2': 0.0,
    'PC1_3': 0.0,
    'PC2_3': 0.0,
    'PC3_3': 1.0,
    'PC4_3': 0.0,
    'PC1_4': 0.0,
    'PC2_4': 0.0,
    'PC3_4': 0.0,
    'PC4_4': 1.0,
    'CTYPE1': 'RA---SIN',
    'CRVAL1': 52.26562499999,
    'CDELT1': -3.333333333333e-06,
    'CRPIX1': 801.0,
    'CUNIT1': 'deg',
    'CTYPE2': 'DEC--SIN',
    'CRVAL2': 31.26777777778,
    'CDELT2': 3.333333333333e-06,
    'CRPIX2': 1121.0,
    'CUNIT2': 'deg',
    'CTYPE3': 'FREQ',
    'CRVAL3': 345693667000.0,
    'CDELT3': 610350.0,
    'CRPIX3': 1.0,
    'CUNIT3': 'Hz',
    'CTYPE4': 'STOKES',
    'CRVAL4': 1.0,
    'CDELT4': 1.0,
    'CRPIX4': 1.0,
    'CUNIT4': '',
    'PV2_1': 0.0,
    'PV2_2': 0.0,
    'RESTFRQ': 345795990000.0,
    'SPECSYS': 'LSRK',
    'ALTRVAL': 88710.29325682,
    'ALTRPIX': 1.0,
    'VELREF': 257,
    'COMMENT': 'casacore non-standard usage: 4 LSD, 5 GEO, 6 SOU, 7 GAL',
    'TELESCOP': 'ALMA',
    'OBSERVER': 'guillem',
    'DATE-OBS': '2016-09-09T08:27:20.928001',
    'TIMESYS': 'UTC',
    'OBSRA': 52.26562499999,
    'OBSDEC': 31.26777777778,
    'OBSGEO-X': 2225142.180269,
    'OBSGEO-Y': -5440307.370349,
    'OBSGEO-Z': -2481029.851874,
    'DATE': '2019-12-05T13:41:06.653000',
    'ORIGIN': 'CASA 5.4.0-68'}


default_header = hdr_lsr

def create_header(dic_of_changes=None):
    hdr = fits.Header()
    for key in default_header:
        hdr[key] = default_header[key]
    if dic_of_changes is not None:
        for key in dic_of_changes:
            hdr[key] = dic_of_changes[key]
    else:
        pass
    return hdr


def read_file(file_path):
    file = [line for line in open(file_path)]
    return {'arcsec': [float(line.split(', ')[0]) for line in file],
            'vel': [float(line.split(', ')[1].strip('\n')) for line in file]}

def rot(vec, axis, i):
    if axis=='x':
        rotated_vec = {'x':vec['x'], 'y':vec['y']*np.cos(i)-vec['z']*np.sin(i),
                      'z':vec['y']*np.sin(i)+vec['z']*np.cos(i)}
    elif axis=='y':
        rotated_vec = {'x':vec['x']*np.cos(i)+vec['z']*np.sin(i), 'y':vec['y'],
                       'z':-vec['x']*np.sin(i)+vec['z']*np.cos(i)}
    elif axis=='z':
        rotated_vec = {'x':vec['x']*np.cos(i)-vec['y']*np.sin(i),
                       'y':vec['x']*np.sin(i)+vec['y']*np.cos(i),
                       'z':vec['z']}
    return rotated_vec

def equal_aspect(func):
    """
    You use this function as decorator, to ensure that axis have the same scale.
    ax.aspect('equal') is not yet available in 3D, so be aware that the size of the plot
    axis will affect the geometry of your model.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        X, Y, Z, ax = func(*args,**kwargs)
        # Create cubic bounding box to simulate equal aspect ratio
#        ax = fig.gca(projection='3d')
        X, Y, Z = np.array(X), np.array(Y), np.array(Z)
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() \
                + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() \
                + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() \
                + 0.5*(Z.max()+Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
               ax.plot([xb], [yb], [zb], 'w')
        print('ax.set_aspect("equal") in 3D is not available yet, you used an equal aspect decorator to at least render axis with the same scale.')
        plt.grid()
        plt.show()
    return wrapper


def kwargs_as_defaults(func):
    """
    TO BE BUILD
    Decorator that passes default_params as kwargs.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
            else self.__dict__[kwarg]
        func(*args, **kwargs)
    return wrapper


def debug():
    #Function to define attributes and retrieve it when desired
    pass
