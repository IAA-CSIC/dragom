import numpy as np
from numpy.linalg import eig, inv

from scipy.optimize import minimize
#from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.colors import to_hex
import matplotlib
from matplotlib.widgets import Slider, Button, RadioButtons
#try:
#    matplotlib.use('Qt4Agg')
#except:
#    print("Qt4Agg backend could not be load")
#matplotlib.rcsetup.interactive_bk ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg',
#'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo']

from IPython.display import display, clear_output

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import copy

from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.modeling import models, fitting

from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model, EllipseSample, EllipseFitter, Isophote
from photutils import EllipticalAperture, EllipticalAnnulus, CircularAperture

import warnings

from itertools import product

import shelve

from SVS13py.SVS13py import show_slice_cube, mosaic_slices, calc_moment, collapse_chans, fit_ellipse, create_cube
import SVS13py.mf as mf
from SVS13py.vertslider import VertSlider




class EllipseFitter(object):
    init_params = {'x0':705, 'y0':975, 'sma':50, 'eps':0.01, 'pa':np.pi}
    init_box = [[556,786],[855,1164]]


    def __init__(self,
                 image_cube,
                 header,
                 db_name='database',
                 x0=None,
                 y0=None,
                 sma=None,
                 eps=None,
                 pa=None,
                 box='init',
                 **kwargs):
        """
        Kwargs can be: init_chan
        """
        self.params = {'x0':x0, 'y0':y0, 'sma':sma, 'eps':eps, 'pa':pa}
        for param in self.params:
            self.params[param] = self.params[param] \
                    if self.params[param] is not None \
                    else self.init_params[param]
        self.geom = EllipseGeometry(**self.params)
        self.box = self.init_box if box=='init' else box
        self.fig = plt.figure(figsize = (8,8))
        self.fig.subplots_adjust(left=0.25, bottom=0.35)
        self.header = header
        self.nchans = self.header['NAXIS3']
        self.init_chan = int(np.round(self.nchans/2))
        if self.header is not None:
            self.wcs = WCS(header).celestial
            self.beam_size_pix = (self.header['BMAJ']+self.header['BMIN']) * 0.5 / self.header['CDELT2']
            self.ax = plt.subplot(projection=self.wcs)
        else:
            self.header=None
            self.wcs=None
            self.beam_size_pix=None
            self.ax = self.fig.add_subplot()
        self.ax_text = self.fig.add_axes([0.05,0.025,0.1,0.04])
        self.ax_text.set_axis_off()
        self.chan = self.init_chan
        self.image_cube = image_cube
        self.operation = kwargs['operation'] if 'operation' in kwargs else None
        self.interpolation = kwargs['interpolation'] \
                             if 'interpolation' in kwargs \
                             else 'bilinear'
        self.filterrad = kwargs['filterrad'] if 'filterrad' in kwargs else 4.0
        self.width_chan = 3
        self.color_ref_ellip = 'r'

        self.max_vmax = np.max([np.max(x) for x in self.image_cube])
        #self.max_vmax = 0.1
        self.vmax = 0.08
        self.vcenter = 0.03
        self.vmin = 0.01
        self.cmap = kwargs['cmap'] if 'cmap' in kwargs else mf.default_params['cmap']
        self.norm = 'divnorm'

        self.v0 = kwargs['v0'] if 'v0' in kwargs else mf.default_params['v0']
        self.vf = kwargs['vf'] if 'vf' in kwargs else mf.default_params['vf']

        self.photomorph = kwargs['photomorph'] if 'photomorph' in kwargs \
                          else 'ellipse'
        self.sample = None
        self.ellip_data = None
        self.q2min = kwargs['q2min'] if 'q2min' in kwargs else 'mean'
        self.mean_intens = 0
        self.sum_intens = 0
        self.fit_results = {}
        self.im = None
        self.fit_tol = 1*10**(-8)
        self.fit_method = None
        self.pickling = False
        self.db_name = db_name
        self.path_database = '{}{}.db'.format(mf.default_params['path_database'],
                                              self.db_name)
        self.limit_fit = kwargs['limit_fit'] if 'limit_fit' in kwargs \
                         else mf.default_params['limit_fit']
        self.slider_ax = None
        self.channels_ax = None
        self.fit_button_ax = None
        self.next_chan_button_ax = None
        self.prev_chan_button_ax = None
        self.max_cbar_ax = None
        self.center_cbar_ax = None
        self.min_cbar_ax = None
        self.create_axes()

        self.param_sliders = None
        self.channels_slider = None
        self.max_cbar_slider = None
        self.center_cbar_slider = None
        self.min_cbar_slider = None
        self.fit_button = None
        self.next_chan_button = None
        self.prev_chan_button = None
        self.update_buttons()

        self.plot_image()
        self.update_ellipse(first_time=True)


    def create_axes(self):
        """
        CAUTION: axis must be created and removed in the same order!
        """
        self.slider_ax = {param: self.fig.add_axes([0.25,0.25-i*0.03,0.65,0.03])
                          for i,param in enumerate(self.params)}
        self.channels_ax = self.fig.add_axes([0.25,
                                              0.25-len(self.slider_ax)*0.03,
                                              0.65,
                                              0.03])
        self.fit_button_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        self.next_chan_button_ax = self.fig.add_axes([0.6, 0.025, 0.1, 0.04])
        self.prev_chan_button_ax = self.fig.add_axes([0.5, 0.025, 0.1, 0.04])
        self.max_cbar_ax = self.fig.add_axes([0.05,0.3,0.03,0.6])
        self.center_cbar_ax = self.fig.add_axes([0.1,0.3,0.03,0.6])
        self.min_cbar_ax = self.fig.add_axes([0.15,0.3,0.03,0.6])

    def remove_axes(self):
        for param in self.params:
            self.slider_ax[param].remove()
        self.channels_ax.remove()
        self.fit_button_ax.remove()
        self.next_chan_button_ax.remove()
        self.prev_chan_button_ax.remove()
        self.max_cbar_ax.remove()
        self.center_cbar_ax.remove()
        self.min_cbar_ax.remove()

    def update_buttons(self,):
        """
        Updates the state of the sliders and buttons (for example, after a fit)
        """
        self.remove_axes()
        self.create_axes()
        if self.box is not None:
            self.limit_slider = {'x0_u':self.box[1][0],
                                 'x0_l':self.box[0][0],
                                 'y0_u':self.box[1][1],
                                 'y0_l':self.box[0][1],
                                 'sma_u':self.params['sma']+100,
                                 'sma_l':0,
                                 'eps_u':0.8,
                                 'eps_l':0.01,
                                 'pa_u':2*np.pi, 'pa_l':0}
        else:
            self.limit_slider = {'x0_u':self.ax.get_xlim()[1],
                                 'x0_l':self.ax.get_ylim()[1],
                                 'y0_u':self.ax.get_ylim()[1],
                                 'y0_l':self.ax.get_ylim()[1],
                                 'sma_u':self.params['sma']+100,
                                 'sma_l':0,
                                 'eps_u':0.8,
                                 'eps_l':0.01,
                                 'pa_u':2*np.pi,
                                 'pa_l':0}
        self.param_sliders = {param: Slider(self.slider_ax[param],
                                            param,
                                            self.limit_slider[param+'_l'],
                                            self.limit_slider[param+'_u'],
                                            valinit=self.params[param])
                              for param in self.params}
        self.channels_slider = Slider(self.channels_ax,
                                      'channel',
                                      0,
                                      self.nchans-1,
                                      valinit=self.chan,
                                      valfmt='%0.0f')
        self.max_cbar_slider = VertSlider(self.max_cbar_ax,
                                          'max',
                                          0,
                                          self.max_vmax,
                                          valinit=self.vmax,valfmt='%0.3f')
        self.center_cbar_slider = VertSlider(self.center_cbar_ax,
                                             'center',
                                             0,
                                             self.max_vmax,
                                             valinit=self.vcenter,
                                             valfmt='%0.3f')
        self.min_cbar_slider = VertSlider(self.min_cbar_ax,
                                          'min',
                                          0,
                                          self.max_vmax,
                                          valinit=self.vmin,
                                          valfmt='%0.3f')
        self.fit_button = Button(self.fit_button_ax,
                                 'Fit!',
                                 color='g',
                                 hovercolor='0.975')
        self.next_chan_button = Button(self.next_chan_button_ax,
                                       'Next chan',
                                       color='r', hovercolor='0.975')
        self.prev_chan_button = Button(self.prev_chan_button_ax,
                                       'Prev. chan',
                                       color='r', hovercolor='0.975')

        for param in self.params:
            self.param_sliders[param].on_changed(self.sliders_on_changed)
        self.channels_slider.on_changed(self.channels_sliders_on_changed)
        self.max_cbar_slider.on_changed(self.max_cbar_slider_on_changed)
        self.center_cbar_slider.on_changed(self.center_cbar_slider_on_changed)
        self.min_cbar_slider.on_changed(self.min_cbar_slider_on_changed)
        self.fit_button.on_clicked(self.fit_button_on_clicked)
        self.next_chan_button.on_clicked(self.next_chan_button_on_clicked)
        self.prev_chan_button.on_clicked(self.prev_chan_button_on_clicked)

    def update_ellipse(self, first_time=False):
        self.measure_photo()
        self.plot_ellipse(first_time)

    def update_image(self, first_time=False):
        self.plot_image()
        self.update_ellipse(first_time)

    def update_params(self, x0, y0, sma, eps, pa):
        self.params = {'x0':x0, 'y0':y0, 'sma':sma, 'eps':eps, 'pa':pa}
        self.geom = EllipseGeometry(**self.params)
        self.update_ellipse()

    def update_channel(self, channel=None, update_chan_button=True):
        chan2update = self.chan+1 if (self.chan+1)<self.nchans else self.chan
        self.chan = channel if channel is not None else chan2update
        self.box = [[int(self.ax.get_xlim()[0]), int(self.ax.get_ylim()[0])],
                    [int(self.ax.get_xlim()[1]), int(self.ax.get_ylim()[1])]]
        self.update_image(first_time=True)

        if update_chan_button:
            self.channels_ax.remove()
            self.channels_ax = self.fig.add_axes([0.25,
                                                  0.25-len(self.slider_ax)*0.03,
                                                  0.65,
                                                  0.03])
            self.channels_slider = Slider(self.channels_ax,
                                          'channel',
                                          0,
                                          self.nchans-1,
                                          valinit=self.chan,
                                          valfmt='%0.0f')
            self.channels_slider.on_changed(self.channels_sliders_on_changed)

    def measure_photo(self):
        """
        sample.extract(), from EllipseSample documentation:
        Extract sample data by scanning an elliptical path over the
        image array.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The rows of the array contain the angles, radii, and
            extracted intensity values, respectively.
        """
        if self.photomorph == 'ellipse':
            sample = EllipseSample(self.im,
                                   sma=self.params['sma'],
                                   astep=0,
                                   sclip=0.3,
                                   nclip=0,
                                   linear_growth=False,
                                   geometry=self.geom,
                                   integrmode='bilinear')
            self.ellip_data = sample.extract()
            self.mean_intens = np.mean(self.ellip_data[2])
        elif self.photomorph == 'annulus':
            aper = EllipticalAnnulus((self.params['x0'], self.params['y0']),
                                     self.params['sma']-self.beam_size_pix/2.,
                                     self.params['sma']+self.beam_size_pix/2.,
                                     (self.params['sma']+self.beam_size_pix/2.) \
                                     *(1. - self.params['eps']),
                                     self.params['pa'])
            annulus_mask = aper.to_mask()
            self.ellip_data = annulus_mask.multiply(self.im)
            self.sum_intens = np.sum(self.ellip_data) / self.beam_size_pix**2 #in Jy
            self.mean_intens = np.mean(self.ellip_data)
#            also: aperture_photometry(self.image_cube[self.chan] / self.beam_size_pix**2, aper)
        elif self.photomorph == 'circle':
            aper = CircularAperture((self.params['x0'], self.params['y0']),
                                    self.params['sma'])
            circle_mask = aper.to_mask()
            self.ellip_data = circle_mask.multiply(self.im)
            self.sum_intens = np.sum(self.ellip_data) / self.beam_size_pix**2
            self.mean_intens = np.mean(self.ellip_data)
        elif self.photomorph == 'ellipse_area':
            aper = EllipticalAperture((self.params['x0'], self.params['y0']),
                                      self.params['sma'],
                                      self.params['sma']*(1. - self.params['eps']),
                                      self.params['pa'])
            ellipse_mask = aper.to_mask()
            self.ellip_data = ellipse_mask.multiply(self.im)
            self.sum_intens = np.sum(self.ellip_data) / self.beam_size_pix**2
            self.mean_intens = np.mean(self.ellip_data)

    def plot_image(self,):
        if self.header is not None:
            self.ax.remove()
            self.ax = plt.subplot(projection=self.wcs)
        else:
#            self.fig.clear()
            self.ax.remove()
            self.ax = self.fig.add_subplot()
            self.fig.canvas.draw_idle()
        self.im = show_slice_cube(self.image_cube,
                                  channel=int(self.chan),
                                  box=self.box,
                                  wcs=self.wcs,
                                  header=self.header,
                                  ax=self.ax,
                                  operation=self.operation,
                                  width_chan=self.width_chan,
                                  vmax=self.vmax,
                                  vcenter=self.vcenter,
                                  vmin=self.vmin,
                                  norm=self.norm,
                                  cmap=self.cmap,
                                  v0=self.v0,
                                  vf=self.vf,
                                  add_scalebar=True if self.header is not None else False,
                                  add_beam=True if self.header is not None else False,
                                  scalebar_loc='lower right',
                                  scalebar_distance=200,
                                  scalebar_units='au',
                                  return_im=True,
                                  output_name=None,
                                  interpolation=self.interpolation,
                                  filterrad=self.filterrad,)

    def plot_ellipse(self, first_time):
        self.ax_text.clear()
        if not first_time:
            self.ax.patches[1].remove()
        else:
            pass
        if self.photomorph == 'ellipse':
            aper = EllipticalAperture((self.params['x0'], self.params['y0']),
                                      self.params['sma'],
                                      self.params['sma']*(1. - self.params['eps']),
                                      self.params['pa'])
            self.ax_text.text(0.,
                              0.5,
                              r"Mean intens path: {:.3} Jy/beam".format(self.mean_intens),
                              fontsize=10,
                              color='k',
                              verticalalignment='top',
                              transform=self.ax_text.transAxes,)
        elif self.photomorph == 'annulus':
            aper = EllipticalAnnulus((self.params['x0'], self.params['y0']),
                                     self.params['sma']-self.beam_size_pix/2.,
                                     self.params['sma']+self.beam_size_pix/2.,
                                     (self.params['sma']+self.beam_size_pix/2.) \
                                     *(1. - self.params['eps']),
                                     self.params['pa'])
            self.ax_text.text(0.,
                              0.5,
                              r"Flux density annulus: {:.3} Jy".format(self.sum_intens),
                              fontsize=10,
                              color='k',
                              verticalalignment='top',
                              transform=self.ax_text.transAxes,)
            self.ax_text.text(0.,
                              1,
                              r"Mean intens path: {:.3} Jy/beam".format(self.mean_intens),
                              fontsize=10,
                              color='k',
                              verticalalignment='top',
                              transform=self.ax_text.transAxes,)
        elif self.photomorph == 'circle':
            aper = CircularAperture((self.params['x0'], self.params['y0']),
                                    self.params['sma'])
            self.ax_text.text(0.,
                              0.5,
                              r"Flux density circle: {:.3} Jy".format(self.sum_intens),
                              fontsize=10,
                              color='k',
                              verticalalignment='top',
                              transform=self.ax_text.transAxes,)
            self.ax_text.text(0.,
                              1,
                              r"Mean intens: {:.3} Jy/beam".format(self.mean_intens),
                              fontsize=10,
                              color='k',
                              verticalalignment='top',
                              transform=self.ax_text.transAxes,)
        elif self.photomorph == 'ellipse_area':
            aper = EllipticalAperture((self.params['x0'],
                                       self.params['y0']),
                                      self.params['sma'],
                                      self.params['sma']*(1. - self.params['eps']),
                                      self.params['pa'])
            self.ax_text.text(0.,
                              0.5,
                              r"Flux density circle: {:.3} Jy".format(self.sum_intens),
                              fontsize=10,
                              color='k',
                              verticalalignment='top',
                              transform=self.ax_text.transAxes,)
            self.ax_text.text(0.,
                              1,
                              r"Mean intens: {:.3} Jy/beam".format(self.mean_intens),
                              fontsize=10,
                              color='k',
                              verticalalignment='top',
                              transform=self.ax_text.transAxes,)

        aper.plot(self.ax, color=self.color_ref_ellip)

        self.ax_text.set_axis_off()
        self.color_ref_ellip = 'r'

    def sliders_on_changed(self, val):
        self.update_params(*[self.param_sliders[param].val
                             for param in self.param_sliders])
        self.fig.canvas.draw_idle()

    def channels_sliders_on_changed(self, val):
        self.update_channel(channel=int(val), update_chan_button=False)
        self.fig.canvas.draw_idle()

    def max_cbar_slider_on_changed(self, val):
        self.vmax = val
        self.update_image(first_time=True)

    def center_cbar_slider_on_changed(self, val):
        self.vcenter = val
        self.update_image(first_time=True)

    def min_cbar_slider_on_changed(self, val):
        self.vmin = val
        self.update_image(first_time=True)

    def fit_button_on_clicked(self,mouse_event):
        self.fit_ellip()

    def next_chan_button_on_clicked(self,mouse_event):
        self.update_channel()

    def prev_chan_button_on_clicked(self,mouse_event):
        self.update_channel(channel=self.chan-1)

    def maximize_intens(self, par):
        if self.photomorph == 'ellipse':
            geom = EllipseGeometry(par[0], par[1], par[2], par[3], par[4],)
            sample = EllipseSample(self.im,
                                   sma=geom.sma,
                                   astep=0,
                                   sclip=0.3,
                                   nclip=0,
                                   linear_growth=False,
                                   geometry=geom,
                                   integrmode='bilinear')
            data2min = np.mean(sample.extract()[2])
        elif self.photomorph == 'annulus':
            aper = EllipticalAnnulus((par[0], par[1]),
                                     par[2]-self.beam_size_pix/2.,
                                     par[2]+self.beam_size_pix/2.,
                                     (par[2]+self.beam_size_pix/2.)*(1. - par[3]),
                                     par[4])
            annulus_mask = aper.to_mask()
            data2min = annulus_mask.multiply(self.im)
        elif self.photomorph == 'circle':
            aper = CircularAperture((par[0], par[1]), par[2])
            circular_mask = aper.to_mask()
            data2min = circular_mask.multiply(self.im)
        elif self.photomorph == 'ellipse_area':
            aper = EllipticalAperture((par[0], par[1]), par[2], par[2]*(1. - par[3]), par[4])
            ellipse_mask = aper.to_mask()
            data2min = ellipse_mask.multiply(self.im)

        if self.q2min == 'sum':
            tominimize = np.sum(data2min)
        elif self.q2min == 'mean':
            tominimize = np.mean(data2min)

        return -tominimize

    def fit_ellip(self):
        """
        The solver method should be Nelder-Mead, Powell, CG, BFGS, Newton-CG, L-BFGS-B, TNC,
        COBYLA, SLSQP, trust-constr, dogleg, trust-ncg, trust-exact, trust-krylov.
        If not given, chosen to be one of BFGS, L-BFGS-B, SLSQP, depending if the problem has constraints or bounds.
        """
        g = self.geom
        bounds = ((g.x0-self.limit_fit['-x0'], g.x0+self.limit_fit['+x0']),
                  (g.y0-self.limit_fit['-y0'], g.y0+self.limit_fit['+y0']),
                  (g.sma-self.limit_fit['-sma'], g.sma+self.limit_fit['+sma']),
                  (g.eps-self.limit_fit['-eps'],g.eps+self.limit_fit['+eps']),
                  (g.pa-self.limit_fit['-pa'], g.pa+self.limit_fit['+pa']))
        self.fit_results[str(self.chan)] = minimize(self.maximize_intens,
                                                    [g.x0, g.y0, g.sma, g.eps, g.pa],
                                                    bounds=bounds,
                                                    tol=self.fit_tol,
                                                    method=self.fit_method)
        self.update_params(*self.fit_results[str(self.chan)].x)
        if self.fit_results[str(self.chan)].success:
            self.color_ref_ellip = 'w'
            self.update_params(*self.fit_results[str(self.chan)].x)
            self.update_buttons()
        else:
            self.color_ref_ellip = 'k'
            self.update_params(*self.fit_results[str(self.chan)].x)
        print(self.fit_results[str(self.chan)])
        if self.pickling:
            self.pickle()

    def save_plot(self, output_name):
        self.im = show_slice_cube(self.image_cube,
                                  channel=int(self.chan),
                                  box=self.box,
                                  wcs=self.wcs,
                                  header=self.header,
                                  ax=self.ax,
                                  operation=self.operation,
                                  width_chan=self.width_chan,
                                  vmax=self.vmax,
                                  vcenter=self.vcenter,
                                  vmin=self.vmin,
                                  norm=self.norm,
                                  cmap=self.cmap,
                                  v0=self.v0,
                                  vf=self.vf,
                                  add_scalebar=True,
                                  add_beam=True,
                                  scalebar_loc='lower right',
                                  scalebar_distance=200,
                                  scalebar_units='au',
                                  return_im=True,
                                  interpolation=self.interpolation,
                                  filterrad=self.filterrad,
                                  output_name=output_name,
                                  path_save=mf.default_params['path_save_bb_characterization'])

    def pickle(self, db_name=None):
        db_name = db_name if db_name is not None else self.db_name
        self.path_database = '{}{}.db'.format(mf.default_params['path_database'],db_name)
        database = shelve.open(self.path_database)
        database[str(self.chan)] = {'fit_results':self.fit_results[str(self.chan)],
                                    'geom':self.geom,
                                    'ellip_data':self.ellip_data}
        print('\n Results from channel {} pickled in {}.db!\n'.format(self.chan, db_name))
        database.close()

    def delete_chan_from_db(self, db_name=None):
        db_name = db_name if db_name is not None else self.db_name
        self.path_database = '{}{}.db'.format(mf.default_params['path_database'],db_name)
        database = shelve.open(self.path_database)
        try:
            del database[str(self.chan)]
            print('\n Results from channel {} DELETED in {}.db\n'.format(self.chan, db_name))
        except:
            print('\n Error when trying to delete channel {} results in {}.db\n'.format(self.chan, db_name))
        database.close()

    def get_from_db(self, channel=None, db_name=None):
        channel = channel if channel is not None else self.chan
        db_name = db_name if db_name is not None else self.db_name
        self.path_database = '{}{}.db'.format(mf.default_params['path_database'],db_name)
        database = shelve.open(self.path_database)
        try:
            self.update_params(*database[str(channel)]['fit_results'].x)
        except:
            print('\n Error while trying to import db data')
        database.close()
        self.update_buttons()


class EllipseFitterFits(EllipseFitter):
    def __init__(self, fits_path,
                 db_name='database',
                 x0=None,
                 y0=None,
                 sma=None,
                 eps=None,
                 pa=None,
                 box='init',
                 **kwargs):

        self.fits_path = fits_path
        self.hdu = None
        self.header = None
        self.wcs = None
        self.image_data = None
        self.args2fitter = [x0, y0, sma, eps, pa, box]

        self.read_fits()

        super().__init__(image_cube=self.image_data, header=self.header, **kwargs)

        plt.show()

    def read_fits(self,):
        self.hdu = fits.open(self.fits_path)[0]
        self.header = self.hdu.header
        self.wcs = WCS(self.header).celestial
        self.image_data = self.hdu.data[0]
