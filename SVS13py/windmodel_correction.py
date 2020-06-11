import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from scipy import optimize

from matplotlib.gridspec import GridSpec

from itertools import product

import SVS13py.main_functions as main_functions
import SVS13py.mf as mf
from SVS13py.vertslider import VertSlider


R_p = lambda z, s, a: (a - z)**(1/s)
z1_0 = lambda z, z1, s, a, i: np.tan(i) * R_p(z1,s,a) + z1 - z
z2_0 = lambda z, z2, s, a, i: np.tan(i) * R_p(z2,s,a) - z2 + z

z1 = lambda z, s, a, i: optimize.brentq(lambda z1: z1_0(z,z1,s,a,i), 0, z)
z2 = lambda z, s, a, i: optimize.brentq(lambda z2: z2_0(z,z2,s,a,i), z, a)

R1 = lambda z, s, a, i: R_p(z1(z,s,a,i),s,a)
R2 = lambda z, s, a, i: R_p(z2(z,s,a,i),s,a)

dx_zaxis = lambda z, s, a, i: 1/(2*np.cos(i)) * np.abs((R2(z,s,a,i) - R1(z,s,a,i)))
dz_zaxis = lambda z, s, a, i: dx_zaxis(z,s,a,i) / np.cos(i)
dz_zaxis_2 = lambda z, s, a, i: 1/np.sin(i) * np.abs((z1(z,s,a,i)+z2(z,s,a,i))/2 - z)

def dz_zaxis_try(z, s, a, i):
    try:
        return dz_zaxis(z, s, a, i)
    except:
        return 0

def dx_zaxis_try(z, s, a, i):
    try:
        return dx_zaxis(z, s, a, i)
    except:
        return 0

def proj_correction(a_p, s_p, i_p, z_point, fig=None, axs=None,):
    #paraboloid
    zs_r = np.linspace(0, a_p, 1000)
    zs_p = np.array([z for z in zs_r] + [z for z in zs_r[::-1]])
    rs_p = np.array([R_p(z,s_p,a_p,) for z in zs_r] + \
                    [-R_p(z,s_p,a_p,) for z in zs_r[::-1]])

    #zaxis
    zaxis_x, zaxis_z = np.array([0]*len(zs_r)), zs_r

    #difference between zaxis and the center of the isovelocities
    zs_dx = np.linspace(1.5,a_p-0.01, 1000)
    dxs_zaxis = np.array([-dx_zaxis(z,s_p,a_p,i_p,) for z in zs_dx])

    #z1 and z2 points, defines the proyected isovelocities
    z1_point, z2_point = z1(z_point,s_p,a_p,i_p), z2(z_point,s_p,a_p,i_p)
    R1_point, R2_point = R1(z_point,s_p,a_p,i_p), R2(z_point,s_p,a_p,i_p)

    #we rotate everything
    p_cart = {'x':rs_p, 'y':0, 'z':zs_p}
    zaxis_cart = {'x':zaxis_x, 'y':0, 'z':zaxis_z}
    dxs_zaxis_cart = {'x':dxs_zaxis, 'y':0, 'z':zs_dx}
    points_cart = {'x':np.array([-R1_point, R2_point]),
                   'y':0,
                   'z':np.array([z1_point, z2_point])}
    z_point_cart = {'x':0,'y':0,'z':z_point}

    p_rot = main_functions.rot(p_cart, 'y', i_p)
    zaxis_rot = main_functions.rot(zaxis_cart, 'y', i_p)
    dxs_zaxis_rot = main_functions.rot(dxs_zaxis_cart, 'y', i_p)
    points_rot = main_functions.rot(points_cart, 'y', i_p)
    z_point_rot = main_functions.rot(z_point_cart, 'y', i_p)

    #lets simulate the deprojection
    x_obs = np.mean(points_rot['x'])
    D_iso = np.abs(points_rot['x'][0]-points_rot['x'][1])
    z_depr = x_obs / np.sin(i_p)
    # we do assume that the radii is the mean obs.
    r_edge = np.sqrt((D_iso/2.)**2 + z_depr**2)
    theta_angle = np.arctan((D_iso/2.) / z_depr)

    z_depr_cart = {'x':0, 'y':0, 'z':z_depr}
    z_depr_rot = main_functions.rot(z_depr_cart, 'y', i_p)

    if axs is None:
        nrow = 1
        ncol = 2
        ngrid = 2
        magical_factor = 15
        wspace = 0
        hspace = 0
        font_size = 15

        fig = plt.figure(figsize=(nrow*magical_factor,(ncol+1)*ngrid,))
        gs1 = GridSpec(nrow, (ncol+1)*ngrid, )
        gs1.update(wspace=wspace, hspace=hspace,)

        axs = {}
        n = 0
        for i,j in product(range(nrow), [i for i in range(ncol*ngrid)][::ngrid]):
            axs[n] = plt.subplot(gs1[i,j:j+ngrid])
            n += 1


    #non_rotated plots
    axs[1].plot(0,0,'k*')
    axs[1].plot(rs_p,zs_p, c='b')
    axs[1].plot(dxs_zaxis, zs_dx, c='r')
    axs[1].plot(zaxis_x, zaxis_z, 'k--')

    #rotated plots
    axs[0].plot(0,0,'k*')
    axs[0].plot(p_rot['x'], p_rot['z'], c='b')
    axs[0].plot(zaxis_rot['x'], zaxis_rot['z'], 'k--')
    axs[0].plot(dxs_zaxis_rot['x'], dxs_zaxis_rot['z'], c='r')
    axs[0].plot(points_rot['x'], points_rot['z'], c='k')
    axs[0].plot(np.mean(points_rot['x']), points_rot['z'][1], 'xk')
    axs[0].plot(z_point_rot['x'], z_point_rot['z'], 'xg')
    axs[0].plot(np.mean(points_rot['x']), z_depr_rot['z'], 'xm')

    #deprojected plots
    axs[1].plot(0, z_depr, 'xm')
    axs[1].plot(points_cart['x'], points_cart['z'], alpha=0.2, c='k')
    axs[1].plot([-D_iso/2,D_iso/2], [z_depr,z_depr], 'k')
    axs[1].plot(0,z_point,'gx')
    #axs[1].plot()

    for n in axs:
        axs[n].set_aspect('equal')

    axs[0].set_title('Projected')
    axs[0].set_xlabel("x' (arcsec)")
    axs[0].set_ylabel("line of sight (arcsec)")

    axs[1].set_title('Deprojected')
    axs[1].set_xlabel("x (arcsec)")
    axs[1].set_ylabel("z (arcsec)")



class ProjCorrect(object):
    init_params = {'a_p':8., 's_p':2.5, 'i_p':np.pi / 9, 'z_point':6.5}
    limit_slider = {'a_p_l':1, 'a_p_u':10.,
                    's_p_l':2.1, 's_p_u':5,
                    'i_p_l':0, 'i_p_u':np.pi,
                    'z_point_l':2.5, 'z_point_u':8}


    def __init__(self, a_p=None, s_p=None, i_p=None, z_point=None, **kwargs):
        """
        Kwargs can be: init_chan
        """
        self.params = {'a_p':a_p, 's_p':s_p, 'i_p':i_p, 'z_point':z_point}
        for param in self.params:
            self.params[param] = self.params[param] \
                    if self.params[param] is not None \
                    else self.init_params[param]

        self.create_fig()
        self.fig.subplots_adjust(left=0.25, bottom=0.35)

        self.create_axes()

        self.param_sliders = None
        self.update_buttons()
        self.proj_correction(axs=self.axs)


    def create_fig(self):
        nrow = 1
        ncol = 2
        ngrid = 2
        magical_factor = 15
        wspace = 0
        hspace = 0
        font_size = 15

        self.fig = plt.figure(figsize=(nrow*magical_factor,(ncol+1)*ngrid,))
        gs1 = GridSpec(nrow, (ncol+1)*ngrid, )
        gs1.update(wspace=wspace, hspace=hspace,)

        self.axs = {}
        n = 0
        for i,j in product(range(nrow), [i for i in range(ncol*ngrid)][::ngrid]):
            self.axs[n] = plt.subplot(gs1[i,j:j+ngrid])
            n += 1

    def create_axes(self):
        """
        CAUTION: axis must be created and removed in the same order!
        """
        self.slider_ax = {param: self.fig.add_axes([0.25,
                                                    0.25-i*0.03,
                                                    0.65,
                                                    0.03])
                          for i,param in enumerate(self.params)}


    def update_buttons(self,):
        """
        Updates the state of the sliders and buttons (for example, after a fit)
        """
#        plt.cla()
#        self.remove_axes()
#        for ax in self.axs:
#
#        self.create_fig()
#        self.create_axes()

        self.param_sliders = {param: Slider(self.slider_ax[param],
                                            param,
                                            self.limit_slider[param+'_l'],
                                            self.limit_slider[param+'_u'],
                                            valinit=self.params[param])
                              for param in self.params}
        for param in self.params:
            self.param_sliders[param].on_changed(self.sliders_on_changed)

    def sliders_on_changed(self, val):
        self.update_params(*[self.param_sliders[param].val
                             for param in self.param_sliders])
        self.fig.canvas.draw_idle()


    def update_params(self, a_p, s_p, i_p, z_point):
        self.params = {'a_p':a_p, 's_p':s_p, 'i_p':i_p, 'z_point':z_point}
        self.axs[0].clear()
        self.axs[1].clear()
        self.proj_correction(axs=self.axs)

    def proj_correction(self, axs=None):
        proj_correction(**self.params, axs=axs)
        self.fig.canvas.draw_idle()



