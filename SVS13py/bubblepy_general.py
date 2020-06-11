import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import functools

from itertools import product, chain

from astropy.io import fits
from astropy.visualization import simple_norm
from astropy import units as u
from astropy import constants as const
from astropy.wcs import WCS

from scipy import stats
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import gaussian_filter

from ipywidgets import interact, interactive, fixed, interact_manual

from SVS13py.main_functions import default_kwargs, default_header, theta_powerspace, create_header, rot, read_file, equal_aspect, debug
from SVS13py.ellipse_fitter import EllipseFitter
import SVS13py.mf as mf
import SVS13py.SVS13py as SVS13py

def add_marks2cmap(V, cmap, mark_velocities, mark_velocities_color_array, mark_rel_width):
    """
    Creates a cmap with marked lines on it with a different color given
    mark_velocities_color_array; e.g. np.array([0/256, 0/256, 0/256, 1])
    """
    color_length = len(V)
    unmarked_cmap = cm.get_cmap(cmap, color_length)
    cmap2mark = unmarked_cmap(np.linspace(0, 1, color_length))
    V_nonans = [v for v in V if ~np.isnan(v)]
    vs_linear = np.linspace(np.min(V_nonans),np.max(V_nonans), color_length)
    closest_vels = [np.abs(vs_linear-mark_vel) for mark_vel in mark_velocities]
    closest_vels_indexes = [np.argmin(closest_vel)
                            for closest_vel in closest_vels]
    for ind in closest_vels_indexes:
        width_marker = int(mark_rel_width * color_length)
        cmap2mark[ind-width_marker:ind+width_marker, :] \
                = mark_velocities_color_array
    cmap = colors.ListedColormap(cmap2mark)
    return cmap


class BuildModel(object):
    """
    Creates the data points for the model you propose. Its methods allow 2D
    and 3D representation, rotation...
    """
    default_kwargs = default_kwargs
    bb2_strong_path = '/home/gblazquez/radio_astronomy/SVS13/PV_bb2_strong.dat'
    bb2_weak_path = '/home/gblazquez/radio_astronomy/SVS13/PV_bb2_weak.dat'

    def __init__(self, model, **kwargs):
        self.m = model
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
            else self.default_kwargs[kwarg]
            setattr(self, kwarg, kwargs[kwarg])

        self.zs2D = [self.m.z(theta) for theta in self.thetas]
        self.rs2D = [self.m.r(theta) for theta in self.thetas]

        self.positions = [self.m.pos2cart(theta, phi)
                          for theta, phi in product(self.thetas, self.phis)]
        self.velocities = [self.m.vel2cart(theta, phi)
                           for theta, phi in product(self.thetas, self.phis)]
        self.arrows = [self.m.arrow2cart(theta, phi)
                       for theta, phi in product(self.thetas_arrows, self.phis)]

        self.rg_positions = None
        self.rg_velocities = None

    def refGeom(self, rg_model, **kwargs):
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
            else self.__dict__[kwarg]

        if rg_model is not None:
            self.rg_positions = [rg_model.pos2cart(theta,phi)
                                 for theta, phi in product(self.thetas, self.phis)]
            self.rg_velocities = [rg_model.v]*len(self.rg_positions)
        else:
            self.rg_positions = None
            self.rg_velocities = None

    def rotate(self, axis, angle_degree):
        i = angle_degree * np.pi / 180
        self.positions = [rot(position,axis,i) for position in self.positions]
        self.velocities = [rot(velocity,axis,i) for velocity in self.velocities]
        self.arrows = [rot(arrow,axis,i) for arrow in self.arrows]

    def updatePoints(self,):
        self.positions = [self.m.pos2cart(theta,phi)
                          for theta, phi in product(self.thetas,self.phis)]
        self.velocities = [self.m.vel2cart(theta)
                           for theta, phi in product(self.thetas,self.phis)]

    def plot2D(self, ax=None, **kwargs):
        """
        Plots the model in 2D, with arrows indicating the direction of the
        velocity. If not all arrows appear is due to the axis limits
        """
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
                else self.__dict__[kwarg]

        if ax is None:
            fig, ax = plt.subplots(figsize=(kwargs['figsize'],
                                            kwargs['figsize']))
        else:
            pass

        ax.plot(self.rs2D, self.zs2D, c='b')
        ax.plot(-np.array(self.rs2D), self.zs2D, c='b')
        for theta in self.thetas_arrows:
            xt, xa = self.m.arrow(theta)['r']
            yt, ya = self.m.arrow(theta)['z']
            if True in np.isnan([xa, xt, ya, yt]):
                pass
            else:
                ax.annotate(
                    "",
                    xy=(xa, ya),
                    xytext=(xt, yt),
                    arrowprops=dict(arrowstyle=default_kwargs['arrowstyle']))
                ax.annotate(
                    "",
                    xy=(-xa, ya),
                    xytext=(-xt, yt),
                    arrowprops=dict(arrowstyle=default_kwargs['arrowstyle']))
        ax.set_xlabel(kwargs['x_label'])
        ax.set_ylabel(kwargs['z_label'])
        ax.set_aspect('equal')

    @equal_aspect
    def scatter3D(self, axis_vel=None, fig_and_ax=None, **kwargs):
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] \
                    if kwarg in kwargs else self.__dict__[kwarg]

        X = [p['x'] for p in self.positions]
        Y = [p['y'] for p in self.positions]
        Z = [p['z'] for p in self.positions]
        if axis_vel is not None:
            V = [v[axis_vel] for v in self.velocities]
            cbar_label = 'Velocity {} (km/s)'.format(axis_vel)
        else:
            V = [np.sqrt(v['x']**2 + v['y']**2 + v['z']**2) for v in self.velocities]
            cbar_label = 'Velocity (km/s)'

        if fig_and_ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        else:
            fig, ax = fig_and_ax

        X, Y, Z = np.array(X), np.array(Y), np.array(Z)
        scat = ax.scatter(X,
                          Y,
                          Z,
                          c=V,
                          s=kwargs['scatter_size'],
                          alpha=kwargs['scatter_alpha'],
                          cmap=kwargs['cmap'])

        if kwargs['show_arrows']:
            for arrow in self.arrows:
                ax.quiver(arrow['x'][0],
                          arrow['y'][0],
                          arrow['z'][0],
                          arrow['x'][1],
                          arrow['y'][1],
                          arrow['z'][1],
                          color=kwargs['color_arrow'],
                          length=kwargs['length_arrow'])

        cbar = fig.colorbar(mappable=scat,
                            shrink=kwargs['colorbar_shrink'],
                            aspect=kwargs['colorbar_aspect'],
                            ax=ax)
        cbar.ax.set_ylabel(cbar_label)
        ax.set_xlabel(kwargs['x_label'])
        ax.set_ylabel(kwargs['y_label'])
        ax.set_zlabel(kwargs['z_label'])

        if kwargs['refgeom'] is not None:
            self.refGeom(rg_model=kwargs['refgeom'])
            kwargs['plot_refgeom'] = True

        if kwargs['plot_refgeom']:
            X_rg = [p['x'] for p in self.rg_positions]
            Y_rg = [p['y'] for p in self.rg_positions]
            Z_rg = [p['z'] for p in self.rg_positions]
            if axis_vel is not None:
                V_rg = [v[axis_vel] for v in self.rg_velocities]
            else:
                V_rg = [np.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
                        for v in self.rg_velocities]
            ax.scatter(X_rg, Y_rg,
                       Z_rg,
                       c=V_rg,
                       s=kwargs['rg_scatter_size'],
                       alpha=kwargs['rg_scatter_alpha'])

        return X, Y, Z, ax

    def plotMayavi(self, axis_vel=None, **kwargs):
        X = [p['x'] for p in self.positions]
        Y = [p['y'] for p in self.positions]
        Z = [p['z'] for p in self.positions]
        if axis_vel is not None:
            V = [v[axis_vel] for v in self.velocities]
            cbar_label = 'Velocity {} (Km/s)'.format(axis_vel)
        else:
            pass
#            V = [np.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
#                 for v in self.velocities]
#            cbar_label = 'Velocity (Km/s)'
#        mlab.points3d(X, Y, Z, V)

    def scatter2D(self, xaxis='x', yaxis='y', vaxis='z', fig_and_ax=None,
                  **kwargs):
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] \
                if kwarg in kwargs else self.__dict__[kwarg]

        coords = {cart: [p[cart] for p in self.positions]
                  for cart in ['x', 'y', 'z']}
        if vaxis is not None:
            V = [v[vaxis] for v in self.velocities]
            cbar_label = "Velocity {}' (km/s)".format(vaxis)
        else:
            V = [np.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
                 for v in self.velocities]
            cbar_label = 'Velocity (km/s)'

        if fig_and_ax is None:
            fig, ax = plt.subplots(figsize=(kwargs['figsize'],
                                            kwargs['figsize']))
        else:
            fig, ax = fig_and_ax

        if kwargs['mark_velocities'] is not None:
            cmap = add_marks2cmap(
                V=V,
                cmap=kwargs['cmap'],
                mark_velocities=kwargs['mark_velocities'],
                mark_velocities_color_array=kwargs['mark_velocities_color_array'],
                mark_rel_width=kwargs['mark_rel_width'])
        else:
            cmap = kwargs['cmap']

        inv = -1 if kwargs['inv_zorder'] else 1
        scat = ax.scatter(coords[xaxis][::inv],
                          coords[yaxis][::inv],
                          c=V[::inv],
                          s=kwargs['scatter_size'],
                          alpha=kwargs['scatter_alpha'],
                          cmap=cmap)

        if kwargs['plot_source']:
            ax.plot(0, 0, '*k', markersize=kwargs['markerstar_size'])

        if kwargs['plot_zaxis']:
            end_index = np.argmax([self.m.z(theta) for theta in self.thetas])
            end_x, end_y = coords[xaxis][end_index], coords[yaxis][end_index]
            ax.plot([0, end_x], [0, end_y], '--')

        if kwargs['plot_cbar']:
            cbar = fig.colorbar(mappable=scat,
                                shrink=kwargs['colorbar_shrink'],
                                aspect=kwargs['colorbar_aspect'],
                                ax=ax)
            cbar.ax.set_ylabel(cbar_label)

        ax.set_xlabel(kwargs['{}_label'.format(xaxis)])
        ax.set_ylabel(kwargs['{}_label'.format(yaxis)])
        ax.set_aspect('equal')


class BuildCube(BuildModel):
    """
    Builds a mock cube from a built model.
    """
    def __init__(self, bmodel,
                 v_per_bin=0.5291,
#                arcsec_per_bin=0.0119999999999988,
                 arcsec_per_bin=0.012,
                 pos_range=[[-13.2598, 13.6562], [-9.72, 9.51]],
                 v_axis='z',
                 automake=True,
                 v_range=None,
                 blueshifted=True,
                 **kwargs):
        if automake:
            BuildModel.__init__(self, bmodel.m, **kwargs)
        else:
            BuildModel.__init__(self, bmodel, **kwargs)
        for attrib in list(bmodel.__dict__):
            self.__dict__[attrib] = bmodel.__dict__[attrib]

        self.v_per_bin = v_per_bin
        self.arcsec_per_bin = arcsec_per_bin
        self.pos_range = pos_range
        self.v_range = v_range
        self.v_axis = v_axis
        self.blueshifted = blueshifted
        self.syst_vel = kwargs['syst_vel'] if 'syst_vel' in kwargs \
                        else mf.default_params['SVS13_vsys']

        self.v_n_bin = None
        self.x_n_bin = None
        self.y_n_bin = None

        self.cube = None
        self.edges = None

        self.intcube = None #internal cube for calculations
        self.intcube = None

        self.pv_diagram = None

        self.header = None
        self.wcs = None

        if automake:
            self.makeCube()

    def remove_nans(self, positions=None, velocities=None):
        positions = positions if positions is not None else self.positions
        velocities = velocities if velocities is not None else self.velocities
        id_nans = lambda p, cart: list(np.where(np.isnan([p[i][cart]
                            for i in range(len(p))]) == True)[0])
        ids2flat = [id_nans(positions, cart) + id_nans(velocities, cart)
                    for cart in ['x', 'y', 'x']]
        ids_nans = {i for i in np.array(ids2flat).flat}
        positions_nonans = [pos for i, pos in enumerate(positions)
                            if i not in ids_nans]
        velocities_nonans = [vel for i, vel in enumerate(velocities)
                            if i not in ids_nans]
        return positions_nonans, velocities_nonans

    def makeCube(self, pos_range=None, intcube=False, rg=False, save_cube=None,
                 **kwargs):
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs else self.__dict__[kwarg]

        pos_range = pos_range if pos_range is not None else self.pos_range
#        self.rg_position = self.rg_position if rg else None
#        self.rg_velocities = self.rg_velocities if rg else None

        poss = (self.positions+self.rg_positions) \
                if self.rg_positions is not None else self.positions
        vels = (self.velocities+self.rg_velocities) \
                if self.rg_velocities is not None else self.velocities
        poss, vels = self.remove_nans(poss, vels)

        cube_points = {cart: np.array([pos[cart] for pos in poss])
                       for cart in ['x','y','z']}
        if self.v_axis is not None:
            cube_points['v'] = np.array([vel[self.v_axis]
                                         for vel in vels]) * (-1)**self.blueshifted \
                                         + self.syst_vel
        else:
            cube_points['v'] = np.array([np.sqrt(vel['x']**2+vel['y']**2+vel['z']**2)
                                         for vel in vels])


        self.v_range = [cube_points['v'].min(), cube_points['v'].max()] \
                        if self.v_range is None else self.v_range
        cube_range = [self.v_range] + pos_range

        self.v_n_bin = abs(self.v_range[1]-self.v_range[0]) / self.v_per_bin
        self.y_n_bin = abs(pos_range[0][1]-pos_range[0][0]) / self.arcsec_per_bin
        self.x_n_bin = abs(pos_range[1][1]-pos_range[1][0]) / self.arcsec_per_bin

        data, edges = np.histogramdd((np.array(cube_points['v'], dtype=np.float32),
                                     np.array(cube_points['y'], dtype=np.float32),
                                     np.array(cube_points['x'], dtype=np.float32)),
                                     bins=(self.v_n_bin,self.y_n_bin,self.x_n_bin),
                                     range=cube_range,
                                     density=kwargs['density'],
                                     weights=kwargs['weights'])

        if kwargs['gaussian_filter'] is not None:
            data = gaussian_filter(data, sigma=kwargs['gaussian_filter'],
                                   output=np.float32)
        else:
            pass

        if intcube:
            self.intcube, self.intedges = data, edges
        else:
            self.cube, self.edges = data[::-1], [k[::-1] for k in edges]

        if save_cube is not None:
            dict_of_changes = {
                               'NAXIS1': self.x_n_bin,
                               'NAXIS2': self.y_n_bin,
                               'NAXIS3': self.v_n_bin,
                               'CDELT1': self.arcsec_per_bin / 3600.,  # deg per bin
                               'CDELT2': self.arcsec_per_bin / 3600.,  # deg per bin
                               'CRPIX1': self.x_n_bin / 2.,
                               'CRPIX2': self.y_n_bin / 2.,
                               'CRVAL3': (kwargs['12CO_J3-2']*u.Hz * \
                                   (1. - self.v_range[1]*(u.km/u.s)/const.c)).to(u.Hz).value, #Hz
                               'CDELT3': (self.v_per_bin*(u.km/u.s) / const.c \
                                         * kwargs['12CO_J3-2']*u.Hz).to(u.Hz).value, #Hz
                               'CRPIX3': 1.0,
                               'RESTFRQ': kwargs['12CO_J3-2'] #Hz
                              }

            if kwargs['change_header']:
                self.header = create_header(dict_of_changes)
            else:
                self.header = create_header(None)
            self.wcs = WCS(self.header).celestial
            fits.writeto(kwargs['path_save_cube']+'{}.fits'.format(save_cube),
                         self.cube,
                         header=self.header,
                         overwrite=True)

    def sliceCube(self, v_slice, nmin=None, nmax=None, show_slice_cube=True,
                  **kwargs):
        """
        Plots a slice of a cube given a velocity
        """
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
                else self.__dict__[kwarg]

        if show_slice_cube is False:
            fig, ax = plt.subplots(figsize=(kwargs['figsize'],
                                            kwargs['figsize']))
            props = dict(boxstyle=kwargs['boxstyle'],
                         facecolor=kwargs['box_facecolor'],)

            plt.xlabel(r'$\Delta$ RA (arcsec)')
            plt.ylabel(r'$\Delta$ Dec (arcsec)')

        extent = [p for coord in self.pos_range for p in coord]
        nmin = nmin if nmin is not None else np.min(self.cube)
        nmax = nmax if nmax is not None else np.max(self.cube)

        channel = np.array([abs(v_slice-v_e)
                            for v_e in self.edges[0]]).argmin()
        vel_str = '-{:.2f} km/s'.format(self.edges[0][channel])
        if show_slice_cube:
            im, ax = SVS13py.show_slice_cube(self.cube,
                                             channel=channel,
                                             wcs=self.wcs,
                                             header=self.header,
                                             add_beam=True,
                                             vmax=nmax,
                                             vmin=nmin,
                                             add_scalebar=True,
                                             return_ax=True,
                                             )
        else:
            norm = simple_norm(self.cube[channel, :, :], kwargs['norm'])
            channel = np.where([np.max(np.abs(v_e)) for v_e in self.edges[0]
                                if np.abs(v_e) < np.abs(v_slice)])
            _ = plt.imshow(self.cube[channel, :, :],
                           origin=kwargs['cube_origin'],
                           aspect=kwargs['cube_aspect'],
                           extent=extent,
                           interpolation=kwargs['interpolation'],
                           filterrad=kwargs['filterrad'],
                           cmap=kwargs['cmap'],
                           vmin=nmin,
                           vmax=nmax)

            plt.text(kwargs['x_box'],
                     kwargs['y_box'],
                     vel_str,
                     fontsize=kwargs['box_fontsize'],
                     verticalalignment='top',
                     transform=ax.transAxes,
                     bbox=props)

            norm = colors.Normalize(vmin=nmin,
                                    vmax=nmax)
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm,
                                                      cmap=kwargs['cmap']),
                                ax=ax,
                                shrink=kwargs['colorbar_shrink'],
                                aspect=kwargs['colorbar_aspect'])
    #                            fraction=kwargs['fraction'], pad=kwargs['pad'])
            n_cbar_ticks = len([str(t.get_position()[1])
                                for t in cbar.ax.get_yticklabels()])
            cbar_ticks = np.linspace(nmin, nmax, n_cbar_ticks)
            max_tick = '>' + str(nmax) if np.max(self.cube) > nmax \
                else str(cbar_ticks[-1])
            cbar_ticks = list(cbar_ticks[:-1]) + [max_tick]
            cbar.ax.set_yticklabels(cbar_ticks)
            cbar.ax.set_ylabel('N particles')

        if kwargs['save_fig']:
            plt.savefig('./slices/{}_{}.{}'.format(self.m.model_type,
                                                   v_slice,
                                                   kwargs['save_format']),
                        format=kwargs['save_format'],
                        bbox_inches='tight')

        if kwargs['show_plot']:
            plt.show()

    def interactiveCube(self,
                        vmax=None,
                        vmin=None,
                        nmin=None,
                        nmax=None,
                        **kwargs):
        """
        Interactive plot of the cube for jupyter notebooks, using ipywidgets
        """
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs else self.__dict__[kwarg]

        fig, ax = plt.subplots(figsize=(kwargs['figsize'],
                                        kwargs['figsize']))
        props = dict(boxstyle=kwargs['boxstyle'],
                     facecolor=kwargs['box_facecolor'],)

        plt.xlabel(r'$\Delta$ RA (arcsec)')
        plt.ylabel(r'$\Delta$ Dec (arcsec)')

        extent = [p for coord in self.pos_range for p in coord]
        nmin = nmin if nmin is not None else np.min(self.cube)
        nmax = nmax if nmax is not None else np.max(self.cube)
        def imcube(channel):
            vel_str = '-{:.2f} km/s'.format(self.edges[0][channel])
#            norm = simple_norm(self.cube[channel,:,:], kwargs['norm'])
            plt.imshow(self.cube[channel,:,:].T,
                       origin=kwargs['cube_origin'],
                       aspect=kwargs['cube_aspect'],
                       extent=extent,
                       interpolation=kwargs['interpolation'],
                       filterrad=kwargs['filterrad'],
                       cmap=kwargs['cmap'],
                       vmin=nmin,
                       vmax=nmax)

            plt.text(kwargs['x_box'],
                     kwargs['y_box'],
                     vel_str,
                     fontsize=kwargs['box_fontsize'],
                     verticalalignment='top',
                     transform=ax.transAxes,
                     bbox=props)
            plt.show()

#        ax = plt.gca()
#        divider = make_axes_locatable(ax)
#        cax = divider.append_axes("right", size="5%", pad=0.05)
        norm = colors.Normalize(vmin=nmin,
                                vmax=nmax)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm,
                                                  cmap=kwargs['cmap']),
                            ax=ax,
                            shrink=kwargs['colorbar_shrink'],
                            aspect=kwargs['colorbar_aspect'])
#                            fraction=kwargs['fraction'],
#                           pad=kwargs['pad'])
        n_cbar_ticks = len([str(t.get_position()[1])
                            for t in cbar.ax.get_yticklabels()])
        cbar_ticks = np.linspace(nmin, nmax, n_cbar_ticks)
        max_tick = '>'+str(nmax) \
                if np.max(self.cube)>nmax \
                else str(cbar_ticks[-1])
        cbar_ticks = list(cbar_ticks[:-1]) + [max_tick]
        cbar.ax.set_yticklabels(cbar_ticks)
        cbar.ax.set_ylabel('N particles')
        vmax = vmax if vmax is not None else len(self.edges[0])-2
        vmin = vmin if vmin is not None else 0
        return interactive(imcube, channel=(vmin,vmax))

    def plotPV(self, segment_angle, width=3, **kwargs):
        """
        Plots the position-velocity diagram of the cube.
        """
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] \
                    if kwarg in kwargs \
                    else self.__dict__[kwarg]

        sim_pr_x = np.max([np.abs(self.pos_range[0][0]),
                           np.abs(self.pos_range[0][1])])
        sim_pr_y = np.max([np.abs(self.pos_range[1][0]),
                           np.abs(self.pos_range[1][1])])

        pos_range = [[-sim_pr_x, sim_pr_x],[-sim_pr_y, sim_pr_y]]

        self.makeCube(pos_range=pos_range, intcube=True)
        self.intcube = rotate(self.intcube,
                              angle=segment_angle,
                              axes=(1,2),
                              reshape=False)

        nx_0 = int(len(self.intcube[:,:,0].T)/2)
        nv = len(self.intcube)
        nxs = [nx for nx in np.arange(nx_0-width, nx_0+width+1)]

        int_segment = self.intcube[:,nxs[0],:]
        for nx in nxs[1:]:
            int_segment += self.intcube[:,nx,:]

        segment = int_segment / nv

        nr = int(len(segment.T) / 2)
        self.pv_diagram = segment[:,:nr]

        extent = [nr*self.arcsec_per_bin, 0, self.v_range[0], self.v_range[1]]

        fig, ax = plt.subplots(figsize=(kwargs['figsize'],kwargs['figsize']))
        plt.xlabel(r'Offset (arcsec)')
        plt.ylabel(r'Velocity (km/s)')
#        extent = [p for coord in self.pos_range for p in coord]
        im = plt.imshow(self.pv_diagram,
                        aspect='auto',
                        origin=kwargs['cube_origin'],
                        extent=extent,
                        interpolation=kwargs['interpolation'])

        ax.invert_xaxis()
        cbar = plt.colorbar(mappable=im,
                            shrink=kwargs['colorbar_shrink'],
                            aspect=kwargs['colorbar_aspect'])
        cbar.ax.set_ylabel('N particles')


class BuildModelInteractive(BuildCube):
    """
    Interactive Model Builder
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, automake=False, **kwargs)
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] \
                    if kwarg in kwargs \
                    else self.__dict__[kwarg]
        self.fig, self.ax = plt.subplots(figsize=(self.figsize, self.figsize))
        self.fig3D = None
        self.ax3D = None
        self.fig2D = None
        self.ax2D = None
#        self.fig.subplots_adjust(left=0.25, bottom=0.35)
        self.fig_sliders = plt.figure(figsize=(8,6))
#        self.fixed_limits = self.init_fixed_limits
        self.fixed_limits = None
        self.set_aspect_equal = False

        self.plot_param4slider = ['thetas_arrows',
                                  'length_arrow',
                                  'thetas',
                                  'theta_0',
                                  'theta_f',
                                  'power_th',
                                  'phis',
                                  'phi_0',
                                  'phi_f']

        self.gaussian_filter = kwargs['gaussian_filter']
        self.change_header = kwargs['change_header']
        self.indep_variable = kwargs['indep_variable']


        self.xaxis = None
        self.yaxis = None
        self.axis_vel_3D = None
        self.box = kwargs['box'] if 'box' in kwargs else None

        self.cube_name = None

        self.angle2rotate = {'x':0, 'y':0, 'z':0}

        self.paramt_sliders_ax = None
        self.rotate_textbox_ax = None
        self.rotate_button_ax = None
        self.scatter_3D_buttons_ax = None
        self.scatter_2D_buttons_ax = None
        self.buildcube_button_ax = None
        self.pv_button_ax = None
        self.create_axes()

        self.paramt_sliders = None
        self.rotate_textbox = None
        self.rotate_button = None
        self.scatter_3D_buttons = None
        self.scatter_2D_buttons = None
        self.buildcube_button = None
        self.pv_button = None
        self.update_buttons()

        self.plot2D(ax=self.ax)

    def create_axes(self):
        """
        CAUTION: axis must be created and removed in the same order!
        """
        all_sliders = [param for param in self.m.params] + ['t_dyn'] + ['d_0'] \
                + self.plot_param4slider
        self.paramt_sliders_ax = {param: self.fig_sliders.add_axes([0.25,0.92-i*0.035,0.65,0.03])
                                  for i,param in enumerate(all_sliders)}
        self.rotate_textbox_ax = {cart: self.fig_sliders.add_axes([0.25,0.2-i*0.04,0.05,0.03])
                                  for i,cart in enumerate(['x','y','z'])}
        self.rotate_button_ax = self.fig_sliders.add_axes([0.4,0.2,0.1,0.04])
        self.scatter_3D_buttons_ax = {proj: self.fig_sliders.add_axes([0.6,0.2-i*0.04,0.1,0.04])
                                      for i,proj in enumerate(['v_tot','v_z'])}
        self.scatter_2D_buttons_ax = {plane: self.fig_sliders.add_axes([0.8,0.2-i*0.04,0.1,0.04])
                                      for i,plane in enumerate(['xy','xz', 'yz'])}
        self.buildcube_button_ax = self.fig_sliders.add_axes([0.4,0.16,0.1,0.04])
        self.pv_button_ax = self.fig_sliders.add_axes([0.4,0.12,0.1,0.04])

    def remove_axes(self):
        for paramt in self.paramt_sliders_ax:
            self.paramt_sliders_ax[paramt].remove()
        for cart in self.rotate_textbox_ax:
            self.rotate_textbox_ax[cart].remove()
        self.rotate_button_ax.remove()
        for proj in self.scatter_3D_buttons_ax:
            self.scatter_3D_buttons_ax[proj].remove()
        for plane in self.scatter_2D_buttons_ax:
            self.scatter_2D_buttons_ax[plane].remove()
        self.buildcube_button_ax.remove()
        self.pv_button_ax.remove()

    def update_buttons(self,):
        """
        Updates the state of the sliders and buttons (for example, after a fit)
        """
        self.remove_axes()
        self.create_axes()

        self.paramt_sliders = {param: Slider(self.paramt_sliders_ax[param], param,
                                            self.m.params[param]-5*abs(self.m.params[param]),
                                            self.m.params[param]+5*abs(self.m.params[param]),
                                            valinit=self.m.params[param])
                               for param in self.m.params}

        self.paramt_sliders['t_dyn'] = Slider(self.paramt_sliders_ax['t_dyn'],
                                              't_dyn (yr)',
                                            0,
                                            self.m.t_dyn+10*self.m.t_dyn,
                                            valinit=self.m.t_dyn)
        self.paramt_sliders['d_0'] = Slider(self.paramt_sliders_ax['d_0'],
                                            'd_0 (arcsec)',
                                            0,
                                            self.m.d_0+10*self.m.d_0,
                                            valinit=self.m.d_0)

        self.paramt_sliders['length_arrow'] = Slider(self.paramt_sliders_ax['length_arrow'],
                                                     'length_arrow',
                                                     0,
                                                     self.length_arrow+10*self.length_arrow,
                                                     valinit=self.length_arrow,)

        self.paramt_sliders['thetas_arrows'] = Slider(self.paramt_sliders_ax['thetas_arrows'],
                                                      'thetas_arrows',
                                                     0,
                                                     len(self.thetas_arrows)*10,
                                                     valinit=len(self.thetas_arrows),
                                                      valfmt='%d',
                                                      valstep=1)

        self.paramt_sliders['thetas'] = Slider(self.paramt_sliders_ax['thetas'],
                                               'thetas',
                                               0,
                                               len(self.thetas)*10,
                                               valinit=len(self.thetas))

        self.paramt_sliders['power_th'] = Slider(self.paramt_sliders_ax['power_th'],
                                                 'power_th',
                                                 1,
                                                 10,
                                                 valinit=self.power_th)

        if self.indep_variable == 'theta':
            self.paramt_sliders['theta_0'] = Slider(self.paramt_sliders_ax['theta_0'],
                                                    'theta_0 (x pi)',
                                                    0,
                                                    1,
                                                    valinit=self.theta_0)

            self.paramt_sliders['theta_f'] = Slider(self.paramt_sliders_ax['theta_f'],
                                                    'theta_f (x pi)',
                                                    0,
                                                    1,
                                                    valinit=self.theta_f)
        elif self.indep_variable == 'z':
            self.paramt_sliders['theta_0'] = Slider(self.paramt_sliders_ax['theta_0'],
                                                    'theta_0 (x pi)',
                                                    self.theta_0,
                                                    self.theta_f,
                                                    valinit=self.theta_0)

            self.paramt_sliders['theta_f'] = Slider(self.paramt_sliders_ax['theta_f'],
                                                    'theta_f (x pi)',
                                                    self.theta_0,
                                                    self.theta_f,
                                                    valinit=self.theta_f)


        self.paramt_sliders['phis'] = Slider(self.paramt_sliders_ax['phis'],
                                             'phis',
                                             0,
                                             len(self.phis)*10,
                                             valinit=len(self.phis))

        self.paramt_sliders['phi_0'] = Slider(self.paramt_sliders_ax['phi_0'],
                                              'phi_0 (x pi)',
                                              0,
                                              2,
                                              valinit=self.phi_0)
        self.paramt_sliders['phi_f'] = Slider(self.paramt_sliders_ax['phi_f'],
                                              'phi_f (x pi)',
                                              0,
                                              2,
                                              valinit=self.phi_f)

        self.rotate_textbox = {cart: TextBox(self.rotate_textbox_ax[cart],
                                             'rotate '+cart,
                                             initial='0')
                               for cart in ['x','y','z']}
        self.rotate_button = Button(self.rotate_button_ax,
                                    'Rotate',
                                    color='#c0fa8b',
                                    hovercolor='0.975')
        self.scatter_3D_buttons = {proj: Button(self.scatter_3D_buttons_ax[proj],
                                                'Plot 3D '+proj,
                                                color='#c0fa8b', hovercolor='0.975')
                                   for proj in ['v_tot', 'v_z']}
        self.scatter_2D_buttons = {plane: Button(self.scatter_2D_buttons_ax[plane],
                                                'Plot '+plane,
                                                color='#c0fa8b',
                                                hovercolor='0.975')
                                   for plane in ['xy', 'xz', 'yz']}
        self.buildcube_button = Button(self.buildcube_button_ax,
                                       'BuildCube',
                                       color='#b1d1fc',
                                       hovercolor='0.975')
        self.pv_button = Button(self.pv_button_ax,
                                'PV',
                                color='#b1d1fc',
                                hovercolor='0.975')

        for paramt in self.paramt_sliders:
            self.paramt_sliders[paramt].on_changed(self.paramt_sliders_on_changed)
#        for cart in self.rotate_textbox:
#            self.rotate_textbox[cart].on_submit(lambda val: self.rotate_textbox_on_submit(val, cart))
        self.rotate_textbox['x'].on_text_change(lambda val:self.rotate_textbox_on_text_change(val, 'x'))
        self.rotate_textbox['y'].on_text_change(lambda val:self.rotate_textbox_on_text_change(val, 'y'))
        self.rotate_textbox['z'].on_text_change(lambda val:self.rotate_textbox_on_text_change(val, 'z'))
        self.rotate_button.on_clicked(self.rotate_button_on_clicked)
        self.scatter_3D_buttons['v_tot'].on_clicked(lambda me:self.scatter_3D_buttons_on_clicked(me, axis_vel=None))
        self.scatter_3D_buttons['v_z'].on_clicked(lambda me:self.scatter_3D_buttons_on_clicked(me, axis_vel='z'))
        self.scatter_2D_buttons['xy'].on_clicked(lambda me:self.scatter_2D_buttons_on_clicked(me, plane='xy'))
        self.scatter_2D_buttons['xz'].on_clicked(lambda me:self.scatter_2D_buttons_on_clicked(me, plane='xz'))
        self.scatter_2D_buttons['yz'].on_clicked(lambda me:self.scatter_2D_buttons_on_clicked(me, plane='yz'))
        self.buildcube_button.on_clicked(self.buildcube_button_on_clicked)
        self.pv_button.on_clicked(self.pv_button_on_clicked)

    def paramt_sliders_on_changed(self, val):
        self.update_model_params({paramt:self.paramt_sliders[paramt].val
                                  for paramt in self.paramt_sliders})
        self.fig.canvas.draw_idle()


    def rotate_textbox_on_text_change(self, val, cart):
        try:
            self.angle2rotate[cart] = float(val)
            print('{} axis rotation of {}'.format(cart, val))
        except:
            pass

    def rotate_button_on_clicked(self, mouse_event):
        for cart in self.angle2rotate:
            self.rotate(axis=cart, angle_degree=self.angle2rotate[cart])
        self.update_image()

    def scatter_3D_buttons_on_clicked(self, mouse_event, axis_vel):
        self.fig3D = plt.figure(figsize=(self.figsize, self.figsize))
        self.ax3D = self.fig3D.add_subplot(projection='3d')
        self.axis_vel_3D = axis_vel
        self.scatter3D(fig_and_ax=[self.fig3D, self.ax3D],
                       axis_vel=self.axis_vel_3D,
                       show_arrows=False)
        self.fig3D.canvas.draw_idle()

    def scatter_2D_buttons_on_clicked(self, mouse_event, plane):
        self.fig2D, self.ax2D = plt.subplots(figsize=(self.figsize,
                                                      self.figsize))
        self.xaxis2D, self.yaxis2D = plane
        self.scatter2D(xaxis=self.xaxis2D,
                       yaxis=self.yaxis2D,
                       fig_and_ax=[self.fig2D,
                                   self.ax2D])
        self.fig2D.canvas.draw_idle()

    def buildcube_button_on_clicked(self, mouse_event):
        self.cube_name = '{:.0f}yr'.format(self.m.t_dyn)
        self.makeCube(save_cube=self.cube_name,
                      gaussian_filter=self.gaussian_filter,
                      change_header=self.change_header)
        self.EF = EllipseFitter(self.cube,
                                header=self.header,
                                name_bb="model",
                                operation=None,
                                box=self.box,
                                interpolation='bilinear')

    def pv_button_on_clicked(self, mouse_event):
        self.plotPV(segment_angle=270, width=3., )

    def update_model_params(self, paramts,):
        self.m.params = {param: paramts[param]
                         for param in self.m.params}
        self.m.t_dyn = paramts['t_dyn']
        self.m.d_0 = paramts['d_0']
        self.m.v_scale = paramts['length_arrow']
        for paramt in self.plot_param4slider:
            if type(getattr(self,paramt))!=type(np.array([])):
                setattr(self, paramt, int(paramts[paramt]))
        if self.indep_variable == 'theta':
            self.thetas_arrows = np.linspace(paramts['theta_0']*np.pi,
                                             paramts['theta_f']*np.pi,
                                             int(paramts['thetas_arrows']))
            self.thetas = theta_powerspace(int(paramts['thetas']),
                                           power=self.power_th,
                                           theta_0=paramts['theta_0'],
                                           theta_f=paramts['theta_f'],)
        if self.indep_variable == 'z':
            pass

        self.phis = np.linspace(paramts['phi_0']*np.pi,
                                paramts['phi_f']*np.pi,
                                int(paramts['phis']))

        self.rebuild_model()

    def rebuild_model(self,):
        self.zs2D = [self.m.z(theta) for theta in self.thetas]
        self.rs2D = [self.m.r(theta) for theta in self.thetas]

        self.positions = [self.m.pos2cart(theta, phi)
                          for theta, phi in product(self.thetas, self.phis)]
        self.velocities = [self.m.vel2cart(theta, phi)
                           for theta, phi in product(self.thetas, self.phis)]
        self.arrows = [self.m.arrow2cart(theta, phi)
                       for theta, phi in product(self.thetas_arrows,
                                                 self.phis)]

        self.update_image()

    def update_image(self,):
        self.ax.clear()
        self.plot2D(ax=self.ax, x_label='x (arcsec)', z_label='z (arcsec)')

        if self.ax3D is not None:
            self.fig3D.clear()
            self.ax3D = self.fig3D.add_subplot(projection='3d')
            self.scatter3D(fig_and_ax=[self.fig3D,
                                       self.ax3D],
                           axis_vel=self.axis_vel_3D,
                           show_arrows=False)
            self.fig3D.canvas.draw_idle()

        if self.ax2D is not None:
            self.fig2D.clear()
            self.ax2D = self.fig2D.add_subplot()
            self.scatter2D(xaxis=self.xaxis2D,
                           yaxis=self.yaxis2D,
                           fig_and_ax=[self.fig2D,
                                       self.ax2D])
            self.fig2D.canvas.draw_idle()

        if self.fixed_limits is not None:
            self.ax.set_xlim([self.fixed_limits[0][0],
                              self.fixed_limits[0][1]])
            self.ax.set_ylim([self.fixed_limits[1][0],
                              self.fixed_limits[1][1]])

        if self.set_aspect_equal:
            self.ax.set_aspect('equal')
