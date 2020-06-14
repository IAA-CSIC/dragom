import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import functools

from itertools import product

from scipy import stats
from scipy.ndimage.interpolation import rotate

from mayavi import mlab

from ipywidgets import interact, interactive, fixed, interact_manual

from astropy.visualization import simple_norm

from SVS13py.main_functions import default_kwargs, rot, read_file, equal_aspect, debug

class BuildModel(object):
    """
    Creates the data points for the model you propose. Its methods allows 2D and 3D representation, rotation...
    """
    default_kwargs = default_kwargs
    bb2_strong_path = '/home/gblazquez/radio_astronomy/SVS13/PV_bb2_strong.dat'
    bb2_weak_path = '/home/gblazquez/radio_astronomy/SVS13/PV_bb2_weak.dat'

    def __init__(self, model, **kwargs):
        self.m = model
        self.bb2_strong = read_file(self.bb2_strong_path)
        self.bb2_weak = read_file(self.bb2_weak_path)
        self.H_0, self.v_0, self.r_value, self.p_value, self.std_err \
                = stats.linregress(self.bb2_strong['arcsec'],
                                   self.bb2_strong['vel'])
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
            else self.default_kwargs[kwarg]
            setattr(self, kwarg, kwargs[kwarg])

#        self.zs2D = [self.m.z_f(r) for r in self.rs2D] if self.m.z_f is not None else
        self.rs2D = [self.m.r(z) for z in self.zs2D]

        self.positions = [self.m.pos2cart(z,theta)
                          for z,theta in product(self.zs, self.thetas)]
        self.velocities = [self.m.vel2cart(z,theta)
                           for z,theta in product(self.zs, self.thetas)]
        self.arrows = [self.m.arrow2cart(z,theta)
                       for z, theta in product(kwargs['zs_arrows'],
                                               kwargs['thetas_arrows'])]
        self.rg_positions = None
        self.rg_velocities = None

    def refGeom(self, rg_model, **kwargs):
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
            else self.__dict__[kwarg]

        if rg_model is not None:
            self.rg_positions = [rg_model.pos2cart(theta,phi)
                                 for theta,phi in product(kwargs['thetas'],
                                                          kwargs['phis'])]
            self.rg_velocities = [rg_model.v]*len(self.rg_positions)
        else:
            self.rg_positions = None
            self.rg_velocities = None

    def rotate(self, axis, angle_degree):
        i = angle_degree * np.pi / 180
        self.positions = [rot(position,axis,i) for position in self.positions]
        self.velocities = [rot(velocity,axis,i)
                           for velocity in self.velocities]
        self.arrows = [rot(arrow,axis,i) for arrow in self.arrows]

    def updatePoints(self,):
        self.positions = [self.m.pos2cart(z,theta)
                          for z,theta in product(self.zs, self.thetas)]
        self.velocities = [self.m.vel2cart(z,theta)
                           for z,theta in product(self.zs, self.thetas)]

    def plot2D(self, **kwargs):
        """
        Plots the model in 2D, with arrows indicating the direction of the velocity. If not all arrows appear
        is due to the axis limits
        """
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs else self.__dict__[kwarg]

        fig, ax = plt.subplots(figsize=(kwargs['figsize'],kwargs['figsize']))
        plt.plot(self.rs2D, self.zs2D, c='b')
        plt.plot(-np.array(self.rs2D), self.zs2D, c='b')
        for z in self.zs2D[::kwargs['step_arrow']]:
            ax.annotate("", xy=(self.m.arrow(z)['r'][1], self.m.arrow(z)['z'][1]),
                        xytext=(self.m.arrow(z)['r'][0], self.m.arrow(z)['z'][0]),
                        arrowprops=dict(arrowstyle=kwargs['arrowstyle']))
        ax.set_xlabel(kwargs['x_label'])
        ax.set_ylabel(kwargs['z_label'])
        #ax.set_aspect('equal')

    @equal_aspect
    def scatter3D(self, axis_vel=None, **kwargs):
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs else self.__dict__[kwarg]

        X = [p['x'] for p in self.positions]
        Y = [p['y'] for p in self.positions]
        Z = [p['z'] for p in self.positions]
        if axis_vel is not None:
            V = [v[axis_vel] for v in self.velocities]
            cbar_label = 'Velocity {} (km/s)'.format(axis_vel)
        else:
            V = [np.sqrt(v['x']**2 + v['y']**2 + v['z']**2) for v in self.velocities]
            cbar_label = 'Velocity (km/s)'

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y, Z = np.array(X), np.array(Y), np.array(Z)

        scat = ax.scatter(X, Y, Z, c=V, s=kwargs['scatter_size'], alpha=kwargs['scatter_alpha'], cmap=kwargs['cmap'])
        if kwargs['show_arrows']:
            for arrow in self.arrows:
                ax.quiver(arrow['x'][0], arrow['y'][0], arrow['z'][0], arrow['x'][1], arrow['y'][1],
                      arrow['z'][1], color=kwargs['color_arrow'], length=kwargs['length_arrow'])

        cbar = plt.colorbar(mappable=scat, shrink=kwargs['colorbar_shrink'], aspect=kwargs['colorbar_aspect'],)
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
            ax.scatter(X_rg,
                       Y_rg,
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
            V = [np.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
                 for v in self.velocities]
            cbar_label = 'Velocity (Km/s)'
        mlab.points3d(X, Y, Z, V)

    def scatter2D(self, xaxis='x', yaxis='z', vaxis='z', **kwargs):
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
                else self.__dict__[kwarg]

        coords = {cart: [p[cart] for p in self.positions]
                  for cart in ['x', 'y', 'z']}
        if vaxis is not None:
            V = [v[vaxis] for v in self.velocities]
            cbar_label = 'Velocity {} (km/s)'.format(vaxis)
        else:
            V = [np.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
                 for v in self.velocities]
            cbar_label = 'Velocity (km/s)'

        fig, ax = plt.subplots(figsize=(kwargs['figsize'],
                                        kwargs['figsize']))
        order = np.argsort(V)
        scat = ax.scatter(coords[xaxis][order],
                          coords[yaxis][order],
                          c=V[order],
                          s=kwargs['scatter_size'],
                          alpha=kwargs['scatter_alpha'],
                          cmap=kwargs['cmap'])

        cbar = plt.colorbar(mappable=scat,
                            shrink=kwargs['colorbar_shrink'],
                            aspect=kwargs['colorbar_aspect'])
        cbar.ax.set_ylabel(cbar_label)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        ax.set_aspect('equal')


class BuildCube(BuildModel):
    """
    Builds a mock cube from a built model.
    """
    def __init__(self, bmodel, v_per_bin=5, arcsec_per_bin=0.03,
                 pos_range=[[-1.5, 0.5], [-2, 0.5]], v_axis='z'):
        BuildModel.__init__(self, bmodel.m)
        for attrib in list(bmodel.__dict__):
            self.__dict__[attrib] = bmodel.__dict__[attrib]

        self.v_per_bin = v_per_bin
        self.arcsec_per_bin = arcsec_per_bin
        self.pos_range = pos_range
        self.v_range = None
        self.v_axis = v_axis

        self.cube = None
        self.edges = None

        self.intcube = None #internal cube for calculations
        self.intcube = None

        self.pv_diagram = None

        self.makeCube()

    def makeCube(self, pos_range=None, intcube=False, rg=False, **kwargs):
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
            else self.__dict__[kwarg]

        pos_range = pos_range if pos_range is not None else self.pos_range
#        self.rg_position = self.rg_position if rg else None
#        self.rg_velocities = self.rg_velocities if rg else None

        poss = (self.positions+self.rg_positions) \
                if self.rg_positions is not None else self.positions
        vels = (self.velocities+self.rg_velocities) \
                if self.rg_velocities is not None else self.velocities

        cube_points = {cart: np.array([pos[cart] for pos in poss])
                       for cart in ['x','y','z']}
        if self.v_axis is not None:
            cube_points['v'] = np.array([vel[self.v_axis] for vel in vels])
        else:
            cube_points['v'] = np.array([np.sqrt(vel['x']**2+vel['y']**2+vel['z']**2)
                                         for vel in vels])

        self.v_range = [cube_points['v'].min(), cube_points['v'].max()]
        cube_range = [self.v_range] + pos_range

        v_n_bin = abs(self.v_range[1]-self.v_range[0]) / self.v_per_bin
        x_n_bin = abs(pos_range[0][1]-pos_range[0][0]) / self.arcsec_per_bin
        y_n_bin = abs(pos_range[1][1]-pos_range[1][0]) / self.arcsec_per_bin

#        debug.cube_points = cube_points
#        debug.bins = (v_n_bin,x_n_bin,y_n_bin)
#        debug.cube_range = cube_range

        data, edges = np.histogramdd((cube_points['v'],
                                      cube_points['x'],
                                      cube_points['y']),
                                     bins=(v_n_bin,x_n_bin,y_n_bin),
                                     range=cube_range,
                                     density=kwargs['density'],
                                     weights=kwargs['weights'])
        if intcube:
            self.intcube, self.intedges = data, edges
        else:
            self.cube, self.edges = data, edges

    def sliceCube(self, v_slice, nmin=None, nmax=None, **kwargs):
        """
        Plots a slice of a cube given a velocity
        """
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
            else self.__dict__[kwarg]

        fig, ax = plt.subplots(figsize=(kwargs['figsize'],kwargs['figsize']))
        props = dict(boxstyle=kwargs['boxstyle'],
                     facecolor=kwargs['box_facecolor'],)

        plt.xlabel(r'$\Delta$ RA (arcsec)')
        plt.ylabel(r'$\Delta$ Dec (arcsec)')

        extent = [p for coord in self.pos_range for p in coord]
        nmin = nmin if nmin is not None else np.min(self.cube)
        nmax = nmax if nmax is not None else np.max(self.cube)

        channel = np.array([abs(abs(v_slice)-abs(v_e))
                            for v_e in self.edges[0]]).argmin()
        vel_str = '-{:.2f} km/s'.format(self.edges[0][channel])
#        norm = simple_norm(self.cube[channel,:,:], kwargs['norm'])
#        channel = np.where([np.max(np.abs(v_e)) for v_e in self.edges[0] if np.abs(v_e)<np.abs(v_slice)])
        im = plt.imshow(self.cube[channel,:,:].T,
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
        norm = mpl.colors.Normalize(vmin=nmin, vmax=nmax)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm,
                                                  cmap=kwargs['cmap']),
                            ax=ax,
                            shrink=kwargs['colorbar_shrink'],
                            aspect=kwargs['colorbar_aspect'])
#                            fraction=kwargs['fraction'], pad=kwargs['pad'])
        n_cbar_ticks = len([str(t.get_position()[1])
                            for t in cbar.ax.get_yticklabels()])
        cbar_ticks = np.linspace(nmin, nmax, n_cbar_ticks)
        max_tick = '>'+str(nmax) if np.max(self.cube)>nmax \
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

    def interactiveCube(self, vmax=None, vmin=None, nmin=None,
                        nmax=None, **kwargs):
        """
        Interactive plot of the cube for jupyter notebooks, using ipywidgets
        """
        for kwarg in self.default_kwargs:
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
            else self.__dict__[kwarg]

        fig, ax = plt.subplots(figsize=(kwargs['figsize'],kwargs['figsize']))
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
        norm = mpl.colors.Normalize(vmin=nmin, vmax=nmax)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm,
                                                  cmap=kwargs['cmap']),
                            ax=ax,
                            shrink=kwargs['colorbar_shrink'],
                            aspect=kwargs['colorbar_aspect'])
#                            fraction=kwargs['fraction'], pad=kwargs['pad'])
        n_cbar_ticks = len([str(t.get_position()[1])
                            for t in cbar.ax.get_yticklabels()])
        cbar_ticks = np.linspace(nmin, nmax, n_cbar_ticks)
        max_tick = '>'+str(nmax) if np.max(self.cube)>nmax \
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
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs \
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

        extent = [nr*self.arcsec_per_bin,
                  0,
                  self.v_range[0],
                  self.v_range[1]]

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


