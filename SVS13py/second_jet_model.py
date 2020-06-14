import numpy as np

from astropy import units as u

#import matplotlib
#matplotlib.use('Qt4Agg')

from SVS13py.main_functions import default_kwargs, rot, read_file, equal_aspect, debug
from SVS13py.models import Wind, Sphere, WindFolium, GeneralAxisymmetricModel, ObsBased
from SVS13py.bubblepy_general import BuildModel, BuildCube, BuildModelInteractive

import matplotlib.pyplot as plt

#matplotlib.use('Qt5Agg')
box = [[556,786],[855,1164]]

P = lambda theta, A, Palpha, B, Pbeta: A*np.cos(theta)**Palpha + B*np.sin(theta)**Pbeta
Q = lambda theta, C, Qalpha, D, Qbeta: C*np.cos(theta)**Qalpha + D*np.sin(theta)**Qbeta


v_theta = lambda theta, A, Palpha, B, Pbeta, C, Qalpha, D, Qbeta,K: K*(P(theta,A,Palpha,B,Pbeta) / Q(theta,C,Qalpha,D,Qbeta))**0.5
params = {'A':3.31,
          'Palpha':2.04,
          'B':41.07,
          'Pbeta':2,
          'C':0.001,
          'Qalpha':2.02,
          'D':0.66,
          'Qbeta':2.65,
          'K':0.90}

gam = GeneralAxisymmetricModel(v_theta, params, t_dyn=86.7, d_0=0.83,)

p = BuildModelInteractive(gam, box=box,
                          gaussian_filter=2,
                          theta_0=0,
                          theta_f=0.09,
                          v_range=[-93.3177-0.5291,88.7103],
                          change_header=False,
                         )

plt.show()
