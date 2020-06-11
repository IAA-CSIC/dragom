import numpy as np

from astropy import units as u


class Wind(object):
    """
    This is the model proposed by Shu et al. (1991), which fits the nearest to the source parts of molecular outflows (Lee et al, 2000)
    beta > 1     -> Parabolic (e.g. C=100, beta=2)
    0 < beta < 1 -> Trumpet (e.g. C)
    beta < 0     -> Inverse Trumpet (e.g. C=-2, C=0.01)
    """
    model_type='wind'

    def __init__(self, C=100, beta=2, H_0=35, v_0=0, v_mode='exp',v_scale=100,**kwargs):
        self.C = C
        self.beta = beta
        self.v_mode = v_mode
        self.H_0 = H_0
        self.v_0 = v_0
        self.v_scale = v_scale

    z_f = lambda self, r: self.C * r**self.beta
    r = lambda self, z: (z/self.C)**(1/self.beta)
    v_z = lambda self, z: self.H_0 * z + self.v_0

    def v_angle(self,z):
        if self.v_mode=='tan':
            alpha = np.arctan(self.beta*self.C*(z/self.C)**(1-1/self.beta))
        elif self.v_mode=='exp':
            alpha = np.arctan(z/self.r(z))
        return alpha

    v = lambda self, z: self.v_z(z) / np.sin(self.v_angle(z))
    v_r = lambda self, z: self.v(z) * np.cos(self.v_angle(z))

    vel2cart = lambda self, z, theta: {'x':self.v_r(z)*np.sin(theta), 'y':self.v_r(z)*np.cos(theta), 'z':self.v_z(z)}
    pos2cart = lambda self, z, theta: {'x':self.r(z)*np.sin(theta), 'y':self.r(z)*np.cos(theta), 'z':z}

    m_arrow = lambda self, z: np.tan(self.v_angle(z))
    r_arrow = lambda self, z: np.array([self.r(z), (self.v(z)/self.v_scale)*np.cos(self.v_angle(z))+self.r(z)])
    z_arrow = lambda self, z: np.array([z, self.m_arrow(z)*self.r_arrow(z)[1]-(z if self.v_mode is 'tan' else 0)])
    arrow = lambda self, z: {'r':self.r_arrow(z), 'z':self.z_arrow(z)}

    arrow2cart = lambda self, z, theta: {'x':self.r_arrow(z) * np.sin(theta),
                                         'y':self.r_arrow(z) * np.cos(theta),
                                         'z':self.z_arrow(z)}



class Sphere(object):
    """
     Usefull to use as reference in the plots, to ensure that "bubbles" are pure circles and not ellipses.
    """
    model_type = 'Sphere'

    def __init__(self, R=0.2, center=[0., -1.5, 1], v=30, **kwargs):
        self.R = R
        self.center = center
        self.v = {'x':v, 'y':v, 'z':v}
    pos2cart = lambda self, theta, phi: {'x':self.center[0]+self.R*np.sin(theta)*np.cos(phi),
                                         'y':self.center[1]+self.R*np.sin(theta)*np.sin(phi),
                                         'z':self.center[2]+self.R*np.cos(theta)}

class WindFolium(object):

    model_type = 'Folium'
    def __init__(self, a=1., n=50., H_0=35, v_0=0, v_mode='exp', v_scale=100, **kwargs):
        self.a = a
        self.n = n
        self.v_mode = v_mode
        self.H_0 = H_0
        self.v_0 = v_0
        self.v_scale = v_scale

    r = lambda self, z: ((self.a * z**self.n)**(2/(self.n+1)) - z**2)**(1/2)
    v_z = lambda self, z: self.H_0 * z + self.v_0

    def v_angle(self,z):
        if self.v_mode=='tan':
            print('No available tan mode for this model')
            pass
#            alpha = np.arctan(self.beta*self.C*(z/self.C)**(1-1/self.beta))
        elif self.v_mode=='exp':
            alpha = np.arctan(z/self.r(z))
        return alpha

    v = lambda self, z: self.v_z(z) / np.sin(self.v_angle(z))
    v_r = lambda self, z: self.v(z) * np.cos(self.v_angle(z))

    vel2cart = lambda self, z, theta: {'x':self.v_r(z)*np.sin(theta), 'y':self.v_r(z)*np.cos(theta), 'z':self.v_z(z)}
    pos2cart = lambda self, z, theta: {'x':self.r(z)*np.sin(theta), 'y':self.r(z)*np.cos(theta), 'z':z}

    m_arrow = lambda self, z: np.tan(self.v_angle(z))
    r_arrow = lambda self, z: np.array([self.r(z), (self.v(z)/self.v_scale)*np.cos(self.v_angle(z))+self.r(z)])
    z_arrow = lambda self, z: np.array([z, self.m_arrow(z)*self.r_arrow(z)[1]-(z if self.v_mode is 'tan' else 0)])
    arrow = lambda self, z: {'r':self.r_arrow(z), 'z':self.z_arrow(z)}

    arrow2cart = lambda self, z, theta: {'x':self.r_arrow(z) * np.sin(theta), 'y':self.r_arrow(z) * np.cos(theta),
                                        'z':self.z_arrow(z)}


#class ClosedWind(object):

#    model_type = 'ClosedWind'
#
#    def __init__(self, R):
#        self.R = R
#        self.center = center
#        self.v = {'x':v, 'y':v, 'z':v}
#
#    z_f = lambda r, theta: r * np.cos(theta)
#
#    pos2cart = lambda self, theta, phi: {'x':self.center[0]+self.R*np.sin(theta)*np.cos(phi),
#                                         'y':self.center[1]+self.R*np.sin(theta)*np.sin(phi),
#                                         'z':self.center[2]+self.R*np.cos(theta)}


class GeneralAxisymmetricModel(object):
    distance = 235 #pc
    toauyears = (u.km/u.s).to(u.au/u.yr)

    model_type = 'General'
    def __init__(self, v_theta, params, t_dyn, v_scale=0.01, d_0=0, v_0=0, **kwargs):
        self.v_theta = v_theta
        self.params = params
        self.t_dyn = t_dyn
        self.v_scale = v_scale
        self.d_0 = d_0
        self.v_0 = v_0

    v = lambda self, theta: self.v_theta(theta, **self.params) + self.v_0
    d = lambda self, theta: self.v(theta) * self.toauyears * self.t_dyn / self.distance + self.d_0 #in arcsec
    r = lambda self, theta: self.d(theta) * np.sin(theta)
    z = lambda self, theta: self.d(theta) * np.cos(theta)

    v_r = lambda self, theta: self.v(theta) * np.sin(theta)
    v_z = lambda self, theta: self.v(theta) * np.cos(theta)

    vel2cart = lambda self, theta, phi: {'x':self.v_r(theta)*np.cos(phi),
                                         'y':self.v_r(theta)*np.sin(phi),
                                         'z':self.v_z(theta)}
    pos2cart = lambda self, theta, phi: {'x':self.r(theta)*np.cos(phi),
                                         'y':self.r(theta)*np.sin(phi),
                                         'z':self.z(theta)}

    m_arrow = lambda self, theta: 1./np.tan(theta)
    r_arrow = lambda self, theta: np.array([self.r(theta),
                                            (self.v(theta)*self.v_scale)*np.sin(theta)+self.r(theta)])
    z_arrow = lambda self, theta: np.array([self.z(theta), self.m_arrow(theta)*self.r_arrow(theta)[1]])
    arrow = lambda self, theta: {'r':self.r_arrow(theta), 'z':self.z_arrow(theta)}

    arrow2cart = lambda self, theta, phi: {'x':self.r_arrow(theta) * np.sin(phi),
                                      'y':self.r_arrow(theta) * np.cos(phi),
                                      'z':self.z_arrow(theta)}

class JetDrivenModel(object):
    """
    Jet driven model from Ostriker et al. (2001)
    """
    distance = 235 #pc
    toauyears = (u.km/u.s).to(u.au/u.yr)

    model_type = 'jet-driven'
    def __init__(self, v_theta, params, t_dyn, v_scale=0.01, d_0=0, v_0=0, **kwargs):
        self.params = params
        self.t_dyn = t_dyn
        self.v_scale = v_scale
        self.d_0 = d_0
        self.v_0 = v_0

#    r = lambda self, theta: self.param['Rj'] * (self.vel_ratio * np.tan(theta) + 1)**0.5
#    z = lambda self, theta: - (1/3*(self.r(theta)/self.params['Rj'])**3 - self.r(theta)/self.params['Rj'] + 2/3) * self.params['Rj'] / self.vel_ratio

#    v = lambda self, theta: self.v_theta(theta, **self.params) + self.v_0
#    d = lambda self, theta: self.v(theta) * self.toauyears * self.t_dyn / self.distance + self.d_0 #in arcsec

    vel_ratio = lambda self,: self.params['beta'] * self.params['cs'] / self.params['vs']
    v_r = lambda self, theta: self.vel_ratio()*self.params['vs'] / (self.vel_ratio()*np.tan(theta) + 1)
    v_z = lambda self, alpha: self.params['vs'] / (self.vel_ratio() * np.tan(theta) + 1)
    r = lambda self, theta: self.v_r(theta) * self.toauyears * self.t_dyn / self.distance
    z = lambda self, theta: self.v_z(theta) * self.toauyears * self.t_dyn / self.distance

    vel2cart = lambda self, theta, phi: {'x':self.v_r(theta)*np.cos(phi),
                                         'y':self.v_r(theta)*np.sin(phi),
                                         'z':self.v_z(theta)}
    pos2cart = lambda self, theta, phi: {'x':self.r(theta)*np.cos(phi),
                                         'y':self.r(theta)*np.sin(phi),
                                         'z':self.z(theta)}

    m_arrow = lambda self, theta: 1./np.tan(theta)
    r_arrow = lambda self, theta: np.array([self.r(theta),
                                            (self.v(theta)*self.v_scale)*np.sin(theta)+self.r(theta)])
    z_arrow = lambda self, theta: np.array([self.z(theta), self.m_arrow(theta)*self.r_arrow(theta)[1]])
    arrow = lambda self, theta: {'r':self.r_arrow(theta), 'z':self.z_arrow(theta)}

    arrow2cart = lambda self, theta, phi: {'x':self.r_arrow(theta) * np.sin(phi),
                                      'y':self.r_arrow(theta) * np.cos(phi),
                                      'z':self.z_arrow(theta)}




class JetDrivenDownes:
    distance = 235 #pc
    toauyears = (u.km/u.s).to(u.au/u.yr)

    model_type = 'jet-driven'
    def __init__(self, params, t_dyn, v_scale=0.01, d_0=0, v_0=0, **kwargs):
        self.params = params #v_j, s, comp_rat
        self.t_dyn = t_dyn
        self.v_scale = v_scale
        self.d_0 = d_0
        self.v_0 = v_0
#        for param in self.params:
#            setattr(self, str(param), params[param])
#        self.vel_ratio = self.params['beta'] * self.params['cs'] / self.params['vs']

    apex = lambda self, : self.t_dyn * np.abs(self.params['v_j']) * self.toauyears/self.distance + self.d_0
    z = lambda self, z: z
#    v_rat = lambda self, z: self.s * (self.apex-z)**((self.s-1)/self.s)
    v_rat = lambda self, z: -self.params['s'] * (self.apex() - z)**((self.params['s']-1)/self.params['s'])
    v_1 = lambda self, z: self.params['v_j'] * \
                        ((self.params['comp_rat']**2+self.v_rat(z)**2)/(1+self.v_rat(z)**2))**0.5
    v_r = lambda self, z: self.v_1(z) * np.cos(np.arctan(self.v_rat(z)))
    v_z = lambda self, z: self.v_1(z) * np.sin(np.arctan(self.v_rat(z))) + self.params['v_j']
    v = lambda self, z: np.sqrt(self.v_r(z)**2 + self.v_z(z)**2)

    r = lambda self, z: (self.apex() - z)**(1/self.params['s'])

    vel2cart = lambda self, z, phi: {'x':self.v_r(z) * np.cos(phi),
                                     'y':self.v_r(z) * np.sin(phi),
                                     'z':self.v_z(z)}
    pos2cart = lambda self, z, phi: {'x':self.r(z) * np.cos(phi),
                                     'y':self.r(z) * np.sin(phi),
                                     'z':z}

    m_arrow = lambda self, z: self.v_z(z) / self.v_r(z)
    theta_v = lambda self, z: np.arctan(self.m_arrow(z))
#    theta_v = lambda self, z: np.arctan(1/self.m_arrow(z))
    z_arrow = lambda self, z: np.array([z,
                             (self.v(z)*self.v_scale)*np.sin(self.theta_v(z)) + z])
    r_arrow = lambda self, z: np.array([self.r(z),
                            (self.v(z)*self.v_scale)*np.cos(self.theta_v(z)) + self.r(z)])
    arrow = lambda self, z: {'r':self.r_arrow(z), 'z':self.z_arrow(z)}

    arrow2cart = lambda self, z, phi: {'x':self.r_arrow(z) * np.sin(phi),
                                       'y':self.r_arrow(z) * np.cos(phi),
                                       'z':self.z_arrow(z)}


class JetDrivenShellOstriker:
    distance = 235 #pc
    toauyears = (u.km/u.s).to(u.au/u.yr)

    model_type = 'jet-driven'
    def __init__(self, params, t_dyn, v_scale=0.01, d_0=0, v_0=0, **kwargs):
        self.params = params #vs, r_j, beta, cs
        self.t_dyn = t_dyn
        self.v_scale = v_scale
        self.d_0 = d_0
        self.v_0 = v_0
#        for param in self.params:
#            setattr(self, str(param), params[param])
#        self.vel_ratio = self.params['beta'] * self.params['cs'] / self.params['vs']
    z_0 = lambda self,: self.t_dyn * np.abs(self.params['vs']) * self.toauyears/self.distance + self.d_0
    vel_ratio = lambda self,: self.params['beta'] * self.params['cs'] / self.params['vs']
    r = lambda self, r: r
    r_ratio = lambda self, r: r / self.params['r_j']
    z = lambda self, r: -(1/3*(self.r_ratio(r))**3-self.r_ratio(r)+2/3) * 1/self.vel_ratio() * self.params['r_j'] + self.z_0()
    v_z = lambda self, r: (self.r_ratio(r))**(-2) * self.params['vs']
    v_r = lambda self, r: (self.r_ratio(r))**(-2) * self.vel_ratio() * self.params['vs']
    v = lambda self, r: np.sqrt(self.v_r(r)**2 + self.v_z(r)**2)

    vel2cart = lambda self, r, phi: {'x':self.v_r(r) * np.cos(phi),
                                     'y':self.v_r(r) * np.sin(phi),
                                     'z':self.v_z(r)}
    pos2cart = lambda self, r, phi: {'x':self.r(r) * np.cos(phi),
                                     'y':self.r(r) * np.sin(phi),
                                     'z':self.z(r)}

    m_arrow = lambda self, r: self.v_z(r) / self.v_r(r)
    theta_v = lambda self, r: np.arctan(self.m_arrow(r))
#    theta_v = lambda self, z: np.arctan(1/self.m_arrow(z))
    z_arrow = lambda self, r: np.array([self.z(r),
                             (self.v(r)*self.v_scale)*np.sin(self.theta_v(r)) + self.z(r)])
    r_arrow = lambda self, r: np.array([r,
                            (self.v(r)*self.v_scale)*np.cos(self.theta_v(r)) + r])
    arrow = lambda self, r: {'r':self.r_arrow(r), 'z':self.z_arrow(r)}

    arrow2cart = lambda self, r, phi: {'x':self.r_arrow(r) * np.sin(phi),
                                       'y':self.r_arrow(r) * np.cos(phi),
                                       'z':self.z_arrow(r)}



class ObsBased(object):

    def __init__(self, inter_obs, a=1., n=50., H_0=35, v_0=0, v_mode='exp', v_scale=100, **kwargs):
        self.v_mode = v_mode
        self.H_0 = H_0
        self.v_0 = v_0
        self.v_scale = v_scale
        self.r = inter_obs

    v_z = lambda self, z: self.H_0 * z + self.v_0

    def v_angle(self,z):
        if self.v_mode=='tan':
            print('No available tan mode for this model')
            pass
#            alpha = np.arctan(self.beta*self.C*(z/self.C)**(1-1/self.beta))
        elif self.v_mode=='exp':
            alpha = np.arctan(z/self.r(z))
        return alpha

    v = lambda self, z: self.v_z(z) / np.sin(self.v_angle(z))
    v_r = lambda self, z: self.v(z) * np.cos(self.v_angle(z))

    vel2cart = lambda self, z, theta: {'x':self.v_r(z)*np.sin(theta),
                                       'y':self.v_r(z)*np.cos(theta),
                                       'z':self.v_z(z)}
    pos2cart = lambda self, z, theta: {'x':self.r(z)*np.sin(theta),
                                       'y':self.r(z)*np.cos(theta),
                                       'z':z}

    m_arrow = lambda self, z: np.tan(self.v_angle(z))
    r_arrow = lambda self, z: np.array([self.r(z), (self.v(z)/self.v_scale)*np.cos(self.v_angle(z))+self.r(z)])
    z_arrow = lambda self, z: np.array([z, self.m_arrow(z)*self.r_arrow(z)[1]-(z if self.v_mode is 'tan' else 0)])
    arrow = lambda self, z: {'r':self.r_arrow(z), 'z':self.z_arrow(z)}

    arrow2cart = lambda self, z, theta: {'x':self.r_arrow(z) * np.sin(theta),
                                         'y':self.r_arrow(z) * np.cos(theta),
                                        'z':self.z_arrow(z)}



class ObsBasedAlpha:
    def __init__(self, r, alpha, v, params, v_scale=0.01, **kwargs):
        self.params = params
        self.v_scale = v_scale
        self.r = r
        self.alpha = alpha
        self.v = v
        self.t_dyn = 0 # needed to build the model, but ignored for this model.
        self.v_0 = 0 # needed to build the model, but ignored for this model.
        self.d_0 = 0 # needed to build the model, but ignored for this model.

    z = lambda self, z: z
    v_z = lambda self, z: self.v(z) * np.cos(self.alpha(z))
    v_r = lambda self, z: self.v(z) * np.sin(self.alpha(z))

    vel2cart = lambda self, z, phi: {'x':self.v_r(z) * np.cos(phi),
                                     'y':self.v_r(z) * np.sin(phi),
                                     'z':self.v_z(z)}
    pos2cart = lambda self, z, phi: {'x':self.r(z) * np.cos(phi),
                                     'y':self.r(z) * np.sin(phi),
                                     'z':z}

    m_arrow = lambda self, z: np.tan(self.v_angle(z))

    z_arrow = lambda self, z: np.array([z,
                                        (self.v(z)*self.v_scale)*np.cos(self.alpha(z)) + z])
    r_arrow = lambda self, z: np.array([self.r(z),
                                        (self.v(z)*self.v_scale)*np.sin(self.alpha(z)) + self.r(z)])
    arrow = lambda self, z: {'r':self.r_arrow(z), 'z':self.z_arrow(z)}

    arrow2cart = lambda self, z, phi: {'x':self.r_arrow(z) * np.sin(phi),
                                         'y':self.r_arrow(z) * np.cos(phi),
                                         'z':self.z_arrow(z)}

#class AlphaObsbasedModel():


