import numpy as np
from mpmath import mp
import matplotlib.pyplot as plt

h = 6.626*10**(-34)        # Planck's constant
c = 2.998*10**(8)          # speed of light
kB = 1.381*10**(-23)       # Boltzmann's constant
nu0 = 3.288*10**(15)       # minimum frequency required to ionize hydrogen
sigma = 5.6704*10**(-8)    # Stefan-Boltzmann constant
R_odot = 6.957*10**(8)     # radius of Sun
lightyear = 9.461*10**(15) # light-year, in meters
alpha = 2.6*10**(-19)      # recombination coefficient
n = 10**(7)                # typical nebular number density
min_mass = 0.1             # lower range of IMF
max_mass = 250            # upper range of IMF

def I(nu,T):
    """The Planck function for a blackbody at temperature T"""
    I = (2*h*nu**(3)*(c**(-2)))/(mp.exp((h*nu)/(kB*T)) - 1)
    return I

def Q_star(T,R_star):
    """The number of ionizing photons emitted by a star per unit time"""
    Q_star = 4*(np.pi*R_star)**(2)*mp.quad(lambda nu: I(nu,T)/(h*nu), [nu0,mp.inf])
    return Q_star

def Rs(Q):
    """The Strömgren radius of a nebula surrounding a cluster emitting Q photons per second"""
    Rs = ((3/(4*np.pi))*Q/(alpha*(n**2)))**(1/3)
    return Rs

def IMF(m,N):
    """The Salpeter initial mass function of a cluster of N stars"""
    xi0 = N/mp.quad(lambda x: x**(-2.35), [min_mass, max_mass]) # normalizes un-nofrmalized IMF
    phi = xi0*m**(-2.35)
    return phi

def diameter(N):
    """The diameter of an HII region, in light-years"""
    # We assume that the effective temperature and stellar radius are proportional to
    # M^(4/7) and M^(3/7), respectively
    Qtot = mp.quad(lambda m: Q_star(5770*(m**(4/7)),R_odot*(m**(3/7)))*IMF(m,N), [min_mass, max_mass])
    radius = Rs(Qtot)/lightyear
    return 2*radius

