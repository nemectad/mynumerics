import numpy as np
import mynumerics.units as units
import mynumerics.FSPA as FSPA

# Some HHG characteristics

def ComputeCutoff(Intensity, omega, Ip):
  # It computes the cutoff from the formula (3.17Up+Ip) Energy [a.u.] and harmonic order
  # inputs are intensity [a.u.], omega [a.u.] and Ip [a.u.]
  # It returns [Energy, order]

  Up = Intensity/(4.0*omega**2)
  Energy = 3.17*Up + Ip
  return Energy, Energy/omega

def ComputeInvCutoff(order,omega,Ip):
  # order [-]
  # omega [a.u.]
  # Ip [a.u.]
  # returns intensity in a.u.
  return (4.0*omega**2) * (omega*order - Ip)/3.17