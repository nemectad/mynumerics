from scipy import special
from scipy import integrate
from scipy import interpolate
import numpy as np
import math
import sys
import units
import h5py
import warnings


## functions to work with indices
def n1n2mapping(k1,k2,N1): ## a mapping from (0,...,N1-1) x (0,...,N2-1) -> (0,...,N1*N2)
  return k1+k2*N1



def n1n2mapping_inv(n,N1): ## inverse to the previous one
  k1 = n % N1
  n = n - k1
  k2 = n // N1
  return k1,k2


def NumOfPointsInRange(N1,N2,k): #number of points between two integers following the Python's range-function logic (0,...,N-1), assumes N1<N2
  if N1 != 0:
    return NumOfPointsInRange(N1-N1,N2-N1,k);
  else:
    if (N2 % k) == 0:
      return N2 // k;
    else:
      return (N2 // k) + 1;


def FindInterval(x,x0): # find an index corresponding to given x0 value interval. ordering <  ), < ),..., < >; throws error otherwise
  if hasattr(x0, "__len__"):
    indices = []
    for x_loc in x0:
      indices.append(FindInterval(x,x_loc))
    return np.asarray(indices)

  else:    
    N = len(x)
    if ( (x0 > x[-1]) or (x0 < x[0]) ): raise LookupError('out of range in FindInterval')
    k1 = 1; k2 = N; length = N;
    while (length > 2):
      if ( x0 < x[k1 - 1 + length//2]): k2 = k1  + length//2
      else: k1 = k1 + length//2
      length = k2-k1

    if ((length == 2) and (x0 >= x[k1])): return k1
    else: return k1-1
    
  # for k1 in range(N-2):
  #   if ( (x[k1]<= x0) and (x0 < x[k1+1]) ): return k1
  # if ( (x[N-2]<= x0) and (x0 <= x[N-1]) ): return N-2
  # sys.exit('out of range in FindInterval')
  # if ( (x0 < x[0]) or (x0 > x[N-1])): sys.exit('out of range in FindInterval')
  # if (N == 2): return k1; # we finished looking
  # else:
  #   if (x0 > x[N//2] ): return FindInterval(x[(N//2):(N-1)],x0,N//2+?); # bookkeeping needed here... best will be additions and subtractions to be in-place
  #   else : return FindInterval(x[0:(N//2)],x0,?);

# a = []
# xxx = np.asarray([1.0, 2, 3, 4, 5, 6, 7, 8, 9])
# for k1 in range(len(xxx)-1): a.append(FindInterval(xxx,0.1+k1+1))

def find_index_across_arrays(vals,lists): ## inverse to the previous one
  if not(len(vals) == len(lists)): raise ValueError('mismatched lengths of values and lists')
  n_list = len(lists[0])
  n_vals = len(vals)
  candidates = []
  for k1 in range(n_list): # create candidates
    if (vals[0] == lists[0][k1]): candidates.append(k1)
  for k1 in range(1,n_vals): # remove candidates not matching conditions
    if not(len(lists[k1]) == n_list): raise ValueError('an array of different size from the first one present')
    old_candidates = candidates.copy()
    for k2 in range(len(old_candidates)):
        if not(vals[k1] == lists[k1][old_candidates[k2]]): del candidates[k2]
  
  if not(len(candidates) == 1): raise ValueError('there is no or multiple candidates')
  return candidates[0]

### low-level routines

def get_odd_interior_points(interval):
    res = np.asarray(
          list(range(int(np.floor(interval[0])),int(np.ceil(interval[1]))))
          )
    return res[(res%2==1)*(res>=interval[0])*(res<=interval[1])]

def get_divisible_interior_points(interval,divisor):
    res = np.asarray(
          list(range(int(np.floor(interval[0])),int(np.ceil(interval[1])+1)))
          )
    return res[(res%divisor==0)*(res>=interval[0])*(res<=interval[1])]

def IsPowerOf2(n):
  if ( (n & (n-1) == 0) and (n != 0) ): return True
  else: return False


def contains_substrings(string,substrings):
    for substring in substrings:
        if substring in string:
            return True
    return False

## CALCULUS
def ddx_arb(k,x,fx):
  h2 = x[k+1]-x[k]
  h1 = x[k]-x[k-1]
  ratio = h2/h1
  return (fx[k+1] - fx[k-1]*ratio**2 - (1.0-ratio**2)*fx[k])/(h2*(1.0+ratio))

def ddx_vec_arb(x,fx):
  dfx = np.zeros(len(fx),dtype=fx.dtype)
  dfx[0] = (fx[1]-fx[0])/(x[1]-x[0])
  dfx[-1] = (fx[-1]-fx[-2])/(x[-1]-x[-2])
  for k1 in range(1,len(fx)-1):
    dfx[k1] = ddx_arb(k1,x,fx)
  return dfx

def complexify_fft(fx):
  N = len(fx)
  fx = np.fft.fft(fx)
  for k1 in range((N // 2) + 1, N):
    fx[k1] = 0.0
  fx = 2.0 * np.fft.ifft(fx)
  return fx

def fft_t_nonorm(t, ft):
  Nt = len(t)
  Ft = np.conj(np.fft.fft(ft)[0:((Nt // 2) + 1)])
  omega = np.linspace(0, (np.pi * (Nt - 1) / (t[-1] - t[0])), (Nt // 2) + 1)
  return omega, Ft, Nt

def fft_t(t, ft):
  Nt = len(t)
  t0_ind = FindInterval(t,0.0)
  dt = t[t0_ind+1] - t[t0_ind]
  Ft = (dt/(np.sqrt(2.0*np.pi)))*np.conj(np.fft.fft(ft)[0:((Nt // 2) + 1)])
  omega = np.linspace(0, (np.pi * (Nt - 1) / (t[-1] - t[0])), (Nt // 2) + 1)
  return omega, Ft, Nt

def ifft_t_nonorm(omega, Ft, Nt):
  Ft = np.append(Ft, np.flip(np.conj(Ft[1:(Nt - len(Ft) + 1)])))
  ft = np.flip(np.fft.ifft(Ft))
  t = np.linspace(0, 2.0 * np.pi * (len(omega)) / (omega[-1] - omega[0]), len(ft))
  return t, ft

def integrate_subinterval(fx,x,xlim):
    if ( (xlim[0]<x[0]) or (xlim[-1]>x[-1]) ):
        raise ValueError('integration out of bounds')
    
    k_low = FindInterval(x, xlim[0]) + 1
    k_up = FindInterval(x, xlim[-1])
    
    x_low = x[k_low]; x_up = x[k_up] 
    fx_low = fx[k_low]; fx_up = fx[k_up] 
    
    delta_x = x[k_low] - x[k_low-1]
    fx_min = (fx[k_low-1] * (xlim[0] - x[k_low-1])/delta_x + fx[k_low] * (x[k_low]-xlim[0])/delta_x )
    delta_x = x[k_up] - x[k_up-1]
    fx_max = (fx[k_up] * (xlim[-1] - x[k_up])/delta_x + fx[k_up+1] * (x[k_up+1]-xlim[-1])/delta_x )
    
    # integrate in the intervale and add trpaezoidal parts of outer intervals
    Ifx = np.trapz(fx[k_low:(k_up+1)],x[k_low:(k_up+1)]) + \
          0.5 * (fx_low + fx_min) * (x_low-xlim[0]) + \
          0.5 * (fx_up + fx_max) * (xlim[-1]-x_up)    
    
    return Ifx

# fx = list(range(10))
# x = fx
# a = integrate_subinterval(fx,x,[1.5,5.5])

def romberg(x_length,fx,eps,n0):
  N = len(fx)
  if ( not IsPowerOf2(N-1) ): sys.exit("romberg: input isn't 2**k+1")
  elif ( not IsPowerOf2(n0) ): sys.exit("romberg: initial stepsize isn't 2**k")
  elif ( n0 > (N-1) ): sys.exit("romberg: initial number of points is larger than provided grid")
  dx = x_length/(N-1)
  step = (N-1)//n0 # adjust to n0 points, divisibility already checked
  k1 = 0
  I = [] # empty list of lists to store the integrals
  while (step >= 1):
    I.append([])
    indices = [k2 for k2 in range(0,N,step)]
    for k2 in range(k1+1):
      if (k2 == 0): value = integrate.trapz(fx[indices],dx=step*dx) # this is inefficient, we already keep the previous results, just refine them
      else: value = (4.0**k2 * I[k1][k2-1] - I[k1-1][k2-1]) / (4.0**k2-1.0)
      I[k1].append(value)

    if (k1>0):# convergence test
      Res = abs(I[k1][k1]-I[k1-1][k1-1])/abs(I[k1][k1])
      if (Res <= eps): return k1, I[k1][k1], Res

    step = step // 2
    k1 = k1+1

  return -1, I[-1][-1], Res # didn't converged in requested precision, returns the last value
  #  return [-1, I[-1][-1]] # didn't converged in requested precision, returns the last value


def romberg(x_length,fx,eps,n0):
  N = len(fx)
  if ( not IsPowerOf2(N-1) ): sys.exit("romberg: input isn't 2**k+1")
  elif ( not IsPowerOf2(n0) ): sys.exit("romberg: initial stepsize isn't 2**k")
  elif ( n0 > (N-1) ): sys.exit("romberg: initial number of points is larger than provided grid")
  dx = x_length/(N-1)
  step = (N-1)//n0 # adjust to n0 points, divisibility already checked
  k1 = 0
  I = [] # empty list of lists to store the integrals
  while (step >= 1):
    I.append([])
    indices = [k2 for k2 in range(0,N,step)]
    for k2 in range(k1+1):
      if (k2 == 0): value = integrate.trapz(fx[indices],dx=step*dx) # this is inefficient, we already keep the previous results, just refine them
      else: value = (4.0**k2 * I[k1][k2-1] - I[k1-1][k2-1]) / (4.0**k2-1.0)
      I[k1].append(value)

    if (k1>0):# convergence test
      Res = abs(I[k1][k1]-I[k1-1][k1-1])/abs(I[k1][k1])
      if (Res <= eps): return k1, I[k1][k1], Res

    step = step // 2
    k1 = k1+1

  return -1, I[-1][-1], Res # didn't converged in requested precision, returns the last value
  #  return [-1, I[-1][-1]] # didn't converged in requested precision, returns the last value

# xgrid = np.linspace(1.0,2.0,2049)
# fx = 1/(xgrid**2)
# nint, Int, full, err = romberg(1.0,fx,1e-15,4)


## Working with field
# conversion of photons
def ConvertPhoton(x,inp,outp):
  # convert to omega in a.u.
  if (inp == 'omegaau'): omega = x
  elif (inp == 'lambdaSI'): omega = 2.0 * np.pi* units.hbar / (x * units.elmass * units.c_light * units.alpha_fine**2);
  elif (inp == 'lambdaau'): omega = 2.0 * np.pi/(units.alpha_fine*x);
  elif (inp == 'omegaSI'): omega = x * units.TIMEau;
  elif (inp == 'eV'): omega = x * units.elcharge/(units.elmass*units.alpha_fine**2*units.c_light**2);
  elif (inp == 'T0SI'): omega = units.TIMEau*2.0*np.pi/x;
  elif (inp == 'T0au'): omega = 2.0*np.pi/x;
  elif (inp == 'Joule'): omega = x / (units.elmass*units.alpha_fine**2 * units.c_light**2);
  else: sys.exit('Wrong input unit')

  # convert to output
  if (outp == 'omegaau'): return omega;
  elif (outp == 'lambdaSI'): return 2.0*np.pi*units.hbar/(omega*units.elmass*units.c_light*units.alpha_fine**2);
  elif (outp == 'lambdaau'): return 2.0*np.pi/(units.alpha_fine*omega);
  elif (outp == 'omegaSI'): return omega/units.TIMEau;
  elif (outp == 'eV'): return omega/(units.elcharge/(units.elmass*units.alpha_fine**2 * units.c_light**2));
  elif (outp == 'T0SI'): return units.TIMEau*2.0*np.pi/omega;
  elif (outp == 'T0au'): return 2.0*np.pi/omega;
  elif (outp == 'Joule'): return omega*(units.elmass*units.alpha_fine**2 * units.c_light**2);
  else: sys.exit('Wrong output unit')
  
  
def FieldToIntensitySI(Efield):
    return 0.5*units.c_light*units.eps0*Efield**2

def Spectrum_lambda2omega(lambdagrid,Spectrum_lambda, include_Jacobian = True):
    ogrid = 2.*np.pi*units.c_light/lambdagrid
    if include_Jacobian: Spectrum_omega = ((ogrid**2)*Spectrum_lambda)/(2.*np.pi*units.c_light)
    else: Spectrum_omega = Spectrum_lambda    
    return ogrid, Spectrum_omega

## Gaussian beam
def GaussianBeamRayleighRange(w0,lambd):
  return np.pi*w0**2/lambd


def invRadius(z,zR):
  return z/(zR**2+z**2)


def GaussianBeamCurvaturePhase(r,z,k0,zR):
  return 0.5*k0*invRadius(z,zR)*r**2


def waist(z,w0,zR):
  return w0*np.sqrt(1.0+(z/zR)**2);


def GaussianBeam(r,z,t,I0,w0,tFWHM,lambd):
  zR = np.pi*w0**2/lambd;
  w=w0*np.sqrt(1.0+(z/zR)**2);
  I=I0*((w0/w)**2)*np.exp(-2.0*(r/w)**2)*np.exp(-(2.0*np.sqrt(np.log(2.0))*t/tFWHM)**2);
  k0=2.0*np.pi/lambd;
  phase = GaussianBeamCurvaturePhase(r,z,k0,zR);
  return I, phase

def GaussianBeamEfield(r,z,t,E0,w0,tFWHM,lambd, comoving = True):
  # Gaussian beam in the comoving frame with c
  if (not(comoving)): t = t - z/units.c_light
  omega0 = ConvertPhoton(lambd, 'lambdaSI', 'omegaSI')
  k0 = 2.0*np.pi/lambd
  zR = np.pi*w0**2/lambd
  w=w0*np.sqrt(1.0+(z/zR)**2)
  phase_Gouy = np.arctan(z/zR)
  phase_curv = GaussianBeamCurvaturePhase(r,z,k0,zR)
  return E0*(w0/w)*np.exp(-(r/w)**2)*np.exp(-(2.0*np.log(2.0)*t/tFWHM)**2)*np.cos(omega0*t -phase_curv + phase_Gouy)
 
 

# def GaussianBeamEfieldConstruct(r,z,t,E0,w0,tFWHM,lambd): # (t,r,z)
#   Nz = len(z); Nr = len(r); Nt = len(t) 
#   Efield = np.zeros((Nt,Nr,Nz))
#   for k1 in range(Nt):
#     for k2 in range(Nr):
#       for k3 in range(Nz):
#         Efield[k1,k2,k3] = GaussianBeamEfield(r[k2],z[k3],t[k1],E0,w0,tFWHM,lambd)
#   # retarded time
#   return Efield

def GaussianBeamEfieldConstruct(rgrid,zgrid,tgrid,E0,w0,tFWHM,lambd, comoving = True):
  r, z, t = np.meshgrid(rgrid,zgrid,tgrid, indexing='ij')
  if (not(comoving)): t = t - z/units.c_light
  omega0 = ConvertPhoton(lambd, 'lambdaSI', 'omegaSI')
  k0 = 2.0*np.pi/lambd
  zR = np.pi*w0**2/lambd
  w=w0*np.sqrt(1.0+(z/zR)**2)
  phase_Gouy = np.arctan(z/zR)
  invRadius = z/(zR**2+z**2)
  phase_curv = 0.5*k0*invRadius*r**2
  return E0*(w0/w)*np.exp(-(r/w)**2 - (2.0*np.log(2.0)*t/tFWHM)**2)*np.cos(omega0*t -phase_curv + phase_Gouy)

## define dipole function
# def dipoleTimeDomainApp(z_medium,tgrid,r,I0,PhenomParams,tcoeff,rcoeff,LaserParams): # some global variables involved
# #  tcoeff = 4.0*np.log(2.0)*units.TIMEau**2 / ( TFWHMSI**2 )
# #  rcoeff = 2.0/(w0r**2)
#   omega0 = LaserParams['omega0']
#   kw0 = 2.0*np.pi/LaserParams['lambda']
#   phiIR = IRphase(r,z_medium,kw0,LaserParams['zR'])
#   res = []
#   NumHarm = PhenomParams.shape[1]
#   for k1 in range(len(tgrid)):
#     res1 = 0.0*1j;
#     intens = I0*np.exp(-tcoeff*(tgrid[k1])**2 - rcoeff*r**2)
#     for k2 in range(NumHarm): res1 = res1 + intens*np.exp(1j*(tgrid[k1]*omega0*PhenomParams[0,k2]-PhenomParams[1,k2]*intens + PhenomParams[0,k2]*phiIR))
#     res.append(res1); ## various points in time
#   return np.asarray(res)



# part to compute the field from the intensity list




## handling HDF5 files
def adddataset(h_path,dset_name,dset_data,unit):
  dset_id = h_path.create_dataset(dset_name,data=dset_data)
  dset_id.attrs['units']=np.string_(unit)
  return dset_id


def addrealdataset_setprec(h_path, dset_name, dset_data, unit, precision):
  if ( precision == 'f'):
    dset_id = h_path.create_dataset(dset_name, data=dset_data, dtype='f')
    dset_id.attrs['units'] = np.string_(unit)
  elif (precision == 'd'):
    dset_id = h_path.create_dataset(dset_name, data=dset_data, dtype='d')
    dset_id.attrs['units'] = np.string_(unit)


def readscalardataset(file,path,type): # type is (S)tring or (N)umber
  if (type == 'N'): return file[path][()]
  elif (type == 'S'): return file[path][()].decode()
  else: sys.exit('wrong type')


def h5_seek_for_scalar(file,dtype,*args):
    for path in args:
        try:
            return readscalardataset(file,path,dtype)
        except:
            pass
    raise ReferenceError('Dateset not found in args.')


## Other functions

def romberg_test(x_length,fx,eps,n0):
  N = len(fx)
  if ( not IsPowerOf2(N-1) ): sys.exit("romberg: input isn't 2**k+1")
  elif ( not IsPowerOf2(n0) ): sys.exit("romberg: initial stepsize isn't 2**k")
  elif ( n0 > (N-1) ): sys.exit("romberg: initial number of points is larger than provided grid")
  dx = x_length/(N-1)
  step = (N-1)//n0 # adjust to n0 points, divisibility already checked
  k1 = 0
  I = [] # empty list of lists to store the integrals
  while (step >= 1):
    I.append([])
    indices = [k2 for k2 in range(0,N,step)]
    for k2 in range(k1+1):
      if (k2 == 0): value = integrate.trapz(fx[indices],dx=step*dx) # this is inefficient, we already keep the previous results, just refine them
      else: value = (4.0**k2 * I[k1][k2-1] - I[k1-1][k2-1]) / (4.0**k2-1.0)
      I[k1].append(value)

    if (k1>0):# convergence test
      Res = abs(I[k1][k1]-I[k1-1][k1-1])/abs(I[k1][k1])
      if (Res <= eps): return k1, I[k1][k1], Res, I

    step = step // 2
    k1 = k1+1

  return -1, I[-1][-1], Res, I # didn't converged in requested precision, returns the last value
  #  return [-1, I[-1][-1]] # didn't converged in requested precision, returns the last value


def rombergeff_test(x_length,fx,eps,n0):
  N = len(fx)
  if ( not IsPowerOf2(N-1) ): sys.exit("romberg: input isn't 2**k+1")
  elif ( not IsPowerOf2(n0) ): sys.exit("romberg: initial stepsize isn't 2**k")
  elif ( n0 > (N-1) ): sys.exit("romberg: initial number of points is larger than provided grid")
  dx = x_length/(N-1)
  step = (N-1)//n0 # adjust to n0 points, divisibility already checked
  k1 = 0
  I = [] # empty list of lists to store the integrals
  while (step >= 1):
    I.append([])
    indices = [k2 for k2 in range(0,N,step)]
    for k2 in range(k1+1):
      if (k2 == 0): value = integrate.trapz(fx[indices],dx=step*dx) # this is inefficient, we already keep the previous results, just refine them
      else: value = (4.0**k2 * I[k1][k2-1] - I[k1-1][k2-1]) / (4.0**k2-1.0)
      I[k1].append(value)

    if (k1>0):# convergence test
      Res = abs(I[k1][k1]-I[k1-1][k1-1])/abs(I[k1][k1])
      if (Res <= eps): return k1, I[k1][k1], Res, I

    step = step // 2
    k1 = k1+1

  return -1, I[-1][-1], Res, I # didn't converged in requested precision, returns the last value
  #  return [-1, I[-1][-1]] # didn't converged in requested precision, returns the last value


def gabor_transf(arr, t, t_min, t_max, N_steps, a, omegamax = -1.0):
  '''
  Gabor transform
  ===============


  gabor_transf(arr, t, t_min, t_max, N_steps = 400, a = 8)


  Computes Gabor transform for arbitrary number array 'arr' of 
  lenght N. 

      Parameters:
          arr (np.array or list of floats): input data for Gabor transform, length N
          t (np.array or list of floats): time domain for data, length N
          t_min (float): minimum time for Gabor transform domain
          t_max (float): maximum time for Gabor transform domain

      Optional parameters:
          N_steps (int): number of discretization points
          a (float): Gabor window parameter
  
      Returns:
          gabor_transf (np.array(N_steps, N)): Gabor transform of arr 

  Note: time domain 't' must correspond to the array 'arr'.

  Example:
      import numpy as np
      import random

      t_min = 0
      t_max = 1
      N_steps = 100

      x = np.linspace(0,1,100)
      y = [np.cos(2*np.pi*t) + np.sin(np.pi*t) + 0.1*random.randrange(-1,1) for t in x]

      z = gabor_transform(y, x, t_min, t_max, N_steps = N_steps)

  '''
  if (len(arr) != len(t)):
    raise ValueError('Arrays must have same dimension.')

  if (t_max < t_min):
    raise ValueError('Maximum time must be larger than minimum time')

  ### Time domain for Gabor transform
  t_0 = np.linspace(t_min, t_max, N_steps)

  ### Number of sample points for fft
  N = len(arr)

  ### Init np.array
  gabor_transf = np.zeros((N_steps, (N // 2) + 1))
  
  omega = np.linspace(0, (np.pi * (N - 1) / (t[-1] - t[0])), (N // 2) + 1)

  ### Compute Gabor transform (normalised)
  for i in range(0, N_steps):
    fft_loc = np.fft.fft(np.exp(-np.power((t-t_0[i])/a, 2))*arr[:])
    gabor_transf[i, :] = 2.0 / N * np.abs(fft_loc[0:((N // 2) + 1)])
    
  if (omegamax < 0.0):    
      return t_0, omega, gabor_transf
  else:
      k_omegamax = FindInterval(omega, omegamax)
      return t_0, omega[:k_omegamax], gabor_transf[:,:k_omegamax]


## SIGNAL PROCESSING
def identity(M):
    return np.ones(M)

def filter_box(M,xgrid,x_box,apply_blackman = False):
    if apply_blackman:
        k1 = FindInterval(xgrid, x_box[0])
        k2 = FindInterval(xgrid, x_box[1])
        # np.blackman(k2-k1)
        return np.concatenate((np.zeros(k1), np.blackman(k2-k1), np.zeros(M-k2)))
    else:
        return (x_box[0] <= xgrid)*(xgrid <= x_box[1])


def apply_filter(fx,filter_funct,*args,**kwargs):
    if (filter_funct == identity): return fx
    else:   return filter_funct(len(fx),*args,**kwargs) * fx


def clamp_array(array,filter_type,filter_threshold):
    array_flat = array.flatten()
    if (filter_type is None): return array
    elif (filter_type == 'low_abs_cut'):
        array_flat[np.abs(array_flat) < filter_threshold*np.max(np.abs(array_flat))] = 0.0
        return array_flat.reshape(array.shape)
    elif ('low_abs2_cut'):
        # array_flat[np.abs(array_flat)**2 <= filter_threshold*np.max(np.abs(array_flat))**2] = 0.0
        array_flat[np.abs(array_flat)**2 <= filter_threshold*np.max(np.abs(array_flat))**2] = filter_threshold*np.max(np.abs(array_flat))**2
        return array_flat.reshape(array.shape)
        #return array[np.abs(array[:])**2 > filter_threshold*np.max(np.abs(array[:]))**2]
    else:
        warnings.warn('filter wrongly specified: returned original')
        return array
## BEAM MEASURE
def measure_beam_RMS(x,fx):
    return np.sqrt(
        integrate.trapz( (x**2) * fx, x=x) / integrate.trapz(fx, x=x)
        )


def measure_beam_max_ratio_zeromax(x,fx,alpha):
    N = len(fx); fxmax = fx[0]; fxFWHM = alpha*fxmax
    
    for k1 in range(N):
        if (fxFWHM > fx[k1]): break
    
    if (k1 == (N-1)): return np.Inf
    
    x1 = x[k1-1]; x2 = x[k1]
    fx1 = fx[k1-1]; fx2 = fx[k1]
    dx = x2 - x1; dfx = fx2 - fx1

    if (abs(dfx) < np.spacing(fxmax)): return x1 # compare with an appropriate machine epsilon for flat cases
    else:   return (dx*fxFWHM + fx2*x1 - fx1*x2) / dfx
    
    
def measure_beam_FWHM_zeromax(x,fx):
    return measure_beam_max_ratio_zeromax(x,fx,0.5)
    

def measure_beam_E_alpha_zeromax(x,fx,alpha): #  taken from my Matlab implementation
    E_tot = integrate.trapz(fx, x=x)
    E_prim_norm = (1.0/E_tot) * integrate.cumtrapz(fx, x=x, initial=0)
    
    
    # k1 = FindInterval(E_prim_norm, alpha) # It'd be faster, need to deal with the boundaries.
    
    N = len(fx)
    for k1 in range(N):
        if (E_prim_norm[k1] > alpha): break
    
    if (k1 == (N-1)): return np.Inf
    elif (k1 == 0): return x[0] 

    x1 = x[k1-1]; x2 = x[k1]
    fx1 = E_prim_norm[k1-1]; fx2 = E_prim_norm[k1]
    dx = x2 - x1; dfx = fx2 - fx1

    if (abs(dfx) < np.spacing(1.0)): return x1 # compare with an appropriate machine epsilon for flat cases
    else:   return (dx*alpha + fx2*x1 - fx1*x2) / dfx
    
    
    # cumtrapz(x, initial=0)
    
    
    
# interpolation etc.
def interpolate_2D(x,y,fxy,Nx,Ny):
    f_interp = interpolate.interp2d(x,y,fxy)
    xnew = np.linspace(x[0], x[-1], Nx)
    ynew = np.linspace(y[0], y[-1], Ny)
    fxy_interp = f_interp(xnew, ynew)
    return xnew, ynew, fxy_interp


