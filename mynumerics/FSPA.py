import numpy as np
from scipy import interpolate

# a = 5

# init from hdf5 table and obtain class with appropriate values and methods, provide opened hdf file for versatility
# print warning here about the alignment of the data
def get_dphase(h5file,Igrid_path,Hgrid_path,dphi_table_path,extrapolate='boundary'):
    Igrid = np.squeeze(h5file[Igrid_path][:])
    Hgrid = np.squeeze(h5file[Hgrid_path][:])
    dphi_table = h5file[dphi_table_path][:]
    # print('shape',dphi_table.shape)
    interp_funct = {}
    for k1 in range(len(Hgrid)):
        # print('I_shape',Igrid.shape)
        # print('phi_shape',np.squeeze(dphi_table[:,k1]).shape)
        if (extrapolate=='boundary'):
            local_table = {
                Hgrid[k1] : interpolate.interp1d(Igrid, dphi_table[:,k1], bounds_error=False, fill_value=(dphi_table[0,k1], np.nan) ) 
            }
        elif (extrapolate=='zero'):
            local_table = {
                Hgrid[k1] : interpolate.interp1d(Igrid, dphi_table[:,k1], bounds_error=False, fill_value=(0.0, np.nan) ) 
            }
        else:
            raise Exception('Wrong boundary method')
        interp_funct.update(local_table)
    return interp_funct

get_interp = get_dphase

#interpolation to get alpha

