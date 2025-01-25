import xarray as xr

import xarray as xr
nc1 = xr.open_dataset("Maindata/era5_2014_2015/data_stream-oper_stepType-accum.nc")
nc2 = xr.open_dataset("Maindata/era5_2014_2015/data_stream-oper_stepType-instant.nc")
print(nc1.variables, nc2.variables)
print(nc1.dims, nc2.dims)
