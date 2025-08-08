import xarray as xr
import numpy as np 

from pathlib import Path 
from typing import Optional, List
from functools import cached_property
from scipy.integrate import cumulative_trapezoid

NET_CDF_DIR = Path(__file__).parent / "netCDF" 

class ParseNetCDF(): 
    def __init__(
        self, 
        cdf_filename:  str, 
        cdf_directory: Path = NET_CDF_DIR, 
    ): 
        self.cdf_filename  = cdf_filename 
        self.cdf_directory = cdf_directory 

    @cached_property 
    def load_cdf(self) -> xr.DataArray: 
        try: 
            return xr.load_dataarray(self.cdf_directory / self.cdf_filename)
        except FileNotFoundError: 
            print(
                f"[ERROR]: "
                f"{self.cdf_filename} not found in "
                f"{self.cdf_directory}"
            ) 
            raise 

    @cached_property 
    def load_B(self) -> np.ndarray: 
        da = self.load_cdf 
        return da.coords["B"].values 

    @cached_property
    def load_dims(self) -> tuple: 
        """ 
        Loads the parameters stored in DataArray 
        ("nu", "omega1") 

        """ 
        da = self.load_cdf
        return tuple(p for p in da.dims if p != 'B')

    @cached_property
    def load_param_dict(self) -> dict[str, List[float]]:
        """ 
        Returns dict of parameters with values swept over 
        { 
            "nu": [1.998e6, ..., 2.002e6], 
            "omega1": [2.000e5, ..., 2.000e6], 
        }

        """
        da   = self.load_cdf 
        dims = [p for p in self.load_dims if p != 'B']
        
        param_dict = {} 
        for param in dims: 
            param_dict[param] = da.coords[param].values
        return param_dict 

class ParameterSlice: 
    def __init__(
        self, 
        dataarray: xr.DataArray, 
        parameter_values: dict[str, float], 
        tolerance: Optional[dict[str, float]] = None, 
        method: Optional[str] = "nearest" 
    ): 
        self._da = dataarray.sel(
            parameter_values, 
            tolerance=tolerance, 
            method=method, 
        )

    @property 
    def B(self) -> np.ndarray: 
        return self._da['B'].values     
    
    @property 
    def dI(self) -> np.ndarray: 
        return self._da.values 

    @property
    def I(self) -> np.ndarray: 
        return cumulative_trapezoid(
            self.dI, 
            self.B,
            initial=0
        ) 

class RenderNetCDF(ParseNetCDF): 
    def __init__(
        self, 
        cdf_filename:   str, 
        cdf_directory:  Path = NET_CDF_DIR
    ): 
        super().__init__(cdf_filename, cdf_directory) 
        self.B: np.ndarray                      = self.load_B
        self.dataarray: xr.DataArray            = self.load_cdf 
        self.parameters: tuple                  = self.load_dims
        self.param_dict: dict[str, List[float]] = self.load_param_dict

    def slice(
        self, 
        parameter_values: dict[str, float], 
        *, 
        tolerance: Optional[dict[str, float]] = None, 
        method: Optional[str] = "nearest"
    ) -> ParameterSlice: 
        return ParameterSlice(
            self.dataarray, 
            parameter_values, 
            tolerance, 
            method 
        )





         

if __name__ == "__main__":
    cdf_filename = "sweep.nc" 

    renderer = RenderNetCDF(cdf_filename) 
    slice = renderer.slice(
                parameter_values={
                    "nu": 1.980e8, 
                    "omega1": 2.0e6
        }
    )

    print(slice.dI)


