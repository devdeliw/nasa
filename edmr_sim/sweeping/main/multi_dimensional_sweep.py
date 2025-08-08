import numpy as np
import xarray as xr 
import itertools, tqdm

from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from ...solving_SLE.main.utils.load_params import load_params
from ...fitting.main.least_squares import EDMRLeastSquaresPlotter


YAML_PATH  = Path(__file__).parent.parent.parent / "solving_SLE/main/utils/params.yaml"
RAW_PARAMS = load_params(YAML_PATH)

def _update_parameters(
    parameter_keys: List[str],
    new_values:     List[float],
    base_params:    dict[str, float] = RAW_PARAMS
):
    u = dict(zip(parameter_keys, new_values))
    return {**base_params, **u}

@dataclass
class SweepConfig:
    B_range:           Tuple[float, float]
    parameter_keys:    List[str]
    parameter_ranges:  List[Tuple[float, float, int]]
    out_directory:     Path
    out_filename:      str 
    n_points_interp:   int = 100
    n_final_points:    int = 4096
    init_check:        bool = True
    disable_warning:   bool = False
    verbose:           bool = False
    save_plot:         bool = False
    dpi:               int = 80

class Sweep:

    """ 
    Performs a sweep across n different parameters *inter-dependently*. 
    i.e., builds an n-dimensional array with simulated currents for every 
    combination of every parameter in the config arguments.

    * computationally intensive for large n * 

    """

    def __init__(self, cfg: SweepConfig):
        assert len(cfg.parameter_keys) == len(cfg.parameter_ranges)
        self.cfg      = cfg
        self.axes = {
            key: np.linspace(*rng)
            for key, rng in zip(cfg.parameter_keys, cfg.parameter_ranges)
        }
        self.B_data = np.linspace(*cfg.B_range, cfg.n_final_points)
        self.I_data = np.zeros_like(self.B_data)

    def _sci_tuple(self, t: tuple[float, ...], precision: int = 3) -> str:
        fmt = f"{{:.{precision}e}}"
        parts = (fmt.format(x) for x in t[:-1])
        return "(" + ", ".join(parts) + f", {t[-1]})"

    def _build_ranges(self): 
        return [ 
            (min, max, n_steps) 
            for min, max, n_steps in self.cfg.parameter_ranges
        ]

    def _print_cfg(self, keys, ranges):
        range_strs = [self._sci_tuple(r) for r in ranges]

        max_key_len   = max(len(k) for k in keys)
        max_range_len = max(len(rs) for rs in range_strs)

        for idx, k in enumerate(keys):
            rs = range_strs[idx]
            print(
                f"[INFO]:: "
                f"{k:^{max_key_len}} :: "
                f"{rs:^{max_range_len}}"
            )

    def run(self) -> np.ndarray:
        axis_lengths = [len(self.axes[k]) for k in self.cfg.parameter_keys]
        total_steps  = int(np.prod(axis_lengths))
        shape        = (*axis_lengths, self.cfg.n_final_points)
        out          = np.empty(shape, dtype=float)

        idx_iter = itertools.product(*(range(L) for L in axis_lengths))
        val_iter = itertools.product(*(self.axes[k] for k in self.cfg.parameter_keys))
 
        keys = self.cfg.parameter_keys
        ranges = self._build_ranges() 
        self._print_cfg(keys, ranges)

        for (idxs, vals) in tqdm.tqdm(
            zip(idx_iter, val_iter),
            total=total_steps,
        ):
            params = _update_parameters(self.cfg.parameter_keys, list(vals))
            plotter = EDMRLeastSquaresPlotter(
                raw_B_data=self.B_data,
                raw_I_data=self.I_data,
                n_points_interp=self.cfg.n_points_interp,
                init_check=self.cfg.init_check,     
                parameters=params,
                save_plot=self.cfg.save_plot,
                verbose=self.cfg.verbose,
                dpi=self.cfg.dpi,
                disable_warning=self.cfg.disable_warning, 
            )
            _, dI = plotter.plot_fitted_EDMR_spectra(
                filename=None, 
            )
            out[(*idxs, slice(None))] = dI

        return out 

    def render(self): 
        """ 
        Runs the multi-dimensional sweep and saves an 
        xarray DataArray for organization. 

        """ 

        dims = [*self.cfg.parameter_keys, "B"] 
        coords = {
            key: self.axes[key] for key in self.cfg.parameter_keys
        }
        coords["B"] = self.B_data 

        out = self.run()

        da = xr.DataArray( 
            data   = out, 
            dims   = dims, 
            coords = coords, 
            name   = "dI/dB", 
        )

        self.cfg.out_directory.mkdir(exist_ok=True, parents=True)
        out_file = self.cfg.out_directory / self.cfg.out_filename 
        da.to_netcdf(out_file) 
        print(f"[INFO] Saved NetCDF to {out_file}")

        return da 

if __name__ == "__main__": 

    cfg = SweepConfig(
        B_range          = (0, 100),
        parameter_keys   = [
            "nu", 
        ],
        parameter_ranges = [
            (1.90e8, 2.1e8, 1000),
        ],
        n_points_interp  = 100,
        n_final_points   = 4096, 
        save_plot        = False,
        init_check       = False,
        disable_warning  = True,
        out_directory    = Path(__file__).parent / "netCDF", 
        out_filename     = "sweep.nc" 
    )
    sweep = Sweep(cfg)
    data  = sweep.render()





