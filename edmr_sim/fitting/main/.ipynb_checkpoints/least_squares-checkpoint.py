# Code written by Deval Deliwala 
# NASA Glenn Research Center 

import time
import numpy as np
import logging, pickle
import matplotlib.pyplot as plt

plt.rcParams['font.family']      = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

from rich import print 
from pathlib import Path 
from datetime import datetime
from scipy.optimize import least_squares 
from dataclasses import dataclass, field
from scipy.interpolate import interp1d

from solving_SLE.main.run_solver import PARAMETER_ORDER, PARAMETER_FILE 
from solving_SLE.main.run_solver import _convert_dict_to_theta
from solving_SLE.main.run_solver import _convert_theta_to_dict
from solving_SLE.main.utils.load_params import load_params, PARAMETER_KEYS
from solving_SLE.main.sle_model  import compute_edmr_spectra
from solving_SLE.main.utils.load_bounds import lower_bounds, upper_bounds

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

@dataclass 
class LeastSquaresParameters: 
    """
    Attributes:
        * default_parameters: bool 
            * Whether to use the parameters in the saved param.yaml 
            * see ...solving_SLE.main.utils.load_params::load_params()
        * ansatz_parameters: dict[str, float] 
            * Initial guess parameters, must have all keys.
        * lower_bound, upper_bound: np.ndarray
            * Parameter bounds for fitting routine 

    ~/ WARNING ~/ 
    Do not change these default parameters unless you are not working 
    with the params.yaml file.  

    """
    default_parameters: bool = True 
    ansatz_parameters: dict[str, float] | None = None
    lower_bound: dict[str, float] = field(default_factory=lambda: lower_bounds.copy())
    upper_bound: dict[str, float] = field(default_factory=lambda: upper_bounds.copy())

@dataclass 
class LeastSquaresMultiThread:
    """ 
    Attributes: 
        * n_jobs: int 
            * Number of cores to use
            * DEFAULT: 1
        * blocks_per_core: int
            * Number of B-blocks per core for multithreading
            * DEFAULT: 1 

    1 core is the fastest -- 0.6s for a 100point sweep. 
    """
    n_jobs: int          = 1  
    blocks_per_core: int = 1

@dataclass 
class LeastSquaresPickling:
    """ 
    Attributes: 
        * save_pickle: bool
            * Saves updating parameters by fitting to pickle file.
        * pickle_dirs: Path 
            * Directory to dump pickled parameters every iteration.
    """
    save_pickle: bool = True 
    pickle_dirs: Path = Path(__file__).parent / "utils/" 


class EDMRLeastSquares(): 
    """ 
    Performs Least-Squares Optimization to fit the derived EDMR model 
    (see ...solving_SLE/main/) against raw EDMR experimental data. 

    """
    def __init__( 
            self, 
            raw_B_array: np.ndarray, 
            raw_I_array: np.ndarray,
            *, 
            n_points_interp: int                        = 100, 
            alpha: float                                = 1.0,
            show_progress: bool                         = True, 
            update_param_yaml: bool                     = False,
            init_check: bool                            = True, 
            modulate: bool                              = True, 
            lsq_params: LeastSquaresParameters          = LeastSquaresParameters(), 
            multithread_params: LeastSquaresMultiThread = LeastSquaresMultiThread(), 
            pickle_params: LeastSquaresPickling         = LeastSquaresPickling(), 
    ): 
        """
        Every parameter below `alpha` is insignificant; 
        Assuming you update parameters via the params.yaml file. 
        Using the default values for the rest are fine. 

        Args: 
            * raw_B_array: np.ndarray
                * Raw EDMR B-field Sweep Data 
            * raw_I_array: np.ndarray 
                * Raw EDMR measured current 

            ~/ Keyword only ~/ 

            * n_points_interp: int 
                * Number of points to discretize B by for simulating
                * Reduces complexity by simulating less B values  
                * with cubic-spline interpolation for residuals. 
                * A good heuristic is 1-point / Gauss
            * alpha: float
                * Residual weight exponent to emphasize derivative 
            * show_progress: bool 
                * Display tqdm progress bar.
            * update_param_yaml: bool 
                * Update the parameter file with the latest stored pickle 
                * from a previous run. Uses these params as well.

        Variable Notation: 
            * `parameters` is a dict[str, float] of parameter names and values 
            * `parameter_vec` is a np.ndarray built from the `parameters` dict
        """

        self._lsq = lsq_params 
        self._mt  = multithread_params 
        self._pkl = pickle_params

        self._save_pickle = self._pkl.save_pickle  
        self._pickle_dirs = self._pkl.pickle_dirs
        self._pickle_dirs.mkdir(exist_ok=True, parents=True)

        # Initial guess parameters 
        if self._lsq.ansatz_parameters is None or self._lsq.default_parameters: 
            ansatz_parameters = load_params(PARAMETER_FILE) 
        else: 
            ansatz_parameters = self._lsq.ansatz_parameters
        if update_param_yaml: 
            _update_yaml_with_pickle(self)
            ansatz_parameters = load_params(PARAMETER_FILE)
        self.verify_parameters_complete(ansatz_parameters)
 
        # Some "parameters" are constants, h, hbar, mu_B, mu_N.
        # Since these are part of the parameter vector, the LSQ routine 
        # would run the EDMR simulation 4 more times than needed *every* iteration. 
        #
        # Hence we remove constants from ansatz_parameters and reinsert them 
        # when calculating the EDMR simulated spectra.  
       
        lower_bound = self._lsq.lower_bound 
        upper_bound = self._lsq.upper_bound 
        assert lower_bound.keys() == upper_bound.keys(), \
            "lower_bound.keys() != upper_bound.keys()"
        assert upper_bound.keys() == ansatz_parameters.keys(), \
            "upper_bound.keys() != ansatz_parameters.keys()" 

        keys_with_common_values = [
            k for k in ansatz_parameters.keys() 
            if (
                np.isclose(ansatz_parameters[k],lower_bound[k], atol=1e-11) and 
                np.isclose(lower_bound[k], upper_bound[k], atol=1e-11)
            )
        ]
        parameters   = {
            k: v for k, v in ansatz_parameters.items() if k not in keys_with_common_values
        }
        lower_bound = {
            k: v for k, v in lower_bound.items() if k not in keys_with_common_values
        } 
        upper_bound = {
            k: v for k, v in upper_bound.items() if k not in keys_with_common_values
        } 

        self.constants     = {k: ansatz_parameters[k] for k in keys_with_common_values}
        self.parameter_vec = _convert_dict_to_theta(parameters)
        self._lower_bounds = _convert_dict_to_theta(lower_bound)
        self._upper_bounds = _convert_dict_to_theta(upper_bound) 

        # Raw EDMR experimental data
        self.raw_B_data = np.asarray(raw_B_array) 
        self.raw_I_data = np.asarray(raw_I_array) 
        if len(raw_B_array) > n_points_interp: 
            self.B_discretized = np.linspace(
                raw_B_array[0], 
                raw_B_array[-1], 
                n_points_interp
            )
        else: self.B_discretized = self.raw_B_data

        self.alpha            = alpha
        self._n_points        = n_points_interp
        self._show_progress   = show_progress
        self.modulate         = modulate

        # multithreading
        self._n_jobs          = self._mt.n_jobs 
        self._blocks_per_core = self._mt.blocks_per_core 

        # preliminary checks 
        if not init_check: 
            print(
                "WARNING: not performing preliminary checks. "
                "This is highly recommended and only takes ~0.1s per iteration."
            )
        self.init_check = init_check 

        self.edmr_functional = compute_edmr_spectra 

    def verify_parameters_complete(self, parameters: dict[str, float]):
        missing = set(PARAMETER_ORDER) - parameters.keys()
        if missing:
            logger.error("`ansatz_parameters` is incomplete:")
            print(f" Missing Parameters: ")
            print(missing)
            raise ValueError("`ansatz_parameters` incomplete. See above.")
 
    def _reinsert_constants_to_parameters(self, parameters: dict[str, float]): 
        return {**self.constants, **parameters}

    def _parameter_vec_to_full_dict(self, parameter_vec: np.ndarray): 
        """ 
        Converts the ndarray parameter vector returned by least_squares 
        back into a parameters dict[str, float] with constants included. 

        """
        parameters = _convert_theta_to_dict(parameter_vec)
        return self._reinsert_constants_to_parameters(parameters) 

    def compute_residuals(self, parameter_vec: np.ndarray):
        """ 
        Args: 
            * parameters: dict[str, float], 
                * Model parameters that are undergoing optimization 
            * alpha: int 
                * Exponent of derivative residual weighting 
                * Weighting by the derivative emphasizes EDMR features 
                * alpha = 0 weights every point equally
            * save_pickle: bool 
                * Saves lsq parameters everytime they're updated. 
        """ 
        parameters = self._parameter_vec_to_full_dict(parameter_vec)
        simulated_current = self.edmr_functional( 
            B_array=self.B_discretized, 
            parameters=parameters, 
            modulate=self.modulate, 
            n_jobs=self._n_jobs, 
            blocks_per_core=self._blocks_per_core, 
            show_progress=self._show_progress, 
            init_check=self.init_check
        )

        # Interpolating discrete simulated current 
        # to same length as self.raw_B_data for residuals 
        I_interpolated = interp1d(
            self.B_discretized, 
            simulated_current, 
            kind="cubic", 
            assume_sorted=True
        )(self.raw_B_data) 
    
        # Weight residuals by derivative (emphasize features) 
        w = np.abs(np.gradient(self.raw_I_data, self.raw_B_data)) 
        w = w ** self.alpha
        return w * (I_interpolated - self.raw_I_data) 

    def _callback_save_pickle(self, parameter_vec: np.ndarray):
        """
        Saves the most recent parameter + constant dictionary in a pickle 
        after every least_squares iteration. 

        * Only if self._save_pickle == True *

        """
        if self._save_pickle:
            parameters = self._parameter_vec_to_full_dict(parameter_vec) 
            pickle_fname = self._pickle_dirs / "lsq_param.pickle"
            with open(pickle_fname, "wb") as f: 
                pickle.dump(parameters, f)
                logger.info(f"~/ Parameter pickle saved. ~/")

    def fit(
        self, 
        ftol: float = 1e-12, 
        xtol: float = 1e-12, 
        gtol: float = 1e-12, 
    ): 
        if self._pkl.save_pickle: 
            logger.info(" ~/ Pickle Saving Active /~") 
            logger.info(" This will consume ~130us every full iteration.")
            logger.info(f" {self._pickle_dirs / 'lsq_param.pickle'}")

        start = datetime.now() 
        print(f"\nLeast-Squares started at {start:%Y‑%m‑%d %H:%M:%S}")
        print("--------------------------------------------") 
        print("Ansatz Parameters: ") 
        print(_convert_theta_to_dict(self.parameter_vec))
        print(f"xtol = {xtol:2g}") 
        print(f"ftol = {ftol:2g}")
        print(f"gtol = {gtol:2g}")
        print(f"# points  = {self._n_points}" )
        print(f"# threads = {self._n_jobs}\n") 

        self.result = least_squares( 
            fun     = self.compute_residuals, 
            x0      = self.parameter_vec, 
            bounds  = (self._lower_bounds, self._upper_bounds), 
            ftol    = ftol, 
            xtol    = xtol, 
            gtol    = gtol, 
            verbose = 2, 
            jac     = '2-point', 
            method  = 'trf', 
            callback= self._callback_save_pickle
        )

        end = datetime.now() 
        delta = (end - start).total_seconds()
        print(f"\nLeast-Squares ended at {end:%Y-%m-%d %H:%M:%S}")
        print(f"Time Elapsed: {delta / 3600}h\n")

        return _convert_theta_to_dict(self.result.x)


class EDMRLeastSquaresPlotter: 
    """ 
    Renders a plot for visualizing the EDMRLeastSquares spectra result. 
    Overlays against the raw experimental EDMR data. 

    """ 

    def __init__(
        self,
        raw_B_data: np.ndarray,
        raw_I_data: np.ndarray, 
        parameters: dict[str, float] | str, 
        *, 
        n_points_interp: int    = 100, 
        save_plot: bool         = True, 
        save_directory: Path    = Path(__file__).parent.parent / "media",
        multithread: bool       = True, 
        init_check: bool        = True,
        dpi: int                = 300, 
        plot_raw: bool          = True,
        verbose: bool           = True,
        modulate: bool          = True, 
        disable_warning: bool   = False
    ): 
        """ 
        Args: 
            * raw_B_data, raw_I_data: np.ndarray 
                * raw EDMR experimental data. 
            * parameters: dict[str, float] | "pickle"  
                * parameters returned from EDMRLeastSquares::fit()
                * or custom params if you want, as long as they're complete. 
                * if "pickle" loads parameters from loaded lsq_param.pickle file
            
            ~/ Keyword only /~ 
            * n_points_interp: int 
                * # points to interpolate simulated spectra from 
                * Good heuristic is 1 point / Gauss 
            * save_plot, save_directory: bool, Path 
                * Whether to save the figure, and where to place it. 
            * multithread: bool 
                * If active, will use the # cores defined in 
                * LeastSquaresMultiThread() 
            * init_check: bool 
                * If active, will run check to ensure calculation 
                * proceeds correctly; Roughly 0.3s every iteration. 
        """
        self.raw_B_data = raw_B_data
        self.raw_I_data = raw_I_data

        if parameters == "pickle": 
            with open(
                LeastSquaresPickling.pickle_dirs / "lsq_param.pickle", 
                "rb"
            ) as f: 
                parameters = pickle.load(f)
        
        self.parameters = parameters 
        self.n_points_interp = n_points_interp 
        
        if multithread: 
            mt_instance  = LeastSquaresMultiThread() 
            self._n_jobs = mt_instance.n_jobs 
            self._blocks_per_core = mt_instance.blocks_per_core
        else: 
            self._n_jobs = 1 
            self._blocks_per_core = 1

        if not init_check and not disable_warning: 
            print(
                "WARNING: not performing preliminary checks. "
                "This is highly recommended and takes ~0.1s per iteration."
            )
        self.init_check     = init_check
        self.save_plot      = save_plot 
        self.save_directory = save_directory
        self.dpi            = dpi 
        self.plot_raw       = plot_raw
        self.verbose        = verbose
        self.modulate       = modulate

    def fitted_EDMR_spectra(self, B_array: np.ndarray):

        if type(self.parameters) != dict: 
            raise TypeError("Parameters must be dict[str, float].")

        I_array = compute_edmr_spectra(
            B_array=B_array, 
            parameters=self.parameters, 
            modulate=self.modulate, 
            n_jobs=self._n_jobs, 
            blocks_per_core=self._blocks_per_core, 
            show_progress=False, 
            init_check=self.init_check
        ) 
        return I_array

    def plot_fitted_EDMR_spectra(self, filename: str | None = None): 

        min_B = min(self.raw_B_data) 
        max_B = max(self.raw_B_data)  

        small_B_array = np.linspace(
            min_B, max_B, 
            self.n_points_interp
        )

        if self.verbose: print(f"Rendering EDMR LSQ Fit with {self.n_points_interp} steps.") 

        fitting_start = time.time()
        dI = self.fitted_EDMR_spectra(B_array=small_B_array)
        interpolator = interp1d(
            small_B_array, 
            dI, 
            kind="cubic",
            assume_sorted=True
        )
        fitting_stop = time.time()

        big_B_array = self.raw_B_data if len(self.raw_B_data) > self.n_points_interp else small_B_array
        dI = interpolator(big_B_array)
       
        if self.save_plot: 
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.subplots_adjust(right=0.8)

            ax.plot(big_B_array, dI, c='r', ls=':', lw=1, label="Simulated")
            if self.plot_raw:
                ax.plot(self.raw_B_data, self.raw_I_data, c='k', lw=1.5, label="Raw") 

            ax.axvline(0, ls=":", color="grey", alpha=0.6)
            ax.axhline(0, ls=":", color="grey", alpha=0.6)
            ax.set_xlabel("B [G]", fontsize=15)
            ax.set_ylabel(r"$\partial I / \partial B$ [nA]", fontsize=15) 
            ax.legend(frameon=False, fontsize=15, loc='upper right')

            max_parameter_key_len = max(len(k) for k in self.parameters)

            if type(self.parameters) == dict: 
                parameter_lines = [
                f"{k.ljust(max_parameter_key_len)} = {self.parameters[k]:.5g}"
                for k in self.parameters
                ]
                parameter_text = "\n".join(parameter_lines)

                bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
                ax.text(
                    1.02, 0.5, parameter_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="center",
                    horizontalalignment="left",
                bbox=bbox_props,
                fontfamily="monospace" 
                )
            else: 
                logger.warning("`self.parameters` not dict[str, float], something went wrong.")
                logger.warning("Rendering EDMR Least-Squares Fit without parameter box.")

            self.save_directory.mkdir(parents=True, exist_ok=True) 
            if not filename: 
                filename = "EDMR_LeastSquares_no_mod.png"  
            plt.show()
            fig.savefig(self.save_directory / filename, dpi=self.dpi) 
            plt.close()
            if self.verbose: logger.info(" fig::%s \n", filename)

        if self.verbose: print(f"Took {fitting_stop - fitting_start} seconds.")
        return big_B_array, dI 

def _update_yaml_with_pickle(fitter: EDMRLeastSquares): 
    from ruamel.yaml import YAML

    with open(fitter._pickle_dirs / "lsq_param_pkl.pickle", "rb") as f: 
        parameter_vector = pickle.load(f)
        parameters = _convert_theta_to_dict(parameter_vector) 

    yaml = YAML() 
    yaml.indent(mapping=2, sequence=4, offset=2) 
    with open(PARAMETER_FILE) as fp: 
        cfg = yaml.load(fp)

    for key, parameter_names in PARAMETER_KEYS.items(): 
        for parameter_name in parameter_names: 
            cfg[key][parameter_name] = parameters[parameter_name] 

    with open(PARAMETER_FILE, 'w') as fp: 
        yaml.dump(cfg, fp) 
        logger.info(f" {PARAMETER_FILE} overwritten.\n") 

def _build_least_squares_fitter(
    raw_B_data: np.ndarray,
    raw_I_data: np.ndarray,
    *,
    n_points_interp: int            = 100,
    B_range: tuple[float, float]    = (-np.inf, np.inf),  
    show_progress: bool             = False,
    init_check: bool                = True
) -> EDMRLeastSquares: 
    """ 
    Builds a `EDMR_LeastSquares` instance for fitting with default parameters 
    (from params.yaml). 

    Args: 
        * raw_B_data, raw_I_data: np.ndarray 
            * raw EDMR experimental data. 
        * n_points_interp: int 
            * How many points to use for simulation (resolution) 
        * n_jobs: int 
            * Number of cores for simulation 
        * blocks_per_core: int
            * Number of B-blocks per core 
        * B_range: tuple[float, float] 
            * B_range to be swept over -- included 
            * if B_range exceeds range of `raw_B_data`: 
            *   `raw_B_data` is used. 
            * if B_range is less than range of `raw_B_data`: 
            *   `raw_B_data` is truncated to match `B_range`
        * show_progress: bool 
            * Display tqdm progress bar 
    """

    raw_B_data = np.asarray(raw_B_data) 
    raw_I_data = np.asarray(raw_I_data) 
    assert len(raw_B_data) == len(raw_I_data), "B and I arrays length mismatch."

    if B_range: 
        mask = (raw_B_data >= B_range[0]) & (raw_B_data <= B_range[1]) 
        raw_B_data = raw_B_data[mask] 
        raw_I_data = raw_I_data[mask] 

    return EDMRLeastSquares( 
            raw_B_array=raw_B_data,
            raw_I_array=raw_I_data, 
            n_points_interp=n_points_interp, 
            show_progress=show_progress, 
            init_check=init_check,
            update_param_yaml=False
    )


if __name__ == "__main__": 
    from .utils.csv_to_npz import npz_to_arrays 


    # Example Usage
    # ------------- 

    raw_B_data, raw_I_data = npz_to_arrays("3V-2G-200MHz.npz")
    
    fitter = _build_least_squares_fitter(
        raw_B_data=raw_B_data, 
        raw_I_data=raw_I_data, 
        n_points_interp=100,
        init_check=False,
    )

    plotter = EDMRLeastSquaresPlotter(
        raw_B_data=raw_B_data, 
        raw_I_data=raw_I_data, 
        parameters=load_params(PARAMETER_FILE), 
        init_check=False,
        n_points_interp=4096,
        dpi=500, 
        plot_raw=False,
        verbose=True,
        modulate=False, 
    )

    # Runs the scipy.optimize.least_squares routine 
    result = fitter.fit() 

    # Displays a fitted EDMR spectra from given parameters. 
    # To use recent fitted parameters, set
    # plotter.parameters = "pickle".

    plotter.plot_fitted_EDMR_spectra()
 





