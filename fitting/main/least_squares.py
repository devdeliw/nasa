import numpy as np 
import logging, pickle, os 

from rich import print 
from pathlib import Path 
from scipy.optimize import OptimizeResult, least_squares 
from scipy.interpolate import interp1d 
from ...solving_SLE.main.run_solver import PARAMETER_ORDER, PARAMETER_FILE 
from ...solving_SLE.main.run_solver import _convert_dict_to_theta
from ...solving_SLE.main.run_solver import _convert_theta_to_dict
from ...solving_SLE.main.utils.load_params import load_params, PARAMETER_KEYS
from ...solving_SLE.main.sle_model  import compute_edmr_spectra
from ...solving_SLE.main.utils.load_bounds import lower_bounds, upper_bounds

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 


class EDMRLeastSquares(): 
    """ 
    Performs Least-Squares Optimization to fit the derived EDMR model 
    (see ...solving_SLE/main/) against raw EDMR experimental data. 

    """

    def __init__( 
            self, 
            raw_B_array: np.ndarray, 
            raw_I_array: np.ndarray, 
            n_points_interp: int                        = 100, 
            alpha: float                                = 1.0,
            ansatz_parameters: dict[str, float] | None  = None,
            default_parameters: bool                    = True, 
            lower_bound: dict[str, float] | None       = None,  
            upper_bound: dict[str, float] | None       = None, 
            n_jobs: int                                 = 2,  
            blocks_per_core: int                        = 8, 
            show_progress: bool                         = True, 
            save_pickle_every_iteration: bool           = True,
            save_pickle_filepath: Path                  = Path(os.getcwd()).parent / "utils", 
            update_param_yaml: bool                     = False
    ): 
        """
        Every parameter below `n_points_interp` is insignificant; 
        Assuming you update parameters via the params.yaml file. 
        Using the default values for the rest are fine. 

        Args: 
            * raw_B_array: np.ndarray
                * Raw EDMR B-field Sweep Data 
            * raw_I_array: np.ndarray 
                * Raw EDMR measured current 
            * n_points_interp: int 
                * Number of points to discretize B by for simulating
                * Reduces complexity by simulating less B values  
                * with cubic-spline interpolation for residuals. 
                * A good heuristic is 1-point / Gauss 

            ~/ INSIGNIFICANT /~
            * alpha: float
                * Residual weight exponent to emphasize derivative 
            * ansatz_parameters: dict[str, float] 
                * Initial guess parameters, must have all keys. 
            * default_parameters: bool 
                * Whether to use the parameters in the saved param.yaml 
                * see ...solving_SLE.main.utils.load_params::load_params() 
            * lower_bounds, upper_bounds: np.ndarray
                * Parameter bounds for fitting routine 
            * n_jobs: int 
                * Number of cores to use 
            * blocks_per_core: int
                * Number of B-blocks per core for multithreading
            * show_progress: bool 
                * Display tqdm progress bar. 
            * save_pickle_every_iteration: bool
                * Saves updating parameters by fitting to pickle file.
            * save_pickle_filepath: Path 
                * Directory to dump pickled parameters
            * update_param_yaml: bool 
                * Update the parameter file with the latest stored pickle 
                * from a previous run. Uses these params as well.

        Variable Notation: 
            * `parameters` is a dict[str, float] of parameter names and values 
            * `parameter_vec` is a np.ndarray built from the `parameters` dict
        """

        self._save_pickle = save_pickle_every_iteration 
        self._pickle_dirs = save_pickle_filepath
        self._pickle_dirs.mkdir(exist_ok=True, parents=True)
        if update_param_yaml: 
            _update_yaml_with_pickle(self)
            ansatz_parameters = load_params(PARAMETER_FILE)
        elif not update_param_yaml and default_parameters: 
            ansatz_parameters = load_params(PARAMETER_FILE)
        elif ansatz_parameters is None: 
            ansatz_parameters = load_params(PARAMETER_FILE)
        self.verify_parameters_complete(ansatz_parameters)

        if lower_bound is None: 
            lower_bound = lower_bounds
        if upper_bound is None: 
            upper_bound = upper_bounds 
         
        # Some "parameters" are constants, like hbar, mu_B, mu_N.
        # Since these are part of the parameter vector, the LSQ routine 
        # would run the EDMR simulation 4 more times than needed every iteration. 
        #
        # Hence we remove constants from ansatz_parameters and reinsert them 
        # when calculating the EDMR simulated spectra.  
       
        assert lower_bound.keys() == upper_bound.keys(), \
            "lower_bound.keys() != upper_bound.keys()"
        assert upper_bound.keys() == ansatz_parameters.keys(), \
            "upper_bound.keys() != ansatz_parameters.keys()" 

        keys_with_common_values = [
            k for k in ansatz_parameters.keys() 
            if (
                np.isclose(ansatz_parameters[k],lower_bound[k]) and 
                np.isclose(lower_bound[k], upper_bound[k])
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

        self.raw_B_data = np.asarray(raw_B_array) 
        self.raw_I_data = np.asarray(raw_I_array) 
        if len(raw_B_array) > n_points_interp: 
            self.B_discretized = np.linspace(
                raw_B_array[0], 
                raw_B_array[-1], 
                n_points_interp
            )
        else: self.B_discretized = self.raw_B_data

        self.alpha = alpha
        self._n_jobs          = n_jobs 
        self._blocks_per_core = blocks_per_core 
        self._n_points        = n_points_interp
        self._show_progress = show_progress 

        if save_pickle_every_iteration: 
            logger.info(" * Pickle Saving Active *") 
            logger.info(
                " This will consume ~130us everytime" 
                " least-squares residuals are calculated."
            )

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
            modulate=True, 
            n_jobs=self._n_jobs, 
            blocks_per_core=self._blocks_per_core, 
            show_progress=self._show_progress
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

    def _parameter_vec_to_full_dict(self, parameter_vec: np.ndarray): 
        """ 
        Converts the ndarray parameter vector returned by least_squares 
        back into a parameters dict[str, float] with constants included. 

        """
        parameters = _convert_theta_to_dict(parameter_vec)
        return self._reinsert_constants_to_parameters(parameters) 

    def _callback_save_pickle(self, lsq_result: OptimizeResult):
        """
        Saves the most recent parameter + constant dictionary in a pickle 
        after every least_squares iteration. 

        * Only if self._save_pickle == True *

        """

        parameter_vec = lsq_result.x
        parameters = self._parameter_vec_to_full_dict(parameter_vec) 
        if self._save_pickle:
            with open(self._pickle_dirs / "lsq_param_pkl.pickle", "wb") as f: 
                pickle.dump(parameters, f)

    def fit(
        self, 
        ftol: float = 1e-14, 
        xtol: float = 3e-16, 
        gtol: float = 1e-14 
    ): 
        print("\nInitializing Least Squares Fitting Routine")
        print("------------------------------------------\n") 
        print("Ansatz Parameters: ") 
        print(_convert_theta_to_dict(self.parameter_vec)) 
        print(f"\n# points  = {self._n_points}" )
        print(f"# threads = {self._n_jobs}") 

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

        return _convert_theta_to_dict(self.result.x) 

    def fitted_EDMR_spectra(
        self, 
        B_array: np.ndarray | None, 
        parameters: dict[str, float], 
    ): 
        if B_array is None: 
            B_array = self.raw_B_data 
        B_array = np.asarray(B_array, dtype=float)

        I_array = self.edmr_functional(
            B_array= B_array, 
            parameters=parameters, 
            modulate=True, 
            n_jobs=self._n_jobs, 
            blocks_per_core=self._blocks_per_core, 
            show_progress=self._show_progress
        ) 
        return I_array

    def plot_fitted_EDMR_spectra(
        self, 
        B_array: np.ndarray | None, 
        parameters: dict[str, float], 
        n_points_interp: int, 
        save: bool, 
        save_directory: Path 
    ): 
        import matplotlib.pyplot as plt 

        if B_array is None:
            B = self.raw_B_data
        else:
            B = B_array
        if n_points_interp and n_points_interp < len(B): 
            B = np.linspace(min(B), max(B), n_points_interp)
        dI = self.fitted_EDMR_spectra(
                B_array=B, 
                parameters=parameters
        )
        dI = interp1d(B, dI, kind="cubic", assume_sorted=True)(self.raw_B_data)
        
        print(f" Rendering Fitted EDMR with {n_points_interp} steps.") 
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.raw_B_data, dI, c='r', ls=':', lw=1, label="Simulated") 
        ax.plot(self.raw_B_data, self.raw_I_data, c='k', lw=2, label="Raw") 

        ax.axvline(0, ls=":", color="grey", alpha=0.6)
        ax.axhline(0, ls=":", color="grey", alpha=0.6)
        ax.set(
            xlabel="B [G]",
            ylabel=r"$\partial I/\partial B$ [nA]",
            title=f"EDMRLSQ Best-Fit",
        )
        ax.legend()

        if save: 
            save_directory.mkdir(parents=True, exist_ok=True) 
            filename = save_directory / "EDMR_LeastSquares.png" 
            fig.savefig(filename, dpi=300) 
            logger.info(" fig::%s \n", filename)
        return fig 

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
    n_points_interp: int            = 100,
    n_jobs: int                     = 2, 
    blocks_per_core: int            = 8, 
    B_range: tuple[float, float]    = (-np.inf, np.inf),  
    show_progress: bool             = True, 
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
            *   if B_range exceeds range of `raw_B_data`, 
            *   `raw_B_data` is used. 
            *   if B_range is less than range of `raw_B_data`, 
            *   raw_B_data is truncated to match `B_range`
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
            n_jobs=n_jobs, 
            blocks_per_core=blocks_per_core, 
            show_progress=show_progress, 
            default_parameters=True, 
            ansatz_parameters=None, 
            save_pickle_every_iteration=True, 
            save_pickle_filepath=Path(os.getcwd()).parent / "utils", 
            lower_bound=lower_bounds, 
            upper_bound=upper_bounds,
            update_param_yaml=False
    )



