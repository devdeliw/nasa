import numpy as np 
from scipy.optimize import least_squares 

class EDMRLSQ:
    """
    Fit EDMR spectra data via least-squares optimization.

    Args:
        * B (np.ndarray): 
            * Magnetic field sweep data.
        * I (np.ndarray): 
            * Experimental current data.
        * p0 (np.ndarray): 
            * Initial parameter guess (19-vector).
        * lower (np.ndarray): 
            * Lower bounds for parameters.
        * upper (np.ndarray): 
            * Upper bounds for parameters.
        *edmr (callable): 
            * Function edmr_spectra(B_array, pvec, modulate=True).
        * result (OptimizeResult):
            *  The result of least_squares after fitting.
    """

    def __init__(self, B_array, I_array, p0, lower, upper, edmr_func):
        self.B = np.asarray(B_array, dtype=float)
        self.I = np.asarray(I_array, dtype=float)
        self.p0 = np.asarray(p0, dtype=float)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.edmr = edmr_func
        self.result = None

    def _residuals(self, pvec):
        """
        Compute residuals for least_squares: model prediction minus data.

        """
        return self.edmr(self.B, pvec, modulate=True) - self.I

    def fit(self, ftol=1e-9, xtol=1e-9, verbose=False):
        """
        Run the least-squares fit.

        Args:
            * ftol (float):
                * Tolerance for change in cost function.
            * xtol (float): 
                * Tolerance for change in parameters.
            * verbose (bool): 
                * If True, prints solver progress.

        Returns:
            * np.ndarray: Best-fit parameter vector.
        """
        self.result = least_squares(
            fun=self._residuals,
            x0=self.p0,
            bounds=(self.lower, self.upper),
            ftol=ftol,
            xtol=xtol,
            verbose=2 if verbose else 0,
            jac='2-point'
        )
        return self.result.x

    @property
    def fitted_params(self):
        """
        Returns the fitted parameters after running fit().

        """
        if self.result is None:
            raise RuntimeError("Call fit() before accessing fitted_params.")
        return self.result.x

    def predict(self, B_array=None, pvec=None):
        """
        Generate model predictions for a given B_array and parameter vector.

        Args:
            * B_array (array-like, optional): 
                * Field values to predict at.
                * Defaults to the original B-array.
            * pvec (array-like, optional): 
                * Parameter vector to use.
                * Defaults to the fitted parameters.

        Returns:
            * np.ndarray: Predicted currents.
        """
        B_eval = np.asarray(B_array, dtype=float) if B_array is not None else self.B
        p_eval = np.asarray(pvec, dtype=float) if pvec is not None else self.fitted_params
        return self.edmr(B_eval, p_eval, modulate=True)


# Example usage (assumes edmr_spectra is defined in scope):
# fitter = EDMRFitter(B_data, I_data, p0, lower, upper, edmr_spectra)
# best_params = fitter.fit(verbose=True)
# print("Best-fit parameters:", best_params)