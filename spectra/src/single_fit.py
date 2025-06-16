# triple_deriv_lorentz_ls.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------
def triple_dlorentz(B, S0, A1, G1, A2, G2, A3, G3):
    """Sum of three first-derivative Lorentzians centred at B = 0."""
    term1 = -2.0 * B * A1 / (B**2 + G1**2) ** 2
    term2 = -2.0 * B * A2 / (B**2 + G2**2) ** 2
    term3 = -2.0 * B * A3 / (B**2 + G3**2) ** 2
    return S0 + term1 + term2 + term3


# ---------------------------------------------------------------------
# initial guesses that rarely fail
# ---------------------------------------------------------------------
def _p0(B, I):
    span = I.max() - I.min()

    S0 = I.mean()
    # rough amplitudes: split span into three decreasing pieces
    A1, A2, A3 = 0.5 * span, 0.25 * span, 0.15 * span

    # broad, medium, narrow widths in Gauss
    full_bw = B.ptp()
    G1, G2, G3 = 0.80 * full_bw, 0.2 * full_bw, 0.05 * full_bw
    return S0, A1, G1, A2, G2, A3, G3


# ---------------------------------------------------------------------
# fitting routine
# ---------------------------------------------------------------------
def fit_triple_dlorentz(B, I,
                        plot=True,
                        out_png="compound_fit.png",
                        param_txt="compound_params.txt"):
    """
    Least-squares fit of a 3-component derivative Lorentzian model.

    Parameters
    ----------
    B : 1-D array_like
        Magnetic-field axis (Gauss).
    I : 1-D array_like
        Current / signal (same length as B).
    plot : bool
        If True, save an overlay PNG.
    out_png : str
        File name for the plot.
    param_txt : str
        File name for best-fit parameters.

    Returns
    -------
    popt : ndarray
        Best-fit parameters (S0, A1, G1, A2, G2, A3, G3).
    pcov : 2-D ndarray
        Covariance matrix from `curve_fit`.
    """
    B = np.asarray(B, dtype=float)
    I = np.asarray(I, dtype=float)

    # positivity bounds: widths > 0
    lower = [-np.inf, -np.inf, 1e-3, -np.inf, 1e-3, -np.inf, 1e-3]
    upper = [ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf]

    popt, pcov = curve_fit(
        triple_dlorentz,
        B,
        I,
        p0=_p0(B, I),
        bounds=(lower, upper),
        method='trf', 
        loss='soft_l1', 
        f_scale=0.1, 
        maxfev=150000,
    )

    # save parameters
    header = (
        "S0  A1    G1    A2    G2    A3    G3\n"
        "(arb) (arb) (G) (arb) (G) (arb) (G)"
    )
    np.savetxt(param_txt, popt[None, :], header=header, fmt="%.6g")
    print(f"parameters → {param_txt}")

    # optional visual check
    if plot:
        B_plot = np.linspace(B.min(), B.max(), 1500)
        fit_tot = triple_dlorentz(B_plot, *popt)

        # individual components
        S0, A1, G1, A2, G2, A3, G3 = popt
        S1 = triple_dlorentz(B_plot, S0, A1, G1, 0, 1, 0, 1) - S0
        S2 = triple_dlorentz(B_plot, S0, 0, 1, A2, G2, 0, 1) - S0
        S3 = triple_dlorentz(B_plot, S0, 0, 1, 0, 1, A3, G3) - S0

        plt.figure(figsize=(6, 4))
        plt.plot(B, I, ".", ms=3, label="data")
        plt.plot(B_plot, S1 + S0, "C0-", lw=1, label="S1")
        plt.plot(B_plot, S2 + S0, "C1-", lw=1, label="S2")
        plt.plot(B_plot, S3 + S0, "C2-", lw=1, label="S3")
        plt.plot(B_plot, fit_tot, "k", lw=2, label="compound")
        plt.xlabel("B (G)")
        plt.ylabel("Current (arb. units)")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"plot → {out_png}")

    return popt, pcov

if __name__ == "__main__":
    import pickle
    file = "./data/raw/[EDMR]_2G_3V_200MHz.pkl" 

    try: 
        with open(file, "rb") as f: 
            spectra = pickle.load(f)

        spectra = spectra[spectra["B (Gauss)"].abs() <= 50]
        B = np.array(spectra["B (Gauss)"])
        I = np.array(spectra["I (nA)"])

        popt, pcov = fit_triple_dlorentz(B, I)
    except FileNotFoundError: 
        print("file not found")


# ------------- sample usage -----------------
# import pandas as pd
# df = pd.read_csv("your_spectrum.csv")
# mask = df["B (Gauss)"].abs() <= 50
# B = df.loc[mask, "B (Gauss)"]
# I = df.loc[mask, "I (nA)"]


# popt, pcov = fit_deriv_lorentz(B, I)
# print("best-fit parameters  [B0, A, G, S0] =\n", popt)

