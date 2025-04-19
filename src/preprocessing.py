import os 
import json
import pickle
import logging
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from pathlib import Path 
from scipy.signal import savgol_filter 
from functools import cached_property

# configure logging and root directory 
logging.basicConfig(level=logging.INFO)
ROOT = Path(__file__).resolve().parents[1] 

class Process(): 
    def __init__(
        self, 
        filename: str, 
        window_length: int = 7, 
        polyorder: int = 3
    ): 
        self.filename       = filename 
        self.window_length  = window_length 
        self.polyorder      = polyorder 

    @cached_property
    def raw_spectra(self) -> pd.DataFrame: 
        pkl = ROOT / "data" / "raw" / f"{self.filename}.pkl" 
        try: 
            return pd.read_pickle(pkl)
        except FileNotFoundError: 
            logging.error(f"{pkl} not found.")
            raise

    @cached_property
    def smoothed_spectra(self, save=True) -> pd.DataFrame:
        # Apply Savitzky-Golay filter to the spin current 
        # window_length must be odd and greater than polyorder
        
        df = self.raw_spectra.copy() 
        if self.window_length %2 == 0: 
            raise ValueError("window_length must be odd") 
        if self.window_length <= self.polyorder: 
            raise ValueError("window_length must be > polyorder") 

        col = df.columns[1] 
        df[f"{col} smoothed"] = savgol_filter(
            df[col], 
            self.window_length, 
            self.polyorder
        )

        if save: 
            pkl = ROOT / "data" / "processed" / f"{self.filename}_processed.pkl"
            with open(pkl, "wb") as f: 
                pickle.dump(df, f)

        return df

    # Helper functions to determine best Savitzky-Golay 
    # window_length/polyorder values 
    @staticmethod
    def _measures(raw: np.ndarray, smooth: np.ndarray, k: int = 12): 
        noise_removed = np.std(raw) - np.std(smooth)
        raw_peaks       = np.sort(raw)[-k:] 
        smooth_peaks    = np.sort(smooth)[-k:] 
        peak_penalty    = np.mean(np.abs(raw_peaks - smooth_peaks)) 
        peak_loss_frac  = peak_penalty / np.mean(raw_peaks) 
        return noise_removed, peak_loss_frac

    def find_best_smoothing(
        self, 
        window_percents=(0.10, 0.05, 0.02, 0.01), 
        polyorders=(2, 3, 4, 5), 
        max_peak_loss: float = 0.05,
        k: int = 30, 
    ) -> dict:

        # initialize 
        y = self.raw_spectra.iloc[:, 1].values
        n = len(y) 
        best = {
            "window_length" : None, 
            "polyorder"     : None, 
            "noise_removed" : -np.inf, 
            "peak_loss_frac": None, 
        }

        # generate odd windows 
        windows = [] 
        for p in window_percents: 
            wl = int(p * n) 
            wl += (wl + 1) % 2 # odd 
            if wl > max(polyorders): 
                windows.append(wl) 
        # grid search 
        for wl in windows: 
            for po in polyorders: 
                if wl <= po:
                    continue 
                sm = savgol_filter(y, wl, po) 
                noise_removed, peak_loss = self._measures(y, sm, k) 
                if peak_loss > max_peak_loss: 
                    continue 
                if noise_removed > best["noise_removed"]: 
                    best.update(
                        window_length=wl, 
                        polyorder=po, 
                        noise_removed=noise_removed, 
                        peak_loss_frac=peak_loss, 
               )

        logging.info(json.dumps(best, indent=2))
        return best 

    def apply_best_smoothing(self, **grid_kwargs) -> pd.DataFrame: 
        """ 
        Finds best Savitzky-Golay parameters via `find_best_smoothing`, 
        and returns optimal smoothed DataFrame object. 
        """ 

        self.best = self.find_best_smoothing(**grid_kwargs) 
        self.window_length  = self.best["window_length"] 
        self.polyoder       = self.best["polyorder"] 

        # clear cache, recompute 
        if "smoothed_spectra" in self.__dict__: 
            del self.__dict__["smoothed_spectra"] 
        return self.smoothed_spectra 

    def plot(self, show_raw: bool = True, save: bool = False, show: bool = True): 
        fig, ax = plt.subplots(1, 1, figsize=(10, 8)) 

        raw_spectra     = self.raw_spectra 
        smooth_spectra  = self.smoothed_spectra

        x_col        = raw_spectra.columns[0] 
        y_raw_col    = raw_spectra.columns[1] 
        y_smooth_col = next(
            (col for col in smooth_spectra.columns if "smoothed" in col), 
            smooth_spectra.columns[2]
        )

        ax.plot(smooth_spectra[x_col], smooth_spectra[y_smooth_col], c='r', label="smooth") 

        if show_raw: 
            ax.scatter(raw_spectra[x_col], raw_spectra[y_raw_col],
                        c='k', marker='+', s=20, label="raw")

        ax.set_xlabel(x_col, fontsize=14) 
        ax.set_ylabel(y_raw_col, fontsize=14)
        ax.set_title(self.filename, fontsize=14)
        ax.legend() 
        ax.grid(True) 

        # display SG parameters 
        ax.text(0.5, -0.12, self.best, transform=ax.transAxes, fontsize=8, ha="center")

        if save: 
            out_dir = "./media/" 
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(f"{out_dir}{self.filename}_processed.png", dpi=300) 
        if show: 
            plt.show() 
        return fig  


if __name__ == "__main__": 
    filename = "[NZFMR]_DER_1.5G_2.5V" 
    process  = Process(filename)
    process.apply_best_smoothing( 
        window_percents=(0.2, 0.1, 0.05, 0.01), 
        polyorders=(2, 3, 4, 5), 
        max_peak_loss=0.01, 
        k=30
    ) 

    process.plot(show_raw=True, save=True)
    







    



            

 



    

