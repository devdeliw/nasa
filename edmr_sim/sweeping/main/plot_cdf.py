import numpy as np 
import matplotlib.pyplot as plt 

from pathlib import Path 
from functools import cached_property
from matplotlib.ticker import MaxNLocator
from .parse_cdf import NET_CDF_DIR, RenderNetCDF 
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['axes.linewidth'] = 0.5

class Load3DSweep(RenderNetCDF): 
    """ 
    Initializes np.ndarray of ParameterSlices 
    for every combination of parameters in the 
    netCDF xrarray DataArray. 

    This will be used for making 3D plots to visualize 
    how EDMR spectra changes as parameter values change. 

    """

    def __init__(
        self, 
        cdf_filename:  str, 
        cdf_directory: Path = NET_CDF_DIR, 
    ): 
        super().__init__(       # Populates 
            cdf_filename,       # self.datarray 
            cdf_directory       # self.parameters 
        )                       # self.param_dict
        
        self.B = self.load_B 

    @cached_property
    def _all_parameter_combinations(self) -> np.ndarray: 
        """
        Returns numpy array containing every combination 
        of every parameter value(s) in self.param_dict. 

        """
        params = [p for p in self.param_dict.keys() if p != "B"]
        arrays = [np.asarray(self.param_dict[p]) for p in params]
        grids  = np.meshgrid(*arrays, indexing='ij')
        flat   = [g.ravel() for g in grids]
        return np.stack(flat, axis=-1)

    def _single_parameter_slices(self, param_name: str): 
        """ 
        Returns numpy array containing ParameterSlices 
        for the evolution of a *single parameteter*, 
        keeping all others constant. 

        The non-param_name parameters will be held at
        their starting value. 

        """
        names      = self.parameters
        param_dict = self.param_dict

        base_params = { 
            name: np.asarray(param_dict[name])[0] 
            for name in names 
            if name != param_name 
        }

        sweep_values = np.asarray(param_dict[param_name])
        slices = [ 
            self.slice(
                parameter_values={**base_params, param_name: val}
            )
            for val in sweep_values 
        ]

        return sweep_values, np.asarray(slices), base_params
 
    @cached_property
    def _all_parameter_slices(self) -> np.ndarray:
        """ 
        Returns numpy array containing ParameterSlices 
        for every combination of every parameter value(s) 
        in self.param_dict. 

        """
        combos   = self._all_parameter_combinations   
        names    = list(self.parameters)              

        slices = [
            self.slice(parameter_values=dict(zip(names, combo))) 
            for combo in combos
        ]
        return np.asarray(slices)


class PlotSingle3DSweep(Load3DSweep):

    def __init__(
        self, 
        param_name:    str,
        cdf_filename:  str, 
        cdf_directory: Path = NET_CDF_DIR,
        out_directory: Path = Path(__file__).parent.parent / "media" 
    ): 
        super().__init__(       
            cdf_filename,        
            cdf_directory, 
        ) 
        self.param_name    = param_name

        out_directory.mkdir(exist_ok=True, parents=True)
        self.out_directory = out_directory 

    def param_evolution(self):

        param_values, param_slices, base_params = self._single_parameter_slices(
            self.param_name
        )
         
        I  = np.stack([sl.I  for sl in param_slices], axis=0)
        dI = np.stack([sl.dI for sl in param_slices], axis=0)
        return param_values, I, dI, base_params 

    def _append_negative_side(self, B, I, dI): 
        """ 
        If the calculated EDMR spectra is only the positive side 
        (which it should be, since its antisymmetric and reduces 
        computation time), this just appends the negative side 
        for plotting the full EDMR spectra 

        """

        has_zero = np.isclose(B[0], 0.0)

        def mirror(arr, antisym=False): 
            neg = arr[:, 1:][:, ::-1] if has_zero else arr[:, ::-1]
            if antisym: 
                neg = -neg 
            return np.concatenate([neg, arr], axis=1)

        B_negative  = -B[1:][::-1] if has_zero else -B[::-1]
        B_complete  = np.concatenate([B_negative, B]) 
        I_complete  = mirror(I, antisym=False) 
        dI_complete = mirror(dI, antisym=True) 

        return B_complete, I_complete, dI_complete

    def current_plot(self, current_type: str = "lockin", plot_full: bool = True, 
                     elev: float = 35, azim: float = -75, figsize=(10, 10)
    ): 
        """ 
        Args: 
            * current_type (str): 
            if "lockin", plots derivative current EDMR spectra 
            if "integrated", plots integrated current EDMR spectra 

            * elev, azim: elevation and azimuthal angle for viewing the 3dplot. 

        """

        assert current_type in ("lockin", "integrated"), (
            f"current_type: {current_type}" 
            f"must either be 'lockin' or 'integrated'" 
        )

        B = self.B 
        param_values, I, dI, base_params = self.param_evolution()

        if B[0] >= 0.0 and plot_full: 
            B, I, dI = self._append_negative_side(B, I, dI) 
        assert B.shape[0] == I.shape[1] == dI.shape[1]

        if current_type == "lockin": 
            current = dI 
            zlabel = r"$\partial I / \partial B$" 
        else: 
            current = I 
            zlabel = "I"

        X = np.broadcast_to(B[np.newaxis, :], (param_values.size, B.size))
        Y = np.broadcast_to(param_values[:, np.newaxis], (param_values.size, B.size))
        Z = current

        lines = np.stack((X, Y, Z), axis=2)

        lc = Line3DCollection(
            lines, 
            linewidths=30/len(param_values), 
            cmap="cool", 
            array=param_values 
        )

        figure = plt.figure(figsize=figsize) 
        axis   = figure.add_subplot(projection='3d')

        axis.add_collection(lc) 
        axis.set_xlim(B.min()-10, B.max()+10)
        axis.set_ylim(param_values.min(), param_values.max()) 
        axis.set_zlim(Z.min(), Z.max())


        axis.grid(False) 
        axis.xaxis.pane.fill = False 
        axis.yaxis.pane.fill = False 
        axis.zaxis.pane.fill = False 
        axis.xaxis.pane.set_edgecolor((1,1,1,0))
        axis.yaxis.pane.set_edgecolor((1,1,1,0))
        axis.zaxis.pane.set_edgecolor((1,1,1,0))

        axis.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
        axis.zaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))

        axis.set_xlabel('B [G]', fontsize=10) 
        axis.set_ylabel(self.param_name, fontsize=10) 
        axis.set_zlabel(zlabel, fontsize=10)
        axis.tick_params(labelsize=8)

        axis.view_init(elev=elev, azim=azim)

        # adding base param textbox 
        keys = list(base_params.keys())
        if keys: 
            maxlen = max(len(k) for k in keys)
            lines = []
            for k in keys:
                lines.append(f"{k.rjust(maxlen)} : {base_params[k]:3g}")

            textstr = "\n".join(lines)

            axis.text2D(
                0.90, 0.90,     
                textstr,
                transform=axis.transAxes,
                fontsize=10,
                fontfamily="monospace",
                verticalalignment="center",
                horizontalalignment="right",
                bbox=dict(facecolor="none", edgecolor="none")
            )

        plt.tight_layout() 
        plt.show()
        out_file = self.out_directory / f"{self.param_name}_{current_type}_surf.png"
        plt.savefig(out_file, dpi=500, bbox_inches='tight', pad_inches=0.5) 
        print(f"[INFO]:: 3D Surface Plot Saved:: {out_file}")

        return figure, axis 




if __name__ == "__main__": 
    single_plot = PlotSingle3DSweep(
        param_name="nu", 
        cdf_filename="sweep.nc", 
    )

    single_plot.current_plot(current_type="lockin", elev=25, azim=-75, figsize=(10, 10)) 








