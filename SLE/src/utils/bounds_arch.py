import numpy as np, time
from pathlib import Path
from lsq import _load_fitter, _update_param_yaml
from run_solver import load_params, make_pvec

def main(n_search: int = 30, sleep_every=20, sleep_time=120): 
    """
    Runs coordinate search to find near-optimal parameters 
    for the least squares initial guess. 

    Args: 
        * n_search: int
            * # steps between the lower and upper bound =
            * of a parameter to measure cost.
    """

    fitter = _load_fitter(
        data_path       = Path.home()/"nasa/spectra/src/data/raw/[EDMR]_2G_3V_200MHz.pkl", 
        n_points_per    = 50, 
        default_params  = True, 
        custom_params   = False, 
        n_jobs          = 5, 
        B_range         = (-50, 50),
        show_progress   = False,
        verbose         = False
    )

    p_best = fitter._p0_nl.copy() 
    lower_nl = fitter._lower_nl 
    upper_nl = fitter._upper_nl 

    sweep_map = { 
        "Aa1"   : 1, 
        "Ab1"   : 2, 
        "Aa2"   : 3, 
        "Ab2"   : 4, 
        "D1"    : 5, 
        "J"     : 0, 
        "k_S"   : 13, 
        "k_D"   : 14, 
        "p"     : 15 
    }

    num_iterations = 0
    def cost(p_nl): 
        nonlocal num_iterations 
        num_iterations += 1
        r = fitter._residuals(p_nl, save_pickle=False)
        return np.sum(r*r)

    print("\nSTARTING COORDINATE SEARCH:")
    print("===========================\n")

    for name, idx in sweep_map.items():     
        print(name)
        print('‾'*len(name))

        lo, hi = lower_nl[idx], upper_nl[idx]
        grid = np.linspace(lo, hi, n_search)

        best_val = p_best[idx]
        best_cost = cost(p_best)
        print(f"Present Value: {best_val:3e}")
        print(f"Present Cost : {best_cost:3e}")

        w_iter, w_val, w_cost, w_diff, w_better = 6, 12, 12, 15, 7
        header = f"{'Iter':^{w_iter}} | {'Value':^{w_val}} | {'Cost':^{w_cost}} | {'δCost':^{w_diff}} | {'Better':^{w_better}}"
        print(header)
        print("‾" * len(header))

        for num, trial in enumerate(grid): 

            if sleep_every and sleep_time and num_iterations and num_iterations % sleep_every == 0: 
                time.sleep(sleep_time)

            p_test = p_best.copy() 
            p_test[idx] = trial 

            c = cost(p_test)
            difference = c - best_cost 
            difference_string = f"+{difference:<2e}" if difference > 0 else f"{difference:<2e}"
            better_string = "✔" if difference < 0 else "✘"

            print(f"{num:^{w_iter}d} | {trial:^{w_val}.3e} | {c:^{w_cost}.3e} | {difference_string:^{w_diff}} | {better_string:^{w_better}} ")

            if difference < 0: 
                best_cost = c 
                best_val = trial 

        p_best[idx] = best_val 
        print(f" {name}: best={best_val:.3e}, cost={best_cost:3e}\n")
        update_yaml(idx, best_val)

    print(" Optimized parameter vector: ")
    print(" ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ")
    for name, idx in sorted(sweep_map.items(), key=lambda x: x[1]):
        print(f"    {name:4s} = {p_best[idx]:3e}")

def update_yaml(idx, val): 
    _, _, hamiltonian_params, *linblad_terms = load_params()
    hamiltonian_params[idx] = val   
    full_pvector = make_pvec(hamiltonian_params, *linblad_terms)
    full_pvector[idx] = val

    _update_param_yaml(
        from_pkl=False, 
        new_vals=np.array(full_pvector)
    )
    

if __name__ == "__main__": 
    main(n_search=120, sleep_time=120, sleep_every=20)