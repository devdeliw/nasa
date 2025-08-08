lower_bounds = {
    # proportion
    "A":   1e-4,
    "I0":  0.0,

    # exchange
    "J":   1e-9,

    # hyperfine
    "Aa1": 1e-9,
    "Ab1": 1e-9,
    "Aa2": 1e-9,
    "Ab2": 1e-9,

    # zero-field splitting
    "D1":  1e-9,
    "D2":  1e-12,

    # Zeeman (B0 is treated like a fit parameter here)
    "B0":  -50.0,
    "g_e": 2.0,
    "g_n1": -1.2,
    "g_n2": 1.2,

    # microwave
    "nu":     1.9e8,
    "omega1": 1e3,

    # sle rates
    "k_S": 1e2,
    "k_D": 1e3,
    "p":   1e1,

    # lock‑in
    "B_mod": 0.1,

    # constants (fixed)
    "h":     4.135667662e-15,
    "hbar":  6.58211e-16,
    "mu_B":  5.7883818e-09,
    "mu_N":  3.1524512e-12,
}

upper_bounds = {
    # proportion
    "A":   1e7,
    "I0":  2e4,

    # exchange
    "J":   5e-7,

    # hyperfine
    "Aa1": 5e-7,
    "Ab1": 5e-7,
    "Aa2": 5e-7,
    "Ab2": 5e-7,

    # zero‑field splitting
    "D1":  1e-7,
    "D2":  1e-10,

    # Zeeman
    "B0":   50.0,
    "g_e":  2.01,
    "g_n1": -1.0,
    "g_n2":  1.6,

    # microwave
    "nu":     2.1e8,
    "omega1": 1e7,

    # sle rates
    "k_S": 1e8,
    "k_D": 1e9,
    "p":   1e6,

    # lock‑in
    "B_mod": 10.0,

    # constants (fixed)
    "h":     4.135667662e-15,
    "hbar":  6.58211e-16,
    "mu_B":  5.7883818e-09,
    "mu_N":  3.1524512e-12,
}

