import numpy as np

# lower bounds
lower = np.array([
    1e4,           # A ≥ 0 (scale factor)
    -1e4,          # I0 offset (can be negative, similar magnitude to A)
    0.0,           # J ≥ 0  (exchange, ≲ 1µeV)
    5.85e-8,         # Aa1 ∈ [−1µeV, +1µeV]
    2.04e-7,         # Ab1
    4.5e-8,         # Aa2
    1e-7,         # Ab2
    1.5e-8,           # D1 ≥ 0 (ZFS, ≲100 neV)
    0.0,           # D2 ≥ 0 (ZFS, ≲1 neV)
    -10.0,         # dummy
    2.001,         # g_e ∈ [2.001, 2.004]
    -1.12,         # g_n1 ∈ [−1.5, −0.5]
    1.39,          # g_n2 ∈ [1.2, 1.6]
    1.995e8,           # nu ∈ [190 MHz, 210 MHz]
    1e3,           # omega1 ∈ [1 kHz, 10 MHz]
    1e3,           # k_S ∈ [1 kHz, 10 MHz]
    1e4,           # k_D ∈ [10 kHz, 100 MHz]
    1e2,           # p ∈ [10, 1e5]
    2,             # B_mod ∈ [0.1 G, 10 G]
], dtype=float)

# upper bounds
upper = np.array([
    1e6,           # A 
    1e4,           # I0
    1e-6,          # J
    5.93e-8,        # Aa1
    2.065e-7,          # Ab1
    5e-8,          # Aa2
    1.01e-7,          # Ab2
    1.7e-8,          # D1
    1e-9,          # D2
    10.0,          # B0 dummy
    2.004,         # g_e
    -1.09,         # g_n1
    1.41,          # g_n2
    2.005e8,       # nu
    1e7,           # omega1
    1e7,           # k_S
    1e8,           # k_D
    1e5,           # p
    5,             # B_mod
], dtype=float)

x_scale = np.array([
    1e-8,   # J       (≈3.3e-9 → 0.33)
    1e-8,   # Aa1     (≈5.2e-8 → 0.52)
    2e-7,   # Ab1     (≈1.9e-7 → 0.95)
    1e-8,   # Aa2     (≈3.7e-8 → 0.37)
    1e-7,   # Ab2     (≈1.0e-7 → 1.0)
    1e-8,   # D1      (≈1.6e-8 → 1.6)
    1e-9,   # D2      (≈1.8e-10 → 0.18)
    1e1,    # B0      (≈0 → 0)
    1.0,    # g_e     (≈2.002 → 2.0)
    1.0,    # g_n1    (≈-1.09 → -1.1)
    1.0,    # g_n2    (≈1.405 → 1.4)
    1e8,    # nu      (≈2.00e8 → 2.0)
    1e6,    # omega1  (≈4.89e5 → 0.49)
    1e5,    # k_S     (≈9.87e4 → 0.99)
    1e5,    # k_D     (≈4.78e4 → 0.48)
    1e3,    # p       (≈8.19e2 → 0.82)
    1.0     # B_mod   (≈3.99 → 4.0)
], dtype=float)