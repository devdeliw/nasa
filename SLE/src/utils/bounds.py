import numpy as np

# lower bounds
lower = np.array([
    1e4,           # A ≥ 0 (scale factor)
    -1e4,          # I0 offset (can be negative, similar magnitude to A)
    0.0,           # J ≥ 0  (exchange, ≲ 1µeV)
    0.0,         # Aa1 ∈ [−1µeV, +1µeV]
    0.0,         # Ab1
    0.0,         # Aa2
    0.0,         # Ab2
    0.0,           # D1 ≥ 0 (ZFS, ≲100 neV)
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
    1e-6,          # Aa1
    1e-6,          # Ab1
    1e-6,          # Aa2
    1e-6,          # Ab2
    1e-7,          # D1
    1e-9,          # D2
    10.0,          # B0 dummy
    2.004,         # g_e
    -1.09,         # g_n1
    1.41,          # g_n2
    2.005e8,           # nu
    1e7,           # omega1
    1e7,           # k_S
    1e8,           # k_D
    1e5,           # p
    5,             # B_mod
], dtype=float)

x_scale = np.array([
    9.90e5,         # A       = max(5e5, 1e6−1e4=9.9e5)
    2.00e4,         # I0      = max(0, 1e4−(−1e4)=2e4)
    1.00e-6,        # J       = max(1e-9, 1e-6)
    1.00e-6,        # Aa1     = max(5.2e-8, 1e-6)
    1.00e-6,        # Ab1     = max(1.9e-7, 1e-6)
    1.00e-6,        # Aa2     = max(3.7e-8, 1e-6)
    1.00e-6,        # Ab2     = max(1e-7, 1e-6)
    1.00e-7,        # D1      = max(3.4e-9, 1e-7)
    1.00e-9,        # D2      = max(1e-12, 1e-9)
    2.00e1,         # B0      = max(0, 10−(−10)=20)
    3.00e-3,        # g_e     = max(2.0023, 0.003)
    3.00e-2,        # g_n1    = max(1.1101, 0.03)
    2.00e-2,        # g_n2    = max(1.4047, 0.02)
    2.00098e8,      # nu      = max(2.00098e8, 1e6)
    1.00e7,         # omega1  = max(2e5, 1e7)
    1.00e7,         # k_S     = max(1e5, 1e7)
    1.00e8,         # k_D     = max(1e6, 1e8)
    1.00e5,         # p       = max(1e3, 1e5)
    3.00e0          # B_mod   = max(4, 5−2=3)
], dtype=float)


