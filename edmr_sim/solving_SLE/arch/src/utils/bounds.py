import numpy as np

lower = np.array([
    1e-4,       #  0: A        
    0,          #  1: I0      
    1e-9,       #  2: J      
    1e-9,       #  3: Aa1      
    1e-9,       #  4: Ab1      
    1e-9,       #  5: Aa2
    1e-9,       #  6: Ab2
    1e-9,       #  7: D1      
    1e-12,      #  8: D2      
    -50.0,      #  9: dummy B0
    2.0,        # 10: g_e      
    -1.2,       # 11: g_n1    
    1.2,        # 12: g_n2     
    1.9e8,      # 13: nu       
    1e3,        # 14: omega1   
    1e2,        # 15: k_S      
    1e3,        # 16: k_D     
    1e1,        # 17: p       
    0.1,        # 18: B_mod   
], dtype=float)

upper = np.array([
    1e7,        #  0: A
    2e4,        #  1: I0
    5e-7,       #  2: J
    5e-7,       #  3: Aa1
    5e-7,       #  4: Ab1
    5e-7,       #  5: Aa2
    5e-7,       #  6: Ab2
    1e-7,       #  7: D1
    1e-10,      #  8: D2
    50.0,       #  9: dummy B0
    2.01,       # 10: g_e
    -1.0,       # 11: g_n1
    1.6,        # 12: g_n2
    2.1e8,      # 13: nu
    1e7,        # 14: omega1
    1e8,        # 15: k_S
    1e9,        # 16: k_D
    1e6,        # 17: p
    10.0,       # 18: B_mod
], dtype=float)

x_scale = np.array([
    1e5,    # A 
    1,      # I0
    1e-8,   # J
    1e-8,   # Aa1
    2e-7,   # Ab1
    1e-8,   # Aa2
    1e-7,   # Ab2
    1e-8,   # D1
    1e-9,   # D2
    1e1,    # B0
    1.0,    # g_e
    1.0,    # g_n1
    1.0,    # g_n2
    1e8,    # nu
    1e6,    # omega1
    1e5,    # k_S
    1e5,    # k_D
    1e3,    # p
    1.0     # B_mod
], dtype=float)
