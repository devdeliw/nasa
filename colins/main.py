# main.py
from plot_sections import (plot_changing_r_ab, plot_changing_r_ae, plot_changing_r_ea,
                           plot_changing_HF1, plot_changing_HF2, plot_changing_HF12)

def main():
    print("Plotting Changing r_ab...")
    plot_changing_r_ab()
    
    print("Plotting Changing r_ae...")
    plot_changing_r_ae()
    
    print("Plotting Changing r_ea...")
    plot_changing_r_ea()
    
    print("Plotting Changing HF1...")
    plot_changing_HF1()
    
    print("Plotting Changing HF2...")
    plot_changing_HF2()
    
    print("Plotting Changing HF1 and HF2 together...")
    plot_changing_HF12()

if __name__ == '__main__':
    main()

