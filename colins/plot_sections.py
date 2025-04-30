# plot_sections.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from spin_helpers import spectra_producer

TextString = 'Simulated NZFMR Spectra (Vary '
Normalized = 'Normalized '
normalizedYlim = 1.2
narrowXlim = 20

def plot_changing_r_ab():
    """
    Replicates the "Changing r_ab" section.
    Varies r_ab from 0.4 to 2.0 GHz in steps of 0.4.
    Saves:
      - Raw plot: imgs/changing_rab.png
      - Normalized (broad): imgs/changing_rab_norm_broader.png
      - Normalized (inflection): imgs/changing_rab_norm_inflection.png
    """
    Field = np.linspace(-10, 10, 4096)
    g = 2.0023
    HF1 = 1.0
    HF2 = 1.0
    r_ae = 0.1
    r_ea = 1.0

    xvals = np.arange(0.4, 2.001, 0.4)
    I_ab_list = []
    for val in xvals:
        spec = spectra_producer(Field, g, HF1, HF2, val, r_ae, r_ea)
        I_ab_list.append(spec)
    I_ab = np.column_stack(I_ab_list)

    # --- Raw offset plot ---
    I_aboffset = np.zeros_like(I_ab)
    I_aboffset[:, 0] = I_ab[:, 0].copy()
    for i in range(1, I_ab.shape[1]):
        offset_val = np.min(I_aboffset[:, i-1]) - np.max(I_ab[:, i]) - 0.001
        I_aboffset[:, i] = I_ab[:, i] + offset_val

    plt.figure(figsize=(6,4))
    varlist = []
    for i in range(I_aboffset.shape[1]):
        plt.plot(Field*10, I_aboffset[:, i])
        varlist.append(f"{xvals[i]:.2f}")
    plt.legend(varlist, loc='best')
    plt.title(TextString + r"r_{\alpha\beta})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("imgs/changing_rab.png", dpi=300)
    plt.close()

    # --- Normalized plots ---
    scale1_ab = np.zeros_like(I_ab)
    scale2_ab = np.zeros_like(I_ab)
    for i in range(I_ab.shape[1]):
        rowdata = I_ab[:, i]
        peaks, _ = find_peaks(rowdata)
        if len(peaks) >= 2:
            first_peak = rowdata[peaks[0]]
            second_peak = rowdata[peaks[1]]
        elif len(peaks) == 1:
            first_peak = rowdata[peaks[0]]
            second_peak = first_peak
        else:
            first_peak = 1.0
            second_peak = 1.0
        scale1_ab[:, i] = rowdata / first_peak
        scale2_ab[:, i] = rowdata / second_peak

    plt.figure(figsize=(6,4))
    for i in range(scale1_ab.shape[1]):
        plt.plot(Field*10, scale1_ab[:, i])
    plt.legend(varlist, loc='best')
    plt.title("a) " + Normalized + TextString + r"r_{\alpha\beta})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.ylim([-normalizedYlim, normalizedYlim])
    plt.tight_layout()
    plt.savefig("imgs/changing_rab_norm_broader.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    for i in range(scale2_ab.shape[1]):
        plt.plot(Field*10, scale2_ab[:, i])
    plt.legend(varlist, loc='best')
    plt.title("b) " + Normalized + TextString + r"r_{\alpha\beta})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.ylim([-normalizedYlim, normalizedYlim])
    plt.tight_layout()
    plt.savefig("imgs/changing_rab_norm_inflection.png", dpi=300)
    plt.close()

def plot_changing_r_ae():
    """
    Replicates the "Changing r_ae" section.
    Varies r_ae from 0.04 to 0.2 GHz in steps of 0.04.
      - Raw plot: imgs/changing_rae.png
      - Normalized (broad): imgs/changing_rae_norm_broader.png
      - Normalized (inflection): imgs/changing_rae_norm_inflection.png
    """
    Field = np.linspace(-10, 10, 4096)
    g = 2.0023
    HF1 = 1.0
    HF2 = 1.0
    r_ab = 1.0
    r_ea = 1.0

    xvals = np.arange(0.04, 0.2001, 0.04)
    I_ae_list = []
    for val in xvals:
        spec = spectra_producer(Field, g, HF1, HF2, r_ab, val, r_ea)
        I_ae_list.append(spec)
    I_ae = np.column_stack(I_ae_list)

    I_aeoffset = np.zeros_like(I_ae)
    I_aeoffset[:, 0] = I_ae[:, 0].copy()
    for i in range(1, I_ae.shape[1]):
        offset_val = np.min(I_aeoffset[:, i-1]) - 1.1 * np.max(I_ae[:, i])
        I_aeoffset[:, i] = I_ae[:, i] + offset_val

    plt.figure(figsize=(6,4))
    varlist = []
    for i in range(I_aeoffset.shape[1]):
        plt.plot(Field*10, I_aeoffset[:, i])
        varlist.append(f"{xvals[i]:.4f}")
    plt.legend(varlist, loc='best')
    plt.title(TextString + r"r_{\alphae})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("imgs/changing_rae.png", dpi=300)
    plt.close()

    scale1_ae = np.zeros_like(I_ae)
    scale2_ae = np.zeros_like(I_ae)
    for i in range(I_ae.shape[1]):
        rowdata = I_ae[:, i]
        peaks, _ = find_peaks(rowdata)
        if len(peaks) >= 2:
            first_peak = rowdata[peaks[0]]
            second_peak = rowdata[peaks[1]]
        elif len(peaks) == 1:
            first_peak = rowdata[peaks[0]]
            second_peak = first_peak
        else:
            first_peak = 1.0
            second_peak = 1.0
        scale1_ae[:, i] = rowdata / first_peak
        scale2_ae[:, i] = rowdata / second_peak

    plt.figure(figsize=(6,4))
    for i in range(scale1_ae.shape[1]):
        plt.plot(Field*10, scale1_ae[:, i])
    plt.legend(varlist, loc='best')
    plt.title("a) " + Normalized + TextString + r"r_{\alphae})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.ylim([-normalizedYlim, normalizedYlim])
    plt.tight_layout()
    plt.savefig("imgs/changing_rae_norm_broader.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    for i in range(scale2_ae.shape[1]):
        plt.plot(Field*10, scale2_ae[:, i])
    plt.legend(varlist, loc='best')
    plt.title("b) " + Normalized + TextString + r"r_{\alphae})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.ylim([-normalizedYlim, normalizedYlim])
    plt.tight_layout()
    plt.savefig("imgs/changing_rae_norm_inflection.png", dpi=300)
    plt.close()

def plot_changing_r_ea():
    """
    Replicates the "Changing r_ea" section.
    Varies r_ea from 0.4 to 2.0 GHz in steps of 0.4, with r_ab = 0.3 and r_ae = 0.1 fixed.
      - Raw plot: imgs/changing_rea.png
      - Normalized (broad): imgs/changing_rea_norm_broader.png
      - Normalized (inflection): imgs/changing_rea_norm_inflection.png
    """
    Field = np.linspace(-10, 10, 4096)
    g = 2.0023
    HF1 = 1.0
    HF2 = 1.0
    r_ab = 0.3
    r_ae = 0.1

    xvals = np.arange(0.4, 2.001, 0.4)
    I_ea_list = []
    for val in xvals:
        spec = spectra_producer(Field, g, HF1, HF2, r_ab, r_ae, val)
        I_ea_list.append(spec)
    I_ea = np.column_stack(I_ea_list)

    I_eaoffset = np.zeros_like(I_ea)
    I_eaoffset[:, 0] = I_ea[:, 0].copy()
    for i in range(1, I_ea.shape[1]):
        offset_val = np.min(I_eaoffset[:, i-1]) - (1.1 * np.max(I_ea[:, i]))
        I_eaoffset[:, i] = I_ea[:, i] + offset_val

    plt.figure(figsize=(6,4))
    varlist = []
    for i in range(I_eaoffset.shape[1]):
        plt.plot(Field*10, I_eaoffset[:, i])
        varlist.append(f"{xvals[i]:.2f}")
    plt.legend(varlist, loc='best')
    plt.title(TextString + r"r_{e\alpha})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("imgs/changing_rea.png", dpi=300)
    plt.close()

    scale1_ea = np.zeros_like(I_ea)
    scale1_eaOffset = np.zeros_like(I_ea)
    for i in range(I_ea.shape[1]):
        rowdata = I_ea[:, i]
        peaks, _ = find_peaks(rowdata)
        if len(peaks) >= 1:
            first_peak = rowdata[peaks[0]]
        else:
            first_peak = 1.0
        scale1_ea[:, i] = rowdata / first_peak
        if i == 0:
            scale1_eaOffset[:, i] = scale1_ea[:, i]
        else:
            offset_val = -0.03 * i
            scale1_eaOffset[:, i] = scale1_ea[:, i] + offset_val

    plt.figure(figsize=(6,4))
    for i in range(scale1_eaOffset.shape[1]):
        plt.plot(Field*10, scale1_eaOffset[:, i])
    plt.legend(varlist, loc='best')
    plt.title(Normalized + TextString + r"r_{e\alpha})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.ylim([-normalizedYlim, normalizedYlim])
    plt.tight_layout()
    plt.savefig("imgs/changing_rea_norm_broader.png", dpi=300)
    plt.close()

def plot_changing_HF1():
    """
    Replicates the "Changing HF1" section.
    Varies HF1 from 0.4 to 2.0 (mT) in steps of 0.4 while HF2 is fixed at 0.
      - Raw plot: imgs/changing_HF1.png
      - Normalized (broad): imgs/changing_HF1_norm_broader.png
      - Normalized (inflection): imgs/changing_HF1_norm_inflection.png
    """
    Field = np.linspace(-10, 10, 4096)
    g = 2.0023
    HF2 = 0.0
    r_ab = 1.0
    r_ae = 0.1
    r_ea = 1.0

    xvals = np.arange(0.4, 2.001, 0.4)
    I_HF1_list = []
    for val in xvals:
        spec = spectra_producer(Field, g, val, HF2, r_ab, r_ae, r_ea)
        I_HF1_list.append(spec)
    I_HF1 = np.column_stack(I_HF1_list)

    I_HF1offset = np.zeros_like(I_HF1)
    I_HF1offset[:, 0] = I_HF1[:, 0].copy()
    varlist = []
    for i in range(I_HF1.shape[1]):
        if i > 0:
            offset_val = np.min(I_HF1offset[:, i-1]) - (1.1 * np.max(I_HF1[:, i]))
            I_HF1offset[:, i] = I_HF1[:, i] + offset_val
        varlist.append(f"{xvals[i]:.2f}")

    plt.figure(figsize=(6,4))
    for i in range(I_HF1offset.shape[1]):
        plt.plot(Field*10, I_HF1offset[:, i])
    plt.legend(varlist, loc='best')
    plt.title(TextString + r"HF_{\alpha})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("imgs/changing_HF1.png", dpi=300)
    plt.close()

    scale1_HF1 = np.zeros_like(I_HF1)
    scale2_HF1 = np.zeros_like(I_HF1)
    for i in range(I_HF1.shape[1]):
        rowdata = I_HF1[:, i]
        peaks, _ = find_peaks(rowdata)
        if len(peaks) >= 2:
            first_peak = rowdata[peaks[0]]
            second_peak = rowdata[peaks[1]]
        elif len(peaks) == 1:
            first_peak = rowdata[peaks[0]]
            second_peak = first_peak
        else:
            first_peak = 1.0
            second_peak = 1.0
        scale1_HF1[:, i] = rowdata / first_peak
        scale2_HF1[:, i] = rowdata / second_peak

    plt.figure(figsize=(6,4))
    for i in range(scale1_HF1.shape[1]):
        plt.plot(Field*10, scale1_HF1[:, i])
    plt.legend(varlist, loc='best')
    plt.title("a) " + Normalized + TextString + r"HF_{\alpha})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.ylim([-normalizedYlim, normalizedYlim])
    plt.tight_layout()
    plt.savefig("imgs/changing_HF1_norm_broader.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    for i in range(1, scale2_HF1.shape[1]):
        plt.plot(Field*10, scale2_HF1[:, i])
    plt.legend(varlist[1:], loc='best')
    plt.title("b) " + Normalized + TextString + r"HF_{\alpha})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.xlim([-narrowXlim, narrowXlim])
    plt.ylim([-normalizedYlim, normalizedYlim])
    plt.tight_layout()
    plt.savefig("imgs/changing_HF1_norm_inflection.png", dpi=300)
    plt.close()

def plot_changing_HF2():
    """
    Replicates the "Changing HF2" section.
    Varies HF2 from 0.4 to 2.0 (mT) in steps of 0.4 while HF1 is fixed at 1.
      - Raw plot: imgs/changing_HF2.png
      - Normalized (broad): imgs/changing_HF2_norm_broader.png
      - Normalized (inflection): imgs/changing_HF2_norm_inflection.png
    """
    Field = np.linspace(-10, 10, 4096)
    g = 2.0023
    HF1 = 1.0
    r_ab = 1.0
    r_ae = 0.1
    r_ea = 1.0

    xvals = np.arange(0.4, 2.001, 0.4)
    I_HF2_list = []
    for val in xvals:
        spec = spectra_producer(Field, g, HF1, val, r_ab, r_ae, r_ea)
        I_HF2_list.append(spec)
    I_HF2 = np.column_stack(I_HF2_list)

    I_HF2offset = np.zeros_like(I_HF2)
    I_HF2offset[:, 0] = I_HF2[:, 0].copy()
    varlist = []
    for i in range(I_HF2.shape[1]):
        if i > 0:
            offset_val = np.min(I_HF2offset[:, i-1]) - (1.1 * np.max(I_HF2[:, i]))
            I_HF2offset[:, i] = I_HF2[:, i] + offset_val
        varlist.append(f"{xvals[i]:.2f}")

    plt.figure(figsize=(6,4))
    for i in range(I_HF2offset.shape[1]):
        plt.plot(Field*10, I_HF2offset[:, i])
    plt.legend(varlist, loc='best')
    plt.title(TextString + r"HF_{\beta})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("imgs/changing_HF2.png", dpi=300)
    plt.close()

    scale1_HF2 = np.zeros_like(I_HF2)
    scale2_HF2 = np.zeros_like(I_HF2)
    for i in range(I_HF2.shape[1]):
        rowdata = I_HF2[:, i]
        peaks, _ = find_peaks(rowdata)
        if len(peaks) >= 2:
            first_peak = rowdata[peaks[0]]
            second_peak = rowdata[peaks[1]]
        elif len(peaks) == 1:
            first_peak = rowdata[peaks[0]]
            second_peak = first_peak
        else:
            first_peak = 1.0
            second_peak = 1.0
        scale1_HF2[:, i] = rowdata / first_peak
        scale2_HF2[:, i] = rowdata / second_peak

    plt.figure(figsize=(6,4))
    for i in range(scale1_HF2.shape[1]):
        plt.plot(Field*10, scale1_HF2[:, i])
    plt.legend(varlist, loc='best')
    plt.title("a) " + Normalized + TextString + r"HF_{\beta})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.ylim([-normalizedYlim, normalizedYlim])
    plt.tight_layout()
    plt.savefig("imgs/changing_HF2_norm_broader.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    for i in range(scale2_HF2.shape[1]):
        plt.plot(Field*10, scale2_HF2[:, i])
    plt.legend(varlist, loc='best')
    plt.title("b) " + Normalized + TextString + r"HF_{\beta})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.xlim([-narrowXlim, narrowXlim])
    plt.ylim([-normalizedYlim, normalizedYlim])
    plt.tight_layout()
    plt.savefig("imgs/changing_HF2_norm_inflection.png", dpi=300)
    plt.close()

def plot_changing_HF12():
    """
    Replicates the "Changing HF1 and HF2 together" section.
    Varies both HF1 and HF2 simultaneously from 0.4 to 2.0 (mT) in steps of 0.4.
      - Raw plot: imgs/changing_HF12.png
      - Normalized (broad): imgs/changing_HF12_norm_broader.png
      - Normalized (inflection): imgs/changing_HF12_norm_inflection.png
    """
    Field = np.linspace(-10, 10, 4096)
    g = 2.0023
    r_ab = 1.0
    r_ae = 0.1
    r_ea = 1.0

    xvals = np.arange(0.4, 2.001, 0.4)
    I_HF12_list = []
    for val in xvals:
        # Both HF1 and HF2 are set to val
        spec = spectra_producer(Field, g, val, val, r_ab, r_ae, r_ea)
        I_HF12_list.append(spec)
    I_HF12 = np.column_stack(I_HF12_list)

    I_HF12offset = np.zeros_like(I_HF12)
    I_HF12offset[:, 0] = I_HF12[:, 0].copy()
    varlist = []
    for i in range(I_HF12.shape[1]):
        if i > 0:
            offset_val = np.min(I_HF12offset[:, i-1]) - (1.1 * np.max(I_HF12[:, i]))
            I_HF12offset[:, i] = I_HF12[:, i] + offset_val
        varlist.append(f"{xvals[i]:.2f}")

    plt.figure(figsize=(6,4))
    for i in range(I_HF12offset.shape[1]):
        plt.plot(Field*10, I_HF12offset[:, i])
    plt.legend(varlist, loc='best')
    plt.title(TextString + r"HF_{\alpha} and HF_{\beta})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("imgs/changing_HF12.png", dpi=300)
    plt.close()

    scale1_HF12 = np.zeros_like(I_HF12)
    scale2_HF12 = np.zeros_like(I_HF12)
    for i in range(I_HF12.shape[1]):
        rowdata = I_HF12[:, i]
        peaks, _ = find_peaks(rowdata)
        if len(peaks) >= 2:
            first_peak = rowdata[peaks[0]]
            second_peak = rowdata[peaks[1]]
        elif len(peaks) == 1:
            first_peak = rowdata[peaks[0]]
            second_peak = first_peak
        else:
            first_peak = 1.0
            second_peak = 1.0
        scale1_HF12[:, i] = rowdata / first_peak
        scale2_HF12[:, i] = rowdata / second_peak

    plt.figure(figsize=(6,4))
    for i in range(scale1_HF12.shape[1]):
        plt.plot(Field*10, scale1_HF12[:, i])
    plt.legend(varlist, loc='best')
    plt.title("a) " + Normalized + TextString + r"HF_{\alpha} and HF_{\beta})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.ylim([-normalizedYlim, normalizedYlim])
    plt.tight_layout()
    plt.savefig("imgs/changing_HF12_norm_broader.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    for i in range(scale2_HF12.shape[1]):
        plt.plot(Field*10, scale2_HF12[:, i])
    plt.legend(varlist, loc='best')
    plt.title("b) " + Normalized + TextString + r"HF_{\alpha} and HF_{\beta})")
    plt.xlabel("Magnetic Field (Gauss)")
    plt.ylabel("Simulated NZFMR Amplitude (arb. units)")
    plt.yticks([])
    plt.xlim([-narrowXlim, narrowXlim])
    plt.ylim([-normalizedYlim, normalizedYlim])
    plt.tight_layout()
    plt.savefig("imgs/changing_HF12_norm_inflection.png", dpi=300)
    plt.close()

