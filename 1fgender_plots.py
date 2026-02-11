# %%
# --------------------------------------
# Figure 1
# --------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from neurodsp.sim import sim_combined
from neurodsp.utils import create_times
import scipy.signal as dsp

np.random.seed(42)
fs = 1000
n_seconds = 60
duration = 4
overlap = 0.5

# --- Panel A+B: Male vs Female ---
sim_components_A = {'sim_powerlaw': {'exponent': -1}}
sig_male = sim_combined(n_seconds=n_seconds, fs=fs, components=sim_components_A)
sig_female = sim_combined(n_seconds=n_seconds, fs=fs, components={'sim_powerlaw': {'exponent': -1.1}})

freq, psd_male = dsp.welch(sig_male, fs=fs, nperseg=duration*fs, noverlap=duration*fs*overlap)
_, psd_female = dsp.welch(sig_female, fs=fs, nperseg=duration*fs, noverlap=duration*fs*overlap)

# --- Panel C+D: Dynamic hormonal modulation ---
n_cycles = 5
hormone = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_seconds*fs)/fs/(n_seconds/n_cycles))
slopes = -1 + hormone*0.5

# Time series for dynamic slope
sig_dynamic = np.array([sim_combined(n_seconds=1, fs=fs, components={'sim_powerlaw': {'exponent': s}}) 
                        for s in slopes[:n_seconds]])
sig_dynamic = sig_dynamic.flatten()
times = create_times(n_seconds=len(sig_dynamic)/fs, fs=fs)

# PSD for dynamic signal
freq_dyn, psd_dyn = dsp.welch(sig_dynamic, fs=fs, nperseg=duration*fs, noverlap=duration*fs*overlap)

# --- Panel E: Hormone vs slope ---
hormone_levels = np.linspace(0, 1, 20)
slopes_linear = -1 + 0.5 * hormone_levels

# --- Figure layout ---
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(10,12))
gs = gridspec.GridSpec(3, 2, height_ratios=[1,1,0.7], hspace=0.4, wspace=0.3)

# Panel A: PSD comparison
axA = fig.add_subplot(gs[0,0])
axA.plot(freq, psd_male, label='Male')
axA.plot(freq, psd_female, label='Female')
axA.set_xscale('log')
axA.set_yscale('log')
axA.set_title('(A) Binary Sex Comparison')
axA.set_xlabel('Frequency (Hz)')
axA.set_ylabel('Power (a.u.)')
axA.legend()

# Panel B: Amplitude distribution
axB = fig.add_subplot(gs[0,1])
axB.hist([sig_male, sig_female], bins=50, label=['Male','Female'], alpha=0.7)
axB.set_title('(B) Amplitude Distribution')
axB.legend()

# Panel C: Dynamic hormonal modulation (time series)
axC = fig.add_subplot(gs[1,0])
axC.plot(times[:fs*5], sig_dynamic[:fs*5], color='tab:blue')
axC.set_title('(C) Dynamic Hormonal Modulation')
axC.set_xlabel('Time (s)')
axC.set_ylabel('Amplitude (a.u.)')

# Panel D: PSD across hormonal fluctuation
axD = fig.add_subplot(gs[1,1])
axD.plot(freq_dyn, psd_dyn, color='tab:green')
axD.set_xscale('log')
axD.set_yscale('log')
axD.set_title('(D) PSD Across Hormonal Fluctuation')
axD.set_xlabel('Frequency (Hz)')
axD.set_ylabel('Power (a.u.)')

# Panel E: 1/f slope vs hormone (spans full bottom row)
axE = fig.add_subplot(gs[2,:])
axE.scatter(hormone_levels, slopes_linear, color='purple')
axE.plot(hormone_levels, slopes_linear, color='purple')
axE.set_title('(E) 1/f Slope as a Neuroendocrine State Marker')
axE.set_xlabel('Hormone Level (a.u.)')
axE.set_ylabel('Slope (exponent)')

plt.tight_layout()
plt.show()


# %%
# --------------------------------------
# Figure 2
# --------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from neurodsp.sim import sim_combined
from neurodsp.utils import create_times
import scipy.signal as dsp

np.random.seed(42)
fs = 1000
n_seconds = 60
duration = 4
overlap = 0.5
n_individuals = 5
n_cycles = 3  # hormone cycles in 60 seconds

#Baselines
baseline_slopes = -1 + 0.1*np.random.randn(n_individuals)  # inter-individual variability
times = np.arange(n_seconds)
all_slope_series = []
all_psds = []

# Simulate each individual
for s0 in baseline_slopes:
    # Hormone fluctuation (sinusoidal)
    hormone = 0.5 + 0.5 * np.sin(2 * np.pi * times / (n_seconds / n_cycles))
    slopes = s0 + 0.5 * hormone
    all_slope_series.append(slopes)
    
    # Generate full signal per individual
    sig = np.array([sim_combined(n_seconds=1, fs=fs, components={'sim_powerlaw': {'exponent': sl}}) 
                    for sl in slopes])
    sig = sig.flatten()
    
    # Compute PSD
    freq, psd = dsp.welch(sig, fs=fs, nperseg=duration*fs, noverlap=duration*fs*overlap)
    all_psds.append(psd)

# --- Plotting ---
fig, axes = plt.subplots(2,1, figsize=(10,8), constrained_layout=True)

colors = plt.cm.viridis(np.linspace(0,1,n_individuals))

for i, slopes in enumerate(all_slope_series):
    axes[0].plot(times, slopes, color=colors[i], lw=2, label=f'Individual {i+1}')
axes[0].set_title('(A) Dynamic 1/f Slope Across Multiple Individuals')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Slope (exponent)')
axes[0].legend()
axes[0].grid(alpha=0.3)

for i, psd in enumerate(all_psds):
    axes[1].loglog(freq, psd, color=colors[i], lw=2, label=f'Individual {i+1}')
axes[1].set_title('(B) PSD Across Individuals with Hormonal Fluctuations')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Power (a.u.)')
axes[1].legend()
axes[1].grid(alpha=0.3, which='both')
plt.show()

# %%
