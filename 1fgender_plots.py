# %%
# --------------------------------------
# Figure 1
# --------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neurodsp.sim import sim_powerlaw
from neurodsp.utils import create_times
import scipy.signal as dsp

np.random.seed(42)
fs = 1000
n_seconds = 60
duration = 4
overlap = 0.5

# PANEL A & B
sig_male = sim_powerlaw(n_seconds, fs, exponent=-1)
sig_female = sim_powerlaw(n_seconds, fs, exponent=-1.1)

freq, psd_male = dsp.welch(sig_male, fs=fs,
                           nperseg=duration*fs,
                           noverlap=int(duration*fs*overlap))

_, psd_female = dsp.welch(sig_female, fs=fs,
                          nperseg=duration*fs,
                          noverlap=int(duration*fs*overlap))

# PANEL C & D 
n_samples = n_seconds * fs
time_vector = np.arange(n_samples) / fs
hormone = 0.5 + 0.5 * np.sin(2 * np.pi * time_vector / 20)
slopes = -1 + 0.5 * hormone
sig_dynamic = np.zeros(n_samples)

for i in range(n_seconds):
    start = i * fs
    end = (i + 1) * fs
    slope = slopes[start]
    sig_dynamic[start:end] = sim_powerlaw(1, fs, exponent=slope)

times = create_times(n_seconds=n_seconds, fs=fs)

# Split into Low vs High Hormone Epochs
median_hormone = np.median(hormone)

low_idx = hormone <= median_hormone
high_idx = hormone > median_hormone

sig_low = sig_dynamic[low_idx]
sig_high = sig_dynamic[high_idx]

freq_low, psd_low = dsp.welch(sig_low, fs=fs,
                              nperseg=duration*fs,
                              noverlap=int(duration*fs*overlap))

freq_high, psd_high = dsp.welch(sig_high, fs=fs,
                                nperseg=duration*fs,
                                noverlap=int(duration*fs*overlap))


# PANEL E
hormone_levels = np.linspace(0, 1, 20)
slopes_linear = -1 + 0.5 * hormone_levels

# FIGURE LAYOUT
fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.7],
                       hspace=0.4, wspace=0.3)

# Panel A
axA = fig.add_subplot(gs[0, 0])
axA.plot(freq, psd_male, label='Male')
axA.plot(freq, psd_female, label='Female')
axA.set_xscale('log')
axA.set_yscale('log')
axA.set_title('(A) Binary Sex Comparison')
axA.set_xlabel('Frequency (Hz)')
axA.set_ylabel('Power (a.u.)')
axA.legend()

# Panel B
axB = fig.add_subplot(gs[0, 1])
axB.hist(sig_male, bins=50, alpha=0.6, label='Male')
axB.hist(sig_female, bins=50, alpha=0.6, label='Female')
axB.set_title('(B) Amplitude Distribution')
axB.set_xlabel('Amplitude')
axB.set_ylabel('Count')
axB.legend()

# Panel C
axC = fig.add_subplot(gs[1, 0])

# Compute slope for each 1-second segment
window = fs  # 1 second
slope_over_time = []
time_points = []

for i in range(0, len(sig_dynamic)-window, window):
    segment = sig_dynamic[i:i+window]
    freqs, psd_segment = dsp.welch(segment, fs=fs, nperseg=window)
    log_freqs = np.log10(freqs[1:])   # ignore DC
    log_psd = np.log10(psd_segment[1:])
    slope = np.polyfit(log_freqs, log_psd, 1)[0]
    slope_over_time.append(slope)
    time_points.append(i/fs)

axC.plot(time_points, slope_over_time, color='tab:blue', linewidth=2)
axC.set_title('(C) Dynamic Hormonal Modulation (Slope over time)')
axC.set_xlabel('Time (s)')
axC.set_ylabel('1/f Slope')

# Panel D
axD = fig.add_subplot(gs[1, 1])
axD.plot(freq_low, psd_low, label='Low Hormone')
axD.plot(freq_high, psd_high, label='High Hormone')
axD.set_xscale('log')
axD.set_yscale('log')
axD.set_title('(D) PSD Across Hormonal States')
axD.set_xlabel('Frequency (Hz)')
axD.set_ylabel('Power (a.u.)')
axD.legend()

# Panel E
axE = fig.add_subplot(gs[2, :])
axE.scatter(hormone_levels, slopes_linear)
axE.plot(hormone_levels, slopes_linear)
axE.set_title('(E) 1/f Slope as Neuroendocrine State Marker (Illustrative)')
axE.set_xlabel('Hormone Level (a.u.)')
axE.set_ylabel('Slope (Exponent)')

plt.tight_layout()
plt.show()
# %%
