# %%
import matplotlib.pyplot as plt
import torch

from taskAttack.util import signal_energy, signal_power


def select_signal_by_label(data, signals, labels, show_label=None, label_rank=0):
	if show_label is None:
		return signals[label_rank], labels[label_rank]

	label_value = data.classes[show_label] if isinstance(show_label, str) else show_label
	matched_idx = (labels == label_value).nonzero(as_tuple=True)[0]
	if matched_idx.numel() == 0:
		raise ValueError(f"Label {show_label} is not available in the selected SNR slice.")
	if label_rank >= matched_idx.numel():
		raise IndexError(
			f"label_rank={label_rank} exceeds the number of samples ({matched_idx.numel()}) for label {show_label}."
		)

	selected_idx = matched_idx[label_rank].item()
	return signals[selected_idx], labels[selected_idx]


def iq_to_real_waveform(signal, carrier_cycles=16):
	i_channel = signal[0]
	q_channel = signal[1]
	time_axis = torch.arange(signal.shape[-1], dtype=signal.dtype, device=signal.device)
	carrier_phase = 2 * torch.pi * carrier_cycles * time_axis / signal.shape[-1]
	cos_carrier = torch.cos(carrier_phase)
	sin_carrier = torch.sin(carrier_phase)
	i_component = i_channel * cos_carrier
	q_component = -q_channel * sin_carrier
	real_waveform = i_component + q_component
	return real_waveform, i_component, q_component

from data.RML import MIMO_Nt4Nr2_Data, MIMO_Nt16Nr4_Data, MIMO_Nt64Nr16_Data, RML2016_10a_Data, RML2016_10b_Data, RML2016_04c_Data, RML2018_01a_Data, Panoradio_HF_Data

data = Panoradio_HF_Data()
data.pack_dataset()

snr = 0
show_label = 'QPSK31'
label_rank = 0
carrier_cycles = 1
sig_i, lab_i, idx_i = data.snr_slice('train', snr)

sig_i, selected_label = select_signal_by_label(data, sig_i, lab_i, show_label, label_rank)
print(sig_i.shape)
X = sig_i.view(1, 2, -1)
print(X.shape)
print(signal_energy(X), signal_power(X))
real_waveform, i_component, q_component = iq_to_real_waveform(sig_i, carrier_cycles)

# Debugging the signal visualization, uncomment the following lines to visualize the signal and its spectrum. Note that you may need to adjust the plotting code based on your specific data format and requirements.
# %matplotlib widget

label_name = next(name for name, value in data.classes.items() if value == int(selected_label))
fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
axes[0].plot(X[0,0], 'r-', linewidth=0.7, label='I channel')
axes[0].plot(X[0,1], 'g-', linewidth=0.7, label='Q channel')
axes[0].set_ylabel("Amplitude")
axes[0].set_title(f"Label: {label_name}, SNR: {snr}")
axes[0].legend(loc='upper right')

# axes[1].plot(i_component, color='tab:blue', linewidth=0.7, alpha=0.8, label='I cos component')
# axes[1].plot(q_component, color='tab:orange', linewidth=0.7, alpha=0.8, label='-Q sin component')
axes[1].plot(real_waveform, 'k-', linewidth=1.0, label='Single-channel waveform')
axes[1].set_xlabel("Time domain")
axes[1].set_ylabel("Amplitude")
axes[1].set_title(f"I/Q to real waveform, carrier cycles = {carrier_cycles}")
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.show()

import numpy as np
def compute_spectrum(signal_np, shift_zero_freq_to_center=True):
    """
    signal_np: shape [2, N] (I/Q)
    Returns magnitude spectrum in dB
    """
    # Complex signal
    complex_signal = signal_np[0] + 1j * signal_np[1]
    N = len(complex_signal)
    if shift_zero_freq_to_center:
        spectrum = np.fft.fftshift(np.fft.fft(complex_signal))
        magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
        freqs = np.fft.fftshift(np.fft.fftfreq(N))
    else:
        spectrum = np.fft.fft(complex_signal).real
        magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
        freqs = np.fft.fftfreq(N)
    return freqs, magnitude_db


freqs, magnitude_db = compute_spectrum(sig_i.cpu().numpy())
plt.figure(figsize=(8, 4))
plt.plot(freqs, magnitude_db, 'm-', linewidth=0.7)
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
plt.title(f"Spectrum of the signal (Label: {label_name}, SNR: {snr})")
plt.grid()
plt.show()
# print(magnitude_db)
# print(freqs)
# %%
