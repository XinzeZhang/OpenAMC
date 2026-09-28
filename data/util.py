
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