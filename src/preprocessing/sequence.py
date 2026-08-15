import numpy as np

def resampling_sequence(sequence, target_length):
    n_frames = sequence.shape[0]
    if n_frames == target_length:
        return sequence

    indices = np.linspace(
        0, n_frames - 1, target_length
    ).round().astype(int)
    return sequence[indices]
