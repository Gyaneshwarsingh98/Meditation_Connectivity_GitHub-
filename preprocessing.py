# scripts/01_preprocessing.py

import os
import numpy as np
import mne
from autoreject import Ransac


def identify_bad_channels(raw):
    """
    Identify and interpolate bad EEG channels using RANSAC.

    Parameters:
    - raw: mne.io.Raw object (raw EEG data)

    Returns:
    - bad_channels: List of identified bad channels
    """
    # Define events every 2 seconds for RANSAC processing
    events = mne.make_fixed_length_events(raw, start=0, duration=2.0)
    event_id = {'Stimulus': 1}

    # Create epochs with 2-second segments
    epochs = mne.Epochs(
        raw, events, event_id, tmin=0, tmax=2.0, baseline=None, preload=True
    )

    # Apply RANSAC to identify bad channels
    ransac = Ransac(n_jobs=1, verbose=True)
    ransac.fit(epochs)

    # Extract bad channels and update raw object
    bad_channels = ransac.bad_chs_
    print(f"Bad channels identified by RANSAC: {bad_channels}")

    if len(bad_channels) > 0:
        raw.info['bads'] = bad_channels
        raw.interpolate_bads(reset_bads=True)
        print("Interpolated bad channels.")

    return bad_channels


def apply_ica(raw):
    """
    Apply Independent Component Analysis (ICA) to remove eye-movement artifacts.

    Parameters:
    - raw: mne.io.Raw object (raw EEG data)

    Returns:
    - raw_ica: Cleaned raw data after ICA artifact removal
    """
    ica = mne.preprocessing.ICA(n_components=0.95, random_state=97, max_iter=800)
    print("Fitting ICA...")
    ica.fit(raw)

    frontal_channels = [
        'E17', 'E21', 'E14', 'E22', 'E15', 'E9', 'E18', 'E16', 'E10',
        'E19', 'E11', 'E4', 'E12', 'E5', 'E25', 'E32', 'E26', 'E23',
        'E38', 'E33', 'E34', 'E28', 'E27', 'E24', 'E20', 'E8', 'E1',
        'E121', 'E2', 'E122', 'E3', 'E123', 'E124', 'E118', 'E116',
        'E117'
    ]

    picks_frontal = mne.pick_channels(raw.info['ch_names'], frontal_channels)
    if len(picks_frontal) == 0:
        raise ValueError("No frontal channels found. Check channel names in your dataset.")

    print("Computing ICA sources...")
    ica_sources = ica.get_sources(raw).get_data()
    frontal_data = raw.get_data(picks=picks_frontal)

    print("Calculating correlations...")
    correlations = np.corrcoef(ica_sources, frontal_data)[len(ica_sources):, :len(ica_sources)]

    correlation_threshold = 2
    eog_inds = np.where(np.max(np.abs(correlations), axis=0) > correlation_threshold)[0]

    print(f"Identified components to exclude: {eog_inds}")
    ica.exclude = eog_inds.tolist()

    print("Applying ICA to raw data...")
    raw_ica = ica.apply(raw.copy())
    print("ICA completed. Eye-movement artifacts removed.")
    return raw_ica


def load_and_preprocess_eeg(file_path):
    """
    Load and preprocess EEG data.

    Parameters:
    - file_path: Path to the .mff EEG file

    Returns:
    - raw: Preprocessed raw EEG data
    """
    raw = mne.io.read_raw_egi(file_path, preload=True)
    raw.drop_channels(['Vertex Reference'])
    raw.filter(0.3, 40, fir_design="firwin")
    raw.notch_filter(60, fir_design="firwin")
    raw = apply_ica(raw)
    bad_channels = identify_bad_channels(raw)
    print("Final list of bad channels:", bad_channels)
    return raw


def save_preprocessed_data(raw, npy_dir, npz_dir, fif_dir, base_name):
    """
    Save preprocessed EEG data in .npy, .npz, and .fif formats.

    Parameters:
    - raw: mne.io.Raw object
    - npy_dir, npz_dir, fif_dir: Output directories
    - base_name: Filename base (no extension)
    """
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    channels = raw.info['ch_names']

    np.save(os.path.join(npy_dir, f"{base_name}.npy"), data)
    print(f"Saved .npy to {npy_dir}")

    np.savez(os.path.join(npz_dir, f"{base_name}.npz"), data=data, sfreq=sfreq, channels=channels)
    print(f"Saved .npz to {npz_dir}")

    raw.save(os.path.join(fif_dir, f"{base_name}.fif"), overwrite=True)
    print(f"Saved .fif to {fif_dir}")


def process_multiple_files(input_dir, output_dir):
    """
    Process all .mff files in input_dir and save preprocessed versions.

    Parameters:
    - input_dir: Directory of .mff files
    - output_dir: Base output directory for preprocessed files
    """
    npy_dir = os.path.join(output_dir, "npy_files")
    npz_dir = os.path.join(output_dir, "npz_files")
    fif_dir = os.path.join(output_dir, "fif_files")

    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(fif_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".mff"):
            file_path = os.path.join(input_dir, file_name)
            base_name = os.path.splitext(file_name)[0]
            print(f"Processing file: {file_name}")
            try:
                raw = load_and_preprocess_eeg(file_path)
                save_preprocessed_data(raw, npy_dir, npz_dir, fif_dir, base_name)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    # Update with your actual paths
    input_dir = "/Users/gyaneshwarsingh/Brain_connectivity/data/Short_term_meditator"
    output_dir = "/Users/gyaneshwarsingh/Brain_connectivity/data/Short_term_meditator/preprocessed"

    process_multiple_files(input_dir, output_dir)
