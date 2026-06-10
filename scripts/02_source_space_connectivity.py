"""
Source-space alpha-band connectivity analysis used in:

Phase-Resolved Modulation of Alpha-Band Functional Connectivity During a Multistage Meditation Protocol

Pipeline:
1. Load preprocessed EEG
2. Source reconstruction using fsaverage BEM-dSPM
3. Desikan-Killiany atlas parcellation
4. Alpha-band (8–12 Hz) weighted Phase Lag Index (wPLI)
5. Export region-to-region connectivity matrices

Note:
Only the analyses reported in the manuscript are included in this script.
"""


import os
import mne
import numpy as np
import pandas as pd
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator
from mne_connectivity import spectral_connectivity_epochs


# ---------------------------
# Configuration
# ---------------------------
subjects_dir = "path_to_fsaverage"
raw_data_dir = "path_to_preprocessed_data"
output_dir = "path_to_output"

spacing = 'ico5'
snr = 3.0
lambda2 = 1.0 / snr ** 2
epoch_length = 10.0  # seconds


all_methods = ['wpli']

bands = {
    'Alpha': (8, 12),
}

os.makedirs(output_dir, exist_ok=True)

def process_subject(raw_fname):
    print(f"Processing {raw_fname} ...")
    base_name = os.path.basename(raw_fname).replace('.fif', '')
    parts = base_name.split('_')
    group = parts[0] if len(parts) >= 3 else 'Unknown'
    phase = parts[-1] if len(parts) >= 3 else 'Unknown'

    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.pick_types(eeg=True)
    raw.filter(0., 40., fir_design='firwin', verbose=False)
    raw.set_eeg_reference('average', projection=True)

    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, preload=True)

    subject = 'fsaverage'
    src = mne.setup_source_space(subject, spacing=spacing, subjects_dir=subjects_dir, add_dist=False, verbose=False)
    model = mne.make_bem_model(subject=subject, ico=4, conductivity=[0.3, 0.006, 0.3], subjects_dir=subjects_dir, verbose=False)
    bem = mne.make_bem_solution(model)

    trans = 'fsaverage'
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True, meg=False, verbose=False)
    cov = mne.compute_covariance(epochs, method='empirical', verbose=False)
    inv = make_inverse_operator(raw.info, fwd, cov, loose=0.2, depth=0.8, verbose=False)

    stcs = apply_inverse_epochs(epochs, inv, lambda2, method='dSPM', pick_ori='normal', verbose=False)

    labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
    labels = [label for label in labels if 'unknown' not in label.name]
    label_ts = mne.extract_label_time_course(stcs, labels, inv['src'], mode='mean_flip', return_generator=False)
    label_ts_array = np.array(label_ts)

    sfreq = raw.info['sfreq']
    n_fft = int(epoch_length * sfreq)

    
    connectivity_data = []
    label_names = [label.name for label in labels]

    for band_name, (fmin, fmax) in bands.items():
        con_results = spectral_connectivity_epochs(label_ts_array, method=all_methods, mode='multitaper',
                                                   sfreq=sfreq, fmin=fmin, fmax=fmax,
                                                   faverage=True, verbose=False)

        for method, con_obj in zip(all_methods, con_results):
            con_matrix = con_obj.get_data(output='dense')[:, :, 0]
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if i != j:
                        connectivity_data.append({
                            'Subject': base_name,
                            'Group': group,
                            'Phase': phase,
                            'Band': band_name,
                            'Method': method,
                            'Label_1': label_names[i],
                            'Label_2': label_names[j],
                            'Value': con_matrix[i, j].item()
                        })

    
    connectivity_df = pd.DataFrame(connectivity_data)
    connectivity_df.to_csv(os.path.join(output_dir, f'{base_name}_connectivity.csv'), index=False)
    print(f"Saved results for {base_name}")

def main():
    raw_files = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith('.fif')]
    for raw_file in raw_files:
        try:
            process_subject(raw_file)
        except Exception as e:
            print(f"Error processing {raw_file}: {e}")

if __name__ == "__main__":
    main()
