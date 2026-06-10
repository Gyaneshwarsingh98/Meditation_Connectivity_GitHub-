# Phase-Resolved Modulation of Alpha-Band Functional Connectivity During a Multistage Meditation Protocol

## Overview

This repository contains the analysis scripts used in the study:


The repository provides scripts for:

- EEG preprocessing
- Source-space reconstruction
- Alpha-band functional connectivity estimation using the weighted Phase Lag Index (wPLI)
- Network-level aggregation of connectivity measures
- Linear mixed-effects modeling
- Generation of publication figures and summary tables

---

## Analysis Workflow

The analysis pipeline follows the steps below:

1. Raw EEG preprocessing
   - Band-pass filtering (0.3–40 Hz)
   - Notch filtering (60 Hz)
   - ICA-based artifact correction
   - RANSAC-based bad-channel detection
   - Channel interpolation

2. Source-space reconstruction
   - Template anatomy: fsaverage
   - Forward model: Boundary Element Model (BEM)
   - Inverse solution: dSPM
   - Cortical parcellation: Desikan–Killiany atlas

3. Functional connectivity analysis
   - Frequency band: Alpha (8–12 Hz)
   - Connectivity metric: weighted Phase Lag Index (wPLI)

4. Network-level aggregation
   - Frontal–Parietal
   - Frontal–Temporal
   - Intra-Frontal
   - Intra-Parietal

5. Statistical analysis
   - Linear mixed-effects model
   - Formula:

     Connectivity ~ Phase * Group + Network

   - Random intercept: Subject
   - Estimation: REML

---

## Repository Structure

scripts/

- 01_preprocessing.py
- 02_source_space_connectivity.py
- 03_final_manuscript_analysis.py

results/

- alpha_wpli_lmm_summary.txt
- Table3_coefficients.csv

requirements.txt

README.md

---

## Group Abbreviations

- CG = Control Group
- STM = Short-Term Meditators
- LTM = Long-Term Meditators

---

## Software Requirements

Python 3.11

Required packages:

- mne
- mne-connectivity
- numpy
- pandas
- scipy
- statsmodels
- matplotlib
- seaborn
- autoreject

Install dependencies:

pip install -r requirements.txt

---

## Data Availability

Raw EEG data are not publicly available because participant consent and institutional approvals do not permit unrestricted public sharing.

Analysis scripts and statistical outputs required to reproduce the reported analyses are provided in this repository.

---

## Citation

If you use this repository, please cite:

Singh G, et al.
Phase-Resolved Modulation of Alpha-Band Functional Connectivity During a Multistage Meditation Protocol.
Clinical EEG & Neuroscience.
