import os
from pathlib import Path
import mne
from mne.preprocessing import find_bad_channels_maxwell
from mne.preprocessing import maxwell_filter
from mne.preprocessing import compute_proj_ecg
from mne.preprocessing import compute_proj_eog
import numpy as np
from pysz import sz, szAlgorithm, szConfig, szErrorBoundMode
import glob
import pickle
import warnings, logging, argparse


WORKSPACE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = WORKSPACE_DIR / "01_raw_data"
METADATA_DIR = WORKSPACE_DIR / "02_metadata"
ANNOTATIONS_DIR = METADATA_DIR / "annotations"

def pysz_decomperssion(file_path):
    with open(file_path, "rb") as file_handle:
        payload = pickle.load(file_handle)
    info = payload["info"]
    first_samp = payload["first_samp"]
    n_times = payload["n_times"]
    n_channels = payload["n_channels"]
    chunks = payload["chunks"]
    data_picks = set(mne.pick_types(info, meg=True, eeg=True, eog=True, ecg=True, exclude=[]))
    data = np.zeros((n_channels, n_times), dtype=np.float32)

    for ch_idx, chunk in chunks.items():
        dtype = np.float64 if ch_idx in data_picks else np.float32
        decompressed, _ = sz.decompress(chunk, dtype, (n_times,))
        data[ch_idx] = decompressed

    raw = mne.io.RawArray(data, info, first_samp=first_samp, verbose=False)
    fif_file_path = Path(file_path).with_suffix('')
    raw.save(fif_file_path, overwrite=True)
    return fif_file_path
def fif_preprocessing(file_path):
    if "type2" in str(file_path) or "shared_empty_rooms" in str(file_path):
        return
    
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)


    bads, flats = mne.preprocessing.find_bad_channels_maxwell(raw, verbose=False)
    raw.info["bads"] += bads
    raw.info["bads"] += flats
    raw = maxwell_filter(raw, st_duration=10.0, st_correlation=0.900, verbose=False)
    
    raw.filter(l_freq=1.0, h_freq=45.0, method='fir', phase='zero', fir_window='hamming', n_jobs=1, verbose=False)

    proj_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=1, 
                                   reject=None, no_proj=True, verbose=False)
    raw.add_proj(proj_ecg) #add projector
    proj_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=1, 
                               reject=None, no_proj=True, verbose=False)
    raw.add_proj(proj_eog) #add projector
    raw.apply_proj()
    

    fif_file_path = Path(file_path).with_name(Path(file_path).stem + "_preprocessed.fif")

    raw.save(fif_file_path, overwrite=True)
    return fif_file_path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument('filecode')
    args = parser.parse_args()
    files = sorted(glob.glob(str(RAW_DATA_DIR / "**/*.pysz*"), recursive=True))
    file_path = files[int(args.filecode)]

    fif_file_path = pysz_decomperssion(file_path)
    fif_preprocessing(fif_file_path)
