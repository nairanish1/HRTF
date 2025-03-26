#### importing necessary libraries ####
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import os
import glob

#############################################
####### HUTUBS Dataset Class ######
#############################################
class HUTUBS_Dataset(Dataset):
    def __init__(self, args, val=False):
        """
        args: A namespace with keys:
              - anthro_mat_path: path to the CSV file with normalized anthropometric data.
              - measured_hrtf_dir: directory containing individual measured HRTF files 
                                   (each with shape [fft_length x 2 x num_freq_bins x num_subjects]).
              - measured_sh_dir: directory containing individual measured SH files 
                                   (each with shape [num_coeffs x fft_length], where num_coeffs = (L+1)^2).
              - simulated_hrtf_dir: directory containing individual simulated HRTF files.
              - simulated_sh_dir: directory containing individual simulated SH files.
              - val_idx: index (integer) of the subject to use for validation.
              - batch_size: batch size for the DataLoader.
              - num_workers: number of workers for the DataLoader.
        val: Boolean flag indicating whether to load the validation subset.
        
        After stacking the individual files along a new subject axis:
              measured_hrtf -> [fft_length, 2, num_freq_bins, num_subjects]
              measured_sh   -> [num_coeffs, fft_length, num_subjects]
        (simulated data are analogous).
        """
        super(HUTUBS_Dataset, self).__init__()
        self.args = args
        self.val = val
        
        # Load anthropometric data (CSV already normalized and with subject IDs removed)
        anthro = pd.read_csv(self.args.anthro_mat_path).values.astype(np.float32)
        if self.val:
            self.anthro_mat = anthro[[self.args.val_idx], :]
        else:
            self.anthro_mat = np.delete(anthro, self.args.val_idx, axis=0)
        
        # Assume columns are ordered as:
        # first 13 columns: head measurements,
        # next 12: left ear measurements,
        # following 12: right ear measurements.
        self.anthro_head = self.anthro_mat[:, :13]
        self.anthro_left = self.anthro_mat[:, 13:25]
        self.anthro_right = self.anthro_mat[:, 25:]
        
        # Load individual measured and simulated HRTF and SH files.
        measured_hrtf_files = sorted(glob.glob(os.path.join(self.args.measured_hrtf_dir, '*_HRTF_measured_dB.mat')))
        measured_sh_files   = sorted(glob.glob(os.path.join(self.args.measured_sh_dir, '*_SH_measured.mat')))
        simulated_hrtf_files = sorted(glob.glob(os.path.join(self.args.simulated_hrtf_dir, '*_HRTF_simulated.mat')))
        simulated_sh_files   = sorted(glob.glob(os.path.join(self.args.simulated_sh_dir, '*_SH_simulated.mat')))
        
        print("Loading measured HRTF files...")
        measured_hrtf_list = [sio.loadmat(f)['hrtf_measured_dB'].astype(np.float32) for f in measured_hrtf_files]
        print("Loading measured SH files...")
        measured_sh_list = [sio.loadmat(f)['sh_coeffs_measured'].astype(np.float32) for f in measured_sh_files]
        print("Loading simulated HRTF files...")
        simulated_hrtf_list = [sio.loadmat(f)['hrtf_simulated_dB'].astype(np.float32) for f in simulated_hrtf_files]
        print("Loading simulated SH files...")
        simulated_sh_list = [sio.loadmat(f)['sh_coeffs_simulated'].astype(np.float32) for f in simulated_sh_files]
        
        # Stack along a new subject axis (last axis).
        measured_hrtf_all = np.stack(measured_hrtf_list, axis=-1)  # shape: [fft_length, 2, num_freq_bins, num_subjects]
        measured_sh_all = np.stack(measured_sh_list, axis=-1)      # shape: [num_coeffs, num_freq_bins = fft_length, num_subjects]
        simulated_hrtf_all = np.stack(simulated_hrtf_list, axis=-1)
        simulated_sh_all = np.stack(simulated_sh_list, axis=-1)
        
        # Split the data along the subject dimension using val_idx.
        if self.val:
            self.measured_hrtf = measured_hrtf_all[..., self.args.val_idx]
            self.measured_sh = measured_sh_all[..., self.args.val_idx]
            self.simulated_hrtf = simulated_hrtf_all[..., self.args.val_idx]
            self.simulated_sh = simulated_sh_all[..., self.args.val_idx]
        else:
            self.measured_hrtf = np.delete(measured_hrtf_all, self.args.val_idx, axis=-1)
            self.measured_sh = np.delete(measured_sh_all, self.args.val_idx, axis=-1)
            self.simulated_hrtf = np.delete(simulated_hrtf_all, self.args.val_idx, axis=-1)
            self.simulated_sh = np.delete(simulated_sh_all, self.args.val_idx, axis=-1)
        
        # Set the number of subjects and fft_length.
        self.num_subjects = self.anthro_head.shape[0]
        self.fft_length = self.measured_hrtf.shape[0]
        # At this point:
        # measured_hrtf: [fft_length, 2, num_freq_bins, num_subjects]
        # measured_sh:   [num_coeffs, num_freq_bins = fft_length, num_subjects]
        # simulated_hrtf and simulated_sh: similar shapes.
    
    def __len__(self):
        # Total number of samples = (# subjects) * (# frequency bins) * 4 (for 4 domain/ear combinations)
        num_freq_bins = self.measured_hrtf.shape[2]
        print("Number of frequency bins:", num_freq_bins)
        return self.num_subjects * num_freq_bins * 4

    def __getitem__(self, idx):
        """
        Returns a tuple:
          (ear_anthro, head_anthro, hrtf, sh, subject, freq, domain_label)
        where:
          - ear_anthro: ear-specific anthropometric vector (left or right)
          - head_anthro: head anthropometric vector
          - hrtf: HRTF magnitude vector for a given frequency bin (shape: [fft_length])
          - sh: SH coefficients matrix for the subject (shape: [num_coeffs, fft_length])
          - subject: subject index (integer)
          - freq: frequency bin index (integer)
          - domain_label: a 4D one-hot vector indicating the data domain:
                [1, 0, 0, 0] for measured left,
                [0, 1, 0, 0] for measured right,
                [0, 0, 1, 0] for simulated left,
                [0, 0, 0, 1] for simulated right.
        """
        num_subjects = self.num_subjects
        num_freq_bins = self.measured_hrtf.shape[2]
        domain = idx // (num_subjects * num_freq_bins)
        new_idx = idx % (num_subjects * num_freq_bins)
        freq = new_idx // num_subjects
        subject = new_idx % num_subjects

        if domain == 0:
            domain_label = np.array([1, 0, 0, 0], dtype=np.float32)  # measured left
            ear_anthro = self.anthro_left[subject]
            hrtf = self.measured_hrtf[:, 0, freq, subject]  # left ear: axis 1 index 0
            sh = self.measured_sh[:, :, subject]            # full SH coefficients matrix for subject
        elif domain == 1:
            domain_label = np.array([0, 1, 0, 0], dtype=np.float32)  # measured right
            ear_anthro = self.anthro_right[subject]
            hrtf = self.measured_hrtf[:, 1, freq, subject]  # right ear: axis 1 index 1
            sh = self.measured_sh[:, :, subject]
        elif domain == 2:
            domain_label = np.array([0, 0, 1, 0], dtype=np.float32)  # simulated left
            ear_anthro = self.anthro_left[subject]
            hrtf = self.simulated_hrtf[:, 0, freq, subject]
            sh = self.simulated_sh[:, :, subject]
        elif domain == 3:
            domain_label = np.array([0, 0, 0, 1], dtype=np.float32)  # simulated right
            ear_anthro = self.anthro_right[subject]
            hrtf = self.simulated_hrtf[:, 1, freq, subject]
            sh = self.simulated_sh[:, :, subject]
        else:
            raise ValueError("Domain index out of range.")
        
        head_anthro = self.anthro_head[subject]
        
        # Convert all outputs to torch tensors.
        ear_anthro = torch.tensor(ear_anthro, dtype=torch.float32)
        head_anthro = torch.tensor(head_anthro, dtype=torch.float32)
        hrtf = torch.tensor(hrtf, dtype=torch.float32)
        sh = torch.tensor(sh, dtype=torch.float32)
        domain_label = torch.tensor(domain_label, dtype=torch.float32)
        
        return ear_anthro, head_anthro, hrtf, sh, subject, freq, domain_label

    def train_test_split(self):
        """
        Returns a dictionary with keys 'train' or 'val', where each entry is a tuple:
          (anthro, measured_hrtf, measured_sh, simulated_hrtf, simulated_sh)
        """
        if self.val:
            return {
                'val': (
                    (self.anthro_head, self.anthro_left, self.anthro_right),
                    self.measured_hrtf,
                    self.measured_sh,
                    self.simulated_hrtf,
                    self.simulated_sh
                )
            }
        else:
            return {
                'train': (
                    (self.anthro_head, self.anthro_left, self.anthro_right),
                    self.measured_hrtf,
                    self.measured_sh,
                    self.simulated_hrtf,
                    self.simulated_sh
                )
            }

#############################################
####### Dataloader Creation ######
def create_dataloader(args):
    train_dataset = HUTUBS_Dataset(args, val=False)
    val_dataset = HUTUBS_Dataset(args, val=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return train_loader, val_loader

#############################################
####### Example Usage ######
class Args:
    anthro_mat_path = '/Users/anishnair/Global_HRTF_VAE/Normalized_Anthropometric_Data.csv'
    measured_hrtf_dir = '/Users/anishnair/Global_HRTF_VAE/Processed_ML/Measured/'
    measured_sh_dir = '/Users/anishnair/Global_HRTF_VAE/Processed_ML/Measured/'
    simulated_hrtf_dir = '/Users/anishnair/Global_HRTF_VAE/Processed_ML/Simulated/'
    simulated_sh_dir = '/Users/anishnair/Global_HRTF_VAE/Processed_ML/Simulated/'
    val_idx = 0  # Use the first subject for validation
    batch_size = 16
    num_workers = 4

if __name__ == '__main__':
    args = Args()
    train_loader, val_loader = create_dataloader(args)
    print("Training set size:", len(train_loader.dataset))
    print("Validation set size:", len(val_loader.dataset))
    
    # Retrieve one batch for demonstration.
    for batch in train_loader:
        ear_anthro, head_anthro, hrtf, sh, subject, freq, domain_label = batch
        print("Batch shapes:")
        print("  ear_anthro:", ear_anthro.shape)
        print("  head_anthro:", head_anthro.shape)
        print("  hrtf:", hrtf.shape)
        print("  sh:", sh.shape)
        print("  domain_label:", domain_label.shape)
        break
