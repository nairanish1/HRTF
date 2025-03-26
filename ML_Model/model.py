import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import e3nn.o3 as o3
from holographic_vae import nn
#Importing SO(3) Equivariant Layers 
from holographic_vae.nn.linearity import SO3_linearity
from holographic_vae.nn.nonlinearity import TP_nonlinearity
#Importing Normalization Layers
from holographic_vae.nn.normalization import magnitudes_norm, signal_norm, layer_norm_nonlinearity, batch_norm
from holographic_vae.so3.functional import make_vec
from typing import Dict, Tuple

### Develop the model for the SO(3) equivariant VAE 


## Part (1) Construct the SHEquivariant Encoder by using Clebsch-Gordon coefficients (respecting the SO(3) symmetry)
## 1: Linear Layer
## 2: Efficient Tensor Product (Non-Linearity)
## 3: Batch Normalization 
## 4: Signal Normalization 
## 5: Extract mean and variance from the SHEquivariant features
class SHEquivariantEncoder(nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, w3j_matrices: Dict[int, torch.Tensor], latent_dim: int):
        super(SHEquivariantEncoder, self).__init__()
        # SO(3)-equivariant linear layer 
        self.linear = SO3_linearity(
            irreps_in = irreps_in, 
            irreps_out = irreps_out, 
            weights_initializer= torch.nn.init.xavier_uniform_,
            bias = True
        )
        # SO(3)-equivariant non-linear layer 
        self.nonlinear = TP_nonlinearity(
            irreps_in = irreps_in, 
            w3j_matrices = w3j_matrices, 
            filter_ir_out = None, 
            ls_rule = 'full',
            channel_rule = 'full',
            filter_symmetric = True
        )
        # Normalization layers 
        self.magnitudes_norm = magnitudes_norm(irreps_in)
        self.signal_norm = signal_norm(irreps_in)
        self.layer_norm = layer_norm_nonlinearity(irreps_in)
        self.batch_norm = batch_norm(irreps_in)

        self.fc = nn.Linear(irreps_in.dim, latent_dim*2)
## Create the forward pass for the SHEquivariant Encoder
    def forward(self, x: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        #Passing through the SO(3) linear layer
        x = self.linear(x)
        #Passing through the SO(3) non-linear layer 
        x = self.nonlinear(x)
        #Passing through the sequential normalization layer 
        x = self.magnitudes_norm(x)
        x = self.signal_norm(x)
        x = self.layer_norm(x)
        x = self.batch_norm(x)
        #Flatten the dictionary of tensors into a single feature vector per sample
        features = make_vec(x)
        #Map the features into its latent parameters (mean and variance)
        latent_params = self.fc(features)
        mu, logvar = latent_params.chunk(2, dim=-1)
        return mu, logvar
# Example usage (with dummy data for demonstration):
if __name__ == '__main__':
    # Dummy irreps for demonstration; replace with actual irreps as needed.
    irreps_in = o3.Irreps("1x0e + 1x1e")   # Example: 1 scalar (l=0) and 1 vector (l=1, 3 components)
    irreps_out = o3.Irreps("1x0e + 1x1e")  # Same structure for output.
    w3j_matrices = {}  # Replace with actual precomputed Wigner 3j matrices (keys like (l1, l2, l3))
    latent_dim = 128

    encoder = SHEquivariantEncoder(irreps_in, irreps_out, w3j_matrices, latent_dim)
    
    # Create dummy input as a dictionary:
    # For l = 0: tensor shape (batch, multiplicity, 1)
    # For l = 1: tensor shape (batch, multiplicity, 3)
    dummy_input = {
        0: torch.randn(16, 1, 1),
        1: torch.randn(16, 1, 3)
    }
    
    mu, logvar = encoder(dummy_input)
    print("Latent mean shape:", mu.shape)      # Expected: (16, 128)
    print("Latent logvar shape:", logvar.shape)  # Expected: (16, 128)
    






## Part (2) Constuct the Anthropomorphic Encoder 
## 1: Fully connected layers with RELU Activation 
## (Could use dropout or batch normalization based on how big is the architecture)


## Part(3) Feature Fusion 
## 1: Creating FilM layers for the anthropomorphic and SHEquivariant features to generate modulation parameters 
# (scaling and shifting) that adjust the SH features directly.

## Part (4): Latent Sampling and Reparameterization
## 1: Reparameterization Trick
## 2: Latent Space Sampling
## 3: Latent Space Sampling with Gaussian Distribution

## Part (5): Decoder which follows SHEquivariant Encoder structure.

## Part (6): Loss Function and Inverse SHT









