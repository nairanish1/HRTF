import os
import sys
print("Python executable:", sys.executable)
print("sys.path:", sys.path)
sys.path.append('/Users/anishnair/Global_HRTF_VAE/Holographic-VAE')
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import e3nn.o3 as o3
#Importing SO(3) Equivariant Layers 
from holographic_vae.nn.linearity import SO3_linearity
from holographic_vae.nn.nonlinearity import TP_nonlinearity
#Importing Normalization Layers
from holographic_vae.nn.normalization import magnitudes_norm, signal_norm, layer_norm_nonlinearity, batch_norm
from holographic_vae.so3.functional import make_vec, make_dict
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
            filter_ir_out=[],
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
## Part (2) Constuct the Anthropomorphic Encoder 
## 1: Fully connected layers with RELU Activation 
## (Could use dropout or batch normalization based on how big is the architecture)
class AnthropometricEncoder(nn.Module):
    def __init__(self, input_dim: int = 37, hidden_dims: Tuple[int,...] = (64,64), output_dim: int = 32):
        super(AnthropometricEncoder, self).__init__()
        layers = []
        prev_dim = input_dim 
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
## Part(3) Feature Fusion 
## 1: Creating FilM layers for the anthropomorphic and SHEquivariant features to generate modulation parameters 
# (scaling and shifting) that adjust the SH features directly.
class FiLM(nn.Module):
    def __init__(self, sh_feature_dim: int, anthropometric_feature_dim: int):
        super(FiLM, self).__init__()
        #Two linear layers to predict the scaling and shifting from the anthropometric parameters (gamma scales and beta shifts)
        self.fc_gamma = nn.Linear(anthropometric_feature_dim, sh_feature_dim)
        self.fc_beta = nn.Linear(anthropometric_feature_dim, sh_feature_dim)

    def forward(self, sh_features: torch.Tensor, anthropometric_features: torch.Tensor) -> torch.Tensor:
        #Compute the scaling and shifting parameters
        scale = self.fc_gamma(anthropometric_features)
        shift = self.fc_beta(anthropometric_features)
        #Apply the feature-wise linear modulation
        return sh_features * scale + shift
## Part (4): Latent Sampling and Reparameterization
## 1: Reparameterization Trick
## 2: Latent Space Sampling
## 3: Latent Space Sampling with Gaussian Distribution
def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std
## Part (5): Decoder which follows SHEquivariant Encoder structure.
class SHEquivariantDecoder(nn.Module):
    def __init__(self, latent_dim: int, irreps_out: o3.Irreps, w3j_matrices: Dict[int, torch.Tensor]):
        super(SHEquivariantDecoder, self).__init__()
        # Fully connected layer to map the latent features to the equivariant feature space 
        # Here we assume the decoder starts from a feature dimension matching irreps_out.dim.
        self.fc = nn.Linear(latent_dim, irreps_out.dim)
        
        # SH Equivariant block to map the features to output SH space
        self.linear = SO3_linearity(
            irreps_in = irreps_out, 
            irreps_out = irreps_out, 
            weights_initializer= torch.nn.init.xavier_uniform_,
            bias = True
        )

        # SO(3)-equivariant non-linear layer 
        self.nonlinear = TP_nonlinearity(
            irreps_in = irreps_out, 
            w3j_matrices = w3j_matrices,
            filter_ir_out=[],
            ls_rule = 'full',
            channel_rule = 'full',
            filter_symmetric = True
        )
        # Normalization layers
        self.mag_norm = magnitudes_norm(irreps_out)
        self.sig_norm = signal_norm(irreps_out)
        self.layer_norm = layer_norm_nonlinearity(irreps_out)
        self.batch_norm = batch_norm(irreps_out)
   
    def forward(self, z: torch.Tensor) -> Dict[int, torch.Tensor]:
        #Map latent vector to the Equivariant feature space 
        features = self.fc(z) # [batch, irreps_out.dim]
        #Converted flattened features to a dictionary using the appropriate inverse irreps_out using make_dict
        x = make_dict(features, self.linear.irreps_out)
        # Process through an equivariant block.
        x = self.linear(x)
        x = self.nonlinear(x)
        x = self.mag_norm(x)
        x = self.sig_norm(x)
        x = self.layer_norm(x)
        x = self.batch_norm(x)
        return x
    
## Part (6): Full VAE Model 
class HRTF_VAE(nn.Module):
    def __init__(self, 
                 sh_irreps_in: o3.Irreps,
                 sh_irreps_hidden: o3.Irreps, 
                 w3j_matrices: Dict[int, torch.Tensor],
                 latent_dim: int, 
                 anthro_input_dim: int = 37):
         """
        A complete SO(3)-equivariant VAE for HRTF personalization.
        It includes:
         - A spherical harmonic encoder (SHEquivariantEncoder)
         - An anthropomorphic encoder (AnthropomorphicEncoder)
         - A FiLM fusion module
         - Latent sampling via reparameterization
         - A spherical harmonic decoder (SHEquivariantDecoder)
        """
         super(HRTF_VAE, self).__init__()
         self.sh_encoder = SHEquivariantEncoder(sh_irreps_in, sh_irreps_hidden, w3j_matrices, latent_dim)
         self.anthro_encoder = AnthropometricEncoder(input_dim=anthro_input_dim, hidden_dims=(64, 64), output_dim=32)
         self.film = FiLM(sh_feature_dim=sh_irreps_in.dim, anthro_feature_dim=32)
         self.fc = nn.Linear(sh_irreps_in.dim + 32, latent_dim * 2)
         self.sh_decoder = SHEquivariantDecoder(latent_dim, sh_irreps_hidden, w3j_matrices)

    def forward(self, sh_input: Dict[int, torch.Tensor], anthro_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get SH features (flattened) from the SH encoder branch.
        sh_features = self.sh_encoder(sh_input)
        # Encode anthropomorphic data.
        anthro_feat = self.anthro_encoder(anthro_input)  # shape: (batch, 32)
        # Fuse the features via FiLM.
        fused_features = self.film(sh_features, anthro_feat)
        # Map fused features to latent parameters.
        latent_params = self.fc(fused_features)  # shape: (batch, latent_dim * 2)
        mu, logvar = latent_params.chunk(2, dim=1)
        # Reparameterize.2
        z = reparameterize(mu, logvar)
        # Decode latent vector.
        decoded = self.sh_decoder(z)
        return decoded, mu, logvar
    
    
    
    

    









        












