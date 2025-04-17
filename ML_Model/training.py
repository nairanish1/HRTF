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
