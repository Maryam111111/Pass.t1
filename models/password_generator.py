import sys
import os
import torch.nn as nn

# Add the parent directory to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
VOCAB_SIZE = 100

import config
class PasswordGenerator(nn.Module):
    def __init__(self, latent_dim=None):
        super().__init__()
        latent_dim = latent_dim if latent_dim is not None else config.LATENT_DIM

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.MAX_LEN * VOCAB_SIZE)
        )
