import torch
import torch.nn as nn

class VoxelDecoder(nn.Module):
    def __init__(self, latent_dim=256, voxel_size=32):
        super().__init__()
        self.voxel_size = voxel_size

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, voxel_size ** 3)
        )

    def forward(self, x):
        x = self.fc(x)
        return x.view(
            -1, 1,
            self.voxel_size,
            self.voxel_size,
            self.voxel_size
        )
