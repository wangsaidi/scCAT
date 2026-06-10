import torch
from torch import nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    Minimal encoder that maps the input feature space to the latent embedding z.
    The PDF defines the loss functions and triplet construction, but does not
    specify the encoder architecture, so this implementation uses a lightweight
    MLP to avoid adding extra objectives or architectural assumptions.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.network(x)

        z = self.network(x)
        return F.normalize(z, p=2, dim=1)
