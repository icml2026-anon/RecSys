"""
GiffCF - Graph Signal Diffusion for Collaborative Filtering
Ported to RecBole framework for fair comparison with QualCF

Original paper: https://github.com/VinciZhu/GiffCF
Uses graph signal processing with heat equation simulation
"""

import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils.enum_type import InputType


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbed(nn.Module):
    """Time embedding module."""

    def __init__(self, hidden_dim, out_dim, activation='swish'):
        super().__init__()
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.SiLU() if activation == 'swish' else nn.ReLU()

    def forward(self, t_emb):
        e = self.activation(self.hidden(t_emb))
        return self.out(e)


class SimpleMixer(nn.Module):
    """Simple mixer for combining features."""

    def __init__(self, n_inputs, hidden_dim, activation='swish'):
        super().__init__()
        self.hidden = nn.Linear(n_inputs, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.activation = nn.SiLU() if activation == 'swish' else nn.ReLU()

    def forward(self, inputs):
        # inputs: list of tensors
        x = torch.stack(inputs, dim=-1)  # (B, D, N)
        x = self.activation(self.hidden(x))
        x = self.out(x)
        return x.squeeze(-1)


class Denoiser(nn.Module):
    """Denoiser network for GiffCF."""

    def __init__(self, n_items, embed_dim=200, activation='swish', norm_ord=1, n_steps=10):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.norm_ord = norm_ord

        # Item embeddings
        self.item_embed = nn.Parameter(torch.empty(n_items, embed_dim))
        nn.init.xavier_uniform_(self.item_embed)

        # Time embeddings
        self.t_embed1 = TimeEmbed(20, 1, activation)
        self.t_embed2 = TimeEmbed(20, 1, activation)

        # Mixers
        self.embed_mixer = SimpleMixer(3, 2, activation)
        self.score_mixer = SimpleMixer(4, 2, activation)

    def norm(self, c):
        """Compute norm for conditioning."""
        if self.norm_ord is not None:
            norm = torch.norm(c, p=self.norm_ord, dim=-1, keepdim=True)
            return torch.maximum(norm, torch.ones_like(norm))
        else:
            return 1.0

    def forward(self, z_t, c, Ac, t):
        """
        Args:
            z_t: noisy input (B, I)
            c: condition (B, I)
            Ac: smoothed condition (B, I)
            t: timestep (B,)
        """
        # Time embeddings
        t_emb = timestep_embedding(t, 20)
        t_embed1 = self.t_embed1(t_emb).repeat(1, self.embed_dim)  # (B, D)
        t_embed2 = self.t_embed2(t_emb).repeat(1, self.n_items)  # (B, I)

        # Embed inputs
        z_embed = torch.matmul(z_t / self.n_items, self.item_embed)  # (B, D)
        c_embed = torch.matmul(c / self.norm(c), self.item_embed)  # (B, D)

        # Mix embeddings
        x_embed = self.embed_mixer([z_embed, c_embed, t_embed1])  # (B, D)

        # Project to item space
        x_mid = torch.matmul(x_embed, self.item_embed.t())  # (B, I)

        # Mix scores
        x_pred = self.score_mixer([x_mid, c, Ac, t_embed2])  # (B, I)

        return x_pred


class GiffCF(GeneralRecommender, AutoEncoderMixin):
    """
    GiffCF: Graph Signal Diffusion for Collaborative Filtering

    Uses graph signal processing with heat equation simulation
    for smoothing and sharpening user preferences.
    """

    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super(GiffCF, self).__init__(config, dataset)
        super().build_histroy_items(dataset)

        # Model parameters
        self.embed_dim = config["embed_dim"] if "embed_dim" in config else 200
        self.dropout = config["dropout"] if "dropout" in config else 0.5
        self.norm_ord = config["norm_ord"] if "norm_ord" in config else 1
        self.T = config["T"] if "T" in config else 3
        self.alpha = config["alpha"] if "alpha" in config else 1.5
        self.ideal_weight = config["ideal_weight"] if "ideal_weight" in config else 0.0
        self.ideal_cutoff = config["ideal_cutoff"] if "ideal_cutoff" in config else 200
        self.noise_decay = config["noise_decay"] if "noise_decay" in config else 1.0
        self.noise_scale = config["noise_scale"] if "noise_scale" in config else 0.0

        # Build denoiser
        self.denoiser = Denoiser(
            self.n_items, self.embed_dim, 'swish', self.norm_ord, self.T
        )

        # Build adjacency matrices
        self._build_adjacency_matrices(dataset)

        # Compute eigendecomposition for ideal filter
        if self.ideal_weight > 0:
            self._compute_eigen()

        # Timesteps
        self.t = torch.linspace(0, self.T, self.T + 1, dtype=torch.long)

        # Apply initialization
        self.apply(xavier_normal_initialization)

    def _build_adjacency_matrices(self, dataset):
        """Build normalized adjacency matrices."""
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # Compute degrees
        user_deg = np.array(interaction_matrix.sum(axis=1)).flatten()
        item_deg = np.array(interaction_matrix.sum(axis=0)).flatten()

        # Normalize: D_u^{-1/4} * A * D_i^{-1/2}
        user_deg_inv = np.power(user_deg + 1e-7, -0.25)
        item_deg_inv = np.power(item_deg + 1e-7, -0.5)

        # Build normalized adjacency
        row, col = interaction_matrix.row, interaction_matrix.col
        data = user_deg_inv[row] * item_deg_inv[col]

        adj_right = sp.coo_matrix(
            (data, (row, col)),
            shape=(self.n_users, self.n_items)
        ).astype(np.float32)

        # Convert to sparse tensors
        self.adj_right = self._sparse_mx_to_torch_sparse_tensor(adj_right).to(self.device)
        self.adj_left = self._sparse_mx_to_torch_sparse_tensor(adj_right.T).to(self.device)

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert scipy sparse matrix to torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def _compute_eigen(self):
        """Compute eigendecomposition for ideal filter."""
        adj_right_dense = self.adj_right.to_dense().cpu().numpy()

        if self.ideal_weight == 0.0:
            cutoff = 1
        else:
            cutoff = min(self.ideal_cutoff, min(adj_right_dense.shape) - 1)

        try:
            _, values, vectors = svds(adj_right_dense, k=cutoff)
            # Sort by eigenvalue magnitude
            idx = np.argsort(np.abs(values))[::-1]
            values = values[idx]
            vectors = vectors[idx, :]

            self.eigen_val = torch.tensor(values, dtype=torch.float32).to(self.device)
            self.eigen_vec = torch.tensor(vectors, dtype=torch.float32).to(self.device)
        except:
            # Fallback if SVD fails
            self.eigen_val = torch.ones(1).to(self.device)
            self.eigen_vec = torch.zeros(1, self.n_items).to(self.device)

    def prop(self, x):
        """Graph propagation: A^T * x * A."""
        x_prop = torch.sparse.mm(self.adj_right, x.t())  # (I, B)
        x_prop = torch.sparse.mm(self.adj_left, x_prop)  # (U, B)
        x_prop = x_prop.t()  # (B, U) -> (B, I) if x is (B, I)
        return x_prop / (self.eigen_val[0] if hasattr(self, 'eigen_val') else 1.0)

    def ideal(self, x):
        """Ideal low-pass filter using eigendecomposition."""
        if not hasattr(self, 'eigen_vec'):
            return x

        x_ideal = torch.matmul(x, self.eigen_vec.t())
        x_ideal = torch.matmul(x_ideal, self.eigen_vec)
        return x_ideal

    def smooth(self, x):
        """Smooth signal using graph filter."""
        if self.ideal_weight > 0:
            x_smooth = self.prop(x) + self.ideal_weight * self.ideal(x)
            return x_smooth / (1 + self.ideal_weight)
        else:
            return self.prop(x)

    def filter(self, x, Ax, t):
        """Apply time-dependent filter."""
        t_float = t.float().unsqueeze(1)
        return x + self.alpha * t_float / self.T * (Ax - x)

    def sigma(self, t):
        """Noise schedule."""
        t_float = t.float().unsqueeze(1)
        return self.noise_scale * (self.noise_decay ** (self.T - t_float))

    def denoise(self, z_t, c, Ac, t):
        """Denoise step."""
        x_pred = self.denoiser(z_t, c, Ac, t)
        return x_pred

    def calculate_loss(self, interaction):
        """Training loss."""
        user = interaction[self.USER_ID]
        x = self.get_rating_matrix(user)

        # Random timestep
        t = torch.randint(1, self.T + 1, (x.size(0),), device=self.device)

        # Smooth
        Ax = self.smooth(x)

        # Forward diffusion
        z_t = self.filter(x, Ax, t)

        # Add noise
        if self.noise_scale > 0.0:
            eps = torch.randn_like(x)
            z_t = z_t + self.sigma(t) * eps

        # Condition with dropout
        c = F.dropout(x, p=self.dropout, training=True)
        Ac = self.smooth(c)

        # Denoise
        x_pred = self.denoise(z_t, c, Ac, t)

        # MSE loss
        loss = F.mse_loss(x, x_pred)

        return loss

    def full_sort_predict(self, interaction):
        """Inference: reverse diffusion."""
        user = interaction[self.USER_ID]
        x = self.get_rating_matrix(user)

        # Smooth
        Ax = self.smooth(x)

        # Start from filtered state
        z_t = self.filter(x, Ax, self.t[-1].repeat(x.size(0)).to(self.device))

        # Reverse diffusion
        for i in range(len(self.t) - 1, 0, -1):
            t_curr = self.t[i].repeat(x.size(0)).to(self.device)
            t_prev = self.t[i - 1].repeat(x.size(0)).to(self.device)

            # Denoise
            x_pred = self.denoise(z_t, x, Ax, t_curr)

            # Smooth prediction
            Ax_pred = self.smooth(x_pred)

            # Reverse step
            z_s_pred = self.filter(x_pred, Ax_pred, t_prev)

            if self.noise_decay > 0.0:
                z_t_pred = self.filter(x_pred, Ax_pred, t_curr)
                decay_factor = self.noise_decay ** (t_curr - t_prev).float().unsqueeze(1)
                z_t = z_s_pred + decay_factor * (z_t - z_t_pred)
            else:
                z_t = z_s_pred

        return z_t

    def predict(self, interaction):
        """Predict scores for specific items."""
        item = interaction[self.ITEM_ID]
        scores = self.full_sort_predict(interaction)
        return scores[torch.arange(scores.size(0), device=scores.device), item]
