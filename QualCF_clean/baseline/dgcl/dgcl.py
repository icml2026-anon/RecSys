"""
DGCL - Diffusion-augmented Graph Contrastive Learning
Ported to RecBole framework for fair comparison with QualCF

Original: https://github.com/huangfan0/DGCL
Based on LightGCN backbone with diffusion-based augmentation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule for diffusion."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    """Linear schedule for diffusion."""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = (self.dim // 2) + 1
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb[:, :self.dim]


class Block(nn.Module):
    """Basic block with time conditioning."""

    def __init__(self, in_ft, out_ft):
        super(Block, self).__init__()
        self.lin = nn.Linear(in_ft, out_ft)
        self.time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(out_ft, out_ft * 2)
        )

    def forward(self, h, t):
        t = self.time(t)
        scale, shift = t.chunk(2, dim=1)
        h = (scale + 1) * h + shift
        return h


class DiffusionEncoder(nn.Module):
    """Diffusion encoder for augmentation."""

    def __init__(self, in_ft, out_ft):
        super(DiffusionEncoder, self).__init__()
        self.l1 = Block(in_ft, out_ft)
        self.l2 = Block(out_ft, out_ft)

        sinu_pos_emb = SinusoidalPosEmb(out_ft)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(out_ft, out_ft),
            nn.GELU(),
            nn.Linear(out_ft, out_ft)
        )

    def forward(self, h, t):
        t = self.time_mlp(t)
        h = self.l1(h, t)
        h = self.l2(h, t)
        return h


class DiffusionModule(nn.Module):
    """Diffusion module for graph augmentation."""

    def __init__(self, emb_size, timesteps=10):
        super(DiffusionModule, self).__init__()
        self.encoder = DiffusionEncoder(emb_size, emb_size)
        self.timesteps = timesteps

        # Define beta schedule
        self.betas = linear_beta_schedule(timesteps=self.timesteps)

        # Define alphas
        self.alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        """Training loss."""
        if noise is None:
            noise = torch.randn_like(x_start)

        target = x_start
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.encoder(x_noisy, t)

        loss = F.mse_loss(target, predicted_noise)
        return loss

    def p_sample(self, x, t, t_index):
        """Reverse diffusion step."""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        predicted_noise = self.encoder(x, t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape):
        """Sample from diffusion model."""
        device = next(self.encoder.parameters()).device
        b = shape[0]
        emb = torch.randn(shape, device=device)

        for i in reversed(range(0, self.timesteps)):
            emb = self.p_sample(emb, torch.full((b,), i, device=device, dtype=torch.long), i)

        return emb

    def forward(self, input):
        """Forward pass for training."""
        device = input.device
        t = torch.randint(0, self.timesteps, (input.shape[0],), device=device).long()
        return self.p_losses(input, t)


class DGCL(GeneralRecommender):
    """
    DGCL: Diffusion-augmented Graph Contrastive Learning

    Combines LightGCN backbone with diffusion-based augmentation
    and contrastive learning.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DGCL, self).__init__(config, dataset)

        # Load parameters
        self.embedding_size = config["embedding_size"]
        self.n_layers = config["n_layers"] if "n_layers" in config else 3
        self.reg_weight = config["reg_weight"] if "reg_weight" in config else 1e-4
        self.cl_rate = config["cl_rate"] if "cl_rate" in config else 0.2
        self.timesteps = config["timesteps"] if "timesteps" in config else 10
        self.temperature = config["temperature"] if "temperature" in config else 0.2

        # User and item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Diffusion module
        self.diffusion = DiffusionModule(self.embedding_size, self.timesteps)

        # Build interaction matrix and adjacency
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # Loss functions
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Initialize
        self.apply(xavier_uniform_initialization)

    def get_norm_adj_mat(self):
        """Get normalized adjacency matrix."""
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()

        # Build adjacency matrix
        for i in range(inter_M.nnz):
            A[inter_M.row[i], inter_M.col[i] + self.n_users] = 1
        for i in range(inter_M_t.nnz):
            A[inter_M_t.row[i] + self.n_users, inter_M_t.col[i]] = 1

        # Normalize
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        # Convert to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def forward(self, perturbed=False):
        """Forward propagation through GCN layers."""
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)

            # Apply diffusion augmentation if perturbed
            if perturbed:
                # Apply small diffusion perturbation to embeddings
                device = all_embeddings.device
                t = torch.randint(0, self.diffusion.timesteps, (all_embeddings.shape[0],), device=device).long()
                noise = torch.randn_like(all_embeddings) * 0.1  # Small noise scale
                all_embeddings = self.diffusion.q_sample(all_embeddings, t, noise)

            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings

    def info_nce_loss(self, view1, view2, temperature):
        """InfoNCE contrastive loss."""
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)

        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        """Calculate total loss."""
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Get embeddings
        user_all_embeddings, item_all_embeddings = self.forward(perturbed=False)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # BPR loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # Regularization loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        # Contrastive loss
        user_view1, item_view1 = self.forward(perturbed=True)
        user_view2, item_view2 = self.forward(perturbed=True)

        u_idx = torch.unique(user)
        i_idx = torch.unique(torch.cat([pos_item, neg_item]))

        user_cl_loss = self.info_nce_loss(user_view1[u_idx], user_view2[u_idx], self.temperature)
        item_cl_loss = self.info_nce_loss(item_view1[i_idx], item_view2[i_idx], self.temperature)
        cl_loss = user_cl_loss + item_cl_loss

        # Total loss
        loss = mf_loss + self.reg_weight * reg_loss + self.cl_rate * cl_loss

        return loss

    def predict(self, interaction):
        """Predict scores for user-item pairs."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward(perturbed=False)
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items."""
        user = interaction[self.USER_ID]

        user_all_embeddings, item_all_embeddings = self.forward(perturbed=False)
        u_embeddings = user_all_embeddings[user]
        scores = torch.matmul(u_embeddings, item_all_embeddings.transpose(0, 1))

        return scores.view(-1)
