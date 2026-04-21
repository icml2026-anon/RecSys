"""
CDiff4Rec - Collaborative Diffusion for Recommendation (WWW 2025)
Ported to RecBole framework for fair comparison with QualCF

Original paper: https://github.com/Gyu-Seok0/CDiff4Rec_WWW25
"""

import math
import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils.enum_type import InputType


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # predict x_0
    EPSILON = enum.auto()  # predict epsilon


class DNN(nn.Module):
    """Deep neural network for reverse diffusion process."""

    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal"
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError(f"Unimplemented timestep embedding type {self.time_type}")

        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([
            nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])
        ])
        self.out_layers = nn.ModuleList([
            nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])
        ])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in list(self.in_layers) + list(self.out_layers):
            size = layer.weight.size()
            fan_out, fan_in = size[0], size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out, fan_in = size[0], size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)

        for layer in self.in_layers:
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


class CDiff4Rec(GeneralRecommender, AutoEncoderMixin):
    """
    CDiff4Rec: Collaborative Diffusion Model for Recommendation

    Key features:
    - Gaussian diffusion process
    - Pseudo-user generation from item features
    - Collaborative neighbor aggregation
    """

    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super(CDiff4Rec, self).__init__(config, dataset)
        super().build_histroy_items(dataset)

        # Diffusion parameters
        self.steps = config["steps"] if "steps" in config else 50
        self.noise_scale = config["noise_scale"] if "noise_scale" in config else 0.1
        self.noise_min = config["noise_min"] if "noise_min" in config else 0.0001
        self.noise_max = config["noise_max"] if "noise_max" in config else 0.02
        self.noise_schedule = config["noise_schedule"] if "noise_schedule" in config else "linear-var"
        self.sampling_noise = config["sampling_noise"] if "sampling_noise" in config else False
        self.sampling_steps = config["sampling_steps"] if "sampling_steps" in config else 0

        # Model parameters
        self.emb_size = config["emb_size"] if "emb_size" in config else 10
        self.mean_type = config["mean_type"] if "mean_type" in config else "x0"
        self.mean_type = ModelMeanType.START_X if self.mean_type == "x0" else ModelMeanType.EPSILON

        # Collaborative parameters
        self.alpha = config["alpha"] if "alpha" in config else 0.1  # user's own preference
        self.beta = config["beta"] if "beta" in config else 0.8  # real users weight
        self.gamma = config["gamma"] if "gamma" in config else 0.1  # pseudo users weight
        self.tau = config["tau"] if "tau" in config else 1.0  # temperature
        self.topk = config["topk_neighbors"] if "topk_neighbors" in config else 10

        # Build denoiser network
        in_dims = [self.n_items] * 2
        out_dims = [self.n_items] * 2
        self.model = DNN(
            in_dims=in_dims,
            out_dims=out_dims,
            emb_size=self.emb_size,
            time_type="cat",
            norm=False,
            dropout=config["dropout"] if "dropout" in config else 0.5
        )

        # Initialize diffusion process
        self._init_diffusion()

        # Build interaction matrix cache
        self.register_buffer("_inter_mat", torch.empty(0))
        self._inter_mat_built = False

        # Apply initialization
        self.apply(xavier_normal_initialization)

    def _init_diffusion(self):
        """Initialize diffusion process parameters."""
        if self.noise_scale != 0.:
            self.betas = torch.tensor(self._get_betas(), dtype=torch.float32).to(self.device)
            self.betas[0] = 0.00001  # Fix first beta to prevent overfitting

            alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
            self.alphas_cumprod_prev = torch.cat([
                torch.tensor([1.0], dtype=torch.float32).to(self.device), self.alphas_cumprod[:-1]
            ]).to(self.device)

            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
            self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

            # Posterior variance
            self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            self.posterior_log_variance_clipped = torch.log(
                torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
            )
            self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
            )

    def _get_betas(self):
        """Get beta schedule for diffusion."""
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max

        if self.noise_schedule == "linear":
            return np.linspace(start, end, self.steps, dtype=np.float64)
        elif self.noise_schedule == "linear-var":
            # Linear variance schedule
            variance = np.linspace(start, end, self.steps, dtype=np.float64)
            alpha_bar = 1 - variance
            betas = []
            betas.append(1 - alpha_bar[0])
            for i in range(1, self.steps):
                betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))
            return np.array(betas)
        else:
            raise NotImplementedError(f"Unknown beta schedule: {self.noise_schedule}")

    @torch.no_grad()
    def _build_interaction_matrix(self, train_inter_feat=None):
        """Build interaction matrix from training data."""
        if self._inter_mat_built:
            return

        mat = torch.zeros(self.n_users, self.n_items, device=self.device)

        if train_inter_feat is not None:
            user_ids = train_inter_feat[self.USER_ID]
            item_ids = train_inter_feat[self.ITEM_ID]
            mat[user_ids, item_ids] = 1.0
        else:
            # Fallback to history
            ids = self.history_item_id
            val = self.history_item_value
            mask = val > 0
            user_idx = torch.arange(self.n_users, device=self.device).unsqueeze(1).expand_as(ids)
            mat[user_idx[mask], ids[mask]] = 1.0

        self._inter_mat = mat
        self._inter_mat_built = True

    def _get_collaborative_prior(self, user_ids):
        """Get collaborative prior from similar users."""
        if not self._inter_mat_built:
            self._build_interaction_matrix()

        B = user_ids.size(0)
        user_inter = self._inter_mat[user_ids]  # (B, I)

        # Compute user-user similarity (cosine)
        user_norms = self._inter_mat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        inter_normed = self._inter_mat / user_norms

        # Get top-k similar users
        sim = torch.mm(inter_normed[user_ids], inter_normed.t())  # (B, U)
        sim[torch.arange(B, device=sim.device), user_ids] = -1  # Mask self

        topk_sim, topk_idx = sim.topk(self.topk, dim=-1)  # (B, K)
        topk_sim = torch.softmax(topk_sim / self.tau, dim=-1)  # Temperature scaling

        # Aggregate neighbor interactions
        neighbor_inter = self._inter_mat[topk_idx.flatten()].view(B, self.topk, self.n_items)
        collaborative_prior = (neighbor_inter * topk_sim.unsqueeze(-1)).sum(dim=1)  # (B, I)

        # Combine with user's own preference
        prior = self.alpha * user_inter + (1 - self.alpha) * collaborative_prior

        return prior.clamp(min=1e-6, max=1-1e-6)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1).to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1).to(x_start.device)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """Predict x_0 from x_t and epsilon."""
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1).to(x_t.device) * x_t
            - self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1).to(x_t.device) * eps
        )

    def p_mean_variance(self, x_t, t):
        """Compute mean and variance for p(x_{t-1} | x_t)."""
        model_output = self.model(x_t, t)

        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        else:  # EPSILON
            pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)

        pred_xstart = pred_xstart.clamp(-1, 1)

        # Compute posterior mean
        posterior_mean = (
            self.posterior_mean_coef1[t].view(-1, 1).to(x_t.device) * pred_xstart
            + self.posterior_mean_coef2[t].view(-1, 1).to(x_t.device) * x_t
        )
        posterior_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1).to(x_t.device)

        return {
            "mean": posterior_mean,
            "log_variance": posterior_log_variance,
            "pred_xstart": pred_xstart
        }

    def p_sample(self, x_t, t, sampling_noise=False):
        """Sample x_{t-1} from p(x_{t-1} | x_t)."""
        out = self.p_mean_variance(x_t, t)

        if sampling_noise:
            noise = torch.randn_like(x_t)
            nonzero_mask = (t != 0).float().view(-1, 1)
            sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        else:
            sample = out["mean"]

        return sample

    def calculate_loss(self, interaction):
        """Training loss."""
        user = interaction[self.USER_ID]
        x_start = self.get_rating_matrix(user)

        batch_size = x_start.size(0)
        device = x_start.device

        # Sample timesteps uniformly
        t = torch.randint(0, self.steps, (batch_size,), device=device).long()

        # Forward diffusion
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, t, noise)
        else:
            x_t = x_start

        # Predict
        model_output = self.model(x_t, t)

        # Compute loss
        target = x_start if self.mean_type == ModelMeanType.START_X else noise
        mse = mean_flat((target - model_output) ** 2)
        loss = mse.mean()

        return loss

    def full_sort_predict(self, interaction):
        """Inference: reverse diffusion."""
        user = interaction[self.USER_ID]
        x_start = self.get_rating_matrix(user)

        # Get collaborative prior
        prior = self._get_collaborative_prior(user)

        # Start from prior or noise
        if self.sampling_steps == 0:
            x_t = prior
        else:
            t = torch.tensor([self.sampling_steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(prior, t)

        # Reverse diffusion
        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = self.model(x_t, t)
            return x_t

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            x_t = self.p_sample(x_t, t, self.sampling_noise)

        return x_t

    def predict(self, interaction):
        """Predict scores for specific items."""
        item = interaction[self.ITEM_ID]
        scores = self.full_sort_predict(interaction)
        return scores[torch.arange(scores.size(0), device=scores.device), item]
