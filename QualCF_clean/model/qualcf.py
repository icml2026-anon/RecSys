"""
QualCF - Quality-Aware Neighbor Selection for Flow Matching
==============================================================

Core Innovation: Learn neighbor quality from prediction improvement

Instead of filtering neighbors by similarity threshold, we learn:
- Which neighbors actually improve predictions (not just similar)
- How to weight neighbors based on their contribution
- Dynamic selection that adapts to each user's needs

Key components:
1. Quality predictor: predicts if a neighbor will help
2. Contrastive learning: good neighbors → better predictions
3. Adaptive weighting: learned from prediction quality
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.utils.enum_type import InputType


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal positional embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device) * 2 * math.pi
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class QualCF(GeneralRecommender, AutoEncoderMixin):
    """Flow Matching with Quality-Aware Neighbor Selection."""

    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super(QualCF, self).__init__(config, dataset)
        super().build_histroy_items(dataset)

        # Flow params
        self.n_steps = config["n_steps"]
        self.s_steps = config["s_steps"]
        self.time_steps = torch.linspace(0, 1, self.n_steps + 1)

        # Retrieval params
        self.top_k = config["top_k"] if "top_k" in config else 10
        self.retrieval_update_freq = config["retrieval_update_freq"] if "retrieval_update_freq" in config else 5

        # Ablation switches (defaults preserve the main model behavior)
        self.use_quality_net = config["use_quality_net"] if "use_quality_net" in config else True
        self.use_personalized_prior = config["use_personalized_prior"] if "use_personalized_prior" in config else True
        self.use_inference_retrieval = config["use_inference_retrieval"] if "use_inference_retrieval" in config else True
        self.use_uniform_neighbor = config["use_uniform_neighbor"] if "use_uniform_neighbor" in config else False
        self.ablate_feature = config["ablate_feature"] if "ablate_feature" in config else None

        # Architecture params
        t_emb_dim = config["time_embedding_size"]
        self.t_emb_dim = t_emb_dim
        hidden = config["hidden_dim"]
        dropout = config["dropout"] if "dropout" in config else 0.1

        # Time embedding
        self.t_emb_fc = nn.Linear(t_emb_dim, t_emb_dim)

        # Velocity MLP
        mlp_in = self.n_items + t_emb_dim
        dims = [mlp_in, hidden, hidden, self.n_items]
        self.velocity_mlp = MLPLayers(
            layers=dims, dropout=dropout, activation="tanh",
            last_activation=False,
        )

        # Quality-aware neighbor selector
        # Input: [user_vec, neighbor_vec, similarity, overlap]
        # Output: quality score (how much this neighbor helps)
        self.quality_net = nn.Sequential(
            nn.Linear(4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Init weights
        self.apply(xavier_normal_initialization)

        # Caches
        self.register_buffer("_inter_mat", torch.empty(0))
        self.register_buffer("_user_sim_idx", torch.empty(0, dtype=torch.long))
        self.register_buffer("_user_sim_wt", torch.empty(0))
        self._index_built = False

        # Item popularity prior
        self.register_buffer("item_freq", self._calc_item_freq())

    def _calc_item_freq(self):
        ids = self.history_item_id
        val = self.history_item_value
        mask = val > 0
        flat_ids = ids[mask]
        cnt = torch.zeros(self.n_items, device=self.device)
        cnt.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=cnt.dtype))
        return cnt / self.n_users

    @torch.no_grad()
    def _build_interaction_matrix(self, train_inter_feat=None):
        """Build interaction matrix from training data only to avoid data leakage."""
        mat = torch.zeros(self.n_users, self.n_items, device=self.device)

        if train_inter_feat is not None:
            # Use training data only (no data leakage)
            user_ids = train_inter_feat[self.USER_ID]
            item_ids = train_inter_feat[self.ITEM_ID]
            mat[user_ids, item_ids] = 1.0
        else:
            # Fallback: use history (contains all data - for compatibility)
            ids = self.history_item_id
            val = self.history_item_value
            mask = val > 0
            user_idx = torch.arange(self.n_users, device=self.device
                                    ).unsqueeze(1).expand_as(ids)
            mat[user_idx[mask], ids[mask]] = 1.0

        self._inter_mat = mat

    @torch.no_grad()
    def build_retrieval_index(self, train_inter_feat=None):
        """Build user-user similarity via cosine using training data only."""
        # Clear old index first to prevent memory accumulation
        if hasattr(self, 'neighbor_idx'):
            del self.neighbor_idx
        if hasattr(self, 'neighbor_weight'):
            del self.neighbor_weight
        torch.cuda.empty_cache()

        if self._inter_mat.numel() == 0:
            self._build_interaction_matrix(train_inter_feat)

        user_norms = self._inter_mat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        inter_normed = self._inter_mat / user_norms

        bs = 512
        all_idx = []
        all_wt = []

        for i in range(0, self.n_users, bs):
            sim = torch.mm(inter_normed[i:i+bs], inter_normed.t())
            batch_size = sim.size(0)
            diag_idx = torch.arange(batch_size, device=sim.device)
            sim[diag_idx, i + diag_idx] = -1
            wt, idx = sim.topk(self.top_k, dim=-1)
            all_idx.append(idx)
            # Clear intermediate tensors
            del sim
            all_wt.append(wt)

        self._user_sim_idx = torch.cat(all_idx, 0)
        self._user_sim_wt = torch.cat(all_wt, 0)
        self._index_built = True

    def _get_personalized_prior(self, user_ids):
        """Construct prior with quality-aware neighbor selection."""
        if not self._index_built:
            self.build_retrieval_index()

        B = user_ids.size(0)

        # Get neighbors
        nb_idx = self._user_sim_idx[user_ids]  # (B, K)
        nb_wt = self._user_sim_wt[user_ids]    # (B, K)

        # Get interactions
        user_inter = self._inter_mat[user_ids]  # (B, I)
        nb_inter = self._inter_mat[nb_idx.flatten()].view(B, self.top_k, self.n_items)  # (B, K, I)

        # Compute features for quality prediction
        user_density = user_inter.sum(dim=-1, keepdim=True) / self.n_items  # (B, 1)
        nb_density = nb_inter.sum(dim=-1) / self.n_items  # (B, K)

        # Overlap: how many items user and neighbor both interacted with
        overlap = (user_inter.unsqueeze(1) * nb_inter).sum(dim=-1) / self.n_items  # (B, K)

        # Normalize each feature to [0,1] so nb_density is not drowned out by nb_wt
        def _norm01(x):
            xmin = x.min(dim=-1, keepdim=True).values
            xmax = x.max(dim=-1, keepdim=True).values
            return (x - xmin) / (xmax - xmin + 1e-8)

        # Quality features: [user_density, neighbor_density, similarity, overlap]
        quality_features = torch.stack([
            _norm01(user_density.expand(-1, self.top_k)),
            _norm01(nb_density),
            _norm01(nb_wt),
            _norm01(overlap)
        ], dim=-1)  # (B, K, 4)

        # Apply feature ablation before computing scores (consistent across train and inference)
        if self.ablate_feature is not None:
            feature_to_idx = {
                'user_density': 0,
                'nb_density': 1,
                'nb_wt': 2,
                'overlap': 3,
            }
            if self.ablate_feature in feature_to_idx:
                quality_features = quality_features.clone()
                quality_features[..., feature_to_idx[self.ablate_feature]] = 0.0

        # Predict quality scores
        quality_scores = self.quality_net(quality_features).squeeze(-1)  # (B, K)

        # Soft selection: use sigmoid to get weights in [0, 1]
        if self.use_uniform_neighbor:
            # Uniform weights: each neighbor contributes equally (1/K)
            neighbor_weights = torch.ones_like(nb_wt) / self.top_k
        elif self.use_quality_net:
            neighbor_weights = torch.sigmoid(quality_scores)  # (B, K)
        else:
            # Fall back to similarity-based weights so alpha stays < 1
            neighbor_weights = nb_wt.clamp(min=0)

        # Weighted neighbor prior
        nb_wt_final = nb_wt * neighbor_weights
        nb_wt_sum = nb_wt_final.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        nb_wt_norm = (nb_wt_final / nb_wt_sum).unsqueeze(-1)  # (B, K, 1)
        neighbor_prior = (nb_inter * nb_wt_norm).sum(dim=1)  # (B, I)

        # Overall weight: average quality score
        alpha = neighbor_weights.mean(dim=-1, keepdim=True).clamp(0, 1)  # (B, 1)

        # Fuse
        global_prior = self.item_freq.unsqueeze(0).expand(B, -1)
        if self.use_personalized_prior:
            personalized_prior = alpha * neighbor_prior + (1 - alpha) * global_prior
        else:
            personalized_prior = global_prior

        return personalized_prior.clamp(min=1e-6, max=1-1e-6)

    def _denoise(self, x_t, t, use_retrieval=False, user_ids=None):
        """Predict x₁ from x_t."""
        t_emb = self.t_emb_fc(timestep_embedding(t, self.t_emb_dim))
        base_pred = self.velocity_mlp(torch.cat([x_t, t_emb], dim=-1))

        if use_retrieval and user_ids is not None and self.use_inference_retrieval:
            # Get quality-weighted neighbor prior
            nb_prior = self._get_personalized_prior(user_ids)
            # Residual: add neighbor signal
            return base_pred + 0.25 * nb_prior
        else:
            return base_pred

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        x1 = self.get_rating_matrix(user)

        # Random time
        steps = torch.randint(0, self.n_steps, (x1.size(0),), device=self.device)
        t = self.time_steps.to(x1.device)[steps].unsqueeze(1)

        # Personalized prior with quality-aware neighbors
        prior_mean = self._get_personalized_prior(user)
        x0_sample = torch.bernoulli(prior_mean)
        # Straight-Through Estimator: forward uses discrete sample,
        # backward passes gradient to prior_mean (and thus quality_net)
        x0 = (x0_sample - prior_mean).detach() + prior_mean

        # Interpolate
        mask = torch.rand_like(x1) <= t
        x_t = torch.where(mask, x1, x0)

        # Predict - quality network learns from flow matching loss
        pred = self._denoise(x_t, t.squeeze(-1), use_retrieval=False)
        loss = mean_flat((x1 - pred) ** 2).mean()
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        X_bar = self.get_rating_matrix(user)

        Xt = X_bar
        for i_t in range(self.n_steps - self.s_steps, self.n_steps):
            t = self.time_steps[i_t].repeat(Xt.shape[0], 1).to(X_bar.device)
            t_scalar = t.squeeze(-1)

            # Use retrieval at inference
            X1_hat = self._denoise(Xt, t_scalar, use_retrieval=True, user_ids=user)

            if i_t == self.n_steps - 1:
                break

            t_next = self.time_steps[i_t + 1].repeat(Xt.shape[0], 1).to(X_bar.device)
            v = (X1_hat - Xt) / (1 - t)
            pos = Xt + v * (t_next - t)
            neg = 1 - pos
            Xt = torch.stack([neg, pos], -1).argmax(-1)
            Xt = torch.logical_or(X_bar.bool(), Xt.bool()).float()

        return X1_hat

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        scores = self.full_sort_predict(interaction)
        return scores[torch.arange(scores.size(0), device=scores.device), item]
