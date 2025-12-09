"""
Lightning Indexer for DeepSeek Sparse Attention.

The Lightning Indexer computes index scores between query and key tokens
to determine which tokens should be selected for sparse attention.

From the DeepSeek-V3.2-Exp paper (Equation 1):

    I_{t,s} = sum_{j=1}^{H^I} w^I_{t,j} * ReLU(q^I_{t,j} . k^I_s)

Where:
- H^I is the number of indexer heads
- q^I_{t,j} and w^I_{t,j} are derived from the query token h_t
- k^I_s is derived from the preceding token h_s
- ReLU is used as activation for throughput consideration

For pupil segmentation, we adapt this for 2D spatial tokens where:
- The pupil is spatially constrained to the eye region
- Local context is critical for precise boundary detection
- We use a lightweight indexer optimized for image patches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LightningIndexer(nn.Module):
    """
    Lightning Indexer for computing sparse attention index scores.

    This module efficiently computes which key tokens are most relevant
    for each query token, enabling sparse attention computation.

    For pupil segmentation:
    - Uses spatial locality bias (pupil pixels attend to nearby pixels)
    - Small number of heads for efficiency
    - ReLU activation for fast computation

    Args:
        dim: Input embedding dimension
        num_heads: Number of indexer heads (H^I in paper)
        key_dim: Dimension of indexer keys/queries (d^I in paper)
        use_rope: Whether to apply partial RoPE to indexer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        key_dim: int = 8,
        use_rope: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.use_rope = use_rope
        self.scale = key_dim ** -0.5

        # Query projections: q^I_{t,j} and weights w^I_{t,j}
        # Output: num_heads * (key_dim + 1) for query vectors + scalar weights
        self.query_proj = nn.Linear(dim, num_heads * (key_dim + 1), bias=False)

        # Key projection: k^I_s
        self.key_proj = nn.Linear(dim, key_dim, bias=False)

        # Optional: Learned position embedding for 2D spatial awareness
        # This helps the indexer understand spatial relationships in images
        self.spatial_bias = nn.Parameter(torch.zeros(1, 1, 1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following DeepSeek initialization."""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.spatial_bias)

    def _apply_rope_2d(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Apply 2D sinusoidal position embedding for spatial tokens.

        For large images (640x400 -> 160x100 = 16k tokens after patch embed),
        we use efficient additive sinusoidal embeddings that scale well.

        Args:
            x: Input tensor (B, N, D) where N = H * W
            height: Height in patches (e.g., 100 for 400px with 4x downsample)
            width: Width in patches (e.g., 160 for 640px with 4x downsample)

        Returns:
            Position-encoded tensor (B, N, D)
        """
        if not self.use_rope:
            return x

        B, N, D = x.shape
        device = x.device

        # For large spatial dimensions, use efficient sinusoidal encoding
        # Split D into half for y-axis and half for x-axis encoding

        # Generate normalized position indices [0, 1]
        y_idx = torch.arange(height, device=device).float() / max(height - 1, 1)
        x_idx = torch.arange(width, device=device).float() / max(width - 1, 1)

        # Create 2D grid and flatten
        y_grid, x_grid = torch.meshgrid(y_idx, x_idx, indexing='ij')
        y_flat = y_grid.reshape(-1)  # (N,)
        x_flat = x_grid.reshape(-1)  # (N,)

        # Frequency bands for each axis (half of D each)
        d_half = D // 2
        # Use geometric spacing for frequencies
        freq_y = torch.exp(torch.arange(0, d_half, 2, device=device).float() *
                          -(math.log(10000.0) / d_half))
        freq_x = torch.exp(torch.arange(0, d_half, 2, device=device).float() *
                          -(math.log(10000.0) / d_half))

        # Compute sin/cos embeddings for y
        y_angles = y_flat.unsqueeze(-1) * freq_y.unsqueeze(0) * height  # Scale by height
        pos_y = torch.zeros(N, d_half, device=device)
        pos_y[:, 0::2] = torch.sin(y_angles)
        pos_y[:, 1::2] = torch.cos(y_angles[:, :pos_y[:, 1::2].size(-1)])

        # Compute sin/cos embeddings for x
        x_angles = x_flat.unsqueeze(-1) * freq_x.unsqueeze(0) * width  # Scale by width
        pos_x = torch.zeros(N, d_half, device=device)
        pos_x[:, 0::2] = torch.sin(x_angles)
        pos_x[:, 1::2] = torch.cos(x_angles[:, :pos_x[:, 1::2].size(-1)])

        # Concatenate y and x embeddings
        pos_emb = torch.cat([pos_y, pos_x], dim=-1)  # (N, D)

        # Handle odd dimensions
        if pos_emb.size(-1) < D:
            pos_emb = F.pad(pos_emb, (0, D - pos_emb.size(-1)))
        elif pos_emb.size(-1) > D:
            pos_emb = pos_emb[:, :D]

        # Add positional encoding (broadcasts across batch)
        return x + pos_emb.unsqueeze(0)

    def forward(
        self,
        query_tokens: torch.Tensor,
        key_tokens: torch.Tensor,
        height: int = None,
        width: int = None,
    ) -> torch.Tensor:
        """
        Compute index scores for sparse attention selection.

        Args:
            query_tokens: Query token embeddings (B, N_q, D)
            key_tokens: Key token embeddings (B, N_k, D)
            height: Height in patches (for 2D RoPE)
            width: Width in patches (for 2D RoPE)

        Returns:
            Index scores (B, N_q, N_k) for top-k selection
        """
        B, N_q, D = query_tokens.shape
        _, N_k, _ = key_tokens.shape

        # Project queries: get q^I and w^I for each head
        qw = self.query_proj(query_tokens)  # (B, N_q, H * (key_dim + 1))
        qw = qw.view(B, N_q, self.num_heads, self.key_dim + 1)

        # Split into query vectors and weights
        q = qw[..., :self.key_dim]  # (B, N_q, H, key_dim)
        w = qw[..., -1]  # (B, N_q, H) - scalar weights

        # Project keys
        k = self.key_proj(key_tokens)  # (B, N_k, key_dim)

        # Apply 2D RoPE if spatial dimensions provided
        if height is not None and width is not None:
            k = self._apply_rope_2d(k.view(B, N_k, self.key_dim), height, width)

        # Compute index scores: I_{t,s} = sum_j w_{t,j} * ReLU(q_{t,j} . k_s)
        # Reshape for batch matmul
        q = q.permute(0, 2, 1, 3)  # (B, H, N_q, key_dim)
        k = k.unsqueeze(1)  # (B, 1, N_k, key_dim)

        # Compute dot products
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_k)

        # Apply ReLU (as per paper for throughput)
        scores = F.relu(scores)

        # Weight and sum across heads
        w = w.permute(0, 2, 1).unsqueeze(-1)  # (B, H, N_q, 1)
        index_scores = (w * scores).sum(dim=1)  # (B, N_q, N_k)

        # Add spatial bias
        index_scores = index_scores + self.spatial_bias

        return index_scores

    def compute_alignment_loss(
        self,
        index_scores: torch.Tensor,
        attention_probs: torch.Tensor,
        selected_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute KL-divergence loss to align indexer with main attention.

        From the paper (Equation 3 and 4):
        - Dense warm-up: L^I = sum_t D_KL(p_{t,:} || Softmax(I_{t,:}))
        - Sparse training: L^I = sum_t D_KL(p_{t,S_t} || Softmax(I_{t,S_t}))

        Where p_{t,:} is the L1-normalized sum of main attention scores.

        Args:
            index_scores: Predicted index scores (B, N_q, N_k)
            attention_probs: Target attention distribution (B, N_q, N_k)
            selected_mask: Optional mask for sparse training (B, N_q, N_k)

        Returns:
            KL-divergence loss for indexer alignment
        """
        # L1-normalize target attention (sum across heads if multi-head)
        target = attention_probs.sum(dim=1) if attention_probs.dim() > 3 else attention_probs
        target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)

        # Apply selection mask for sparse training
        if selected_mask is not None:
            # Mask out non-selected positions
            index_scores = index_scores.masked_fill(~selected_mask, float('-inf'))
            target = target.masked_fill(~selected_mask, 0.0)
            target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)

        # Softmax over index scores
        pred = F.softmax(index_scores, dim=-1)

        # KL-divergence: D_KL(target || pred) = sum(target * log(target/pred))
        # Use stable computation
        kl_loss = F.kl_div(
            pred.log().clamp(min=-100),
            target,
            reduction='batchmean',
            log_target=False,
        )

        return kl_loss


class SpatialLightningIndexer(LightningIndexer):
    """
    Spatial-aware Lightning Indexer for image segmentation.

    Extends the base indexer with:
    - 2D spatial position encoding
    - Local window bias for efficient local attention
    - Pupil-specific spatial priors

    For pupil segmentation, we know:
    - Pupil is a roughly circular region
    - Pupil is always within the eye boundary
    - Local context (neighboring pixels) is most important

    Args:
        dim: Input embedding dimension
        num_heads: Number of indexer heads
        key_dim: Dimension of indexer keys/queries
        local_window: Size of local attention window (for bias)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        key_dim: int = 8,
        local_window: int = 7,
    ):
        super().__init__(dim, num_heads, key_dim, use_rope=True)
        self.local_window = local_window

        # Learnable local attention bias
        # Encourages attention to nearby spatial positions
        self.local_bias = nn.Parameter(
            torch.zeros(1, 1, local_window, local_window)
        )

        # Initialize with Gaussian-like local preference
        self._init_local_bias()

    def _init_local_bias(self):
        """Initialize local bias with Gaussian-like pattern."""
        center = self.local_window // 2
        for i in range(self.local_window):
            for j in range(self.local_window):
                dist = math.sqrt((i - center)**2 + (j - center)**2)
                self.local_bias.data[0, 0, i, j] = math.exp(-dist / 2)

    def _create_local_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a local attention mask for spatial bias.

        Each query position gets higher scores for nearby key positions.

        Args:
            height: Feature map height
            width: Feature map width
            device: Device to create tensor on

        Returns:
            Local bias tensor (1, H*W, H*W)
        """
        N = height * width
        bias = torch.zeros(1, N, N, device=device)

        half_win = self.local_window // 2

        for i in range(height):
            for j in range(width):
                q_idx = i * width + j

                for di in range(-half_win, half_win + 1):
                    for dj in range(-half_win, half_win + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            k_idx = ni * width + nj
                            local_i = di + half_win
                            local_j = dj + half_win
                            bias[0, q_idx, k_idx] = self.local_bias[0, 0, local_i, local_j]

        return bias

    def forward(
        self,
        query_tokens: torch.Tensor,
        key_tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Compute spatially-aware index scores.

        Args:
            query_tokens: Query token embeddings (B, H*W, D)
            key_tokens: Key token embeddings (B, H*W, D)
            height: Feature map height
            width: Feature map width

        Returns:
            Index scores with spatial bias (B, H*W, H*W)
        """
        # Get base index scores
        index_scores = super().forward(query_tokens, key_tokens, height, width)

        # Add local spatial bias
        local_bias = self._create_local_mask(height, width, query_tokens.device)
        index_scores = index_scores + local_bias

        return index_scores
