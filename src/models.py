import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==========================================
# Model MLP Regressor
# ==========================================

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ==========================================
# Model Autodencoder
# ==========================================

class Encoder(nn.Module):
    def __init__(self, in_dim=768, z_dim=16):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.ReLU(),

            nn.LayerNorm(512),
            nn.Linear(512, 384),
            nn.ReLU(),

            nn.LayerNorm(384),
            nn.Linear(384, 256),
            nn.ReLU(),

            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, z_dim)
        )

    def forward(self, x):
        return self.net(x)
    

class Decoder(nn.Module):
    def __init__(self, out_dim=768, z_dim=16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),

            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.ReLU(),

            nn.LayerNorm(256),
            nn.Linear(256, 384),
            nn.ReLU(),

            nn.LayerNorm(384),
            nn.Linear(384, 512),
            nn.ReLU(),

            nn.LayerNorm(512),
            nn.Linear(512, out_dim)
        )

    def forward(self, z):
        return self.net(z)

class MirrorAutoEncoder(nn.Module):
    def __init__(self, in_dim=768, z_dim=16):
        super().__init__()
        self.encoder = Encoder(in_dim, z_dim)
        self.decoder = Decoder(in_dim, z_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class FrozenEncoder(nn.Module):
    """
    Wrapper inference-only pour un encodeur pre-entraine.
    - Tous les poids sont geles (requires_grad = False)
    - Mode eval par defaut
    """
    def __init__(self, trained_encoder: nn.Module):
        super().__init__()
        self.encoder = trained_encoder

        # Gel total des poids
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.encoder(x)

# ==========================================
# Model transformer
# ==========================================


class ContinuousTokenEmbedding(nn.Module):
    """
    Embed continuous features as tokens:
      token_i = x_i * W_i + b_i + E_feat_i
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Initialisation stabilisée
        self.W = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_features, d_model))

        self.feat_emb = nn.Embedding(n_features, d_model)
        nn.init.normal_(self.feat_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_features)
        returns: (B, n_features, d_model)
        """
        B, F = x.shape
        assert F == self.n_features

        feat_ids = torch.arange(F, device=x.device)
        e_feat = self.feat_emb(feat_ids)

        tokens = x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)
        tokens = tokens + e_feat.unsqueeze(0)

        return tokens


class NutritionTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        n_phys: int = 10,
        out_dim: int = 11,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_cls: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_phys = n_phys
        self.seq_len = latent_dim + n_phys + (1 if use_cls else 0)
        self.use_cls = use_cls

        # Token embeddings for continuous features (latent and phys)
        self.latent_embed = ContinuousTokenEmbedding(latent_dim, d_model)
        self.phys_embed = ContinuousTokenEmbedding(n_phys, d_model)

        # Special tokens
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)  # [MASK]
        if use_cls:
            self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)  # [CLS]

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    @staticmethod
    def _make_src_key_padding_mask(attn_keep: torch.Tensor) -> torch.Tensor:
        """
        PyTorch Transformer expects: True where padding (i.e., to ignore).
        attn_keep: (B, S) with 1=keep, 0=ignore
        returns: (B, S) bool with True=ignore
        """
        return (attn_keep == 0)

    def forward(self, x_concat: torch.Tensor, phys_present_mask: torch.Tensor | None = None):
        """
        x_concat: (B, 26) = [latent_16 | phys_10]
        phys_present_mask: optional (B, 10) with 1=present, 0=missing
                           If None, we infer missing from NaNs in phys part.

        returns: y_pred (B, out_dim)
        """

        B, D = x_concat.shape
        assert D == self.latent_dim + self.n_phys, f"Expected {self.latent_dim + self.n_phys}, got {D}"

        latent = x_concat[:, : self.latent_dim]          # (B,16)
        phys   = x_concat[:, self.latent_dim : ]         # (B,10)

        # Infer missing phys from NaNs if mask not provided
        if phys_present_mask is None:
            # present = not NaN
            phys_present_mask = (~torch.isnan(phys)).to(phys.dtype)  # (B,10) float 0/1
            # Replace NaNs by 0 for embedding arithmetic (mask will handle ignoring)
            phys = torch.nan_to_num(phys, nan=0.0)

        # Build tokens
        t_latent = self.latent_embed(latent)  # (B,16,d_model)
        t_phys   = self.phys_embed(phys)      # (B,10,d_model)

        # Apply [MASK] to missing phys tokens (value-level representation)
        # phys_present_mask: (B,10) -> (B,10,1)
        keep = phys_present_mask.unsqueeze(-1)  # float
        t_phys = t_phys * keep + self.mask_token.view(1, 1, -1) * (1.0 - keep)

        # Concatenate tokens (optionally prepend CLS)
        if self.use_cls:
            cls = self.cls_token.view(1, 1, -1).expand(B, 1, -1)  # (B,1,d_model)
            tokens = torch.cat([cls, t_latent, t_phys], dim=1)    # (B,S,d_model)
            # attention keep mask: CLS + latent(always 1) + phys_present
            attn_keep = torch.cat(
                [
                    torch.ones(B, 1, device=x_concat.device, dtype=phys_present_mask.dtype),
                    torch.ones(B, self.latent_dim, device=x_concat.device, dtype=phys_present_mask.dtype),
                    phys_present_mask,
                ],
                dim=1,
            )  # (B,S)
        else:
            tokens = torch.cat([t_latent, t_phys], dim=1)  # (B,26,d_model)
            attn_keep = torch.cat(
                [
                    torch.ones(B, self.latent_dim, device=x_concat.device, dtype=phys_present_mask.dtype),
                    phys_present_mask,
                ],
                dim=1,
            )

        # src_key_padding_mask expects True for tokens to ignore
        src_key_padding_mask = self._make_src_key_padding_mask(attn_keep)

        # Encode
        h = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)  # (B,S,d_model)

        # Pool -> prediction
        if self.use_cls:
            pooled = h[:, 0, :]  # CLS
        else:
            # masked mean pooling
            m = attn_keep.unsqueeze(-1)  # (B,S,1)
            pooled = (h * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1e-6))

        y_pred = self.head(pooled)
        return y_pred
    
    
