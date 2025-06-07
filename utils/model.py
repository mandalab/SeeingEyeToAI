import torch.nn as nn
import math
import torch

class FourierPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self):
        out = self.dropout(self.pe)
        return out.unsqueeze(0)  # (1, max_len, d_model)


class MultiModalTransformerEncoder(nn.Module):
    def __init__(
        self,
        patch_x,
        patch_y,
        video_feature_dim,
        hidden_dim,
        num_frames,
        num_encoder_layers,
        n_heads,
        dropout_rate=0.1,
    ):
        super(MultiModalTransformerEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.video_linear = nn.Linear(video_feature_dim, hidden_dim)
        self.patch_x = patch_x
        self.patch_y = patch_y

        # cls learnable token
        self.cls_token = nn.Embedding(1, hidden_dim)
        # positional embeddings
        self.pos_temporal_embed = nn.Embedding(
            num_frames, hidden_dim
        )
        self.init_pos_temporal_embed = torch.arange(
            0, num_frames
        ).unsqueeze(0)

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dropout=dropout_rate,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, video_features):
        device = video_features.device
        batch_size = video_features.shape[0]
        num_frames = video_features.shape[1]

        self.init_cls_token = torch.zeros(batch_size, 1).long()

        self.init_video_att_mask = torch.ones(
            batch_size, num_frames * self.patch_x * self.patch_y
        ).bool()
        self.init_cls_att_mask = torch.ones(batch_size, 1).bool()

        video_embeddings = self.video_linear(video_features.float())

        # temporal positional embeddings
        video_pos_encodings_temporal = self.pos_temporal_embed(
            self.init_pos_temporal_embed.repeat(batch_size, 1).to(device)
        )

        video_pos_encodings_temporal = video_pos_encodings_temporal.unsqueeze(
            2
        ).unsqueeze(3)
        video_embeddings += video_pos_encodings_temporal

        # flatten video
        video_embeddings = video_embeddings.view(
            batch_size, -1, self.hidden_dim
        )  # (B x (5x7x7) x hidden_dim)
        video_embeddings = self.layer_norm(video_embeddings)

        cls_tokens = self.cls_token(self.init_cls_token.to(device))

        combined_embeddings = torch.cat([cls_tokens, video_embeddings], dim=1)
        combined_attention_mask = torch.cat(
            [
                self.init_cls_att_mask.to(device),
                self.init_video_att_mask.to(device),
            ],
            dim=1,
        )
        combined_attention_mask = ~combined_attention_mask
        encoded = self.transformer_encoder(
            combined_embeddings, src_key_padding_mask=combined_attention_mask
        )

        cls_output = encoded[:, 0, :]  # cls output

        out = self.mlp(cls_output)

        return out

