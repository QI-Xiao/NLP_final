import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, embeddings=None):
        super().__init__()
        self.embedding_dim = d_model
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.segment_embedding = nn.Embedding(num_embeddings=2, embedding_dim=d_model)
        self.positional_embedding = PositionalEncoding(d_model, dropout)
        if embeddings is not None:
            self.token_embedding.weight.data.copy_(embeddings)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, input_ids, segment_ids):
        # Token embedding
        token_embeddings = self.token_embedding(input_ids)

        # Positional embedding
        positional_embeddings = self.positional_embedding(token_embeddings)

        # Segment embedding
        segment_embeddings = self.segment_embedding(segment_ids)

        # Concatenate all embeddings
        embeddings = token_embeddings + positional_embeddings + segment_embeddings

        # Apply attention mechanism
        attention_mask = self._generate_attention_mask(input_ids)
        embeddings = self.transformer_encoder(embeddings, src_key_padding_mask=attention_mask)

        # Batch normalization and collapsing
        embeddings = self.norm(embeddings)
        embeddings = torch.mean(embeddings, dim=1)

        # Feed forward layer
        logits = self.fc(embeddings)
        return logits

    def _generate_attention_mask(self, input_ids):
        padding_mask = (input_ids == 0)
        attention_mask = padding_mask.unsqueeze(1).repeat(1, input_ids.shape[1], 1)
        return attention_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

model = TransformerModel(vocab_size, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, embeddings=embeddings)
