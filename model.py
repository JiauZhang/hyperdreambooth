import torch
from torch import nn

class VisualTransformerEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        ...

    def forward(self, face):
        ...

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, K, N):
        self.N = N
        self.K = K
        self.embed_dim = embed_dim
        self.model = ...

    def forward(self, face):
        batch = face.shape[0]
        weights = torch.zeros(
            batch, self.N, self.embed_dim, device=face.device,
            dtype=face.dtype,
        )
        for _ in range(self.K):
            weights += self.model(face, weights)
        return weights

class HyperNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, K, L):
        self.face_encoder = VisualTransformerEncoder(in_dim, out_dim, K)
        self.proj = nn.Linear(out_dim, out_dim)
        self.weight_decoder = TransformerDecoder(out_dim, out_dim)
        self.affine = [nn.Linear(out_dim, out_dim) for _ in range(L)]

    def forward(self, face):
        face = self.face_encoder(face)
        face = self.proj(face)
        delta_weights = self.weight_decoder(face) # batch, seq_len, embed_dim
        seq_len = delta_weights.shape[1]
        for i in range(seq_len):
            delta_weights[:, i] = self.affine[i](delta_weights[:, i])
        return delta_weights
