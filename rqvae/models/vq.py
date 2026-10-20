import torch
import torch.nn as nn
from .layers import kmeans


class VectorQuantizer(nn.Module):
    """Nearest-neighbor quantizer (paper §3.1; Sinkhorn reallocation removed).

    forward returns the raw quantized vector q_d (gradient flows to the
    codebook only via codebook_loss). There is no per-layer STE: the single
    straight-through reparametrization is applied once at the decoder input in
    RQVAE.forward.
    """

    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x):
        # x: residual r_d, shape [*, e_dim]
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t())
        indices = torch.argmin(d, dim=-1)

        q = self.embedding(indices).view(x.shape)

        # §3.1: codebook loss ||sg[r_d] - e||^2, commitment beta*||r_d - sg[e]||^2
        # reduction: sum over the feature dim, mean over the batch
        codebook_loss = (q - x.detach()).pow(2).sum(-1).mean()
        commit_loss = self.beta * (x - q.detach()).pow(2).sum(-1).mean()

        indices = indices.view(x.shape[:-1])

        return q, codebook_loss, commit_loss, indices
