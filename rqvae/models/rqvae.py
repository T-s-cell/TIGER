import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x):
        z = self.encoder(x)
        x_q, quant_loss, indices, layer_losses = self.rq(z)
        # single straight-through estimator at the decoder input (§3.1):
        # gradients reach the encoder through z, the codebooks through their
        # own loss terms only
        z_q = z + (x_q - z).detach()
        out = self.decoder(z_q)

        return out, quant_loss, indices, layer_losses

    @torch.no_grad()
    def get_indices(self, xs):
        x_e = self.encoder(xs)
        _, _, indices, _ = self.rq(x_e)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        # reduction per §3.1: per-sample sum over the feature dim, batch mean
        if self.loss_type == 'mse':
            loss_recon = (out - xs).pow(2).sum(-1).mean()
        elif self.loss_type == 'l1':
            loss_recon = (out - xs).abs().sum(-1).mean()
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon
