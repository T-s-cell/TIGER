import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ Residual quantization per paper §3.1.

    residual_{d+1} = residual_d - q_d.detach(): the next layer's residual is
    detached from the current layer's codebook selection, so every layer's
    commitment loss trains the encoder through the residual chain (the frozen
    version's per-layer STE made layers 2+ commitment gradients vanish).
    Layer losses are summed, not averaged.
    """

    def __init__(self, n_e_list, e_dim, beta = 0.25,
                 kmeans_init = False, kmeans_iters = 100):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters)
                                        for n_e in n_e_list])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x):
        x_q = 0
        residual = x
        quant_loss = 0
        all_indices = []
        layer_losses = []

        for quantizer in self.vq_layers:
            q_d, codebook_loss, commit_loss, indices = quantizer(residual)
            residual = residual - q_d.detach()
            x_q = x_q + q_d
            quant_loss = quant_loss + codebook_loss + commit_loss
            layer_losses.append({"codebook": codebook_loss.detach(),
                                 "commitment": commit_loss.detach()})
            all_indices.append(indices)

        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, quant_loss, all_indices, layer_losses
