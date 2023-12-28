import numpy as np
import torch
import torch.nn as nn

from algorithms.modules import weight_init


class EmbeddingAttentionSelectorLayers(nn.Module):
    """ Using "attention" to select and combine the region in a soft way
    """

    def __init__(self, input_shape, total_shape, num_layers, embed_dim):
        super().__init__()

        self.S, self.R, self.C, self.H, self.W = total_shape
        self.R = self.R + 1

        self.layers = [nn.Linear(input_shape[-1], embed_dim)]
        for _ in range(1, num_layers):
            self.layers.append(nn.Linear(embed_dim, embed_dim))
            self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)
        self.layers.apply(weight_init)

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.out_shape = [embed_dim * self.S]

    def forward(self, x, original_frame, return_logits=False):
        """
        x:              shape = (B * (sample_region + 1) * RGB_Channel, flatten_dim)
        original_frame: shape = (B, sample_region + 1) * RGB_Channel, W, H)
        Each last region represents the complete pixel image.
        """
        B = x.shape[0] // self.S // self.R

        tokens = self.layers(x).reshape(B * self.S, self.R, -1)
        tokens_frame = tokens[:, -1:, :]
        tokens_segment = tokens[:, :-1, :]
        q = self.q(tokens_frame)
        k = self.k(tokens_segment)
        v = self.v(tokens_segment)

        original_frame = original_frame.reshape(-1, self.C, self.H, self.W)
        mask = torch.sum(original_frame, dim=(1, 2, 3)).reshape(
            B * self.S, 1, -1)[:, :, :-1]
        mask = torch.where(mask != 0, False, True)

        attention = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.shape[-1])
        attention = attention.masked_fill_(mask, float("-inf"))

        probs = torch.softmax(attention, dim=-1)
        ret = torch.matmul(probs, v)
        ret = torch.reshape(ret, (B, -1))

        if return_logits:
            return probs
        else:
            return ret
