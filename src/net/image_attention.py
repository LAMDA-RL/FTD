import numpy as np
import torch
import torch.nn as nn

from algorithms.modules import NormalizeImg, Flatten
from algorithms.modules import weight_init


class ImageAttentionSelectorLayers(nn.Module):
    """ Using "attention" to select and combine the region in a soft way
    """

    def __init__(self, obs_shape, region_num, in_channels, stack_num, num_layers, num_filters, embed_dim, num_heads):
        super().__init__()

        self.preprocess_layer = nn.Sequential(*[NormalizeImg()])
        self.layers = [nn.Conv2d(in_channels, num_filters, 3, stride=2, padding=1)]
        self.shape = obs_shape[1:]
        self.region_num = region_num
        self.in_channels = in_channels
        self.stack_num = stack_num
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.current_image_size = obs_shape[1] // 2
        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.current_image_size = self.current_image_size // 2
        self.layers.append(Flatten())
        out_num = num_filters * self.current_image_size ** 2
        self.layers.append(nn.Linear(out_num, embed_dim))
        self.layers = nn.Sequential(*self.layers)
        self.layers.apply(weight_init)

        self.q = nn.Linear(embed_dim, num_heads * embed_dim)
        self.k = nn.Linear(embed_dim, num_heads * embed_dim)
        # no 'v' network, we use the raw input images as the values

    def forward(self, x, return_logits=False, return_head_logits=False, return_all=False):
        # (batch_size, stack_num * (region_num + 1) * channels , height, width)
        # Last region is the whole frame
        S, R, C, H, W = self.stack_num, self.region_num + 1, self.in_channels, self.shape[0], self.shape[1]
        x = x.reshape(-1, C, H, W)
        B = x.shape[0] // S // R
        x = self.preprocess_layer(x)

        mask = torch.sum(x, dim=(1, 2, 3)).reshape(B * S, 1, -1)[:, :, :-1]
        mask = torch.where(mask != 0, False, True)

        tokens = self.layers(x).reshape(B * S, R, -1)
        tokens_frame = tokens[:, -1:, :]
        tokens_segment = tokens[:, :-1, :]
        q = self.q(tokens_frame).reshape(B * S, 1, self.num_heads, self.embed_dim).transpose(-3, -2)
        k = self.k(tokens_segment).reshape(B * S, R - 1, self.num_heads, self.embed_dim).transpose(-3, -2)
        v = x.reshape(B * S, R, C * H * W)[:, :-1, :]

        attention = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.shape[-1])
        mask = torch.cat([torch.unsqueeze(mask, dim=1)] * self.num_heads, dim=1)
        attention = attention.masked_fill_(mask, float("-inf"))

        multi_probs = torch.softmax(attention, dim=-1)
        probs = torch.mean(multi_probs, dim=1)
        ret_obs = torch.matmul(probs, v)

        # vector 2 image
        ret_obs = ret_obs.reshape(-1, S * C, H, W)

        if return_logits:
            return probs
        elif return_head_logits:
            return multi_probs
        elif return_all:
            return ret_obs, probs
        else:
            return ret_obs
