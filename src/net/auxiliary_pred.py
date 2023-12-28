import torch
import torch.nn as nn

from algorithms.modules import weight_init


class RewardPredictor(nn.Module):
    def __init__(self, encoder, action_shape, accumulate_steps, hidden_dim, num_filters=32):
        super().__init__()
        self.num_filters = num_filters

        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim + action_shape[0] * accumulate_steps, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.mlp.apply(weight_init)

    def forward(self, x, action):
        x = self.encoder(x)
        x = torch.cat([x, action], dim=1)
        x = self.mlp(x)

        return x


class InverseDynamicPredictor(nn.Module):
    def __init__(self, encoder, action_shape, accumulate_steps, hidden_dim, num_filters=32):
        super().__init__()
        self.num_filters = num_filters

        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim * 2 + (accumulate_steps - 1) * action_shape[0], hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0])
        )

        self.mlp.apply(weight_init)

    def forward(self, x, next_x, pre_actions):
        x = self.encoder(x)
        next_x = self.encoder(next_x)

        x = torch.cat((x, next_x, pre_actions), dim=1)
        x = self.mlp(x)

        return x


class ForwardDynamicPredictor(nn.Module):
    def __init__(self, encoder, action_shape, accumulate_steps, hidden_dim, num_filters=32):
        super().__init__()
        self.num_filters = num_filters

        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim + action_shape[0] * accumulate_steps, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.encoder.out_dim)
        )

        self.mlp.apply(weight_init)

    def forward(self, x, next_x, action):
        x = self.encoder(x)
        next_x = self.encoder(next_x)

        x = torch.cat((x, action), dim=1)
        x = self.mlp(x)

        return x, next_x
