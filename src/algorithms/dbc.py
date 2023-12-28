# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

import algorithms.modules as m
import utils


class ProbabilisticTransitionModel(nn.Module):

    def __init__(self, projection, action_shape, hidden_dim, max_sigma=1e1, min_sigma=1e-4,
                 encoder_max_norm=None):
        super().__init__()
        # self.shared_cnn = shared_cnn
        # self.head_cnn = head_cnn
        # self.projection = projection

        self.fc = nn.Linear(projection.out_dim + action_shape[0], hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, projection.out_dim)
        self.fc_sigma = nn.Linear(hidden_dim, projection.out_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert (self.max_sigma >= self.min_sigma)
        self.max_norm = encoder_max_norm

        print("Probabilistic transition model chosen.")

    def forward(self, x, action, normalize=True):
        # x = self.shared_cnn(x)
        # x = self.head_cnn(x)
        # x = self.projection(x)

        x = torch.cat([x, action], dim=1)
        x = self.fc(x)
        x = self.ln(x)
        x = F.relu(x)

        mu = self.fc_mu(x)
        if self.max_norm and normalize:
            mu = self.normalize(mu)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x, action):
        mu, sigma = self(x, action)
        eps = torch.randn_like(sigma)
        ret = mu + sigma * eps
        if self.max_norm:
            ret = self.normalize(ret)
            # WARNING: not adjusting for non-linear change in distribution.
        return ret

    def normalize(self, x):
        norms = x.norm(dim=-1)
        norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
        x = x / norm_to_max

        return x


class DeterministicTransitionModel(nn.Module):

    def __init__(self, projection, action_shape, hidden_dim,
                 encoder_max_norm=None):
        super().__init__()
        # self.shared_cnn = shared_cnn
        # self.head_cnn = head_cnn
        # self.projection = projection

        self.fc = nn.Linear(projection.out_dim + action_shape[0], hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, projection.out_dim)

        self.max_norm = encoder_max_norm

        print("Deterministic transition model chosen.")

    def forward(self, x, action, normalize=True):
        # x = self.shared_cnn(x)
        # x = self.head_cnn(x)
        # x = self.projection(x)

        x = torch.cat([x, action], dim=1)
        x = self.fc(x)
        x = self.ln(x)
        x = F.relu(x)

        mu = self.fc_mu(x)
        if self.max_norm and normalize:
            mu = self.normalize(mu)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x, action):
        mu, sigma = self(x, action)
        return mu

    def normalize(self, x):
        norms = x.norm(dim=-1)
        norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
        x = x / norm_to_max

        return x


class RewardModel(nn.Module):
    def __init__(self, proj_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(proj_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = F.relu(x)
        return self.out(x)


class BisimAgent(object):
    """Bisimulation metric algorithm."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            args
    ):

        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.transition_model_type = args.transition_model_type
        self.bisim_coef = args.bisim_coef
        self.dyn_loss = args.dbc_dyn_loss

        shared_cnn = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters).cuda()
        head_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).cuda()
        actor_projection = m.RLProjection(head_cnn.out_shape, args.projection_dim)
        critic_projection = m.RLProjection(head_cnn.out_shape, args.projection_dim)
        actor_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            actor_projection
        )
        critic_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            critic_projection
        )

        self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim, args.actor_log_std_min,
                             args.actor_log_std_max).cuda()
        self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim).cuda()
        self.critic_target = deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
        )
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
        )

        if self.transition_model_type == 'prob':
            self.transition_model = ProbabilisticTransitionModel(critic_projection, action_shape, args.hidden_dim,
                                                                 max_sigma=1e1, min_sigma=1e-4,
                                                                 encoder_max_norm=args.encoder_max_norm).cuda()
        elif self.transition_model_type == 'deter':
            self.transition_model = DeterministicTransitionModel(critic_projection, action_shape, args.hidden_dim,
                                                                 encoder_max_norm=args.encoder_max_norm).cuda()

        self.reward_model = RewardModel(critic_projection.out_dim, args.hidden_dim).cuda()
        self.trans_optimizer = torch.optim.Adam(
            list(self.reward_model.parameters()) + list(self.transition_model.parameters()),
            lr=args.selector_lr,
            weight_decay=args.decoder_weight_lambda
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
        else:
            _obs = obs
        _obs = torch.FloatTensor(_obs).cuda()
        if len(_obs.shape) == 3:
            _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
        return pi.cpu().data.numpy()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_encoder(self, obs, action, reward, L, step):
        h = self.critic.encoder(obs)

        # Sample random states across episodes at random
        batch_size = obs.size(0)
        perm = np.random.permutation(batch_size)
        h2 = h[perm]

        with torch.no_grad():
            # action, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(h, action)
            # reward = self.reward_decoder(pred_next_latent_mu1)
            reward2 = reward[perm]
        if pred_next_latent_sigma1 is None:
            pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
        if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
        elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
        else:
            raise NotImplementedError

        z_dist = F.smooth_l1_loss(h, h2, reduction='none')
        r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')
        if self.transition_model_type == '':
            transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none')
        else:
            if self.dyn_loss == 'mse':
                transition_dist = torch.sqrt(
                    (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
                    (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
                )
            else:
                transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none') \
                                  + F.smooth_l1_loss(pred_next_latent_sigma1, pred_next_latent_sigma2, reduction='none')

        bisimilarity = r_dist + self.discount * transition_dist
        loss = (z_dist - bisimilarity).pow(2).mean()
        L.log('train_dbcencoder/loss', loss, step)
        return loss

    def update_transition_reward_model(self, obs, action, next_obs, reward, L, step):
        h = self.critic.encoder(obs)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(h, action)
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.critic.encoder(next_obs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_dbctransition/loss', loss, step)

        pred_next_latent = self.transition_model.sample_prediction(h, action)
        pred_next_reward = self.reward_model(pred_next_latent)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        L.log('train_dbcreward/loss', reward_loss, step)

        total_loss = loss + reward_loss
        return total_loss

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        transition_reward_loss = self.update_transition_reward_model(obs, action, next_obs, reward, L, step)
        encoder_loss = self.update_encoder(obs, action, reward, L, step)
        total_loss = self.bisim_coef * encoder_loss + transition_reward_loss
        self.encoder_optimizer.zero_grad()
        self.trans_optimizer.zero_grad()
        total_loss.backward()
        self.trans_optimizer.step()
        self.encoder_optimizer.step()

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
