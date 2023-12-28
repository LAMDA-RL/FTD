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


class MICo(object):
    """Baseline algorithm with transition model and various decoder types."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            args,
            # mico_weight=0.01,
            # beta=0.1,
    ):
        # self.device = device
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.mico_weight = args.mico_weight
        self.beta = args.beta
        self.max_norm = args.max_norm

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

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        obs = self._obs_to_input(obs)
        with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            # obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
        return mu.cpu().data.numpy().flatten()

    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
        else:
            _obs = obs
        _obs = torch.FloatTensor(_obs).cuda()
        if len(_obs.shape) == 3:
            _obs = _obs.unsqueeze(0)
        return _obs

    def sample_action(self, obs):
        obs = self._obs_to_input(obs)
        with torch.no_grad():
            # obs = np.array(obs)
            # obs = np.vstack(obs)
            # obs = np.expand_dims(obs, 0)
            # if obs.shape[-1] != self.image_size:
            # obs = utils.random_crop(obs, self.image_size)
            # obs = torch.FloatTensor(obs).to(self.device)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    def update_encoder(self, obs, reward, next_obs, L, step):
        h = self.critic.encoder(obs)
        h_target = self.critic_target.encoder(obs)
        next_h_target = self.critic_target.encoder(next_obs)

        h_target = h_target.detach()
        next_h_target = next_h_target.detach()

        online_dist = utils.representation_distances(h, h_target, beta=self.beta)
        reward_dist = utils.reward_distance(reward)
        target_dist = utils.target_distances(next_h_target, reward_dist, self.discount)
        metric_loss = self.mico_weight * F.smooth_l1_loss(online_dist, target_dist)

        L.log('train/encoder_loss', metric_loss, step)

        self.encoder_optimizer.zero_grad()
        metric_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.critic.encoder.parameters(), max_norm=self.max_norm)
        self.encoder_optimizer.step()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step, frozen=False):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        if frozen:
            current_Q1, current_Q2 = self.critic(obs, action, detach=True)
        else:
            current_Q1, current_Q2 = self.critic(obs, action, detach=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(L, step)

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

        # self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step, load_encoder=False, frozen=False, retrain_metric=False):

        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step, frozen=frozen)

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

            if not load_encoder:
                self.update_encoder(obs, reward, next_obs, L, step)
            else:
                if not frozen and retrain_metric:
                    self.update_encoder(obs, reward, next_obs, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
