# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from itertools import count

import algorithms.modules as m
import utils


class Q2(object):
    """Baseline algorithm with transition model and various decoder types."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            args,
            # sia_hidden_dims=[256,4],
            # sia_hidden_dims_act=[256],
            # noise_scale=0.05,
            # dsa_lr=1e-3,
            # distance_measure='abs',
            # be_double=False,
            # sia_tau=0.05,
            # use_tanh=args.ds=0.99,
            # encoder_lr=1e-3,
            # start_update_encoder=5000,
            # dsa_target_update_freq=2,
            # use_relu=False,
            # oa_embed_dim=256,
            # dsa_act_update_freq=1,
            # sl1=False,
            # uniform_noise=False,
            # q2_weight=1.0
    ):

        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.projection_dim = args.projection_dim

        self.num_dsa = 0
        self.dsa_discount = args.discount
        self.sia_tau = args.encoder_tau

        self.noise_scale = args.noise_scale
        self.dsa_target_update_freq = args.critic_target_update_freq
        self.dsa_act_update_freq = args.dsa_act_update_freq

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

        self.dsa_critic = m.make_siamese_net('critic', args.projection_dim, action_shape[0], [1024, 32],
                                             noise_scale=self.noise_scale).cuda()
        self.dsa_actor = m.make_siamese_net('actor', args.projection_dim, action_shape[0], [1024],
                                            max_action=1.0).cuda()
        self.dsa_critic_target = copy.deepcopy(self.dsa_critic)

        # optimizers

        self.dsa_critic_optimizer = torch.optim.Adam(
            self.dsa_critic.parameters(), lr=args.critic_lr
        )

        self.dsa_actor_optimizer = torch.optim.Adam(
            self.dsa_actor.parameters(), lr=args.critic_lr
        )

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
            self.critic.encoder.parameters(), lr=args.critic_lr
        )

        self.train()
        self.critic_target.train()
        self.dsa_critic_target.train()

    def update_encoder(self, obs, action, reward, next_obs, L, step):
        batch_size = obs.size(0)
        action_dim = action.size(-1)

        h = self.critic.encoder(obs)
        h_target = self.critic.encoder(obs)
        h_target = h_target.detach()

        with torch.no_grad():
            next_h = self.critic.encoder(next_obs)
            next_h = next_h.detach()
            next_h_sq = utils.squarify(next_h)
            next_h_target_sq = next_h_sq.permute(1, 0, 2)
            next_h_sq = torch.reshape(next_h_sq, (batch_size ** 2, self.projection_dim))
            next_h_target_sq = torch.reshape(next_h_target_sq, (batch_size ** 2, self.projection_dim))
            next_a_sq = self.dsa_actor(next_h_sq, next_h_target_sq)  # pa (bs**2, ad)

            rew_sq = utils.squarify(reward)
            rew_target_sq = rew_sq.permute(1, 0, 2)
            rew_sq = torch.reshape(rew_sq, (batch_size ** 2, 1))
            rew_target_sq = torch.reshape(rew_target_sq, (batch_size ** 2, 1))
            rew_diffs = F.smooth_l1_loss(rew_sq, rew_target_sq, reduction='none')

            next_h = self.critic_target.encoder(next_obs)
            next_h_sq = utils.squarify(next_h)
            next_h_target_sq = next_h_sq.permute(1, 0, 2)
            next_h_sq = torch.reshape(next_h_sq, (batch_size ** 2, self.projection_dim))
            next_h_target_sq = torch.reshape(next_h_target_sq, (batch_size ** 2, self.projection_dim))

            point1 = self.dsa_critic_target(next_h_sq, next_a_sq)
            point2 = self.dsa_critic_target(next_h_target_sq, next_a_sq)
            target_dsa = F.l1_loss(point1, point2, reduction='none')
            target_dsa = target_dsa.sum(1).unsqueeze(-1)

            target_dsa = rew_diffs + self.dsa_discount * target_dsa

        h_sq = utils.squarify(h)  # (bs,bs,fs)
        h_sq = torch.reshape(h_sq, (batch_size ** 2, self.projection_dim))
        h_target_sq = utils.squarify(h_target)
        h_target_sq = h_target_sq.permute(1, 0, 2)
        h_target_sq = torch.reshape(h_target_sq, (batch_size ** 2, self.projection_dim))

        a_sq = utils.squarify(action)
        a_target_sq = a_sq.permute(1, 0, 2)
        a_sq = torch.reshape(a_sq, (batch_size ** 2, action_dim))
        a_target_sq = torch.reshape(a_target_sq, (batch_size ** 2, action_dim))

        point1 = self.dsa_critic(h_sq, a_sq)
        point2 = self.dsa_critic(h_target_sq, a_target_sq)
        point2 = point2.detach()

        dsa = F.l1_loss(point1, point2, reduction='none')
        dsa = dsa.sum(1).unsqueeze(-1)

        dsa_loss = F.mse_loss(dsa, target_dsa)

        self.dsa_critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        dsa_loss.backward()
        self.dsa_critic_optimizer.step()
        self.encoder_optimizer.step()

        L.log('train/dsa_critic_loss', dsa_loss.item(), step)

        self.num_dsa += 1

        if self.num_dsa % self.dsa_act_update_freq == 0:
            h = self.critic.encoder(obs)
            h = h.detach()
            h_target = self.critic.encoder(obs)
            h_target = h_target.detach()

            h_sq = utils.squarify(h)
            h_target_sq = utils.squarify(h_target)
            h_target_sq = h_target_sq.permute(1, 0, 2)
            h_sq = torch.reshape(h_sq, (batch_size ** 2, self.projection_dim))
            h_target_sq = torch.reshape(h_target_sq, (batch_size ** 2, self.projection_dim))
            max_a_sq = self.dsa_actor(h_sq, h_target_sq)

            point1 = self.dsa_critic(h_sq, max_a_sq)
            point2 = self.dsa_critic(h_target_sq, max_a_sq)
            point2 = point2.detach()
            dsa = F.l1_loss(point1, point2, reduction='none')
            dsa = dsa.sum(1).unsqueeze(-1)
            dsa_loss = -dsa.mean()

            # optimizer update
            self.dsa_actor_optimizer.zero_grad()
            # self.encoder_optimizer.zero_grad()
            dsa_loss.backward()
            self.dsa_actor_optimizer.step()
            # self.encoder_optimizer.step()

            L.log('train/dsa_actor_loss', dsa_loss.item(), step)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.dsa_critic.train(training)
        self.dsa_actor.train(training)

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
            # obs = utils.center_crop_image(obs, self.image_size)
            # obs = torch.FloatTensor(obs).to(self.device)
            # obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

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

    def update(self, replay_buffer, L, step):

        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

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

            # if step > self.start_update_encoder:
            self.update_encoder(obs, action, reward, next_obs, L, step)

            if step % self.dsa_target_update_freq == 0:
                for param, target_param in zip(self.dsa_critic.parameters(), self.dsa_critic_target.parameters()):
                    target_param.data.copy_(self.sia_tau * param.data + (1 - self.sia_tau) * target_param.data)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

        torch.save(
            self.dsa_critic.state_dict(), '%s/dsa_critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.dsa_actor.state_dict(), '%s/dsa_actor_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
