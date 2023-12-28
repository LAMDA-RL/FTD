import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

import algorithms.modules as m
import net.auxiliary_pred as aux
import utils


class SAC(object):
    def __init__(self, obs_shape, action_shape, args):
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.max_grad_norm = args.max_grad_norm

        self.unsupervised_update_freq = args.unsupervised_update_freq
        self.unsupervised_update_num = args.unsupervised_update_num
        self.reward_factor = args.reward_factor
        self.inverse_factor = args.inv_factor
        self.forward_factor = args.fwd_factor

        shared_cnn = m.SharedCNN(
            obs_shape, args.num_shared_layers, args.num_filters).cuda()
        head_cnn = m.HeadCNN(shared_cnn.out_shape,
                             args.num_head_layers, args.num_filters).cuda()
        actor_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim)
        )
        critic_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim)
        )

        self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim, args.actor_log_std_min,
                             args.actor_log_std_max).cuda()
        self.critic = m.Critic(
            critic_encoder, action_shape, args.hidden_dim).cuda()
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

        self.reward_predictor = aux.RewardPredictor(critic_encoder, action_shape, args.reward_accumulate_steps,
                                                    args.hidden_dim, args.num_filters).cuda()
        self.reward_predictor_optimizer = torch.optim.Adam(
            self.reward_predictor.parameters(), lr=args.selector_lr, betas=(args.selector_beta, 0.999)
        )

        self.inverse_dynamic_predictor = aux.InverseDynamicPredictor(critic_encoder, action_shape,
                                                                     args.inv_accumulate_steps,
                                                                     args.hidden_dim, args.num_filters).cuda()
        self.inverse_dynamic_predictor_optimizer = torch.optim.Adam(
            self.inverse_dynamic_predictor.parameters(), lr=args.selector_lr, betas=(args.selector_beta, 0.999)
        )

        self.forward_dynamic_predictor = aux.ForwardDynamicPredictor(critic_encoder, action_shape,
                                                                     args.fwd_accumulate_steps,
                                                                     args.hidden_dim, args.num_filters).cuda()
        self.forward_dynamic_predictor_optimizer = torch.optim.Adam(
            self.forward_dynamic_predictor.parameters(), lr=args.selector_lr, betas=(args.selector_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        self.train(False)

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
            mu, _, _, _ = self.actor(
                _obs, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
        return pi.cpu().data.numpy()

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        """
        The soft value function V is defined as:
        V(s) = Expected value over actions from policy of [Q(s, a) - alpha * log(policy(a|s))]
        """
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if L is not None:
            L.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                                ) + log_std.sum(dim=-1)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                    self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def soft_update_critic_target(self):
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

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if step % self.unsupervised_update_freq == 0:
            for _ in range(self.unsupervised_update_num):
                if self.reward_factor != 0:
                    self.update_reward_predictor(obs, action, reward, L, step)
                if self.inverse_factor != 0:
                    self.update_inverse_dynamic_predictor(
                        obs, action, next_obs, L, step)
                if self.forward_factor != 0:
                    self.update_forward_dynamic_predictor(
                        obs, action, next_obs, L, step)

    def update_reward_predictor(self, obs, action, reward, L, step):
        predicted_reward = self.reward_predictor(obs, action)
        predict_loss = self.reward_factor * \
                       F.mse_loss(reward, predicted_reward)

        self.reward_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.reward_predictor.parameters(), self.max_grad_norm)
        self.reward_predictor_optimizer.step()

        L.log('train_rewardpred/loss', predict_loss, step)

    def update_inverse_dynamic_predictor(self, obs, action, next_obs, L, step):
        predicted_action = self.inverse_dynamic_predictor(obs, next_obs)
        predict_loss = self.inverse_factor * \
                       F.mse_loss(action, predicted_action)

        self.inverse_dynamic_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.inverse_dynamic_predictor.parameters(), self.max_grad_norm)
        self.inverse_dynamic_predictor_optimizer.step()

        L.log('train_inversepred/loss', predict_loss, step)

    def update_forward_dynamic_predictor(self, obs, action, next_obs, L, step):
        predicted_next_obs, embeded_next_obs = self.forward_dynamic_predictor(
            obs, next_obs, action)
        predict_loss = self.forward_factor * \
                       F.mse_loss(predicted_next_obs, embeded_next_obs)

        self.forward_dynamic_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.forward_dynamic_predictor.parameters(), self.max_grad_norm)
        self.forward_dynamic_predictor_optimizer.step()

        L.log('train_forwardpred/loss', predict_loss, step)
