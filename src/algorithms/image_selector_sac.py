import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

import algorithms.modules as m
import net.auxiliary_pred as aux
from algorithms.sac import SAC
from net.image_attention import ImageAttentionSelectorLayers


class Image_Selector_SAC(SAC):
    def __init__(self, obs_shape, action_shape, args):
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.unsupervised_update_freq = args.unsupervised_update_freq
        self.unsupervised_update_num = args.unsupervised_update_num
        self.unsupervised_update_slow_freq = args.unsupervised_update_slow_freq
        self.stack_num = args.frame_stack
        self.channels = args.channels
        self.region_num = args.masked_region_num
        self.attention_heads = args.attention_heads
        self.max_grad_norm = args.max_grad_norm

        self.reward_factor = args.reward_factor
        self.inverse_factor = args.inv_factor
        self.forward_factor = args.fwd_factor
        self.reward_accumulated_steps = args.reward_accumulate_steps
        self.inv_accumulated_steps = args.inv_accumulate_steps
        self.fwd_accumulated_steps = args.fwd_accumulate_steps
        self.unsupervised_warmup_steps = args.unsupervised_warmup_steps
        self.reward_first_sampling = args.reward_first_sampling

        selector_layers = ImageAttentionSelectorLayers(obs_shape, args.masked_region_num, args.channels,
                                                       args.frame_stack, args.num_selector_layers, args.num_filters,
                                                       args.embed_dim, args.attention_heads)
        selector_cnn = m.SelectorCNN(selector_layers, obs_shape, args.masked_region_num, args.channels,
                                     args.frame_stack, args.num_shared_layers, args.num_filters).cuda()
        head_cnn = m.HeadCNN(selector_cnn.out_shape,
                             args.num_head_layers, args.num_filters).cuda()
        actor_projection = m.RLProjection(
            head_cnn.out_shape, args.projection_dim)
        critic_projection = m.RLProjection(
            head_cnn.out_shape, args.projection_dim)
        actor_encoder = m.Encoder(
            selector_cnn,
            head_cnn,
            actor_projection
        )
        critic_encoder = m.Encoder(
            selector_cnn,
            head_cnn,
            critic_projection
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

        self.complete_selector = selector_layers.cuda()
        self.selector_optimizer = torch.optim.Adam(
            self.complete_selector.parameters(), lr=args.selector_lr, betas=(args.selector_beta, 0.999)
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

    def update(self, replay_buffer, L, step):
        if step % self.unsupervised_update_slow_freq == 0 and step > self.unsupervised_warmup_steps:
            self.unsupervised_update_freq = self.unsupervised_update_freq + 1
            print("Update frequency slow down to ", str(self.unsupervised_update_freq))

        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if step % self.unsupervised_update_freq == 0 and step > self.unsupervised_warmup_steps:
            for _ in range(self.unsupervised_update_num):
                if self.reward_factor != 0:
                    self.update_reward_predictor(replay_buffer, L, step)
                if self.inverse_factor != 0:
                    self.update_inverse_dynamic_predictor(replay_buffer, L, step)
                if self.forward_factor != 0:
                    self.update_forward_dynamic_predictor(replay_buffer, L, step)

    def update_reward_predictor(self, replay_buffer, L, step):
        obs, action, reward, _ = replay_buffer.sample_multi_step(self.reward_accumulated_steps,
                                                                 reward_first=self.reward_first_sampling)
        concatenated_action = torch.concatenate(action, dim=1)
        concatenated_reward = torch.sum(torch.concatenate(reward, dim=1), dim=1, keepdim=True)
        predicted_reward = self.reward_predictor(obs, concatenated_action)
        predict_loss = self.reward_factor * \
                       F.mse_loss(concatenated_reward, predicted_reward)

        self.reward_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.reward_predictor.parameters(), self.max_grad_norm)
        self.reward_predictor_optimizer.step()

        L.log('train_rewardpred/loss', predict_loss, step)

    def update_inverse_dynamic_predictor(self, replay_buffer, L, step):
        obs, action, reward, next_obs = replay_buffer.sample_multi_step(self.inv_accumulated_steps,
                                                                        reward_first=self.reward_first_sampling)
        concatenated_pre_action = torch.concatenate(action[:-1], dim=1) if len(action) > 1 else torch.tensor([]).cuda()
        predicted_action = self.inverse_dynamic_predictor(obs, next_obs, concatenated_pre_action)
        predict_loss = self.inverse_factor * \
                       F.mse_loss(action[-1], predicted_action)

        self.inverse_dynamic_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.inverse_dynamic_predictor.parameters(), self.max_grad_norm)
        self.inverse_dynamic_predictor_optimizer.step()

        L.log('train_inversepred/loss', predict_loss, step)

    def update_forward_dynamic_predictor(self, replay_buffer, L, step):
        obs, action, reward, next_obs = replay_buffer.sample_multi_step(self.fwd_accumulated_steps,
                                                                        reward_first=self.reward_first_sampling)
        concatenated_action = torch.concatenate(action, dim=1)
        predicted_next_obs, embeded_next_obs = self.forward_dynamic_predictor(
            obs, next_obs, concatenated_action)
        predict_loss = self.forward_factor * \
                       F.mse_loss(predicted_next_obs, embeded_next_obs)

        self.forward_dynamic_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.forward_dynamic_predictor.parameters(), self.max_grad_norm)
        self.forward_dynamic_predictor_optimizer.step()

        L.log('train_forwardpred/loss', predict_loss, step)

    def select_image(self, obs):
        with torch.no_grad():
            current_obs = self._obs_to_input(obs)
            obs, logits = self.complete_selector(current_obs, return_all=True)
            selected_obs = torch.squeeze(obs)[-self.channels:].cpu().numpy()
            logits = logits.reshape(-1, self.region_num)[-1].cpu().detach().tolist()
            return logits, np.transpose(selected_obs * 255, (1, 2, 0)).astype(np.uint8)
