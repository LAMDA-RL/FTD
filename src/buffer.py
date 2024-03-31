import numpy as np
import os
import pickle
import random
import torch

import augmentations


def prefill_memory(obses, capacity, obs_shape, type):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for i in range(capacity):
        frame = np.ones((c // 3, h, w), dtype=type)
        obses[i] = (frame, frame)
    return obses


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, reward_first_capacity, batch_size):
        self.capacity = capacity
        self.reward_first_capacity = reward_first_capacity
        self.batch_size = batch_size

        self._obses = [None] * self.capacity
        self._obses = prefill_memory(
            self._obses, capacity, obs_shape, type=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def _add_observation(self, obs, next_obs, action, reward, done):
        obses = (obs, next_obs)
        self._obses[self.idx] = (obses)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add(self, obses, actions, rewards, next_obses, dones):
        self._add_observation(obses, next_obses, actions, rewards, dones)

    def _get_idxs(self, n=None, reward_first=False, save_steps=1):
        if n is None:
            n = self.batch_size
        if not reward_first:
            return np.random.randint(
                0, self.capacity if self.full else self.idx - save_steps, size=n
            )
        else:
            size = self.capacity if self.full else self.idx - save_steps
            sorted_indexes = sorted(range(size), key=lambda i: self.rewards[i], reverse=True)
            candi_index = sorted_indexes[:min(self.reward_first_capacity, size)]
            return np.random.choice(candi_index, size=n, replace=True)

    def _encode_obses(self, idxs):
        obses, next_obses = [], []
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        return np.array(obses), np.array(next_obses)

    def __sample__(self, n=None):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, actions, rewards, next_obs, not_dones

    def sample_multi_step(self, step=1, n=None, reward_first=False):
        idxs = self._get_idxs(n, reward_first, step)

        obses, next_obses = [], []
        for i in idxs:
            obs, _ = self._obses[i]
            next_obs, _ = self._obses[(i + step) % self.capacity]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        obs, next_obs = np.array(obses), np.array(next_obses)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        actions, rewards = [], []
        for add_idx in range(step):
            actions.append(torch.as_tensor(self.actions[(idxs + add_idx) % self.capacity]).cuda())
            rewards.append(torch.as_tensor(self.rewards[(idxs + add_idx) % self.capacity]).cuda())

        return obs, actions, rewards, next_obs

    def sample(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones

    def __sample_multi_rewards__(self, n=None, n_step=3, def_discount=0.99):
        if n is None:
            n = self.batch_size
        idxs = np.random.randint(0, self.capacity - n_step if self.full else self.idx - n_step, size=n)
        obs, _ = self._encode_obses(idxs)
        _, next_obs = self._encode_obses(idxs + n_step - 1)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        reward = np.zeros_like(self.rewards[idxs])
        discount = np.ones_like(self.not_dones[idxs])
        for i in range(n_step):
            step_reward = self.rewards[idxs + i]
            reward += discount * step_reward
            discount *= self.not_dones[idxs + i] * def_discount
        reward = torch.as_tensor(reward).cuda()
        discount = torch.as_tensor(discount).cuda()
        return obs, actions, reward, discount, next_obs

    def sample_drqv2(self, n=None, n_step=3, discount=0.99):
        obs, action, reward, discount, next_obs = self.__sample_multi_rewards__(n=n, n_step=n_step,
                                                                                def_discount=discount)
        return obs, action, reward, discount, next_obs
