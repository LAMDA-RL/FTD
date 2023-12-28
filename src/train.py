import datetime
import gym
import numpy as np
import os
import psutil
import time
import torch
import warnings
from pympler.asizeof import asizeof

import utils
from algorithms.drqv2 import DrQV2Agent
from algorithms.factory import make_agent
from arguments import parse_args
from buffer import ReplayBuffer
from env.wrappers import make_env
from logger import Logger
from video import VideoRecorder

warnings.filterwarnings("ignore")

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def evaluate(env, agent, video, num_episodes, L, step, mode, domain_name, test_env=False):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0) and step % video.save_freq == 0)
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                if isinstance(agent, DrQV2Agent):
                    action = agent.select_action(obs, step=step, eval_mode=True)
                else:
                    action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            if video.plot_segment or video .plot_selected:
                logits, selected_image = agent.select_image(
                    obs)
            if video.plot_segment:
                segmented_image = utils.transfer_to_ann(obs, logits, video.channels,
                                                        video.region_num,
                                                        video.stack_num)
            else:
                segmented_image = None
            if not video.plot_selected:
                selected_image = None
            if "robosuite" in domain_name:
                video.record(env, mode, selected_image, segmented_image, obs)
            else:
                video.record(env, mode, selected_image, segmented_image)
            episode_reward += reward

        if L is not None:
            _test_env = '_test_env' if test_env else ''
            video.save(f'{step}{_test_env}.mp4')
            L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


def main(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.train_mode,
        color_type=args.color_type,
        apply_sam=args.apply_sam,
        args=args,
        is_train_env=True
    )
    test_env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed + 42,
        episode_length=args.episode_length,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.eval_mode,
        color_type=args.color_type,
        apply_sam=args.apply_sam,
        args=args,
        is_train_env=False
    ) if args.eval_mode is not None else env

    # Create working directory
    work_dir = os.path.join(args.log_dir, args.domain_name + '_' + args.task_name, args.algorithm, curr_time)
    print('Working directory:', work_dir)
    assert not os.path.exists(os.path.join(
        work_dir, 'train.log')), 'specified working directory already exists'
    utils.make_dir(work_dir)
    if not os.path.exists("figures"):
        utils.make_dir("figures")
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, args.plot_segment, args.plot_selected, height=84,
                          width=84, channels=args.channels, region_num=args.masked_region_num,
                          stack_num=args.frame_stack,
                          save_freq=args.save_video_freq)
    utils.write_info(args, os.path.join(work_dir, 'info.log'))

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    obs_shape = env.observation_space.shape
    true_obs_shape = list(obs_shape)
    true_obs_shape[0] = true_obs_shape[0] + args.channels * args.frame_stack if args.add_original_frame else \
    true_obs_shape[0]
    action_shape = env.action_space[0].shape if isinstance(env.action_space,
                                                           gym.spaces.tuple.Tuple) else env.action_space.shape
    assert len(obs_shape) >= 3, "Dimension of observation must be 3 or 4"
    if len(obs_shape) == 4:
        obs_shape = obs_shape[1:]
    assert len(action_shape) <= 2, "Dimension of action must be 1 or 2"
    if len(action_shape) == 2:
        action_shape = action_shape[1:]
    print('Observation space:', obs_shape)
    print('Action space:', action_shape)
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args
    )
    replay_buffer = ReplayBuffer(
        obs_shape=true_obs_shape,
        action_shape=action_shape,
        capacity=args.capacity,
        reward_first_capacity=args.reward_first_capacity,
        batch_size=args.batch_size
    )

    print("=====Start training=====")
    done = True
    start_time = time.time()
    episode = 0
    episode_reward = 0
    episode_step = 0
    obs = env.reset()
    L = Logger(work_dir, args)

    for step in range(args.train_steps):
        if done:
            if step > 0:
                L.log('train/duration', time.time() -
                      start_time, step)
                L.log('train/episode', episode, step)
                L.dump(step)

            # Evaluate agent periodically
            if step % args.eval_freq == 0:
                print('Evaluating:', work_dir)
                L.log('eval/episode', episode, step)
                if test_env is not None:
                    evaluate(test_env, agent, video, args.eval_episodes, L, step,
                             args.eval_mode, args.domain_name, test_env=True)
                L.dump(step)

            # Save agent periodically
            if step > 0 and step % args.save_freq == 0:
                torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

            L.log('train/episode_reward', episode_reward, step)

            L.log('train/memory_usage', psutil.virtual_memory().percent, step)
            L.log('train/buffer_size', asizeof(replay_buffer) /
                  np.power(2, 30), step)

            start_time = time.time()
            episode_reward = 0
            episode_step = 0
            episode = episode + 1

            obs = env.reset()

        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                if isinstance(agent, DrQV2Agent):
                    action = agent.select_action(obs, step=step, eval_mode=False)
                else:
                    action = agent.sample_action(obs)

                action = action.flatten()

        # Run training update
        if step > args.init_steps:
            agent.update(replay_buffer, L, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == args.max_episode_steps else float(done)
        episode_reward = episode_reward + reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_step = episode_step + 1
        obs = next_obs

    torch.save(agent, os.path.join(model_dir, f'{args.train_steps}.pt'))
    replay_buffer.save(args.train_steps)

    print('Completed training for', work_dir)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_idx)
    args.description = "_".join(
        [args.algorithm, args.train_mode, str(args.seed)])
    main(args)
