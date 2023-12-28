import copy
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import xmltodict
from collections import deque
from dm_control.suite import common
from dm_control.utils import io as resources
from gym import spaces
from gym.vector import AsyncVectorEnv
from numpy.random import randint

import dmc2gym
import utils
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator


def make_env(
        domain_name,
        task_name,
        seed=0,
        episode_length=1000,
        frame_stack=3,
        action_repeat=4,
        image_size=100,
        mode='train',
        color_type='rgb',
        apply_sam=False,
        args=None,
        cuda_idx=0,
        is_train_env=True
):
    """Make environment for experiments"""
    assert mode in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard'}, \
        f'specified mode "{mode}" is not supported'

    paths = []
    if "robosuite" in domain_name:
        import robosuite
        env = robosuite.make(
            env_name=task_name,
            robots="Panda",
            gripper_types="default",  # use default grippers per robot arm
            use_latch=False,  # use a spring-loaded handle
            has_renderer=False,  # no on-screen rendering
            has_offscreen_renderer=True,  # off-screen rendering needed for image obs
            control_freq=20,  # 20 hz control for applied actions
            horizon=1000,  # each episode terminates after 1000 steps
            use_object_obs=False,  # don't provide object observations to agent
            use_camera_obs=True,  # provide image observations to agent
            camera_names="frontview",  # use "agentview" camera for observations
            camera_heights=image_size,  # image height
            camera_widths=image_size,  # image width
            reward_shaping=True,  # use a dense reward signal for learning
        )
        env = robosuite_wrapper(env, action_repeat=action_repeat)
    else:
        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=seed,
            visualize_reward=False,
            from_pixels=True,
            height=image_size,
            width=image_size,
            episode_length=episode_length,
            frame_skip=action_repeat,
            background_dataset_paths=paths,
            camera_id=args.camera_id
        )
    env = VideoWrapper(env, mode, seed, is_train_env)
    env = ColorWrapper(env, mode, color_type, seed)
    if apply_sam:
        env = SegmentWrapper(env, args, cuda_idx)
    env = FrameStack(env, frame_stack)

    return env


class ColorWrapper(gym.Wrapper):
    """Wrapper for the color experiments"""

    def __init__(self, env, mode, color_type, seed=None):
        # assert isinstance(env, FrameStack), 'wrapped env must be a framestack'
        gym.Wrapper.__init__(self, env)
        self._mode = mode
        self._random_state = np.random.RandomState(seed)
        self.time_step = 0
        self._color_type = color_type

        shp = env.observation_space.shape
        if color_type == "gray":
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=((1,) + shp[1:]),
                dtype=env.observation_space.dtype
            )

        if 'color' in self._mode:
            self._load_colors()

    def reset(self):
        self.time_step = 0
        if 'color' in self._mode:
            self.randomize()
        # elif 'video' in self._mode:
        #     # apply greenscreen
        #     setting_kwargs = {
        #         'skybox_rgb': [.2, .8, .2],
        #         'skybox_rgb2': [.2, .8, .2],
        #         'skybox_markrgb': [.2, .8, .2]
        #     }
        #     if self._mode == 'video_hard':
        #         setting_kwargs['grid_rgb1'] = [.2, .8, .2]
        #         setting_kwargs['grid_rgb2'] = [.2, .8, .2]
        #         setting_kwargs['grid_markrgb'] = [.2, .8, .2]
        #     self.reload_physics(setting_kwargs)

        if self._color_type == "rgb":
            return self.env.reset()
        else:
            return self._rgb_to_gray(self.env.reset())

    def step(self, action):
        self.time_step += 1
        if self._color_type == "rgb":
            return self.env.step(action)
        else:
            next_obs, reward, done, info = self.env.step(action)
            return self._rgb_to_gray(next_obs), reward, done, info

    def randomize(self):
        assert 'color' in self._mode, f'can only randomize in color mode, received {self._mode}'
        self.reload_physics(self.get_random_color())

    def _load_colors(self):
        assert self._mode in {'color_easy', 'color_hard'}
        self._colors = torch.load(f'src/env/data/{self._mode}.pt')

    def get_random_color(self):
        assert len(self._colors) >= 100, 'env must include at least 100 colors'
        return self._colors[self._random_state.randint(len(self._colors))]

    def reload_physics(self, setting_kwargs=None, state=None):
        domain_name = self._get_dmc_wrapper()._domain_name
        if setting_kwargs is None:
            setting_kwargs = {}
        if state is None:
            state = self._get_state()
        self._reload_physics(
            *get_model_and_assets_from_setting_kwargs(
                domain_name + '.xml', setting_kwargs
            )
        )
        self._set_state(state)

    def get_state(self):
        return self._get_state()

    def set_state(self, state):
        self._set_state(state)

    def _get_dmc_wrapper(self):
        _env = self.env
        while not isinstance(_env, dmc2gym.wrappers.DMCWrapper) and hasattr(_env, 'env'):
            _env = _env.env
        assert isinstance(
            _env, dmc2gym.wrappers.DMCWrapper), 'environment is not dmc2gym-wrapped'

        return _env

    def _reload_physics(self, xml_string, assets=None):
        _env = self.env
        while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
            _env = _env.env
        assert hasattr(
            _env, '_physics'), 'environment does not have physics attribute'
        _env.physics.reload_from_xml_string(xml_string, assets=assets)

    def _get_physics(self):
        _env = self.env
        while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
            _env = _env.env
        assert hasattr(
            _env, '_physics'), 'environment does not have physics attribute'

        return _env._physics

    def _get_state(self):
        return self._get_physics().get_state()

    def _set_state(self, state):
        self._get_physics().set_state(state)

    def _rgb_to_gray(self, image):
        assert image.shape[0] == 3, "Input color channel must be 3"
        gray_image = np.expand_dims(image.mean(axis=0), axis=0).astype("uint8")

        return gray_image


class FrameStack(gym.Wrapper):
    """Stack frames as observation"""

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return utils.LazyFrames(list(self._frames))


def rgb_to_hsv(r, g, b):
    """Convert RGB color to HSV color"""
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc - minc) / maxc
    rc = (maxc - r) / (maxc - minc)
    gc = (maxc - g) / (maxc - minc)
    bc = (maxc - b) / (maxc - minc)
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = (h / 6.0) % 1.0
    return h, s, v


def do_green_screen(x, bg):
    """Removes green background from observation and replaces with bg; not optimized for speed"""
    assert isinstance(x, np.ndarray) and isinstance(
        bg, np.ndarray), 'inputs must be numpy arrays'
    assert x.dtype == np.uint8 and bg.dtype == np.uint8, 'inputs must be uint8 arrays'

    # Get image sizes
    x_h, x_w = x.shape[1:]

    # Convert to RGBA images
    im = TF.to_pil_image(torch.ByteTensor(x))
    im = im.convert('RGBA')
    pix = im.load()
    bg = TF.to_pil_image(torch.ByteTensor(bg))
    bg = bg.convert('RGBA')
    bg = bg.load()

    # Replace pixels
    for x in range(x_w):
        for y in range(x_h):
            r, g, b, a = pix[x, y]
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(
                r / 255., g / 255., b / 255.)
            h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

            min_h, min_s, min_v = (100, 80, 70)
            max_h, max_s, max_v = (185, 255, 255)
            if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
                pix[x, y] = bg[x, y]

    return np.moveaxis(np.array(im).astype(np.uint8), -1, 0)[:3]


class VideoWrapper(gym.Wrapper):
    """Green screen for video experiments"""

    def __init__(self, env, mode, seed, is_train_env):
        gym.Wrapper.__init__(self, env)
        self._mode = mode
        self._seed = seed
        self._random_state = np.random.RandomState(seed)
        self._index = 0
        self._video_paths = []
        if 'video' in mode:
            self._get_video_paths(is_train_env)
        self._num_videos = len(self._video_paths)

    def _get_video_paths(self, is_train_env):
        if is_train_env:
            start_idx = 0
            end_idx = 80
        else:
            start_idx = 80
            end_idx = 100
        if 'video_easy' in self._mode:
            video_dir = os.path.join('src/env/data', self._mode)
            self._video_paths = [os.path.join(
                video_dir, f'video{i}.mp4') for i in range(0, 10)]
        elif 'video_hard' in self._mode:
            video_dir = os.path.join('src/env/data', self._mode)
            self._video_paths = [os.path.join(
                video_dir, f'video{i}.mp4') for i in range(start_idx, end_idx)]
        else:
            raise ValueError(f'received unknown mode "{self._mode}"')

    def _load_video(self, video):
        """Load video from provided filepath and return as numpy array"""
        import cv2
        cap = cv2.VideoCapture(video)
        assert cap.get(
            cv2.CAP_PROP_FRAME_WIDTH) >= 100, 'width must be at least 100 pixels'
        assert cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT) >= 100, 'height must be at least 100 pixels'
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty((n, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
                       np.dtype('uint8'))
        i, ret = 0, True
        while (i < n and ret):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf[i] = frame
            i += 1
        cap.release()
        return np.moveaxis(buf, -1, 1)

    def _reset_video(self):
        self._index = (self._index + 1) % self._num_videos
        self._data = self._load_video(self._video_paths[self._index])

    def reset(self):
        if 'video' in self._mode:
            self._reset_video()
        self._current_frame = 0
        return self._greenscreen(self.env.reset())

    def step(self, action):
        self._current_frame += 1
        obs, reward, done, info = self.env.step(action)
        return self._greenscreen(obs), reward, done, info

    def _interpolate_bg(self, bg, size: tuple):
        """Interpolate background to size of observation"""
        bg = torch.from_numpy(bg).float().unsqueeze(0) / 255.
        bg = F.interpolate(bg, size=size, mode='bilinear', align_corners=False)
        return (bg * 255.).byte().squeeze(0).numpy()

    def _greenscreen(self, obs):
        """Applies greenscreen if video is selected, otherwise does nothing"""
        if 'video' in self._mode:
            bg = self._data[self._current_frame %
                            len(self._data)]  # select frame
            # scale bg to observation size
            bg = self._interpolate_bg(bg, obs.shape[1:])
            return do_green_screen(obs, bg)  # apply greenscreen
        return obs

    def apply_to(self, obs):
        """Applies greenscreen mode of object to observation"""
        obs = obs.copy()
        channels_last = obs.shape[-1] == 3
        if channels_last:
            obs = torch.from_numpy(obs).permute(2, 0, 1).numpy()
        obs = self._greenscreen(obs)
        if channels_last:
            obs = torch.from_numpy(obs).permute(1, 2, 0).numpy()
        return obs


def get_model_and_assets_from_setting_kwargs(model_fname, setting_kwargs=None):
    """"Returns a tuple containing the model XML string and a dict of assets."""
    assets = {filename: resources.GetResource(os.path.join(os.path.dirname(os.path.dirname(__file__)), filename))
              for filename in [
                  "./env/distracting_control/assets/materials.xml",
                  "./env/distracting_control/assets/skybox.xml",
                  "./env/distracting_control/assets/visual.xml",
              ]}

    if setting_kwargs is None:
        return common.read_model(model_fname), assets

    # Convert XML to dicts
    model = xmltodict.parse(common.read_model(model_fname))
    materials = xmltodict.parse(
        assets["./env/distracting_control/assets/materials.xml"])
    skybox = xmltodict.parse(
        assets["./env/distracting_control/assets/skybox.xml"])

    # Edit grid floor
    if 'grid_rgb1' in setting_kwargs:
        assert isinstance(
            setting_kwargs['grid_rgb1'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['texture']['@rgb1'] = \
            f'{setting_kwargs["grid_rgb1"][0]} {setting_kwargs["grid_rgb1"][1]} {setting_kwargs["grid_rgb1"][2]}'
    if 'grid_rgb2' in setting_kwargs:
        assert isinstance(
            setting_kwargs['grid_rgb2'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['texture']['@rgb2'] = \
            f'{setting_kwargs["grid_rgb2"][0]} {setting_kwargs["grid_rgb2"][1]} {setting_kwargs["grid_rgb2"][2]}'
    if 'grid_markrgb' in setting_kwargs:
        assert isinstance(
            setting_kwargs['grid_markrgb'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['texture']['@markrgb'] = \
            f'{setting_kwargs["grid_markrgb"][0]} {setting_kwargs["grid_markrgb"][1]} {setting_kwargs["grid_markrgb"][2]}'
    if 'grid_texrepeat' in setting_kwargs:
        assert isinstance(
            setting_kwargs['grid_texrepeat'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
        materials['mujoco']['asset']['material'][0]['@texrepeat'] = \
            f'{setting_kwargs["grid_texrepeat"][0]} {setting_kwargs["grid_texrepeat"][1]}'

    # Edit self
    if 'self_rgb' in setting_kwargs:
        assert isinstance(
            setting_kwargs['self_rgb'], (list, tuple, np.ndarray))
        assert materials['mujoco']['asset']['material'][1]['@name'] == 'self'
        materials['mujoco']['asset']['material'][1]['@rgba'] = \
            f'{setting_kwargs["self_rgb"][0]} {setting_kwargs["self_rgb"][1]} {setting_kwargs["self_rgb"][2]} 1'

    # Edit skybox
    if 'skybox_rgb' in setting_kwargs:
        assert isinstance(
            setting_kwargs['skybox_rgb'], (list, tuple, np.ndarray))
        assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
        skybox['mujoco']['asset']['texture']['@rgb1'] = \
            f'{setting_kwargs["skybox_rgb"][0]} {setting_kwargs["skybox_rgb"][1]} {setting_kwargs["skybox_rgb"][2]}'
    if 'skybox_rgb2' in setting_kwargs:
        assert isinstance(
            setting_kwargs['skybox_rgb2'], (list, tuple, np.ndarray))
        assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
        skybox['mujoco']['asset']['texture']['@rgb2'] = \
            f'{setting_kwargs["skybox_rgb2"][0]} {setting_kwargs["skybox_rgb2"][1]} {setting_kwargs["skybox_rgb2"][2]}'
    if 'skybox_markrgb' in setting_kwargs:
        assert isinstance(
            setting_kwargs['skybox_markrgb'], (list, tuple, np.ndarray))
        assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
        skybox['mujoco']['asset']['texture']['@markrgb'] = \
            f'{setting_kwargs["skybox_markrgb"][0]} {setting_kwargs["skybox_markrgb"][1]} {setting_kwargs["skybox_markrgb"][2]}'

    # Convert back to XML
    model_xml = xmltodict.unparse(model)
    assets["./env/distracting_control/assets/materials.xml"] = xmltodict.unparse(
        materials).encode("utf-8")
    assets["./env/distracting_control/assets/skybox.xml"] = xmltodict.unparse(
        skybox).encode("utf-8")

    return model_xml, assets


class SegmentWrapper(gym.Wrapper):
    def __init__(self, env, args, cuda_idx=0):
        gym.Wrapper.__init__(self, env)
        self.stack_num = args.frame_stack
        self.image_crop_size = args.image_crop_size
        self.masked_region_num = args.masked_region_num
        self.max_area = args.max_area
        self.min_area = args.min_area
        self.clip_range = args.clip_range
        self.reverse_sort = args.reverse_sort
        self.color_type = args.color_type
        self.add_original_frame = args.add_original_frame
        self.env_name = args.domain_name
        sam = sam_model_registry[args.model_type](checkpoint=args.model_path).to(
            device="cuda:{}".format(str(cuda_idx)))
        sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(sam, pred_iou_thresh=args.pred_iou_thresh,
                                                        stability_score_thresh=args.stability_score_thresh,
                                                        points_per_side=args.points_per_side,
                                                        points_per_batch=args.points_per_batch)
        self.ball_mask = None
        shp = env.observation_space.shape
        if self.color_type == "rgb":
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=((shp[0] * args.masked_region_num,) + shp[1:]),
                dtype=np.uint8
            )
        elif self.color_type == "gray":
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=((args.masked_region_num,) + shp[1:]),
                dtype=np.uint8
            )

        self.plot = args.plot_segment
        self.timer = args.segment_timer

    def mask_filter(self, masks, image, initial_flag=False, overlap_threshold=0.95):
        if len(masks) == 0:
            return []

        masks = sorted(masks, key=(
            lambda x: x['area']), reverse=self.reverse_sort)
        filtered_masks = []
        for mask in masks:
            if mask['area'] > self.max_area or mask['area'] < self.min_area:
                continue
            if np.sum(mask['segmentation'][self.clip_range[0]: self.clip_range[1],
                      self.clip_range[0]: self.clip_range[1]]) == 0:
                continue

            contain_flag = False
            for previous_mask in filtered_masks:
                if np.sum(mask['segmentation'] * previous_mask['segmentation']) > overlap_threshold * np.sum(
                        mask['segmentation']):
                    contain_flag = True
                    break
            if contain_flag:
                continue

            filtered_masks.append(mask)

        if len(filtered_masks) == 0:
            return []

        if len(filtered_masks) > self.masked_region_num:
            filtered_masks = filtered_masks[:self.masked_region_num]
        elif len(filtered_masks) < self.masked_region_num:
            black_mask = copy.deepcopy(masks[0])
            black_mask['segmentation'] = np.zeros_like(
                black_mask['segmentation'])
            filtered_masks = filtered_masks + \
                             [black_mask] * (self.masked_region_num - len(filtered_masks))

        if self.add_original_frame:
            white_mask = copy.deepcopy(masks[0])
            white_mask['segmentation'] = np.ones_like(
                white_mask['segmentation'])
            filtered_masks = filtered_masks + [white_mask]

        return filtered_masks

    def _generate_image_mask(self, image, initial_flag=False, dtype=np.uint8):
        start = time.time()
        assert image.shape[0] in [1, 3], "Image can only be gray or rgb"
        if image.shape[0] == 1:
            image = np.concatenate([image, image, image], axis=0)
        image = np.transpose(image, [1, 2, 0])
        masks = self.mask_generator.generate(image)
        masks = self.mask_filter(masks, image, initial_flag)

        if len(masks) == 0:
            print("No mask detected!")
            plt.imshow(image)
            plt.savefig("./figures/alert.jpg")
            if self.color_type == "rgb":
                black_image = np.zeros((3, self.image_crop_size, self.image_crop_size))
                total_images = [image.transpose((2, 0, 1))] + [black_image] * (self.masked_region_num - 1)
                if self.add_original_frame:
                    total_images = total_images + [image.transpose((2, 0, 1))]
                total_images = np.concatenate(total_images, axis=0).astype(dtype)
                return total_images
            elif self.color_type == "gray":
                black_image = np.zeros((1, self.image_crop_size, self.image_crop_size))
                gray_image = image.mean(axis=2).astype("uint8")
                total_images = [np.expand_dims(gray_image, axis=0)] + [black_image] * (self.masked_region_num - 1)
                if self.add_original_frame:
                    total_images = total_images + [np.expand_dims(gray_image, axis=0)]
                total_images = np.concatenate(total_images).astype(dtype)
                return total_images

        if self.color_type == "rgb":
            total_masks = []
            for item in masks:
                m = np.expand_dims(item["segmentation"], axis=0)
                m = np.expand_dims(np.concatenate(
                    [m for _ in range(3)], axis=0), axis=0)
                total_masks.append(m)
            total_masks = np.concatenate(total_masks, axis=0)
            total_images = np.concatenate(
                [np.expand_dims(image.transpose((2, 0, 1)), axis=0) for _ in range(len(masks))])
            masked_images = (total_masks * total_images).reshape(len(masks) * 3, self.image_crop_size,
                                                                 self.image_crop_size)

            zero_position_sum = np.sum(
                masked_images.reshape((3, 10, self.image_crop_size, self.image_crop_size))[:, 0, :, :], axis=(0, 1, 2))
            if zero_position_sum == 0:
                print("zero position is black")
                masked_images[:3, :, :] = image.transpose((2, 0, 1))
                plt.imshow(masked_images[:3, :, :].transpose(1, 2, 0))
                plt.savefig("./figures/zero.jpg")
        elif self.color_type == "gray":
            total_masks = []
            for item in masks:
                m = np.expand_dims(item["segmentation"], axis=0)
                total_masks.append(m)
            total_masks = np.concatenate(total_masks, axis=0)
            gray_image = image.mean(axis=2).astype("uint8")
            total_images = np.concatenate(
                [np.expand_dims(gray_image, axis=0) for _ in range(len(masks))])
            masked_images = total_masks * total_images

        if self.timer:
            print(time.time() - start)
        return masked_images.astype(dtype)

    def reset(self):
        obs = self.env.reset()
        return self._generate_image_mask(obs, initial_flag=True)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._generate_image_mask(obs), reward, done, info


class robosuite_wrapper(gym.Wrapper):
    def __init__(self, env, action_repeat):
        gym.Wrapper.__init__(self, env)
        self.key = env.camera_names[0] + "_image"
        shp = env.observation_spec()[self.key].shape
        shp = ((shp[2],) + shp[0:2])
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shp,
            dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_spec[0].shape,
            dtype=np.float32
        )

        self._env = env
        self.action_repeat = action_repeat

    def reset(self):
        obs = self._env.reset()[self.key]
        obs = obs.transpose((2, 0, 1))
        obs = np.copy(obs[:, ::-1, :])
        return obs

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            obs, reward_single, done, extra = self._env.step(action)
            reward = reward + reward_single
            if done:
                break
        obs = obs[self.key].transpose((2, 0, 1))
        obs = np.copy(obs[:, ::-1, :])
        return obs, reward, done, extra
