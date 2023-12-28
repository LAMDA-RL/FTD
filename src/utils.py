import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import torch
import torch.cuda
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from pynvml import *
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        'timestamp': str(datetime.now()),

        'args': vars(args)
    }
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
    path = os.path.join('setup', 'config.cfg')
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
    fpath = os.path.join(dir_path, f'*.{filetype}')
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return fpaths


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=np.uint8):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0] // 3

    def frame(self, i):
        return self._force()[i * 3:(i + 1) * 3]


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f'{count:,}'


def show_anns(image, anns):
    if len(anns) == 0:
        return
    plt.imshow(image)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m)))
    plt.axis('off')
    plt.savefig("./figures/segment.jpg")

    plt.imshow(image)
    plt.axis('off')
    plt.savefig("./figures/original.jpg")


def transfer_to_ann(obs, logits, channels, region_num, stack_num, text_height=12):
    obs = np.array(obs)
    if channels * region_num * stack_num == obs.shape[0]:
        obs = obs[-channels * region_num:]
    else:
        obs = obs[-channels * (region_num + 1): -channels]
    obs = obs.reshape([region_num, channels, obs.shape[-2], obs.shape[-1]])
    obs = np.transpose(obs, (0, 2, 3, 1))
    logits = [str(round(i, 2)) for i in logits]
    assert len(logits) == 9 or len(
        logits) == 16, "Currently only support 9 or 16 subfigures."
    side_num = int(np.sqrt(len(logits)))

    image_size = obs.shape[-2] + text_height
    total_images = Image.new(
        'RGB', (side_num * image_size, side_num * image_size), (255, 255, 255))
    for index in range(region_num):
        image = obs[index]
        logit = logits[index]

        if channels == 1:
            image = np.concatenate([image, image, image], axis=2)
        pil_image = Image.fromarray(image).convert('RGB')
        font = ImageFont.load_default()
        new_image = Image.new(
            'RGB', (pil_image.width + text_height, pil_image.height + text_height), (255, 255, 255))
        new_image.paste(pil_image, (text_height // 2, 0))
        draw = ImageDraw.Draw(new_image)
        draw.multiline_text((29 + text_height / 2, pil_image.height),
                            logit, fill=(65, 105, 225), font=font)
        total_images.paste(new_image, (index // side_num *
                                       image_size, index % side_num * image_size))

    return np.array(total_images)


def transfer_to_segments(obs, channels, region_num, stack_num, plot_num, blank_height=4):
    obs = np.array(obs)
    if channels * region_num * stack_num == obs.shape[0]:
        obs = obs[-channels * region_num:]
    else:
        obs = obs[-channels * (region_num + 1): -channels]
    obs = obs.reshape([region_num, channels, obs.shape[-2], obs.shape[-1]])
    obs = np.transpose(obs, (0, 2, 3, 1))
    side_num = int(np.sqrt(plot_num))

    for i in range(7):
        temp = obs[i]
        if np.sum((np.sum(temp, axis=-1) > 0)) != 0:
            judge = np.sum((temp[:, :, 1] >= 130) & (temp[:, :, 1] <= 160) & (temp[:, :, 2] >= 80) & (
                    temp[:, :, 2] <= 110)) / np.sum((np.sum(temp, axis=-1) > 0))
            if judge > 0.2:
                obs[random.randint(0, plot_num - 1)] = temp
                break

    image_size = obs.shape[-2] + blank_height
    total_images = Image.new(
        'RGB', (side_num * image_size, side_num * image_size), (255, 255, 255))

    for index in range(plot_num):
        image = obs[index]

        pil_image = Image.fromarray(image).convert('RGB')
        new_image = Image.new(
            'RGB', (pil_image.width + blank_height, pil_image.height + blank_height), (255, 255, 255))
        new_image.paste(pil_image, (blank_height // 2, 0))
        total_images.paste(new_image, (index // side_num *
                                       image_size, index % side_num * image_size))

    return np.array(total_images)


def show_gpu(simlpe=False):
    # 初始化
    nvmlInit()
    # 获取GPU个数
    deviceCount = nvmlDeviceGetCount()
    total_memory = 0
    total_free = 0
    total_used = 0
    gpu_name = ""
    gpu_num = deviceCount

    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_name = nvmlDeviceGetName(handle)
        # 查看型号、显存、温度、电源
        if not simlpe:
            print("GPU{}: {}".format(i, gpu_name), end="    ")
            print("Memory: {}G".format((info.total // 1048576) / 1024), end="    ")
            print("Free: {}G".format((info.free // 1048576) / 1024), end="    ")
            print("Used: {}G".format((info.used // 1048576) / 1024), end="    ")
            print("Usage: {}%".format(info.used / info.total))

        total_memory += (info.total // 1048576) / 1024
        total_free += (info.free // 1048576) / 1024
        total_used += (info.used // 1048576) / 1024

    print("Name: [{}], GPU num: [{}], Total mem: [{}G], Free mem: [{}G], Used mem: [{}G], Usage: [{}%]".format(
        gpu_name, gpu_num, total_memory, total_free, total_used, (total_used / total_memory)))
    # 关闭管理工具
    nvmlShutdown()


def check_parameters(model):
    problem_flag = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Parameter {name} contains NaN or Inf values.")
            problem_flag = True

    return problem_flag


def _sqrt(x, tol=0.):
    tol = torch.tensor(tol).type_as(x)
    return torch.sqrt(torch.max(x, tol))


def cosine_distance(x, y):
    cos_sim = F.cosine_similarity(x, y, dim=1)
    return torch.atan2(_sqrt(1. - cos_sim ** 2), cos_sim)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def representation_distances(first_representations, second_representations, beta=0.1,
                             return_distance_components=False):
    batch_size = first_representations.size(0)
    representation_dim = first_representations.size(-1)
    first_squared_reps = squarify(first_representations)
    first_squared_reps = torch.reshape(first_squared_reps,
                                       (batch_size ** 2, representation_dim))
    second_squared_reps = squarify(second_representations)
    second_squared_reps = second_squared_reps.permute(1, 0, 2)
    second_squared_reps = torch.reshape(second_squared_reps,
                                        (batch_size ** 2, representation_dim))
    base_distances = cosine_distance(first_squared_reps, second_squared_reps)
    norm_average = 0.5 * (torch.sum(first_squared_reps ** 2, -1) +
                          torch.sum(second_squared_reps ** 2, -1))
    if return_distance_components:
        return norm_average + beta * base_distances, norm_average, base_distances
    return (norm_average + beta * base_distances).unsqueeze(-1)


def target_distances(representations, reward_diffs, cumulative_gamma):
    next_state_distances = representation_distances(
        representations, representations)
    target_dis = reward_diffs + cumulative_gamma * next_state_distances
    target_dis = target_dis.detach()
    return target_dis


def reward_distance(rewards):
    squared_rews = squarify(rewards)
    squared_rews_transp = squared_rews.permute(1, 0, 2)
    squared_rews = torch.reshape(squared_rews, (squared_rews.shape[0] ** 2, 1))
    squared_rews_transp = torch.reshape(squared_rews_transp,
                                        (squared_rews_transp.shape[0] ** 2, 1))
    # have to be this in sarsa2
    reward_diffs = F.smooth_l1_loss(squared_rews, squared_rews_transp, reduction='none')
    # have to be this in mico
    # reward_diffs = F.l1_loss(squared_rews, squared_rews_transp, reduction='none')
    return reward_diffs


def squarify(x):
    batch_size = x.size(0)
    if len(x.size()) > 1:
        repre_dim = x.size(-1)
        return torch.reshape(x.repeat(1, batch_size), (batch_size, batch_size, repre_dim))
    return torch.reshape(x.repeat(batch_size), (batch_size, batch_size))
