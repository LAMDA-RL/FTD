import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


class VideoRecorder(object):
    def __init__(self, dir_name, plot_segment, plot_selected, height=84, width=84, channels=3, region_num=9,
                 stack_num=3,
                 camera_id=0, fps=25, save_freq=1e4):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.channels = channels
        self.region_num = region_num
        self.stack_num = stack_num
        self.camera_id = camera_id
        self.fps = fps
        self.plot_segment = plot_segment
        self.plot_selected = plot_selected
        self.frames = []
        self.save_freq = save_freq

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env, mode=None, selected_image=None, segmented_image=None, original_image=None):
        if self.enabled:
            if original_image is None:
                frame = env.render(
                    mode='rgb_array',
                    height=segmented_image.shape[0] if self.plot_segment else self.height,
                    width=segmented_image.shape[1] if self.plot_segment else self.width,
                    camera_id=self.camera_id
                )
                if mode is not None and 'video' in mode:
                    _env = env
                    while 'video' not in _env.__class__.__name__.lower():
                        _env = _env.env
                    frame = _env.apply_to(frame)
            else:
                frame = np.transpose(np.array(original_image)[-3:, :, :], (1, 2, 0))
                frame = Image.fromarray(frame).convert('RGB')
                frame = np.array(frame.resize((segmented_image.shape[0], segmented_image.shape[1])))

            # plt.imshow(frame)
            # plt.axis('off')
            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # plt.savefig("./figures/example.pdf", format="pdf")
            # assert 0

            if self.plot_selected:
                if self.channels == 1:
                    selected_image = np.concatenate([selected_image, selected_image, selected_image], axis=2)
                if self.plot_segment:
                    pil_image = Image.fromarray(selected_image).convert('RGB')
                    selected_image = np.array(pil_image.resize((segmented_image.shape[0], segmented_image.shape[1])))
                frame = np.concatenate([frame, selected_image], axis=1)
            if self.plot_segment:
                frame = np.concatenate([frame, segmented_image], axis=1)

            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
