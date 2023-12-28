import json
import os
import torch
import wandb
import warnings
from collections import defaultdict
from termcolor import colored

warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

FORMAT_CONFIG = {
    'rl': {
        'train': [
            ('episode', 'E', 'int'), ('step', 'S', 'int'),
            ('duration', 'D', 'time'), ('episode_reward', 'R', 'float'),
            ('actor_loss', 'ALOSS', 'float'), ('critic_loss', 'CLOSS', 'float'),
            ('rewardpred_loss', 'RPredLOSS', 'float'), ('inversepred_loss', 'APredLOSS', 'float')
        ],
        'eval': [('step', 'S', 'int'),
                 ('episode_reward_test_env', 'ERTEST', 'float')]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04e'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, args, config='rl'):
        self._log_dir = log_dir
        self._train_mg = MetersGroup(
            os.path.join(log_dir, 'train.log'),
            formating=FORMAT_CONFIG[config]['train']
        )
        self._eval_mg = MetersGroup(
            os.path.join(log_dir, 'eval.log'),
            formating=FORMAT_CONFIG[config]['eval']
        )
        self.board = Board()
        self.board.set_log(SummaryWriter(log_dir=log_dir))

        self.use_wandb = args.use_wandb
        if self.use_wandb:
            wandb.init(project="sarl", config=args,
                       name=args.description, tags=args.description.split("_"))

    def log(self, key, value, step, n=1):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)
        self.board.log(key, value, step)
        if self.use_wandb:
            wandb.log({key: value}, step=step)

    def log_param(self, key, param, step):
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def dump(self, step):
        self._train_mg.dump(step, 'train')
        self._eval_mg.dump(step, 'eval')


class Singleton:
    def __new__(cls, *arg, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class Board(Singleton):
    logger = None

    def set_log(self, logger):
        self.logger = logger

    def log(self, key, value, idx):
        if self.logger is not None:
            self.logger.add_scalar(key, value, idx)