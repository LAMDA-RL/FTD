from algorithms.dbc import BisimAgent
from algorithms.drqv2 import DrQV2Agent
from algorithms.image_selector_sac import Image_Selector_SAC
from algorithms.mico import MICo
from algorithms.q2 import Q2
from algorithms.sac import SAC

algorithm = {
    'sac': SAC,
    'sam_sac': SAC,
    'ftd': Image_Selector_SAC,
    'dbc': BisimAgent,
    'drqv2': DrQV2Agent,
    'mico': MICo,
    'q2': Q2
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
