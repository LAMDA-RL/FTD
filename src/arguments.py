import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--domain_name', default='walker',
                        choices=['walker', 'finger', 'cheetah', 'hopper', 'cartpole', 'pendulum', 'franka',
                                 'robosuite'])
    parser.add_argument('--task_name', default='walk')
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--episode_length', default=1000, type=int)
    parser.add_argument('--train_mode', default='video_hard', type=str,
                        choices=['train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'none'],
                        help='video_hard is used in FTD experiment')
    parser.add_argument('--eval_mode', default='video_hard', type=str,
                        choices=['train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'none'],
                        help='video_hard is used in FTD experiment')
    parser.add_argument('--color_type', default="rgb", type=str, choices=['rgb'],
                        help='FTD performs well on rgb mode, but fails in gray mode to due poor performance of SAM')
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--cuda_idx', default=0, type=int)

    # agent
    parser.add_argument('--algorithm', default='ftd', type=str,
                        choices=['sac', 'sam_sac', 'ftd', 'dbc', 'drqv2', 'mico', 'q2'])
    parser.add_argument('--train_steps', default='500k', type=str)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_steps', default='10k', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--capacity', default='100k', type=str, help='size of replay buffer')
    parser.add_argument('--max_grad_norm', default=5, type=float, help='used in grad clip')

    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)

    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)

    # architecture
    parser.add_argument('--num_shared_layers', default=11, type=int)
    parser.add_argument('--num_head_layers', default=0, type=int)
    parser.add_argument('--num_selector_layers', default=5, type=int)
    parser.add_argument('--num_pred_layers', default=5, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--projection_dim', default=100, type=int)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--attention_heads', default=4, type=int)

    parser.add_argument('--embed_dim', default=128, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--mlp_ratio', default=1., type=float)
    parser.add_argument('--qvk_bias', default=False, action='store_true')

    # entropy maximization
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)

    # auxiliary tasks
    parser.add_argument('--reward_factor', default=1.0,
                        type=float, help="Factor for reward loss")
    parser.add_argument('--inv_factor', default=1.0, type=float,
                        help="Factor for inverse dynamic loss")
    parser.add_argument('--fwd_factor', default=0.0, type=float,
                        help="Factor for forward dynamic loss, currently not used in FTD")
    parser.add_argument('--reward_accumulate_steps', default=1, type=int,
                        help='reward prediction using multi_step rewards, currently not used in FTD')
    parser.add_argument('--inv_accumulate_steps', default=1, type=int,
                        help='multi_step inverse dynamic prediction, currently not used in FTD')
    parser.add_argument('--fwd_accumulate_steps', default=1, type=int,
                        help='multi_step forward dynamic prediction, currently not used in FTD')
    parser.add_argument('--selector_lr', default=1e-4, type=float)
    parser.add_argument('--selector_beta', default=0.9, type=float)
    parser.add_argument('--unsupervised_update_freq', default=1, type=int)
    parser.add_argument('--unsupervised_update_num', default=1, type=int)
    parser.add_argument('--unsupervised_update_slow_freq', default='50k', type=str,
                        help='to save computational resources, '
                             'the frequency of self-supervised update gradually decreases during training')
    parser.add_argument('--unsupervised_warmup_steps', default='10k', type=str,
                        help='self-supervised update will not be called initially')
    parser.add_argument('--reward_first_sampling', default=False, type=bool,
                        help='used to improve performance of FTD in sparse-reward environments')
    parser.add_argument('--reward_first_capacity', default='5k', type=str)

    # eval
    parser.add_argument('--save_freq', default='100k', type=str, help='frequency of saving models')
    parser.add_argument('--eval_freq', default='5k', type=str, help='frequency of evaluation')
    parser.add_argument('--save_video_freq', default='10k', type=str, help='frequency of saving videos')
    parser.add_argument('--eval_episodes', default=1, type=int, help='episodes of each evaluation')
    parser.add_argument('--use_wandb', default=False, action='store_true',
                        help='whether use wandb to record tensorboard')

    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--save_video', default=False, action='store_true')

    # sam
    parser.add_argument('--apply_sam', action='store_false', help='whether apply sam to segment original frames')
    parser.add_argument('--model_type', default="vit_t",
                        type=str, help='model type of sam, currently only support vit_t')
    parser.add_argument('--pred_iou_thresh', default=0.5, type=float)
    parser.add_argument('--stability_score_thresh', default=0.5, type=float)
    parser.add_argument('--masked_region_num', default=9, type=int,
                        help="maximum number of segments")
    parser.add_argument('--max_area', default=2000, type=float,
                        help="segments larger than this area will be removed")
    parser.add_argument('--min_area', default=100, type=float,
                        help="segments smaller than this area will be removed")
    parser.add_argument('--clip_range', nargs="+", default=[0, 84], type=int,
                        help="segments outside this range will be removed")
    parser.add_argument('--reverse_sort', default=True, type=bool)
    parser.add_argument('--points_per_side', default=8, type=int)
    parser.add_argument('--points_per_batch', default=64, type=int)
    parser.add_argument('--add_original_frame', action='store_false',
                        help='whether add original frame in the segmented observation, depending on the algorithm')

    parser.add_argument('--plot_segment', default=False, action='store_true',
                        help='whether plot the segments')
    parser.add_argument('--plot_selected', default=False, action='store_true',
                        help='whether plot the selected frames')
    parser.add_argument('--segment_timer', default=False, action='store_true',
                        help='whether print the time consumption of segmentation')

    # dbc
    parser.add_argument('--bisim_coef', default=0.5, type=float)
    parser.add_argument('--transition_model_type', default="prob", type=str)
    parser.add_argument('--encoder_max_norm', default=False, action='store_true')
    parser.add_argument('--decoder_weight_lambda', default=3e-6, type=float)
    parser.add_argument('--dbc_dyn_loss', default='mse', type=str)
    parser.add_argument('--load_video', default=False, action='store_true')

    # drqv2
    parser.add_argument('--n_step', default=3, type=int)
    parser.add_argument('--num_expl_steps', default=2000, type=int)
    parser.add_argument('--stddev_schedule', default='linear(1.0,0.1,100000)',
                        type=str)  # medium: linear(1.0,0.1,500000)
    parser.add_argument('--stddev_clip', default=0.3, type=float)

    # mico
    parser.add_argument('--mico_weight', default=1.0, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--max_norm', default=10, type=float)

    # q2
    parser.add_argument('--dsa_act_update_freq', default=1, type=int)
    parser.add_argument('--noise_scale', default=0.05, type=float)

    args = parser.parse_args()

    assert args.seed is not None, 'must provide seed for experiment'
    assert args.log_dir is not None, 'must provide a log directory for experiment'
    assert not (args.masked_region_num != 9 and args.save_video), "currently only support saving video under 9 segments"

    if args.eval_mode == 'none':
        args.eval_mode = None

    args.image_size = 84
    args.image_crop_size = 84

    args.max_episode_steps = (args.episode_length +
                              args.action_repeat - 1) // args.action_repeat
    args.channels = 3 if args.color_type == "rgb" else 1
    args.model_path = "src/mobile_sam/weights/mobile_sam.pt"

    # parameters adjust for ftd-related methods
    if args.algorithm == 'ftd':
        args.apply_sam = True
        args.add_original_frame = True
    elif args.algorithm == 'sam_sac':
        args.apply_sam = True
        args.add_original_frame = False
        args.plot_segment = False
        args.plot_selected = False
    else:
        args.apply_sam = False
        args.add_original_frame = False
        args.plot_segment = False
        args.plot_selected = False

    # parameter specification for each task
    if args.domain_name == 'walker':
        args.max_area = 2000
        args.min_area = 100
        args.action_repeat = 4
    elif args.domain_name == 'finger':
        args.max_area = 300
        args.min_area = 100
        args.action_repeat = 1
        args.clip_range = [20, 64]
    elif args.domain_name == 'cheetah':
        args.max_area = 300
        args.min_area = 100
        args.action_repeat = 4
    elif args.domain_name == 'hopper':
        args.max_area = 2000
        args.min_area = 100
        args.action_repeat = 4
        args.init_steps = '10k'
        args.unsupervised_warmup_steps = '10k'
        args.reward_first_sampling = True
    elif args.domain_name == 'cartpole':
        args.max_area = 300
        args.min_area = 100
        args.action_repeat = 4
    elif args.domain_name == 'fish':
        args.max_area = 1000
        args.min_area = 100
        args.action_repeat = 4
        args.camera_id = 0
    elif args.domain_name == 'pendulum':
        args.max_area = 300
        args.min_area = 100
        args.action_repeat = 4
        args.init_steps = '10k'
        args.unsupervised_warmup_steps = '10k'
        args.reward_first_sampling = True
    elif args.domain_name == 'franka':
        args.max_area = 2000
        args.min_area = 100
        args.action_repeat = 4
        args.clip_range = [10, 74]
        args.init_steps = '100k'
        args.unsupervised_warmup_steps = '100k'
    elif args.domain_name == 'robosuite':
        args.max_area = 2000
        args.min_area = 100
        args.action_repeat = 1
        args.train_steps = '1000k'

    # special for drq-v2, to ensure convergence
    if args.algorithm == 'drqv2':
        args.actor_lr = 1e-4
        args.critic_lr = 1e-4
        args.batch_size = 512
        args.projection_dim = 50

    args.train_steps = int(args.train_steps.replace('k', '000'))
    args.save_freq = int(args.save_freq.replace('k', '000'))
    args.eval_freq = int(args.eval_freq.replace('k', '000'))
    args.capacity = int(args.capacity.replace('k', '000'))
    args.init_steps = int(args.init_steps.replace('k', '000'))
    args.save_video_freq = int(args.save_video_freq.replace('k', '000'))
    args.unsupervised_update_slow_freq = int(args.unsupervised_update_slow_freq.replace('k', '000'))
    args.reward_first_capacity = int(args.reward_first_capacity.replace('k', '000'))
    args.unsupervised_warmup_steps = int(args.unsupervised_warmup_steps.replace('k', '000'))

    return args
