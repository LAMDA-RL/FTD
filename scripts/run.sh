xvfb-run -a python3 src/train.py --algorithm $1 --seed 0 --cuda_idx 0 --domain_name $2 --task_name $3 --train_mode video_hard --eval_mode video_hard --reward_factor 1.0 --inv_factor 1.0 --fwd_factor 0.0 --save_video --plot_selected --plot_segment