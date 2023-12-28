# SASeg w/o SEL
xvfb-run -a python3 src/train.py --algorithm sac --seed 0 --cuda_idx 0 --domain_name finger --task_name spin --train_mode video_hard --eval_mode video_hard --save_video --reward_factor 1.0 --inv_factor 1.0 --fwd_factor 0.0 --add_original_frame

# SASeg w/o SSO
xvfb-run -a python3 src/train.py --algorithm ftd --seed 0 --cuda_idx 0 --domain_name finger --task_name spin --train_mode video_hard --eval_mode video_hard --reward_factor 0.0 --inv_factor 0.0 --fwd_factor 0.0 --save_video --plot_selected --plot_segment

# SASeg w/o SEL&SSO
xvfb-run -a python3 src/train.py --algorithm sac --seed 0 --cuda_idx 0 --domain_name finger --task_name spin --train_mode video_hard --eval_mode video_hard --save_video --reward_factor 0.0 --inv_factor 0.0 --fwd_factor 0.0 --add_original_frame