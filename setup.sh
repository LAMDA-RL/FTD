#!/bin/bash

conda create -n ftd python=3.7 -y
conda activate ftd

pip install mujoco==2.3.0
pip install dm_control==1.0.8
pip uninstall dm_control -y
cd src/env/dm_control
python setup.py install
cd ../../..

pip install -r requirements.txt

pip install src/env/dmc2gym

pip install src/mobile_sam

export MUJOCO_GL=glfw