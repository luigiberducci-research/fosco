<h1 align="center">
FoSRL: A Tool for Formally-Guaranteed Safe Reinforcement Learning
</h1>

<p align="center">
<a href="https://opensource.org/license/bsd-3-clause/"><img alt="License" src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg"></a>
<a href="https://python.org"><img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg"></a>
<a href="https://github.com/luigiberducci-research/fosrl/actions/workflows/tests_on_push.yml/badge.svg"><img alt="Tests Status" src="https://github.com/luigiberducci-research/fosrl/actions/workflows/tests_on_push.yml/badge.svg"></a>
<a href="https://github.com/luigiberducci-research/fosrl/actions/workflows/linting_on_push.yml/badge.svg"><img alt="Linting Status" src="https://github.com/luigiberducci-research/fosrl/actions/workflows/linting_on_push.yml/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


**Learning framework** for Safe Reinforcement Learning with Control Barrier Functions (**CBFs**) 
for control-affine systems.

We combine **counterexample-guided pretraining** and **RL algorithms** to
learn a policy which is guaranteed to be safe to a set of environmental assumptions.

![Example CBF Single-Integrator](docs%2Fsingle_integrator.gif)

## :wrench: Installation 
The code is written in Python 3.10 and uses [PyTorch](https://pytorch.org/) for
learning a CBF.
We recommend using a virtual environment.

To install the required dependencies, run
```bash
pip install -r requirements.txt
```

## :rocket: Examples 
We provide a simple example for a single-integrator system in 
[`run_example.py`](run_example.py).

To run the example, run
```bash
python run_example.py
```

## :warning: Disclaimer
This is a research prototype, which builds on prior work on Lyapunov functions,
and extend it to support CBF and Robust CBF.
Our implementation refactored part of the original codebase.

However, if interested in other class of Lyapunov certificates,
we invite to refer to the [FOSSIL](https://github.com/oxford-oxcav/fossil) codebase.


# Known issues

1. Rendering in Mujoco env
If when using mujoco environments, you run into the following error
```
mujoco.FatalError: gladLoadGL error
```
Please have a look at [rendering in mujoco](https://pytorch.org/rl/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)
Ensure dependencies are there and env variables.
For ubuntu 20.04 install: 

`sudo apt-get install libglfw3 libglew2.1 libgl1-mesa-glx libosmesa6`


2. Video recording with Moviepy.
```
TypeError: must be real number, not NoneType
```
Solved by reinstalling moviepy and decorator packages
and reinstalling moviepy.
```
pip uninstall moviepy decorator
pip install moviepy
```
