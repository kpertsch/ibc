# Implicit Behavioral Cloning

This codebase contains the official implementation of the Implicit Behavioral Cloning (IBC) algorithm from the paper:

***Florence et al., [Implicit Behavioral Cloning (arxiv link)](https://arxiv.org/abs/2109.00137), Conference on Robotic Learning (CoRL) 2021.***

![](./docs/insert.gif)  |  ![](./docs/sort.gif)
:-------------------------:|:-------------------------:|

<img src="docs/energy_pop_teaser.png"/>

## Abstract

We find that across a wide range of robot policy learning scenarios, treating supervised policy learning with an implicit model generally performs better, on average, than commonly used explicit models. We present extensive experiments on this finding, and we provide both intuitive insight and theoretical arguments distinguishing the properties of implicit models compared to their explicit counterparts, particularly with respect to approximating complex, potentially discontinuous and multi-valued (set-valued) functions. On robotic policy learning tasks we show that implicit behavioral cloning policies with energy-based models (EBM) often outperform common explicit (Mean Square Error, or Mixture Density) behavioral cloning policies, including on tasks with high-dimensional action spaces and visual image inputs. We find these policies provide competitive results or outperform state-of-the-art offline reinforcement learning methods on the challenging human-expert tasks from the D4RL benchmark suite, despite using no reward information. In the real world, robots with implicit policies can learn complex and remarkably subtle behaviors on contact-rich tasks from human demonstrations, including tasks with high combinatorial complexity and tasks requiring 1mm precision.

## Prerequisites

The code for this project uses python 3.7+ and the following pip packages:

```
python3 -m pip install --upgrade pip
pip install \
  absl-py==0.12.0 \
  gin-config==0.4.0 \
  matplotlib==3.4.3 \
  mediapy==1.0.3 \
  pybullet==3.1.6 \
  scipy==1.7.1 \
  tensorflow==2.6.0 \
  tensorflow-probability==0.13.0 \
  tf-agents-nightly==0.10.0.dev20210930 \
  tqdm==4.62.2
```


For (optional) Mujoco / D4RL support, you also need to install:

```
pip install \
  mujoco_py==2.0.2.5 \
  git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

Note that the above will require that you have a local Mujoco installed (see
[here](https://github.com/openai/mujoco-py) for installation details).


## Quickstart

**Step 1**: Install listed Python packages above in  [Prerequisites](#Prequisites).

**Step 2**: Run unit tests (should take less than a minute):

```
cd <path_to>/ibc
./run_tests.sh
```

**Step 3**: Check that Tensorflow has GPU access:

```
python3 -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
```

If the above prints `False`, see the following requirements, notably CUDA 11.2 and cuDNN 8.1.0: https://www.tensorflow.org/install/gpu#software_requirements.

**Step 4**: Set PYTHONPATH to include the directory *just above `ibc`*.

```
cd <path_to>/ibc/..
export PYTHONPATH=$PYTHONPATH:${PWD}
```

**Step 5**: Let's do an example task, a Block Pushing task, so first let's **generate oracle data** (should take ~2 minutes):

```
python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=200 \
 --policy=oracle_push \
 --task=PUSH \
 --dataset_path=/tmp/blocks/dataset/oracle_push.tfrecord \
 --replicas=10  \
 --use_image_obs=False
```

**Step 6**: On that example Block Pushing task, let's do a **training + evaluation** with Implicit BC. The following trains at about 18 steps/sec on a GTX 2080 Ti, and an example training run gets to about ~100% success in 5,000 steps (roughly 5 minutes):

```
python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/mlp_ebm.gin \
  --task=PUSH \
  --tag=name_this_experiment \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='/tmp/blocks/dataset/oracle_push*.tfrecord'"
```


You're done with Quickstart!  See other sections below for [Codebase Overview](#codebase-overview), [Workflow](#workflow), and more [Tasks](#tasks).

## Codebase Overview

The highest level structure contains:


- `ibc/`
    - `data/` <-- tools to generate datasets, and feed data for training
    - `environments/` <-- a collection of environments
    - `networks/` <-- TensorFlow models for state inputs and/or vision inputs
    - ...

The above directories are algorithm-agnostic, and the implementation of specific algorithms
are mostly in:

- `ibc/ibc/`
    - `agents/` <-- holds the majority of the BC algorithm details, including:
        - `ibc_agent.py` <-- class for IBC training
        - `ibc_policy.py` <-- class for IBC inference
        - `mcmc.py` <-- implements optimizers used for IBC training/inference
        - similar files for MSE and MDN policies.
    - `losses/` <-- loss functions
        - `ebm_loss.py` <-- several different EBM-style loss functions.
        - `gradient_loss.py` <-- gradient penalty for Langevin
    - `configs/` <-- configurations for different trainings (including hyperparams)
    - ... other various utils for making training and evaluation happen.

A couple more notes for you the reader:

1. The codebase was optimized for large-scale experimentation and trying out many different ideas.  With hindsight it could be much simpler to implement a simplified version of only the core essentials.
2. The codebase heavily uses TF Agents, so we don't have to re-invent various wheels, and it is recommended you take a look at the Guide to get a sense: https://www.tensorflow.org/agents/overview




## Workflow

For each task we will **(1) acquire data** either by:

  - (a) Generating training data from scratch with scripted oracles (via `policy_eval.py`), **OR**
  - (b) Downloading training data from the web.

And then **(2) run a train+eval** by:

  - Running both training and evaluation in one script (via `train_eval.py`)

Note that each train+eval will spend a minute or two
computing normalization statistics, then start training with example printouts:

```
I1013 22:26:42.807687 139814213846848 triggers.py:223] Step: 100, 11.514 steps/sec
I1013 22:26:48.352215 139814213846848 triggers.py:223] Step: 200, 18.036 steps/sec
```

And at certain intervals (set in the configs), run evaluations:

```
I1013 22:19:30.002617 140341789730624 train_eval.py:343] Evaluating policy.
...
I1013 22:21:11.054836 140341789730624 actor.py:196]  
		 AverageReturn = 21.162763595581055
		 AverageEpisodeLength = 48.79999923706055
		 AverageFinalGoalDistance = 0.016136236488819122
		 AverageSuccessMetric = 1.0

```

There is **Tensorboard** support which can be obtained (for default configs) by running the following (and then going to `localhost:6006` in a browser.  (Might be slightly different for you to set up -- let us know if there are any issues.)

```
tensorboard --logdir /tmp/ibc_logs
```

And several chunks of useful information can be found in the train+eval log dirs for each experiment, which will end up for example at `/tmp/ibc_logs/mlp_ebm` after running the first suggested training.  For example `operative-gin-config.txt` will save out all the hyperparameters used for that training.



## Tasks

### Task: Block Pushing (from state observations)

We can either generate data from scratch (only takes a couple minutes):

```
python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=200 \
 --policy=oracle_push \
 --task=PUSH \
 --dataset_path=/tmp/blocks/dataset/oracle_push.tfrecord \
 --replicas=10  \
 --use_image_obs=False
```

Or we can download data from the web:

```
wget # TBD location on the web for the datasets
```

For **IBC** train+eval:

```
python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/mlp_ebm.gin \
  --task=PUSH \
  --tag=name_this_experiment \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='/tmp/blocks/dataset/oracle_push*.tfrecord'"
```

For **MSE** train+eval:

```
python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/mlp_mse.gin \
  --task=PUSH \
  --tag=name_this_experiment \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='/tmp/blocks/dataset/oracle_push*.tfrecord'"
```

For **MDN** train+eval:

```
python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/mlp_mdn.gin \
  --task=PUSH \
  --tag=name_this_experiment \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='/tmp/blocks/dataset/oracle_push*.tfrecord'"
```

### Task: Block Pushing (from image observations)

### Task: Particle

We can either generate data from scratch, for example for 2D (takes 15 seconds):

```
python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=200 \
 --policy=particle_green_then_blue \
 --task=PARTICLE \
 --dataset_path=/tmp/particle/dataset/2d_oracle_particle.tfrecord \
 --replicas=10  \
 --use_image_obs=False \
```

Or to do N-D, change `ParticleEnv.n_dim` in `third_party/py/ibc/environments/particle/particle.py`.

TODO(peteflorence): this should be settable in gin once this commit lands in a tf-agents release: https://github.com/tensorflow/agents/commit/1ef31b9a8a037924d6c33307650958130e1bb140

Or download all the data:

```
wget # TBD location on the web for the datasets
```

For **IBC** train+eval, the following trains at about 21 steps/sec on a GTX 2080 Ti, and an example training run gets to about ~100% success in 5,000 steps (roughly 5 minutes):

```
python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/mlp_ebm.gin \
  --task=PARTICLE \
  --tag=name_this_experiment \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='/tmp/particle/dataset/2d_oracle_particle*.tfrecord'" \
  --gin_bindings="ParticleEnv.n_dim=2"
```


### Task: D4RL Adroit and Kitchen


### Run Train Eval:

EBM:

```
# pen-human-v0
./ibc/main.sh \
  --mode=train \
  --task=pen-human-v0 \
  --gin_bindings="train_eval.root_dir='/tmp/ebm'" \
  --train_dataset_glob="$(pwd)/data/d4rl_trajectories/pen-human-v0/*.tfrecord" \
  --train_gin_file=mlp_ebm.gin \
  --train_tag=name_this_experiment
```


## Citation

If you found our paper/code useful in your research, please consider citing:

```
@article{
  author = {Pete Florence, Corey Lynch, Andy Zeng, Oscar Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, Jonathan Tompson},
  title = {Implicit Behavioral Cloning},
  journal = {Conference on Robotic Learning (CoRL)},
  month = {November},
  year = {2021},
}
```
