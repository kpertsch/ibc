# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gets observation and action normalizers from data."""
import collections
import gin
import tqdm
from ibc.ibc import tasks
from ibc.ibc.train import stats
import ibc.ibc.utils.constants as constants
import tensorflow as tf
from tf_agents.networks import nest_map
from ibc.train.get_normalizers import NormalizationInfo
from ibc.ibc.train.stats import *


@gin.configurable
def compute_dataset_statistics_rb(dataset, num_samples,
                                  min_max_actions=False,
                                  use_sqrt_std=False):
  """Uses Chan's algorithm to compute mean, std in a single pass on reverb dataset.

  Args:
    dataset: Dataset to compute statistics on. Should return batches of (obs,
      action) tuples. If `nested` is not True obs, and actions should be
      flattened first.
    num_samples: Number of samples to take from the dataset.
    min_max_actions: If True, use [-1,1] instead of 0-mean, unit-var.
    use_sqrt_std: If True, divide by sqrt(std) instead of std for normalization.

  Returns:
    obs_norm_layer, per-dimension normalizer to 0-mean, unit-variance
    act_norm_layer, per-dimension normalizer to 0-mean, unit-variance
    min_action, shape [dim_A], per-dimension max actions in dataset
    max_action, shape [dim_A], per-dimension min actions in dataset
  """
  obs_statistics = None
  act_statistics = None

  def identity(x):
    return x

  def sqrt(x):
    return np.sqrt(x)

  if use_sqrt_std:
    get = sqrt
  else:
    get = identity

  with tqdm.tqdm(
      desc="Computing Dataset Statistics", total=num_samples) as progress_bar:

    observation = None
    action = None

    for trajectory in dataset.unbatch().take(num_samples):
      # format: [T, dim] for action and observation, where T=2
      observation, action = trajectory[0].observation[:, 0], trajectory[0].action[0]
      flat_obs = tf.nest.flatten(observation)
      flat_actions = tf.nest.flatten(action)

      if obs_statistics is None:
        # Initialize all params
        num_obs = len(flat_obs)
        num_act = len(flat_actions)

        if num_obs > 1:
          raise ValueError("Found too many observations, make sure you set "
                           "`nested=True` or you flatten them.")

        if num_act > 1:
          raise ValueError("Found too many actions, make sure you set "
                           "`nested=True` or you flatten them.")

        # [0] on the observation to take single value out of time dim.
        obs_statistics = [ChanRunningStatistics(o[0].numpy()) for o in flat_obs]
        act_statistics = [
            ChanRunningStatistics(a.numpy()) for a in flat_actions
        ]

        min_actions = [None for _ in range(num_act)]
        max_actions = [None for _ in range(num_act)]

      for obs, obs_stat in zip(flat_obs, obs_statistics):
        # Iterate over time dim.
        for o in obs:
          obs_stat.update_running_statistics(o.numpy())

      for act, act_stat in zip(flat_actions, act_statistics):
        act_stat.update_running_statistics(act.numpy())

      min_actions, max_actions = zip(*tf.nest.map_structure(
          stats._action_update,
          flat_actions,
          min_actions,
          max_actions,
          check_types=False))

      progress_bar.update(1)

  assert obs_statistics[0].n > 2

  obs_norm_layers = []
  act_norm_layers = []
  act_denorm_layers = []
  for obs_stat in obs_statistics:
    obs_norm_layers.append(
        StdNormalizationLayer(mean=obs_stat.mean, std=get(obs_stat.std)))

  for act_stat in act_statistics:
    if not min_max_actions:
      act_norm_layers.append(
          StdNormalizationLayer(mean=act_stat.mean, std=get(act_stat.std)))
      act_denorm_layers.append(
          StdDenormalizationLayer(mean=act_stat.mean, std=get(act_stat.std)))
    else:
      act_norm_layers.append(
          MinMaxNormalizationLayer(vmin=min_actions[0], vmax=max_actions[0]))
      act_denorm_layers.append(
          MinMaxDenormalizationLayer(vmin=min_actions[0], vmax=max_actions[0]))

  obs_norm_layers = obs_norm_layers[0]
  act_norm_layers = act_norm_layers[0]
  act_denorm_layers = act_denorm_layers[0]
  min_actions = min_actions[0]
  max_actions = max_actions[0]

  # Initialize act_denorm_layers:
  act_denorm_layers(min_actions)
  return (obs_norm_layers, act_norm_layers, act_denorm_layers, min_actions,
          max_actions)


@gin.configurable
def get_normalizers_rb(train_data,
                    batch_size,
                    env_name,
                    nested_obs=False,
                    nested_actions=False,
                    num_batches=100,
                    num_samples=None):
  """Computes stats and creates normalizer layers from stats for reverb dataset."""
  statistics_dataset = train_data
  assert not nested_obs and not nested_actions    # currently don't support nested obs/acts, since not needed for D4RL

  # You can either ask for num_batches (by default, used originally),
  # or num_samples (which doesn't penalize you for using bigger batches).
  if num_samples is None:
    num_samples = num_batches * batch_size

  # Create observation and action normalization layers.
  (obs_norm_layer, act_norm_layer, act_denorm_layer,
   min_actions, max_actions) = (
       compute_dataset_statistics_rb(
           statistics_dataset,
           num_samples=num_samples,))
           # nested_obs=nested_obs,
           # nested_actions=nested_actions))

  # # Define a function used to normalize training data inside a tf.data .map().
  # def norm_train_data_fn(obs_and_act, nothing):
  #   obs = obs_and_act[0]
  #   for img_key in constants.IMG_KEYS:
  #     if isinstance(obs, dict) and img_key in obs:
  #       obs[img_key] = tf.image.convert_image_dtype(
  #           obs[img_key], dtype=tf.float32)
  #   act = obs_and_act[1]
  #   normalized_obs = obs_norm_layer(obs)
  #   if isinstance(obs_norm_layer, nest_map.NestMap):
  #     normalized_obs, _ = normalized_obs
  #   normalized_act = act_norm_layer(act)
  #   if isinstance(act_norm_layer, nest_map.NestMap):
  #     normalized_act, _ = normalized_act
  #   return ((normalized_obs, normalized_act), nothing)

  norm_info = NormalizationInfo(obs_norm_layer,
                                act_norm_layer,
                                act_denorm_layer,
                                min_actions,
                                max_actions)
  return norm_info, None  # norm_train_data_fn
