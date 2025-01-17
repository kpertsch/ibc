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

"""Main binary to train a Behavioral Cloning agent."""
#  pylint: disable=g-long-lambda
import collections
import datetime
import functools
import os
import reverb
import rlds
import tensorflow_datasets as tfds

from absl import app
from absl import flags
from absl import logging
import gin
from ibc.environments.block_pushing import block_pushing  # pylint: disable=unused-import
from ibc.environments.block_pushing import block_pushing_discontinuous  # pylint: disable=unused-import
from ibc.environments.particle import particle  # pylint: disable=unused-import
from ibc.ibc import tasks
from ibc.ibc.utils import strategy_policy
from ibc.ibc.agents import ibc_policy  # pylint: disable=unused-import
from ibc.ibc.agents.ft_agent import IBCFinetuneAgent
from ibc.ibc.eval import eval_env as eval_env_module
from ibc.ibc.train import get_agent as agent_module
from ibc.ibc.train import get_cloning_network as cloning_network_module
from ibc.ibc.train import get_data as data_module
from ibc.ibc.train import get_eval_actor as eval_actor_module
from ibc.ibc.train import get_learner as learner_module
from ibc.ibc.train import get_learner_rb as learner_module_rb
from ibc.ibc.train import get_normalizers as normalizers_module
from ibc.ibc.train import get_normalizers_rb as normalizers_module_rb
from ibc.ibc.train import get_sampling_spec as sampling_spec_module
from ibc.ibc.utils import make_video as video_module
import tensorflow as tf
from tf_agents.examples.cql_sac.kumar20.data_utils import create_tf_record_dataset
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.metrics import py_metrics
from tf_agents.specs import tensor_spec
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.utils import common

flags.DEFINE_string('tag', None,
                    'Tag for the experiment. Appended to the root_dir.')
flags.DEFINE_bool('add_time', False,
                  'If True current time is added to the experiment path.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')
flags.DEFINE_bool('shared_memory_eval', False,
                  'If true the eval_env uses shared_memory.')
flags.DEFINE_bool('video', False,
                  'If true, write out one rollout video after eval.')
flags.DEFINE_multi_enum(
    'task', 'pen-human-v0',
    (tasks.IBC_TASKS + tasks.D4RL_TASKS),
    'If True the reach task is evaluated.')
flags.DEFINE_string('offline_dataset_name', 'd4rl_adroit_pen',
                    'D4RL dataset name.')
flags.DEFINE_boolean('viz_img', default=False,
                     help='Whether to save out imgs of what happened.')
flags.DEFINE_bool('skip_eval', False,
                  'If true the evals are skipped and instead run from '
                  'policy_eval binary.')
flags.DEFINE_bool('multi_gpu', False,
                  'If true, run in multi-gpu setting.')

flags.DEFINE_enum('device_type', 'gpu', ['gpu', 'tpu'],
                  'Where to perform training.')
flags.DEFINE_integer(
    'reverb_port', None,
    'Port for reverb server, if None, use a randomly chosen unused port.')

FLAGS = flags.FLAGS
VIZIER_KEY = 'success'
_SEQUENCE_LENGTH = 2
_STRIDE_LENGTH = 1


@gin.configurable
def train_eval(
    task=None,
    dataset_path=None,
    dataset_name=None,
    root_dir=None,
    # 'ebm' or 'mse' or 'mdn'.
    loss_type=None,
    # Name of network to train. see get_cloning_network.
    network=None,
    # Training params
    batch_size=512,
    num_iterations=20000,
    num_rl_iterations=1e6,
    learning_rate=1e-3,
    decay_steps=100,
    replay_capacity=100000,
    initial_collect_steps=10000,
    eval_interval=1000,
    eval_loss_interval=100,
    eval_episodes=1,
    fused_train_steps=100,
    data_shuffle_buffer_size=100,
    sequence_length=2,
    uniform_boundary_buffer=0.05,
    critic_learning_rate=3e-4,
    for_rnn=False,
    flatten_action=True,
    dataset_eval_fraction=0.0,
    goal_tolerance=0.02,
    tag=None,
    add_time=False,
    seed=0,
    viz_img=False,
    skip_eval=False,
    num_envs=1,
    reverb_port=None,
    offline_reverb_port=None,
    shared_memory_eval=False,
    image_obs=False,
    strategy=None,
    policy_save_interval=10000,
    replay_buffer_save_interval=100000,
    load_dataset_fn=tfds.load,
    # Use this to sweep amount of tfrecords going into training.
    # -1 for 'use all'.
    max_data_shards=-1,
    use_warmup=False):
  """Trains a BC agent on the given datasets."""
  if task is None:
    raise ValueError('task argument must be set.')
  logging.info(('Using task:', task))

  tf.random.set_seed(seed)
  if not tf.io.gfile.exists(root_dir):
    tf.io.gfile.makedirs(root_dir)

  # Logging.
  if tag:
    root_dir = os.path.join(root_dir, tag)
  if add_time:
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(root_dir, current_time)

  # Define eval env.
  eval_envs = []
  env_names = []
  for task_id in task:
    env_name = eval_env_module.get_env_name(task_id, shared_memory_eval,
                                            image_obs)
    logging.info(('Got env name:', env_name))
    eval_env = eval_env_module.get_eval_env(
        env_name, sequence_length, goal_tolerance, num_envs)
    logging.info(('Got eval_env:', eval_env))
    eval_envs.append(eval_env)
    env_names.append(env_name)

  obs_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(eval_envs[0]))

  # Create training and validation offline dataset.
  if not dataset_path.endswith('.tfrecord'):
      dataset_path = os.path.join(dataset_path, env_name,
                                  '%s*.tfrecord' % env_name)
  logging.info('Loading dataset from %s', dataset_path)
  dataset_paths = tf.io.gfile.glob(dataset_path)
  num_eval_shards = int(len(dataset_paths) * dataset_eval_fraction)
  num_train_shards = len(dataset_paths) - num_eval_shards
  train_shards = dataset_paths[:num_train_shards]
  if num_eval_shards > 0:
      eval_shards = dataset_paths[num_train_shards:]

  if not strategy:
    strategy = tf.distribute.get_strategy()

  with strategy.scope():
      train_data = create_tf_record_dataset(
          train_shards,
          batch_size,
          shuffle_buffer_size_per_record=1,
          shuffle_buffer_size=data_shuffle_buffer_size,
          num_shards=1,
          cycle_length=10,
          block_length=10,
          num_parallel_reads=None,
          num_parallel_calls=10,
          num_prefetch=10,
          strategy=strategy,
          reward_shift=0.0,
          action_clipping=None,
          use_trajectories=True)

      if num_eval_shards > 0:
          val_data = create_tf_record_dataset(
              dataset_paths,
              batch_size,
              shuffle_buffer_size_per_record=1,
              shuffle_buffer_size=data_shuffle_buffer_size,
              num_shards=1,
              cycle_length=10,
              block_length=10,
              num_parallel_reads=None,
              num_parallel_calls=10,
              num_prefetch=10,
              strategy=strategy,
              reward_shift=0.0,
              action_clipping=None,
              use_trajectories=True)
          dist_eval_data_iter = iter(       # properly distributed data loader
                  strategy.distribute_datasets_from_function(lambda: val_data))
      else:
          val_data = None
          dist_eval_data_iter = None

  (norm_info, _) = normalizers_module_rb.get_normalizers_rb(
      train_data, batch_size, env_name)

  # # Create dataset of TF-Agents trajectories from RLDS D4RL dataset.
  # #
  # # The RLDS dataset will be converted to trajectories and pushed to Reverb.
  # rlds_data = load_dataset_fn(dataset_name)['train']
  # trajectory_data_spec = rlds_to_reverb.create_trajectory_data_spec(rlds_data)
  # offline_table_name = 'offline_uniform_table'
  # offline_table = reverb.Table(
  #     name=offline_table_name,
  #     max_size=data_shuffle_buffer_size,
  #     sampler=reverb.selectors.Uniform(),
  #     remover=reverb.selectors.Fifo(),
  #     rate_limiter=reverb.rate_limiters.MinSize(1),
  #     signature=tensor_spec.add_outer_dim(trajectory_data_spec))
  # offline_reverb_server = reverb.Server([offline_table], port=offline_reverb_port)
  # offline_reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
  #     trajectory_data_spec,
  #     sequence_length=_SEQUENCE_LENGTH,
  #     table_name=offline_table_name,
  #     local_server=offline_reverb_server)
  # offline_rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  #     offline_reverb_replay.py_client,
  #     offline_table_name,
  #     sequence_length=_SEQUENCE_LENGTH,
  #     stride_length=_STRIDE_LENGTH,
  #     pad_end_of_episodes=False)
  #
  # rlds_to_reverb.push_rlds_to_reverb(rlds_data, offline_rb_observer)
  #
  # def _experience_dataset() -> tf.data.Dataset:
  #     """Reads and returns the experiences dataset from Reverb Replay Buffer."""
  #     return offline_reverb_replay.as_dataset(
  #         sample_batch_size=batch_size,
  #         num_steps=_SEQUENCE_LENGTH).prefetch(50)

  # create_train_and_eval_fns_unnormalized = data_module.get_data_fns(
  #     dataset_path,
  #     sequence_length,
  #     replay_capacity,
  #     batch_size,
  #     for_rnn,
  #     dataset_eval_fraction,
  #     flatten_action)
  # train_data, _ = create_train_and_eval_fns_unnormalized()
  # (norm_info, _) = normalizers_module_rb.get_normalizers_rb(
  #     train_data, batch_size, env_name)
  #
  # # Create normalized training data.
  # if not strategy:
  #   strategy = tf.distribute.get_strategy()
  # per_replica_batch_size = batch_size // strategy.num_replicas_in_sync
  # create_train_and_eval_fns = data_module.get_data_fns(
  #     dataset_path,
  #     sequence_length,
  #     replay_capacity,
  #     per_replica_batch_size,
  #     for_rnn,
  #     dataset_eval_fraction,
  #     flatten_action,
  #     norm_function=norm_train_data_fn,
  #     max_data_shards=max_data_shards)

  # # Create properly distributed eval data iterator.
  # dist_eval_data_iter = get_distributed_eval_data(create_train_and_eval_fns,
  #                                                 strategy)

  # Create normalization layers for obs and action.
  with strategy.scope():
    # Create train step counter.
    train_step = train_utils.create_train_step()

    # Define action sampling spec.
    action_sampling_spec = sampling_spec_module.get_sampling_spec(
        action_tensor_spec,
        min_actions=norm_info.min_actions,
        max_actions=norm_info.max_actions,
        uniform_boundary_buffer=uniform_boundary_buffer,
        act_norm_layer=norm_info.act_norm_layer)

    # This is a common opportunity for a bug, having the wrong sampling min/max
    # so log this.
    logging.info(('Using action_sampling_spec:', action_sampling_spec))

    # Define keras cloning network.
    cloning_network = cloning_network_module.get_cloning_network(
        network,
        obs_tensor_spec,
        action_tensor_spec,
        norm_info.obs_norm_layer,
        norm_info.act_norm_layer,
        sequence_length,
        norm_info.act_denorm_layer)

    # Define tfagent.
    agent = agent_module.get_agent(loss_type,
                                   time_step_tensor_spec,
                                   action_tensor_spec,
                                   action_sampling_spec,
                                   norm_info.obs_norm_layer,
                                   norm_info.act_norm_layer,
                                   norm_info.act_denorm_layer,
                                   learning_rate,
                                   use_warmup,
                                   cloning_network,
                                   train_step,
                                   decay_steps)

    # Define offline learner.
    offline_learner = learner_module_rb.get_learner(
        loss_type,
        root_dir,
        agent,
        train_step,
        lambda: train_data,
        fused_train_steps,
        strategy)

    # # TODO(karl): create online RL agent (w/ online RL _train() function)
    # rl_agent = agent = IBCFinetuneAgent(time_step_spec=time_step_tensor_spec,
    #                                     action_spec=action_tensor_spec,
    #                                     action_sampling_spec=action_sampling_spec,
    #                                     obs_norm_layer=norm_info.obs_norm_layer,
    #                                     act_norm_layer=norm_info.act_norm_layer,
    #                                     act_denorm_layer=norm_info.act_denorm_layer,
    #                                     cloning_network=cloning_network,
    #                                     optimizer=tf.keras.optimizers.Adam(learning_rate=critic_learning_rate),
    #                                     train_step_counter=train_step)
    #

    # create reverb online replay buffer
    table_name = 'uniform_table'
    table = reverb.Table(
        table_name,
        max_size=replay_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))

    reverb_checkpoint_dir = os.path.join(root_dir, "rl", "replay_checkpoints")
    reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
        path=reverb_checkpoint_dir)
    reverb_server = reverb.Server([table],
                                  port=reverb_port,
                                  checkpointer=reverb_checkpointer)
    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        sequence_length=2,
        table_name=table_name,
        local_server=reverb_server)
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client,
        table_name,
        sequence_length=2,
        stride_length=1)

    def experience_dataset_fn():
        return reverb_replay.as_dataset(
            sample_batch_size=batch_size, num_steps=2).prefetch(50)

    # create online RL actors
    greedy_policy = strategy_policy.StrategyPyTFEagerPolicy(
        agent.policy, strategy=strategy)
    initial_collect_actor = actor.Actor(
        eval_envs[0],
        greedy_policy,
        train_step,
        steps_per_run=initial_collect_steps,
        observers=[rb_observer])

    collect_actor = actor.Actor(
        eval_envs[0],
        greedy_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(root_dir, "rl"),
        observers=[rb_observer, py_metrics.EnvironmentSteps()])

    # create online RL learner
    saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            agent,
            train_step,
            interval=policy_save_interval,
            metadata_metrics={triggers.ENV_STEP_METADATA_KEY: py_metrics.EnvironmentSteps()}),
        triggers.ReverbCheckpointTrigger(
            train_step,
            interval=replay_buffer_save_interval,
            reverb_client=reverb_replay.py_client),
        # TODO(b/165023684): Add SIGTERM handler to checkpoint before preemption.
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    online_learner = learner.Learner(
        root_dir,
        train_step,
        agent,
        experience_dataset_fn,
        triggers=learning_triggers,
        strategy=strategy)

    # Define eval.
    eval_actors, eval_success_metrics = [], []
    for eval_env, env_name in zip(eval_envs, env_names):
      env_name_clean = env_name.replace('/', '_')
      eval_actor, success_metric = eval_actor_module.get_eval_actor(
          agent,
          env_name,
          eval_env,
          train_step,
          eval_episodes,
          root_dir,
          viz_img,
          num_envs,
          strategy,
          summary_dir_suffix=env_name_clean)
      eval_actors.append(eval_actor)
      eval_success_metrics.append(success_metric)

    get_eval_loss = tf.function(agent.get_eval_loss)

    # Get summary writer for aggregated metrics.
    aggregated_summary_dir = os.path.join(root_dir, 'eval')
    summary_writer = tf.summary.create_file_writer(
        aggregated_summary_dir, flush_millis=10000)
  logging.info('Saving operative-gin-config.')
  with tf.io.gfile.GFile(
      os.path.join(root_dir, 'operative-gin-config.txt'), 'wb') as f:
    f.write(gin.operative_config_str())

  # Main train and eval loop.
  while train_step.numpy() < (num_iterations + num_rl_iterations):
    if train_step.numpy() < num_iterations:
      # Run offline_learner for fused_train_steps.
      training_step(agent, offline_learner, fused_train_steps, train_step)

      if (dist_eval_data_iter is not None and
              train_step.numpy() % eval_loss_interval == 0):
          # Run a validation step.
          validation_step(
              dist_eval_data_iter, bc_learner, train_step, get_eval_loss)

    else:
      if train_step.numpy() == num_iterations:
          # before starting online RL, collect initial samples for replay buffer
          logging.info(f"Collecting {initial_collect_steps} samples to initialize online replay buffer.")
          initial_collect_actor.run()
          logging.info("Done!")
      # Run online RL for 1 step
      collect_actor.run()
      online_learner.run(iterations=1)

    if not skip_eval and train_step.numpy() % eval_interval == 0:

      all_metrics = []
      for eval_env, eval_actor, env_name, success_metric in zip(
          eval_envs, eval_actors, env_names, eval_success_metrics):
        # Run evaluation.
        metrics = evaluation_step(
            eval_episodes,
            eval_env,
            eval_actor,
            name_scope_suffix=f'_{env_name}')
        all_metrics.append(metrics)

        # rendering on some of these envs is broken
        if FLAGS.video and 'kitchen' not in task:
          if 'PARTICLE' in task:
            # A seed with spread-out goals is more clear to visualize.
            eval_env.seed(42)
          # Write one eval video.
          video_module.make_video(
              agent,
              eval_env,
              root_dir,
              step=train_step.numpy(),
              strategy=strategy)

      metric_results = collections.defaultdict(list)
      for env_metrics in all_metrics:
        for metric in env_metrics:
          metric_results[metric.name].append(metric.result())

      with summary_writer.as_default(), \
         common.soft_device_placement(), \
         tf.summary.record_if(lambda: True):
        for key, value in metric_results.items():
          tf.summary.scalar(
              name=os.path.join('AggregatedMetrics/', key),
              data=sum(value) / len(value),
              step=train_step)

  summary_writer.flush()
  rb_observer.close()
  reverb_server.stop()


def training_step(agent, bc_learner, fused_train_steps, train_step):
  """Runs bc_learner for fused training steps."""
  reduced_loss_info = None
  if not hasattr(agent, 'ebm_loss_type') or agent.ebm_loss_type != 'cd_kl':
    reduced_loss_info = bc_learner.run(iterations=fused_train_steps)
  else:
    for _ in range(fused_train_steps):
      # I think impossible to do this inside tf.function.
      agent.cloning_network_copy.set_weights(
          agent.cloning_network.get_weights())
      reduced_loss_info = bc_learner.run(iterations=1)

  if reduced_loss_info:
    # Graph the loss to compare losses at the same scale regardless of
    # number of devices used.
    with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(
        True):
      tf.summary.scalar(
          'reduced_loss', reduced_loss_info.loss, step=train_step)


def validation_step(dist_eval_data_iter, bc_learner, train_step,
                    get_eval_loss_fn):
  """Runs a validation step."""
  losses_dict = get_eval_loss_fn(next(dist_eval_data_iter))

  with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(
      True):
    common.summarize_scalar_dict(
        losses_dict, step=train_step, name_scope='Eval_Losses/')


def evaluation_step(eval_episodes, eval_env, eval_actor, name_scope_suffix=''):
  """Evaluates the agent in the environment."""
  logging.info('Evaluating policy.')
  with tf.name_scope('eval' + name_scope_suffix):
    # This will eval on seeds:
    # [0, 1, ..., eval_episodes-1]
    for eval_seed in range(eval_episodes):
      eval_env.seed(eval_seed)
      eval_actor.reset()  # With the new seed, the env actually needs reset.
      eval_actor.run()

    eval_actor.log_metrics()
    eval_actor.write_metric_summaries()
  return eval_actor.metrics


def get_distributed_eval_data(data_fn, strategy):
  """Gets a properly distributed evaluation data iterator."""
  _, eval_data = data_fn()
  dist_eval_data_iter = None
  if eval_data:
    dist_eval_data_iter = iter(
        strategy.distribute_datasets_from_function(lambda: eval_data))
  return dist_eval_data_iter


def main(_):
  logging.set_verbosity(logging.INFO)

  gin.add_config_file_search_path(os.getcwd())
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings,
                                      # TODO(coreylynch): This is a temporary
                                      # hack until we get proper distributed
                                      # eval working. Remove it once we do.
                                      skip_unknown=True)

  # For TPU, FLAGS.tpu will be set with a TPU address and FLAGS.use_gpu
  # will be False.
  # For GPU, FLAGS.tpu will be None and FLAGS.use_gpu will be True.
  strategy = strategy_utils.get_strategy(
      tpu=FLAGS.tpu, use_gpu=FLAGS.use_gpu)

  task = FLAGS.task or gin.REQUIRED
  # If setting this to True, change `my_rangea in mcmc.py to `= range`
  tf.config.experimental_run_functions_eagerly(False)

  train_eval(
      task=task,
      tag=FLAGS.tag,
      add_time=FLAGS.add_time,
      viz_img=FLAGS.viz_img,
      skip_eval=FLAGS.skip_eval,
      shared_memory_eval=FLAGS.shared_memory_eval,
      reverb_port=FLAGS.reverb_port,
      dataset_name=FLAGS.offline_dataset_name,
      strategy=strategy)


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
