# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger
from dopamine.utils import threading_utils
import gin.tf
import numpy as np
import queue
import tensorflow as tf


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  """Creates an agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: An Atari 2600 Gym environment.
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval'):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return Runner(base_dir, create_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent)
  elif schedule == 'async_train':
    return AsyncRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class Runner(object):
  """Object that handles running Dopamine experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  import dopamine.discrete_domains.atari_lib
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
  runner.run()
  ```
  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000,
               reward_clipping=(-1, 1)):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      reward_clipping: Tuple(int, int), with the minimum and maximum bounds for
        reward at each step. If `None` no clipping is applied.

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    assert base_dir is not None
    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._create_directories()
    self._summary_writer = tf.summary.FileWriter(self._base_dir)

    self._environment = create_environment_fn()
    # Set up a session and initialize variables.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self._sess = tf.Session('', config=config)

    self._agent = create_agent_fn(self._sess, self._environment,
                                  summary_writer=self._summary_writer)
    self._summary_writer.add_graph(graph=tf.get_default_graph())
    self._sess.run(tf.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)
    self._reward_clipping = reward_clipping

  def _create_directories(self):
    """Create necessary sub-directories."""
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix)
    self._start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self._checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      if self._agent.unbundle(
          self._checkpoint_dir, latest_checkpoint_version, experiment_data):
        assert 'logs' in experiment_data
        assert 'current_iteration' in experiment_data
        self._logger.data = experiment_data['logs']
        self._start_iteration = experiment_data['current_iteration'] + 1
        tf.logging.info('Reloaded checkpoint and will start from iteration %d',
                        self._start_iteration)

  def _initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    return self._begin_episode(initial_observation)

  def _run_one_step(self, action):
    """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, _ = self._environment.step(action)
    return observation, reward, is_terminal

  def _end_episode(self, reward):
    """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
    """
    self._agent.end_episode(reward)

  def _begin_episode(self, observation):
    return self._agent.begin_episode(observation)

  def _step(self, reward, observation):
    return self._agent.step(reward, observation)

  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    action = self._initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      # Perform reward clipping.
      if self._reward_clipping:
        min_bound, max_bound = self._reward_clipping
        reward = np.clip(reward, min_bound, max_bound)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward)
        action = self._begin_episode(observation)
      else:
        action = self._step(reward, observation)

    self._end_episode(reward)

    return step_number, total_reward

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    while step_count < min_steps:
      episode_length, episode_return = self._run_one_episode()
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      # We use sys.stdout.write instead of tf.logging so as to flush frequently
      # without generating a line break.
      sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Return: {}\r'.format(episode_return))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_train_phase(self, statistics):
    """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: The average reward generated in this phase.
    """
    # Perform the training phase, during which the agent learns.
    # TODO(#137): Replace `eval_mode` with TF ModeKeys.
    self._agent.eval_mode = False
    start_time = time.time()
    number_steps, sum_returns, num_episodes = self._run_one_phase(
        self._training_steps, statistics, 'train')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    time_delta = time.time() - start_time
    tf.logging.info('Average undiscounted return per training episode: %.2f',
                    average_return)
    tf.logging.info('Average training steps per second: %.2f',
                    number_steps / time_delta)
    return num_episodes, average_return

  def _run_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase(
        self._evaluation_steps, statistics, 'eval')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    tf.logging.info('Average undiscounted return per evaluation episode: %.2f',
                    average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    num_episodes_train, average_reward_train = self._run_train_phase(
        statistics)
    num_episodes_eval, average_reward_eval = self._run_eval_phase(
        statistics)

    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train, tag='Train')

    self._save_tensorboard_summaries(iteration, num_episodes_eval,
                                     average_reward_eval, tag='Eval')
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward, tag):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes: int, number of training episodes run.
      average_reward: float, The average reward.
      tag: str, Tag to apply to Tensorboard summaries (e.g `train`, `eval`).
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(
            tag='{}/NumEpisodes'.format(tag), simple_value=num_episodes),
        tf.Summary.Value(
            tag='{}/AverageReturns'.format(tag), simple_value=average_reward),
    ])
    self._summary_writer.add_summary(summary, iteration)

  def _log_experiment(self, iteration, statistics, suffix=''):
    """Records the results of the current iteration.

    Args:
      iteration: int, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
      suffix: string, suffix to add to the logging key.
    """
    self._logger['iteration_{:d}{}'.format(iteration, suffix)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration)

  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
    experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data)

  def _run_iterations(self):
    """Runs required number of training iterations sequentially.

    Statistics from each iteration are logged and exported for tensorboard.
    """
    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    tf.logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                         self._num_iterations, self._start_iteration)
      return

    self._run_iterations()


@gin.configurable
class TrainRunner(Runner):
  """Object that handles running experiments.

  The `TrainRunner` differs from the base `Runner` class in that it does not
  the evaluation phase. Checkpointing and logging for the train phase are
  preserved as before.
  """

  def __init__(self, base_dir, create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment):
    """Initialize the TrainRunner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
    """
    tf.logging.info('Creating TrainRunner ...')
    super(TrainRunner, self).__init__(base_dir, create_agent_fn,
                                      create_environment_fn)
    self._agent.eval_mode = False

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. This method differs from the `_run_one_iteration` method
    in the base `Runner` class in that it only runs the train phase.

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    num_episodes_train, average_reward_train = self._run_train_phase(
        statistics)

    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train, tag='Train')
    return statistics.data_lists


def _start_worker_thread(task_queue):
  """Starts and returns a thread working on tasks in provided queue.

  Tasks in `task_queue` needs to be stored as tuple of:
    - function: function taking positional arguments and returning None.
    - task: tuple of positional arguments to pass to the function.
  Each task is executed by calling `function(*task)`.

  The worker thread stops when a task `None` is added to the task queue and
  processed by the worker.

  Args:
    task_queue: Queue object containing tasks to perform.

  Returns:
    Thread object running and performing the tasks in `task_queue`.
  """
  def _worker(q):
    while True:
      item = q.get()
      if item is None:
        q.task_done()
        break
      function, task = item
      function(*task)
      q.task_done()
  thread = threading.Thread(target=_worker, args=(task_queue,))
  thread.start()
  return thread


# TODO(aarg): Add more details about this runner and the way thread and local
# variables are managed. This is somewhat hidden to the user.
@threading_utils.local_attributes(['_environment'])
@gin.configurable
class AsyncRunner(Runner):
  """Defines a train runner for asynchronous training.

  See `_run_one_iteration` for more details on how iterations are ran
  asynchronously.
  """

  def __init__(
      self, base_dir, create_agent_fn,
      create_environment_fn=atari_lib.create_atari_environment,
      num_simultaneous_iterations=1, **kwargs):
    """Creates an asynchronous runner.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      num_simultaneous_iterations: int, number of iterations running
        simultaneously in separate threads.
      **kwargs: Additional positional arguments.
    """
    threading_utils.initialize_local_attributes(
        self, _environment=create_environment_fn)
    self._eval_period = num_simultaneous_iterations
    self._num_simultaneous_iterations = num_simultaneous_iterations
    self._output_lock = threading.Lock()
    self._training_queue = queue.Queue(num_simultaneous_iterations)

    super(AsyncRunner, self).__init__(
        base_dir=base_dir, create_agent_fn=create_agent_fn,
        create_environment_fn=create_environment_fn, **kwargs)

  def _run_iterations(self):
    """Runs required number of training iterations sequentially.

    Statistics from each iteration are logged and exported for tensorboard.

    Iterations are run in multiple threads simultaneously (number of
    simultaneous threads is specified by `num_simultaneous_iterations`). Each
    time an iteration completes a new one starts until the right number of
    iterations is run.
    """
    experience_queue = queue.Queue()
    worker_threads = []

    for _ in range(self._num_simultaneous_iterations):
      worker_threads.append(_start_worker_thread(experience_queue))
    worker_threads.append(_start_worker_thread(self._training_queue))

    # TODO(westurner): See how to refactor the code to avoid setting an internal
    # attribute.
    self._completed_iteration = self._start_iteration
    for iteration in range(self._start_iteration, self._num_iterations):
      if (iteration + 1) % self._eval_period == 0:
        # TODO(aarg): Replace with ModeKeys.
        experience_queue.put((self._run_one_iteration, (iteration, True)))
      experience_queue.put((self._run_one_iteration, (iteration, False)))

    # Wait for all tasks to complete.
    experience_queue.join()
    self._training_queue.join()

    # Indicate workers to stop.
    for _ in range(self._num_simultaneous_iterations):
      experience_queue.put(None)
    self._training_queue.put(None)

    # Wait for all running threads to complete.
    for thread in worker_threads:
      thread.join()

  def _begin_episode(self, observation):
    # Increments training steps and blocks if training is too slow.
    self._enqueue_training_step()
    return self._agent.begin_episode(observation, training=False)

  def _step(self, reward, observation):
    # Increments training steps and blocks if training is too slow.
    self._enqueue_training_step()
    return self._agent.step(reward, observation, training=False)

  def _enqueue_training_step(self):
    """Increments training steps to run and blocks if training is too slow.

    If training is delayed, this will block episode generation until training
    catches up. This ensures that training orccurs simultaneously to episode
    generation with a constant training step / episode steps ratio.
    """
    if self._agent.eval_mode:
      return
    self._training_queue.put((self._agent.train_step, tuple([])))

  def _run_one_iteration(self, iteration, eval_mode):
    """Runs one iteration in separate thread, logs and checkpoints results.

    Same as parent Runner implementation except that summary statistics are
    directly logged instead of being returned.

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.
      eval_mode: bool, whether this is an evaluation iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    iteration_name = '{}iteration {}'.format(
        'eval ' if eval_mode else '', iteration)
    tf.logging.info('Starting %s.', iteration_name)
    run_phase = self._run_eval_phase if eval_mode else self._run_train_phase
    num_episodes, average_reward = run_phase(statistics)
    with self._output_lock:
      logging_iteration = iteration if eval_mode else self._completed_iteration
      self._log_experiment(
          logging_iteration, statistics, suffix='_eval' if eval_mode else '')
      self._save_tensorboard_summaries(
          logging_iteration, num_episodes, average_reward,
          tag='Eval' if eval_mode else 'Train')
      if not eval_mode:
        self._checkpoint_experiment(self._completed_iteration)
        self._completed_iteration += 1
    tf.logging.info('Completed %s.', iteration_name)
