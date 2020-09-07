from __future__ import absolute_import, division, print_function

from MyUtils import Stage
import pygame
import random
from robo_rugby.gym_env.RoboRugby import get_tf_wrapped_robo_rugby_env
import robo_rugby.gym_env.RR_Constants as const

import base64
import imageio
# import IPython
import matplotlib.pyplot as plt
import os
import reverb
import tempfile
import PIL.Image
import sys
import datetime

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
# from tf_agents.environments import suite_pybullet
# from tf_agents.environments import suite_gym
from tf_agents.experimental.train import actor
from tf_agents.experimental.train import learner
from tf_agents.experimental.train import triggers
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.experimental.train.utils import strategy_utils
from tf_agents.experimental.train.utils import train_utils
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils

tempdir = tempfile.gettempdir()

def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


"""
Tensorflow example tutorial from here:

https://www.tensorflow.org/agents/tutorials/7_SAC_minitaur_tutorial

but adapted to try and play RoboRugby instead of drive a robot.

"""

class HyperParms:  # Hyper Parameters
    # Use "num_iterations = 1e6" for better results (2 hrs)
    # 1e5 is just so this doesn't take too long (1 hr)
    # num_iterations = 100000  # @param {type:"integer"}
    num_iterations = 20000  # @param {type:"integer"}

    initial_collect_steps = 10000  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_capacity = 10000  # @param {type:"integer"}

    batch_size = 256  # @param {type:"integer"}

    critic_learning_rate = 3e-4  # @param {type:"number"}
    actor_learning_rate = 3e-4  # @param {type:"number"}
    alpha_learning_rate = 3e-4  # @param {type:"number"}
    target_update_tau = 0.005  # @param {type:"number"}
    target_update_period = 1  # @param {type:"number"}
    gamma = 0.99  # @param {type:"number"}
    reward_scale_factor = 1.0  # @param {type:"number"}

    actor_fc_layer_params = (256, 256)
    critic_joint_fc_layer_params = (256, 256)

    log_interval = 1000  # @param {type:"integer"}

    num_eval_episodes = 20  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    policy_save_interval = 5000  # @param {type:"integer"}

env = get_tf_wrapped_robo_rugby_env()
print('Observation Spec:')
print(env.time_step_spec().observation)
print('Action Spec:')
print(env.action_spec())
collect_env = get_tf_wrapped_robo_rugby_env()
eval_env = get_tf_wrapped_robo_rugby_env()
objStrategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

specObservation, specAction, specTimeStep = (spec_utils.get_tensor_specs(collect_env))

with objStrategy.scope():
    # Critic network trains the Actor network
    nnCritic = critic_network.CriticNetwork(
        (specObservation, specAction),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=HyperParms.critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform'
    )

with objStrategy.scope():
    nnActor = actor_distribution_network.ActorDistributionNetwork(
        specObservation,
        specAction,
        fc_layer_params=HyperParms.actor_fc_layer_params,
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork
        )
    )

with objStrategy.scope():
    train_step = train_utils.create_train_step()

    tf_agent = sac_agent.SacAgent(
        specTimeStep,
        specAction,
        actor_network=nnActor,
        critic_network=nnCritic,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=HyperParms.actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=HyperParms.critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=HyperParms.alpha_learning_rate),
        target_update_tau=HyperParms.target_update_tau,
        target_update_period=HyperParms.target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=HyperParms.gamma,
        reward_scale_factor=HyperParms.reward_scale_factor,
        train_step_counter=train_step
    )

    tf_agent.initialize()

print(f" --  REPLAY BUFFER  ({now()})  -- ")
rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
    samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0
)
table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=HyperParms.replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([table])

reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server)

dataset = reverb_replay.as_dataset(
      sample_batch_size=HyperParms.batch_size, num_steps=2).prefetch(50)
experience_dataset_fn = lambda: dataset

print(f" --  POLICIES  ({now()})  -- ")
tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_eval_policy, use_tf_function=True)
tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy, use_tf_function=True)
random_policy = random_py_policy.RandomPyPolicy(collect_env.time_step_spec(), collect_env.action_spec())

print(f" --  ACTORS  ({now()})  -- ")
rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  reverb_replay.py_client,
  table_name,
  sequence_length=2,
  stride_length=1)

initial_collect_actor = actor.Actor(
  collect_env,
  random_policy,
  train_step,
  steps_per_run=HyperParms.initial_collect_steps,
  observers=[rb_observer])
initial_collect_actor.run()

env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
  collect_env,
  collect_policy,
  train_step,
  steps_per_run=1,
  metrics=actor.collect_metrics(10),
  summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
  observers=[rb_observer, env_step_metric])

eval_actor = actor.Actor(
  eval_env,
  eval_policy,
  train_step,
  episodes_per_run=HyperParms.num_eval_episodes,
  metrics=actor.eval_metrics(HyperParms.num_eval_episodes),
  summary_dir=os.path.join(tempdir, 'eval'),
)

print(f" --  LEARNERS  ({now()})  -- ")
saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        train_step,
        interval=HyperParms.policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000),
]

agent_learner = learner.Learner(
    tempdir,
    train_step,
    tf_agent,
    experience_dataset_fn,
    triggers=learning_triggers)

print(f" --  METRICS AND EVALUATION  ({now()})  -- ")
def get_eval_metrics():
  eval_actor.run()
  results = {}
  for metric in eval_actor.metrics:
    results[metric.name] = metric.result()
  return results

metrics = get_eval_metrics()

def log_eval_metrics(step, metrics):
  eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
  print('step = {0}: {1}'.format(step, eval_results))

log_eval_metrics(0, metrics)

print(f" --  TRAINING  ({now()})  -- ")

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = metrics["AverageReturn"]  # didn't we already do this? get_eval_metrics()["AverageReturn"]
returns = [avg_return]

print(f" --  LET US BEGIN! ({now()})  -- ")
for progress in range(HyperParms.num_iterations):
    sys.stdout.write("Iteration num: %d%%  \r" % (progress))
    sys.stdout.flush()
    # Training.
    collect_actor.run()
    loss_info = agent_learner.run(iterations=1)

    # Evaluating.
    step = agent_learner.train_step_numpy

    if HyperParms.eval_interval and step % HyperParms.eval_interval == 0:
        metrics = get_eval_metrics()
        log_eval_metrics(step, metrics)
        returns.append(metrics["AverageReturn"])

    if HyperParms.log_interval and step % HyperParms.log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

rb_observer.close()
reverb_server.stop()

print(f" --  VISUALIZATION  ({now()})  -- ")

steps = range(0, HyperParms.num_iterations + 1, HyperParms.eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()

num_episodes = 1
video_filename = f"sac_robo_{HyperParms.num_iterations}_iterations.mp4"
with imageio.get_writer(video_filename, fps=const.FRAMERATE) as video:
  for _ in range(num_episodes):
    time_step = eval_env.reset()
    video.append_data(eval_env.render())
    while not time_step.is_last():
      action_step = eval_actor.policy.action(time_step)
      time_step = eval_env.step(action_step.action)
      video.append_data(eval_env.render())

print("VICTORY?")