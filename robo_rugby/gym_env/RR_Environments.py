import robo_rugby.gym_env.RR_ScoreKeepers as sk
import robo_rugby.gym_env.RR_EnvBase as base
import gym

"""
Create different possible entry point classes for loading environments with gym.
"""

class SimpleChasePos(sk.ChasePosBall, base.GameEnv_Simple_AngleDistObs):
    """ Simple (discrete) environment that just tries to get the stupid robots to chase the positive ball."""

class SimpleDuel(sk.PushPosBallsToGoal, sk.PushNegBallsFromGoal, base.GameEnv_Simple):
    """ Simple (discrete) environment that only awards points for pushing balls towards the right goal. """





def get_tf_wrapped_robo_rugby_env():
    """Wraps given gym environment with TF Agent's GymWrapper.
      Note that by default a TimeLimit wrapper is used to limit episode lengths
      to the default benchmarks defined by the registered environments.
      Args:
        gym_env: An instance of OpenAI gym environment.
        discount: Discount to use for the environment.
        max_episode_steps: Used to create a TimeLimitWrapper. No limit is applied
          if set to None or 0. Usually set to `gym_spec.max_episode_steps` in `load.
        gym_env_wrappers: Iterable with references to wrapper classes to use
          directly on the gym environment.
        time_limit_wrapper: Wrapper that accepts (env, max_episode_steps) params to
          enforce a TimeLimit. Usuaully this should be left as the default,
          wrappers.TimeLimit.
        env_wrappers: Iterable with references to wrapper classes to use on the
          gym_wrapped environment.
        spec_dtype_map: A dict that maps gym specs to tf dtypes to use as the
          default dtype for the tensors. An easy way how to configure a custom
          mapping through Gin is to define a gin-configurable function that returns
          desired mapping and call it in your Gin config file, for example:
          `suite_gym.load.spec_dtype_map = @get_custom_mapping()`.
        auto_reset: If True (default), reset the environment automatically after a
          terminal state is reached.
        render_kwargs: Optional `dict` of keywoard arguments for rendering.
      Returns:
        A PyEnvironment instance.
      """
    from tf_agents.environments import suite_gym
    gym_spec = gym.spec("RoboRugby-v0")
    gym_env = gym_spec.make()
    return suite_gym.wrap_env(
        gym_env,
        discount=1.0,  # discount TODO research that more
        max_episode_steps=gym_spec.max_episode_steps,
        auto_reset=False)
    # gym_env_wrappers=gym_env_wrappers,
    # time_limit_wrapper=wrappers.TimeLimit,  # default
    # env_wrappers=env_wrappers,
    # spec_dtype_map=spec_dtype_map,