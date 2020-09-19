from gym.envs.registration import register
import robo_rugby.gym_env.RR_Constants as const

register(
    id='RoboRugby-v0',
    entry_point='robo_rugby.gym_env:GameEnv',
    max_episode_steps=const.GAME_LENGTH_STEPS,
    nondeterministic=True,  # opposing team will pretty much always be nondeterministic
    reward_threshold=1.0  # no idea
)

register(
    id='RoboRugbySimple-v0',
    entry_point='robo_rugby.gym_env.RR_Environments:SimpleChasePos',
    max_episode_steps=const.GAME_LENGTH_STEPS,
    nondeterministic=True,  # opposing team will pretty much always be nondeterministic
    reward_threshold=1.0  # no idea
)

register(
    id='RoboRugbySimpleDuel-v2',
    entry_point='robo_rugby.gym_env.RR_Environments:SimpleDuel2',
    max_episode_steps=const.GAME_LENGTH_STEPS,
    nondeterministic=True,  # opposing team will pretty much always be nondeterministic
    reward_threshold=1.0  # no idea
)
