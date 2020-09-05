from gym.envs.registration import register

register(
    id='RoboRugby-v0',
    entry_point='robo_rugby.gym_env:GameEnv'
)
