from gym.envs.registration import register

register(
    id='BubbleShooter-v0',
    entry_point='gym_bubbleshooter.envs:BubbleShooterEnv',
    timestep_limit=10000
)
