from bubbleshooter_env import BubbleShooterEnv
import random


env = BubbleShooterEnv()
#env.render()
episodes = 100
for e in range(episodes):
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    print(reward)
    if done:
        break
    #env.render()
