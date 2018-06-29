import gym, random
import numpy as np

LEARNING_RATE = 0.1
DISCOUNT = 0.99

"""Defines some frozen lake maps."""
from gym.envs.toy_text import frozen_lake, discrete
from gym.envs.registration import register

# Load Deterministic Frozen lake.
register(
    id='Deterministic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False})

class qTable:
    """
    Implements a table tracking the estimated values
    for state action pairs in an MDP.
    """
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.table = [[0 for i in range(nA)] for j in range(nS)]

    def getQ(self, s, a):
        return(self.table[s][a])

    def setQ(self, s, a, value):
        self.table[s][a] = value

    def getMaxQ(self, s):
        """
        Returns the highest Q-value
        for a given state.
        """
        hVal = self.table[s][0]
        for a in range(self.nA):
            aVal = self.table[s][a]
            if aVal > hVal:
                hVal = aVal
        return hVal

    def getMaxQAction(self, s):
        """
        Returns the action that has the highest Q-value
        for a given state.
        """
        h = 0
        hVal = self.table[s][0]
        for a in range(self.nA):
            aVal = self.table[s][a]
            if aVal > hVal:
                h = a
                hVal = aVal
        return h

def epsilonGreedy(epsilon, env, obs, qtab):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = qtab.getMaxQAction(obs)
    return action

def main():
    rewards = []
    episodes = []
    running_avg = []
    env = gym.make('Deterministic-4x4-FrozenLake-v0')
    #env.monitor.start('/tmp/frozenlake-experiment-1')
    rewardWindow = [0 for _ in range(100)]
    qtab = qTable(env.observation_space.n, env.action_space.n)
    epsilon = 1
    for i_episode in range(8000):
        observation = env.reset()
        accumulatedReward = 0
        for t in range(10000):
            #Render enviorment
            env.render()
            #Select action
            action = epsilonGreedy(epsilon, env, observation, qtab)
            #Perform action
            prevObs = observation
            observation, reward, done, info = env.step(action)
            accumulatedReward += reward
            #Update Q
            oldQ = qtab.getQ(prevObs, action)
            maxCurrQ = qtab.getMaxQ(observation)
            newQ = oldQ + LEARNING_RATE*(reward + DISCOUNT*maxCurrQ - oldQ)
            qtab.setQ(prevObs, action, newQ)
            #Check if episode is done
            if done:
                rewardWindow[i_episode % 99] = accumulatedReward
                break
        #Decrease exploration rate
        epsilon *= 0.998
        windowAvg = 0


        for i in rewardWindow:
            windowAvg += i

        print(i_episode, " ", windowAvg)
        if windowAvg >= 78:
            break

        if i_episode <= 1000:
            rewards.append(accumulatedReward)
            running_avg.append(np.array(rewards).mean())
            episodes.append(i_episode)
    #env.monitor.close()
    print(epsilon)
    print(qtab.table)
    import matplotlib.pyplot as plt
    plt.plot(episodes, running_avg)
    plt.show()

if __name__ == '__main__':
    main()
