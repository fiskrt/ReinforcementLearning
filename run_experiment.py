import argparse
import gym
import gym_toytext
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="sarsa.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)
reward = []

try:
    env = gym.make(args.env)
    print("Loaded ", args.env)
except:
    print(args.env +':Env')
    gym.envs.register(
        id=args.env + "-v0",
        entry_point=args.env +':Env',
    )
    env = gym.make(args.env + "-v0")
    print("Loaded", args.env)

action_dim = env.action_space.n
state_dim = env.observation_space.n

print(args.env)

N = 50000
n_runs = 1
rewards = np.zeros((n_runs, N))
n_episodes = []
Q_avg = np.zeros((state_dim, action_dim))
for i in range(n_runs):
    agent = agentfile.Agent(state_dim, action_dim)
    observation = env.reset()

    episode = 0
    rewards_episode = []
    dct = {i:0 for i in range(6)}
    dct2 = {i:0 for i in range(2)}
    for j in range(N): 
#        env.render()
        action = agent.act(observation) # (this takes random actions)
        observation, reward, done, info = env.step(action)
        agent.observe(observation, reward, done)

        dct[observation] += 1
        dct2[action] += 1

        rewards_episode.append(reward)

        if ((j-1) % 10000) == 0:
            print(1/agent.learning_rate)

        if done:
            rewards[i][episode] = sum(rewards_episode)
            episode += 1
            rewards_episode = []
            observation = env.reset() 
        elif 'riverswim' in args.env:
            rewards[i][episode] = sum(rewards_episode)
            episode += 1
            rewards_episode = []
            

    print(agent.Q)
    Q_avg += agent.Q
    n_episodes.append(episode)
print(dct)
print(dct2)
print('AVG:')
print(np.sum(rewards[:])/sum(n_episodes))
Q_avg /= n_runs 
print('-'*25)
print('Average Q table')
np.savetxt(sys.stdout.buffer, Q_avg, fmt='%.4f')
print('-'*25)
print(n_episodes)
rewards = rewards[:,:min(n_episodes)]
print(f'reward dimension: {rewards.shape}')


def greedy_policy(Q):
    """ Print greedy frozenlake policy """
    print('Greedy policy derived from Q_avg:')
    H = [5,7,11,12,15]
    for i,a in enumerate(Q):
        if i in H: 
            print('.', end='')
        else:
            dir = ['L', 'D', 'R', 'U']
            print(dir[np.argmax(a)], end='')
        if (i+1)%4 == 0:
            print()

greedy_policy(Q_avg)

# Plot Q - table
#fig, ax = plt.subplots()
#fig.patch.set_visible(False)
#ax.axis('off')
#ax.axis('tight')
#ax.table(cellText=Q_avg, colLabels=['W', 'S', 'E', 'N'], loc='center')
#fig.tight_layout()
#plt.show()
#
try:
    from scipy.stats import t
    t_val = t.ppf(0.975, n_runs-1)/(n_runs**0.5)
    print(f'Using t-value: {t.ppf(0.975, n_runs-1)}')
except ModuleNotFoundError:
    t_val = 2.7/(n_runs**0.5)

# Calculate trending avgs. for each run.
W=500 # Use a window of size 100
rewards_MA = np.zeros((n_runs, min(n_episodes)-W+1))
for i in range(n_runs):
    rewards_MA[i] = np.convolve(rewards[i], np.ones((W,))/W, mode='valid')

mean = np.mean(rewards_MA, axis=0)
std = np.std(rewards_MA, axis=0)
plt.plot(mean, label=f'Reward ({W} episodes MA)')
plt.fill_between(np.arange(mean.shape[0]), mean-t_val*std, mean+t_val*std, alpha=0.2, label='95% CI')
if 'riverswim' in args.env:
    plt.axhline(y=0.23, color='r', linestyle='-', label='Optimal',zorder=0)
    plt.ylabel('Average reward')
    plt.xlabel('Timestep')
else:
    plt.axhline(y=0.55, color='r', linestyle='-', label='Optimal',zorder=0)
    plt.ylabel('Average reward for episode')
    plt.xlabel('Episode')

plt.legend(loc='lower right')
plt.show()
env.close()