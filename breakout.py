import gym
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)

    def add(self, experience_tuple):
        self.buffer.append(experience_tuple)

    def sample(self, size=64):
        sample = random.sample(self.buffer, size)

        batch_state, batch_action, batch_next_state, batch_reward, batch_done = map(np.vstack, zip(*sample))
        batch_state = torch.from_numpy(batch_state).float().to(device)
        batch_action = torch.from_numpy(batch_action).long().to(device)
        batch_next_state = torch.from_numpy(batch_next_state).float().to(device)
        batch_reward = torch.from_numpy(batch_reward).float().to(device)
        batch_done = torch.from_numpy(batch_done * 1).float().to(device)

        return batch_state, batch_action, batch_next_state, batch_reward, batch_done


class DQN(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()
        self.learning_rate = .0005
        self.seed = torch.manual_seed(0)
        self.conv1 = torch.nn.Conv2d(160, 13440, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = self.pool(x)
        x = x.view(-1, 18 * 16 *16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def backward(self, expected, actual):
        loss = F.mse_loss(expected, actual)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Agent:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space.n
        self.state_space = env.observation_space.shape[0]

        self.memory = Memory()
        self.model = self.create_model()
        self.target = self.create_model()
        self.gamma = .99

        self.epsilon_decay = .995
        self.epsilon = 1
        self.epsilon_min = .01
        self.tau = 0.001

    def create_model(self):
        return DQN(self.state_space, self.action_space).to(device)

    def is_ready(self, threshold):
        return len(self.memory.buffer) > threshold

    def get_max_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()
        return np.argmax(action_values.cpu().data.numpy())

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        return self.get_max_action(state)

    def train(self, batch_size=64):
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = self.memory.sample(size=batch_size)

        Q_next = self.target(batch_next_state).detach().max(1)[0].unsqueeze(1)
        Q_actual = batch_reward + (self.gamma * Q_next * (1 - batch_done))
        Q_expected = self.model(batch_state).gather(1, batch_action)

        self.model.backward(Q_expected, Q_actual)
        self.update_target()

    def remember(self, experience_tuple):
        self.memory.add(experience_tuple)

    def update_target(self):
        for target_param, local_param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.seed(0)

    agent = Agent(env=env)

    total_episodes = 2000
    max_steps = 1000
    batch_size = 64

    scores = []
    last_100_scores = deque(maxlen=100)
    average_rewards = []
    epsilons = []

    for episode in range(1, total_episodes + 1):
        state = env.reset()
        score = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            agent.remember((state, action, next_state, reward, done))
            score += reward

            if step % 4 == 0:
                if agent.is_ready(batch_size):
                    agent.train(batch_size=batch_size)

            state = next_state
            if done:
                break

        last_100_scores.append(score)
        scores.append(score)
        average_rewards.append(np.mean(last_100_scores))
        epsilons.append(agent.epsilon)

        agent.decay_exploration()

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_100_scores)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_100_scores)))
        if np.mean(last_100_scores) >= 200:
            print("Goal Reached")
            torch.save(agent.model.state_dict(), 'checkpoint.pth')
            break
