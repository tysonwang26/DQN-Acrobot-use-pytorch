import time
import gym
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

env = gym.make("Acrobot-v1")
env = env.unwrapped
env.seed(0)
N_STATES = env.observation_space.shape[0]   # How many States
N_ACTIONS = env.action_space.n              # How many Actions
EPSILON = 0.99

BATCH_SIZE = 32
GAMMA = 0.99
MEMORY_CAPACITY = 300
LR = 0.01
NET_UPDATE = 80                   # update net
TRAIN_TIME = 8

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.func = nn.Linear(N_STATES, 25)
        self.func.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(25, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.func(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value

class DQN(object):
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()

        self.learn_counter = 0
        self.memory_counter = 0
        self.memory = numpy.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))      # state, action, reward, next_state
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if numpy.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)     # get 2 Actions
            action = torch.argmax(action_value)
            action = action.data.numpy()
        else:
            action = numpy.random.randint(0, N_ACTIONS)
        return action

    def store_memory(self, s, a, r, s_):
        memory = numpy.hstack((s, a, r, s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = memory
        self.memory_counter += 1

    def learn(self):
        if self.learn_counter % NET_UPDATE == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())     # update the net
        self.learn_counter += 1

        index = numpy.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES: N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    # Train
    dqn = DQN()
    x = []
    y = []
    for episode in range(1, TRAIN_TIME + 1):
        tStart = time.time()        # time counter Start
        x.append(episode)           # plt X
        s = env.reset()
        while True:
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)
            dqn.store_memory(s, a, r, s_)

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
            if done:
                tEnd = time.time()  # time counter End
                time.sleep(2)
                y.append(tEnd-tStart)   # plt Y
                print("Episode:", episode, "   Time:", ("%.2f" % (tEnd-tStart)))
                break
            env.render()
            s = s_
    env.close()

    import numpy
    print("Average:", ("%.2f" % (numpy.average(y))), "秒")
    plt.title("Result")
    plt.xlabel("Episode")
    plt.ylabel("Time")
    plt.plot(x, y, "-o")
    for i in range(len(x)):
        plt.text(x[i], y[i], ("%.1f" % y[i]), fontsize='10', fontweight='semibold')
    plt.show()