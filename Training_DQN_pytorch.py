import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import robo_rugby
import robo_rugby.gym_env.RR_Constants as const
from robo_rugby.gym_env import GameEnv_Simple
import datetime
import imageio
import os
import pygame
# import gym.envs.box2d.lunar_lander
from Training_SAC_pytorch import plot_learning_curve

""" Only using Linear layers instead of convolutional because the observation array is so basic (len 8 array) """

# region DeepQNetwork + Agent

class DeepQNetwork(nn.Module):
    """ Inheriting from nn.Module gives us stuff like backpropagation """
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        """

        :param lr: learning rate
        :param input_dims:
        :param fc1_dims:
        :param fc2_dims:
        :param n_actions:
        """
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # * operator is unpacking the list
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims).float()
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims).float()
        self.fc3 = nn.Linear(self.fc2_dims, n_actions).float()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        # self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.device = T.device("cpu")

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        var = F.relu(self.fc1(state.float()))
        var = F.relu(var)
        var = F.relu(self.fc2(var))
        var = F.relu(var)
        actions = self.fc3(var)

        return actions

    def save_checkpoint(self, str_file):
        T.save(self.state_dict(), str_file)

    def load_checkpoint(self, str_file):
        self.load_state_dict(T.load(str_file))

class DQNAgent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size = 100000, eps_end=0.01, eps_dec=5e-4,
                 fc1_dims=256, fc2_dims=256):
        """

        :param gamma: Determines the weighting of future rewards (vs immediate)
        :param epsilon: How often to pick random action to explore? 1 = 100% random, 0 = never random
        :param lr: learning rate. #todo some form of learning rate decay would probably be beneficial.
        :param input_dims: input dimensions
        :param batch_size:
        :param n_actions:
        :param max_mem_size:
        :param eps_end: Minimum value of epsilon (we probly never want to TRULY degrade
        :param eps_dec: Tutorial is using linear decay (others definitely valid, he says). Decay by this amount.
        :param fc1_dims: Size of layer 1 in the Q network
        :param fc2_dims: Size of layer 2 in the Q network
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(lr, input_dims, fc1_dims, fc2_dims, n_actions)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        i = self.mem_cntr % self.mem_size
        self.state_memory[i] = state
        self.new_state_memory[i] = state_
        self.action_memory[i] = action
        self.terminal_memory[i] = done
        self.reward_memory[i] = reward
        self.mem_cntr += 1

    def choose_action(self, observation, bln_from_policy = False):
        if bln_from_policy or np.random.random() > self.epsilon:
            """ stick to policy """
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval(state)
            action = T.argmax(actions).item()
        else:
            """ pick random """
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            # memory is initialized to all 0's
            # sampling before it's actually full is dumm
            return

        # zero the gradient on our optimizer (yup...)
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_size, self.mem_cntr)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # Ex. np.arange(4, int32) returns a np array: [0, 1, 2, 3]
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        # todo what is even happening with the 'dereferencing' at the end of this statement...
        q_eval = self.Q_eval(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval(new_state_batch)
        # todo this has issues trying to use cuda device (although... not any faster...)
        q_next[terminal_batch] = 0.0
        # TODO Target network for better DQN (have to buy Phil's udemy course)
        # T.max() returns a tuple (value of max element, index of max element) - hence the [0]
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        # q_eval = how well did it actualllly go?
        # q_target = how did our Q network predict it would go?
        # Backpropagate!
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon -= self.eps_dec
        self.epsilon = max(self.epsilon, self.eps_end)

    def save_checkpoint(self, str_file):
        self.Q_eval.save_checkpoint(str_file)

    def load_checkpoint(self, str_file):
        self.Q_eval.load_checkpoint(str_file)

# endregion

def get_action_from_player() -> int:
    # Process player input to send to the step function
    dctEvent = pygame.event.get()  # if you don't call this, the code below never finds any keys...
    dctKeyDown = pygame.key.get_pressed()
    if dctKeyDown[const.KEY_BOTH_MOTOR_FORWARD]:
        if dctKeyDown[const.KEY_BOTH_MOTOR_LEFT] and not dctKeyDown[const.KEY_BOTH_MOTOR_RIGHT]:
            return GameEnv_Simple.Direction.F_L.value
        elif dctKeyDown[const.KEY_BOTH_MOTOR_RIGHT] and not dctKeyDown[const.KEY_BOTH_MOTOR_LEFT]:
            return GameEnv_Simple.Direction.F_R.value
        else:
            return GameEnv_Simple.Direction.FORWARD.value

    elif dctKeyDown[const.KEY_BOTH_MOTOR_BACKWARD]:
        if dctKeyDown[const.KEY_BOTH_MOTOR_LEFT] and not dctKeyDown[const.KEY_BOTH_MOTOR_RIGHT]:
            return GameEnv_Simple.Direction.B_L.value
        elif dctKeyDown[const.KEY_BOTH_MOTOR_RIGHT] and not dctKeyDown[const.KEY_BOTH_MOTOR_LEFT]:
            return GameEnv_Simple.Direction.B_R.value
        else:
            return GameEnv_Simple.Direction.BACKWARD.value

    elif dctKeyDown[const.KEY_BOTH_MOTOR_LEFT]:
        return GameEnv_Simple.Direction.LEFT.value

    elif dctKeyDown[const.KEY_BOTH_MOTOR_RIGHT]:
        return GameEnv_Simple.Direction.RIGHT.value

    else:
        return np.random.choice(list(i for i in range(len(GameEnv_Simple.Direction))))

if __name__ == "__main__":
    # str_env = "LunarLander-v2"
    str_env = "RoboRugbySimple-v0"
    env = gym.make(str_env)
    """
    2020_09_13__00_56 - lr .0001, gamma .95, eps_dec 1e-6, fc1/fc2 256, no human start - 1000+, NEVER CONVERGED
    2020_09_13__11_41 - lr .001, gamma .5 (or .8?), 1e-6, fc1/2 256, 50 game human start - ...
    """
    agent = DQNAgent(
        gamma=.5,  #0.99,
        epsilon=1.0,
        lr=.001, #0.001,
        input_dims=env.observation_space.shape,
        batch_size=env.spec.max_episode_steps + 100,
        n_actions=len(GameEnv_Simple.Direction),
        eps_end=0.01,
        eps_dec=1e-6,  # 5e-4,  1e-5 ~= 100 runs till min
        fc1_dims=256,
        fc2_dims=256
    )

    scores, eps_history = [], []
    n_games = 100000
    n_games_human = 50
    bln_render = True

    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/dqn'):
        os.makedirs('checkpoints/dqn')

    """
    SHOULD WE LOAD A CHECKPOINT OR NOT? (CAREFUL WITH THIS)
    """

    lng_start_episode = 1050
    str_session = "2020_09_13__11_41"
    assert (lng_start_episode == 0 and str_session == "") or \
           (lng_start_episode != 0 and str_session != "")
    
    if str_session == "":
        dtm_start = datetime.datetime.now()
        dir = f"DQN_Pytorch_{str_env}_{dtm_start.strftime('%Y_%m_%d__%H_%M')}"
        os.makedirs(f"checkpoints/dqn/{dir}")        
    else:
        dir = f"DQN_Pytorch_{str_env}_{str_session}"
        agent.load_checkpoint(f"checkpoints/dqn/{dir}/{dir}_Ep_{lng_start_episode}.dat")
        print(f"Resuming from Episode {lng_start_episode}, Session {str_session}, Env {str_env}")

    checkpoint_freq = 50  # save/record every X games
    dct_save_files = dict.fromkeys(i for i in range(lng_start_episode + checkpoint_freq, n_games, checkpoint_freq))
    for k in dct_save_files:
        dct_save_files[k] = f"checkpoints/dqn/{dir}/{dir}_Ep_{k}"

    clock = pygame.time.Clock()

    for i in range(lng_start_episode + 1, n_games, 1):
        score = 0
        done = False
        observation = env.reset()
        obs_grumpy = env.unwrapped.get_game_state(intTeam=const.TEAM_GRUMPY)
        bln_checkpoint = (i in dct_save_files)
        bln_player = i < n_games_human

        if bln_checkpoint:
            video_stream = imageio.get_writer(dct_save_files[i] + ".mp4", fps=env.metadata['video.frames_per_second'])

        while not done:

            if bln_player and not bln_checkpoint:
                action = get_action_from_player()
            else:
                action = agent.choose_action(observation, bln_from_policy=bln_checkpoint)

            action_grumpy = agent.choose_action(obs_grumpy, bln_from_policy=bln_checkpoint)

            info :robo_rugby.gym_env.GameEnv.DebugInfo

            observation_, reward, done, info = env.step([action, action_grumpy])
            score += reward

            if not bln_checkpoint:
                obs_grumpy_ = info.adblGrumpyState
                reward_grumpy = info.dblGrumpyScore
                agent.store_transition(observation, action, reward, observation_, done)
                agent.store_transition(obs_grumpy, action_grumpy, reward_grumpy, obs_grumpy_, done)
                agent.learn()

            observation = observation_

            if bln_checkpoint:
                video_stream.append_data(env.render(mode='rgb_array'))
            if bln_render:
                env.render()
            if bln_player:
                clock.tick(const.FRAMERATE)

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        print("episode", i, "epsilon %.5f" % agent.epsilon, "avg %.2f" % avg_score, "score %.2f" % score)

        if bln_checkpoint:
            agent.save_checkpoint(dct_save_files[i] + ".dat")
            video_stream.close()

    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander_2020_tutorial.png'
    plot_learning_curve(x, scores, eps_history, filename)







