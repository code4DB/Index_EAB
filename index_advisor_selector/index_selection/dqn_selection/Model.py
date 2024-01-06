import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import math
import numpy as np
import matplotlib.pyplot as plt

import pickle
import logging
from itertools import count

from tensorboardX import SummaryWriter

import Env as env
from index_advisor_selector.index_selection.dqn_selection.dqn_utils import Common
from index_advisor_selector.index_selection.dqn_selection.dqn_utils.Common import plot_report
from index_advisor_selector.index_selection.dqn_selection.dqn_utils import PR_Buffer as BufferX
from index_advisor_selector.index_selection.dqn_selection.dqn_utils import ReplyBuffer as Buffer


class NN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            # nn.Sigmoid()
        )

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, state):
        actions = self.layers(state)
        return actions


class DNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DNN, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 256)
        # self.l3 = nn.Linear(256, 128)
        self.adv1 = nn.Linear(256, 256)
        self.adv2 = nn.Linear(256, action_dim)
        self.val1 = nn.Linear(256, 64)
        self.val2 = nn.Linear(64, 1)

    def _init_weights(self):
        self.l1.weight.data.normal_(0.0, 1e-2)
        self.l1.weight.data.uniform_(-0.1, 0.1)
        self.l2.weight.data.normal_(0.0, 1e-2)
        self.l2.weight.data.uniform_(-0.1, 0.1)

        # self.l3.weight.data.normal_(0.0, 1e-2)
        # self.l3.weight.data.uniform_(-0.1, 0.1)

        self.adv1.weight.data.normal_(0.0, 1e-2)
        self.adv1.weight.data.uniform_(-0.1, 0.1)
        self.adv2.weight.data.normal_(0.0, 1e-2)
        self.adv2.weight.data.uniform_(-0.1, 0.1)
        self.val1.weight.data.normal_(0.0, 1e-2)
        self.val1.weight.data.uniform_(-0.1, 0.1)
        self.val2.weight.data.normal_(0.0, 1e-2)
        self.val2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, state):
        # actions = self.layers(state)
        x = self.relu(self.l1(state))
        x = self.relu(self.l2(x))
        # x = self.relu(self.l3(x))

        adv = self.relu(self.adv1(x))
        val = self.relu(self.val1(x))
        adv = self.relu(self.adv2(adv))
        val = self.relu(self.val2(val))
        qvals = val + (adv - adv.mean())

        return qvals


class DQN:
    def __init__(self, args, workload, frequency, action, index_mode,
                 conf, is_dnn, is_ps, is_double, a, action_mode="train"):
        self.args = args
        self.conf = conf

        self.action_mode = action_mode

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index_mode = index_mode

        # (0813): newly added.
        self.constraint = args.constraint
        self.max_count = args.max_count
        self.max_storage = args.max_storage * 1000 * 1000  # mb_to_b()

        self.workload = workload
        self.frequency = frequency

        self.action = action

        self.state_dim = len(workload) + len(action)
        # we do not need another flag to indicate "deletion/creation"
        self.action_dim = len(action)
        self.is_ps = is_ps
        self.is_double = is_double
        if is_dnn:
            self.actor = DNN(self.state_dim, self.action_dim).to(self.device)
            self.actor_target = DNN(self.state_dim, self.action_dim).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
        else:
            self.actor = NN(self.state_dim, self.action_dim).to(self.device)
            self.actor_target = NN(self.state_dim, self.action_dim).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), conf["LR"])
        # self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=self.conf["LR"], momentum=0.9)

        self.replay_buffer = None

        # some monitor information
        self.num_training = 0
        self.num_actor_update_iteration = 0
        self.actor_loss_trace = list()

        # environment
        self.envx = env.Env(args, self.workload, self.frequency, self.action, self.index_mode, a)

        # store the parameters
        self.writer = SummaryWriter(args.logdir.format(args.exp_id))

        self.learn_step_counter = 0

    def select_action(self, ep, state):
        if self.action_mode == "train":
            if not self.replay_buffer.can_update():
                action = np.random.randint(0, len(self.action))
                action = [action]

                return action

            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)

            # 1. greedy policy
            # if np.random.randn() <= self.conf["EPSILON"] * (1 - math.pow(0.5, ep / self.conf["DECAY_EP"])):
            # if np.random.randn() <= self.conf["EPSILON"] * (ep / self.conf["DECAY_EP"]):

            # if np.random.random() >= self.conf["EPSILON"]:
            if np.random.randn() <= self.conf["EPSILON"]:
                action_value = self.actor.forward(state)
                action = torch.max(action_value, 1)[1].data.cpu().numpy()
                return action

            # 2. random policy
            else:
                action = np.random.randint(0, len(self.action))
                action = [action]
                return action

        elif self.action_mode == "infer":
            chosen_idx = np.where(self.envx.current_index == 1.)

            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
            action_value = self.actor.forward(state)
            action_value = action_value.index_fill(1, torch.tensor(chosen_idx[0]).to(self.device), -torch.inf)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()

            return action

    def _sample(self):
        batch, idx = self.replay_buffer.sample(self.conf["BATCH_SIZE"])
        # state, next_state, action, reward, np.float(done))
        # batch = self.replay_memory.sample(self.batch_size)

        x, y, u, r, d = list(), list(), list(), list(), list()
        for _b in batch:
            x.append(np.array(_b[0], copy=False))
            y.append(np.array(_b[1], copy=False))
            u.append(np.array(_b[2], copy=False))
            r.append(np.array(_b[3], copy=False))
            d.append(np.array(_b[4], copy=False))

        return idx, np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.conf["LR"] * (0.1 ** (epoch // self.conf["DECAY_EP"]))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def update(self, ep):
        if self.learn_step_counter % self.conf["Q_ITERATION"] == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
        self.learn_step_counter += 1
        self.adjust_learning_rate(self.actor_optimizer, ep)

        for it in range(self.conf["U_ITERATION"]):
            idxs = None
            if self.is_ps:
                idxs, x, y, u, r, d = self._sample()
            else:
                x, y, u, r, d = self.replay_buffer.sample(self.conf["BATCH_SIZE"])

            state = torch.FloatTensor(x).to(self.device)
            action = torch.LongTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            q_eval = self.actor(state).gather(1, action)
            if self.is_double:
                next_batch = self.actor(next_state)
                nx = next_batch.max(1)[1][:, None]
                # max_act4next = np.argmax(q_eval_next, axis=1)
                q_next = self.actor_target(next_state)
                qx = q_next.gather(1, nx)
                # q_target = reward + (1 - done) * self.conf["GAMMA"] * qx.max(1)[0].view(self.conf["BATCH_SIZE"], 1)
                q_target = reward + (1 - done) * self.conf["GAMMA"] * qx
            else:
                q_next = self.actor_target(next_state).detach()
                q_target = reward + (1 - done) * self.conf["GAMMA"] * q_next.max(1)[0].view(self.conf["BATCH_SIZE"], 1)

            actor_loss = F.mse_loss(q_eval, q_target)
            error = torch.abs(q_eval - q_target).data.cpu().numpy()
            if self.is_ps:
                for i in range(self.conf["BATCH_SIZE"]):
                    idx = idxs[i]
                    self.replay_buffer.update(idx, error[i][0])

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.actor_loss_trace.append(actor_loss.data.item())

            # (0807): newly added.
            self.writer.add_scalar("actor_loss", actor_loss.data.item(), global_step=Common.tf_step)
            Common.tf_step += 1
            # for item in self.actor.named_parameters():
            # h = item[1].register_hook(lambda grad: print(grad))

    def load_model(self):
        logging.info(f"Load Model from: `{self.args.model_load}`.")
        # directory + "dqn.pth"
        self.actor.load_state_dict(torch.load(self.args.model_load))

    def save_model(self, model_save):
        # directory + "dqn.pth"
        torch.save(self.actor_target.state_dict(), model_save)

    def train(self):
        if os.path.exists(self.args.model_load):
            self.load_model()

        # (0813): newly added.
        if self.constraint == "number":
            self.envx.max_count = self.max_count
        elif self.constraint == "storage":
            self.envx.max_storage = self.max_storage

        # Greedy: check whether have an index will 60% improvement
        if self.args.pre_create:
            pre_create = self.envx.checkout()
        else:
            pre_create = list()
        logging.info(f"The set of the pre-create index is: {pre_create}.")

        # (0813): newly added.
        if self.constraint == "number" and len(pre_create) >= self.max_count:
            return pre_create

        if self.is_ps:
            self.replay_buffer = BufferX.PrioritizedReplayMemory(self.conf["MEMORY_CAPACITY"],
                                                                 min(self.conf["LEARNING_START"],
                                                                     200 * self.envx.max_count))
        else:
            self.replay_buffer = Buffer.ReplayBuffer(self.conf["MEMORY_CAPACITY"],
                                                     min(self.conf["LEARNING_START"],
                                                         200 * self.envx.max_count))

        current_best_reward = 0
        current_best_index = None
        rewards = list()
        for ep in range(self.conf["EPISODES"]):
            state = self.envx.reset()

            t_r = 0
            _state, _next_state = list(), list()
            _action, _reward, _done = list(), list(), list()
            for _ in count():
                action = self.select_action(ep, state)
                next_state, reward, done = self.envx.step(action)

                # (0813): newly added.
                if reward == -1:
                    break

                t_r += reward
                """_state.append(state)
                _next_state.append(next_state)
                _action.append(action)
                _reward.append(reward)
                _done.append(np.float(done))"""

                if self.is_ps:
                    self.replay_buffer.add(1.0, (state, next_state, action, reward, np.float(done)))
                else:
                    self.replay_buffer.push((state, next_state, action, reward, np.float(done)))
                # if self.replay_buffer.can_update():
                #    self.update()

                if done:
                    """for i in range(len(_state)):
                        if self.isPS:
                            self.replay_buffer.add(1.0, (_state[i], _next_state[i], _action[i], _reward[i]+t_r/max_count, _done[i]))
                        else:
                            self.replay_buffer.push((_state[i], _next_state[i], _action[i], _reward[i]+t_r/max_count, _done[i]))"""

                    if ep > (self.conf["EPISODES"] - 100) and t_r > current_best_reward:
                        current_best_reward = t_r
                        current_best_index = self.envx.index_trace_overall[-1]

                    # self.replay_buffer.add(1.0, (state, next_state, action, reward, np.float(done)))
                    if self.replay_buffer.can_update() and ep % 5 == 0:
                        self.update(ep)
                    break

                state = next_state

            # (0816): newly added.
            self.envx.measure["Reward"].append(t_r)

            rewards.append(t_r)
            self.writer.add_scalar("episode_reward", t_r, ep)

            if ep % self.args.save_gap == 0:
                model_save = self.args.model_save.format(self.args.exp_id, ep)
                self.save_model(model_save)
            logging.info(f"The reward of current episode {ep + 1} is: `{t_r}`.")

        exp_dir = os.path.dirname(self.args.runlog).format(self.args.exp_id)
        plot_report(exp_dir, self.envx.measure)

        cost_trace_save = f"{os.path.dirname(self.args.runlog).format(self.args.exp_id)}/cost_trace.pickle"
        with open(cost_trace_save, "wb") as wf:
            pickle.dump(self.envx.cost_trace_overall, wf, protocol=0)

        reward_save = f"{os.path.dirname(self.args.runlog).format(self.args.exp_id)}/reward.pickle"
        with open(reward_save, "wb") as wf:
            pickle.dump(rewards, wf, protocol=0)

        # plot the figure
        # self.plt_figure(rewards)

        # return current_best_index
        return current_best_index, self.envx.index_trace_overall[-1]

    # (0817): newly added.
    def infer(self):
        # assert os.path.exists(self.args.model_load), "The corresponding model does not exist!"
        if os.path.exists(self.args.model_load):
            self.load_model()

        if self.constraint == "number":
            self.envx.max_count = self.max_count
        elif self.constraint == "storage":
            self.envx.max_storage = self.max_storage

        # Greedy: check whether have an index will 60% improvement
        if self.args.pre_create:
            pre_create = self.envx.checkout()
        else:
            pre_create = list()
        logging.info(f"The set of the pre-create index is: {pre_create}.")

        if self.constraint == "number" and len(pre_create) >= self.max_count:
            return pre_create

        state = self.envx.reset()
        for _ in count():
            action = self.select_action(-1, state)
            next_state, reward, done = self.envx.step(action)

            state = next_state

            if done:
                break

        return self.envx.index_trace_overall[-1]

    def plt_figure(self, rewards):
        # plot the cost
        cost_save = f"{os.path.dirname(self.args.runlog).format(self.args.exp_id)}/cost.png"

        # (0813): newly added / modified.
        if self.constraint == "number":
            plt.figure(self.max_count)
        elif self.constraint == "storage":
            plt.figure(self.max_storage)

        x = range(len(self.envx.cost_trace_overall))
        y = [math.log(a, 10) for a in self.envx.cost_trace_overall]
        plt.plot(x, y, marker="x")
        plt.savefig(cost_save, dpi=120)
        plt.clf()
        plt.close()

        logging.info(f"Dump the 'cost' info into the fig: `{cost_save}`.")

        # plot the reward
        reward_save = f"{os.path.dirname(self.args.runlog).format(self.args.exp_id)}/reward.png"

        # (0813): newly added / modified.
        if self.constraint == "number":
            plt.figure(self.max_count + 1)
        elif self.constraint == "storage":
            plt.figure(self.max_storage + 1)

        x = range(len(rewards))
        y = rewards
        plt.plot(x, y, marker="x")
        plt.savefig(reward_save, dpi=120)
        plt.clf()
        plt.close()

        logging.info(f"Dump the 'reward' info into the fig: `{reward_save}`.")
