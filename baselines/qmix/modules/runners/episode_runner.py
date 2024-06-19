import numpy as np
import torch

from baselines.common.runners import AbstractEnvRunner


class EpisodeRunner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, n_agents=3):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.env=env
        self.model=model
        self.nsteps=nsteps

        # Discount rate
        self.gamma = gamma

        # nn
        self.hidden_states = [
            [self.model.init_hidden() for _ in range(n_agents)]
            for _ in range(self.env.num_envs)
        ]

        self.n_agents = n_agents
        print(f"self._n_agents {self.n_agents}")

    def run(self, actionSelector, T, test_mode=False):
        mb_obs, mb_avail_actions, mb_rewards, mb_actions, mb_states, mb_dones = [], [], [], [], [], []

        env_obs, _, _, _ = self.env.step([
            self.env.action_space.sample() for _ in range(self.env.num_envs)
        ])

        for _ in range(self.nsteps):
            actions = []
            for env_id in range(self.env.num_envs):
                obss, avail_actionss, rewardss, actionss, states, doness = [], [], [], [], [], []
                for agent_id in range(self.n_agents):
                    obs = torch.tensor(env_obs[env_id][agent_id].flatten(), dtype=torch.float32).unsqueeze(0)
                    q, self.hidden_states[env_id][agent_id] = self.model.forward(obs, self.hidden_states[env_id][agent_id])
                    avail_actions = torch.ones(1, q.size(1))
                    action = actionSelector.select_action(q, avail_actions, T)

                    obss.append(obs.numpy())  # Convert tensor to numpy
                    avail_actionss.append(avail_actions.numpy())  # Convert tensor to numpy
                    rewardss.append(0)
                    actionss.append(action)  # Convert tensor to numpy
                    states.append(self.hidden_states[env_id][agent_id].detach().numpy())  # Convert tensor to numpy

                mb_obs.append(obss)
                mb_avail_actions.append(avail_actionss)
                mb_rewards.append(rewardss)
                mb_actions.append(actionss)
                mb_states.append(states)
                mb_dones.append(False)
                actions.append(actionss)

            env_obs, env_rewards, env_dones, env_infos = self.env.step(actions)

            for env_id in range(self.env.num_envs):
                for i in range(self.n_agents):
                    mb_rewards[env_id][i] = env_rewards[env_id][i]
                    mb_dones[i] = env_dones[env_id]

        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)
        mb_states = np.asarray(mb_states, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        return mb_obs, mb_avail_actions, mb_rewards, mb_actions, mb_states, mb_dones
