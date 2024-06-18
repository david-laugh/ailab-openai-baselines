import numpy as np
import torch


class EpisodeRunner:
    def __init__(self, *, env, model, nsteps, gamma):
        self.env=env
        self.model=model
        self.nsteps=nsteps

        # Discount rate
        self.gamma = gamma

        # nn
        self.hidden_states = [self.model.init_hidden() for _ in range(3)]
        self.inputs = self.model.init_inputs()

    def run(self, actionSelector, T, test_mode=False):
        mb_obs, mb_avail_actions, mb_rewards, mb_actions, mb_state, mb_dones = [], [], [], [], [], []

        env_obs, env_reward, env_done, env_info = self.env.step([
            self.env.action_space.sample(),
            self.env.action_space.sample(),
            self.env.action_space.sample(),
        ])

        actions = []
        q_values = []
        for agent_id, obs in enumerate(env_obs):
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            print(obs.shape)
            q, self.hidden_states[agent_id] = self.model.forward(obs, self.hidden_states[agent_id])
            avail_actions = torch.ones(1, q.size(1))
            action = actionSelector.select_action(q, avail_actions, T)
            actions.append(action)
            q_values.append(q)

            mb_obs.append(obs)
            mb_avail_actions.append(avail_actions)
            mb_rewards.append(0)
            mb_actions.append(action)
            mb_state.append(self.hidden_states[agent_id])
            mb_dones.append(False)

        env_obs, env_rewards, env_dones, env_infos = self.env.step(actions)
        for i in range(len(env_rewards)):
            mb_rewards[-len(env_rewards) + i] = env_rewards[i]
            mb_dones[-len(env_dones) + i] = env_dones[i]

        return mb_obs, mb_avail_actions, mb_rewards, mb_actions, mb_state, mb_dones

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
