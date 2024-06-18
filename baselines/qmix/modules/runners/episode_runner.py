import numpy as np


class EpisodeRunner:
    def __init__(self, *, env, model, nsteps, gamma):
        self.env=env
        self.model=model
        self.nsteps=nsteps

        # Discount rate
        self.gamma = gamma

    def run(self, test_mode=False):
        # mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        # mb_states = self.states
        # epinfos = []

        # self.reset()

        mb_states, mb_obs, mb_avail_actions, mb_rewards, mb_actions, mb_terminated = [], [], [], [], [], []
        episode_return = 0
        # self.mac.init_hidden(batch_size=self.batch_size)
        epinfos = []

        while True:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            mb_states.append(self.env.get_state())
            mb_avail_actions.append(self.env.get_avail_actions())
            mb_obs.append(self.env.get_obs())
            mb_actions.append(actions)
            mb_rewards.append(reward)
            mb_terminated.append(terminated)

            self.t += 1

            if terminated:
                epinfos.append(env_info)
                break

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        mb_states = np.asarray(mb_states, dtype=np.float32)
        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_avail_actions = np.asarray(mb_avail_actions, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)
        mb_terminated = np.asarray(mb_terminated, dtype=np.bool)

        return mb_states, mb_obs, mb_avail_actions, mb_rewards, mb_actions, mb_terminated, epinfos

        # terminated = False
        # episode_return = 0
        # self.mac.init_hidden(batch_size=self.batch_size)

        # while not terminated:

        #     pre_transition_data = {
        #         "state": [self.env.get_state()],
        #         "avail_actions": [self.env.get_avail_actions()],
        #         "obs": [self.env.get_obs()]
        #     }

        #     self.batch.update(pre_transition_data, ts=self.t)

        #     # Pass the entire batch of experiences up till now to the agents
        #     # Receive the actions for each agent at this timestep in a batch of size 1
        #     actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        #     reward, terminated, env_info = self.env.step(actions[0])
        #     episode_return += reward

        #     post_transition_data = {
        #         "actions": actions,
        #         "reward": [(reward,)],
        #         "terminated": [(terminated != env_info.get("episode_limit", False),)],
        #     }

        #     self.batch.update(post_transition_data, ts=self.t)

        #     self.t += 1

        # last_data = {
        #     "state": [self.env.get_state()],
        #     "avail_actions": [self.env.get_avail_actions()],
        #     "obs": [self.env.get_obs()]
        # }
        # self.batch.update(last_data, ts=self.t)

        # # Select actions in the last stored state
        # actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # self.batch.update({"actions": actions}, ts=self.t)

        # cur_stats = self.test_stats if test_mode else self.train_stats
        # cur_returns = self.test_returns if test_mode else self.train_returns
        # log_prefix = "test_" if test_mode else ""
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        # cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        # cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # if not test_mode:
        #     self.t_env += self.t

        # cur_returns.append(episode_return)

        # if test_mode and (len(self.test_returns) == self.args.test_nepisode):
        #     self._log(cur_returns, cur_stats, log_prefix)
        # elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
        #     self._log(cur_returns, cur_stats, log_prefix)
        #     if hasattr(self.mac.action_selector, "epsilon"):
        #         self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
        #     self.log_train_stats_t = self.t_env
        mb_returns = np.zeros_like(mb_rewards)
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
