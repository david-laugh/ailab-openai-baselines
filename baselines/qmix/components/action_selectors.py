import torch as th
from torch.distributions import Categorical

from baselines.qmix.components.epsilon_schedules import DecayThenFlatSchedule


class EpsilonGreedyActionSelector:
    def __init__(self):
        self.schedule = DecayThenFlatSchedule(
            1.0, 0.05, 50000000, decay="linear"
        )
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs)
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=1)[1]
        return picked_actions
