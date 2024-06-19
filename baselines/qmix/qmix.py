import time
import random
from collections import deque

import numpy as np
from baselines import logger
from baselines.common import set_global_seeds
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.qmix.modules.runners.episode_runner import EpisodeRunner
from baselines.qmix.components.action_selectors import EpsilonGreedyActionSelector


def constfn(val):
    def f(_):
        return val
    return f


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)

    def add(self, obs, avail_actions, rewards, actions, states, dones):
        self.buffer.append((obs, avail_actions, rewards, actions, states, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, avail_actions, rewards, actions, states, dones = zip(*batch)
        return np.array(obs), np.array(avail_actions), np.array(rewards), np.array(actions), np.array(states), np.array(dones)

    def __len__(self):
        return len(self.buffer)



def learn(
    *, env, total_timesteps,
    eval_env=None, seed=None, nsteps=32,
    ent_coef=0.0, lr=3e-4, vf_coef=0.5,
    max_grad_norm=0.5, gamma=0.99, lam=0.95,
    log_interval=10, nminibatches=4, noptepochs=4,
    cliprange=0.2, save_interval=0, load_path=None,
    model_fn=None, update_fn=None, init_fn=None,
    mpi_rank_weight=1, comm=None, rnn_hidden_dim=64,
    buffer_size=5000, num_agents=3, **network_kwargs
):
    # episode_batch = runner.run()
    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    # ob_space value: Box(72, 96, 16), type: <class 'gym.spaces.box.Box'>
    print(f"ob_space value: {np.prod(ob_space.shape) // num_agents}, type: {type(ob_space)}")
    ac_space = env.action_space
    # Discrete(19), type: <class 'gym.spaces.discrete.Discrete'>
    
    print(f"n_actions value: {env.action_space.nvec[1]}, type: {type(ac_space)}")

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.qmix.modules.agents.rnn_agent import RNNAgent
        model_fn = RNNAgent

    model = model_fn(
        input_shape=np.prod(ob_space.shape) // num_agents,
        rnn_hidden_dim=rnn_hidden_dim,
        n_actions=env.action_space.nvec[1]
    )

    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = EpisodeRunner(env=env, model=model, nsteps=nsteps, gamma=gamma, n_agents=num_agents)
    if eval_env is not None:
        eval_runner = EpisodeRunner(env=eval_env, model=model, nsteps=nsteps, gamma=gamma)

    replay_buffer = ReplayBuffer(buffer_size)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    # EpsilonGreedyActionSelector
    actionSelector = EpsilonGreedyActionSelector()

    nupdates = total_timesteps//nbatch

    env.reset()
    for update in range(1, nupdates+1):
        print(f"{update} / {nupdates+1}")
        # assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange; ppo
        # cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        print("Before runner.run")
        obs, avail_actions, rewards, actions, state, dones = \
            runner.run(actionSelector, update*nbatch) #pylint: disable=E0632
        if eval_env is not None:
            obs, avail_actions, rewards, actions, state, dones = \
                eval_runner.run(actionSelector, update*nbatch) #pylint: disable=E0632
        print("After runner.run")
        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        # Store the episodes in the replay buffer
        replay_buffer.add(obs, avail_actions, rewards, actions, state, dones)

        # Sample a batch from the replay buffer
        if len(replay_buffer) > nbatch_train:
            for _ in range(noptepochs):
                batch_obs, batch_avail_actions, batch_rewards, batch_actions, batch_states, batch_dones = replay_buffer.sample(nbatch_train)
                
                # # Compute targets and update the model
                # q_vals, target_q_vals = model.compute_q_values(batch_obs, batch_states)
                # targets = batch_rewards + gamma * np.max(target_q_vals, axis=1) * (1 - batch_dones)
                # loss = model.update(batch_obs, batch_actions, targets)

                # if update_fn is not None:
                #     update_fn(update, model, loss)

        # if save_interval and (update % save_interval == 0 or update == 1) and is_mpi_root:
        #     path = "20240620"
        #     model.save(f"./models/{path}.pt")

        # if update % log_interval == 0 and is_mpi_root:
        #     logger.info(f"Update {update}, loss {loss}")

    return 1

