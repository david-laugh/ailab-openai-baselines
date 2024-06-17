from baselines.qmix.replay_buffer import ReplayBuffer


def constfn(val):
    def f(_):
        return val
    return 


def learn(
    *, network, env, total_timesteps,
    eval_env = None, seed=None, nsteps=2048,
    ent_coef=0.0, lr=3e-4, vf_coef=0.5,
    max_grad_norm=0.5, gamma=0.99, lam=0.95,
    log_interval=10, nminibatches=4, noptepochs=4,
    cliprange=0.2, save_interval=0, load_path=None,
    model_fn=None, update_fn=None, init_fn=None,
    mpi_rank_weight=1, comm=None, **network_kwargs
):
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    print("Hello, world!")
    return
