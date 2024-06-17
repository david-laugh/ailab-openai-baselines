from baselines.qmix.replay_buffer import ReplayBuffer
from baselines.qmix.runner import Runner
from baselines.common import explained_variance, set_global_seeds
from baselines.qmix.policy import Policy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def constfn(val):
    def f(_):
        return val
    return f


def learn(
    *, network, env, total_timesteps,
    eval_env=None, seed=None, nsteps=2048,
    ent_coef=0.0, lr=3e-4, vf_coef=0.5,
    max_grad_norm=0.5, gamma=0.99, lam=0.95,
    log_interval=10, nminibatches=4, noptepochs=4,
    cliprange=0.2, save_interval=0, load_path=None,
    model_fn=None, update_fn=None, init_fn=None,
    mpi_rank_weight=1, comm=None, **network_kwargs
):
    # 전역 시드 설정
    set_global_seeds(seed)

    # 학습률을 함수로 변환
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    
    # 클립 범위를 함수로 변환
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    
    total_timesteps = int(total_timesteps)

    # 정책 빌드 - 여기서는 주석처리 되어 있음
    # policy = build_policy(env, network, **network_kwargs)

    # 환경의 개수, 관찰 공간, 행동 공간, 배치 크기 설정
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Actor 모델 선언
    if model_fn is None:
        from baselines.qmix.model import Model
        model_fn = Model

    # Actor 모델 초기화
    model = model_fn(
        policy=Policy, ob_space=ob_space, ac_space=ac_space,
        nbatch_act=nenvs, nbatch_train=nbatch_train, nsteps=nsteps,
        ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
        comm=comm, mpi_rank_weight=mpi_rank_weight
    )

    # Actor 모델 로드
    if load_path is not None:
        model.load(load_path)

    # Actor 모델이 환경마다 runner 생성
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    # runner가 모은 데이터를 replay buffer에 저장하고 학습 진행
    replay_buffer = ReplayBuffer(size=50000)

    for update in range(1, total_timesteps // nbatch + 1):
        # runner가 nsteps만큼의 데이터를 수집
        obs, returns, masks, actions, values, neglogpacs, states = runner.run()

        # 수집한 데이터를 replay buffer에 저장
        for ob, ret, mask, action, value, neglogpac in zip(obs, returns, masks, actions, values, neglogpacs):
            replay_buffer.add(ob, action, ret, mask, value, neglogpac)

        # 학습
        for _ in range(noptepochs):
            for i in range(nminibatches):
                # replay buffer에서 데이터를 샘플링
                sample = replay_buffer.sample(nbatch_train)
                obs_batch, actions_batch, returns_batch, masks_batch, values_batch, neglogpacs_batch = sample

                # 모델 업데이트
                model.train(obs_batch, actions_batch, returns_batch, values_batch, neglogpacs_batch, lr(update), cliprange(update))

        if update % log_interval == 0 and is_mpi_root:
            print(f"Update {update}, explained variance: {explained_variance(values, returns)}")

    print("Hello, world!")
    return model
